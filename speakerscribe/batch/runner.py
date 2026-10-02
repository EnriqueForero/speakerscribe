"""The batch orchestrator (composition root of `speakerscribe.batch`).

One call does a whole session::

    report = BatchRunner(settings).run()

Order of work: preflight (no GPU) -> lock -> workspace binding -> recover
interrupted commits -> discover -> plan -> housekeeping -> one source at a
time (decide, execute, retire) -> summary files -> unlock. The GPU stack is
validated and models are loaded only when the first real job appears.

Everything confirmed is durable before the next file starts: a session
killed at any point loses at most the file in progress.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from speakerscribe.batch.colab import in_colab
from speakerscribe.batch.discovery import (
    SourceChangingError,
    discover_sources,
    probe_media,
    stage_source,
)
from speakerscribe.batch.engine import (
    DiarizationCacheStore,
    SpeakerscribeEngine,
    TranscriptionEngine,
)
from speakerscribe.batch.errors import (
    DiarizationFileError,
    FatalDiarizationSetupError,
    QualityRejectedError,
    safe_traceback,
    sanitize_error,
)
from speakerscribe.batch.executor import JobExecutor, JobResult
from speakerscribe.batch.fsio import atomic_write_json, utc_now
from speakerscribe.batch.guards import (
    CircuitBreaker,
    RamGuard,
    SessionBudget,
    error_signature,
    learned_rtf,
    shutdown_decision,
)
from speakerscribe.batch.identity import SourceInfo, bind_workspace, root_id, source_payload
from speakerscribe.batch.journal import (
    Event,
    Journal,
    JournalIndex,
    previous_input_roots,
    unfinished_signatures,
)
from speakerscribe.batch.layout import DeliverableLayout, OutputNamer
from speakerscribe.batch.locking import LockError, StateLock
from speakerscribe.batch.masters import MasterStore
from speakerscribe.batch.paths import BatchPaths
from speakerscribe.batch.planner import Action, Decision, Planner, PlanState, ProfileChangeStopError
from speakerscribe.batch.preflight import check_gpu_stack, check_storage
from speakerscribe.batch.profiles import (
    enabled_deliverables,
    engine_config,
    presentation_global,
    render_options,
)
from speakerscribe.batch.publisher import OK_STATUSES, OutputRecords, Publisher
from speakerscribe.batch.reporting import Row, SessionReport, console_summary, write_session_files
from speakerscribe.batch.retention import (
    KeepSources,
    MoveToProcessed,
    RetentionPolicy,
    prune_diar_cache,
    purge_processed,
)
from speakerscribe.batch.settings import BatchSettings
from speakerscribe.batch.speakers import RenameStore
from speakerscribe.batch.telemetry import ResourceMonitor, ram_pct, release_memory
from speakerscribe.environment import is_environment_error

DIAR_BREAKER_SIGNATURE = "diarizacion_por_archivo"


def _producer() -> str:
    from speakerscribe import __version__

    return f"speakerscribe v{__version__}"


@dataclass
class RunnerDeps:
    """Injectable collaborators (tests replace the GPU and the clocks)."""

    engine_factory: Callable[[Any], TranscriptionEngine] = SpeakerscribeEngine
    gpu_preflight: Callable[..., dict[str, Any]] = check_gpu_stack
    storage_preflight: Callable[[BatchPaths, bool], None] = check_storage
    config_factory: Callable[[BatchSettings], Any] = engine_config
    probe: Callable[..., Any] = probe_media
    stage: Callable[..., Any] = stage_source
    ram: Callable[[], float] = ram_pct
    monotonic: Callable[[], float] = time.monotonic
    wall: Callable[[], float] = time.time
    in_colab: Callable[[], bool] = in_colab
    emit: Callable[[str], None] = print
    producer: Callable[[], str] = _producer


@dataclass
class _Session:
    """Mutable state of one run (kept out of the runner's attributes)."""

    paths: BatchPaths
    journal: Journal
    lock: StateLock
    report: SessionReport
    where: dict[str, Any]
    config: Any = None
    engine: TranscriptionEngine | None = None
    environment_checked: bool = False
    files_processed: int = 0
    files_since_recycle: int = 0
    end_reason: str | None = None
    breaker: CircuitBreaker | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class BatchRunner:
    """Runs one batch session end to end.

    Args:
        settings: Validated settings.
        deps: Collaborators (defaults are the production ones).
        run_id: Explicit run id (default: random).
    """

    def __init__(
        self, settings: BatchSettings, deps: RunnerDeps | None = None, run_id: str | None = None
    ) -> None:
        self.s = settings
        self.deps = deps or RunnerDeps()
        self.run_id = run_id or uuid.uuid4().hex

    # ── Public entry point ───────────────────────────────────────────
    def run(self) -> SessionReport:
        """Process everything pending. Never raises for per-file problems.

        Returns:
            The session report (also persisted under the state folder).
        """
        d = self.deps
        paths = BatchPaths.from_settings(self.s)
        started = d.monotonic()
        session = _Session(
            paths=paths,
            journal=Journal(paths.events, self.run_id),
            lock=StateLock(paths.lock, self.run_id, force_take=self.s.force_take_lock),
            report=SessionReport(run_id=self.run_id),
            where={"archivo": None, "etapa": "preflight"},
            breaker=CircuitBreaker(self.s.breaker_threshold),
        )
        monitor: ResourceMonitor | None = None
        fatal = False
        lock_held = False
        try:
            d.storage_preflight(paths, d.in_colab())
            if self.s.resource_monitor:
                monitor = ResourceMonitor(
                    paths.telemetry,
                    run_id=self.run_id,
                    started_monotonic=started,
                    scratch=paths.scratch,
                    where=session.where,
                    heartbeat_min=self.s.heartbeat_minutes,
                    emit=d.emit,
                )
                monitor.start()
            session.lock.acquire()
            lock_held = True
            self._run_locked(session, started)
        except KeyboardInterrupt:
            session.end_reason = "interrumpido"
            d.emit("⏹ Interrumpido: todo lo confirmado está a salvo.")
        except Exception as e:
            fatal = True
            session.end_reason = session.end_reason or "error_orquestador"
            session.report.fatal_error = sanitize_error(e)
            d.emit(f"🛑 ERROR del orquestador: {sanitize_error(e)}")
            if lock_held:
                session.journal.append(
                    Event.BATCH_ABORTED, error=sanitize_error(e), traceback=safe_traceback(e)
                )
        finally:
            self._finish(session, started, monitor, fatal, lock_held)
        return session.report

    # ── Session body (lock held) ─────────────────────────────────────
    def _run_locked(self, session: _Session, started: float) -> None:
        d, s, paths, journal, report = (
            self.deps,
            self.s,
            session.paths,
            session.journal,
            session.report,
        )
        events = journal.read()
        report.environment["workspace"] = bind_workspace(paths, previous_input_roots(events))
        session.config = d.config_factory(s)

        layout = DeliverableLayout(paths)
        masters = MasterStore(paths.masters, paths.deliverables)
        presentation = presentation_global(s)
        publisher = Publisher(
            layout=layout,
            journal=journal,
            lock=session.lock,
            masters=masters,
            renames=RenameStore(paths.renames),
            options=render_options(s, d.producer()),
            deliverables=enabled_deliverables(s),
            input_root=paths.input,
            hash_mode=s.hash_mode,
            presentation_global_id=presentation["id"],
            repair_orphans_enabled=s.repair_orphans,
            orphan_tolerance_s=s.orphan_tolerance_s,
            clock=utc_now,
        )
        report.recovered_commits = publisher.recover_prepared(events)
        if report.recovered_commits:
            events = journal.read()
        discovery = discover_sources(
            s, paths.input, skip_dirs=(*paths.output_dirs(), paths.scratch)
        )
        report.warnings += discovery.warnings
        report.environment["detectados"] = len(discovery.sources)
        index = JournalIndex.build(events)
        records = OutputRecords(layout, s.verify_output_hash)
        planner = Planner(
            settings=s,
            journal=journal,
            index=index,
            records=records,
            publisher=publisher,
            layout=layout,
            namer=OutputNamer(s.name_template, s.model, s.mirror_subfolders, paths.deliverables),
            config=session.config,
            input_root=paths.input,
            staging_dir=paths.staging,
            audit_cursor=paths.audit_cursor,
            probe=d.probe,
            stage=d.stage,
            clock=d.wall,
        )
        state = planner.prepare(discovery)
        rtf = learned_rtf(events, s.model, s.rtf_asr_floor)
        report.rtf_estimate = round(rtf, 2)
        budget = SessionBudget(
            max_minutes=s.max_session_minutes,
            margin_minutes=s.close_margin_minutes,
            rtf_asr=rtf,
            rtf_diar=s.rtf_diar_floor,
            overhead_min=s.per_file_overhead_min,
            started=started,
            clock=d.monotonic,
        )
        executor = JobExecutor(
            settings=s,
            journal=journal,
            publisher=publisher,
            masters=masters,
            diar_cache=DiarizationCacheStore(paths.diar_cache),
            config=session.config,
            input_root=paths.input,
            jobs_dir=paths.jobs,
            failure_logs=paths.failure_logs,
            where=session.where,
        )
        retention: RetentionPolicy = (
            MoveToProcessed(paths.input, paths.processed, journal)
            if s.after_success == "move_to_processed"
            else KeepSources()
        )
        journal.append(
            Event.BATCH_STARTED,
            settings={
                "producer": d.producer(),
                "input_root_sha256": root_id(paths.input),
                "model": s.model,
                "language": s.language,
                "beam": s.beam_size,
                "batch": s.batch_size,
                "presentacion_id": presentation["id"],
                "detectados": len(discovery.sources),
            },
        )
        self._housekeeping(session, events, {src.source_id for src in discovery.sources})
        d.emit("════════ INICIO ════════")
        d.emit(
            f"  Detectados: {len(discovery.sources)} · auditorías: {len(state.strict_ids)} "
            f"(≤{state.audit_gb} GB) · RTF para estimar: {rtf:.1f}x · "
            + (
                f"presupuesto {s.max_session_minutes} min"
                if s.max_session_minutes
                else "sin límite"
            )
        )
        if report.recovered_commits:
            d.emit(f"  ♻️ {report.recovered_commits} commit(s) interrumpidos recuperados.")
        session.extras.update(
            planner=planner,
            state=state,
            executor=executor,
            retention=retention,
            index=index,
            budget=budget,
        )
        self._loop(session)
        session.end_reason = session.end_reason or "completo"

    def _housekeeping(
        self, session: _Session, events: list[dict[str, Any]], current: set[str]
    ) -> None:
        s, journal, paths = self.s, session.journal, session.paths
        info: dict[str, Any] = {}
        try:
            if s.after_success == "move_to_processed":
                info["procesados_purgados"] = purge_processed(
                    paths.processed, s.processed_retention_days, journal
                )
            if s.include_glob:
                # A filtered run only sees part of data/: it cannot tell which
                # caches still belong to pending files, so it never prunes.
                info["diar_cache_podados"] = 0
            else:
                protected = unfinished_signatures(events) | {
                    str((e.get("source") or {}).get("content_signature"))
                    for e in events
                    if (e.get("source") or {}).get("id") in current
                }
                info["diar_cache_podados"] = prune_diar_cache(
                    paths.diar_cache,
                    s.diar_cache_retention_days,
                    journal,
                    protected,
                    now=self.deps.wall(),
                )
        except OSError as e:
            session.report.warnings.append(f"Mantenimiento omitido: {sanitize_error(e)}")
        session.report.housekeeping = info

    # ── Main loop ────────────────────────────────────────────────────
    def _loop(self, session: _Session) -> None:
        s, report = self.s, session.report
        planner: Planner = session.extras["planner"]
        state: PlanState = session.extras["state"]
        budget: SessionBudget = session.extras["budget"]
        for source in state.ordered:
            if budget.exhausted():
                session.end_reason = "limite_sesion"
                break
            if s.max_files_per_session and session.files_processed >= s.max_files_per_session:
                session.end_reason = "limite_archivos"
                break
            try:
                decision = planner.decide(source, state)
            except (ProfileChangeStopError, LockError):
                raise
            except Exception as e:  # one odd file must not stop the batch
                report.count("error_planificacion")
                report.warnings.append(
                    f"No se pudo planificar {source.relative_posix}: {sanitize_error(e)}"
                )
                report.rows.append(
                    Row(source.relative_posix, "✖ planificación", detail=sanitize_error(e))
                )
                continue
            report.count(decision.counter)
            try:
                self._handle(session, decision)
            finally:
                if decision.action is not Action.PROCESS and decision.staged is not None:
                    path = decision.staged.staged_path
                    if path is not None:
                        path.unlink(missing_ok=True)
            if source.source_id in state.strict_ids:
                session.lock.assert_owned()
                atomic_write_json(
                    session.paths.audit_cursor,
                    {
                        "last_source_id": source.source_id,
                        "relative_path": source.relative_posix,
                        "updated_utc": utc_now(),
                    },
                )
            if session.end_reason:
                break

    def _handle(self, session: _Session, decision: Decision) -> None:
        report = session.report
        rel = decision.source.relative_posix
        if decision.action is Action.READY:
            self._retire(session, decision.source, (decision.prior or {}).get("quality"))
            return
        if decision.action is Action.SKIP:
            if decision.counter in {"salida_cambiada", "bloqueados", "invalidos"}:
                report.rows.append(Row(rel, f"⏸ {decision.counter}", detail=decision.detail))
            return
        executor: JobExecutor = session.extras["executor"]
        index: JournalIndex = session.extras["index"]
        if decision.action is Action.REPUBLISH:
            try:
                record = executor.republish(decision.source, decision.prior or {})
            except LockError:
                raise
            except Exception as e:
                report.warnings.append(f"Republicación falló para {rel}: {sanitize_error(e)[:200]}")
                report.count("republicacion_fallida")
                return
            if record is None:
                report.count("sin_maestro")
                return
            index.record_success(record)
            report.republished += 1
            report.rows.append(Row(rel, "📝 republicado", detail=record["output"]["relative_path"]))
            self._retire(session, decision.source, record.get("quality"))
            return
        if decision.action is Action.REUSE and decision.job is not None:
            try:
                record = executor.reuse(decision.job, decision.donor or {})
            except LockError:
                raise
            except Exception as e:
                report.warnings.append(
                    f"Reutilización no aplicable a {rel}: {sanitize_error(e)[:200]}"
                )
                record = None
            if record is not None:
                index.record_success(record)
                report.reused += 1
                report.rows.append(Row(rel, "♻️ reusado", detail=record["output"]["relative_path"]))
                self._retire(session, decision.job.source, record.get("quality"))
                if decision.staged is not None and decision.staged.staged_path is not None:
                    decision.staged.staged_path.unlink(missing_ok=True)
                return
            decision.action = Action.PROCESS  # fall back to the GPU
        if decision.action is Action.PROCESS:
            self._process(session, decision)

    # ── GPU job with guards ──────────────────────────────────────────
    def _process(self, session: _Session, decision: Decision) -> None:
        d, s, report = self.deps, self.s, session.report
        job = decision.job
        assert job is not None
        rel = job.source.relative_posix
        executor: JobExecutor = session.extras["executor"]
        budget: SessionBudget = session.extras["budget"]
        index: JournalIndex = session.extras["index"]
        try:
            guard = RamGuard(s.ram_pct_recycle, s.ram_pct_stop, s.recycle_every_n_files)
            action = guard.decide(
                d.ram(), session.files_since_recycle, bool(session.engine and session.engine.loaded)
            )
            if action == "recycle":
                self._recycle(
                    session,
                    "ram" if d.ram() >= s.ram_pct_recycle else f"cada {s.recycle_every_n_files}",
                )
                action = "stop" if d.ram() >= s.ram_pct_stop else "continue"
            if action == "stop":
                session.end_reason = "limite_ram"
                d.emit(f"   🛡️ RAM al {d.ram():.0f}%: cierre LIMPIO antes del OOM.")
                return
            cached = executor.diarization_cached(job)
            if s.max_session_minutes and not budget.fits(job.duration_seconds, cached):
                report.deferred_time += 1
                report.rows.append(Row(rel, "⏭️ diferido", detail="no cabe en el tiempo restante"))
                d.emit(
                    f"   ⏭️ APLAZADO: ~{budget.estimate_minutes(job.duration_seconds, cached):.0f} min "
                    f"estimados, quedan {budget.remaining_minutes():.0f}."
                )
                return
            d.emit(
                f"▶ {rel} ({job.source.size_bytes / 1e6:.1f} MB"
                + (f", {job.duration_seconds / 60:.0f} min" if job.duration_seconds else "")
                + (", diarización en caché" if cached else "")
                + ")"
            )
            if not self._ensure_engine(session, job):
                return
            session.files_processed += 1
            session.files_since_recycle += 1
            attempts = index.attempts.get(job.job_id, 0)
            try:
                result = executor.process(job, session.engine, attempts)  # type: ignore[arg-type]
            except LockError:
                raise  # never keep writing without the lock
            except Exception as e:
                self._on_failure(session, job, e)
                return
            self._on_success(session, job, result)
        finally:
            if job.staged_path is not None:
                job.staged_path.unlink(missing_ok=True)
            release_memory()

    def _ensure_engine(self, session: _Session, job: Any) -> bool:
        d, report = self.deps, session.report
        if session.engine is not None and session.engine.loaded:
            return True
        try:
            if not session.environment_checked:
                session.where.update(archivo=job.source.relative_posix, etapa="preflight GPU")
                env = d.gpu_preflight(
                    session.config, session.paths, (job.source.size_bytes, job.duration_seconds)
                )
                report.environment.update(env)
                report.warnings += list(env.get("warnings") or [])
                session.environment_checked = True
                d.emit(f"   🖥️ {env.get('gpu')} ({env.get('vram_gb')} GB) · cargando modelos…")
            session.where.update(etapa="cargando modelos")
            if session.engine is None:
                session.engine = d.engine_factory(session.config)
            session.engine.load()
            return True
        except Exception as e:
            session.journal.append(
                Event.DIARIZATION_SETUP_FAILED
                if "pyannote" in str(e).lower()
                else Event.ENVIRONMENT_FAILURE_SUSPECTED,
                error=sanitize_error(e),
                traceback=safe_traceback(e),
            )
            report.fatal_error = sanitize_error(e)
            session.end_reason = "abort_fatal"
            d.emit(
                "🛑 El entorno no puede procesar (GPU, token o librerías). Lote detenido sin gastar intentos."
            )
            d.emit("   " + sanitize_error(e)[:400])
            return False

    def _recycle(self, session: _Session, reason: str) -> None:
        before = self.deps.ram()
        if session.engine is not None:
            session.engine.unload()
        release_memory()
        after = self.deps.ram()
        session.files_since_recycle = 0
        session.report.model_recycles += 1
        session.journal.append(
            Event.MODELS_RECYCLED,
            reason=reason,
            ram_pct_before=before,
            ram_pct_after=after,
            files_done=session.files_processed,
        )
        self.deps.emit(f"   🔄 Modelos reciclados ({reason}): RAM {before:.0f}% → {after:.0f}%")

    def _on_success(self, session: _Session, job: Any, result: JobResult) -> None:
        report, rel = session.report, job.source.relative_posix
        index: JournalIndex = session.extras["index"]
        index.record_success(result.record)
        assert session.breaker is not None
        session.breaker.record_success()
        report.audio_s += result.duration_s
        session.journal.append(
            Event.JOB_METRICS,
            source=source_payload(job.source, session.paths.input, job.signature, self.s.hash_mode),
            profile=job.profile,
            duration_minutes=result.duration_s / 60,
            real_time_factor=result.rtf,
        )
        output = result.record["output"]["relative_path"]
        if result.quality in OK_STATUSES:
            report.ok += 1
            report.rows.append(
                Row(rel, f"✔ {result.quality}", result.duration_s, result.words, output)
            )
            self.deps.emit(
                f"   ✔ OK → {output} · {result.words:,} palabras · {result.speakers} hablantes"
                + (f" · RTF {result.rtf:.1f}x" if isinstance(result.rtf, int | float) else "")
            )
            self._retire(session, job.source, result.quality)
        else:
            report.flagged += 1
            report.flagged_items.append(
                {
                    "archivo": rel,
                    "salida": output,
                    "estado": result.quality,
                    "motivo": result.reason,
                }
            )
            report.rows.append(
                Row(rel, f"🧯 {result.quality}", result.duration_s, result.words, result.reason)
            )
            self.deps.emit(f"   🧯 PUBLICADO MARCADO ({result.quality}) → {output}")

    def _on_failure(self, session: _Session, job: Any, exc: Exception) -> None:
        d, report, journal = self.deps, session.report, session.journal
        index: JournalIndex = session.extras["index"]
        breaker = session.breaker
        assert breaker is not None
        rel = job.source.relative_posix
        payload = source_payload(job.source, session.paths.input, job.signature, self.s.hash_mode)
        common = {
            "job_id": job.job_id,
            "source": payload,
            "profile": job.profile,
            "error": sanitize_error(exc),
        }
        if isinstance(exc, SourceChangingError):
            journal.append(Event.SOURCE_CHANGED, **common)
            report.rows.append(Row(rel, "⏸ pospuesto", detail="el archivo cambió"))
            return
        report.failed += 1
        report.errors.append({"archivo": rel, "error": sanitize_error(exc)[:300]})
        own = isinstance(exc, DiarizationFileError | QualityRejectedError)
        if isinstance(exc, FatalDiarizationSetupError) or (not own and is_environment_error(exc)):
            journal.append(Event.FAILED_ENVIRONMENT, traceback=safe_traceback(exc), **common)
            report.rows.append(Row(rel, "✖ entorno", detail=sanitize_error(exc)))
            d.emit("   ✖ ENTORNO (no consume intentos): " + sanitize_error(exc)[:200])
            tripped = isinstance(exc, FatalDiarizationSetupError) or breaker.record_failure(
                error_signature(exc)
            )
            if tripped:
                journal.append(Event.ENVIRONMENT_FAILURE_SUSPECTED, error=sanitize_error(exc))
                session.end_reason = (
                    "abort_fatal"
                    if isinstance(exc, FatalDiarizationSetupError)
                    else "entorno_incompatible"
                )
                d.emit("🛑 Fallo de entorno repetido: el lote se detiene para no quemar cuota.")
            return
        index.attempts[job.job_id] = index.attempts.get(job.job_id, 0) + 1
        if isinstance(exc, QualityRejectedError):
            journal.append(Event.QUALITY_REJECTED, **common)
            report.rows.append(Row(rel, "✖ calidad", detail=sanitize_error(exc)))
            d.emit("   ✖ RECHAZADO por calidad (reintentará): " + sanitize_error(exc)[:200])
            breaker.record_success()
            return
        if isinstance(exc, DiarizationFileError):
            common["stage"] = "diar_file"
        journal.append(Event.FAILED_RETRYABLE, traceback=safe_traceback(exc), **common)
        report.rows.append(Row(rel, "✖ error", detail=sanitize_error(exc)))
        d.emit("   ✖ ERROR (reintentable): " + sanitize_error(exc)[:200])
        signature = (
            DIAR_BREAKER_SIGNATURE
            if isinstance(exc, DiarizationFileError)
            else error_signature(exc)
        )
        if breaker.record_failure(signature) and isinstance(exc, DiarizationFileError):
            journal.append(
                Event.DIARIZATION_SETUP_FAILED, error="archivos consecutivos sin diarizar"
            )
            session.end_reason = "abort_fatal"
            d.emit("🛑 Varios archivos seguidos sin diarizar: se corta el lote.")

    # ── Retention ────────────────────────────────────────────────────
    def _retire(self, session: _Session, info: SourceInfo, quality: str | None) -> None:
        if self.s.after_success != "move_to_processed" or quality not in OK_STATUSES:
            return
        masters: MasterStore = session.extras["executor"].masters
        if self.s.keep_master_json and not masters.path(info.source_id).is_file():
            return  # without its master the audio is the only way back
        retention: RetentionPolicy = session.extras["retention"]
        payload = source_payload(info, session.paths.input, None, self.s.hash_mode)
        try:
            moved = retention.retire(info, payload)
        except OSError as e:
            session.report.warnings.append(
                f"No se pudo mover {info.relative_posix} a _procesados: {sanitize_error(e)}"
            )
            return
        if moved is not None:
            session.report.retired += 1

    # ── Finalization ─────────────────────────────────────────────────
    def _finish(
        self,
        session: _Session,
        started: float,
        monitor: ResourceMonitor | None,
        fatal: bool,
        lock_held: bool,
    ) -> None:
        d, s, report = self.deps, self.s, session.report
        if session.engine is not None:
            try:
                session.engine.unload()
            except Exception as e:  # never mask the run's outcome
                report.warnings.append(f"Liberación de modelos: {sanitize_error(e)}")
        report.end_reason = session.end_reason or ("error_orquestador" if fatal else "completo")
        report.finished_utc = utc_now()
        report.session_minutes = round((d.monotonic() - started) / 60, 1)
        if monitor is not None:
            monitor.stop(f"fin:{report.end_reason}")
            report.peak_ram_gb = monitor.peak_ram_gb
        decision = shutdown_decision(
            end_reason=report.end_reason,
            fatal_exception=fatal,
            deliverables=report.deliverables,
            already_done=report.plan.get("ya_listos", 0),
            failures=report.failed,
            discovered=int(report.environment.get("detectados", 0)),
            session_minutes=report.session_minutes,
            shutdown_at_end=s.shutdown_at_end,
            shutdown_on_fatal=s.shutdown_on_fatal,
            only_if_work=s.shutdown_only_if_work,
        )
        report.shutdown = {
            "apagar": decision.shutdown,
            "sospechoso": decision.suspicious,
            "motivo": decision.reason,
        }
        if lock_held:
            write_session_files(
                report,
                summary_path=session.paths.summary,
                review_path=session.paths.review,
                resumen_path=session.paths.resumen_md,
                context={
                    "producer": d.producer(),
                    "input": str(session.paths.input),
                    "deliverables": str(session.paths.deliverables),
                    "engine": f"{s.model} · idioma {s.language or 'auto'} · beam {s.beam_size} "
                    f"· batch {s.batch_size}",
                },
            )
            try:
                session.journal.append(
                    Event.BATCH_FINISHED,
                    summary={
                        k: v
                        for k, v in report.to_dict().items()
                        if k not in {"rows", "environment"}
                    },
                )
            except OSError as e:
                report.warnings.append(f"No se pudo cerrar el journal: {sanitize_error(e)}")
            session.lock.release()
        for line in console_summary(report):
            d.emit(line)


def run_batch(
    settings: BatchSettings, *, shutdown: bool = True, deps: RunnerDeps | None = None
) -> SessionReport:
    """Run a session and, on Colab, release the VM when the settings say so.

    The shutdown decision is computed by `shutdown_decision` (the machine is
    kept alive when there is an error to read).
    """
    report = BatchRunner(settings, deps).run()
    if shutdown and report.shutdown.get("apagar"):
        from speakerscribe.batch.colab import shutdown_runtime

        shutdown_runtime(settings.shutdown_delay_s, str(report.end_reason))
    elif report.shutdown:
        (deps or RunnerDeps()).emit(f"🔌 Máquina activa: {report.shutdown.get('motivo')}")
    return report


__all__ = ["BatchRunner", "RunnerDeps", "run_batch"]
