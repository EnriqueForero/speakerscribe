"""Execute decisions: GPU jobs, CPU re-renders and content reuse.

`JobExecutor.process` is the port of notebook v5's ``process_job``: one
file in, every deliverable committed plus its master JSON out. On the LAST
attempt (or from the first one when ``publish_if_quality_wont_improve``,
because a quality rejection is deterministic), the valve publishes the
result FLAGGED instead of leaving the user with nothing.
"""

from __future__ import annotations

import contextlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from speakerscribe.batch.discovery import PROMPT_SUFFIX, SourceChangingError, source_unchanged
from speakerscribe.batch.engine import DiarizationCacheStore, TranscriptionEngine
from speakerscribe.batch.errors import (
    DiarizationFileError,
    FatalDiarizationSetupError,
    QualityRejectedError,
)
from speakerscribe.batch.fsio import atomic_write_json, is_within, utc_now
from speakerscribe.batch.identity import source_payload
from speakerscribe.batch.journal import Event, Journal
from speakerscribe.batch.masters import MasterStore
from speakerscribe.batch.profiles import diar_cache_name
from speakerscribe.batch.publisher import (
    REASON_MAX_CHARS,
    STATUS_DEGRADED,
    STATUS_FLAGGED,
    STATUS_OK,
    Job,
    Publisher,
    job_id_for,
    status_from_master,
    strip_error_prefix,
)
from speakerscribe.batch.renderers import TranscriptRejectedError
from speakerscribe.batch.settings import BatchSettings
from speakerscribe.batch.telemetry import release_memory

PIPELINE_LOAD_FAILURE = "Could not load pyannote pipeline"
LOG_FILES_KEPT = 2


@dataclass(frozen=True)
class JobResult:
    """Outcome of one committed job."""

    record: dict[str, Any]
    quality: str
    reason: str | None
    words: int
    duration_s: float
    speakers: int
    rtf: float | None
    n_extras: int


class JobExecutor:
    """Runs GPU jobs and CPU re-renders and commits their results.

    Args:
        settings: Batch settings.
        journal: Event journal.
        publisher: Deliverable writer/committer.
        masters: Master JSON store.
        diar_cache: Durable diarization cache.
        config: Engine configuration (diarization cache key).
        input_root: Input folder (source payloads).
        jobs_dir: Local scratch folder for per-job workspaces.
        failure_logs: Durable folder for logs of failed jobs.
        where: Shared "current file / stage" dict read by telemetry.
    """

    def __init__(
        self,
        *,
        settings: BatchSettings,
        journal: Journal,
        publisher: Publisher,
        masters: MasterStore,
        diar_cache: DiarizationCacheStore,
        config: Any,
        input_root: Path,
        jobs_dir: Path,
        failure_logs: Path,
        where: dict[str, Any],
    ) -> None:
        self.s = settings
        self.journal = journal
        self.publisher = publisher
        self.masters = masters
        self.diar_cache = diar_cache
        self.config = config
        self.input_root = input_root
        self.jobs_dir = jobs_dir
        self.failure_logs = failure_logs
        self.where = where

    def _payload(self, job: Job) -> dict[str, Any]:
        return source_payload(job.source, self.input_root, job.signature, self.s.hash_mode)

    def diarization_cached(self, job: Job) -> bool:
        if self.s.force_reprocess:
            return False
        return self.diar_cache.lookup(diar_cache_name(self.config, job.signature)) is not None

    # ── GPU job ──────────────────────────────────────────────────────
    def process(self, job: Job, engine: TranscriptionEngine, attempts_before: int) -> JobResult:
        """Transcribe, publish and commit one file.

        Raises:
            FatalDiarizationSetupError: pyannote cannot load at all.
            DiarizationFileError: Diarization failed for this file (retryable).
            QualityRejectedError: Quality gate rejected it (retryable).
            SourceChangingError: The source changed during the job.
            Exception: Anything else from the engine (classified by the runner).
        """
        rel = job.source.relative_posix
        workdir = self.jobs_dir / job.job_id[:24]
        shutil.rmtree(workdir, ignore_errors=True)
        workdir.mkdir(parents=True, exist_ok=True)
        media, staged_prompt = self._media_for(job)
        cache_name = diar_cache_name(self.config, job.signature)
        cached = None if self.s.force_reprocess else self.diar_cache.lookup(cache_name)
        self.journal.append(
            Event.PROCESSING,
            job_id=job.job_id,
            source=self._payload(job),
            profile=job.profile,
            diarization_cached=cached is not None,
        )
        last_attempt = attempts_before + 1 >= self.s.max_attempts
        quality_valve = self.s.publish_degraded_after_attempts and (
            last_attempt or self.s.publish_if_quality_wont_improve
        )
        diar_out = workdir / "diarization.json"
        metadata: dict[str, Any] = {}
        try:
            self.where.update(archivo=rel, etapa="transcribiendo")
            metadata = engine.transcribe(media, workdir, cached, diar_out)
            self.where.update(etapa="publicando")
            status, reason = self._classify(metadata, last_attempt, quality_valve)
            master_meta = (
                self.masters.save(metadata, job.source, job.signature, job.profile_id)
                if self.s.keep_master_json
                else None
            )
            try:
                pub = self.publisher.render_and_write(
                    job.source,
                    job.output_relative,
                    metadata,
                    status,
                    reason,
                    job.job_id,
                    valve=quality_valve,
                )
            except TranscriptRejectedError as e:
                raise QualityRejectedError(str(e)) from e
            if not source_unchanged(job.source):
                with contextlib.suppress(OSError):
                    pub.part.unlink(missing_ok=True)
                raise SourceChangingError("la fuente o su prompt cambió durante la transcripción")
            record = self.publisher.commit(job, pub, Event.COMPLETED, master_meta)
            return JobResult(
                record=record,
                quality=pub.status,
                reason=pub.reason,
                words=int(metadata.get("total_words") or 0),
                duration_s=float(metadata.get("duration_seconds") or 0.0),
                speakers=len(metadata.get("speakers_summary") or {}),
                rtf=metadata.get("real_time_factor"),
                n_extras=len(pub.extras),
            )
        except BaseException:
            self._keep_failure_logs(job, workdir)
            raise
        finally:
            self.where.update(archivo=None, etapa="entre archivos")
            with contextlib.suppress(OSError):
                if cached is None and self.diar_cache.persist(diar_out, cache_name):
                    self.journal.append(
                        Event.DIAR_CACHE_PERSISTED, job_id=job.job_id, cache=cache_name
                    )
            self._cleanup(job, workdir, staged_prompt)
            release_memory()

    def _classify(
        self, metadata: dict[str, Any], last_attempt: bool, quality_valve: bool
    ) -> tuple[str, str | None]:
        error_text = str(metadata.get("diarization_error") or "")
        detail = strip_error_prefix(error_text)
        if metadata.get("status") == "ok_degraded":
            if PIPELINE_LOAD_FAILURE in error_text:
                raise FatalDiarizationSetupError(detail)
            if not (last_attempt and self.s.publish_degraded_after_attempts):
                raise DiarizationFileError(detail or "diarización degradada")
            return STATUS_DEGRADED, (detail or "diarización degradada")[:REASON_MAX_CHARS]
        critical = [
            str(f) for f in (metadata.get("quality_flags") or []) if str(f).startswith("[CRITICAL]")
        ]
        if self.s.reject_critical_quality and critical:
            if not quality_valve:
                raise QualityRejectedError("Calidad crítica: " + "; ".join(critical))
            return STATUS_FLAGGED, "; ".join(critical)[:REASON_MAX_CHARS]
        return STATUS_OK, None

    def _media_for(self, job: Job) -> tuple[Path, Path | None]:
        """Media path for the engine; per-file glossary placed next to it."""
        if job.staged_path is None:
            return job.source.path, None  # fast mode: sidecar already beside the source
        staged_prompt = None
        if job.source.prompt_path is not None:
            staged_prompt = job.staged_path.with_suffix(PROMPT_SUFFIX)
            shutil.copy2(job.source.prompt_path, staged_prompt)
        return job.staged_path, staged_prompt

    def _cleanup(self, job: Job, workdir: Path, staged_prompt: Path | None) -> None:
        for path in (job.staged_path, staged_prompt):
            if path is not None:
                with contextlib.suppress(OSError):
                    path.unlink(missing_ok=True)
        if workdir.exists() and is_within(workdir, self.jobs_dir) and workdir != self.jobs_dir:
            shutil.rmtree(workdir, ignore_errors=True)

    def _keep_failure_logs(self, job: Job, workdir: Path) -> None:
        if not self.s.save_failure_logs:
            return
        with contextlib.suppress(Exception):
            target = self.failure_logs / job.job_id[:12]
            target.mkdir(parents=True, exist_ok=True)
            for log in sorted((workdir / "_logs").glob("*.log"))[-LOG_FILES_KEPT:]:
                shutil.copy2(log, target / log.name)
            atomic_write_json(
                target / "contexto.json",
                {"source": self._payload(job), "profile_id": job.profile_id, "utc": utc_now()},
            )

    # ── CPU paths ────────────────────────────────────────────────────
    def republish(self, job_source: Any, prior: dict[str, Any]) -> dict[str, Any] | None:
        """Re-render every deliverable from the master JSON (no GPU).

        Returns:
            The ``republished`` record, or None when no master exists.
        """
        master = self.masters.load(job_source.source_id)
        if master is None:
            return None
        signature = (prior.get("source") or {}).get("content_signature") or ""
        output_rel = Path(str((prior.get("output") or {}).get("relative_path") or ""))
        if not output_rel.name:
            return None
        profile = prior.get("profile") or {
            "id": (master.get("_v5") or {}).get("motor_profile_id", "?")
        }
        presentation = self.publisher.presentation(job_source.source_id)
        job = Job(
            source=job_source,
            signature=signature,
            profile=profile,
            job_id=job_id_for(
                signature or job_source.source_id,
                str(profile.get("id")),
                output_rel,
                presentation["id"],
            ),
            output_relative=output_rel,
            duration_seconds=master.get("duration_seconds"),
        )
        status, reason = status_from_master(master, self.s.reject_critical_quality)
        pub = self.publisher.render_and_write(
            job_source, output_rel, master, status, reason, job.job_id, valve=True
        )
        return self.publisher.commit(
            job, pub, Event.REPUBLISHED, self.masters.meta(job_source.source_id)
        )

    def reuse(self, job: Job, donor: dict[str, Any]) -> dict[str, Any] | None:
        """Dedup without GPU: adopt the donor's master, render for THIS source."""
        donor_id = (donor.get("source") or {}).get("id")
        if not donor_id:
            return None
        master = (
            self.masters.adopt(donor_id, job.source.source_id)
            if self.s.keep_master_json
            else self.masters.load(donor_id)
        )
        if master is None:
            return None
        presentation = self.publisher.presentation(job.source.source_id)
        reuse_job = Job(
            source=job.source,
            signature=job.signature,
            profile=job.profile,
            job_id=job_id_for(
                job.signature, job.profile_id, job.output_relative, presentation["id"]
            ),
            output_relative=job.output_relative,
            duration_seconds=master.get("duration_seconds"),
        )
        status, reason = status_from_master(master, self.s.reject_critical_quality)
        pub = self.publisher.render_and_write(
            job.source, job.output_relative, master, status, reason, reuse_job.job_id, valve=True
        )
        return self.publisher.commit(
            reuse_job,
            pub,
            Event.REUSED,
            self.masters.meta(job.source.source_id),
            reused_from=donor_id,
        )


__all__ = ["JobExecutor", "JobResult"]
