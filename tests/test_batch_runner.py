"""End-to-end behavior of the batch runner with a scripted engine (no GPU)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from speakerscribe.batch import BatchRunner, rebind_workspace
from speakerscribe.batch.errors import DiarizationFileError
from speakerscribe.batch.fsio import canonical_hash
from tests.batch_fakes import (
    FakeEngine,
    event_names,
    events,
    fake_probe,
    make_deps,
    make_settings,
    metadata_for,
    write_audio,
)

STAR_NAME = "2026-10-01 *Reunión de seguimiento"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    project = tmp_path / "Transcripcion-Diarizacion"
    (project / "data").mkdir(parents=True)
    return project


def run(root: Path, engine: FakeEngine, **settings_overrides):
    deps_kwargs = settings_overrides.pop("_deps", {})
    emitted: list[str] = []
    deps = make_deps(engine, emitted=emitted, **deps_kwargs)
    report = BatchRunner(make_settings(root, **settings_overrides), deps).run()
    return report, emitted


class TestHappyPath:
    def test_new_file_is_published_with_star_name_and_retired(self, root: Path):
        write_audio(root, f"{STAR_NAME}.mkv", b"AUDIO-1")
        engine = FakeEngine()
        report, _ = run(root, engine)

        assert report.end_reason == "completo"
        assert report.ok == 1 and report.failed == 0
        canonical = root / "entregables" / f"{STAR_NAME}.txt"
        assert canonical.is_file(), "the * must be kept in the output name"
        text = canonical.read_text(encoding="utf-8")
        assert "[00:00:00 - 00:00:04] SPEAKER_00: Buenos días a todos." in text
        assert "# archivo_origen: 2026-10-01 *Reunión de seguimiento.mkv" in text
        assert (root / "splits" / f"{STAR_NAME}.full_for_llm.txt").is_file()
        assert not list((root / "entregables").glob("*.part.*")), "no partial left behind"

        # Audio moved out of data/ into _procesados/<today>/ (recoverable).
        assert not (root / "data" / f"{STAR_NAME}.mkv").exists()
        moved = root / "_procesados" / date.today().isoformat() / f"{STAR_NAME}.mkv"
        assert moved.read_bytes() == b"AUDIO-1"
        assert report.retired == 1

        names = event_names(root)
        for expected in ("batch_started", "processing", "prepared", "completed", "job_metrics",
                         "source_retired", "diar_cache_persisted", "batch_finished"):  # fmt: skip
            assert expected in names, expected
        assert names.index("prepared") < names.index("completed") < names.index("source_retired")
        state = root / "entregables" / ".speakerscribe_state"
        assert not (state / "active.lock.json").exists(), "lock released"
        assert (state / "intermedios").glob("*.json.gz")
        assert (root / "entregables" / "_resumen.md").read_text(encoding="utf-8").count("✔ ok") == 1
        assert engine.loads == 1 and engine.unloads >= 1

    def test_second_run_with_empty_input_does_nothing_and_never_loads_models(self, root: Path):
        write_audio(root, "a.wav", b"AUDIO-A")
        run(root, FakeEngine())
        engine = FakeEngine()
        report, _ = run(root, engine)
        assert report.end_reason == "completo"
        assert report.deliverables == 0 and engine.loads == 0

    def test_keep_policy_is_idempotent(self, root: Path):
        write_audio(root, "a.wav", b"AUDIO-A")
        run(root, FakeEngine(), after_success="keep")
        engine = FakeEngine()
        report, _ = run(root, engine, after_success="keep")
        assert report.plan.get("ya_listos") == 1
        assert engine.calls == [] and engine.loads == 0
        assert event_names(root).count("completed") == 1

    def test_presentation_change_republishes_on_cpu(self, root: Path):
        write_audio(root, "a.wav", b"AUDIO-A")
        run(root, FakeEngine(), after_success="keep")
        engine = FakeEngine()
        report, _ = run(root, engine, after_success="keep", deliver_srt=True)
        assert report.republished == 1 and engine.calls == []
        srt = (root / "transcripts" / "a.srt").read_text(encoding="utf-8")
        assert srt.startswith("1\n00:00:00,000 --> 00:00:04,000\nSPEAKER_00: Buenos días")

    def test_subfolders_are_mirrored(self, root: Path):
        write_audio(root, "ClienteA/2026/reunion.ts", b"AUDIO-TS")
        run(root, FakeEngine())
        assert (root / "entregables" / "ClienteA" / "2026" / "reunion.txt").is_file()
        assert (root / "splits" / "ClienteA" / "2026" / "reunion.full_for_llm.txt").is_file()
        assert not (root / "data" / "ClienteA").exists(), "empty input folders are pruned"

    def test_per_file_glossary_reaches_the_engine(self, root: Path):
        write_audio(root, "a.wav", b"AUDIO-A")
        (root / "data" / "a.prompt.txt").write_text("Glosario, Términos", encoding="utf-8")
        engine = FakeEngine()
        run(root, engine)
        assert engine.calls[0]["prompt"] == "Glosario, Términos"
        assert (root / "_procesados" / date.today().isoformat() / "a.prompt.txt").exists()


class TestUserOwnedOutputs:
    def test_renamed_output_is_respected_not_regenerated(self, root: Path):
        write_audio(root, "*reunion.wav", b"AUDIO-R")
        run(root, FakeEngine(), after_success="keep")
        canonical = root / "entregables" / "*reunion.txt"
        canonical.rename(root / "entregables" / "reunion.txt")  # user removed the *
        engine = FakeEngine()
        report, _ = run(root, engine, after_success="keep")
        assert engine.calls == []
        assert report.plan.get("salida_cambiada") == 1
        assert not canonical.exists(), "the * file must not reappear"

    def test_existing_unrelated_file_is_never_clobbered(self, root: Path):
        (root / "entregables").mkdir()
        mine = root / "entregables" / "acta.txt"
        mine.write_text("mi resumen", encoding="utf-8")
        write_audio(root, "acta.wav", b"AUDIO-ACTA")
        run(root, FakeEngine())
        assert mine.read_text(encoding="utf-8") == "mi resumen"
        produced = [p.name for p in (root / "entregables").glob("acta~*.txt")]
        assert len(produced) == 1


class TestFailures:
    def test_identical_environment_errors_trip_the_breaker_without_consuming_attempts(
        self, root: Path
    ):
        for i in range(3):
            write_audio(root, f"f{i}.wav", f"AUDIO-{i}".encode())
        boom = TypeError("open() got an unexpected keyword argument 'metadata_errors'")
        engine = FakeEngine(default=boom)
        report, emitted = run(root, engine)

        assert report.end_reason == "entorno_incompatible"
        assert len(engine.calls) == 2, "stops after two identical environment failures"
        names = event_names(root)
        assert names.count("failed_environment") == 2
        assert "failed_retryable" not in names
        assert report.shutdown["apagar"] is False and report.shutdown["sospechoso"] is True
        assert all((root / "data" / f"f{i}.wav").exists() for i in range(3))
        # Diarization survived the ASR failure: a retry only pays ASR.
        assert (
            len(list((root / "entregables/.speakerscribe_state/diar_cache").glob("*.diar.json")))
            == 2
        )

    def test_diarization_cache_is_reused_on_retry(self, root: Path):
        write_audio(root, "a.wav", b"AUDIO-A")
        run(root, FakeEngine(default=TypeError("boom")))
        engine = FakeEngine()
        run(root, engine)
        assert engine.calls[0]["cache_in"] is not None

    def test_quality_rejection_consumes_attempt_then_valve_publishes_flagged(self, root: Path):
        write_audio(root, "a.wav", b"AUDIO-A")
        bad = metadata_for(quality_flags=["[CRITICAL] REPETITIONS: bucle"])
        settings = {"after_success": "keep", "publish_if_quality_wont_improve": False}
        report1, _ = run(root, FakeEngine(default=bad), **settings)
        assert report1.failed == 1 and "quality_rejected" in event_names(root)
        report2, _ = run(root, FakeEngine(default=bad), **settings)
        assert report2.flagged == 1
        text = (root / "entregables" / "a.txt").read_text(encoding="utf-8")
        assert "estado: publicado_con_flags_criticos" in text
        review = json.loads(
            (root / "entregables/.speakerscribe_state/pendientes_revision.json").read_text()
        )
        assert review["publicados_marcados"][0]["archivo"] == "a.wav"
        assert (root / "data" / "a.wav").exists(), "flagged results keep their audio"

    def test_per_file_diarization_failure_publishes_degraded_on_last_attempt(self, root: Path):
        write_audio(root, "a.wav", b"AUDIO-A")
        degraded = metadata_for(status="ok_degraded", diarization_enabled=False,
                                diarization_error="RuntimeError: CUDA error in pyannote")  # fmt: skip
        report1, _ = run(root, FakeEngine(default=degraded))
        assert report1.failed == 1
        report2, _ = run(root, FakeEngine(default=degraded))
        assert report2.flagged == 1
        assert "estado: degradado_sin_diarizacion" in (root / "entregables/a.txt").read_text(
            "utf-8"
        )

    def test_pipeline_load_failure_aborts_the_batch(self, root: Path):
        write_audio(root, "a.wav", b"AUDIO-A")
        write_audio(root, "b.wav", b"AUDIO-B")
        fatal = metadata_for(status="ok_degraded", diarization_enabled=False,
                             diarization_error="RuntimeError: Could not load pyannote pipeline")  # fmt: skip
        engine = FakeEngine(default=fatal)
        report, _ = run(root, engine)
        assert report.end_reason == "abort_fatal" and len(engine.calls) == 1

    def test_engine_load_failure_stops_without_attempts(self, root: Path):
        write_audio(root, "a.wav", b"AUDIO-A")
        engine = FakeEngine(load_error=RuntimeError("Could not load pyannote pipeline: token"))
        report, _ = run(root, engine)
        assert report.end_reason == "abort_fatal"
        assert "diarization_setup_failed" in event_names(root)
        assert "failed_retryable" not in event_names(root)

    def test_generic_error_is_retryable_and_blocks_after_max_attempts(self, root: Path):
        write_audio(root, "a.wav", b"AUDIO-A")
        for _ in range(2):
            run(
                root,
                FakeEngine(default=DiarizationFileError("x")),
                publish_degraded_after_attempts=False,
            )
        engine = FakeEngine()
        report, _ = run(root, engine)
        assert report.plan.get("bloqueados") == 1 and engine.calls == []

    def test_invalid_media_is_journaled_once_and_skipped(self, root: Path):
        write_audio(root, "roto.mp4", b"BROKEN")
        probe = fake_probe(invalid={b"BROKEN"})
        report, _ = run(root, FakeEngine(), _deps={"probe": probe})
        assert report.plan.get("invalidos") == 1
        run(root, FakeEngine(), _deps={"probe": probe})
        assert event_names(root).count("invalid_media") == 1


class TestDedupAndRecovery:
    def test_same_content_in_two_places_uses_gpu_once(self, root: Path):
        write_audio(root, "a.wav", b"SAME")
        write_audio(root, "copia/a-copia.wav", b"SAME")
        engine = FakeEngine()
        report, _ = run(root, engine)
        assert len(engine.calls) == 1 and report.reused == 1 and report.ok == 1
        assert (root / "entregables" / "copia" / "a-copia.txt").is_file()

    def test_interrupted_commit_is_recovered(self, root: Path):
        write_audio(root, "a.wav", b"AUDIO-A")
        run(root, FakeEngine(), after_success="keep")
        evs = events(root)
        prepared = next(e for e in evs if e["event"] == "prepared")
        final = root / "entregables" / "a.txt"
        part = root / "entregables" / prepared["output"]["part_relative_path"]
        final.rename(part)  # killed between "prepared" and the promotion
        log = root / "entregables/.speakerscribe_state/events.jsonl"
        kept = [e for e in evs if e["event"] not in {"completed", "batch_finished"}]
        log.write_text("".join(json.dumps(e) + "\n" for e in kept), encoding="utf-8")

        engine = FakeEngine()
        report, _ = run(root, engine, after_success="keep")
        assert report.recovered_commits == 1
        assert final.is_file() and not part.exists()
        assert engine.calls == [], "recovered result is not transcribed again"
        recovered = [e for e in events(root) if e.get("recovered_after_interruption")]
        assert len(recovered) == 1


class TestGuards:
    def test_file_that_does_not_fit_the_budget_is_deferred(self, root: Path):
        write_audio(root, "largo.wav", b"LONG")
        probe = fake_probe(durations={b"LONG": 10 * 3600.0})
        engine = FakeEngine()
        report, _ = run(
            root, engine, max_session_minutes=60, close_margin_minutes=5, _deps={"probe": probe}
        )
        assert report.deferred_time == 1 and engine.calls == []
        assert not list((root.parent / f"{root.name}_scratch" / "staging").glob("*")), (
            "staged copy removed"
        )

    def test_ram_at_stop_threshold_closes_cleanly(self, root: Path):
        write_audio(root, "a.wav", b"AUDIO-A")
        engine = FakeEngine()
        report, _ = run(root, engine, _deps={"ram": lambda: 95.0}, ram_pct_stop=90)
        assert report.end_reason == "limite_ram" and engine.calls == []
        assert report.shutdown["apagar"] is False

    def test_models_are_recycled_every_n_files(self, root: Path):
        for i in range(3):
            write_audio(root, f"f{i}.wav", f"A{i}".encode())
        engine = FakeEngine()
        report, _ = run(root, engine, recycle_every_n_files=1)
        assert report.ok == 3 and report.model_recycles == 2
        assert event_names(root).count("models_recycled") == 2

    def test_max_files_per_session(self, root: Path):
        for i in range(3):
            write_audio(root, f"f{i}.wav", f"A{i}".encode())
        report, _ = run(root, FakeEngine(), max_files_per_session=2)
        assert report.ok == 2 and report.end_reason == "limite_archivos"

    def test_lock_held_by_live_session_refuses_to_run(self, root: Path):
        state = root / "entregables" / ".speakerscribe_state"
        state.mkdir(parents=True)
        import time

        (state / "active.lock.json").write_text(
            json.dumps({"owner": "other", "heartbeat_epoch": time.time()})
        )
        write_audio(root, "a.wav", b"AUDIO-A")
        report, _ = run(root, FakeEngine())
        assert report.end_reason == "error_orquestador"
        assert "otra ejecución" in (report.fatal_error or "")
        assert (state / "active.lock.json").exists(), "never delete someone else's lock"


class TestWorkspaceBinding:
    def test_moved_input_requires_explicit_rebind(self, root: Path):
        state = root / "entregables" / ".speakerscribe_state"
        state.mkdir(parents=True)
        old = "/content/drive/MyDrive/Pruebas/Speakerscribe/data"
        (state / "workspace_identity.json").write_text(
            json.dumps(
                {"schema_version": 1, "input_root": old, "input_root_sha256": canonical_hash(old)}
            )
        )
        write_audio(root, "a.wav", b"AUDIO-A")
        report, _ = run(root, FakeEngine())
        assert report.end_reason == "error_orquestador" and "rebind_workspace" in (
            report.fatal_error or ""
        )

        result = rebind_workspace(make_settings(root))
        assert result["anterior"]["input_root"] == old
        assert "workspace_rebound" in event_names(root)
        report, _ = run(root, FakeEngine())
        assert report.ok == 1


class TestTools:
    def test_status_published_rename_and_autopsy(self, root: Path):
        from speakerscribe.batch import autopsy, published, rename_speakers, status

        write_audio(root, "a.wav", b"AUDIO-A")
        write_audio(root, "b.wav", b"AUDIO-B")
        settings = make_settings(root, after_success="keep", deliver_srt=True)
        report = BatchRunner(settings, make_deps(FakeEngine(), emitted=[])).run()
        assert report.ok == 2
        write_audio(root, "c.wav", b"AUDIO-C")

        census = status(settings)
        assert census.total == 3 and census.confirmed == 2 and census.pending == ["c.wav"]
        assert any("Pendientes" in line for line in census.lines())

        rows = published(settings)
        assert {row["audio"] for row in rows} == {"a.wav", "b.wav"}

        record = rename_speakers(settings, "a.txt", {"SPEAKER_00": "Ana", "SPEAKER_01": "Luis"})
        assert record["event"] == "republished"
        text = (root / "entregables" / "a.txt").read_text(encoding="utf-8")
        assert "Ana: Buenos días" in text and "SPEAKER_00" not in text.split("─")[-1]
        assert "Luis: Gracias" in (root / "transcripts" / "a.srt").read_text(encoding="utf-8")
        assert "speakers_renamed" in event_names(root)
        # Renames persist: a later presentation change keeps the names.
        BatchRunner(
            make_settings(root, after_success="keep", deliver_plain=True), make_deps(FakeEngine())
        ).run()
        assert "[Ana]" in (root / "transcripts" / "a.plano.txt").read_text(encoding="utf-8")

        with pytest.raises(LookupError):
            rename_speakers(settings, "no-existe.wav", {})
        assert any("AUTOPSIA" in line for line in autopsy(settings))

    def test_run_batch_shuts_down_only_when_decided(self, root: Path, monkeypatch):
        from speakerscribe.batch import run_batch

        calls: list[tuple[int, str]] = []
        monkeypatch.setattr(
            "speakerscribe.batch.colab.shutdown_runtime",
            lambda delay, reason: calls.append((delay, reason)),
        )
        write_audio(root, "a.wav", b"AUDIO-A")
        report = run_batch(
            make_settings(root, shutdown_at_end=True, shutdown_delay_s=7),
            deps=make_deps(FakeEngine()),
        )
        assert report.shutdown["apagar"] is True and calls == [(7, "completo")]
        report = run_batch(
            make_settings(root, shutdown_at_end=True), deps=make_deps(FakeEngine()), shutdown=False
        )
        assert report.shutdown["apagar"] is True, "empty input is the normal steady state"
        assert len(calls) == 1, "shutdown=False never releases the VM"


class TestPlannerPaths:
    def test_profile_change_policies(self, root: Path):
        write_audio(root, "a.wav", b"AUDIO-A")
        run(root, FakeEngine(), after_success="keep")
        engine = FakeEngine()
        report, _ = run(root, engine, after_success="keep", beam_size=3)
        assert report.plan.get("perfil_cambiado") == 1 and engine.calls == []
        assert "profile_change_deferred" in event_names(root)

        report, _ = run(
            root, FakeEngine(), after_success="keep", beam_size=3, profile_change_policy="stop"
        )
        assert (
            report.end_reason == "error_orquestador"
            and "perfil" in (report.fatal_error or "").lower()
        )

        engine = FakeEngine()
        report, _ = run(
            root, engine, after_success="keep", beam_size=3, profile_change_policy="reprocess"
        )
        assert report.ok == 1 and len(engine.calls) == 1

    def test_replaced_content_is_detected_and_retranscribed(self, root: Path):
        path = write_audio(root, "a.wav", b"AUDIO-A")
        run(root, FakeEngine(), after_success="keep")
        path.write_bytes(b"AUDIO-A-v2")
        engine = FakeEngine()
        report, _ = run(root, engine, after_success="keep")
        assert report.ok == 1 and engine.calls[0]["payload"] == b"AUDIO-A-v2"
        assert "content_change_detected" in event_names(root)

    def test_touched_file_with_same_content_needs_no_gpu(self, root: Path):
        import os
        import time as _time

        path = write_audio(root, "a.wav", b"AUDIO-A")
        run(root, FakeEngine(), after_success="keep")
        os.utime(path, (_time.time() - 100, _time.time() - 100))
        engine = FakeEngine()
        report, _ = run(root, engine, after_success="keep")
        assert engine.calls == [] and report.plan.get("ya_listos") == 1

    def test_rotating_strict_audit_rehashes_and_moves_the_cursor(self, root: Path):
        write_audio(root, "a.wav", b"AUDIO-A")
        write_audio(root, "b.wav", b"AUDIO-B")
        run(root, FakeEngine(), after_success="keep")
        engine = FakeEngine()
        report, _ = run(root, engine, after_success="keep", audits_per_session=1)
        assert engine.calls == [] and report.plan.get("ya_listos") == 2
        cursor = json.loads(
            (root / "entregables/.speakerscribe_state/audit_cursor.json").read_text()
        )
        assert cursor["relative_path"] == "a.wav"
        run(root, FakeEngine(), after_success="keep", audits_per_session=1)
        cursor = json.loads(
            (root / "entregables/.speakerscribe_state/audit_cursor.json").read_text()
        )
        assert cursor["relative_path"] == "b.wav"

    def test_force_reprocess_ignores_confirmations_and_cache(self, root: Path):
        write_audio(root, "a.wav", b"AUDIO-A")
        run(root, FakeEngine(), after_success="keep")
        engine = FakeEngine()
        report, _ = run(root, engine, after_success="keep", force_reprocess=True)
        assert report.ok == 1 and engine.calls[0]["cache_in"] is None

    def test_fast_hash_mode_reads_from_the_source(self, root: Path):
        write_audio(root, "a.wav", b"AUDIO-A")
        engine = FakeEngine()
        report, _ = run(root, engine, hash_mode="fast")
        assert report.ok == 1 and engine.calls[0]["media"].name == "a.wav"

    def test_silent_audio_is_a_first_class_result(self, root: Path):
        write_audio(root, "mudo.wav", b"SILENCE")
        engine = FakeEngine(default=metadata_for([], total_words=0))
        report, _ = run(root, engine)
        assert report.ok == 1
        text = (root / "entregables" / "mudo.txt").read_text(encoding="utf-8")
        assert "SIN_VOZ: (sin voz detectada)" in text and "estado: ok_sin_voz" in text

    def test_splits_and_all_formats(self, root: Path):
        write_audio(root, "a.wav", b"AUDIO-A")
        run(root, FakeEngine(), deliver_markdown=True, deliver_srt=True, deliver_json=True,
            deliver_plain=True, deliver_splits=True, split_words=100)  # fmt: skip
        for rel in ("transcripts/a.transcript.md", "transcripts/a.srt", "transcripts/a.json",
                    "transcripts/a.plano.txt", "splits/a.full_for_llm.txt", "splits/a.parte_01.txt"):  # fmt: skip
            assert (root / rel).stat().st_size > 0, rel
        data = json.loads((root / "transcripts" / "a.json").read_text(encoding="utf-8"))
        assert data["segments"][0] == {
            "start": 0.0,
            "end": 4.0,
            "text": "Buenos días a todos.",
            "speaker": "SPEAKER_00",
        }


class TestSafetyNets:
    def test_lost_lock_aborts_instead_of_writing_on(self, root: Path, monkeypatch):
        from speakerscribe.batch.locking import LockError, StateLock

        for i in range(2):
            write_audio(root, f"f{i}.wav", f"A{i}".encode())

        def lost(self):
            raise LockError("Se perdió el lock de estado: prueba")

        monkeypatch.setattr(StateLock, "assert_owned", lost)
        engine = FakeEngine()
        report, _ = run(root, engine)
        assert report.end_reason == "error_orquestador" and "lock" in (report.fatal_error or "")
        assert len(engine.calls) == 1, "no second file after losing the lock"
        assert "failed_retryable" not in event_names(root)
        assert "completed" not in event_names(root)

    def test_one_unplannable_file_does_not_stop_the_batch(self, root: Path, monkeypatch):
        from speakerscribe.batch.planner import Planner

        write_audio(root, "a.wav", b"AUDIO-A")
        write_audio(root, "b.wav", b"AUDIO-B")
        original = Planner.decide

        def flaky(self, source, state):
            if source.relative_posix == "a.wav":
                raise ValueError("nombre imposible")
            return original(self, source, state)

        monkeypatch.setattr(Planner, "decide", flaky)
        report, _ = run(root, FakeEngine())
        assert report.ok == 1 and report.plan.get("error_planificacion") == 1
        assert report.end_reason == "completo"
