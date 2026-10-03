"""Unit tests of `speakerscribe.batch` building blocks."""

from __future__ import annotations

import json
import os
import sys
import time
import types
from datetime import date, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from speakerscribe.batch import BatchPaths, BatchSettings, PathLayoutError
from speakerscribe.batch.colab import shutdown_runtime
from speakerscribe.batch.discovery import (
    SourceChangingError,
    discover_sources,
    parse_extensions,
    stage_source,
)
from speakerscribe.batch.engine import DiarizationCacheStore
from speakerscribe.batch.errors import redact, sanitize_error
from speakerscribe.batch.fsio import (
    atomic_write_gz_json,
    copy_and_hash,
    hash_file,
    read_gz_json,
)
from speakerscribe.batch.guards import (
    CircuitBreaker,
    RamGuard,
    SessionBudget,
    error_signature,
    learned_rtf,
    shutdown_decision,
)
from speakerscribe.batch.journal import Event, Journal, JournalIndex
from speakerscribe.batch.layout import OutputNamer
from speakerscribe.batch.locking import LockError, StateLock
from speakerscribe.batch.preflight import PreflightError, check_hf_access, check_storage
from speakerscribe.batch.reporting import Row, SessionReport, autopsy, render_resumen_md
from speakerscribe.batch.retention import MoveToProcessed, prune_diar_cache, purge_processed
from speakerscribe.environment import EnvironmentIncompatibleError
from tests.batch_fakes import make_settings


# ── Settings and paths ───────────────────────────────────────────────
class TestSettings:
    def test_defaults_and_derived_layout(self, tmp_path: Path):
        s = make_settings(tmp_path / "root")
        p = BatchPaths.from_settings(s)
        assert p.input == (tmp_path / "root" / "data").resolve()
        assert p.state == p.deliverables / ".speakerscribe_state"
        assert p.llm.name == "splits" and p.formats.name == "transcripts"
        assert s.deliver_full_llm is True and s.after_success == "move_to_processed"

    def test_unknown_field_is_rejected(self, tmp_path: Path):
        with pytest.raises(ValidationError):
            BatchSettings(root=tmp_path, modelo="large-v3")

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"ram_pct_recycle": 90, "ram_pct_stop": 88},
            {"max_session_minutes": 30, "close_margin_minutes": 30},
            {"min_speakers": 5, "max_speakers": 2},
            {"name_template": "{fecha}"},
            {"name_template": "{stem} {otro}"},
            {"batch_size": 3},
        ],
    )
    def test_inconsistent_values_fail_fast(self, tmp_path: Path, kwargs):
        with pytest.raises(ValidationError):
            BatchSettings(root=tmp_path, **kwargs)

    def test_language_normalized(self, tmp_path: Path):
        assert BatchSettings(root=tmp_path, language="  ES ").language == "es"
        assert BatchSettings(root=tmp_path, language="").language is None

    def test_settings_are_immutable(self, tmp_path: Path):
        s = BatchSettings(root=tmp_path)
        with pytest.raises(ValidationError):
            s.model = "small"  # type: ignore[misc]

    def test_output_inside_input_is_rejected(self, tmp_path: Path):
        with pytest.raises(PathLayoutError):
            BatchPaths.from_settings(
                BatchSettings(root=tmp_path, deliverables_dir=tmp_path / "data" / "out")
            )

    def test_scratch_on_drive_is_rejected(self, tmp_path: Path):
        with pytest.raises(PathLayoutError):
            BatchPaths.from_settings(BatchSettings(root=tmp_path, scratch_dir=tmp_path / "tmp"))

    def test_durable_folders_must_be_on_drive_in_colab(self, tmp_path: Path):
        paths = BatchPaths.from_settings(make_settings(tmp_path / "r"))
        paths.check_durable_on_colab(in_colab=False)
        fake = BatchPaths(*(Path("/content/x") for _ in range(7)), scratch=Path("/content/s"))
        with pytest.raises(PathLayoutError):
            fake.check_durable_on_colab(in_colab=True)


# ── File primitives ──────────────────────────────────────────────────
class TestFsio:
    def test_copy_and_hash_equals_full_hash(self, tmp_path: Path):
        src = tmp_path / "a.bin"
        src.write_bytes(os.urandom(3 * 1024 * 1024 + 7))
        digest = copy_and_hash(src, tmp_path / "x" / "b.bin", chunk_size=1 << 20)
        assert digest == hash_file(src, "full") == hash_file(tmp_path / "x" / "b.bin", "full")
        assert not list((tmp_path / "x").glob("*.tmp.*"))

    def test_fast_hash_is_prefixed_and_size_sensitive(self, tmp_path: Path):
        a = tmp_path / "a"
        a.write_bytes(b"x" * 100)
        assert hash_file(a, "fast").startswith("fast:")
        with pytest.raises(ValueError):
            hash_file(a, "md5")

    def test_gz_json_roundtrip(self, tmp_path: Path):
        size, sha = atomic_write_gz_json(tmp_path / "m.json.gz", {"a": "ñ"})
        assert size > 0 and sha == hash_file(tmp_path / "m.json.gz")
        assert read_gz_json(tmp_path / "m.json.gz") == {"a": "ñ"}
        assert read_gz_json(tmp_path / "missing.json.gz") is None


# ── Discovery ────────────────────────────────────────────────────────
class TestDiscovery:
    def test_finds_media_skips_noise_and_flags_homonyms(self, tmp_path: Path):
        data = tmp_path / "data"
        (data / "sub").mkdir(parents=True)
        for name in ("a.wav", "a.mp4", "notas.txt", ".oculto.wav", "~lock.wav", "a.prompt.txt"):
            (data / name).write_bytes(b"x")
        (data / "sub" / "b.MKV").write_bytes(b"y")
        (data / "vacio.wav").write_bytes(b"")
        found = discover_sources(make_settings(tmp_path), data, skip_dirs=())
        rels = [s.relative_posix for s in found.sources]
        assert rels == ["a.mp4", "a.wav", "sub/b.MKV"]
        assert ("", "a") in {(k[0].replace(".", ""), k[1]) for k in found.stem_collisions}
        wav = next(s for s in found.sources if s.relative_posix == "a.wav")
        assert wav.prompt_path == data / "a.prompt.txt"

    def test_include_glob_limits_the_run(self, tmp_path: Path):
        data = tmp_path / "data"
        data.mkdir()
        for name in ("2026-09-28 *Prueba corta.wav", "otra.wav"):
            (data / name).write_bytes(b"x")
        settings = make_settings(tmp_path, include_glob="*prueba CORTA*")
        found = discover_sources(settings, data, skip_dirs=())
        assert [s.relative_posix for s in found.sources] == ["2026-09-28 *Prueba corta.wav"]

    def test_parse_extensions(self):
        assert parse_extensions("ts, .MP4;wav") == {".ts", ".mp4", ".wav"}

    def test_stage_source_copies_while_hashing(self, tmp_path: Path):
        from speakerscribe.batch.identity import SourceInfo

        src = tmp_path / "a.wav"
        src.write_bytes(b"audio" * 1000)
        st = src.stat()
        info = SourceInfo(src, Path("a.wav"), "sid", st.st_size, st.st_mtime_ns, None, None, None)
        staged = stage_source(info, hash_mode="full", stability_seconds=0,
                              staging_dir=tmp_path / "stage", staged_name="s.wav")  # fmt: skip
        assert staged.signature == hash_file(src)
        assert (
            staged.staged_path is not None and staged.staged_path.read_bytes() == src.read_bytes()
        )
        with pytest.raises(SourceChangingError):
            stage_source(info, hash_mode="full", stability_seconds=3600,
                         staging_dir=None, staged_name="s.wav")  # fmt: skip


# ── Naming ───────────────────────────────────────────────────────────
class TestNaming:
    def _info(self, rel: str):
        from speakerscribe.batch.identity import SourceInfo

        return SourceInfo(Path("/in") / rel, Path(rel), "f" * 64, 1, 1, None, None, None)

    def test_template_and_long_name_guard(self, tmp_path: Path):
        namer = OutputNamer("{fecha} {stem}", "large-v3", True, tmp_path)
        assert namer.relative_for(self._info("x/*a.wav"), set(), "2026-10-02") == Path(
            "x/2026-10-02 *a.txt"
        )
        long = OutputNamer("{stem}", "m", False, tmp_path).relative_for(
            self._info("d/" + "ñ" * 300 + ".wav"), set()
        )
        assert len(long.name.encode()) <= 240 and long.parent == Path(".")


# ── Journal and lock ─────────────────────────────────────────────────
class TestJournal:
    def test_truncated_last_line_is_tolerated(self, tmp_path: Path):
        j = Journal(tmp_path / "events.jsonl", "run")
        j.append(Event.BATCH_STARTED)
        with (tmp_path / "events.jsonl").open("a") as fh:
            fh.write('{"event": "comple')
        assert [e["event"] for e in j.read()] == ["batch_started"]

    def test_environment_failures_never_consume_attempts_and_reset_clears(self):
        log = [
            {"event": "failed_environment", "job_id": "j"},
            {"event": "failed_retryable", "job_id": "j"},
            {"event": "quality_rejected", "job_id": "j"},
        ]
        assert JournalIndex.build(log).attempts == {"j": 2}
        log.append({"event": "attempts_reset", "job_id": "j"})
        assert JournalIndex.build(log).attempts == {}

    def test_old_failures_are_reread_with_todays_rules(self):
        """2026-10-03: libcublas.so.12 was journaled as per-file before its marker existed."""
        cublas = "RuntimeError: Library libcublas.so.12 is not found or cannot be loaded"
        log = [
            {"event": "failed_retryable", "job_id": "env", "error": cublas},
            {"event": "failed_retryable", "job_id": "file", "error": "ValueError: bad segment"},
            {"event": "failed_retryable", "job_id": "diar", "error": cublas, "stage": "diar_file"},
            {"event": "quality_rejected", "job_id": "q", "error": cublas},
        ]
        assert JournalIndex.build(log).attempts == {"file": 1, "diar": 1, "q": 1}


class TestUnfinishedSignatures:
    def test_attempted_but_never_published_content_is_protected(self):
        from speakerscribe.batch.journal import unfinished_signatures

        log = [
            {"event": "failed_retryable", "source": {"id": "old-root", "content_signature": "A"}},
            {"event": "processing", "source": {"id": "x", "content_signature": "B"}},
            {"event": "completed", "source": {"id": "y", "content_signature": "B"}},
            {"event": "failed_environment", "source": {"id": "z", "content_signature": "C"}},
        ]
        assert unfinished_signatures(log) == {"A", "C"}


class TestLock:
    def test_second_owner_is_refused_and_stale_lock_is_taken(self, tmp_path: Path):
        path = tmp_path / "lock.json"
        with StateLock(path, "a") as lock:
            lock.assert_owned()
            with pytest.raises(LockError):
                StateLock(path, "b").acquire()
        assert not path.exists()
        path.write_text(json.dumps({"owner": "zombie", "heartbeat_epoch": time.time() - 9999}))
        with StateLock(path, "c"):
            assert json.loads(path.read_text())["owner"] == "c"
        assert list(tmp_path.glob("lock.json.stale.*"))

    def test_unreadable_lock_is_never_stolen_by_default(self, tmp_path: Path):
        path = tmp_path / "lock.json"
        path.write_text("{corrupt")
        with pytest.raises(LockError):
            StateLock(path, "a").acquire()
        with StateLock(path, "a", force_take=True):
            pass

    def test_lost_ownership_is_detected(self, tmp_path: Path):
        path = tmp_path / "lock.json"
        lock = StateLock(path, "a")
        lock.acquire()
        path.write_text(json.dumps({"owner": "intruder", "heartbeat_epoch": time.time()}))
        with pytest.raises(LockError):
            lock.assert_owned()
        lock.release()
        assert path.exists(), "someone else's lock is left alone"


# ── Guards ───────────────────────────────────────────────────────────
class TestGuards:
    def test_learned_rtf_uses_recent_p10_and_floor(self):
        def ev(rtf: float) -> dict:
            return {"event": "job_metrics", "profile": {"asr_model": "m"}, "real_time_factor": rtf}

        assert learned_rtf([ev(20.0)] * 4, "m", 8.0) == 8.0
        old_fast = [ev(50.0)] * 100
        recent_slow = [ev(float(x)) for x in range(10, 20)] * 10
        assert learned_rtf(old_fast + recent_slow, "m", 8.0) == 10.0

    def test_budget(self):
        now = [0.0]
        b = SessionBudget(
            100, 10, rtf_asr=20, rtf_diar=10, overhead_min=5, started=0.0, clock=lambda: now[0]
        )
        assert b.estimate_minutes(3600) == pytest.approx(60 / 20 + 60 / 10 + 5)
        assert b.estimate_minutes(3600, diarization_cached=True) == pytest.approx(8)
        assert b.fits(3600)
        now[0] = 85 * 60
        assert not b.fits(3600) and not b.exhausted()
        now[0] = 95 * 60
        assert b.exhausted()
        assert SessionBudget(0, 0, 1, 1, 0, 0.0).fits(10**9)

    def test_ram_guard(self):
        g = RamGuard(70, 88, recycle_every_n=3)
        assert g.decide(90, 0, True) == "stop"
        assert g.decide(75, 0, True) == "recycle"
        assert g.decide(75, 0, False) == "continue"
        assert g.decide(10, 3, True) == "recycle"

    def test_breaker_needs_identical_consecutive_signatures(self):
        b = CircuitBreaker(2)
        assert not b.record_failure("A")
        assert not b.record_failure("B")
        b.record_success()
        assert not b.record_failure("A") and b.record_failure("A") and b.tripped

    def test_error_signature_ignores_numbers_and_paths(self):
        a = error_signature(RuntimeError("fallo en /content/a.wav tras 12 s"))
        b = error_signature(RuntimeError("fallo en /content/b.wav tras 99 s"))
        assert a == b

    @pytest.mark.parametrize(
        ("reason", "fatal", "deliv", "failures", "expected"),
        [
            ("completo", False, 3, 0, True),
            ("limite_sesion", False, 1, 0, True),
            ("completo", False, 0, 6, False),  # 2026-10-01: never hide the traceback
            ("limite_ram", False, 2, 0, False),
            ("abort_fatal", False, 0, 1, False),
            ("entorno_incompatible", False, 1, 2, False),
        ],
    )
    def test_shutdown_decision(self, reason, fatal, deliv, failures, expected):
        d = shutdown_decision(end_reason=reason, fatal_exception=fatal, deliverables=deliv,
                              already_done=0, failures=failures, discovered=3, session_minutes=30,
                              shutdown_at_end=True, shutdown_on_fatal=False, only_if_work=True)  # fmt: skip
        assert d.shutdown is expected

    def test_empty_input_is_normal_and_shuts_down(self):
        kw = dict(end_reason="completo", fatal_exception=False, deliverables=0, already_done=0,
                  failures=0, session_minutes=1, shutdown_at_end=True, shutdown_on_fatal=False,
                  only_if_work=True)  # fmt: skip
        assert shutdown_decision(discovered=0, **kw).shutdown is True
        assert shutdown_decision(discovered=2, **kw).suspicious is True


# ── Retention ────────────────────────────────────────────────────────
class TestRetention:
    def test_move_then_purge_after_retention(self, tmp_path: Path):
        from speakerscribe.batch.identity import SourceInfo

        data, processed = tmp_path / "data", tmp_path / "_procesados"
        (data / "sub").mkdir(parents=True)
        src = data / "sub" / "a.wav"
        src.write_bytes(b"x")
        st = src.stat()
        info = SourceInfo(
            src, Path("sub/a.wav"), "sid", st.st_size, st.st_mtime_ns, None, None, None
        )
        journal = Journal(tmp_path / "events.jsonl")
        day = date(2026, 10, 2)
        moved = MoveToProcessed(data, processed, journal, today=lambda: day).retire(
            info, {"id": "sid"}
        )
        assert moved == processed / "2026-10-02" / "sub" / "a.wav" and moved.read_bytes() == b"x"
        assert not (data / "sub").exists() and data.exists()
        (processed / "notas").mkdir()
        assert purge_processed(processed, 30, journal, today=day + timedelta(days=30)) == []
        assert purge_processed(processed, 30, journal, today=day + timedelta(days=31)) == [
            "2026-10-02"
        ]
        assert (processed / "notas").exists(), "non-date folders are never touched"
        assert [e["event"] for e in journal.read()] == ["source_retired", "processed_purged"]

    def test_changed_source_is_not_retired(self, tmp_path: Path):
        from speakerscribe.batch.identity import SourceInfo

        src = tmp_path / "a.wav"
        src.write_bytes(b"x")
        info = SourceInfo(src, Path("a.wav"), "sid", 999, 1, None, None, None)
        assert (
            MoveToProcessed(tmp_path, tmp_path / "p", Journal(tmp_path / "e")).retire(info, {})
            is None
        )
        assert src.exists()

    def test_diar_cache_prune_respects_age_and_protection(self, tmp_path: Path):
        cache = tmp_path / "diar_cache"
        cache.mkdir()
        now = time.time()
        for name, age_days in (
            ("old_x.diar.json", 100),
            ("keep_x.diar.json", 100),
            ("new_x.diar.json", 5),
        ):
            path = cache / name
            path.write_text("{}")
            os.utime(path, (now - age_days * 86400, now - age_days * 86400))
        journal = Journal(tmp_path / "events.jsonl")
        assert prune_diar_cache(cache, 90, journal, {"keep"}, now=now) == 1
        assert sorted(p.name for p in cache.iterdir()) == ["keep_x.diar.json", "new_x.diar.json"]
        assert prune_diar_cache(cache, 0, journal, set(), now=now) == 0


class TestDiarizationCacheStore:
    def test_lookup_quarantines_corrupt_and_persist_is_idempotent(self, tmp_path: Path):
        store = DiarizationCacheStore(tmp_path / "c")
        (tmp_path / "c").mkdir()
        (tmp_path / "c" / "bad.diar.json").write_text("{nope")
        assert store.lookup("bad.diar.json") is None
        assert (tmp_path / "c" / "bad.diar.json.invalid").exists()
        local = tmp_path / "local.json"
        local.write_text('{"turns": []}')
        assert store.persist(local, "k.diar.json") is True
        assert store.persist(local, "k.diar.json") is False
        assert store.lookup("k.diar.json") is not None
        assert store.persist(tmp_path / "missing.json", "z") is False


# ── Preflight, errors, Colab ─────────────────────────────────────────
class TestPreflight:
    def test_missing_input_is_actionable(self, tmp_path: Path):
        paths = BatchPaths.from_settings(make_settings(tmp_path / "r"))
        with pytest.raises(PreflightError, match="entrada"):
            check_storage(paths, in_colab=False)

    def test_missing_token(self, monkeypatch):
        cfg = types.SimpleNamespace(resolve_hf_token=lambda: None, diarization_model="org/model")
        with pytest.raises(PreflightError, match="HF_TOKEN"):
            check_hf_access(cfg)  # type: ignore[arg-type]

    def test_errors_never_leak_tokens(self):
        token = "hf_" + "A" * 30
        assert token not in sanitize_error(RuntimeError(f"bad token {token}"))
        assert "redactado" in redact(token)


class TestColab:
    def test_outside_colab_does_nothing(self):
        out: list[str] = []
        assert shutdown_runtime(0, "x", emit=out.append) is False

    def test_countdown_can_be_cancelled_and_completes(self, monkeypatch):
        calls: list[str] = []
        colab = types.ModuleType("google.colab")
        colab.drive = types.SimpleNamespace(flush_and_unmount=lambda: calls.append("flush"))  # type: ignore[attr-defined]
        colab.runtime = types.SimpleNamespace(unassign=lambda: calls.append("unassign"))  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "google.colab", colab)
        monkeypatch.setattr("speakerscribe.batch.colab.in_colab", lambda: True)

        def interrupt(_):
            raise KeyboardInterrupt

        assert shutdown_runtime(10, "x", emit=lambda _: None, sleep=interrupt) is False
        assert calls == []
        assert shutdown_runtime(10, "x", emit=lambda _: None, sleep=lambda _: None) is True
        assert calls == ["flush", "unassign"]


# ── Reporting ────────────────────────────────────────────────────────
class TestReporting:
    def test_resumen_md_escapes_table_cells(self):
        report = SessionReport(run_id="r" * 32, ok=1)
        report.rows.append(Row("a|b.wav", "✔ ok", 65.0, 1200, "línea\nnueva"))
        md = render_resumen_md(report, {"producer": "p"})
        assert "| `a/b.wav` | ✔ ok | 00:01:05 | 1,200 | línea nueva |" in md

    def test_autopsy_detects_external_kill(self, tmp_path: Path):
        tele = tmp_path / "t.jsonl"
        tele.write_text(
            "\n".join(
                json.dumps(
                    {"ram_usada_gb": 4, "ram_total_gb": 12, "motivo": "periodico", "min_sesion": m}
                )
                for m in range(5)
            )
            + "\n{trunc"  # fmt: skip
        )
        lines = autopsy(tmp_path / "none.json", tele, tmp_path / "none.jsonl")
        assert any("EXTERNO" in line for line in lines)
        assert any("NUNCA" in line for line in lines)


# ── GPU adapter, preflight and telemetry with doubles ────────────────
class TestSpeakerscribeEngine:
    def test_loads_diarization_first_persists_cache_even_on_failure(self, tmp_path, monkeypatch):
        import contextlib as _ctx

        from speakerscribe.batch import engine as eng

        order: list[str] = []

        class FakeDiar:
            def __init__(self, config):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                order.append("diar_closed")

            def load(self):
                order.append("diar_loaded")

        @_ctx.contextmanager
        def fake_whisper(config):
            order.append("whisper_loaded")
            yield "MODEL"
            order.append("whisper_closed")

        def fake_process_one(media, paths, model, config, diar_engine=None):
            cache = next(paths.diar_cache.glob("*.diar.json"), None)
            if cache is None:
                (paths.diar_cache / f"{media.stem}_hash.diar.json").write_text("{}")
            raise TypeError("asr exploded")

        monkeypatch.setattr("speakerscribe.diarization.DiarizationEngine", FakeDiar)
        monkeypatch.setattr("speakerscribe.diarization.diarization_params_hash", lambda c: "hash")
        monkeypatch.setattr("speakerscribe.transcription.loaded_whisper", fake_whisper)
        monkeypatch.setattr("speakerscribe.pipeline.process_one", fake_process_one)
        engine = eng.SpeakerscribeEngine(config=object())
        engine.load()
        assert order == ["diar_loaded", "whisper_loaded"] and engine.loaded
        media = tmp_path / "m.wav"
        media.write_bytes(b"x")
        out = tmp_path / "diar.json"
        with pytest.raises(TypeError):
            engine.transcribe(media, tmp_path / "w", None, out)
        assert out.read_text() == "{}", "diarization survives the ASR failure"
        engine.unload()
        assert order[-2:] == ["whisper_closed", "diar_closed"] and not engine.loaded


class TestGpuPreflight:
    def test_no_cuda_and_small_vram_are_actionable(self, tmp_path, monkeypatch):
        from speakerscribe.batch import preflight as pf

        torch = types.ModuleType("torch")
        cuda = types.SimpleNamespace(is_available=lambda: False)
        torch.cuda = cuda  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "torch", torch)
        paths = BatchPaths.from_settings(make_settings(tmp_path / "r"))
        cfg = types.SimpleNamespace(
            model="large-v3", resolve_hf_token=lambda: "hf_x", diarization_model="m"
        )
        with pytest.raises(PreflightError, match="GPU"):
            pf.check_gpu_stack(cfg, paths, (1, 60.0))  # type: ignore[arg-type]

        cuda.is_available = lambda: True  # type: ignore[assignment]
        cuda.get_device_name = lambda i: "Tesla T4"  # type: ignore[attr-defined]
        cuda.get_device_properties = lambda i: types.SimpleNamespace(total_memory=4e9)  # type: ignore[attr-defined]
        monkeypatch.setattr(pf, "importlib", types.SimpleNamespace(import_module=lambda name: None))
        monkeypatch.setattr(
            "speakerscribe.environment.check_audio_decoding", lambda: {"native_wav": True}
        )
        with pytest.raises(PreflightError, match="VRAM"):
            pf.check_gpu_stack(cfg, paths, (1, 60.0))  # type: ignore[arg-type]

        cuda.get_device_properties = lambda i: types.SimpleNamespace(total_memory=16e9)  # type: ignore[attr-defined]
        paths.ensure()
        monkeypatch.setattr(pf, "check_hf_access", lambda c: ["aviso"])

        def no_cublas():
            raise EnvironmentIncompatibleError("Library libcublas.so.12 is not found")

        monkeypatch.setattr(pf, "provide_ctranslate2_cuda_libs", no_cublas)
        with pytest.raises(PreflightError, match="CTranslate2 no puede usar la GPU"):
            pf.check_gpu_stack(cfg, paths, (1000, 60.0))  # type: ignore[arg-type]

        cublas = {"soname": "libcublas.so.12", "source": "system"}
        monkeypatch.setattr(pf, "provide_ctranslate2_cuda_libs", lambda: cublas)
        env = pf.check_gpu_stack(cfg, paths, (1000, 60.0))  # type: ignore[arg-type]
        assert env["gpu"] == "Tesla T4" and env["warnings"] == ["aviso"]
        assert env["cuda_libs"] == cublas


class TestProvideCtranslate2CudaLibs:
    """Colab's CUDA 13 images lack the CUDA 12 cuBLAS CTranslate2 dlopens (2026-10-03)."""

    def _patch(self, monkeypatch, outcomes):
        calls = iter(outcomes)

        def ensure():
            outcome = next(calls)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        monkeypatch.setattr("speakerscribe.environment.ensure_ctranslate2_cuda_libs", ensure)
        monkeypatch.setattr(
            "speakerscribe.environment.ctranslate2_cublas_soname", lambda: "libcublas.so.12"
        )

    def test_already_loadable_installs_nothing(self, monkeypatch):
        from speakerscribe.batch.colab import provide_ctranslate2_cuda_libs

        self._patch(monkeypatch, [{"soname": "libcublas.so.12", "source": "system"}])
        ran = []
        info = provide_ctranslate2_cuda_libs(install=True, run=lambda *a, **k: ran.append(a))
        assert info["source"] == "system" and ran == []

    def test_missing_outside_colab_raises_with_the_fix(self, monkeypatch):
        from speakerscribe.batch.colab import provide_ctranslate2_cuda_libs

        self._patch(monkeypatch, [EnvironmentIncompatibleError("pip install nvidia-cublas-cu12")])
        with pytest.raises(EnvironmentIncompatibleError, match="nvidia-cublas-cu12"):
            provide_ctranslate2_cuda_libs(install=False, run=lambda *a, **k: None)

    def test_missing_in_colab_installs_once_then_preloads(self, monkeypatch):
        from speakerscribe.batch.colab import provide_ctranslate2_cuda_libs

        self._patch(
            monkeypatch,
            [
                EnvironmentIncompatibleError("missing"),
                {"soname": "libcublas.so.12", "source": "/x"},
            ],
        )
        commands = []

        def run(cmd, **kwargs):
            commands.append(cmd)
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        info = provide_ctranslate2_cuda_libs(install=True, run=run, emit=lambda _: None)
        assert commands[0][-1] == "nvidia-cublas-cu12>=12.4,<13"
        assert info == {"soname": "libcublas.so.12", "source": "/x",
                        "installed": "nvidia-cublas-cu12>=12.4,<13"}  # fmt: skip

    def test_pip_failure_is_an_environment_error(self, monkeypatch):
        from speakerscribe.batch.colab import provide_ctranslate2_cuda_libs

        self._patch(monkeypatch, [EnvironmentIncompatibleError("missing")])
        failed = types.SimpleNamespace(returncode=1, stdout="", stderr="No space left on device")
        with pytest.raises(EnvironmentIncompatibleError, match="No space left"):
            provide_ctranslate2_cuda_libs(
                install=True, run=lambda *a, **k: failed, emit=lambda _: None
            )


class TestTelemetry:
    def test_monitor_writes_samples_and_final_reason(self, tmp_path):
        from speakerscribe.batch.telemetry import ResourceMonitor, ram_pct, sample

        assert 0.0 <= ram_pct() <= 100.0
        data = sample("run", time.monotonic(), tmp_path, {"archivo": "a", "etapa": "x"})
        assert data["archivo"] == "a" and "disco_libre_gb" in data
        monitor = ResourceMonitor(tmp_path / "t.jsonl", run_id="r", started_monotonic=time.monotonic(),
                                  scratch=tmp_path, where={}, emit=lambda _: None)  # fmt: skip
        monitor.start()
        monitor.stop("fin:test")
        monitor.join(timeout=5)
        lines = [json.loads(line) for line in (tmp_path / "t.jsonl").read_text().splitlines()]
        assert lines[0]["motivo"] == "inicio" and lines[-1]["motivo"] == "fin:test"
