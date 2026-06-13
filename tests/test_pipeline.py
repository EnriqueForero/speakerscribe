"""Orchestration: idempotency, hash migration, degraded diarization, auto-retry.

Runs the REAL process_one end-to-end against the fake faster_whisper —
only diarization and (optionally) quality evaluation are stubbed.
"""

from __future__ import annotations

import json
import sys

import pytest

from speakerscribe.audio import file_signature
from speakerscribe.config import TranscriptionConfig, WorkspacePaths
from speakerscribe.persistence import _record_from_metadata, _sqlite_register, find_run_by_hash
from speakerscribe.pipeline import process_one
from speakerscribe.quality import QualityFlag, QualityReport, Severity
from tests.fakes import (
    TWO_SPEAKER_TURNS,
    FakeModel,
    FakeSegment,
    FakeWord,
    make_fake_faster_whisper,
)


@pytest.fixture(autouse=True)
def fake_fw(monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", make_fake_faster_whisper())


@pytest.fixture
def ws(tmp_path):
    paths = WorkspacePaths(workspace=tmp_path, scratch=str(tmp_path / "scratch"))
    paths.create_directories()
    media = paths.data / "meeting.mp4"
    media.write_bytes(b"fake-media-content-" * 1000)
    return paths, media


class StubEngine:
    """Duck-typed DiarizationEngine stand-in."""

    def __init__(self, turns=None, error: Exception | None = None):
        self._turns = turns
        self._error = error
        self.calls = 0

    def diarize(self, audio_path, cache_path=None):
        self.calls += 1
        if self._error:
            raise self._error
        return self._turns or []


def _cfg(**over) -> TranscriptionConfig:
    base = {
        "model": "large-v3-turbo",
        "language": "es",
        "batch_size": 1,
        "extract_temp_wav": False,  # no ffmpeg in unit tests
        "enable_diarization": False,
        "evaluate_quality": False,
        "auto_retry_on_critical": False,
    }
    base.update(over)
    return TranscriptionConfig(**base)


class TestHappyPathAndIdempotency:
    def test_ok_run_writes_outputs_and_ledger(self, ws):
        paths, media = ws
        meta = process_one(media, paths, FakeModel(), _cfg())
        assert meta["status"] == "ok"
        base = meta["base_name"]
        for ext in (".txt", ".srt", ".json", ".transcript.md"):
            assert (paths.transcripts / f"{base}{ext}").exists(), ext
        assert paths.ledger_path.exists()
        rec = find_run_by_hash(paths.ledger_path, meta["file_hash"], "large-v3-turbo", None)
        assert rec is not None and rec["status"] == "ok"

    def test_second_run_is_skipped_by_content_hash(self, ws):
        paths, media = ws
        process_one(media, paths, FakeModel(), _cfg())
        again = process_one(media, paths, FakeModel(), _cfg())
        assert again["status"] == "skipped"
        assert again["previous_run_id"] is not None

    def test_rename_does_not_defeat_idempotency(self, ws):
        """Same bytes, new filename -> hash hit (but outputs are name-based,
        so a rename means missing outputs -> reprocess; the LEDGER hit is the
        contract under test here)."""
        paths, media = ws
        first = process_one(media, paths, FakeModel(), _cfg())
        renamed = media.with_name("renamed.mp4")
        media.rename(renamed)
        rec = find_run_by_hash(paths.ledger_path, first["file_hash"], "large-v3-turbo", None)
        assert rec is not None
        assert file_signature(renamed, "fast") == first["file_hash"]

    def test_force_reprocess_bypasses_skip(self, ws):
        paths, media = ws
        process_one(media, paths, FakeModel(), _cfg())
        again = process_one(media, paths, FakeModel(), _cfg(force_reprocess=True))
        assert again["status"] == "ok"

    def test_no_ledger_falls_back_to_filename_skip(self, ws):
        paths, media = ws
        process_one(media, paths, FakeModel(), _cfg(enable_runs_db=False))
        again = process_one(media, paths, FakeModel(), _cfg(enable_runs_db=False))
        assert again["status"] == "skipped"
        assert not paths.ledger_path.exists()


class TestLegacyHashMigration:
    def test_pre03_full_hash_history_is_recognized_once_then_fast(self, ws):
        """A legacy SQLite row keyed by FULL SHA-256 must (a) skip the file
        and (b) leave a fast-keyed migration record so the full hash is
        never recomputed again."""
        paths, media = ws
        cfg = _cfg()
        full = file_signature(media, "full")
        fast = file_signature(media, "fast")
        legacy_row = _record_from_metadata(
            {
                "audio_file": media.name,
                "processed_at": "2026-01-01T00:00:00+00:00",
                "model": "large-v3-turbo",
                "diarization_model": None,
                "config": {},
            },
            full,
            1,
            None,
            "ok",
            None,
            1,
        )
        _sqlite_register(legacy_row, paths.db_path)
        base = f"{media.stem}_{cfg.model}"
        (paths.transcripts / f"{base}.json").write_text("{}")
        (paths.transcripts / f"{base}.txt").write_text("x")

        result = process_one(media, paths, FakeModel(), cfg)
        assert result["status"] == "skipped"
        migrated = find_run_by_hash(paths.ledger_path, fast, "large-v3-turbo", None)
        assert migrated is not None, "migration record under the fast signature"
        assert json.loads(migrated["config_json"])["migrated_from_full_hash"] == full


class TestDegradedDiarization:
    def test_diar_failure_yields_ok_degraded_with_critical_flag(self, ws):
        paths, media = ws
        cfg = _cfg(enable_diarization=True, evaluate_quality=True)
        engine = StubEngine(error=RuntimeError("HF token rejected"))
        meta = process_one(media, paths, FakeModel(), cfg, diar_engine=engine)
        assert meta["status"] == "ok_degraded"
        assert meta["diarization_failed"] is True
        assert any("DIARIZATION_FAILED" in f for f in meta["quality_flags"])
        rec = find_run_by_hash(paths.ledger_path, meta["file_hash"], cfg.model, None)
        assert rec["status"] == "ok_degraded"

    def test_degraded_run_does_not_block_future_diarized_run(self, ws):
        paths, media = ws
        cfg = _cfg(enable_diarization=True, evaluate_quality=True)
        process_one(
            media, paths, FakeModel(), cfg, diar_engine=StubEngine(error=RuntimeError("boom"))
        )
        ok = process_one(
            media, paths, FakeModel(), cfg, diar_engine=StubEngine(turns=TWO_SPEAKER_TURNS)
        )
        assert ok["status"] == "ok", "degraded row (diar_model=None) must not satisfy the lookup"
        assert ok["speakers_summary"]

    def test_engine_reused_not_reloaded(self, ws):
        paths, media = ws
        cfg = _cfg(enable_diarization=True)
        engine = StubEngine(turns=TWO_SPEAKER_TURNS)
        process_one(media, paths, FakeModel(), cfg, diar_engine=engine)
        process_one(
            media,
            paths,
            FakeModel(),
            cfg.model_copy(update={"force_reprocess": True}),
            diar_engine=engine,
        )
        assert engine.calls == 2


def _report(critical_codes: list[str]) -> QualityReport:
    flags = [
        QualityFlag(severity=Severity.CRITICAL, code=c, message=c, context={})
        for c in critical_codes
    ]
    return QualityReport(quality_ok=not flags, flags=flags, metrics={})


class TestAutoRetry:
    def _media_with_two_attempts(self, ws):
        """Fake model: attempt 1 emits looped garbage, attempt 2 clean text."""
        bad = [FakeSegment(0, 1, " mil gracias mil gracias", [FakeWord(" mil", 0, 0.4)])]
        good = [FakeSegment(0, 1, " contenido limpio", [FakeWord(" contenido", 0, 0.4)])]
        return FakeModel(segments_per_call=[bad, good])

    def test_retry_promotes_better_attempt(self, ws, monkeypatch):
        paths, media = ws
        import speakerscribe.pipeline as pl

        reports = iter([_report(["REPETITIONS"]), _report([])])
        monkeypatch.setattr(pl, "evaluate_transcription_quality", lambda m: next(reports))

        cfg = _cfg(evaluate_quality=True, auto_retry_on_critical=True)
        model = self._media_with_two_attempts(ws)
        meta = process_one(media, paths, model, cfg)

        assert meta["attempt_kept"] == 2
        assert meta["attempts"] == [
            {"attempt": 1, "n_critical": 1, "kept": False},
            {"attempt": 2, "n_critical": 0, "kept": True},
        ]
        txt = (paths.transcripts / f"{meta['base_name']}.txt").read_text()
        assert "contenido limpio" in txt and "mil gracias" not in txt
        # retry decoding used anti-loop overrides
        retry_call = model.calls[-1]
        assert retry_call["condition_on_previous_text"] is False
        assert retry_call["repetition_penalty"] == 1.15
        assert retry_call["no_repeat_ngram_size"] == 3
        # both attempts ledgered: losing attempt as 'retried', winner as 'ok'
        lines = [json.loads(line) for line in paths.ledger_path.read_text().splitlines()]
        statuses = [(r["status"], r["attempt"]) for r in lines]
        assert ("retried", 1) in statuses and ("ok", 2) in statuses
        # no .retry leftovers
        assert not list(paths.transcripts.glob("*.retry"))

    def test_retry_kept_first_when_not_better(self, ws, monkeypatch):
        paths, media = ws
        import speakerscribe.pipeline as pl

        reports = iter([_report(["REPETITIONS"]), _report(["REPETITIONS"])])
        monkeypatch.setattr(pl, "evaluate_transcription_quality", lambda m: next(reports))

        cfg = _cfg(evaluate_quality=True, auto_retry_on_critical=True)
        model = self._media_with_two_attempts(ws)
        meta = process_one(media, paths, model, cfg)
        assert meta["attempt_kept"] == 1
        txt = (paths.transcripts / f"{meta['base_name']}.txt").read_text()
        assert "mil gracias" in txt
        assert not list(paths.transcripts.glob("*.retry"))

    def test_no_retry_when_disabled_or_clean(self, ws, monkeypatch):
        paths, media = ws
        import speakerscribe.pipeline as pl

        monkeypatch.setattr(pl, "evaluate_transcription_quality", lambda m: _report([]))
        model = self._media_with_two_attempts(ws)
        meta = process_one(
            media, paths, model, _cfg(evaluate_quality=True, auto_retry_on_critical=True)
        )
        assert meta["attempt_kept"] == 1
        assert len(model.calls) == 1


class TestWavBySignature:
    def test_wav_name_embeds_content_signature(self, ws, monkeypatch):
        paths, media = ws
        import speakerscribe.pipeline as pl

        created: list[str] = []

        def fake_extract(input_file, output_file, sample_rate, *, timeout_s=0):
            created.append(output_file.name)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_bytes(b"wav")
            return output_file

        monkeypatch.setattr(pl, "extract_audio_wav", fake_extract)
        cfg = _cfg(extract_temp_wav=True, delete_temp_wav=True)
        meta = process_one(media, paths, FakeModel(), cfg)
        sig = meta["file_hash"]
        assert created == [f"{media.stem}_{sig[:10]}.wav"]
        assert not list(paths.audio_tmp.glob("*.wav")), "temp wav deleted after run"
