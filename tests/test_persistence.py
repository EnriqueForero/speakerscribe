"""Runs ledger: JSONL backend, legacy SQLite fallback, torn-line tolerance."""

from __future__ import annotations

import json

from speakerscribe.persistence import (
    _record_from_metadata,
    _sqlite_register,
    find_run_by_hash,
    global_stats,
    list_runs,
    register_run,
)


def _meta(**over):
    base = {
        "audio_file": "a.mp4",
        "processed_at": "2026-06-11T00:00:00+00:00",
        "duration_minutes": 10.0,
        "elapsed_seconds": 60.0,
        "real_time_factor": 10.0,
        "model": "large-v3-turbo",
        "diarization_model": "pyannote/x",
        "speakers_summary": {"S0": 5, "S1": 4},
        "total_segments": 9,
        "total_words": 100,
        "config": {},
        "package_version": "0.3.0",
    }
    base.update(over)
    return base


class TestJsonlBackend:
    def test_register_and_find(self, tmp_path):
        ledger = tmp_path / "_runs.jsonl"
        run_id = register_run(ledger, _meta(), file_hash="H1")
        assert run_id > 0
        rec = find_run_by_hash(ledger, "H1", "large-v3-turbo", "pyannote/x")
        assert rec is not None
        assert rec["n_words"] == 100
        assert rec["attempt"] == 1

    def test_last_record_wins_for_same_key(self, tmp_path):
        ledger = tmp_path / "_runs.jsonl"
        register_run(ledger, _meta(), file_hash="H1", status="ok_degraded", attempt=1)
        register_run(ledger, _meta(), file_hash="H1", status="ok", attempt=2)
        rec = find_run_by_hash(ledger, "H1", "large-v3-turbo", "pyannote/x")
        assert rec["status"] == "ok"
        assert rec["attempt"] == 2

    def test_diar_model_none_is_a_distinct_key(self, tmp_path):
        """A degraded run (diar_model=None) must not satisfy a diarized lookup."""
        ledger = tmp_path / "_runs.jsonl"
        register_run(ledger, _meta(diarization_model=None), file_hash="H1", status="ok_degraded")
        assert find_run_by_hash(ledger, "H1", "large-v3-turbo", "pyannote/x") is None
        assert find_run_by_hash(ledger, "H1", "large-v3-turbo", None) is not None

    def test_torn_trailing_line_is_skipped(self, tmp_path):
        ledger = tmp_path / "_runs.jsonl"
        register_run(ledger, _meta(), file_hash="H1")
        with ledger.open("a") as f:
            f.write('{"torn": ')  # simulated crash mid-append
        assert find_run_by_hash(ledger, "H1", "large-v3-turbo", "pyannote/x") is not None

    def test_missing_ledger_returns_none(self, tmp_path):
        assert find_run_by_hash(tmp_path / "_runs.jsonl", "X", "m", None) is None


class TestLegacySqliteFallback:
    def test_jsonl_miss_falls_back_to_sibling_db(self, tmp_path):
        """Pre-0.3 SQLite histories keep preventing reprocessing."""
        legacy = tmp_path / "_runs.db"
        rec = _record_from_metadata(_meta(audio_file="old.mp4"), "FULL", 1, None, "ok", None, 1)
        _sqlite_register(rec, legacy)
        hit = find_run_by_hash(tmp_path / "_runs.jsonl", "FULL", "large-v3-turbo", "pyannote/x")
        assert hit is not None
        assert hit["audio_file"] == "old.mp4"

    def test_list_runs_merges_both_sources(self, tmp_path):
        legacy = tmp_path / "_runs.db"
        _sqlite_register(
            _record_from_metadata(_meta(audio_file="old.mp4"), "FULL", 1, None, "ok", None, 1),
            legacy,
        )
        ledger = tmp_path / "_runs.jsonl"
        register_run(ledger, _meta(audio_file="new.mp4"), file_hash="FAST")
        runs = list_runs(ledger, limit=10)
        names = {r["audio_file"] for r in runs}
        assert names == {"old.mp4", "new.mp4"}

    def test_jsonl_overrides_legacy_on_same_key(self, tmp_path):
        legacy = tmp_path / "_runs.db"
        _sqlite_register(_record_from_metadata(_meta(), "H1", 0, None, "error", "boom", 1), legacy)
        ledger = tmp_path / "_runs.jsonl"
        register_run(ledger, _meta(), file_hash="H1", status="ok")
        runs = list_runs(ledger, limit=10)
        assert len(runs) == 1
        assert runs[0]["status"] == "ok"


class TestGlobalStats:
    def test_empty(self, tmp_path):
        assert global_stats(tmp_path / "_runs.jsonl") == {"total_runs": 0}

    def test_aggregates(self, tmp_path):
        ledger = tmp_path / "_runs.jsonl"
        register_run(ledger, _meta(), file_hash="A", status="ok")
        register_run(ledger, _meta(audio_file="b.mp4"), file_hash="B", status="ok_degraded")
        register_run(ledger, _meta(audio_file="c.mp4"), file_hash="C", status="error")
        s = global_stats(ledger)
        assert s["total_runs"] == 3
        assert s["ok_runs"] == 1
        assert s["ok_degraded_runs"] == 1
        assert s["error_runs"] == 1
        assert s["avg_rtf"] == 10.0
        assert s["total_words"] == 200

    def test_status_filter(self, tmp_path):
        ledger = tmp_path / "_runs.jsonl"
        register_run(ledger, _meta(), file_hash="A", status="ok")
        register_run(ledger, _meta(audio_file="b.mp4"), file_hash="B", status="error")
        only_err = list_runs(ledger, limit=10, status="error")
        assert len(only_err) == 1 and only_err[0]["file_hash"] == "B"


class TestSqliteDirect:
    def test_db_path_still_fully_functional(self, tmp_path):
        """Local (non-Drive) users can keep using SQLite by passing a .db path."""
        db = tmp_path / "runs.db"
        register_run(db, _meta(), file_hash="H1")
        assert find_run_by_hash(db, "H1", "large-v3-turbo", "pyannote/x") is not None
        assert global_stats(db)["total_runs"] == 1
        assert json.loads(list_runs(db, 1)[0]["config_json"]) == {}
