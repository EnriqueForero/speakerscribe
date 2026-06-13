"""Run history: append-only JSON-Lines ledger (primary) + legacy SQLite.

Enabled by default (`TranscriptionConfig.enable_runs_db=True`): every run is
logged under the workspace for robust content-hash idempotency.

Why JSON-Lines replaced SQLite as the default backend in 0.3:
    The ledger lives in the workspace, which on Colab is Google Drive
    mounted via FUSE. SQLite depends on POSIX file locking that Drive/FUSE
    does not guarantee — long or interrupted sessions risk corruption and
    phantom locks. An append-only JSONL file needs no locking, survives
    mid-write crashes (at worst one torn trailing line, which readers
    skip), and is human-readable.

Backend selection is path-driven and the public API is unchanged:
    - ``*.jsonl`` -> JSONL backend (primary). Lookups that miss fall back
      to a sibling ``_runs.db`` (read-only) so pre-0.3 histories keep
      working; ``list_runs``/``global_stats`` merge both sources.
    - ``*.db``    -> SQLite backend (still fully functional for local,
      non-Drive use).

Record semantics: one line per run attempt; for a given key
``(file_hash, asr_model, diar_model)`` the LAST line wins (same effect as
SQLite's ``INSERT OR REPLACE``).
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from speakerscribe.io_utils import append_jsonl_line
from speakerscribe.logging_config import logger

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    audio_file      TEXT NOT NULL,
    file_hash       TEXT NOT NULL,
    processed_at    TEXT NOT NULL,
    duration_min    REAL,
    elapsed_sec     REAL,
    rtf             REAL,
    asr_model       TEXT,
    diar_model      TEXT,
    n_speakers      INTEGER,
    n_segments      INTEGER,
    n_words         INTEGER,
    quality_ok      INTEGER,           -- 0/1
    quality_flags   TEXT,              -- JSON
    config_json     TEXT,
    package_version TEXT,
    status          TEXT NOT NULL,     -- ok | ok_degraded | error | skipped
    error_message   TEXT,
    UNIQUE(file_hash, asr_model, diar_model)
);

CREATE INDEX IF NOT EXISTS idx_runs_audio_file  ON runs(audio_file);
CREATE INDEX IF NOT EXISTS idx_runs_file_hash   ON runs(file_hash);
CREATE INDEX IF NOT EXISTS idx_runs_processed   ON runs(processed_at);
CREATE INDEX IF NOT EXISTS idx_runs_status      ON runs(status);
"""

_LEGACY_DB_NAME = "_runs.db"


def _is_jsonl(path: Path) -> bool:
    return path.suffix.lower() == ".jsonl"


def _legacy_db_for(ledger_path: Path) -> Path:
    """Sibling legacy SQLite path for a JSONL ledger."""
    return ledger_path.with_name(_LEGACY_DB_NAME)


def _record_from_metadata(
    metadata: dict[str, Any],
    file_hash: str,
    quality_ok: bool | None,
    quality_flags_json: str | None,
    status: str,
    error_message: str | None,
    attempt: int,
) -> dict[str, Any]:
    """Normalize run metadata into a flat ledger record (shared by backends)."""
    return {
        "id": time.time_ns() // 1_000,  # monotonic-enough unique id (µs)
        "audio_file": metadata.get("audio_file"),
        "file_hash": file_hash,
        "processed_at": metadata.get("processed_at"),
        "duration_min": metadata.get("duration_minutes"),
        "elapsed_sec": metadata.get("elapsed_seconds"),
        "rtf": metadata.get("real_time_factor"),
        "asr_model": metadata.get("model"),
        "diar_model": metadata.get("diarization_model"),
        "n_speakers": len(metadata.get("speakers_summary") or {}) or None,
        "n_segments": metadata.get("total_segments"),
        "n_words": metadata.get("total_words"),
        "quality_ok": int(quality_ok) if quality_ok is not None else None,
        "quality_flags": quality_flags_json,
        "config_json": json.dumps(metadata.get("config", {}), ensure_ascii=False),
        "package_version": metadata.get("package_version"),
        "status": status,
        "error_message": error_message,
        "attempt": attempt,
    }


# ─── JSONL backend ─────────────────────────────────────────────────
def _iter_jsonl_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield records from a JSONL ledger, skipping torn/garbage lines.

    A crash during append can leave one incomplete trailing line; that is
    by design recoverable — we log it at DEBUG and move on.
    """
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logger.debug(f"Skipping malformed ledger line {n} in {path.name}")
                continue
            if isinstance(rec, dict):
                yield rec


def _jsonl_find(
    path: Path,
    file_hash: str,
    asr_model: str,
    diar_model: str | None,
) -> dict[str, Any] | None:
    """Last record matching the key wins (append-only REPLACE semantics)."""
    found: dict[str, Any] | None = None
    for rec in _iter_jsonl_records(path):
        if (
            rec.get("file_hash") == file_hash
            and rec.get("asr_model") == asr_model
            and rec.get("diar_model") == diar_model
        ):
            found = rec
    return found


def _dedupe_latest(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse records to the latest per (file_hash, asr_model, diar_model)."""
    latest: dict[tuple, dict[str, Any]] = {}
    for rec in records:  # input is in append order -> later overwrites
        key = (rec.get("file_hash"), rec.get("asr_model"), rec.get("diar_model"))
        latest[key] = rec
    return list(latest.values())


# ─── SQLite backend (legacy / local) ───────────────────────────────
@contextmanager
def _connection(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Context manager for SQLite with automatic commit and Row factory."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA_SQL)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _sqlite_register(record: dict[str, Any], db_path: Path) -> int:
    with _connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT OR REPLACE INTO runs (
                audio_file, file_hash, processed_at,
                duration_min, elapsed_sec, rtf,
                asr_model, diar_model,
                n_speakers, n_segments, n_words,
                quality_ok, quality_flags,
                config_json, package_version,
                status, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["audio_file"],
                record["file_hash"],
                record["processed_at"],
                record["duration_min"],
                record["elapsed_sec"],
                record["rtf"],
                record["asr_model"],
                record["diar_model"],
                record["n_speakers"],
                record["n_segments"],
                record["n_words"],
                record["quality_ok"],
                record["quality_flags"],
                record["config_json"],
                record["package_version"],
                record["status"],
                record["error_message"],
            ),
        )
        return cur.lastrowid or 0


def _sqlite_find(
    db_path: Path,
    file_hash: str,
    asr_model: str,
    diar_model: str | None,
) -> dict[str, Any] | None:
    if not db_path.exists():
        return None
    with _connection(db_path) as conn:
        if diar_model is None:
            cur = conn.execute(
                "SELECT * FROM runs WHERE file_hash=? AND asr_model=? AND diar_model IS NULL",
                (file_hash, asr_model),
            )
        else:
            cur = conn.execute(
                "SELECT * FROM runs WHERE file_hash=? AND asr_model=? AND diar_model=?",
                (file_hash, asr_model, diar_model),
            )
        row = cur.fetchone()
        return dict(row) if row else None


def _sqlite_list(db_path: Path, limit: int, status: str | None) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    with _connection(db_path) as conn:
        if status:
            cur = conn.execute(
                "SELECT * FROM runs WHERE status=? ORDER BY id DESC LIMIT ?",
                (status, limit),
            )
        else:
            cur = conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(r) for r in cur.fetchall()]


# ─── Public API (backend-agnostic) ─────────────────────────────────
def register_run(
    db_path: Path,
    metadata: dict[str, Any],
    file_hash: str,
    quality_ok: bool | None = None,
    quality_flags_json: str | None = None,
    status: str = "ok",
    error_message: str | None = None,
    attempt: int = 1,
) -> int:
    """Record a run in the ledger (JSONL append) or database (SQLite REPLACE).

    Args:
        db_path: ``*.jsonl`` ledger (primary) or ``*.db`` SQLite path.
        metadata: Run metadata, typically the dict returned by
            ``transcribe_streaming``.
        file_hash: Content signature of the source audio (see
            ``audio.file_signature``).
        quality_ok: True/False/None — result of the quality check.
        quality_flags_json: JSON-serialized list of flag strings.
        status: "ok" | "ok_degraded" | "error" | "skipped".
        error_message: Error message when status="error".
        attempt: 1 for the first pass; 2 for an auto-retry pass. Both
            attempts of a retried run are recorded for auditability.

    Returns:
        Record id (microsecond timestamp for JSONL; rowid for SQLite).
    """
    record = _record_from_metadata(
        metadata, file_hash, quality_ok, quality_flags_json, status, error_message, attempt
    )
    if _is_jsonl(db_path):
        append_jsonl_line(db_path, record)
        run_id = int(record["id"])
    else:
        run_id = _sqlite_register(record, db_path)
    logger.debug(f"Run registered: id={run_id} hash={file_hash[:8]} status={status}")
    return run_id


def find_run_by_hash(
    db_path: Path,
    file_hash: str,
    asr_model: str,
    diar_model: str | None = None,
) -> dict[str, Any] | None:
    """Look up a previous run by content hash + model combination.

    If the file was renamed but its content is identical, this still detects
    the duplicate. On a JSONL ledger, a miss transparently falls back to the
    sibling legacy ``_runs.db`` (read-only) so histories recorded before 0.3
    keep preventing reprocessing.

    Args:
        db_path: ``*.jsonl`` ledger or ``*.db`` SQLite path.
        file_hash: Content signature of the source audio.
        asr_model: Whisper model name used for the run.
        diar_model: Diarization model id (or None for no-diarization runs).
            Note: a run whose diarization FAILED is recorded with
            ``diar_model=None`` and ``status="ok_degraded"``, so it never
            blocks a future run that succeeds at diarizing.

    Returns:
        Dict with the record if found, else None.
    """
    if _is_jsonl(db_path):
        rec = _jsonl_find(db_path, file_hash, asr_model, diar_model)
        if rec is not None:
            return rec
        return _sqlite_find(_legacy_db_for(db_path), file_hash, asr_model, diar_model)
    return _sqlite_find(db_path, file_hash, asr_model, diar_model)


def list_runs(
    db_path: Path,
    limit: int = 50,
    status: str | None = None,
) -> list[dict[str, Any]]:
    """List the most recent runs, newest first.

    On a JSONL ledger, legacy SQLite rows are merged in (deduplicated by
    run key, ledger wins) so the full history stays visible after the 0.3
    backend migration.

    Args:
        db_path: ``*.jsonl`` ledger or ``*.db`` SQLite path.
        limit: Maximum number of records to return.
        status: Filter by status or None for all.

    Returns:
        List of dicts, newest first.
    """
    if not _is_jsonl(db_path):
        return _sqlite_list(db_path, limit, status)

    legacy = _sqlite_list(_legacy_db_for(db_path), limit=1_000_000, status=None)
    legacy.reverse()  # oldest first, so JSONL records override on dedupe
    merged = _dedupe_latest(legacy + list(_iter_jsonl_records(db_path)))
    if status:
        merged = [r for r in merged if r.get("status") == status]
    merged.sort(key=lambda r: (r.get("processed_at") or "", r.get("id") or 0), reverse=True)
    return merged[:limit]


def global_stats(db_path: Path) -> dict[str, Any]:
    """Compute aggregate statistics over all runs (latest attempt per key).

    Args:
        db_path: ``*.jsonl`` ledger or ``*.db`` SQLite path.

    Returns:
        Dict with totals, average RTF, total hours processed, etc.
        ``{"total_runs": 0}`` when no history exists.
    """
    records = list_runs(db_path, limit=1_000_000, status=None)
    if not records:
        return {"total_runs": 0}

    ok = [r for r in records if r.get("status") in ("ok", "ok_degraded")]
    rtfs = [r["rtf"] for r in ok if isinstance(r.get("rtf"), int | float)]
    durations = [r["duration_min"] for r in ok if isinstance(r.get("duration_min"), int | float)]
    words = [r["n_words"] for r in ok if isinstance(r.get("n_words"), int | float)]
    return {
        "total_runs": len(records),
        "ok_runs": sum(1 for r in records if r.get("status") == "ok"),
        "ok_degraded_runs": sum(1 for r in records if r.get("status") == "ok_degraded"),
        "error_runs": sum(1 for r in records if r.get("status") == "error"),
        "skipped_runs": sum(1 for r in records if r.get("status") == "skipped"),
        "avg_rtf": round(sum(rtfs) / len(rtfs), 2) if rtfs else None,
        "hours_processed": round(sum(durations) / 60, 2) if durations else None,
        "total_words": int(sum(words)) if words else None,
    }


__all__ = [
    "find_run_by_hash",
    "global_stats",
    "list_runs",
    "register_run",
]
