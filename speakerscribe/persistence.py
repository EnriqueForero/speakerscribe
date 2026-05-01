"""Optional SQLite-backed run history.

When `TranscriptionConfig.enable_runs_db=True`, every run is logged to a
local `_runs.db` under the workspace. Disabled by default — most users
process audios one at a time and do not need a history database.

Benefits when enabled:
    - Robust idempotency by file hash (not by filename)
    - Queryable history: how many audios, average RTF, etc.
    - Audit trail for institutional use

The schema is intentionally simple: ~15 columns with appropriate indices.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

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
    status          TEXT NOT NULL,     -- ok | error | skipped
    error_message   TEXT,
    UNIQUE(file_hash, asr_model, diar_model)
);

CREATE INDEX IF NOT EXISTS idx_runs_audio_file  ON runs(audio_file);
CREATE INDEX IF NOT EXISTS idx_runs_file_hash   ON runs(file_hash);
CREATE INDEX IF NOT EXISTS idx_runs_processed   ON runs(processed_at);
CREATE INDEX IF NOT EXISTS idx_runs_status      ON runs(status);
"""


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


def register_run(
    db_path: Path,
    metadata: dict[str, Any],
    file_hash: str,
    quality_ok: bool | None = None,
    quality_flags_json: str | None = None,
    status: str = "ok",
    error_message: str | None = None,
) -> int:
    """Insert (or replace) a run row in the database.

    Conflict on UNIQUE(file_hash, asr_model, diar_model) triggers a REPLACE.

    Args:
        db_path: Path to the .db file.
        metadata: Run metadata, typically the dict returned by
            `transcribe_streaming`.
        file_hash: SHA-256 of the source audio.
        quality_ok: True/False/None — result of the quality check.
        quality_flags_json: JSON-serialized list of flag strings.
        status: "ok" | "error" | "skipped".
        error_message: Error message when status="error".

    Returns:
        Row id of the inserted (or replaced) record.
    """
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
                metadata.get("audio_file"),
                file_hash,
                metadata.get("processed_at"),
                metadata.get("duration_minutes"),
                metadata.get("elapsed_seconds"),
                metadata.get("real_time_factor"),
                metadata.get("model"),
                metadata.get("diarization_model"),
                len(metadata.get("speakers_summary") or {}) or None,
                metadata.get("total_segments"),
                metadata.get("total_words"),
                int(quality_ok) if quality_ok is not None else None,
                quality_flags_json,
                json.dumps(metadata.get("config", {}), ensure_ascii=False),
                metadata.get("package_version"),
                status,
                error_message,
            ),
        )
        run_id = cur.lastrowid or 0
        logger.debug(f"Run registered in DB: id={run_id} hash={file_hash[:8]}")
        return run_id


def find_run_by_hash(
    db_path: Path,
    file_hash: str,
    asr_model: str,
    diar_model: str | None = None,
) -> dict[str, Any] | None:
    """Look up a previous run by content hash + model combination.

    If the file was renamed but its content is identical, this still detects
    the duplicate.

    Args:
        db_path: Path to the .db file.
        file_hash: SHA-256 of the source audio.
        asr_model: Whisper model name used for the run.
        diar_model: Diarization model id (or None for no-diarization runs).

    Returns:
        Dict with the row data if found, else None.
    """
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


def list_runs(
    db_path: Path,
    limit: int = 50,
    status: str | None = None,
) -> list[dict[str, Any]]:
    """List the most recent runs.

    Args:
        db_path: Path to the .db file.
        limit: Maximum number of rows to return.
        status: Filter by status (ok/error/skipped) or None for all.

    Returns:
        List of dicts, newest first.
    """
    if not db_path.exists():
        return []

    with _connection(db_path) as conn:
        if status:
            cur = conn.execute(
                "SELECT * FROM runs WHERE status=? ORDER BY id DESC LIMIT ?",
                (status, limit),
            )
        else:
            cur = conn.execute(
                "SELECT * FROM runs ORDER BY id DESC LIMIT ?",
                (limit,),
            )
        return [dict(r) for r in cur.fetchall()]


def global_stats(db_path: Path) -> dict[str, Any]:
    """Compute aggregate statistics over all runs.

    Args:
        db_path: Path to the .db file.

    Returns:
        Dict with totals, average RTF, total hours processed, etc. If the DB
        does not exist, returns `{"total_runs": 0}`.
    """
    if not db_path.exists():
        return {"total_runs": 0}

    with _connection(db_path) as conn:
        cur = conn.execute(
            """
            SELECT
                COUNT(*)                                          AS total_runs,
                SUM(CASE WHEN status='ok' THEN 1 ELSE 0 END)      AS ok_runs,
                SUM(CASE WHEN status='error' THEN 1 ELSE 0 END)   AS error_runs,
                SUM(CASE WHEN status='skipped' THEN 1 ELSE 0 END) AS skipped_runs,
                ROUND(AVG(rtf), 2)                                AS avg_rtf,
                ROUND(SUM(duration_min) / 60, 2)                  AS hours_processed,
                SUM(n_words)                                      AS total_words
            FROM runs
            """
        )
        row = cur.fetchone()
        return dict(row) if row else {"total_runs": 0}


__all__ = [
    "find_run_by_hash",
    "global_stats",
    "list_runs",
    "register_run",
]
