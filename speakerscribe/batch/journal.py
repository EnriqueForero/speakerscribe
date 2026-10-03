"""Append-only event journal (``events.jsonl``) and its in-memory indices.

The journal is the source of truth of the batch: every state transition is
one JSON line, fsync'ed. Readers tolerate a truncated last line (a session
killed mid-write). Event names and fields are a superset of notebook v5's,
so the 463 events recorded until 2026-10-01 index exactly as before.
"""

from __future__ import annotations

import contextlib
import json
import os
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from speakerscribe.batch.fsio import utc_now
from speakerscribe.environment import is_environment_error_text

JOURNAL_SCHEMA = 6
"""v5 wrote 5; v6 only adds event types and fields (backward compatible)."""


class Event(str, Enum):
    """Journal event types."""

    BATCH_STARTED = "batch_started"
    BATCH_FINISHED = "batch_finished"
    BATCH_ABORTED = "batch_aborted"
    PROCESSING = "processing"
    PREPARED = "prepared"
    COMPLETED = "completed"
    REUSED = "reused"
    REPUBLISHED = "republished"
    PREPARED_ABANDONED = "prepared_abandoned"
    JOB_METRICS = "job_metrics"
    FAILED_RETRYABLE = "failed_retryable"
    FAILED_ENVIRONMENT = "failed_environment"
    QUALITY_REJECTED = "quality_rejected"
    INVALID_MEDIA = "invalid_media"
    PROBE_RETRYABLE = "probe_retryable"
    SOURCE_CHANGED = "source_changed"
    CONTENT_CHANGE_DETECTED = "content_change_detected"
    CONTENT_CHANGE_RESOLVED = "content_change_resolved"
    PROFILE_CHANGE_DEFERRED = "profile_change_deferred"
    SPEAKERS_RENAMED = "speakers_renamed"
    MODELS_RECYCLED = "models_recycled"
    DIARIZATION_SETUP_FAILED = "diarization_setup_failed"
    ENVIRONMENT_FAILURE_SUSPECTED = "environment_failure_suspected"
    ATTEMPTS_RESET = "attempts_reset"
    WORKSPACE_REBOUND = "workspace_rebound"
    SOURCE_RETIRED = "source_retired"
    PROCESSED_PURGED = "processed_purged"
    DIAR_CACHE_PERSISTED = "diar_cache_persisted"
    DIAR_CACHE_PRUNED = "diar_cache_pruned"


SUCCESS_EVENTS = frozenset({Event.COMPLETED.value, Event.REUSED.value, Event.REPUBLISHED.value})
ATTEMPT_EVENTS = frozenset({Event.FAILED_RETRYABLE.value, Event.QUALITY_REJECTED.value})


def _environmental_failure(rec: dict[str, Any]) -> bool:
    """A ``failed_retryable`` that today's rules classify as environmental.

    Errors are re-read with the current classifier, so a failure journaled
    as per-file before its marker existed (``libcublas.so.12`` on
    2026-10-03) stops consuming a retry attempt. Diarization failures of one
    file (``stage == "diar_file"``) always count.
    """
    return (
        rec.get("event") == Event.FAILED_RETRYABLE.value
        and rec.get("stage") != "diar_file"
        and is_environment_error_text(rec.get("error"))
    )


"""Events that consume one retry attempt of a job. Environment failures and
probe/stability hiccups never do: retrying them cannot change the outcome."""


class Journal:
    """Append-only JSONL journal bound to one run id.

    Args:
        path: ``events.jsonl`` location.
        run_id: Identifier stamped on every event of this run.
    """

    def __init__(self, path: Path, run_id: str | None = None) -> None:
        self.path = path
        self.run_id = run_id or uuid.uuid4().hex

    def append(self, event: Event | str, **fields: Any) -> dict[str, Any]:
        """Append one event durably (flush + fsync) and return it."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema_version": JOURNAL_SCHEMA,
            "event_id": uuid.uuid4().hex,
            "run_id": self.run_id,
            "timestamp_utc": utc_now(),
            "event": event.value if isinstance(event, Event) else str(event),
            **fields,
        }
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            with contextlib.suppress(OSError):
                os.fsync(fh.fileno())
        return record

    def read(self) -> list[dict[str, Any]]:
        """All events, skipping blank and truncated lines."""
        if not self.path.exists():
            return []
        events: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue  # last line cut by a killed session
                if isinstance(value, dict):
                    events.append(value)
        return events


@dataclass
class JournalIndex:
    """Lookup tables built once per session from the journal.

    Attributes:
        latest_by_job: Last event of each job id.
        ok_by_source: Success records (completed/reused/republished) per source id.
        ok_by_content_profile: Success records per (strong signature, profile id),
            used for dedup by content.
        invalid: Last ``invalid_media`` record per source id.
        dirty: Sources whose content changed and await resolution.
        attempts: Consumed retry attempts per job id.
        renames: Last speaker-rename hash per source id.
    """

    latest_by_job: dict[str, dict[str, Any]] = field(default_factory=dict)
    ok_by_source: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    ok_by_content_profile: dict[tuple[str, str], list[dict[str, Any]]] = field(default_factory=dict)
    invalid: dict[str, dict[str, Any]] = field(default_factory=dict)
    dirty: set[str] = field(default_factory=set)
    attempts: dict[str, int] = field(default_factory=dict)
    renames: dict[str, str] = field(default_factory=dict)

    @classmethod
    def build(cls, events: list[dict[str, Any]]) -> JournalIndex:
        """Fold the event list into indices (order matters: later wins)."""
        index = cls()
        for rec in events:
            job_id = rec.get("job_id")
            if job_id:
                index.latest_by_job[job_id] = rec
            event = rec.get("event")
            source = rec.get("source") or {}
            sid = source.get("id")
            if event in SUCCESS_EVENTS and sid:
                index.record_success(rec)
            elif event == Event.INVALID_MEDIA.value and sid:
                index.invalid[sid] = rec
            elif event == Event.CONTENT_CHANGE_DETECTED.value and sid:
                index.dirty.add(sid)
            elif event == Event.CONTENT_CHANGE_RESOLVED.value and sid:
                index.dirty.discard(sid)
            elif event == Event.SPEAKERS_RENAMED.value and sid:
                index.renames[sid] = str(rec.get("rename_sha") or "")
            if job_id and event in ATTEMPT_EVENTS and not _environmental_failure(rec):
                index.attempts[job_id] = index.attempts.get(job_id, 0) + 1
            elif job_id and event == Event.ATTEMPTS_RESET.value:
                index.attempts.pop(job_id, None)
        return index

    def record_success(self, rec: dict[str, Any]) -> None:
        """Index a success record (also used live, right after a commit)."""
        source = rec.get("source") or {}
        sid = source.get("id")
        if not sid:
            return
        self.dirty.discard(sid)
        self.ok_by_source.setdefault(sid, []).append(rec)
        signature = source.get("content_signature")
        profile_id = (rec.get("profile") or {}).get("id")
        if signature and not str(signature).startswith("fast:") and profile_id:
            self.ok_by_content_profile.setdefault((signature, profile_id), []).append(rec)

    def latest_success(self, source_id: str) -> dict[str, Any] | None:
        """Most recent success record of a source."""
        for rec in reversed(self.ok_by_source.get(source_id, [])):
            if rec.get("event") in SUCCESS_EVENTS:
                return rec
        return None


def unfinished_signatures(events: list[dict[str, Any]]) -> set[str]:
    """Content signatures that were attempted but never published.

    Keyed on content, not location, so it survives a workspace rebind (the
    source ids change, the bytes do not). Their diarization caches are
    still worth keeping.
    """
    attempted: set[str] = set()
    done: set[str] = set()
    for rec in events:
        signature = (rec.get("source") or {}).get("content_signature")
        if not signature:
            continue
        if rec.get("event") in SUCCESS_EVENTS:
            done.add(str(signature))
        elif rec.get("event") in {
            Event.PROCESSING.value,
            Event.FAILED_RETRYABLE.value,
            Event.FAILED_ENVIRONMENT.value,
            Event.QUALITY_REJECTED.value,
        }:
            attempted.add(str(signature))
    return attempted - done


def previous_input_roots(events: list[dict[str, Any]]) -> set[str]:
    """``input_root_sha256`` of every recorded ``batch_started``."""
    roots = {
        (rec.get("settings") or {}).get("input_root_sha256")
        for rec in events
        if rec.get("event") == Event.BATCH_STARTED.value
    }
    roots.discard(None)
    return {str(r) for r in roots}


__all__ = [
    "ATTEMPT_EVENTS",
    "JOURNAL_SCHEMA",
    "SUCCESS_EVENTS",
    "Event",
    "Journal",
    "JournalIndex",
    "previous_input_roots",
    "unfinished_signatures",
]
