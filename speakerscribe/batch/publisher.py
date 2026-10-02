"""Render, write and transactionally commit every deliverable of one recording.

Commit protocol (notebook v5, unchanged on disk):

1. Secondary deliverables are written atomically (idempotent).
2. The canonical ``.txt`` is written COMPLETE as ``<name>.txt.part.<job>``.
3. ``prepared`` is journaled with the part's size and SHA-256.
4. Lock ownership is verified, then ``os.replace(part, final)``.
5. ``completed`` / ``reused`` / ``republished`` is journaled.

A session killed between 3 and 5 is repaired by `Publisher.recover_prepared`
on the next run: it promotes the part if it matches, accepts the final if it
already matches, or journals ``prepared_abandoned``.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from speakerscribe.batch.fsio import atomic_write_text, canonical_hash, fsync_directory, hash_file
from speakerscribe.batch.identity import SourceInfo, source_payload
from speakerscribe.batch.journal import SUCCESS_EVENTS, Event, Journal
from speakerscribe.batch.layout import DeliverableLayout, UnsafeOutputPathError
from speakerscribe.batch.locking import StateLock
from speakerscribe.batch.masters import MasterStore
from speakerscribe.batch.profiles import presentation_for
from speakerscribe.batch.renderers import (
    DELIVERABLES,
    RENDER_SCHEMA,
    SPLITS_KEY,
    RenderContext,
    RenderOptions,
    TranscriptRejectedError,
    render_full_llm,
    render_splits,
    render_transcript,
)
from speakerscribe.batch.speakers import RenameStore, apply_renames, rename_hash, repair_orphans

STATUS_OK = "ok"
STATUS_SILENT = "ok_sin_voz"
STATUS_DEGRADED = "degradado_sin_diarizacion"
STATUS_FLAGGED = "publicado_con_flags_criticos"
OK_STATUSES = frozenset({STATUS_OK, STATUS_SILENT})
REASON_MAX_CHARS = 300
_ERROR_PREFIX = re.compile(r"^\w+Error:\s*")


@dataclass(frozen=True)
class Job:
    """One unit of GPU work (or of CPU re-rendering)."""

    source: SourceInfo
    signature: str
    profile: dict[str, Any]
    job_id: str
    output_relative: Path
    duration_seconds: float | None = None
    staged_path: Path | None = None

    @property
    def profile_id(self) -> str:
        return str(self.profile.get("id", ""))


def job_id_for(
    signature: str, profile_id: str, output_relative: Path, presentation_id: str | None = None
) -> str:
    """Deterministic job id (v5 formula; presentation only for CPU re-renders)."""
    payload: dict[str, Any] = {
        "content": signature,
        "profile": profile_id,
        "output": output_relative.as_posix(),
    }
    if presentation_id is not None:
        payload["presentacion"] = presentation_id
    return canonical_hash(payload)


@dataclass
class Publication:
    """Everything written for one recording, ready to commit."""

    final: Path
    part: Path
    size: int
    sha256: str
    extras: list[dict[str, Any]]
    status: str
    reason: str | None
    repair: dict[str, int] = field(default_factory=dict)


def strip_error_prefix(text: str) -> str:
    return _ERROR_PREFIX.sub("", text)


def status_from_master(metadata: dict[str, Any], reject_critical: bool) -> tuple[str, str | None]:
    """Quality category of an already-accepted result (CPU re-render).

    Content accepted once is never rejected here; it is classified honestly.
    """
    if metadata.get("diarization_failed") or not metadata.get("diarization_enabled"):
        detail = strip_error_prefix(str(metadata.get("diarization_error") or ""))
        return STATUS_DEGRADED, (detail or "diarización no disponible")[:REASON_MAX_CHARS]
    critical = [
        str(f) for f in (metadata.get("quality_flags") or []) if str(f).startswith("[CRITICAL]")
    ]
    if reject_critical and critical:
        return STATUS_FLAGGED, "; ".join(critical)[:REASON_MAX_CHARS]
    return STATUS_OK, None


def _text_digest(text: str) -> tuple[int, str]:
    data = text.encode("utf-8")
    return len(data), hashlib.sha256(data).hexdigest()


class OutputRecords:
    """Checks journal output records against the files on disk."""

    def __init__(self, layout: DeliverableLayout, verify_hash: bool) -> None:
        self.layout = layout
        self.verify_hash = verify_hash

    def path(self, record: dict[str, Any] | None) -> Path | None:
        rel = ((record or {}).get("output") or {}).get("relative_path")
        if not rel:
            return None
        try:
            return self.layout.canonical(Path(rel))
        except UnsafeOutputPathError:
            return None

    def stat_valid(self, record: dict[str, Any] | None) -> bool:
        """The recorded canonical file exists with the recorded size."""
        final = self.path(record)
        try:
            return bool(
                final is not None
                and final.is_file()
                and final.stat().st_size == ((record or {}).get("output") or {}).get("size_bytes")
            )
        except OSError:
            return False

    def valid(self, record: dict[str, Any] | None) -> bool:
        """`stat_valid` plus, when enabled, a matching SHA-256."""
        if not self.stat_valid(record):
            return False
        if not self.verify_hash:
            return True
        final = self.path(record)
        try:
            return final is not None and hash_file(final, "full") == (
                (record or {}).get("output") or {}
            ).get("sha256")
        except OSError:
            return False

    def latest_valid(self, records: list[dict[str, Any]]) -> dict[str, Any] | None:
        for rec in reversed(records):
            if rec.get("event") in SUCCESS_EVENTS and self.valid(rec):
                return rec
        return None


class Publisher:
    """Writes deliverables and commits them to the journal.

    Args:
        layout: Deliverable locations.
        journal: Event journal.
        lock: Held state lock (ownership verified before every promotion).
        masters: Master JSON store.
        renames: Persistent speaker renames.
        options: Renderer options.
        deliverables: Active secondary deliverable keys.
        input_root: Input folder (for source payloads).
        hash_mode: Signature mode recorded in source payloads.
        presentation_global_id: Id of the global presentation profile.
        repair_orphans_enabled / orphan_tolerance_s: Orphan-speaker repair.
        clock: Returns the UTC timestamp printed in headers.
    """

    def __init__(
        self,
        *,
        layout: DeliverableLayout,
        journal: Journal,
        lock: StateLock | None,
        masters: MasterStore,
        renames: RenameStore,
        options: RenderOptions,
        deliverables: tuple[str, ...],
        input_root: Path,
        hash_mode: str,
        presentation_global_id: str,
        repair_orphans_enabled: bool,
        orphan_tolerance_s: float,
        clock: Any,
    ) -> None:
        self.layout = layout
        self.journal = journal
        self.lock = lock
        self.masters = masters
        self.renames = renames
        self.options = options
        self.deliverables = deliverables
        self.input_root = input_root
        self.hash_mode = hash_mode
        self.presentation_global_id = presentation_global_id
        self.repair_orphans_enabled = repair_orphans_enabled
        self.orphan_tolerance_s = orphan_tolerance_s
        self.clock = clock

    # ── Preparation ──────────────────────────────────────────────────
    def presentation(self, source_id: str) -> dict[str, Any]:
        return presentation_for(
            self.presentation_global_id, rename_hash(self.renames.get(source_id))
        )

    def prepare_segments(
        self, metadata: dict[str, Any], source_id: str
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Publishable segments: orphan repair + persistent renames.

        The master keeps RAW engine labels; this step is deterministic and
        re-applicable, so renames can change later without GPU.
        """
        segments = list(metadata.get("segments") or [])
        repair = {"huerfanos": 0, "adoptados": 0, "hablantes_reales": 0}
        if self.repair_orphans_enabled and segments:
            segments, repair = repair_orphans(segments, self.orphan_tolerance_s)
        mapping = self.renames.get(source_id)
        if mapping:
            segments = apply_renames(segments, mapping)
        return segments, repair

    # ── Writing ──────────────────────────────────────────────────────
    def _write(self, path: Path, text: str, kind: str) -> dict[str, Any]:
        atomic_write_text(path, text)
        size, sha = _text_digest(text)
        return {
            "tipo": kind,
            "relative_path": self.layout.root_relative(path),
            "base": "root",
            "size_bytes": size,
            "sha256": sha,
        }

    def write_extras(self, output_relative: Path, ctx: RenderContext) -> list[dict[str, Any]]:
        """Write every active secondary deliverable (atomic, idempotent)."""
        extras: list[dict[str, Any]] = []
        full_text: str | None = None
        for key in self.deliverables:
            if key == SPLITS_KEY:
                continue
            spec = DELIVERABLES[key]
            text = spec.render(ctx)
            if key == "full_llm":
                full_text = text
            target = (
                self.layout.formats_file(output_relative, spec.suffix)
                if spec.location == "formats"
                else self.layout.llm_file(output_relative, spec.suffix)
            )
            extras.append(self._write(target, text, key))
        if SPLITS_KEY in self.deliverables:
            parts = render_splits(full_text or render_full_llm(ctx), self.options.split_words)
            for stale in self.layout.split_parts(output_relative):
                with contextlib.suppress(OSError):
                    stale.unlink()
            for i, part in enumerate(parts, 1):
                target = self.layout.llm_file(output_relative, f".parte_{i:02d}.txt")
                extras.append(self._write(target, part, f"split_{i:02d}"))
        return extras

    def extras_present(self, output_relative: Path) -> bool:
        """True if every active secondary deliverable exists and is non-empty."""
        try:
            for key in self.deliverables:
                if key == SPLITS_KEY:
                    first = self.layout.llm_file(output_relative, ".parte_01.txt")
                    if not first.is_file():
                        return False
                    continue
                spec = DELIVERABLES[key]
                target = (
                    self.layout.formats_file(output_relative, spec.suffix)
                    if spec.location == "formats"
                    else self.layout.llm_file(output_relative, spec.suffix)
                )
                if not target.is_file() or target.stat().st_size == 0:
                    return False
        except (OSError, UnsafeOutputPathError):
            return False
        return True

    def render_and_write(
        self,
        info: SourceInfo,
        output_relative: Path,
        metadata: dict[str, Any],
        status: str,
        reason: str | None,
        job_id: str,
        *,
        valve: bool,
    ) -> Publication:
        """Render everything; leave the canonical ``.txt`` as a complete ``.part``.

        Args:
            valve: When the strict render rejects an 'ok' result, True
                publishes it FLAGGED (``publicado_con_flags_criticos``)
                instead of losing it; False re-raises.

        Raises:
            TranscriptRejectedError: Only with ``valve=False``.
        """
        final = self.layout.canonical(output_relative)
        segments, repair = self.prepare_segments(metadata, info.source_id)
        ctx = RenderContext(
            segments=segments,
            metadata=metadata,
            source_rel=info.relative_posix,
            options=self.options,
            processed_utc=self.clock(),
            repair=repair,
        )
        try:
            text = render_transcript(ctx, status, reason)
        except TranscriptRejectedError as exc:
            if not valve:
                raise
            status, reason = STATUS_FLAGGED, str(exc)[:REASON_MAX_CHARS]
            text = render_transcript(ctx, status, reason)
        if status == STATUS_OK and not segments and self.options.accept_silent_audio:
            status = STATUS_SILENT
        extras = self.write_extras(output_relative, ctx)
        part = final.with_name(final.name + f".part.{job_id[:16]}")
        atomic_write_text(part, text)
        size, sha = _text_digest(text)
        return Publication(final, part, size, sha, extras, status, reason, repair)

    # ── Commit ───────────────────────────────────────────────────────
    def _promote(self, part: Path, final: Path) -> None:
        if part.parent.resolve() != final.parent.resolve():
            raise RuntimeError("El parcial y el final deben compartir carpeta")
        if self.lock is not None:
            self.lock.assert_owned()
        os.replace(part, final)
        fsync_directory(final.parent)

    def commit(
        self,
        job: Job,
        publication: Publication,
        event: Event,
        master_meta: dict[str, Any] | None,
        **extra: Any,
    ) -> dict[str, Any]:
        """``prepared`` -> promotion -> final event. Returns the final record."""
        output = {
            "relative_path": job.output_relative.as_posix(),
            "part_relative_path": publication.part.relative_to(
                self.layout.paths.deliverables
            ).as_posix(),
            "size_bytes": publication.size,
            "sha256": publication.sha256,
            "encoding": "utf-8",
            "schema": RENDER_SCHEMA,
        }
        fields: dict[str, Any] = {
            "job_id": job.job_id,
            "source": source_payload(job.source, self.input_root, job.signature, self.hash_mode),
            "profile": job.profile,
            "presentacion": self.presentation(job.source.source_id),
            "extras": publication.extras,
            "master_json": master_meta,
            "quality": publication.status,
        }
        if publication.reason:
            fields["degraded_reason"] = publication.reason
        if publication.repair.get("adoptados"):
            fields["speaker_repair"] = publication.repair
        fields.update(extra)
        self.journal.append(Event.PREPARED, output=output, **fields)
        self._promote(publication.part, publication.final)
        final_output = {k: v for k, v in output.items() if k != "part_relative_path"}
        return self.journal.append(event, output=final_output, **fields)

    def recover_prepared(self, events: list[dict[str, Any]]) -> int:
        """Close every ``prepared`` left open by a killed session.

        Returns:
            Number of commits completed (promoted or confirmed).
        """
        closers = {*SUCCESS_EVENTS, Event.PREPARED_ABANDONED.value}
        pending: dict[str, dict[str, Any]] = {}
        for rec in events:
            job_id = rec.get("job_id")
            if not job_id:
                continue
            if rec.get("event") == Event.PREPARED.value:
                pending[job_id] = rec
            elif rec.get("event") in closers:
                pending.pop(job_id, None)
        recovered = 0
        for rec in pending.values():
            output = rec.get("output") or {}
            final_rel, part_rel = output.get("relative_path"), output.get("part_relative_path")
            if not final_rel:
                continue
            try:
                final = self.layout.canonical(Path(final_rel))
                part = self.layout.canonical(Path(part_rel)) if part_rel else None
            except UnsafeOutputPathError:
                continue

            def matches(path: Path | None, out: dict[str, Any] = output) -> bool:
                try:
                    return bool(
                        path is not None
                        and path.is_file()
                        and path.stat().st_size == out.get("size_bytes")
                        and hash_file(path, "full") == out.get("sha256")
                    )
                except OSError:
                    return False

            if matches(final):
                if part is not None and part != final:
                    with contextlib.suppress(OSError):
                        part.unlink(missing_ok=True)
            elif part is not None and matches(part):
                self._promote(part, final)
            else:
                self.journal.append(
                    Event.PREPARED_ABANDONED,
                    job_id=rec.get("job_id"),
                    source=rec.get("source"),
                    profile=rec.get("profile"),
                    error="No se pudo recuperar el commit preparado",
                )
                continue
            self.journal.append(
                Event.COMPLETED,
                job_id=rec.get("job_id"),
                source=rec.get("source"),
                profile=rec.get("profile"),
                presentacion=rec.get("presentacion"),
                extras=rec.get("extras"),
                master_json=rec.get("master_json"),
                output={k: v for k, v in output.items() if k != "part_relative_path"},
                quality=rec.get("quality", STATUS_OK),
                recovered_after_interruption=True,
            )
            recovered += 1
        return recovered


__all__ = [
    "OK_STATUSES",
    "STATUS_DEGRADED",
    "STATUS_FLAGGED",
    "STATUS_OK",
    "STATUS_SILENT",
    "Job",
    "OutputRecords",
    "Publication",
    "Publisher",
    "job_id_for",
    "status_from_master",
    "strip_error_prefix",
]
