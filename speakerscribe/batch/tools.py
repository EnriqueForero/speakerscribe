"""Operator tools that run without a GPU: status, rebind, rename, autopsy.

Each tool takes the same `BatchSettings` as the batch, so the notebook keeps
one configuration cell.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from speakerscribe.batch.discovery import discover_sources
from speakerscribe.batch.engine import DiarizationCacheStore
from speakerscribe.batch.executor import JobExecutor
from speakerscribe.batch.fsio import utc_now
from speakerscribe.batch.identity import SourceInfo, rebind_document, source_stat_matches
from speakerscribe.batch.journal import Event, Journal, JournalIndex
from speakerscribe.batch.layout import DeliverableLayout
from speakerscribe.batch.locking import StateLock
from speakerscribe.batch.masters import MasterStore
from speakerscribe.batch.paths import BatchPaths
from speakerscribe.batch.profiles import (
    enabled_deliverables,
    engine_config,
    presentation_global,
    render_options,
)
from speakerscribe.batch.publisher import OK_STATUSES, OutputRecords, Publisher
from speakerscribe.batch.reporting import autopsy as _autopsy
from speakerscribe.batch.settings import BatchSettings
from speakerscribe.batch.speakers import RenameStore


def rebind_workspace(settings: BatchSettings) -> dict[str, Any]:
    """Re-link the state folder to the CURRENT input folder (audited).

    Use once after moving the input folder on purpose (e.g. from
    ``Pruebas/Speakerscribe/data`` to ``…/Transcripcion-Diarizacion/data``).
    Source ids depend on the input root, so files already confirmed under
    the old root will be seen as new if they reappear in the new input;
    already-published outputs are never touched.

    Returns:
        ``{"nuevo": …, "anterior": …, "respaldo": …}``.
    """
    paths = BatchPaths.from_settings(settings)
    paths.ensure()
    run_id = uuid.uuid4().hex
    with StateLock(paths.lock, run_id, force_take=settings.force_take_lock):
        new, previous, backup = rebind_document(paths)
        Journal(paths.events, run_id).append(
            Event.WORKSPACE_REBOUND,
            new_input_root=new.get("input_root"),
            new_input_root_sha256=new.get("input_root_sha256"),
            previous_input_root=(previous or {}).get("input_root"),
            previous_input_root_sha256=(previous or {}).get("input_root_sha256"),
            backup=str(backup) if backup else None,
        )
    return {"nuevo": new, "anterior": previous, "respaldo": str(backup) if backup else None}


@dataclass
class Census:
    """Global state of the input folder against the journal."""

    total: int = 0
    confirmed: int = 0
    flagged: list[tuple[str, str]] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    invalid: int = 0
    output_changed: list[str] = field(default_factory=list)
    blocked_jobs: int = 0

    def lines(self) -> list[str]:
        out = [
            "═" * 68,
            f"  ESTADO — {self.total} archivo(s) en la entrada",
            "═" * 68,
            f"  ✔ Confirmados OK              : {self.confirmed:>5d}",
            f"  🧯 Confirmados MARCADOS        : {len(self.flagged):>5d}",
            f"  ⏳ Pendientes / cambiados      : {len(self.pending):>5d}",
            f"  ✋ Salida movida/editada (usted): {len(self.output_changed):>4d}",
            f"  🚫 Inválidos (sin audio)       : {self.invalid:>5d}",
            f"  ⛔ Jobs con intentos agotados  : {self.blocked_jobs:>5d}",
        ]
        if self.pending:
            out.append("  Próximos pendientes:")
            out += [f"     · {p}" for p in self.pending[:15]]
            if len(self.pending) > 15:
                out.append(f"     … y {len(self.pending) - 15} más")
        if self.flagged:
            out.append("  Marcados (publicados con reservas):")
            out += [f"     🧯 {rel} ({q})" for rel, q in self.flagged[:10]]
        if self.output_changed:
            out.append("  Con salida movida/editada por usted (no se regeneran):")
            out += [f"     ✋ {rel}" for rel in self.output_changed[:10]]
        return out


def status(settings: BatchSettings) -> Census:
    """Read-only census (stat-based; no hashes, no GPU, no lock)."""
    paths = BatchPaths.from_settings(settings)
    discovery = discover_sources(
        settings, paths.input, skip_dirs=(*paths.output_dirs(), paths.scratch)
    )
    index = JournalIndex.build(Journal(paths.events).read())
    records = OutputRecords(DeliverableLayout(paths), verify_hash=False)
    census = Census(total=len(discovery.sources))
    for source in discovery.sources:
        prior = index.latest_success(source.source_id)
        if prior and source_stat_matches(prior, source):
            if not records.stat_valid(prior):
                census.output_changed.append(source.relative_posix)
            elif str(prior.get("quality") or "ok") in OK_STATUSES:
                census.confirmed += 1
            else:
                census.flagged.append((source.relative_posix, str(prior.get("quality"))))
            continue
        invalid = index.invalid.get(source.source_id)
        if invalid and source_stat_matches(invalid, source):
            census.invalid += 1
            continue
        census.pending.append(source.relative_posix)
    census.blocked_jobs = sum(1 for n in index.attempts.values() if n >= settings.max_attempts)
    return census


def published(settings: BatchSettings, limit: int = 0) -> list[dict[str, str]]:
    """Recordings with a published result (newest last), from the journal.

    Works even after the audio left the input folder.
    """
    paths = BatchPaths.from_settings(settings)
    index = JournalIndex.build(Journal(paths.events).read())
    rows = []
    for sid, records in index.ok_by_source.items():
        last = records[-1]
        rows.append(
            {
                "source_id": sid,
                "audio": str((last.get("source") or {}).get("relative_path")),
                "salida": str((last.get("output") or {}).get("relative_path")),
                "fecha": str(last.get("timestamp_utc")),
            }
        )
    rows.sort(key=lambda r: r["fecha"])
    return rows[-limit:] if limit else rows


def _source_from_record(record: dict[str, Any], input_root: Path) -> SourceInfo:
    src = record.get("source") or {}
    prompt = src.get("prompt_path")
    return SourceInfo(
        path=input_root / str(src.get("relative_path")),
        relative=Path(str(src.get("relative_path"))),
        source_id=str(src.get("id")),
        size_bytes=int(src.get("size_bytes") or 0),
        mtime_ns=int(src.get("mtime_ns") or 0),
        prompt_path=(input_root / prompt) if prompt else None,
        prompt_size=src.get("prompt_size"),
        prompt_mtime_ns=src.get("prompt_mtime_ns"),
    )


def rename_speakers(
    settings: BatchSettings, audio_or_output: str, mapping: dict[str, str]
) -> dict[str, Any]:
    """Persist a speaker mapping and re-render that recording (CPU only).

    Args:
        settings: Batch settings.
        audio_or_output: Relative path of the audio (as in the input) or of
            its canonical ``.txt`` — see `published`.
        mapping: e.g. ``{"SPEAKER_00": "Ana", "SPEAKER_01": "Luis"}``. An
            empty mapping removes the renames.

    Returns:
        The ``republished`` journal record.

    Raises:
        LookupError: Unknown recording.
        RuntimeError: The recording has no master JSON.
    """
    paths = BatchPaths.from_settings(settings)
    run_id = uuid.uuid4().hex
    journal = Journal(paths.events, run_id)
    index = JournalIndex.build(journal.read())
    wanted = audio_or_output.strip()
    prior = None
    for records in index.ok_by_source.values():
        last = records[-1]
        if wanted in {
            (last.get("source") or {}).get("relative_path"),
            (last.get("output") or {}).get("relative_path"),
        }:
            prior = last
            break
    if prior is None:
        raise LookupError(f"No hay un resultado publicado para {wanted!r} (vea published()).")
    info = _source_from_record(prior, paths.input)
    masters = MasterStore(paths.masters, paths.deliverables)
    if not masters.path(info.source_id).is_file():
        raise RuntimeError(
            "Esta grabación no tiene JSON maestro: no se puede re-renderizar sin GPU."
        )
    renames = RenameStore(paths.renames)
    with StateLock(paths.lock, run_id, force_take=settings.force_take_lock) as lock:
        rename_sha = renames.save(info.source_id, mapping)
        journal.append(
            Event.SPEAKERS_RENAMED,
            source=prior.get("source"),
            rename_sha=rename_sha,
            mapping=mapping,
            renamed_utc=utc_now(),
        )
        layout = DeliverableLayout(paths)
        from speakerscribe import __version__

        publisher = Publisher(
            layout=layout,
            journal=journal,
            lock=lock,
            masters=masters,
            renames=renames,
            options=render_options(settings, f"speakerscribe v{__version__}"),
            deliverables=enabled_deliverables(settings),
            input_root=paths.input,
            hash_mode=settings.hash_mode,
            presentation_global_id=presentation_global(settings)["id"],
            repair_orphans_enabled=settings.repair_orphans,
            orphan_tolerance_s=settings.orphan_tolerance_s,
            clock=utc_now,
        )
        executor = JobExecutor(
            settings=settings,
            journal=journal,
            publisher=publisher,
            masters=masters,
            diar_cache=DiarizationCacheStore(paths.diar_cache),
            config=engine_config(settings),
            input_root=paths.input,
            jobs_dir=paths.jobs,
            failure_logs=paths.failure_logs,
            where={},
        )
        record = executor.republish(info, prior)
    if record is None:  # pragma: no cover - master checked above
        raise RuntimeError("No se pudo re-renderizar (falta el JSON maestro).")
    return record


def autopsy(settings: BatchSettings) -> list[str]:
    """Why did the last session end? (CPU only; run in a fresh session.)"""
    paths = BatchPaths.from_settings(settings)
    return _autopsy(paths.summary, paths.telemetry, paths.events)


__all__ = ["Census", "autopsy", "published", "rebind_workspace", "rename_speakers", "status"]
