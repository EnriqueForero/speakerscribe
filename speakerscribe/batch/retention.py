"""What happens to inputs and caches after success (Strategy pattern).

Decision of 2026-10-02: audio is removed from the input once its
transcription is exported successfully, and the diarization cache keeps
three months. To make "removed" recoverable, audio is first MOVED to
``_procesados/YYYY-MM-DD/`` (a rename inside Drive: no data copied) and only
purged after `processed_retention_days`. Every move and purge is journaled.

Only verified successes are retired: quality ``ok``/``ok_sin_voz``, output
committed and its master JSON saved. Flagged or degraded results stay in
the input so the user can decide.
"""

from __future__ import annotations

import contextlib
import shutil
import time
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from speakerscribe.batch.discovery import PROMPT_SUFFIX, source_unchanged
from speakerscribe.batch.fsio import is_within, utc_now
from speakerscribe.batch.identity import SourceInfo
from speakerscribe.batch.journal import Event, Journal

DATE_FOLDER_FORMAT = "%Y-%m-%d"
DIAR_CACHE_GLOB = "*.diar.json"


class RetentionPolicy(Protocol):
    """Decides what happens to a source after a verified success."""

    def retire(self, info: SourceInfo, source_payload: dict[str, Any]) -> Path | None:
        """Return the new location, or None if the source stays in place."""
        ...


class KeepSources:
    """Leave inputs where they are (``after_success="keep"``)."""

    def retire(self, info: SourceInfo, source_payload: dict[str, Any]) -> Path | None:
        return None


def _unique(target: Path) -> Path:
    if not target.exists():
        return target
    stem, suffix = target.stem, target.suffix
    for n in range(2, 10_000):
        candidate = target.with_name(f"{stem} ({n}){suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Sin nombre libre para {target}")


class MoveToProcessed:
    """Move a transcribed source into ``<processed>/<YYYY-MM-DD>/<mirror>/``.

    Args:
        input_root: Input folder (empty subfolders are pruned after a move).
        processed_root: ``_procesados`` folder.
        journal: Event journal.
        today: Date provider (injected in tests).
    """

    def __init__(
        self,
        input_root: Path,
        processed_root: Path,
        journal: Journal,
        today: Callable[[], date] = date.today,
    ) -> None:
        self.input_root = input_root
        self.processed_root = processed_root
        self.journal = journal
        self.today = today

    def retire(self, info: SourceInfo, source_payload: dict[str, Any]) -> Path | None:
        if not source_unchanged(info):
            return None  # changed after transcription: keep it for the next run
        folder = self.processed_root / self.today().strftime(DATE_FOLDER_FORMAT)
        target = _unique(folder / info.relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(info.path), str(target))
        moved_prompt = None
        if info.prompt_path is not None and info.prompt_path.exists():
            moved_prompt = target.with_name(target.stem + PROMPT_SUFFIX)
            with contextlib.suppress(OSError):
                shutil.move(str(info.prompt_path), str(moved_prompt))
        self._prune_empty_parents(info.path.parent)
        self.journal.append(
            Event.SOURCE_RETIRED,
            source=source_payload,
            moved_to=target.relative_to(self.processed_root).as_posix(),
            prompt_moved_to=(
                moved_prompt.relative_to(self.processed_root).as_posix() if moved_prompt else None
            ),
        )
        return target

    def _prune_empty_parents(self, folder: Path) -> None:
        current = folder
        while current != self.input_root and is_within(current, self.input_root):
            try:
                current.rmdir()  # only succeeds when empty
            except OSError:
                return
            current = current.parent


def purge_processed(
    processed_root: Path,
    retention_days: int,
    journal: Journal,
    today: date | None = None,
) -> list[str]:
    """Delete ``_procesados/<date>`` folders older than `retention_days`.

    Folders whose name is not a date are never touched.

    Returns:
        Names of the purged date folders.
    """
    if not processed_root.is_dir():
        return []
    cutoff = (today or date.today()) - timedelta(days=retention_days)
    purged: list[str] = []
    for folder in sorted(processed_root.iterdir()):
        if not folder.is_dir():
            continue
        try:
            folder_date = datetime.strptime(folder.name, DATE_FOLDER_FORMAT).date()
        except ValueError:
            continue
        if folder_date >= cutoff:
            continue
        files = [p for p in folder.rglob("*") if p.is_file()]
        size = sum(p.stat().st_size for p in files)
        shutil.rmtree(folder)
        journal.append(
            Event.PROCESSED_PURGED,
            folder=folder.name,
            files=len(files),
            size_bytes=size,
            retention_days=retention_days,
            purged_utc=utc_now(),
        )
        purged.append(folder.name)
    return purged


def prune_diar_cache(
    cache_dir: Path,
    retention_days: int,
    journal: Journal,
    protected_signatures: set[str],
    now: float | None = None,
) -> int:
    """Delete diarization caches older than `retention_days` (by mtime).

    Caches of sources still awaiting work (`protected_signatures`) are kept:
    pruning them would only cost another diarization pass.

    Returns:
        Number of files deleted.
    """
    if retention_days <= 0 or not cache_dir.is_dir():
        return 0
    limit = (now if now is not None else time.time()) - retention_days * 86_400
    removed: list[str] = []
    freed = 0
    for path in cache_dir.glob(DIAR_CACHE_GLOB):
        signature = path.name.split("_", 1)[0]
        if signature in protected_signatures:
            continue
        try:
            st = path.stat()
            if st.st_mtime >= limit:
                continue
            path.unlink()
        except OSError:
            continue
        removed.append(path.name)
        freed += st.st_size
    if removed:
        journal.append(
            Event.DIAR_CACHE_PRUNED,
            files=len(removed),
            size_bytes=freed,
            retention_days=retention_days,
        )
    return len(removed)


__all__ = [
    "DATE_FOLDER_FORMAT",
    "KeepSources",
    "MoveToProcessed",
    "RetentionPolicy",
    "prune_diar_cache",
    "purge_processed",
]
