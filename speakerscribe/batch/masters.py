"""Master JSON store: the raw engine result per source (``intermedios/<id>.json.gz``).

The master keeps the engine's segments (raw speaker labels, no word timings)
so every deliverable can be re-rendered, renamed or reformatted on CPU
without another GPU pass. Format is identical to notebook v5's, so the 94
masters written before 0.4 remain usable.
"""

from __future__ import annotations

import contextlib
import os
import shutil
from pathlib import Path
from typing import Any

from speakerscribe.batch.fsio import atomic_write_gz_json, hash_file, read_gz_json, utc_now
from speakerscribe.batch.identity import SourceInfo
from speakerscribe.batch.renderers import strip_words

MASTER_SUFFIX = ".json.gz"


class MasterStore:
    """Read/write master JSON files.

    Args:
        folder: ``<state>/intermedios``.
        records_base: Folder that journal ``relative_path`` values are
            relative to (the deliverables folder, as in v5).
    """

    def __init__(self, folder: Path, records_base: Path) -> None:
        self.folder = folder
        self.records_base = records_base

    def path(self, source_id: str) -> Path:
        return self.folder / f"{source_id}{MASTER_SUFFIX}"

    def _relative(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.records_base.resolve()).as_posix()
        except ValueError:
            return str(path)

    def save(
        self, metadata: dict[str, Any], info: SourceInfo, signature: str, motor_profile_id: str
    ) -> dict[str, Any]:
        """Persist the engine result; return ``{relative_path, size_bytes, sha256}``."""
        master = strip_words(metadata)
        master["_v5"] = {  # key name kept for compatibility with existing masters
            "words_stripped": True,
            "source_rel": info.relative_posix,
            "source_id": info.source_id,
            "content_signature": signature,
            "motor_profile_id": motor_profile_id,
            "creado": utc_now(),
        }
        target = self.path(info.source_id)
        size, sha = atomic_write_gz_json(target, master)
        return {"relative_path": self._relative(target), "size_bytes": size, "sha256": sha}

    def load(self, source_id: str) -> dict[str, Any] | None:
        return read_gz_json(self.path(source_id))

    def meta(self, source_id: str) -> dict[str, Any] | None:
        """``{relative_path, size_bytes, sha256}`` of the stored master, or None."""
        target = self.path(source_id)
        try:
            if not target.is_file():
                return None
            return {
                "relative_path": self._relative(target),
                "size_bytes": target.stat().st_size,
                "sha256": hash_file(target, "full"),
            }
        except OSError:
            return None

    def adopt(self, donor_source_id: str, source_id: str) -> dict[str, Any] | None:
        """Copy a donor's master for `source_id` (dedup by content).

        Returns:
            The donor master (parsed), or None if it is missing/unreadable.
        """
        origin = self.path(donor_source_id)
        master = read_gz_json(origin)
        if master is None:
            return None
        target = self.path(source_id)
        if origin.resolve() != target.resolve():
            tmp = target.with_name(target.name + ".tmp.adopt")
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(origin, tmp)
                os.replace(tmp, target)
            finally:
                with contextlib.suppress(OSError):
                    tmp.unlink(missing_ok=True)
        return master


__all__ = ["MASTER_SUFFIX", "MasterStore"]
