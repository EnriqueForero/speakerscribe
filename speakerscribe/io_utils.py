"""Atomic file-write helpers.

Why this module exists:
    A Colab session can die mid-write (12 h cap, OOM, network drop). A
    truncated ``.json`` is worse than a missing one: it still satisfies
    ``output_json.exists()`` in the idempotency check, so a corrupt file
    silently blocks reprocessing. Writing to a temp file in the same
    directory and ``os.replace``-ing guarantees the destination is either
    the previous complete version or the new complete version — never a
    torn file. ``os.replace`` is atomic on POSIX same-filesystem renames.

Streaming outputs (``.txt``/``.srt``/``.segments.jsonl``) are intentionally
NOT atomic: their value is being written incrementally. The ``.json`` is
the source of truth for idempotency, and that one IS atomic.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Any


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> Path:
    """Write ``text`` to ``path`` atomically (temp file + ``os.replace``).

    The temp file lives in the same directory as the destination so the
    final rename never crosses filesystems (a cross-device rename is not
    atomic and raises ``OSError`` on POSIX).

    Args:
        path: Destination file path. Parent directories are created.
        text: Full content to write.
        encoding: Text encoding. Default UTF-8.

    Returns:
        The destination path (for chaining).

    Raises:
        OSError: If the write or the replace fails. The destination is
            left untouched in that case; the temp file is removed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", encoding=encoding) as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():  # replace failed or write raised
            with contextlib.suppress(OSError):
                tmp.unlink()
    return path


def atomic_write_json(
    path: Path,
    obj: Any,
    *,
    indent: int | None = 2,
    ensure_ascii: bool = False,
) -> Path:
    """Serialize ``obj`` as JSON and write it atomically.

    Args:
        path: Destination ``.json`` path.
        obj: JSON-serializable object.
        indent: Indentation passed to ``json.dumps``. None = compact.
        ensure_ascii: Passed to ``json.dumps``. Default False (keep UTF-8).

    Returns:
        The destination path.

    Raises:
        TypeError: If ``obj`` is not JSON-serializable (raised BEFORE any
            file is touched — fail fast).
        OSError: If the filesystem write fails.
    """
    payload = json.dumps(obj, ensure_ascii=ensure_ascii, indent=indent)
    return atomic_write_text(path, payload)


def append_jsonl_line(path: Path, obj: dict[str, Any]) -> None:
    """Append one JSON object as a single line, flushed and fsynced.

    Append-only writes need no file locking (safe on Drive/FUSE where
    SQLite locks are unreliable). A crash can at worst leave one torn
    trailing line, which readers must skip (see ``persistence``).

    Args:
        path: Destination ``.jsonl`` path. Parents are created.
        obj: JSON-serializable dict (one record).

    Raises:
        TypeError: If ``obj`` is not JSON-serializable.
        OSError: If the filesystem write fails.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


__all__ = [
    "append_jsonl_line",
    "atomic_write_json",
    "atomic_write_text",
]
