"""Durable file primitives for the batch layer: atomic writes, hashing, paths.

Context: Google Colab with Google Drive mounted through FUSE. A session can
die mid-write (12 h cap, OOM, network drop), so every durable write goes
through a uniquely named temp file + ``fsync`` + ``os.replace``: readers see
either the previous complete file or the new complete file, never a torn one.

These helpers are ported verbatim in behavior from notebook v5 (the
transactional core that produced the 94 confirmed transcriptions of
2026-08/09), so on-disk formats stay compatible.
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HASH_CHUNK_BYTES = 8 << 20
"""Read size for content hashing (8 MiB: good throughput over Drive FUSE)."""

FAST_SAMPLE_BYTES = 8 << 20
"""Bytes hashed at each end of a file in ``fast`` signature mode."""

_RUN_TOKEN = uuid.uuid4().hex
"""Per-process token for temp-file names (two sessions never collide)."""


def utc_now() -> str:
    """Current UTC time in ISO 8601 (journal timestamps)."""
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value: Any) -> str:
    """SHA-256 of the canonical JSON of `value` (sorted keys, compact).

    Identities stored in the journal (source ids, job ids, profiles, the
    diarization-cache key) are canonical hashes; changing this function
    would orphan every existing record.
    """
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def hash_file(path: Path, mode: str = "full", chunk_size: int = HASH_CHUNK_BYTES) -> str:
    """Content signature of a file.

    Args:
        path: File to hash.
        mode: ``"full"`` = SHA-256 of every byte (strong identity, enables
            dedup by content); ``"fast"`` = size + first and last 8 MiB,
            prefixed ``"fast:"`` so it is never mistaken for a strong id.
        chunk_size: Read size in bytes.

    Returns:
        Hex digest (``"fast:"``-prefixed in fast mode).

    Raises:
        ValueError: If `mode` is not ``"full"`` or ``"fast"``.
        OSError: If the file cannot be read.
    """
    if mode not in {"full", "fast"}:
        raise ValueError(f"hash mode must be 'full' or 'fast', got {mode!r}")
    digest = hashlib.sha256()
    if mode == "fast":
        size = path.stat().st_size
        digest.update(str(size).encode())
        with path.open("rb") as fh:
            digest.update(fh.read(FAST_SAMPLE_BYTES))
            if size > 2 * FAST_SAMPLE_BYTES:
                fh.seek(size - FAST_SAMPLE_BYTES)
                digest.update(fh.read(FAST_SAMPLE_BYTES))
            elif size > FAST_SAMPLE_BYTES:
                fh.seek(FAST_SAMPLE_BYTES)
                digest.update(fh.read())
        return "fast:" + digest.hexdigest()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_and_hash(source: Path, destination: Path, chunk_size: int = HASH_CHUNK_BYTES) -> str:
    """Copy `source` to `destination` and return its full SHA-256 in ONE read.

    Why: on Drive every byte read costs FUSE latency. v5 read each audio
    twice (once to hash it, once when ffmpeg extracted the WAV from Drive).
    Staging the file on local disk while hashing it halves Drive reads, and
    ffmpeg then reads local NVMe.

    Args:
        source: File to copy (typically on Drive).
        destination: Target path (typically on local scratch). Replaced
            atomically; parent directories are created.
        chunk_size: Read size in bytes.

    Returns:
        Hex SHA-256 of the copied bytes.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + f".tmp.{_RUN_TOKEN}")
    digest = hashlib.sha256()
    try:
        with source.open("rb") as src, tmp.open("wb") as dst:
            for block in iter(lambda: src.read(chunk_size), b""):
                digest.update(block)
                dst.write(block)
        os.replace(tmp, destination)
    finally:
        with contextlib.suppress(OSError):
            if tmp.exists():
                tmp.unlink()
    return digest.hexdigest()


def fsync_directory(path: Path) -> None:
    """fsync a directory so a rename inside it is durable (FUSE-tolerant)."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_text(path: Path, text: str) -> None:
    """Write `text` atomically (unique temp + fsync + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{_RUN_TOKEN}")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            with contextlib.suppress(OSError):
                os.fsync(fh.fileno())
        os.replace(tmp, path)
        fsync_directory(path.parent)
    finally:
        with contextlib.suppress(OSError):
            if tmp.exists():
                tmp.unlink()


def atomic_write_json(path: Path, value: Any) -> None:
    """Write `value` as indented UTF-8 JSON, atomically."""
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, default=str))


def atomic_write_gz_json(path: Path, value: Any) -> tuple[int, str]:
    """Write `value` as gzip-compressed JSON, atomically.

    Returns:
        (size in bytes, full SHA-256 of the written file).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{_RUN_TOKEN}")
    try:
        with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as fh:
            json.dump(value, fh, ensure_ascii=False, default=str)
        os.replace(tmp, path)
        fsync_directory(path.parent)
    finally:
        with contextlib.suppress(OSError):
            if tmp.exists():
                tmp.unlink()
    return path.stat().st_size, hash_file(path, "full")


def read_gz_json(path: Path) -> dict[str, Any] | None:
    """Read a ``.json.gz`` (or plain ``.json``); None if missing or unreadable."""
    try:
        if path.name.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                value = json.load(fh)
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, EOFError):
        return None
    return value if isinstance(value, dict) else None


def is_within(path: Path, parent: Path) -> bool:
    """True if `path` resolves inside `parent` (or equals it)."""
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


__all__ = [
    "FAST_SAMPLE_BYTES",
    "HASH_CHUNK_BYTES",
    "atomic_write_gz_json",
    "atomic_write_json",
    "atomic_write_text",
    "canonical_hash",
    "copy_and_hash",
    "fsync_directory",
    "hash_file",
    "is_within",
    "read_gz_json",
    "utc_now",
]
