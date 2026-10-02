"""Find input media, classify it with ffprobe and take a stable content signature.

Ported from notebook v5 with one efficiency change: in ``full`` hash mode the
file is copied to local scratch WHILE it is hashed (`stage_source`), so the
audio is read from Drive once instead of twice.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import subprocess
import time
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from speakerscribe.batch.fsio import copy_and_hash, hash_file, is_within
from speakerscribe.batch.identity import SourceInfo, source_id_for
from speakerscribe.batch.settings import BatchSettings

KNOWN_MEDIA_EXTENSIONS = frozenset(
    {
        ".3gp", ".3g2", ".aac", ".ac3", ".aif", ".aiff", ".amr", ".ape", ".asf",
        ".avi", ".caf", ".dts", ".eac3", ".f4a", ".f4v", ".flac", ".m2ts", ".m4a",
        ".m4v", ".mka", ".mkv", ".mov", ".mp2", ".mp3", ".mp4", ".mpeg", ".mpg",
        ".mts", ".oga", ".ogg", ".ogv", ".opus", ".rm", ".rmvb", ".ts", ".vob",
        ".wav", ".webm", ".wma", ".wmv",
    }
)  # fmt: skip
OBVIOUS_NON_MEDIA = frozenset(
    {
        ".txt", ".md", ".json", ".jsonl", ".csv", ".tsv", ".xlsx", ".xls", ".doc",
        ".docx", ".pdf", ".ppt", ".pptx", ".py", ".ipynb", ".html", ".htm", ".xml",
        ".yaml", ".yml", ".ini", ".cfg", ".log", ".zip", ".rar", ".7z", ".tar",
        ".gz", ".bz2", ".xz", ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp",
        ".svg", ".exe", ".dll", ".so", ".whl", ".srt", ".vtt",
    }
)  # fmt: skip
PROMPT_SUFFIX = ".prompt.txt"
FFPROBE_TIMEOUT_S = 60
_PERMANENT_PROBE_MARKERS = (
    "invalid data found when processing input",
    "moov atom not found",
    "could not find codec parameters",
    "end of file",
)


def nfc(value: str) -> str:
    """NFC-normalize (Drive and macOS mix NFC/NFD in accented names)."""
    return unicodedata.normalize("NFC", value)


def parse_extensions(raw: str) -> frozenset[str]:
    """'ts, .MP4;wav' -> {'.ts', '.mp4', '.wav'}."""
    out = set()
    for token in re.split(r"[,;\s]+", (raw or "").strip()):
        if token:
            out.add(token.lower() if token.startswith(".") else "." + token.lower())
    return frozenset(out)


def resolved_extensions(settings: BatchSettings) -> tuple[frozenset[str], bool, frozenset[str]]:
    """(accepted, probe_unknown, excluded) from the discovery settings."""
    only = parse_extensions(settings.only_extensions)
    excluded = parse_extensions(settings.exclude_extensions)
    if only:
        return only, False, excluded
    accepted = (KNOWN_MEDIA_EXTENSIONS | parse_extensions(settings.extra_extensions)) - excluded
    return frozenset(accepted), settings.probe_unknown_extensions, excluded


def prompt_sidecar(path: Path) -> tuple[Path | None, int | None, int | None]:
    """(path, size, mtime_ns) of ``<stem>.prompt.txt`` next to a media file."""
    candidate = path.with_suffix(PROMPT_SUFFIX)
    try:
        if not candidate.is_file():
            return None, None, None
        st = candidate.stat()
    except OSError:  # name at the 255-byte limit
        return None, None, None
    return candidate, st.st_size, st.st_mtime_ns


@dataclass(frozen=True)
class Discovery:
    """Result of scanning the input folder."""

    sources: list[SourceInfo]
    warnings: list[str]
    stem_collisions: set[tuple[str, str]]


def discover_sources(
    settings: BatchSettings, input_root: Path, skip_dirs: tuple[Path, ...]
) -> Discovery:
    """Walk `input_root` (recursively unless disabled) and list media files.

    Args:
        settings: Discovery options.
        input_root: Folder to scan.
        skip_dirs: Folders never descended into (state, outputs).

    Returns:
        Sorted sources, warnings, and (folder, stem) keys shared by two or
        more files (homonyms with different extensions).

    Raises:
        FileNotFoundError: If `input_root` does not exist.
    """
    if not input_root.is_dir():
        raise FileNotFoundError(f"La carpeta de entrada no existe: {input_root}")
    accepted, probe_unknown, excluded = resolved_extensions(settings)
    pattern = nfc(settings.include_glob.strip()).casefold()
    warnings: list[str] = []
    found: list[SourceInfo] = []
    unknown, cap_warned = 0, False
    visited: set[Path] = set()

    def on_error(exc: OSError) -> None:
        warnings.append(f"No se pudo recorrer {getattr(exc, 'filename', '?')}: {exc}")

    for dirpath, dirnames, filenames in os.walk(
        input_root, topdown=True, onerror=on_error, followlinks=settings.follow_symlinks
    ):
        current = Path(dirpath)
        real = current.resolve()
        if not is_within(real, input_root) or real in visited:
            dirnames[:] = []
            continue
        visited.add(real)
        if not settings.recursive:
            dirnames[:] = []
        else:
            keep = []
            for name in dirnames:
                child = current / name
                if name.startswith(".") or any(is_within(child, s) for s in skip_dirs):
                    continue
                if child.is_symlink() and not settings.follow_symlinks:
                    continue
                keep.append(name)
            dirnames[:] = keep

        for name in filenames:
            if name.startswith((".", "~")) or name.endswith(PROMPT_SUFFIX):
                continue
            path = current / name
            if path.is_symlink() and (
                not settings.follow_symlinks or not is_within(path.resolve(), input_root)
            ):
                continue
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix not in accepted:
                if suffix in excluded or not probe_unknown or suffix in OBVIOUS_NON_MEDIA:
                    continue
                unknown += 1
                if unknown > settings.max_unknown_probes:
                    if not cap_warned:
                        warnings.append("Tope de extensiones desconocidas alcanzado; se omiten.")
                        cap_warned = True
                    continue
            try:
                st = path.stat()
                prompt, prompt_size, prompt_mtime = prompt_sidecar(path)
                relative = path.relative_to(input_root)
            except OSError as e:
                warnings.append(f"No se pudo inspeccionar {path}: {e}")
                continue
            if st.st_size == 0:
                continue
            if pattern and not fnmatch.fnmatchcase(nfc(relative.as_posix()).casefold(), pattern):
                continue
            found.append(
                SourceInfo(
                    path=path,
                    relative=relative,
                    source_id=source_id_for(input_root, relative),
                    size_bytes=st.st_size,
                    mtime_ns=st.st_mtime_ns,
                    prompt_path=prompt,
                    prompt_size=prompt_size,
                    prompt_mtime_ns=prompt_mtime,
                )
            )
    found.sort(key=lambda i: (nfc(i.relative_posix).casefold(), i.source_id))

    counts: dict[tuple[str, str], int] = {}
    for info in found:
        key = (nfc(info.relative.parent.as_posix()).casefold(), nfc(info.relative.stem).casefold())
        counts[key] = counts.get(key, 0) + 1
    return Discovery(found, warnings, {k for k, n in counts.items() if n > 1})


ProbeStatus = Literal["ok", "invalid", "retryable"]


@dataclass(frozen=True)
class ProbeResult:
    """ffprobe verdict: permanent invalid vs transient failure vs ok."""

    status: ProbeStatus
    duration_s: float | None
    detail: str | None


def probe_media(path: Path) -> ProbeResult:
    """Classify a file with ffprobe without decoding it.

    A transient Drive hiccup is ``retryable`` (never condemns the file);
    only known corruption markers make it ``invalid``.
    """
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=codec_name:format=duration", "-of", "json", str(path),
    ]  # fmt: skip
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=FFPROBE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return ProbeResult("retryable", None, f"ffprobe excedió {FFPROBE_TIMEOUT_S} s")
    except OSError as e:
        return ProbeResult("retryable", None, f"ffprobe no ejecutable: {e}")
    if result.returncode != 0:
        detail = result.stderr.strip()[:1000]
        lowered = detail.lower()
        permanent = any(m in lowered for m in _PERMANENT_PROBE_MARKERS)
        return ProbeResult("invalid" if permanent else "retryable", None, detail)
    try:
        payload = json.loads(result.stdout)
        streams = payload.get("streams") or []
        raw = (payload.get("format") or {}).get("duration")
        duration = float(raw) if raw not in (None, "N/A") else None
    except (ValueError, TypeError) as e:
        return ProbeResult("retryable", None, f"respuesta de ffprobe inválida: {e}")
    if not streams:
        return ProbeResult("invalid", duration, "el contenedor no tiene stream de audio")
    return ProbeResult("ok", duration, None)


class SourceChangingError(RuntimeError):
    """The source changed while it was being read; retry later."""


@dataclass(frozen=True)
class StagedSource:
    """A source with its content signature, optionally copied to local disk."""

    info: SourceInfo
    signature: str
    staged_path: Path | None


def _refresh(info: SourceInfo, st: os.stat_result) -> SourceInfo:
    prompt, size, mtime = prompt_sidecar(info.path)
    return replace(
        info,
        size_bytes=st.st_size,
        mtime_ns=st.st_mtime_ns,
        prompt_path=prompt,
        prompt_size=size,
        prompt_mtime_ns=mtime,
    )


def stage_source(
    info: SourceInfo,
    *,
    hash_mode: str,
    stability_seconds: int,
    staging_dir: Path | None,
    staged_name: str,
) -> StagedSource:
    """Take a stable content signature; in ``full`` mode also stage the bytes.

    Args:
        info: Source to read.
        hash_mode: ``"full"`` or ``"fast"``.
        stability_seconds: Files modified more recently are postponed.
        staging_dir: Local folder for the staged copy (None = no copy).
        staged_name: File name for the staged copy.

    Returns:
        `StagedSource` (``staged_path`` is None in ``fast`` mode or when
        `staging_dir` is None).

    Raises:
        SourceChangingError: If the file is too recent or changed while read.
        OSError: On read errors (transient on Drive; the caller retries).
    """
    before = info.path.stat()
    age = time.time() - before.st_mtime
    if age < stability_seconds:
        raise SourceChangingError(f"modificado hace {age:.1f} s; se pospone hasta estabilizar")
    staged: Path | None = None
    if hash_mode == "full" and staging_dir is not None:
        staged = staging_dir / staged_name
        signature = copy_and_hash(info.path, staged)
    else:
        signature = hash_file(info.path, hash_mode)
    after = info.path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        if staged is not None:
            staged.unlink(missing_ok=True)
        raise SourceChangingError("el archivo cambió mientras se leía")
    return StagedSource(_refresh(info, after), signature, staged)


def source_unchanged(info: SourceInfo) -> bool:
    """True if the source and its sidecar still match the stat in `info`."""
    try:
        st = info.path.stat()
    except OSError:
        return False
    prompt, size, mtime = prompt_sidecar(info.path)
    return (
        st.st_size == info.size_bytes
        and st.st_mtime_ns == info.mtime_ns
        and size == info.prompt_size
        and mtime == info.prompt_mtime_ns
        and prompt == info.prompt_path
    )


__all__ = [
    "KNOWN_MEDIA_EXTENSIONS",
    "OBVIOUS_NON_MEDIA",
    "PROMPT_SUFFIX",
    "Discovery",
    "ProbeResult",
    "SourceChangingError",
    "StagedSource",
    "discover_sources",
    "nfc",
    "parse_extensions",
    "probe_media",
    "prompt_sidecar",
    "resolved_extensions",
    "source_unchanged",
    "stage_source",
]
