"""Output names and locations of every deliverable.

Naming (decision of 2026-10-02): the output keeps the audio's name EXACTLY,
including the leading ``*`` the user uses to mark "not yet summarized by an
LLM" (notebook v5 replaced it with ``_`` and the user renamed 57 files by
hand). Only characters no filesystem accepts are replaced: ``/`` and control
characters.

Locations::

    <deliverables>/<mirror>/<name>.txt                   canonical, always
    <formats>/<mirror>/<name>.transcript.md|.srt|.json|.plano.txt
    <llm>/<mirror>/<name>.full_for_llm.txt | <name>.parte_NN.txt

Set the three folders to the same path to get v5's "everything together".
The canonical path is journaled RELATIVE TO `deliverables`, exactly like v5,
so the 94 records written before this module keep resolving.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from speakerscribe.batch.discovery import nfc
from speakerscribe.batch.fsio import is_within
from speakerscribe.batch.identity import SourceInfo
from speakerscribe.batch.paths import BatchPaths

CANONICAL_SUFFIX = ".txt"
MAX_NAME_BYTES = 240
MAX_PATH_BYTES = 3000
LONG_PATHS_DIR = "_rutas_largas"
_UNSAFE_CHARS = re.compile(r"[/\x00-\x1f\x7f]")


class UnsafeOutputPathError(RuntimeError):
    """A computed output path would escape its folder."""


def sanitize_name(name: str) -> str:
    """Replace only characters no filesystem accepts; keep everything else."""
    cleaned = _UNSAFE_CHARS.sub("_", nfc(name)).strip()
    return cleaned.strip(".") or "_"


def render_name(template: str, stem: str, relative: Path, model: str, today: str) -> str:
    """Apply the name template (markers: {stem} {carpeta} {modelo} {fecha})."""
    raw = template.format(
        stem=stem, carpeta=relative.parent.name or "raiz", modelo=model, fecha=today
    )
    return sanitize_name(raw) or sanitize_name(stem)


@dataclass(frozen=True)
class OutputNamer:
    """Decides the canonical relative path of a NEW source's ``.txt``.

    Args:
        template: Name template (must contain {stem}).
        model: Whisper model name (for {modelo}).
        mirror_subfolders: Reproduce the input's subfolders.
        deliverables: Deliverables folder (for the total path-length guard).
    """

    template: str
    model: str
    mirror_subfolders: bool
    deliverables: Path

    def relative_for(
        self, info: SourceInfo, stem_collisions: set[tuple[str, str]], today: str | None = None
    ) -> Path:
        """``A/reunion.wav`` -> ``A/reunion.txt``; homonyms -> ``reunion (wav).txt``."""
        today = today or datetime.now().strftime("%Y-%m-%d")
        stem = info.relative.stem
        key = (nfc(info.relative.parent.as_posix()).casefold(), nfc(stem).casefold())
        if key in stem_collisions and info.relative.suffix:
            stem = f"{stem} ({info.relative.suffix.lstrip('.').lower()})"
        name = render_name(self.template, stem, info.relative, self.model, today) + CANONICAL_SUFFIX
        if len(name.encode("utf-8")) > MAX_NAME_BYTES:
            budget = MAX_NAME_BYTES - len(CANONICAL_SUFFIX) - 14
            head = name[: -len(CANONICAL_SUFFIX)].encode("utf-8")[: max(1, budget)]
            name = (
                head.decode("utf-8", errors="ignore") + f"~{info.source_id[:12]}{CANONICAL_SUFFIX}"
            )
        folder = info.relative.parent if self.mirror_subfolders else Path(".")
        relative = folder / name
        if len(str(self.deliverables / relative).encode("utf-8")) > MAX_PATH_BYTES:
            relative = Path(LONG_PATHS_DIR) / info.source_id[:2] / info.source_id / name
        return relative


def disambiguate(candidate: Path, source_id: str, attempt: int) -> Path:
    """Collision ladder: ``name~<source_id prefix>.txt`` with growing prefixes."""
    width = (10, 16, 24, 32, 64)[min(attempt, 4)]
    stem = candidate.name[: -len(CANONICAL_SUFFIX)]
    return candidate.with_name(f"{stem}~{source_id[:width]}{CANONICAL_SUFFIX}")


def _safe_child(base: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise UnsafeOutputPathError(f"Ruta de salida insegura: {relative}")
    candidate = (base / relative).resolve()
    if not is_within(candidate, base):
        raise UnsafeOutputPathError(f"La salida quedaría fuera de {base}: {relative}")
    return candidate


@dataclass(frozen=True)
class DeliverableLayout:
    """Maps a canonical relative path to every deliverable location."""

    paths: BatchPaths

    def canonical(self, relative: Path) -> Path:
        """Absolute path of the canonical ``.txt``."""
        return _safe_child(self.paths.deliverables, relative)

    @staticmethod
    def base_name(relative: Path) -> str:
        """Name without the ``.txt`` suffix (names may contain dots)."""
        name = relative.name
        return name[: -len(CANONICAL_SUFFIX)] if name.endswith(CANONICAL_SUFFIX) else name

    def formats_file(self, relative: Path, suffix: str) -> Path:
        """``<formats>/<mirror>/<name><suffix>``."""
        return _safe_child(
            self.paths.formats, relative.parent / (self.base_name(relative) + suffix)
        )

    def llm_file(self, relative: Path, suffix: str) -> Path:
        """``<llm>/<mirror>/<name><suffix>``."""
        return _safe_child(self.paths.llm, relative.parent / (self.base_name(relative) + suffix))

    def split_parts(self, relative: Path) -> list[Path]:
        """Existing ``.parte_NN.txt`` files of a recording (for cleanup)."""
        folder = (
            _safe_child(self.paths.llm, relative.parent)
            if relative.parent != Path(".")
            else self.paths.llm
        )
        if not folder.is_dir():
            return []
        prefix = self.base_name(relative) + ".parte_"
        return sorted(
            p for p in folder.iterdir() if p.name.startswith(prefix) and p.suffix == ".txt"
        )

    def root_relative(self, path: Path) -> str:
        """Path relative to the project root, for journal records."""
        try:
            return path.resolve().relative_to(self.paths.root).as_posix()
        except ValueError:
            return str(path)


__all__ = [
    "CANONICAL_SUFFIX",
    "DeliverableLayout",
    "OutputNamer",
    "UnsafeOutputPathError",
    "disambiguate",
    "render_name",
    "sanitize_name",
]
