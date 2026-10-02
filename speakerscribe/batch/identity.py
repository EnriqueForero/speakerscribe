"""Source identity and the input↔output workspace binding.

Identity rules (unchanged from notebook v5, so existing journals stay valid):

* ``source_id = canonical_hash({"root": canonical_hash(str(input_root)),
  "relative": relative_posix_path})`` — identifies a *location*.
* ``content signature`` = full SHA-256 of the bytes (or a ``fast:`` sample)
  — identifies *content*, enables dedup across locations.

The workspace binding (``workspace_identity.json``) ties a state folder to
ONE input root, so outputs of two collections never mix by mistake. Moving
the input folder therefore requires an explicit, audited `rebind_workspace`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from speakerscribe.batch.fsio import atomic_write_json, canonical_hash, utc_now
from speakerscribe.batch.paths import BatchPaths

IDENTITY_SCHEMA = 1


class WorkspaceBindingError(RuntimeError):
    """The state folder belongs to a different input root."""


@dataclass(frozen=True)
class SourceInfo:
    """Identity and stat of one input file (plus its glossary sidecar)."""

    path: Path
    relative: Path
    source_id: str
    size_bytes: int
    mtime_ns: int
    prompt_path: Path | None
    prompt_size: int | None
    prompt_mtime_ns: int | None

    @property
    def relative_posix(self) -> str:
        return self.relative.as_posix()


def root_id(input_root: Path) -> str:
    """Hash of the input root path (component of every source id)."""
    return canonical_hash(str(input_root))


def source_id_for(input_root: Path, relative: Path) -> str:
    """Location identity of a file under `input_root`."""
    return canonical_hash({"root": root_id(input_root), "relative": relative.as_posix()})


def source_payload(
    info: SourceInfo, input_root: Path, signature: str | None = None, hash_mode: str = "full"
) -> dict[str, Any]:
    """JSON-serializable description of a source for journal events (v5 schema)."""
    prompt_rel = None
    if info.prompt_path is not None:
        try:
            prompt_rel = info.prompt_path.relative_to(input_root).as_posix()
        except ValueError:
            prompt_rel = info.prompt_path.name
    return {
        "id": info.source_id,
        "relative_path": info.relative_posix,
        "size_bytes": info.size_bytes,
        "mtime_ns": info.mtime_ns,
        "content_signature": signature,
        "signature_mode": hash_mode,
        "prompt_path": prompt_rel,
        "prompt_size": info.prompt_size,
        "prompt_mtime_ns": info.prompt_mtime_ns,
    }


def source_stat_matches(record: dict[str, Any], info: SourceInfo) -> bool:
    """True if the journal record describes the same bytes on disk (by stat)."""
    src = record.get("source") or {}
    return (
        src.get("size_bytes") == info.size_bytes
        and src.get("mtime_ns") == info.mtime_ns
        and src.get("prompt_size") == info.prompt_size
        and src.get("prompt_mtime_ns") == info.prompt_mtime_ns
    )


def _identity_document(input_root: Path) -> dict[str, Any]:
    return {
        "schema_version": IDENTITY_SCHEMA,
        "input_root": str(input_root),
        "input_root_sha256": canonical_hash(str(input_root)),
    }


def bind_workspace(paths: BatchPaths, previous_roots: set[str]) -> dict[str, Any]:
    """Bind the state folder to `paths.input`, or verify an existing binding.

    Args:
        paths: Resolved batch paths.
        previous_roots: ``input_root_sha256`` values seen in earlier
            ``batch_started`` events (used only when no identity file exists).

    Returns:
        The identity document in force.

    Raises:
        WorkspaceBindingError: If the state belongs to another input root.
    """
    wanted = _identity_document(paths.input)
    if paths.identity.exists():
        try:
            current = json.loads(paths.identity.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise WorkspaceBindingError(
                f"No se pudo leer {paths.identity}; no se tocará la salida."
            ) from e
        if current.get("input_root_sha256") != wanted["input_root_sha256"]:
            raise WorkspaceBindingError(
                "El estado está vinculado a OTRA carpeta de entrada.\n"
                f"  Registrada: {current.get('input_root')}\n"
                f"  Actual:     {paths.input}\n"
                "  Si usted movió la carpeta de entrada a propósito, ejecute una vez "
                "rebind_workspace(settings) (celda 4 del notebook de lote, REVINCULAR=True)."
            )
        return current
    if previous_roots and wanted["input_root_sha256"] not in previous_roots:
        raise WorkspaceBindingError(
            "El historial de este estado corresponde a otra carpeta de entrada. "
            "Use rebind_workspace(settings) si la movió a propósito."
        )
    atomic_write_json(paths.identity, wanted)
    return wanted


def rebind_document(paths: BatchPaths) -> tuple[dict[str, Any], dict[str, Any] | None, Path | None]:
    """Write a new identity for `paths.input`, backing up the previous one.

    Returns:
        (new identity, previous identity or None, backup path or None).
    """
    previous: dict[str, Any] | None = None
    backup: Path | None = None
    if paths.identity.exists():
        try:
            previous = json.loads(paths.identity.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = None
        backup = paths.identity.with_name(f"{paths.identity.name}.bak.{int(time.time())}")
        backup.write_bytes(paths.identity.read_bytes())
    new = {**_identity_document(paths.input), "rebound_utc": utc_now()}
    if previous is not None:
        new["rebound_from"] = previous.get("input_root")
    atomic_write_json(paths.identity, new)
    return new, previous, backup


__all__ = [
    "IDENTITY_SCHEMA",
    "SourceInfo",
    "WorkspaceBindingError",
    "bind_workspace",
    "rebind_document",
    "root_id",
    "source_id_for",
    "source_payload",
    "source_stat_matches",
]
