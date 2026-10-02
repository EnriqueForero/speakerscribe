"""Speaker post-processing: orphan repair and persistent renames.

Ported unchanged from notebook v5 (characterization-tested against it).
"""

from __future__ import annotations

import bisect
import contextlib
import json
from pathlib import Path
from typing import Any

from speakerscribe.batch.fsio import atomic_write_json, canonical_hash, utc_now

NULL_SPEAKERS = frozenset({None, "", "(no diarization)", "SPEAKER_NO_OVERLAP", "None"})
"""Labels meaning 'no diarization turn covers this segment'."""

UNLABELED = "SIN_HABLANTE"


def is_null_speaker(value: Any) -> bool:
    return value in NULL_SPEAKERS


def repair_orphans(
    segments: list[dict[str, Any]], tolerance_s: float
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Give orphan segments (no diarization turn) a neighbor's speaker.

    ``pyannote/speaker-diarization-community-1`` returns EXCLUSIVE diarization:
    overlapped or low-confidence stretches fall outside every turn. Rules,
    safest first:

    1. Only one real speaker found: every orphan is theirs.
    2. Both temporal neighbors agree: inherit that speaker.
    3. Otherwise inherit from the nearest neighbor if within `tolerance_s`.
    4. The rest stay unlabeled and count against the unlabeled-fraction cap.

    Args:
        segments: Dicts with ``start``, ``end``, ``text``, ``speaker``.
        tolerance_s: Max distance (s) to inherit from a neighbor.

    Returns:
        (segments, stats) with ``huerfanos``, ``adoptados``,
        ``hablantes_reales``. Input segments are never mutated.
    """
    stats = {"huerfanos": 0, "adoptados": 0, "hablantes_reales": 0}
    if not segments:
        return segments, stats
    labels: list[str | None] = [
        None if is_null_speaker(s.get("speaker")) else str(s.get("speaker")) for s in segments
    ]
    real = sorted({label for label in labels if label})
    stats["hablantes_reales"] = len(real)
    stats["huerfanos"] = sum(1 for label in labels if label is None)
    if not real or stats["huerfanos"] == 0:
        return segments, stats

    new_labels = list(labels)
    if len(real) == 1:
        new_labels = [real[0] if label is None else label for label in labels]
    else:
        labeled_idx = [i for i, label in enumerate(labels) if label]
        for i, label in enumerate(labels):
            if label is not None:
                continue
            pos = bisect.bisect_left(labeled_idx, i)
            prev_i = labeled_idx[pos - 1] if pos > 0 else None
            next_i = labeled_idx[pos] if pos < len(labeled_idx) else None
            if prev_i is not None and next_i is not None and labels[prev_i] == labels[next_i]:
                new_labels[i] = labels[prev_i]
                continue
            candidate, distance = None, float("inf")
            if prev_i is not None:
                d = abs(
                    float(segments[i].get("start", 0.0)) - float(segments[prev_i].get("end", 0.0))
                )
                if d < distance:
                    candidate, distance = labels[prev_i], d
            if next_i is not None:
                d = abs(
                    float(segments[next_i].get("start", 0.0)) - float(segments[i].get("end", 0.0))
                )
                if d < distance:
                    candidate, distance = labels[next_i], d
            if candidate is not None and distance <= float(tolerance_s):
                new_labels[i] = candidate

    stats["adoptados"] = sum(
        1 for old, new in zip(labels, new_labels, strict=True) if old is None and new is not None
    )
    if not stats["adoptados"]:
        return segments, stats
    out = []
    for seg, label in zip(segments, new_labels, strict=True):
        if label and is_null_speaker(seg.get("speaker")):
            seg = {**seg, "speaker": label}
        out.append(seg)
    return out, stats


def rename_hash(mapping: dict[str, str]) -> str:
    """Fingerprint of a rename mapping ('' when empty)."""
    return canonical_hash(mapping) if mapping else ""


def apply_renames(segments: list[dict[str, Any]], mapping: dict[str, str]) -> list[dict[str, Any]]:
    """Apply a mapping in ONE pass (a swap 00→01, 01→Ana never chains)."""
    if not mapping:
        return segments
    out = []
    for seg in segments:
        raw = seg.get("speaker")
        if raw is not None and str(raw) in mapping:
            seg = {**seg, "speaker": mapping[str(raw)]}
        out.append(seg)
    return out


class RenameStore:
    """Persistent speaker renames per source id (``renombres/<id>.json``)."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self._cache: dict[str, dict[str, str]] | None = None

    def _load_all(self) -> dict[str, dict[str, str]]:
        if self._cache is None:
            self._cache = {}
            if self.folder.is_dir():
                for path in self.folder.glob("*.json"):
                    with contextlib.suppress(OSError, ValueError):
                        data = json.loads(path.read_text(encoding="utf-8"))
                        mapping = data.get("mapping") or {}
                        if isinstance(mapping, dict) and mapping:
                            self._cache[path.stem] = {str(k): str(v) for k, v in mapping.items()}
        return self._cache

    def get(self, source_id: str) -> dict[str, str]:
        return dict(self._load_all().get(source_id, {}))

    def save(self, source_id: str, mapping: dict[str, str]) -> str:
        """Persist (or clear) a mapping; return its fingerprint."""
        cache = self._load_all()
        clean = {str(k): str(v) for k, v in mapping.items() if str(k) != str(v)}
        target = self.folder / f"{source_id}.json"
        if clean:
            cache[source_id] = clean
            atomic_write_json(target, {"mapping": clean, "actualizado": utc_now()})
        else:
            cache.pop(source_id, None)
            target.unlink(missing_ok=True)
        return rename_hash(clean)


__all__ = [
    "NULL_SPEAKERS",
    "UNLABELED",
    "RenameStore",
    "apply_renames",
    "is_null_speaker",
    "rename_hash",
    "repair_orphans",
]
