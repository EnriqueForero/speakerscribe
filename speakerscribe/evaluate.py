"""Quantitative evaluation against user references: WER (jiwer) and DER.

Optional dependencies — install with: pip install 'speakerscribe[bench]'
    jiwer            — Word Error Rate / Match Error Rate / Word Info Lost.
    pyannote.metrics — Diarization Error Rate (with pyannote.core).

These metrics turn "the transcript looks good" into numbers you can put in
BENCHMARKS.md and compare across releases. The DER computed here is
END-TO-END: it scores the speaker labels of the FINAL transcript segments
(diarization + assignment combined), which is what a reader experiences —
not the raw diarizer output alone.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from speakerscribe.config import SPK_NO_OVERLAP, WorkspacePaths
from speakerscribe.logging_config import logger

_BENCH_INSTALL_HINT = (
    "Benchmark extras not installed. Run: pip install 'speakerscribe[bench]' "
    "(provides jiwer and pyannote.metrics)."
)

_SPEAKER_PREFIX_RE = re.compile(r"^\[[^\]]+\]\s*")


def _require_jiwer() -> Any:
    try:
        import jiwer
    except ImportError as e:  # pragma: no cover - exercised via bench tests
        raise ImportError(_BENCH_INSTALL_HINT) from e
    return jiwer


def normalize_for_wer(text: str) -> str:
    """Normalize text for fair WER comparison.

    Lowercases, strips punctuation (Unicode-aware for es/pt accents are
    KEPT — they are letters), collapses whitespace, and removes
    `[SPEAKER_XX]` prefixes. Both reference and hypothesis go through the
    SAME normalization, so stylistic differences don't count as errors.

    Args:
        text: Raw transcript text.

    Returns:
        Normalized single-line text.
    """
    lines = [_SPEAKER_PREFIX_RE.sub("", line) for line in text.splitlines()]
    text = " ".join(lines).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compute_wer(reference: str, hypothesis: str) -> dict[str, Any]:
    """Compute WER/MER/WIL between a reference and a hypothesis transcript.

    Args:
        reference: Human ground-truth transcript (raw text).
        hypothesis: System transcript (raw text; speaker prefixes tolerated).

    Returns:
        Dict with keys: wer, mer, wil, reference_words, hypothesis_words.

    Raises:
        ImportError: If jiwer is not installed (actionable message).
        ValueError: If the normalized reference is empty.
    """
    jiwer = _require_jiwer()
    ref = normalize_for_wer(reference)
    hyp = normalize_for_wer(hypothesis)
    if not ref:
        raise ValueError("Reference transcript is empty after normalization.")
    out = jiwer.process_words(ref, hyp)
    return {
        "wer": round(float(out.wer), 4),
        "mer": round(float(out.mer), 4),
        "wil": round(float(out.wil), 4),
        "reference_words": len(ref.split()),
        "hypothesis_words": len(hyp.split()),
    }


def parse_rttm(path: Path) -> list[dict[str, Any]]:
    """Parse a SPEAKER-type RTTM file into turn dicts.

    RTTM line format (NIST):
        SPEAKER <uri> <chan> <start> <dur> <NA> <NA> <speaker> <NA> <NA>

    Args:
        path: Path to the .rttm file.

    Returns:
        List of `{"start", "end", "speaker"}` dicts sorted by start.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If no SPEAKER lines could be parsed.
    """
    if not path.exists():
        raise FileNotFoundError(f"RTTM not found: {path}")
    turns: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if len(parts) < 8 or parts[0].upper() != "SPEAKER":
            continue
        start = float(parts[3])
        dur = float(parts[4])
        turns.append({"start": start, "end": start + dur, "speaker": parts[7]})
    if not turns:
        raise ValueError(f"No SPEAKER lines parsed from {path}")
    turns.sort(key=lambda t: t["start"])
    return turns


def write_rttm(turns: list[dict[str, Any]], uri: str, path: Path) -> Path:
    """Write turns as a SPEAKER-type RTTM file (for external scoring tools).

    Args:
        turns: `{"start", "end", "speaker"}` dicts.
        uri: Recording identifier (RTTM field 2).
        path: Destination .rttm path.

    Returns:
        The destination path.
    """
    lines = [
        f"SPEAKER {uri} 1 {t['start']:.3f} {max(0.0, t['end'] - t['start']):.3f} "
        f"<NA> <NA> {t['speaker']} <NA> <NA>"
        for t in turns
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _turns_to_annotation(turns: list[dict[str, Any]], uri: str) -> Any:
    from pyannote.core import Annotation, Segment

    ann = Annotation(uri=uri)
    for t in turns:
        if t["end"] > t["start"]:
            ann[Segment(t["start"], t["end"])] = t["speaker"]
    return ann


def compute_der(
    reference_turns: list[dict[str, Any]],
    hypothesis_turns: list[dict[str, Any]],
    *,
    uri: str = "audio",
    collar: float = 0.25,
    skip_overlap: bool = False,
) -> dict[str, Any]:
    """Compute the Diarization Error Rate between two turn lists.

    Args:
        reference_turns: Ground-truth turns (`parse_rttm` output).
        hypothesis_turns: System turns (e.g. final segments' speakers).
        uri: Recording identifier.
        collar: Forgiveness collar in seconds around reference boundaries
            (0.25 is the common convention).
        skip_overlap: Exclude overlapped-speech regions from scoring.

    Returns:
        Dict with keys: der, confusion, missed_detection, false_alarm,
        total (all in seconds except `der`, a ratio).

    Raises:
        ImportError: If pyannote.metrics is not installed.
    """
    try:
        from pyannote.metrics.diarization import DiarizationErrorRate
    except ImportError as e:  # pragma: no cover - exercised via bench tests
        raise ImportError(_BENCH_INSTALL_HINT) from e

    metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    ref = _turns_to_annotation(reference_turns, uri)
    hyp = _turns_to_annotation(hypothesis_turns, uri)
    components = metric(ref, hyp, detailed=True)
    return {
        "der": round(float(components["diarization error rate"]), 4),
        "confusion": round(float(components["confusion"]), 3),
        "missed_detection": round(float(components["missed detection"]), 3),
        "false_alarm": round(float(components["false alarm"]), 3),
        "total": round(float(components["total"]), 3),
    }


def bench_run(
    paths: WorkspacePaths,
    base_name: str,
    reference_txt: Path,
    *,
    reference_rttm: Path | None = None,
    collar: float = 0.25,
) -> dict[str, Any]:
    """Benchmark a finished run against user references and ledger the result.

    Reads `<transcripts>/<base_name>.txt` (hypothesis text) and
    `<transcripts>/<base_name>.json` (segments with speakers for end-to-end
    DER), computes WER (+DER when an RTTM reference is provided), logs a
    summary and appends a `kind="bench"` record to the runs ledger.

    Args:
        paths: WorkspacePaths of the project.
        base_name: Output base name (e.g. "meeting_large-v3-turbo").
        reference_txt: Human ground-truth transcript.
        reference_rttm: Optional ground-truth diarization (enables DER).
        collar: DER collar in seconds.

    Returns:
        Dict with wer/mer/wil (+ der components when applicable).

    Raises:
        FileNotFoundError: If outputs or references are missing.
        ImportError: If bench extras are not installed.
    """
    hyp_txt = paths.transcripts / f"{base_name}.txt"
    hyp_json = paths.transcripts / f"{base_name}.json"
    if not hyp_txt.exists():
        raise FileNotFoundError(f"Hypothesis transcript not found: {hyp_txt}")
    if not reference_txt.exists():
        raise FileNotFoundError(f"Reference transcript not found: {reference_txt}")

    result = compute_wer(reference_txt.read_text(encoding="utf-8"), hyp_txt.read_text("utf-8"))
    result["base_name"] = base_name
    result["der"] = None

    if reference_rttm is not None:
        if not hyp_json.exists():
            raise FileNotFoundError(f"Hypothesis JSON not found (needed for DER): {hyp_json}")
        meta = json.loads(hyp_json.read_text(encoding="utf-8"))
        hyp_turns = [
            {"start": s["start"], "end": s["end"], "speaker": s.get("speaker")}
            for s in meta.get("segments", [])
            if s.get("speaker") and s["speaker"] != SPK_NO_OVERLAP
        ]
        der = compute_der(parse_rttm(reference_rttm), hyp_turns, uri=base_name, collar=collar)
        result.update(der)

    logger.success(
        f"Bench {base_name}: WER={result['wer']:.2%}"
        + (f" DER={result['der']:.2%}" if result.get("der") is not None else "")
    )

    from speakerscribe.io_utils import append_jsonl_line

    append_jsonl_line(
        paths.ledger_path,
        {"kind": "bench", "base_name": base_name, **{k: v for k, v in result.items()}},
    )
    return result


__all__ = [
    "bench_run",
    "compute_der",
    "compute_wer",
    "normalize_for_wer",
    "parse_rttm",
    "write_rttm",
]
