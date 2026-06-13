"""Transcript Markdown generation grouped by speaker turns + text splitting.

Filler filtering (`remove_fillers`):
    "safe"       — drop pure hesitation sounds only (eh, um, mmm).
    "aggressive" — additionally drop discourse markers (sí, no, claro, ok).
                   WARNING: those can be real answers ("¿Aprobamos?" — "Sí.").
    "off"        — keep everything.
    The .json and .srt ALWAYS keep all segments for traceability; the filter
    only affects the readable .transcript.md.
"""

from __future__ import annotations

import re
from pathlib import Path

from speakerscribe.audio import format_hms
from speakerscribe.config import (
    FILLERS_SAFE,
    SPK_NO_DIARIZATION,
    FillerMode,
    fillers_for,
)
from speakerscribe.io_utils import atomic_write_text
from speakerscribe.logging_config import logger

# Punctuation stripped before checking against the filler set
_PUNCT_STRIP_RE = re.compile(r"[\.\,\;\:\!\?\¿\¡\…\-\–\—\(\)\"'`]+$")
_LEADING_PUNCT_RE = re.compile(r"^[\.\,\;\:\!\?\¿\¡\…\-\–\—\(\)\"'`]+")


def is_filler_only(text: str, language: str | None, mode: FillerMode = "safe") -> bool:
    """Return True if `text` is composed entirely of filler tokens.

    Comparison is case-insensitive and strips leading/trailing punctuation.
    The filler set depends on `mode` (see `FILLERS_SAFE` /
    `FILLERS_AGGRESSIVE` in config). Languages without a defined set always
    return False.

    Args:
        text: Segment text to evaluate.
        language: Two-letter language code (e.g. "es", "en"). None disables
            the filter (returns False).
        mode: "off" | "safe" | "aggressive".

    Returns:
        True if the segment should be considered a pure filler.
    """
    if not text or not language or mode == "off":
        return False
    fillers = fillers_for(language, mode)
    if not fillers:
        return False
    cleaned = _LEADING_PUNCT_RE.sub("", text.strip().lower())
    cleaned = _PUNCT_STRIP_RE.sub("", cleaned).strip()
    if not cleaned:
        return True
    return cleaned in fillers


def generate_transcript_md(
    segments: list[dict],
    output_md: Path,
    metadata: dict,
    gap_max_s: float = 2.0,
    *,
    remove_fillers: FillerMode | bool = "safe",
) -> int:
    """Generate a `.transcript.md` file by grouping consecutive same-speaker segments.

    A "turn" is a sequence of consecutive segments by the same speaker without
    a pause longer than `gap_max_s`. Long silences from the same speaker open
    a new block (more readable for meeting transcripts).

    If diarization was not run, the header makes that explicit so the reader
    does not confuse the file with one that has real speaker labels.

    Args:
        segments: List of dicts with keys: start, end, text, speaker (optional).
        output_md: Destination Markdown path.
        metadata: Run metadata. Used keys: audio_file, duration_minutes,
            speakers_summary, diarization_enabled, model, package_version, etc.
        gap_max_s: If two consecutive segments by the same speaker are more
            than N seconds apart, they are split into separate blocks.
        remove_fillers: Filler filtering mode ("off" | "safe" | "aggressive").
            Booleans are accepted for backward compatibility
            (True -> "safe", False -> "off").

    Returns:
        Number of turn blocks written.
    """
    if isinstance(remove_fillers, bool):  # legacy callers
        remove_fillers = "safe" if remove_fillers else "off"
    output_md.parent.mkdir(parents=True, exist_ok=True)

    if not segments:
        output_md.write_text("# Empty transcript\n", encoding="utf-8")
        return 0

    diar_enabled = bool(metadata.get("diarization_enabled"))
    language = metadata.get("language_detected")

    # ── Filter fillers if requested
    if remove_fillers != "off":
        before = len(segments)
        segments = [
            s for s in segments if not is_filler_only(s.get("text", ""), language, remove_fillers)
        ]
        n_dropped = before - len(segments)
        if n_dropped:
            logger.info(
                f"   Filtered {n_dropped} filler-only segments "
                f"(mode={remove_fillers}) from .transcript.md"
            )

    if not segments:
        output_md.write_text("# Empty transcript (all fillers)\n", encoding="utf-8")
        return 0

    # ── Group consecutive segments by the same speaker
    blocks: list[dict] = []
    current: dict | None = None
    for seg in segments:
        speaker = seg.get("speaker") or SPK_NO_DIARIZATION
        if (
            current is None
            or current["speaker"] != speaker
            or seg["start"] - current["end"] > gap_max_s
        ):
            if current is not None:
                blocks.append(current)
            current = {
                "speaker": speaker,
                "start": seg["start"],
                "end": seg["end"],
                "texts": [seg["text"]],
            }
        else:
            current["end"] = seg["end"]
            current["texts"].append(seg["text"])
    if current is not None:
        blocks.append(current)

    # ── Header
    speakers = sorted({b["speaker"] for b in blocks})
    lines: list[str] = [
        f"# Transcript — {metadata.get('audio_file', 'audio')}",
        "",
    ]

    if not diar_enabled:
        lines.extend(
            [
                "> **Diarization not available** — turns shown below are not "
                "speaker-separated. They are grouped by pauses in the audio, "
                "not by who is speaking.",
                "",
            ]
        )

    lines.extend(
        [
            f"- **Processed at (UTC):** {metadata.get('processed_at', '-')}",
            f"- **Duration:** {metadata.get('duration_minutes', 0):.1f} min",
            f"- **ASR model:** {metadata.get('model', '-')} "
            f"(faster-whisper {metadata.get('faster_whisper_version', '-')})",
        ]
    )

    if diar_enabled:
        lines.append(
            f"- **Diarization model:** {metadata.get('diarization_model', '-')} "
            f"(pyannote.audio {metadata.get('pyannote_version', '-')})"
        )
        lines.append(f"- **Speakers detected:** {len(speakers)} ({', '.join(speakers)})")

    lines.extend(
        [
            f"- **Total turns:** {len(blocks)}",
            f"- **Total words:** {metadata.get('total_words', 0):,}",
            "",
        ]
    )

    if metadata.get("chunked"):
        lines.append(
            f"> Long-audio mode: this transcript was assembled from "
            f"{metadata.get('n_chunks', 0)} overlapping chunks."
        )
        lines.append("")

    if remove_fillers != "off" and language in FILLERS_SAFE:
        lines.append(
            f"> Filler words for `{language}` removed from this readable view "
            f"(mode: {remove_fillers}). The .json contains all original segments."
        )
        lines.append("")

    if diar_enabled:
        lines.extend(
            [
                "> Labels `SPEAKER_XX` are auto-assigned. Use "
                "`rename_speakers_in_outputs()` to map them to real names.",
                "",
            ]
        )

    lines.extend(["---", ""])

    # ── Blocks
    for b in blocks:
        text = " ".join(t.strip() for t in b["texts"] if t.strip())
        lines.append(f"### {b['speaker']} · {format_hms(b['start'])} → {format_hms(b['end'])}")
        lines.append("")
        lines.append(text)
        lines.append("")

    atomic_write_text(output_md, "\n".join(lines))
    logger.info(f"Transcript MD: {len(blocks)} turns in {output_md.name}")
    return len(blocks)


def split_text_by_words(
    txt_file: Path,
    output_dir: Path,
    max_words: int,
    has_speakers: bool | None = None,
) -> int:
    """Split a `.txt` file into chunks of at most `max_words` words each.

    Behavior depends on whether speaker labels are present:
        - Without labels: pure word-count split.
        - With labels: line-based split that never breaks a turn in half.
          The last chunk may have fewer words.

    Args:
        txt_file: Source .txt file.
        output_dir: Destination directory for split files.
        max_words: Soft cap on words per output file.
        has_speakers: If known from metadata, pass it directly. If None, the
            function inspects the first lines of the file to decide.

    Returns:
        Number of split files generated.
    """
    if not txt_file.exists():
        logger.warning(f"File not found: {txt_file}")
        return 0

    text = txt_file.read_text(encoding="utf-8")
    if not text.strip():
        logger.warning(f"Empty file: {txt_file.name}")
        return 0

    base, ext = txt_file.stem, txt_file.suffix
    lines = [ln for ln in text.splitlines() if ln.strip()]

    output_dir.mkdir(parents=True, exist_ok=True)

    if has_speakers is None:
        has_speakers = any(ln.startswith("[SPEAKER_") for ln in lines)

    if not has_speakers:
        words = text.split()
        n_files = (len(words) + max_words - 1) // max_words
        for i in range(n_files):
            start, end = i * max_words, (i + 1) * max_words
            (output_dir / f"{base}_{i + 1}{ext}").write_text(
                " ".join(words[start:end]), encoding="utf-8"
            )
        logger.info(f"Word-based split: {n_files} files")
        return n_files

    # Line-based split, preserving turns
    buffers: list[str] = []
    current: list[str] = []
    current_words = 0
    for ln in lines:
        n_words = len(ln.split())
        if current_words + n_words > max_words and current:
            buffers.append("\n".join(current))
            current = []
            current_words = 0
        current.append(ln)
        current_words += n_words
    if current:
        buffers.append("\n".join(current))

    for i, content in enumerate(buffers, 1):
        (output_dir / f"{base}_{i}{ext}").write_text(content, encoding="utf-8")
    logger.info(f"Turn-aware split: {len(buffers)} files")
    return len(buffers)


def write_unified_for_llm(
    txt_file: Path,
    output_dir: Path,
) -> Path | None:
    """Write a single unified transcript file convenient for LLMs with large context.

    Modern LLMs (Claude, GPT-4, Gemini) accept >100k tokens. For audios up to
    ~5h, a single concatenated file is more useful than the chunked splits
    (one paste, one prompt).

    The file content is the same as the source `.txt` (already includes
    `[SPEAKER_XX]` labels when available). Naming convention:
    `<txt_stem>_full_for_llm.txt`.

    Args:
        txt_file: Source .txt produced by `transcribe_streaming`.
        output_dir: Destination directory (typically `paths.splits`).

    Returns:
        Path to the unified file, or None if the source is missing/empty.
    """
    if not txt_file.exists():
        logger.warning(f"File not found: {txt_file}")
        return None
    text = txt_file.read_text(encoding="utf-8")
    if not text.strip():
        logger.warning(f"Empty file: {txt_file.name}")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{txt_file.stem}_full_for_llm.txt"
    target.write_text(text, encoding="utf-8")
    n_words = len(text.split())
    logger.info(f"Unified for LLM: {target.name} ({n_words:,} words)")
    return target


__all__ = [
    "generate_transcript_md",
    "is_filler_only",
    "split_text_by_words",
    "write_unified_for_llm",
]
