"""Pure renderers: segments + metadata -> deliverable text.

No I/O except the Markdown renderer, which delegates to
`speakerscribe.output.generate_transcript_md` (it writes a file) and reads
the result back so every deliverable goes through the same atomic writer.

Ported from notebook v5 (cell 5) with two deliberate changes:

* The strict monotonic check tolerates `RenderOptions.monotonic_tolerance_s`
  of backwards drift (v5 hard-coded 0.05 s). Word-level alignment
  produces 0.39-0.85 s boundary overlaps in real meetings (11 of 94 files
  were flagged "Timestamps no monotónicos" until 2026-10-01); the library
  now clamps those at the source, and this tolerance is the second net.
* The header names the library version instead of the notebook version.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from speakerscribe.batch.speakers import UNLABELED, is_null_speaker

RENDER_SCHEMA = "ss-render-2"
"""Bumped from v5's ``ss-v5-render-1``: header line changed."""

SILENT_LABEL = "SIN_VOZ"
NO_SPEECH_TEXT = "(sin voz detectada)"


class TranscriptRejectedError(ValueError):
    """A strict render found the transcript unpublishable as 'ok'."""


@dataclass(frozen=True)
class RenderOptions:
    """Everything a renderer needs besides the transcript itself."""

    include_header: bool = True
    timestamp_ms: bool = False
    monotonic_tolerance_s: float = 1.0
    max_unlabeled_fraction: float = 0.35
    accept_silent_audio: bool = True
    md_gap_s: float = 3.0
    md_fillers: str = "safe"
    split_words: int = 1950
    model: str = "large-v3"
    beam_size: int = 5
    batch_size: int = 8
    producer: str = "speakerscribe"


@dataclass(frozen=True)
class RenderContext:
    """Input shared by every renderer of one recording."""

    segments: list[dict[str, Any]]
    metadata: dict[str, Any]
    source_rel: str
    options: RenderOptions
    processed_utc: str = ""
    repair: dict[str, int] = field(default_factory=dict)


# ── Small formatters ─────────────────────────────────────────────────
def format_ts(seconds: float, with_ms: bool = False) -> str:
    """``01:02:05`` or, with `with_ms`, ``01:02:05.240``."""
    if with_ms:
        total_ms = max(0, int(round(float(seconds) * 1000)))
        h, rest = divmod(total_ms, 3_600_000)
        m, rest = divmod(rest, 60_000)
        s, ms = divmod(rest, 1000)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    total = max(0, int(round(float(seconds))))
    h, rest = divmod(total, 3600)
    m, s = divmod(rest, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def srt_ts(seconds: float) -> str:
    """SRT timestamp ``HH:MM:SS,mmm``."""
    total_ms = max(0, int(round(float(seconds) * 1000)))
    h, rest = divmod(total_ms, 3_600_000)
    m, rest = divmod(rest, 60_000)
    s, ms = divmod(rest, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def clean_text(value: Any) -> str:
    """Collapse internal whitespace."""
    return " ".join(str(value or "").split())


def speaker_label(raw: Any) -> str:
    return UNLABELED if is_null_speaker(raw) else str(raw)


# ── Canonical .txt ───────────────────────────────────────────────────
def header_lines(
    ctx: RenderContext, status: str, reason: str | None, speakers: set[str]
) -> list[str]:
    """Commented header of the canonical ``.txt`` (``# key: value`` lines)."""
    md = ctx.metadata
    opts = ctx.options
    prob = md.get("language_probability")
    language = md.get("language_detected") or md.get("language") or "?"
    lines = [
        f"# archivo_origen: {ctx.source_rel}",
        f"# duracion: {format_ts(md.get('duration_seconds') or 0)} · idioma: {language}"
        + (f" (p={prob:.2f})" if isinstance(prob, int | float) else ""),
        f"# hablantes: {len(speakers)}" + (f" ({', '.join(sorted(speakers))})" if speakers else ""),
        f"# modelo: faster-whisper {md.get('model', opts.model)} + "
        + str(md.get("diarization_model") or "pyannote community-1")
        + f" · beam {opts.beam_size} · batch {opts.batch_size}",
        f"# palabras: {md.get('total_words', '?')} · estado: {status}"
        + (f" · motivo: {reason}" if reason else ""),
    ]
    if ctx.repair.get("adoptados"):
        lines.append(
            f"# reparacion_hablantes: {ctx.repair['adoptados']} de "
            f"{ctx.repair['huerfanos']} segmentos huérfanos adoptados "
            f"({ctx.repair['hablantes_reales']} hablantes)"
        )
    lines += [
        f"# esquema: {RENDER_SCHEMA} · {opts.producer} · procesado: {ctx.processed_utc}",
        "# " + "─" * 68,
        "",
    ]
    return lines


def render_transcript(ctx: RenderContext, status: str, reason: str | None = None) -> str:
    """The canonical ``.txt``: ``[start - end] SPEAKER: text`` per segment.

    ``status == "ok"`` enables strict checks (diarization present,
    monotonic timestamps within tolerance, unlabeled-word cap). Silent audio
    yields a first-class ``SIN_VOZ`` line. Degraded statuses render leniently
    but say so in the header.

    Raises:
        TranscriptRejectedError: Only in strict mode, when the result is not
            publishable as 'ok'.
    """
    opts = ctx.options
    md = ctx.metadata
    segments = ctx.segments
    strict = status == "ok"
    if strict and not segments:
        if not opts.accept_silent_audio:
            raise TranscriptRejectedError("La transcripción no contiene segmentos")
        duration = max(0.0, float(md.get("duration_seconds") or 0.0))
        body = [
            f"[{format_ts(0, opts.timestamp_ms)} - {format_ts(duration, opts.timestamp_ms)}] "
            f"{SILENT_LABEL}: {NO_SPEECH_TEXT}"
        ]
        head = header_lines(ctx, "ok_sin_voz", None, set()) if opts.include_header else []
        return "\n".join(head + body) + "\n"

    if strict and not md.get("diarization_enabled"):
        raise TranscriptRejectedError("La diarización no está presente")
    speakers = {str(s.get("speaker")) for s in segments if not is_null_speaker(s.get("speaker"))}
    if strict and segments and not speakers:
        raise TranscriptRejectedError("No se detectó ningún hablante válido")

    lines: list[str] = []
    latest_start = -1.0
    words_total = words_unlabeled = 0
    for idx, seg in enumerate(segments, 1):
        text = clean_text(seg.get("text"))
        if not text:
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        if strict:
            if end < start:
                raise TranscriptRejectedError(f"Segmento {idx} termina antes de empezar")
            if start + opts.monotonic_tolerance_s < latest_start:
                raise TranscriptRejectedError(
                    f"Timestamps no monotónicos en segmento {idx} "
                    f"(retrocede {latest_start - start:.2f} s)"
                )
        else:
            end = max(end, start)
        latest_start = max(latest_start, start)
        label = speaker_label(seg.get("speaker"))
        n_words = len(text.split())
        words_total += n_words
        if label == UNLABELED:
            words_unlabeled += n_words
        lines.append(
            f"[{format_ts(start, opts.timestamp_ms)} - {format_ts(end, opts.timestamp_ms)}] "
            f"{label}: {text}"
        )
    if strict and words_total:
        fraction = words_unlabeled / words_total
        if fraction > opts.max_unlabeled_fraction:
            raise TranscriptRejectedError(
                f"{fraction:.1%} de palabras sin hablante (máx {opts.max_unlabeled_fraction:.1%})"
            )
    if not lines:
        if strict:
            raise TranscriptRejectedError("No quedó texto publicable")
        lines.append(NO_SPEECH_TEXT)
    head = header_lines(ctx, status, reason, speakers) if opts.include_header else []
    return "\n".join(head + lines) + "\n"


# ── Secondary deliverables ───────────────────────────────────────────
def render_srt(ctx: RenderContext) -> str:
    """SRT subtitles, consecutive numbering, speaker as prefix."""
    blocks: list[str] = []
    n = 0
    for seg in ctx.segments:
        text = clean_text(seg.get("text"))
        if not text:
            continue
        n += 1
        start = float(seg.get("start", 0.0))
        end = max(float(seg.get("end", start)), start)
        raw = seg.get("speaker")
        prefix = "" if is_null_speaker(raw) else f"{raw}: "
        blocks.append(f"{n}\n{srt_ts(start)} --> {srt_ts(end)}\n{prefix}{text}\n")
    return "\n".join(blocks) + ("\n" if blocks else "")


def speaker_turns(segments: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Merge consecutive same-speaker segments -> ``[(speaker, text)]``."""
    turns: list[tuple[str, str]] = []
    current: str | None = None
    buffer: list[str] = []
    for seg in segments:
        text = clean_text(seg.get("text"))
        if not text:
            continue
        label = speaker_label(seg.get("speaker"))
        if label != current and buffer:
            turns.append((current or UNLABELED, " ".join(buffer)))
            buffer = []
        current = label
        buffer.append(text)
    if buffer:
        turns.append((current or UNLABELED, " ".join(buffer)))
    return turns


def render_full_llm(ctx: RenderContext) -> str:
    """Running text by speaker turn, no timestamps — to paste into an LLM."""
    md = ctx.metadata
    language = md.get("language_detected") or md.get("language") or "?"
    parts = [
        f"Transcripción de: {ctx.source_rel}",
        f"Idioma: {language} · Duración: {format_ts(md.get('duration_seconds') or 0)} "
        f"· Palabras: {md.get('total_words', '?')}",
        "",
    ]
    turns = speaker_turns(ctx.segments)
    if not turns:
        parts.append(NO_SPEECH_TEXT)
    for speaker, text in turns:
        parts.append(f"{speaker}: {text}")
        parts.append("")
    return "\n".join(parts).rstrip("\n") + "\n"


def render_plain(ctx: RenderContext) -> str:
    """One line per segment ``[SPEAKER] text``, no timestamps."""
    lines = [
        f"[{speaker_label(seg.get('speaker'))}] {clean_text(seg.get('text'))}"
        for seg in ctx.segments
        if clean_text(seg.get("text"))
    ]
    return "\n".join(lines) + "\n" if lines else NO_SPEECH_TEXT + "\n"


def render_splits(full_text: str, words_per_split: int) -> list[str]:
    """~N-word chunks on whole lines, each with a ``# parte i/n`` header."""
    chunks: list[list[str]] = [[]]
    count = 0
    for line in full_text.splitlines():
        n = len(line.split())
        if count + n > words_per_split and count > 0:
            chunks.append([])
            count = 0
        chunks[-1].append(line)
        count += n
    total = len(chunks)
    parts = [
        f"# parte {i}/{total}\n" + "\n".join(chunk).strip() + "\n"
        for i, chunk in enumerate(chunks, 1)
        if any(line.strip() for line in chunk)
    ]
    return parts or [f"# parte 1/1\n{full_text}"]


def strip_words(metadata: dict[str, Any]) -> dict[str, Any]:
    """Metadata with segments reduced to start/end/text/speaker (no words)."""
    out = {k: v for k, v in metadata.items() if k != "segments"}
    out["segments"] = [
        {
            "start": s.get("start"),
            "end": s.get("end"),
            "text": s.get("text"),
            "speaker": s.get("speaker"),
        }
        for s in (metadata.get("segments") or [])
    ]
    return out


def render_json(ctx: RenderContext) -> str:
    """Engine metadata (raw speakers, no word timings) as indented JSON."""
    return json.dumps(strip_words(ctx.metadata), ensure_ascii=False, indent=1, default=str)


def render_markdown(ctx: RenderContext) -> str:
    """Readable ``.transcript.md`` via the library's turn grouper."""
    from speakerscribe.output import generate_transcript_md

    with tempfile.TemporaryDirectory(prefix="ss_md_") as tmp:
        target = Path(tmp) / "transcript.md"
        generate_transcript_md(
            list(ctx.segments),
            target,
            dict(ctx.metadata),
            gap_max_s=ctx.options.md_gap_s,
            remove_fillers=ctx.options.md_fillers,  # type: ignore[arg-type]
        )
        return target.read_text(encoding="utf-8")


# ── Registry (Open/Closed: add a deliverable without touching callers) ──
Location = Literal["formats", "llm"]


@dataclass(frozen=True)
class TextDeliverable:
    """A single-file deliverable rendered next to the canonical name."""

    key: str
    location: Location
    suffix: str
    render: Callable[[RenderContext], str]


DELIVERABLES: dict[str, TextDeliverable] = {
    d.key: d
    for d in (
        TextDeliverable("md", "formats", ".transcript.md", render_markdown),
        TextDeliverable("srt", "formats", ".srt", render_srt),
        TextDeliverable("json", "formats", ".json", render_json),
        TextDeliverable("plano", "formats", ".plano.txt", render_plain),
        TextDeliverable("full_llm", "llm", ".full_for_llm.txt", render_full_llm),
    )
}
SPLITS_KEY = "splits"


__all__ = [
    "DELIVERABLES",
    "NO_SPEECH_TEXT",
    "RENDER_SCHEMA",
    "SILENT_LABEL",
    "SPLITS_KEY",
    "RenderContext",
    "RenderOptions",
    "TextDeliverable",
    "TranscriptRejectedError",
    "clean_text",
    "format_ts",
    "header_lines",
    "render_full_llm",
    "render_json",
    "render_markdown",
    "render_plain",
    "render_splits",
    "render_srt",
    "render_transcript",
    "speaker_label",
    "speaker_turns",
    "srt_ts",
    "strip_words",
]
