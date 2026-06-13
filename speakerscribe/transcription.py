"""Streaming transcription with faster-whisper + per-segment speaker labeling.

Pipeline core. Each emitted segment is written to disk IMMEDIATELY, keeping
RAM under ~500 MB even for multi-hour audio.

Batched inference (new in 0.3):
    `batch_size > 1` (default 8) routes through faster-whisper's
    `BatchedInferencePipeline` — ~3-4x faster than sequential on large
    models per the official benchmark
    (https://github.com/SYSTRAN/faster-whisper#benchmark). On CUDA OOM the
    batch size is automatically halved down to 1 (exact sequential path,
    parity with pre-0.3 outputs). Useful side effect: batched mode segments
    via VAD and decodes windows independently, which also limits
    hallucination-loop propagation across windows.

Speaker attribution:
    With `speaker_assignment="word"` (default) each segment's words are
    attributed individually and the segment is re-split at speaker changes
    (see `diarization.assign_speakers_by_words`). `"segment"` reproduces
    the legacy one-speaker-per-segment behavior.

Two transcription paths:
    transcribe_streaming  — Single audio file (any duration; batched mode
                            handles long audio natively via its VAD).
    transcribe_chunked    — DEPRECATED legacy path for externally chunked
                            audio. Kept for `long_audio_threshold_min > 0`.
"""

from __future__ import annotations

import gc
import json
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from speakerscribe.audio import AudioChunk, format_srt_timestamp
from speakerscribe.config import (
    WHISPER_PROMPT_TOKEN_LIMIT,
    TranscriptionConfig,
)
from speakerscribe.diarization import (
    assign_speaker_to_segment,
    assign_speakers_by_words,
)
from speakerscribe.io_utils import atomic_write_json
from speakerscribe.logging_config import logger

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

# Heuristic chars-per-token for es/en prose, used only to WARN about prompt
# budget without importing the tokenizer eagerly.
_APPROX_CHARS_PER_TOKEN = 4


def load_whisper_model(config: TranscriptionConfig) -> WhisperModel:
    """Load a faster-whisper model on GPU (float16) or CPU (int8).

    Approximate VRAM usage on NVIDIA T4:
        - large-v3 fp16:        ~3.0 GB
        - large-v3-turbo fp16:  ~1.6 GB  (recommended for diar + ASR)
        - medium fp16:          ~1.5 GB
        - large-v3 int8 CPU:    ~2.3 GB RAM

    Args:
        config: Pipeline configuration.

    Returns:
        Loaded WhisperModel instance.
    """
    import torch
    from faster_whisper import WhisperModel

    device, compute_type = config.resolve_device()
    logger.info(f"Loading model '{config.model}' on {device.upper()} ({compute_type})...")
    t0 = time.time()
    model = WhisperModel(
        config.model,
        device=device,
        compute_type=compute_type,
    )
    logger.success(f"Model loaded in {time.time() - t0:.1f}s")
    if device == "cuda":
        vram_used = torch.cuda.memory_allocated() / 1e9
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"   VRAM: {vram_used:.2f} / {vram_total:.1f} GB")
    return model


def release_whisper_model(model: WhisperModel | None = None) -> None:
    """Best-effort release: drop OUR reference and clear the CUDA cache.

    Honesty note: ``del model`` here only removes this function's local
    reference. If the CALLER still holds a reference, the model is NOT
    freed until that reference dies — Python cannot free an object someone
    else points to. To guarantee release, either delete your own reference
    before calling this, or use the `loaded_whisper` context manager,
    which scopes the only reference for you.

    Args:
        model: Instance to release. If None, only clears the CUDA cache.
    """
    if model is not None:
        del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass
    logger.debug("Whisper reference dropped, CUDA cache cleared")


@contextmanager
def loaded_whisper(config: TranscriptionConfig) -> Iterator[WhisperModel]:
    """Context manager that guarantees the Whisper model is released.

    Unlike calling `release_whisper_model(model)` while still holding
    `model` yourself (which cannot free anything), this scopes the only
    strong reference:

        >>> with loaded_whisper(config) as model:   # doctest: +SKIP
        ...     process_one(media, paths, model, config)
        ... # model freed + CUDA cache cleared here, even on exceptions

    Args:
        config: Pipeline configuration.

    Yields:
        Loaded WhisperModel instance.
    """
    model: WhisperModel | None = load_whisper_model(config)
    try:
        yield model
    finally:
        model = None  # drop the only reference BEFORE collecting
        release_whisper_model(None)


def _config_to_dict(config: TranscriptionConfig) -> dict[str, Any]:
    """Convert a Config object (Pydantic v2) into a serializable dict."""
    if hasattr(config, "model_dump"):
        return config.model_dump()
    if is_dataclass(config):
        return asdict(config)
    return dict(config.__dict__)


def _is_cuda_oom(exc: BaseException) -> bool:
    """Detect a CUDA out-of-memory error from its message (torch/CT2 style)."""
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda oom" in msg


def _clear_cuda_cache() -> None:
    """Free cached CUDA blocks between OOM retries (no-op without torch)."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _batch_size_plan(requested: int) -> list[int]:
    """OOM degradation ladder: requested, halved repeatedly, down to 1."""
    plan = [max(1, requested)]
    while plan[-1] > 1:
        plan.append(plan[-1] // 2)
    return plan


def _resolve_initial_prompt(config: TranscriptionConfig, source_media: Path | None) -> str | None:
    """Resolve initial prompt with per-file override if applicable."""
    if source_media is not None:
        return config.resolve_initial_prompt(source_media)
    return config.initial_prompt


def _warn_prompt_token_budget(prompt: str | None) -> None:
    """Warn when a prompt likely exceeds Whisper's ~224-token window.

    Whisper conditions on the TRAILING tokens, so the actionable advice is
    to put the most important terms at the END of the glossary.
    """
    if not prompt:
        return
    approx_tokens = max(1, len(prompt) // _APPROX_CHARS_PER_TOKEN)
    if approx_tokens > WHISPER_PROMPT_TOKEN_LIMIT:
        logger.warning(
            f"initial_prompt is ~{approx_tokens} tokens; Whisper only uses the "
            f"trailing ~{WHISPER_PROMPT_TOKEN_LIMIT}. Put the most important "
            f"terms at the END of the prompt."
        )


def _build_transcribe_kwargs(
    config: TranscriptionConfig,
    effective_prompt: str | None,
    *,
    word_timestamps: bool,
) -> dict[str, Any]:
    """Decoding kwargs shared by the sequential and batched APIs.

    Anti-hallucination controls are forwarded explicitly (defaults equal
    faster-whisper's, so default configs keep parity with pre-0.3 output).
    """
    temperature: Any = config.temperature
    if isinstance(temperature, tuple):
        temperature = list(temperature)
    return {
        "beam_size": config.beam_size,
        "language": config.language,
        "initial_prompt": effective_prompt,
        "vad_filter": config.use_vad,
        "vad_parameters": (
            {"min_silence_duration_ms": config.vad_min_silence_ms} if config.use_vad else None
        ),
        "word_timestamps": word_timestamps,
        "condition_on_previous_text": config.condition_on_previous_text,
        "temperature": temperature,
        "compression_ratio_threshold": config.compression_ratio_threshold,
        "log_prob_threshold": config.log_prob_threshold,
        "no_speech_threshold": config.no_speech_threshold,
        "hallucination_silence_threshold": config.hallucination_silence_threshold,
        "repetition_penalty": config.repetition_penalty,
        "no_repeat_ngram_size": config.no_repeat_ngram_size,
    }


def _launch_transcription(
    model: WhisperModel,
    audio_path: Path,
    kwargs: dict[str, Any],
    batch_size: int,
) -> tuple[Any, Any]:
    """Start sequential or batched decoding; both return (segments, info).

    `BatchedInferencePipeline.transcribe` is a drop-in for
    `WhisperModel.transcribe` (same lazy segment generator, same kwargs):
    https://github.com/SYSTRAN/faster-whisper#batched-transcription
    """
    if batch_size > 1:
        from faster_whisper import BatchedInferencePipeline

        pipe = BatchedInferencePipeline(model=model)
        return pipe.transcribe(str(audio_path), batch_size=batch_size, **kwargs)
    return model.transcribe(str(audio_path), **kwargs)


def _words_to_dicts(words: Any, offset_s: float = 0.0) -> list[dict[str, Any]] | None:
    """Normalize faster-whisper Word objects to plain dicts (+time offset)."""
    if not words:
        return None
    return [
        {
            "word": w.word,
            "start": round(w.start + offset_s, 3),
            "end": round(w.end + offset_s, 3),
            "probability": round(w.probability, 3),
        }
        for w in words
    ]


class _SegmentWriter:
    """Owns output handles, counters and per-speaker stats for one run.

    Single emission path for the streaming and chunked routes — guarantees
    SRT numbering, txt/jsonl content and `segments` metadata can never
    diverge between paths again (they did before 0.3: the streaming path
    incremented the SRT counter BEFORE the empty-text filter, producing
    non-consecutive SRT indices and an inflated `total_segments`).
    """

    def __init__(
        self,
        f_txt: IO[str],
        f_srt: IO[str],
        f_jsonl: IO[str] | None,
        *,
        has_diar: bool,
        include_words: bool,
        flush_every: int = 100,
    ) -> None:
        self._f_txt = f_txt
        self._f_srt = f_srt
        self._f_jsonl = f_jsonl
        self._has_diar = has_diar
        self._include_words = include_words
        self._flush_every = flush_every
        self.counter = 0
        self.total_words = 0
        self.n_empty = 0
        self.speaker_counts: dict[str, int] = defaultdict(int)
        self.segments_meta: list[dict] = []

    def note_empty(self) -> None:
        """Count a Whisper-emitted segment whose text was empty (discarded)."""
        self.n_empty += 1

    def emit(
        self,
        *,
        start: float,
        end: float,
        text: str,
        speaker: str | None,
        overlap: float,
        words: list[dict[str, Any]] | None,
    ) -> bool:
        """Write one display segment to txt/srt/json(l). Returns False if empty."""
        text = text.strip()
        if not text:
            return False
        self.counter += 1

        if self._has_diar and speaker is not None:
            self.speaker_counts[speaker] += 1
            label = f"[{speaker}] "
        else:
            label = ""

        self._f_txt.write(f"{label}{text}\n")
        self._f_srt.write(f"{self.counter}\n")
        self._f_srt.write(f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}\n")
        self._f_srt.write(f"{label}{text}\n\n")

        entry: dict[str, Any] = {
            "id": self.counter,
            "start": round(start, 3),
            "end": round(end, 3),
            "text": text,
        }
        if self._has_diar:
            entry["speaker"] = speaker
            entry["speaker_overlap_s"] = overlap
        if self._include_words and words:
            entry["words"] = words
        self.segments_meta.append(entry)
        self.total_words += len(text.split())

        if self._f_jsonl is not None:
            self._f_jsonl.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if self.counter % self._flush_every == 0:
            self.flush()
        return True

    def flush(self) -> None:
        """Flush all open output handles."""
        self._f_txt.flush()
        self._f_srt.flush()
        if self._f_jsonl is not None:
            self._f_jsonl.flush()


def _process_whisper_segment(
    writer: _SegmentWriter,
    *,
    start: float,
    end: float,
    text: str,
    words: list[dict[str, Any]] | None,
    has_diar: bool,
    diar_turns: list[dict] | None,
    word_mode: bool,
) -> None:
    """Route one Whisper segment to the writer (word-level split or whole).

    Word-level mode re-segments at speaker changes; when a segment carries
    no word timing (rare VAD edge), it falls back to segment-level
    attribution rather than dropping content.
    """
    if not text.strip():
        writer.note_empty()
        return

    if has_diar and word_mode and words:
        for piece in assign_speakers_by_words(words, diar_turns or []):
            writer.emit(
                start=piece["start"],
                end=piece["end"],
                text=piece["text"],
                speaker=piece["speaker"],
                overlap=piece["overlap"],
                words=piece["words"],
            )
        return

    if has_diar:
        speaker, overlap = assign_speaker_to_segment(start, end, diar_turns or [])
        writer.emit(start=start, end=end, text=text, speaker=speaker, overlap=overlap, words=words)
        return

    writer.emit(start=start, end=end, text=text, speaker=None, overlap=0.0, words=words)


def transcribe_streaming(
    model: WhisperModel,
    audio_path: Path,
    output_txt: Path,
    output_srt: Path,
    output_json: Path,
    config: TranscriptionConfig,
    diar_turns: list[dict] | None = None,
    *,
    source_media: Path | None = None,
    output_jsonl: Path | None = None,
) -> dict:
    """Transcribe a single audio file, writing each segment to disk as produced.

    Why this pattern:
        faster-whisper's transcribe() returns a lazy generator. On each step
        a segment (5-15 words) is yielded and written to disk IMMEDIATELY,
        so RAM stays below ~500 MB even for 4+ hour audio files.

    Resilience:
        On CUDA out-of-memory with `batch_size > 1`, the batch size is
        halved and the file restarted (output files are reopened, so a
        partial attempt never leaks into the final output), down to the
        exact sequential path at `batch_size=1`.

    Args:
        model: A WhisperModel instance already loaded by `load_whisper_model`.
        audio_path: 16 kHz mono WAV file.
        output_txt: Plain text output. Lines are prefixed with [SPEAKER_XX] when
            a diarization is supplied.
        output_srt: SubRip subtitle output.
        output_json: Structured metadata output (segments + run config +
            versions). Written atomically at the end.
        config: Pipeline configuration.
        diar_turns: List of pyannote turns, or None to skip speaker labeling.
        source_media: Original media file. Used to resolve per-file glossary
            (`<stem>.prompt.txt`). If None, only the global prompt is used.
        output_jsonl: Optional path to additionally stream segments as JSON Lines.

    Returns:
        Metadata dict with stats (language, duration, segments, words, RTF,
        speaker distribution, library versions, per-stage timings).

    Raises:
        RuntimeError: On unrecoverable decoding errors (including OOM that
            persists at batch_size=1).
    """
    plan = _batch_size_plan(config.batch_size)
    last_exc: BaseException | None = None
    for attempt_index, batch_size in enumerate(plan):
        try:
            return _transcribe_streaming_once(
                model,
                audio_path,
                output_txt,
                output_srt,
                output_json,
                config,
                diar_turns,
                source_media=source_media,
                output_jsonl=output_jsonl,
                batch_size=batch_size,
            )
        except RuntimeError as e:
            if _is_cuda_oom(e) and attempt_index < len(plan) - 1:
                next_bs = plan[attempt_index + 1]
                logger.warning(
                    f"CUDA OOM at batch_size={batch_size} — retrying with "
                    f"batch_size={next_bs} ({audio_path.name})"
                )
                _clear_cuda_cache()
                last_exc = e
                continue
            raise
    raise RuntimeError(f"Transcription failed after OOM retries: {last_exc}")  # pragma: no cover


def _transcribe_streaming_once(
    model: WhisperModel,
    audio_path: Path,
    output_txt: Path,
    output_srt: Path,
    output_json: Path,
    config: TranscriptionConfig,
    diar_turns: list[dict] | None,
    *,
    source_media: Path | None,
    output_jsonl: Path | None,
    batch_size: int,
) -> dict:
    """One full streaming pass at a fixed batch size (see `transcribe_streaming`)."""
    from speakerscribe import __version__

    logger.info(f"Transcribing: {audio_path.name} (batch_size={batch_size})")
    t0 = time.time()
    timings: dict[str, Any] = {}

    has_diar = diar_turns is not None and len(diar_turns) > 0
    if has_diar:
        n_unique_speakers = len({t["speaker"] for t in diar_turns})  # type: ignore[union-attr]
        logger.info(f"   Labeling with {n_unique_speakers} speakers")

    word_mode = has_diar and config.speaker_assignment == "word"
    word_timestamps = config.effective_word_timestamps(has_diar)

    effective_prompt = _resolve_initial_prompt(config, source_media)
    if effective_prompt and effective_prompt != config.initial_prompt:
        logger.info(f"   Using per-file prompt ({len(effective_prompt)} chars)")
    _warn_prompt_token_budget(effective_prompt)

    kwargs = _build_transcribe_kwargs(config, effective_prompt, word_timestamps=word_timestamps)

    # ── Launch transcription (lazy generator)
    t_launch = time.time()
    segments_iter, info = _launch_transcription(model, audio_path, kwargs, batch_size)
    timings["whisper_launch_s"] = round(time.time() - t_launch, 2)
    logger.info(f"   Language: {info.language} (prob {info.language_probability:.2%})")
    logger.info(f"   Duration: {info.duration:.1f}s = {info.duration / 60:.1f} min")

    output_txt.parent.mkdir(parents=True, exist_ok=True)
    f_jsonl = output_jsonl.open("w", encoding="utf-8") if output_jsonl else None
    try:
        with (
            output_txt.open("w", encoding="utf-8") as f_txt,
            output_srt.open("w", encoding="utf-8") as f_srt,
        ):
            writer = _SegmentWriter(
                f_txt,
                f_srt,
                f_jsonl,
                has_diar=has_diar,
                include_words=config.word_timestamps,
            )
            for segment in segments_iter:
                _process_whisper_segment(
                    writer,
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                    words=_words_to_dicts(getattr(segment, "words", None)),
                    has_diar=has_diar,
                    diar_turns=diar_turns,
                    word_mode=word_mode,
                )
                if writer.counter and writer.counter % 50 == 0:
                    progress = (segment.end / info.duration) * 100 if info.duration else 0.0
                    logger.info(
                        f"   {writer.counter} segments | {writer.total_words:,} words | "
                        f"{progress:.1f}%"
                    )
    finally:
        if f_jsonl is not None:
            f_jsonl.close()

    elapsed = time.time() - t0
    timings["transcribe_total_s"] = round(elapsed, 2)
    rtf = info.duration / elapsed if elapsed > 0 else 0

    metadata = _build_run_metadata(
        audio_file=audio_path.name,
        model=config.model,
        info=info,
        elapsed=elapsed,
        rtf=rtf,
        writer=writer,
        has_diar=has_diar,
        config=config,
        timings=timings,
        package_version=__version__,
        batch_size_requested=config.batch_size,
        batch_size_effective=batch_size,
        word_timestamps_effective=word_timestamps,
    )

    atomic_write_json(output_json, metadata)

    if has_diar:
        summary = " | ".join(f"{k}: {v}" for k, v in sorted(writer.speaker_counts.items()))
        logger.info(f"   Segment distribution: {summary}")
    logger.success(
        f"{writer.counter} segments | {writer.total_words:,} words | "
        f"{elapsed / 60:.1f} min | RTF={rtf:.1f}x"
    )
    return metadata


def transcribe_chunked(
    model: WhisperModel,
    chunks: list[AudioChunk],
    output_txt: Path,
    output_srt: Path,
    output_json: Path,
    config: TranscriptionConfig,
    diar_turns: list[dict] | None = None,
    *,
    source_media: Path | None = None,
    output_jsonl: Path | None = None,
) -> dict:
    """Transcribe a list of overlapping chunks and concatenate timestamps.

    .. deprecated:: 0.3
        The batched path (`transcribe_streaming` with `batch_size > 1`)
        handles long audio natively; external chunking risks truncating
        sentences at boundaries. Active only when the user sets
        `long_audio_threshold_min > 0`.

    Strategy:
        1. Run Whisper on each chunk independently.
        2. Adjust each segment's (and word's) timestamps by the chunk's
           `start_s` offset so they refer to the ORIGINAL audio timeline.
        3. Drop segments that BEGIN inside the trailing overlap region of
           every chunk except the last; those words are re-transcribed by
           the next chunk. Known limitation: a sentence crossing the hard
           cut at `chunk.end_s` is still truncated mid-word — raising
           `chunk_overlap_s` (default 30 s since 0.3) reduces but does not
           eliminate this. This is the core reason the path is deprecated.
        4. Speaker assignment uses the diarization turns of the FULL audio
           (computed once before chunking).

    Args:
        model: WhisperModel already loaded.
        chunks: Sequence of AudioChunk produced by `split_long_audio`.
        output_txt: Plain text output (concatenated).
        output_srt: SubRip output (concatenated, consecutive numbering).
        output_json: Structured metadata output. Written atomically.
        config: Pipeline configuration.
        diar_turns: Diarization turns for the FULL audio, or None.
        source_media: Original media file path (for per-file glossary lookup).
        output_jsonl: Optional path to stream concatenated segments as JSON Lines.

    Returns:
        Metadata dict with the same shape as `transcribe_streaming`.

    Raises:
        ValueError: If `chunks` is empty.
        RuntimeError: On unrecoverable decoding errors.
    """
    if not chunks:
        raise ValueError("chunks list is empty")

    plan = _batch_size_plan(config.batch_size)
    for attempt_index, batch_size in enumerate(plan):
        try:
            return _transcribe_chunked_once(
                model,
                chunks,
                output_txt,
                output_srt,
                output_json,
                config,
                diar_turns,
                source_media=source_media,
                output_jsonl=output_jsonl,
                batch_size=batch_size,
            )
        except RuntimeError as e:
            if _is_cuda_oom(e) and attempt_index < len(plan) - 1:
                next_bs = plan[attempt_index + 1]
                logger.warning(
                    f"CUDA OOM at batch_size={batch_size} — retrying chunked run "
                    f"with batch_size={next_bs}"
                )
                _clear_cuda_cache()
                continue
            raise
    raise RuntimeError("Chunked transcription failed after OOM retries")  # pragma: no cover


def _transcribe_chunked_once(
    model: WhisperModel,
    chunks: list[AudioChunk],
    output_txt: Path,
    output_srt: Path,
    output_json: Path,
    config: TranscriptionConfig,
    diar_turns: list[dict] | None,
    *,
    source_media: Path | None,
    output_jsonl: Path | None,
    batch_size: int,
) -> dict:
    """One full chunked pass at a fixed batch size (see `transcribe_chunked`)."""
    from speakerscribe import __version__

    logger.info(f"Chunked transcription: {len(chunks)} chunks (batch_size={batch_size})")
    t_total = time.time()
    timings: dict[str, Any] = {"chunks": []}

    has_diar = diar_turns is not None and len(diar_turns) > 0
    word_mode = has_diar and config.speaker_assignment == "word"
    word_timestamps = config.effective_word_timestamps(has_diar)

    effective_prompt = _resolve_initial_prompt(config, source_media)
    if effective_prompt and effective_prompt != config.initial_prompt:
        logger.info(f"   Using per-file prompt ({len(effective_prompt)} chars)")
    _warn_prompt_token_budget(effective_prompt)

    kwargs = _build_transcribe_kwargs(config, effective_prompt, word_timestamps=word_timestamps)

    last_info: Any = None
    total_audio_duration = chunks[-1].end_s

    output_txt.parent.mkdir(parents=True, exist_ok=True)
    f_jsonl = output_jsonl.open("w", encoding="utf-8") if output_jsonl else None
    try:
        with (
            output_txt.open("w", encoding="utf-8") as f_txt,
            output_srt.open("w", encoding="utf-8") as f_srt,
        ):
            writer = _SegmentWriter(
                f_txt,
                f_srt,
                f_jsonl,
                has_diar=has_diar,
                include_words=config.word_timestamps,
            )
            for chunk in chunks:
                t_chunk = time.time()
                logger.info(
                    f"   Chunk {chunk.index + 1}/{len(chunks)} "
                    f"({chunk.start_s / 60:.1f}-{chunk.end_s / 60:.1f} min)"
                )
                segments_iter, info = _launch_transcription(model, chunk.path, kwargs, batch_size)
                last_info = info

                # Trailing overlap cutoff (absolute timestamp in original audio).
                # For non-last chunks, drop any segment that starts >= this point;
                # those words will be transcribed by the next chunk.
                cutoff_abs: float | None = (
                    None if chunk.is_last else chunk.end_s - config.chunk_overlap_s
                )

                for segment in segments_iter:
                    abs_start = segment.start + chunk.start_s
                    abs_end = segment.end + chunk.start_s
                    if cutoff_abs is not None and abs_start >= cutoff_abs:
                        continue
                    _process_whisper_segment(
                        writer,
                        start=abs_start,
                        end=abs_end,
                        text=segment.text,
                        words=_words_to_dicts(
                            getattr(segment, "words", None), offset_s=chunk.start_s
                        ),
                        has_diar=has_diar,
                        diar_turns=diar_turns,
                        word_mode=word_mode,
                    )

                writer.flush()
                t_chunk_elapsed = round(time.time() - t_chunk, 2)
                timings["chunks"].append({"index": chunk.index, "elapsed_s": t_chunk_elapsed})
                logger.info(f"      done in {t_chunk_elapsed}s")
    finally:
        if f_jsonl is not None:
            f_jsonl.close()

    elapsed = time.time() - t_total
    timings["transcribe_total_s"] = round(elapsed, 2)
    rtf = total_audio_duration / elapsed if elapsed > 0 else 0

    metadata = _build_run_metadata(
        audio_file=source_media.name if source_media else chunks[0].path.name,
        model=config.model,
        info=None,
        elapsed=elapsed,
        rtf=rtf,
        writer=writer,
        has_diar=has_diar,
        config=config,
        timings=timings,
        package_version=__version__,
        detected_language=(last_info.language if last_info else (config.language or "unknown")),
        detected_lang_prob=(last_info.language_probability if last_info else 0.0),
        total_duration_s=total_audio_duration,
        chunked=True,
        n_chunks=len(chunks),
        batch_size_requested=config.batch_size,
        batch_size_effective=batch_size,
        word_timestamps_effective=word_timestamps,
    )

    atomic_write_json(output_json, metadata)

    if has_diar:
        summary = " | ".join(f"{k}: {v}" for k, v in sorted(writer.speaker_counts.items()))
        logger.info(f"   Segment distribution: {summary}")
    logger.success(
        f"{writer.counter} segments | {writer.total_words:,} words | {elapsed / 60:.1f} min | "
        f"RTF={rtf:.1f}x | chunks={len(chunks)}"
    )
    return metadata


def _build_run_metadata(
    *,
    audio_file: str,
    model: str,
    info: Any,
    elapsed: float,
    rtf: float,
    writer: _SegmentWriter,
    has_diar: bool,
    config: TranscriptionConfig,
    timings: dict[str, Any],
    package_version: str,
    detected_language: str | None = None,
    detected_lang_prob: float | None = None,
    total_duration_s: float | None = None,
    chunked: bool = False,
    n_chunks: int = 1,
    batch_size_requested: int = 1,
    batch_size_effective: int = 1,
    word_timestamps_effective: bool = False,
) -> dict[str, Any]:
    """Build the unified metadata dict written to .json."""
    try:
        import pyannote.audio

        pyannote_version = pyannote.audio.__version__
    except Exception:
        pyannote_version = "unknown"
    try:
        import faster_whisper

        fw_version = faster_whisper.__version__
    except Exception:
        fw_version = "unknown"

    if info is not None:
        lang = info.language
        lang_prob = round(info.language_probability, 4)
        duration_s = round(info.duration, 2)
    else:
        lang = detected_language or "unknown"
        lang_prob = round(detected_lang_prob or 0.0, 4)
        duration_s = round(total_duration_s or 0.0, 2)

    return {
        # Identity
        "audio_file": audio_file,
        "processed_at": datetime.now(tz=timezone.utc).isoformat(),
        # Versions
        "package_version": package_version,
        "pyannote_version": pyannote_version,
        "faster_whisper_version": fw_version,
        # ASR
        "model": model,
        "language_detected": lang,
        "language_probability": lang_prob,
        "duration_seconds": duration_s,
        "duration_minutes": round(duration_s / 60, 2),
        "elapsed_seconds": round(elapsed, 2),
        "real_time_factor": round(rtf, 2),
        "total_segments": len(writer.segments_meta),
        "total_words": writer.total_words,
        "empty_segments_discarded": writer.n_empty,
        # Decoding
        "batch_size_requested": batch_size_requested,
        "batch_size_effective": batch_size_effective,
        # Chunking info (legacy path)
        "chunked": chunked,
        "n_chunks": n_chunks,
        # Diarization
        "diarization_enabled": has_diar,
        "diarization_model": config.diarization_model if has_diar else None,
        "speakers_summary": dict(writer.speaker_counts) if has_diar else None,
        "speaker_assignment": config.speaker_assignment if has_diar else None,
        "word_timestamps": config.word_timestamps,
        "word_timestamps_effective": word_timestamps_effective,
        # Timings
        "timings": timings,
        # Full config
        "config": _config_to_dict(config),
        # Segments
        "segments": writer.segments_meta,
    }


__all__ = [
    "load_whisper_model",
    "loaded_whisper",
    "release_whisper_model",
    "transcribe_chunked",
    "transcribe_streaming",
]
