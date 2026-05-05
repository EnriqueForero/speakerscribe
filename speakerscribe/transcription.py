"""Streaming transcription with faster-whisper + per-segment speaker labeling.

Pipeline core. Each Whisper segment is written to disk IMMEDIATELY, keeping
RAM under ~500 MB even for multi-hour audio.

Two transcription paths:
    transcribe_streaming  — Single audio file (short or moderate duration).
    transcribe_chunked    — List of overlapping chunks for very long audios.
                            Drops segments that fall in the trailing overlap
                            of every chunk except the last to avoid duplicates.
"""

from __future__ import annotations

import gc
import json
import time
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from speakerscribe.audio import AudioChunk, format_srt_timestamp
from speakerscribe.config import TranscriptionConfig
from speakerscribe.diarization import assign_speaker_to_segment
from speakerscribe.logging_config import logger

if TYPE_CHECKING:
    from faster_whisper import WhisperModel


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
    """Release a Whisper model and clear the CUDA cache.

    Args:
        model: Instance to release. If None, only clears the CUDA cache.
    """
    import torch

    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    logger.debug("Whisper model released, CUDA cache cleared")


def _config_to_dict(config: TranscriptionConfig) -> dict[str, Any]:
    """Convert a Config object (Pydantic v2) into a serializable dict."""
    if hasattr(config, "model_dump"):
        return config.model_dump()
    if is_dataclass(config):
        return asdict(config)
    return dict(config.__dict__)


def _build_segment_entry(
    counter: int,
    start: float,
    end: float,
    text: str,
    has_diar: bool,
    speaker: str | None,
    overlap: float,
    word_timestamps: bool,
    words: list | None,
) -> dict[str, Any]:
    """Build the metadata dict for a single segment."""
    entry: dict[str, Any] = {
        "id": counter,
        "start": round(start, 3),
        "end": round(end, 3),
        "text": text,
    }
    if has_diar:
        entry["speaker"] = speaker
        entry["speaker_overlap_s"] = overlap
    if word_timestamps and words:
        entry["words"] = [
            {
                "word": w.word,
                "start": round(w.start, 3),
                "end": round(w.end, 3),
                "probability": round(w.probability, 3),
            }
            for w in words
        ]
    return entry


def _resolve_initial_prompt(config: TranscriptionConfig, source_media: Path | None) -> str | None:
    """Resolve initial prompt with per-file override if applicable."""
    if source_media is not None:
        return config.resolve_initial_prompt(source_media)
    return config.initial_prompt


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
        faster-whisper.transcribe() returns a generator. On each step:
            1. ~30 seconds of audio are processed on the GPU.
            2. A segment (5-15 words) is yielded.
            3. We write it to disk IMMEDIATELY.
            4. If a diarization is supplied, we assign a speaker by overlap.
        RAM stays below ~500 MB constantly, even for 4+ hour audio files.

    Args:
        model: A WhisperModel instance already loaded by `load_whisper_model`.
        audio_path: 16 kHz mono WAV file.
        output_txt: Plain text output. Lines are prefixed with [SPEAKER_XX] when
            a diarization is supplied.
        output_srt: SubRip subtitle output.
        output_json: Structured metadata output (segments + run config + versions).
        config: Pipeline configuration.
        diar_turns: List of pyannote turns, or None to skip speaker labeling.
        source_media: Original media file. Used to resolve per-file glossary
            (`<stem>.prompt.txt`). If None, only the global prompt is used.
        output_jsonl: Optional path to additionally stream segments as JSON Lines.

    Returns:
        Metadata dict with stats (language, duration, segments, words, RTF,
        speaker distribution, library versions, per-stage timings).
    """
    from speakerscribe import __version__

    logger.info(f"Transcribing: {audio_path.name}")
    t0 = time.time()
    timings: dict[str, float] = {}

    has_diar = diar_turns is not None and len(diar_turns) > 0
    if has_diar:
        n_unique_speakers = len({t["speaker"] for t in diar_turns})  # type: ignore[union-attr]
        logger.info(f"   Labeling with {n_unique_speakers} speakers")

    effective_prompt = _resolve_initial_prompt(config, source_media)
    if effective_prompt and effective_prompt != config.initial_prompt:
        logger.info(f"   Using per-file prompt ({len(effective_prompt)} chars)")

    # ── Launch transcription (generator)
    t_launch = time.time()
    segments_iter, info = model.transcribe(
        str(audio_path),
        beam_size=config.beam_size,
        language=config.language,
        initial_prompt=effective_prompt,
        vad_filter=config.use_vad,
        vad_parameters=(
            {"min_silence_duration_ms": config.vad_min_silence_ms} if config.use_vad else None
        ),
        word_timestamps=config.word_timestamps,
    )
    timings["whisper_launch_s"] = round(time.time() - t_launch, 2)
    logger.info(f"   Language: {info.language} (prob {info.language_probability:.2%})")
    logger.info(f"   Duration: {info.duration:.1f}s = {info.duration / 60:.1f} min")

    # ── Streaming write
    counter = 0
    total_words = 0
    speaker_counts: dict[str, int] = defaultdict(int)
    segments_meta: list[dict] = []

    output_txt.parent.mkdir(parents=True, exist_ok=True)

    f_jsonl = output_jsonl.open("w", encoding="utf-8") if output_jsonl else None
    try:
        with (
            output_txt.open("w", encoding="utf-8") as f_txt,
            output_srt.open("w", encoding="utf-8") as f_srt,
        ):
            for segment in segments_iter:
                counter += 1
                text = segment.text.strip()
                if not text:
                    continue

                if has_diar:
                    speaker, overlap = assign_speaker_to_segment(
                        segment.start,
                        segment.end,
                        diar_turns,  # type: ignore[arg-type]
                    )
                    speaker_counts[speaker] += 1
                    label = f"[{speaker}] "
                else:
                    speaker, overlap = (None, 0.0)
                    label = ""

                f_txt.write(f"{label}{text}\n")

                f_srt.write(f"{counter}\n")
                f_srt.write(
                    f"{format_srt_timestamp(segment.start)} --> "
                    f"{format_srt_timestamp(segment.end)}\n"
                )
                f_srt.write(f"{label}{text}\n\n")

                entry = _build_segment_entry(
                    counter=counter,
                    start=segment.start,
                    end=segment.end,
                    text=text,
                    has_diar=has_diar,
                    speaker=speaker,
                    overlap=overlap,
                    word_timestamps=config.word_timestamps,
                    words=segment.words,
                )
                segments_meta.append(entry)
                total_words += len(text.split())

                if f_jsonl is not None:
                    f_jsonl.write(json.dumps(entry, ensure_ascii=False) + "\n")

                if counter % 50 == 0:
                    progress = (segment.end / info.duration) * 100
                    logger.info(f"   {counter} segments | {total_words:,} words | {progress:.1f}%")

                if counter % 100 == 0:
                    f_txt.flush()
                    f_srt.flush()
                    if f_jsonl is not None:
                        f_jsonl.flush()
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
        counter=counter,
        total_words=total_words,
        has_diar=has_diar,
        speaker_counts=speaker_counts,
        config=config,
        segments_meta=segments_meta,
        timings=timings,
        package_version=__version__,
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    if has_diar:
        summary = " | ".join(f"{k}: {v}" for k, v in sorted(speaker_counts.items()))
        logger.info(f"   Segment distribution: {summary}")
    logger.success(
        f"{counter} segments | {total_words:,} words | {elapsed / 60:.1f} min | RTF={rtf:.1f}x"
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

    Strategy:
        1. Run Whisper on each chunk independently.
        2. Adjust each segment's timestamps by the chunk's `start_s` offset
           so they refer to positions in the ORIGINAL audio.
        3. Drop segments that begin inside the trailing overlap region of
           every chunk except the last (avoiding duplicates with no chance
           of cutting words mid-segment, since Whisper's VAD ensures
           segments do not span chunk boundaries).
        4. Speaker assignment uses the diarization turns of the FULL audio
           (computed once before chunking).

    Args:
        model: WhisperModel already loaded.
        chunks: Sequence of AudioChunk produced by `split_long_audio`.
        output_txt, output_srt, output_json: Output paths (concatenated).
        config: Pipeline configuration.
        diar_turns: Diarization turns for the FULL audio, or None.
        source_media: Original media file path (for per-file glossary lookup).
        output_jsonl: Optional path to stream concatenated segments as JSON Lines.

    Returns:
        Metadata dict with the same shape as `transcribe_streaming`.
    """
    from faster_whisper.transcribe import TranscriptionInfo

    from speakerscribe import __version__

    if not chunks:
        raise ValueError("chunks list is empty")

    logger.info(f"Chunked transcription: {len(chunks)} chunks")
    t_total = time.time()
    timings: dict[str, float] = {"chunks": []}

    has_diar = diar_turns is not None and len(diar_turns) > 0
    effective_prompt = _resolve_initial_prompt(config, source_media)
    if effective_prompt and effective_prompt != config.initial_prompt:
        logger.info(f"   Using per-file prompt ({len(effective_prompt)} chars)")

    counter = 0
    total_words = 0
    speaker_counts: dict[str, int] = defaultdict(int)
    segments_meta: list[dict] = []
    last_info: TranscriptionInfo | None = None
    total_audio_duration = chunks[-1].end_s

    output_txt.parent.mkdir(parents=True, exist_ok=True)

    f_jsonl = output_jsonl.open("w", encoding="utf-8") if output_jsonl else None
    try:
        with (
            output_txt.open("w", encoding="utf-8") as f_txt,
            output_srt.open("w", encoding="utf-8") as f_srt,
        ):
            for chunk in chunks:
                t_chunk = time.time()
                logger.info(
                    f"   Chunk {chunk.index + 1}/{len(chunks)} "
                    f"({chunk.start_s / 60:.1f}-{chunk.end_s / 60:.1f} min)"
                )
                segments_iter, info = model.transcribe(
                    str(chunk.path),
                    beam_size=config.beam_size,
                    language=config.language,
                    initial_prompt=effective_prompt,
                    vad_filter=config.use_vad,
                    vad_parameters=(
                        {"min_silence_duration_ms": config.vad_min_silence_ms}
                        if config.use_vad
                        else None
                    ),
                    word_timestamps=config.word_timestamps,
                )
                last_info = info

                # Trailing overlap cutoff (absolute timestamp in original audio).
                # For non-last chunks, drop any segment that starts >= this point;
                # those words will be transcribed by the next chunk.
                if chunk.is_last:
                    cutoff_abs: float | None = None
                else:
                    cutoff_abs = chunk.end_s - config.chunk_overlap_s

                for segment in segments_iter:
                    text = segment.text.strip()
                    if not text:
                        continue

                    abs_start = segment.start + chunk.start_s
                    abs_end = segment.end + chunk.start_s

                    # Drop segments inside the trailing overlap (will be in next chunk)
                    if cutoff_abs is not None and abs_start >= cutoff_abs:
                        continue

                    counter += 1

                    if has_diar:
                        speaker, overlap = assign_speaker_to_segment(
                            abs_start,
                            abs_end,
                            diar_turns,  # type: ignore[arg-type]
                        )
                        speaker_counts[speaker] += 1
                        label = f"[{speaker}] "
                    else:
                        speaker, overlap = (None, 0.0)
                        label = ""

                    f_txt.write(f"{label}{text}\n")
                    f_srt.write(f"{counter}\n")
                    f_srt.write(
                        f"{format_srt_timestamp(abs_start)} --> {format_srt_timestamp(abs_end)}\n"
                    )
                    f_srt.write(f"{label}{text}\n\n")

                    entry = _build_segment_entry(
                        counter=counter,
                        start=abs_start,
                        end=abs_end,
                        text=text,
                        has_diar=has_diar,
                        speaker=speaker,
                        overlap=overlap,
                        word_timestamps=config.word_timestamps,
                        words=None,  # word offsets would need adjustment; omit for chunked path
                    )
                    segments_meta.append(entry)
                    total_words += len(text.split())

                    if f_jsonl is not None:
                        f_jsonl.write(json.dumps(entry, ensure_ascii=False) + "\n")

                f_txt.flush()
                f_srt.flush()
                if f_jsonl is not None:
                    f_jsonl.flush()
                t_chunk_elapsed = round(time.time() - t_chunk, 2)
                timings["chunks"].append({"index": chunk.index, "elapsed_s": t_chunk_elapsed})
                logger.info(f"      ✅ {t_chunk_elapsed}s")
    finally:
        if f_jsonl is not None:
            f_jsonl.close()

    elapsed = time.time() - t_total
    timings["transcribe_total_s"] = round(elapsed, 2)
    rtf = total_audio_duration / elapsed if elapsed > 0 else 0

    detected_lang = last_info.language if last_info else (config.language or "unknown")
    detected_prob = last_info.language_probability if last_info else 0.0

    metadata = _build_run_metadata(
        audio_file=source_media.name if source_media else chunks[0].path.name,
        model=config.model,
        info=None,
        elapsed=elapsed,
        rtf=rtf,
        counter=counter,
        total_words=total_words,
        has_diar=has_diar,
        speaker_counts=speaker_counts,
        config=config,
        segments_meta=segments_meta,
        timings=timings,
        package_version=__version__,
        detected_language=detected_lang,
        detected_lang_prob=detected_prob,
        total_duration_s=total_audio_duration,
        chunked=True,
        n_chunks=len(chunks),
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    if has_diar:
        summary = " | ".join(f"{k}: {v}" for k, v in sorted(speaker_counts.items()))
        logger.info(f"   Segment distribution: {summary}")
    logger.success(
        f"{counter} segments | {total_words:,} words | {elapsed / 60:.1f} min | "
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
    counter: int,
    total_words: int,
    has_diar: bool,
    speaker_counts: dict[str, int],
    config: TranscriptionConfig,
    segments_meta: list[dict],
    timings: dict[str, Any],
    package_version: str,
    detected_language: str | None = None,
    detected_lang_prob: float | None = None,
    total_duration_s: float | None = None,
    chunked: bool = False,
    n_chunks: int = 1,
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
        "total_segments": counter,
        "total_words": total_words,
        # Chunking info
        "chunked": chunked,
        "n_chunks": n_chunks,
        # Diarization
        "diarization_enabled": has_diar,
        "diarization_model": config.diarization_model if has_diar else None,
        "speakers_summary": dict(speaker_counts) if has_diar else None,
        "word_timestamps": config.word_timestamps,
        # Timings
        "timings": timings,
        # Full config
        "config": _config_to_dict(config),
        # Segments
        "segments": segments_meta,
    }


__all__ = [
    "load_whisper_model",
    "release_whisper_model",
    "transcribe_chunked",
    "transcribe_streaming",
]
