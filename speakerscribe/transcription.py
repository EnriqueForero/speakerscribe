"""Streaming transcription with faster-whisper + per-segment speaker labeling.

Pipeline core. Each Whisper segment is written to disk IMMEDIATELY, keeping
RAM under ~500 MB even for multi-hour audio.
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

from speakerscribe.audio import format_srt_timestamp
from speakerscribe.config import TranscriptionConfig
from speakerscribe.diarization import assign_speaker_to_segment
from speakerscribe.logging_config import logger

if TYPE_CHECKING:
    from faster_whisper import WhisperModel


def load_whisper_model(config: TranscriptionConfig) -> "WhisperModel":
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


def release_whisper_model(model: "WhisperModel | None" = None) -> None:
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


def transcribe_streaming(
    model: "WhisperModel",
    audio_path: Path,
    output_txt: Path,
    output_srt: Path,
    output_json: Path,
    config: TranscriptionConfig,
    diar_turns: list[dict] | None = None,
) -> dict:
    """Transcribe an audio file, writing each segment to disk as it is produced.

    Why this pattern:
        faster-whisper.transcribe() returns a generator. On each step:
            1. ~30 seconds of audio are processed on the GPU.
            2. A segment (5-15 words) is yielded.
            3. We write it to disk IMMEDIATELY.
            4. If a diarization is supplied, we assign a speaker by overlap.
        RAM stays below ~500 MB constantly, even for 4+ hour audio files.

    Supports `word_timestamps=True` for word-level alignment, useful when there
    is cross-talk inside a single Whisper segment.

    Args:
        model: A WhisperModel instance already loaded by `load_whisper_model`.
        audio_path: 16 kHz mono WAV file.
        output_txt: Plain text output. Lines are prefixed with [SPEAKER_XX] when
            a diarization is supplied.
        output_srt: SubRip subtitle output.
        output_json: Structured metadata output (segments + run config + versions).
        config: Pipeline configuration.
        diar_turns: List of pyannote turns, or None to skip speaker labeling.

    Returns:
        Metadata dict with stats (language, duration, segments, words, RTF,
        speaker distribution, library versions).
    """
    from speakerscribe import __version__

    logger.info(f"Transcribing: {audio_path.name}")
    t0 = time.time()

    has_diar = diar_turns is not None and len(diar_turns) > 0
    if has_diar:
        n_unique_speakers = len({t["speaker"] for t in diar_turns})  # type: ignore[union-attr]
        logger.info(f"   Labeling with {n_unique_speakers} speakers")

    # ── Launch transcription (generator)
    segments_iter, info = model.transcribe(
        str(audio_path),
        beam_size=config.beam_size,
        language=config.language,
        initial_prompt=config.initial_prompt,
        vad_filter=config.use_vad,
        vad_parameters=(
            {"min_silence_duration_ms": config.vad_min_silence_ms} if config.use_vad else None
        ),
        word_timestamps=config.word_timestamps,
    )
    logger.info(f"   Language: {info.language} (prob {info.language_probability:.2%})")
    logger.info(f"   Duration: {info.duration:.1f}s = {info.duration / 60:.1f} min")

    # ── Streaming write
    counter = 0
    total_words = 0
    speaker_counts: dict[str, int] = defaultdict(int)
    segments_meta: list[dict] = []

    output_txt.parent.mkdir(parents=True, exist_ok=True)

    with output_txt.open("w", encoding="utf-8") as f_txt, output_srt.open(
        "w", encoding="utf-8"
    ) as f_srt:
        for segment in segments_iter:
            counter += 1
            text = segment.text.strip()
            if not text:
                continue

            # Speaker assignment
            if has_diar:
                speaker, overlap = assign_speaker_to_segment(
                    segment.start, segment.end, diar_turns  # type: ignore[arg-type]
                )
                speaker_counts[speaker] += 1
                label = f"[{speaker}] "
            else:
                speaker, overlap = (None, 0.0)
                label = ""

            # TXT
            f_txt.write(f"{label}{text}\n")

            # SRT
            f_srt.write(f"{counter}\n")
            f_srt.write(
                f"{format_srt_timestamp(segment.start)} --> "
                f"{format_srt_timestamp(segment.end)}\n"
            )
            f_srt.write(f"{label}{text}\n\n")

            # Metadata
            entry: dict[str, Any] = {
                "id": counter,
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "text": text,
            }
            if has_diar:
                entry["speaker"] = speaker
                entry["speaker_overlap_s"] = overlap

            # Word-level timestamps if enabled
            if config.word_timestamps and segment.words:
                entry["words"] = [
                    {
                        "word": w.word,
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "probability": round(w.probability, 3),
                    }
                    for w in segment.words
                ]

            segments_meta.append(entry)
            total_words += len(text.split())

            if counter % 50 == 0:
                progress = (segment.end / info.duration) * 100
                logger.info(
                    f"   {counter} segments | {total_words:,} words | "
                    f"{progress:.1f}%"
                )

            if counter % 100 == 0:
                f_txt.flush()
                f_srt.flush()

    elapsed = time.time() - t0
    rtf = info.duration / elapsed if elapsed > 0 else 0

    # Library versions for reproducibility
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

    metadata = {
        # Identity
        "audio_file": audio_path.name,
        "processed_at": datetime.now(tz=timezone.utc).isoformat(),
        # Versions
        "package_version": __version__,
        "pyannote_version": pyannote_version,
        "faster_whisper_version": fw_version,
        # ASR
        "model": config.model,
        "language_detected": info.language,
        "language_probability": round(info.language_probability, 4),
        "duration_seconds": round(info.duration, 2),
        "duration_minutes": round(info.duration / 60, 2),
        "elapsed_seconds": round(elapsed, 2),
        "real_time_factor": round(rtf, 2),
        "total_segments": counter,
        "total_words": total_words,
        # Diarization
        "diarization_enabled": has_diar,
        "diarization_model": config.diarization_model if has_diar else None,
        "speakers_summary": dict(speaker_counts) if has_diar else None,
        "word_timestamps": config.word_timestamps,
        # Full config
        "config": _config_to_dict(config),
        # Segments
        "segments": segments_meta,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    if has_diar:
        summary = " | ".join(f"{k}: {v}" for k, v in sorted(speaker_counts.items()))
        logger.info(f"   Segment distribution: {summary}")
    logger.success(
        f"{counter} segments | {total_words:,} words | "
        f"{elapsed / 60:.1f} min | RTF={rtf:.1f}x"
    )
    return metadata


__all__ = [
    "load_whisper_model",
    "release_whisper_model",
    "transcribe_streaming",
]
