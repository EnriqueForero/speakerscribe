"""WAV extraction and long-audio chunking decisions."""

from __future__ import annotations

import time
from pathlib import Path

from speakerscribe.audio import extract_audio_wav, get_audio_duration_seconds
from speakerscribe.config import TranscriptionConfig, WorkspacePaths


def _maybe_extract_wav(
    media_path: Path,
    paths: WorkspacePaths,
    config: TranscriptionConfig,
    timings: dict[str, float],
) -> tuple[Path, Path | None]:
    """Extract WAV if configured, return (audio_for_model, wav_path_to_cleanup_or_None).

    When `config.extract_temp_wav=True`, runs ffmpeg to produce a 16 kHz mono
    PCM WAV and returns its path twice (once as the model input, once as a
    cleanup hint). When disabled, the original media is fed to Whisper directly.
    """
    if config.extract_temp_wav:
        t = time.time()
        wav_path = paths.audio_tmp / f"{media_path.stem}.wav"
        extract_audio_wav(media_path, wav_path, config.sample_rate)
        timings["extract_wav_s"] = round(time.time() - t, 2)
        return wav_path, wav_path
    return media_path, None


def _should_chunk_audio(audio_path: Path, config: TranscriptionConfig) -> tuple[bool, float]:
    """Decide whether to chunk based on duration vs threshold.

    Returns (should_chunk, total_duration_seconds). When chunking is disabled
    (`long_audio_threshold_min <= 0`), returns (False, 0.0) without probing
    the audio.
    """
    if config.long_audio_threshold_min <= 0:
        return False, 0.0
    duration_s = get_audio_duration_seconds(audio_path)
    threshold_s = config.long_audio_threshold_min * 60
    return duration_s > threshold_s, duration_s
