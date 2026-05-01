"""Audio extraction with ffmpeg + time formatting helpers + file hashing."""

from __future__ import annotations

import hashlib
import subprocess
import time
from datetime import timedelta
from pathlib import Path

from speakerscribe.logging_config import logger


def extract_audio_wav(
    input_file: Path,
    output_file: Path,
    sample_rate: int = 16_000,
) -> Path:
    """Extract a 16 kHz mono WAV from an MP4/MKV/MP3/M4A/etc input file.

    Why pre-extract:
        1. faster-whisper AND pyannote both read the SAME WAV -> zero re-decoding.
        2. Guarantees 16 kHz mono, the format both models expect.
        3. 16 kHz mono PCM ~= 115 MB per hour of audio.

    Idempotent: if the output WAV already exists, the function returns its path
    without re-running ffmpeg.

    Args:
        input_file: Path to the source media file (MP4, MP3, MKV, M4A, WAV, etc).
        output_file: Destination WAV path.
        sample_rate: Sampling rate in Hz. 16000 is the standard for both
            Whisper and pyannote.

    Returns:
        Path to the output WAV.

    Raises:
        FileNotFoundError: If the input file does not exist.
        RuntimeError: If ffmpeg returns a non-zero exit code.
    """
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if output_file.exists():
        logger.debug(f"WAV already exists, reusing: {output_file.name}")
        return output_file

    output_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(input_file),
        "-ac",
        "1",  # mono
        "-ar",
        str(sample_rate),  # sampling rate
        "-vn",  # no video
        "-c:a",
        "pcm_s16le",  # 16-bit PCM
        str(output_file),
    ]
    logger.info(f"Extracting audio: {input_file.name} -> {output_file.name}")
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (exit {res.returncode}):\n{res.stderr}")

    size_mb = output_file.stat().st_size / 1e6
    logger.success(f"Audio extracted in {time.time() - t0:.1f}s ({size_mb:.1f} MB)")
    return output_file


def calculate_file_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute the SHA-256 of a file by streaming chunks (low memory footprint).

    Useful for content-based idempotency: if a file is renamed but its content
    is identical, the hash will detect that it was already processed.

    Args:
        path: File to hash.
        chunk_size: Read block size in bytes (default 1 MB).

    Returns:
        Hexadecimal SHA-256 digest.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def format_srt_timestamp(seconds: float) -> str:
    """Convert seconds to the SRT format `HH:MM:SS,mmm`.

    Args:
        seconds: Time in seconds. Negative values are clamped to 0.

    Returns:
        SRT-formatted timestamp string.

    Examples:
        >>> format_srt_timestamp(125.789)
        '00:02:05,789'
        >>> format_srt_timestamp(0)
        '00:00:00,000'
        >>> format_srt_timestamp(-5.0)
        '00:00:00,000'
    """
    td = timedelta(seconds=max(0.0, seconds))
    total_ms = int(td.total_seconds() * 1000)
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_hms(seconds: float) -> str:
    """Convert seconds to `HH:MM:SS` (no milliseconds) for human-readable output.

    Args:
        seconds: Time in seconds. Negative values are clamped to 0.

    Returns:
        `HH:MM:SS` string.

    Examples:
        >>> format_hms(3661)
        '01:01:01'
        >>> format_hms(0)
        '00:00:00'
    """
    td = timedelta(seconds=max(0.0, seconds))
    total_s = int(td.total_seconds())
    h, rem = divmod(total_s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


__all__ = [
    "calculate_file_hash",
    "extract_audio_wav",
    "format_hms",
    "format_srt_timestamp",
]
