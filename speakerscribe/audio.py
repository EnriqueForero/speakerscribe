"""Audio extraction with ffmpeg + duration probe + long-audio splitting + helpers.

Public API:
    extract_audio_wav        — Extract 16 kHz mono WAV from any media file.
    get_audio_duration_seconds — Probe duration via ffprobe (no decoding).
    split_long_audio         — Split a long WAV into overlapping chunks.
    calculate_file_hash      — SHA-256 streaming hash (low memory).
    format_srt_timestamp     — Seconds to SRT 'HH:MM:SS,mmm' format.
    format_hms               — Seconds to 'HH:MM:SS' for human reading.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from speakerscribe.logging_config import logger


@dataclass(frozen=True)
class AudioChunk:
    """A chunk of a longer audio file produced by `split_long_audio`.

    Attributes:
        path: Path to the chunk WAV file.
        index: Zero-based chunk index.
        start_s: Start time of the chunk in the ORIGINAL audio (seconds).
        end_s: End time of the chunk in the ORIGINAL audio (seconds).
        is_last: True when this is the final chunk (no trailing overlap to discard).
    """

    path: Path
    index: int
    start_s: float
    end_s: float
    is_last: bool

    @property
    def duration_s(self) -> float:
        """Chunk duration in seconds."""
        return self.end_s - self.start_s


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


def get_audio_duration_seconds(path: Path) -> float:
    """Probe a media file's duration in seconds using ffprobe.

    No decoding, no full-file read — uses the container metadata. Works on
    any format supported by ffmpeg (WAV, MP3, MP4, MKV, M4A, FLAC, ...).

    Args:
        path: Path to the media file.

    Returns:
        Duration in seconds. Returns 0.0 if the file has no measurable duration.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If ffprobe is not installed or fails.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe not found in PATH. Install ffmpeg (which bundles ffprobe).")

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {res.stderr.strip()}")
    try:
        data = json.loads(res.stdout)
        duration = float(data.get("format", {}).get("duration", 0.0))
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        raise RuntimeError(f"Could not parse ffprobe output: {e}") from e
    return duration


def split_long_audio(
    input_wav: Path,
    output_dir: Path,
    chunk_duration_s: int = 1800,
    overlap_s: int = 5,
    sample_rate: int = 16_000,
) -> list[AudioChunk]:
    """Split a long WAV file into overlapping fixed-duration chunks.

    Strategy:
        Chunk N covers [N*step, N*step + chunk_duration_s] where
        step = chunk_duration_s - overlap_s. Consecutive chunks share `overlap_s`
        seconds. Whisper transcribes each chunk independently; the consumer is
        expected to drop segments that fall in the trailing overlap of every
        chunk except the last (handled by `is_last`). This guarantees no word
        is split across chunk boundaries while avoiding duplicate output.

    Diarization is NOT chunked: it must run on the full original audio
    (pyannote.audio handles long audio with its own sliding window).

    Idempotent: if the chunks already exist with the expected file sizes,
    they are reused without re-running ffmpeg.

    Args:
        input_wav: Source WAV file (typically already extracted to 16 kHz mono).
        output_dir: Directory where chunk WAVs are written.
        chunk_duration_s: Target chunk duration in seconds. Default 1800 (30 min).
        overlap_s: Overlap between consecutive chunks in seconds. Default 5.
        sample_rate: Sampling rate of the chunk WAVs. Default 16000.

    Returns:
        List of `AudioChunk` instances sorted by chunk index. If the audio is
        shorter than `chunk_duration_s`, returns a single chunk pointing to
        the original file (no splitting performed).

    Raises:
        FileNotFoundError: If the input WAV does not exist.
        RuntimeError: If ffmpeg fails for any chunk.
        ValueError: If `chunk_duration_s <= overlap_s` (would cause infinite loop).
    """
    if not input_wav.exists():
        raise FileNotFoundError(f"Input WAV not found: {input_wav}")
    if chunk_duration_s <= overlap_s:
        raise ValueError(
            f"chunk_duration_s ({chunk_duration_s}) must be greater than overlap_s ({overlap_s})."
        )

    duration = get_audio_duration_seconds(input_wav)
    if duration <= chunk_duration_s:
        logger.info(
            f"Audio duration ({duration / 60:.1f} min) <= chunk size "
            f"({chunk_duration_s / 60:.1f} min); no splitting needed."
        )
        return [AudioChunk(path=input_wav, index=0, start_s=0.0, end_s=duration, is_last=True)]

    output_dir.mkdir(parents=True, exist_ok=True)
    step = chunk_duration_s - overlap_s
    starts: list[float] = []
    s = 0.0
    while s < duration:
        starts.append(s)
        s += step

    chunks: list[AudioChunk] = []
    logger.info(
        f"Splitting {duration / 60:.1f} min into {len(starts)} chunks of "
        f"{chunk_duration_s / 60:.1f} min (overlap {overlap_s}s)"
    )
    t0 = time.time()
    for i, start_s in enumerate(starts):
        end_s = min(start_s + chunk_duration_s, duration)
        is_last = i == len(starts) - 1
        chunk_path = output_dir / f"{input_wav.stem}_chunk{i:03d}.wav"

        # Skip if already produced and non-empty
        if chunk_path.exists() and chunk_path.stat().st_size > 1024:
            logger.debug(f"Reusing existing chunk: {chunk_path.name}")
            chunks.append(
                AudioChunk(path=chunk_path, index=i, start_s=start_s, end_s=end_s, is_last=is_last)
            )
            continue

        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{start_s:.3f}",
            "-i",
            str(input_wav),
            "-t",
            f"{end_s - start_s:.3f}",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(chunk_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed splitting chunk {i} (exit {res.returncode}):\n{res.stderr}"
            )
        chunks.append(
            AudioChunk(path=chunk_path, index=i, start_s=start_s, end_s=end_s, is_last=is_last)
        )

    logger.success(f"{len(chunks)} chunks produced in {time.time() - t0:.1f}s")
    return chunks


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
    "AudioChunk",
    "calculate_file_hash",
    "extract_audio_wav",
    "format_hms",
    "format_srt_timestamp",
    "get_audio_duration_seconds",
    "split_long_audio",
]
