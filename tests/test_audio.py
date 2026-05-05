"""Tests for speakerscribe.audio (v0.2.0)."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from speakerscribe.audio import (
    AudioChunk,
    calculate_file_hash,
    format_hms,
    format_srt_timestamp,
    get_audio_duration_seconds,
    split_long_audio,
)


class TestFormatSrtTimestamp:
    def test_normal_case(self):
        assert format_srt_timestamp(125.789) == "00:02:05,789"

    def test_zero(self):
        assert format_srt_timestamp(0) == "00:00:00,000"

    def test_negative_clamped_to_zero(self):
        assert format_srt_timestamp(-5.0) == "00:00:00,000"

    def test_hours(self):
        assert format_srt_timestamp(3661.5) == "01:01:01,500"

    @pytest.mark.parametrize(
        "seconds,expected",
        [
            (0.001, "00:00:00,001"),
            (59.999, "00:00:59,999"),
            (60.0, "00:01:00,000"),
            (3600, "01:00:00,000"),
        ],
    )
    def test_parametrized(self, seconds, expected):
        assert format_srt_timestamp(seconds) == expected


class TestFormatHms:
    def test_normal_case(self):
        assert format_hms(3661) == "01:01:01"

    def test_zero(self):
        assert format_hms(0) == "00:00:00"

    def test_negative_clamped_to_zero(self):
        assert format_hms(-100) == "00:00:00"

    @pytest.mark.parametrize(
        "seconds,expected",
        [(60, "00:01:00"), (3600, "01:00:00"), (86399, "23:59:59")],
    )
    def test_parametrized(self, seconds, expected):
        assert format_hms(seconds) == expected


class TestCalculateFileHash:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            calculate_file_hash(tmp_path / "does-not-exist.txt")

    def test_hash_is_consistent(self, tmp_path):
        file = tmp_path / "test.txt"
        file.write_text("hello world")
        h1 = calculate_file_hash(file)
        h2 = calculate_file_hash(file)
        assert h1 == h2
        assert len(h1) == 64

    def test_different_contents_different_hashes(self, tmp_path):
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("content A")
        b.write_text("content B")
        assert calculate_file_hash(a) != calculate_file_hash(b)

    def test_large_file_streamed_correctly(self, tmp_path):
        large_file = tmp_path / "large.bin"
        large_file.write_bytes(b"x" * (3 * 1024 * 1024))
        h = calculate_file_hash(large_file, chunk_size=1024)
        assert len(h) == 64


class TestAudioChunk:
    def test_duration_property(self):
        c = AudioChunk(
            path=Path("/tmp/x.wav"),
            index=0,
            start_s=10.0,
            end_s=1810.0,
            is_last=False,
        )
        assert c.duration_s == 1800.0

    def test_immutable(self):
        c = AudioChunk(path=Path("/tmp/x.wav"), index=0, start_s=0.0, end_s=10.0, is_last=True)
        with pytest.raises(FrozenInstanceError):
            c.index = 5  # type: ignore[misc]


# ─── ffmpeg-dependent tests (skip if ffmpeg/ffprobe missing) ────────
ffmpeg_available = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _make_silence_wav(path: Path, duration_s: float, sample_rate: int = 16_000) -> None:
    """Generate a silent WAV using ffmpeg's anullsrc filter."""
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"anullsrc=r={sample_rate}:cl=mono",
        "-t",
        f"{duration_s}",
        "-c:a",
        "pcm_s16le",
        str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


@pytest.mark.skipif(not ffmpeg_available, reason="ffmpeg/ffprobe not installed")
class TestGetAudioDurationSeconds:
    def test_returns_correct_duration(self, tmp_path):
        wav = tmp_path / "silence.wav"
        _make_silence_wav(wav, duration_s=2.5)
        assert abs(get_audio_duration_seconds(wav) - 2.5) < 0.05

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            get_audio_duration_seconds(tmp_path / "no.wav")


@pytest.mark.skipif(not ffmpeg_available, reason="ffmpeg/ffprobe not installed")
class TestSplitLongAudio:
    def test_short_audio_returns_single_chunk_no_split(self, tmp_path):
        """Audio <= chunk_duration_s returns the original as a single chunk."""
        wav = tmp_path / "short.wav"
        _make_silence_wav(wav, duration_s=3.0)
        out_dir = tmp_path / "chunks"
        chunks = split_long_audio(wav, out_dir, chunk_duration_s=10, overlap_s=1)
        assert len(chunks) == 1
        assert chunks[0].path == wav  # original file reused
        assert chunks[0].is_last is True
        assert chunks[0].index == 0

    def test_long_audio_split_into_chunks_with_overlap(self, tmp_path):
        """Audio of 20s split into chunks of 10s with 2s overlap -> 3 chunks."""
        wav = tmp_path / "long.wav"
        _make_silence_wav(wav, duration_s=20.0)
        out_dir = tmp_path / "chunks"
        chunks = split_long_audio(wav, out_dir, chunk_duration_s=10, overlap_s=2)
        assert len(chunks) >= 2
        assert chunks[0].start_s == 0.0
        assert chunks[0].is_last is False
        assert chunks[-1].is_last is True
        # Step = chunk - overlap = 8s, so chunk1 starts at 8s
        assert abs(chunks[1].start_s - 8.0) < 0.01
        # Each chunk file exists
        for c in chunks:
            assert c.path.exists()

    def test_idempotent_reuses_existing_chunks(self, tmp_path):
        """Running split twice does not regenerate chunks."""
        wav = tmp_path / "long.wav"
        _make_silence_wav(wav, duration_s=15.0)
        out_dir = tmp_path / "chunks"
        chunks1 = split_long_audio(wav, out_dir, chunk_duration_s=8, overlap_s=1)
        # Capture mtimes
        mtimes_before = {c.path: c.path.stat().st_mtime_ns for c in chunks1}
        chunks2 = split_long_audio(wav, out_dir, chunk_duration_s=8, overlap_s=1)
        mtimes_after = {c.path: c.path.stat().st_mtime_ns for c in chunks2}
        # Chunks reused: mtimes preserved
        assert mtimes_before == mtimes_after

    def test_overlap_must_be_smaller_than_chunk(self, tmp_path):
        wav = tmp_path / "x.wav"
        wav.write_bytes(b"\x00" * 100)
        with pytest.raises(ValueError, match="chunk_duration_s"):
            split_long_audio(wav, tmp_path / "c", chunk_duration_s=10, overlap_s=10)

    def test_missing_input_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            split_long_audio(tmp_path / "no.wav", tmp_path / "c")
