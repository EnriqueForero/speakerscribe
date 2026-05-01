"""Tests for speakerscribe.audio."""

from __future__ import annotations

import pytest

from speakerscribe.audio import (
    calculate_file_hash,
    format_hms,
    format_srt_timestamp,
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
        assert len(h1) == 64  # SHA-256 hex digest

    def test_different_contents_different_hashes(self, tmp_path):
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("content A")
        b.write_text("content B")
        assert calculate_file_hash(a) != calculate_file_hash(b)

    def test_large_file_streamed_correctly(self, tmp_path):
        """Hash must work for files larger than the default chunk size."""
        large_file = tmp_path / "large.bin"
        large_file.write_bytes(b"x" * (3 * 1024 * 1024))  # 3 MB
        h = calculate_file_hash(large_file, chunk_size=1024)
        assert len(h) == 64
