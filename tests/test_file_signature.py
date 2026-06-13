"""Content signatures (fast/full) and ffmpeg timeout conversion."""

from __future__ import annotations

import hashlib
import subprocess

import pytest

from speakerscribe.audio import calculate_file_hash, file_signature

MB8 = 8 << 20


class TestFileSignature:
    def test_full_equals_legacy_sha256(self, tmp_path):
        """Migration depends on this: full mode must match pre-0.3 hashes."""
        p = tmp_path / "a.bin"
        p.write_bytes(b"abc" * 1000)
        assert file_signature(p, "full") == hashlib.sha256(b"abc" * 1000).hexdigest()
        assert calculate_file_hash(p) == file_signature(p, "full")

    def test_fast_detects_head_tail_and_size_changes(self, tmp_path):
        p = tmp_path / "a.bin"
        data = bytearray(b"a" * (20 << 20))
        p.write_bytes(bytes(data))
        sig = file_signature(p, "fast")

        data[0] = ord("X")  # head change
        p.write_bytes(bytes(data))
        assert file_signature(p, "fast") != sig

        data[0] = ord("a")
        data[-1] = ord("X")  # tail change
        p.write_bytes(bytes(data))
        assert file_signature(p, "fast") != sig

        p.write_bytes(b"a" * ((20 << 20) + 1))  # size change
        assert file_signature(p, "fast") != sig

    def test_fast_documented_blind_spot_middle_only_change(self, tmp_path):
        """Documented trade-off: a middle-only edit with identical size/extremes
        is invisible to fast mode (not a realistic pattern for re-encoded media).
        """
        p = tmp_path / "a.bin"
        data = bytearray(b"a" * (20 << 20))
        p.write_bytes(bytes(data))
        sig = file_signature(p, "fast")
        data[10 << 20] = ord("X")
        p.write_bytes(bytes(data))
        assert file_signature(p, "fast") == sig
        assert file_signature(p, "full") != hashlib.sha256(b"a" * (20 << 20)).hexdigest()

    def test_small_file_consistent(self, tmp_path):
        p = tmp_path / "small.bin"
        p.write_bytes(b"tiny")
        assert file_signature(p, "fast") == file_signature(p, "fast")
        assert file_signature(p, "fast") != file_signature(p, "full")

    def test_medium_file_between_8_and_16mb(self, tmp_path):
        p = tmp_path / "mid.bin"
        p.write_bytes(b"m" * (MB8 + 100))
        s1 = file_signature(p, "fast")
        p.write_bytes(b"m" * MB8 + b"X" * 100)
        assert file_signature(p, "fast") != s1

    def test_invalid_mode(self, tmp_path):
        p = tmp_path / "a.bin"
        p.write_bytes(b"x")
        with pytest.raises(ValueError, match="fast"):
            file_signature(p, "sha1")

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            file_signature(tmp_path / "nope.bin", "fast")


class TestFfmpegTimeouts:
    def test_ffprobe_timeout_becomes_actionable_runtimeerror(self, tmp_path, monkeypatch):
        from speakerscribe import audio as audio_mod

        p = tmp_path / "a.wav"
        p.write_bytes(b"RIFF")
        monkeypatch.setattr(audio_mod.shutil, "which", lambda _: "/usr/bin/ffprobe")

        def fake_run(cmd, **kw):
            raise subprocess.TimeoutExpired(cmd, kw.get("timeout", 0))

        monkeypatch.setattr(audio_mod.subprocess, "run", fake_run)
        with pytest.raises(RuntimeError, match="timed out"):
            audio_mod.get_audio_duration_seconds(p)

    def test_auto_timeout_formula(self):
        from speakerscribe.audio import _auto_ffmpeg_timeout

        assert _auto_ffmpeg_timeout(None, 99) == 99.0  # explicit config wins
        assert _auto_ffmpeg_timeout(600.0, 0) == 120.0 + 1200.0
        assert _auto_ffmpeg_timeout(None, 0) == 3600.0  # unknown duration: hard cap
