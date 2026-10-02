"""Environment self-test and error triage (incident 2026-10-01, PyAV 19)."""

from __future__ import annotations

import sys
import types

import pytest

from speakerscribe import environment as env
from speakerscribe.environment import (
    EnvironmentIncompatibleError,
    check_audio_decoding,
    is_environment_error,
    package_versions,
)


class TestIsEnvironmentError:
    @pytest.mark.parametrize(
        "exc",
        [
            TypeError("open() got an unexpected keyword argument 'metadata_errors'"),
            ImportError("libcudnn_ops.so.9: cannot open shared object file"),
            ModuleNotFoundError("No module named 'ctranslate2'"),
            AttributeError("module 'av' has no attribute 'open'"),
            RuntimeError("CUDA error: no kernel image is available for execution"),
            RuntimeError("undefined symbol: _ZN3c104cuda"),
        ],
    )
    def test_environmental(self, exc):
        assert is_environment_error(exc) is True

    @pytest.mark.parametrize(
        "exc",
        [
            OSError("[Errno 5] Input/output error"),  # Drive/FUSE hiccup: per file
            ValueError("bad segment"),
            RuntimeError("CUDA failed to allocate: out of memory"),  # handled by OOM ladder
            RuntimeError("ffmpeg failed: Invalid data found when processing input"),
            KeyError("speaker"),
        ],
    )
    def test_not_environmental(self, exc):
        assert is_environment_error(exc) is False


def test_package_versions_reports_missing_as_none():
    versions = package_versions(("pytest", "a-distribution-that-does-not-exist-xyz"))
    assert versions["pytest"]
    assert versions["a-distribution-that-does-not-exist-xyz"] is None


def _fake_fw_audio(decode):
    mod = types.ModuleType("faster_whisper")
    audio = types.ModuleType("faster_whisper.audio")
    audio.decode_audio = decode  # type: ignore[attr-defined]
    mod.audio = audio  # type: ignore[attr-defined]
    return mod, audio


class TestCheckAudioDecoding:
    def test_native_ok_and_pyav_ok(self, monkeypatch):
        import numpy as np

        mod, audio = _fake_fw_audio(lambda path, sampling_rate: np.zeros(sampling_rate))
        monkeypatch.setitem(sys.modules, "faster_whisper", mod)
        monkeypatch.setitem(sys.modules, "faster_whisper.audio", audio)
        report = check_audio_decoding()
        assert report["native_wav"] == "ok"
        assert report["samples"] == 16_000
        assert report["pyav"] == "ok"
        assert "versions" in report

    def test_pyav_failure_is_warning_by_default(self, monkeypatch):
        def broken(path, sampling_rate):
            raise TypeError("open() got an unexpected keyword argument 'metadata_errors'")

        mod, audio = _fake_fw_audio(broken)
        monkeypatch.setitem(sys.modules, "faster_whisper", mod)
        monkeypatch.setitem(sys.modules, "faster_whisper.audio", audio)
        report = check_audio_decoding()
        assert report["native_wav"] == "ok"
        assert report["pyav"].startswith("error: TypeError")

    def test_pyav_failure_raises_when_required(self, monkeypatch):
        def broken(path, sampling_rate):
            raise TypeError("open() got an unexpected keyword argument 'metadata_errors'")

        mod, audio = _fake_fw_audio(broken)
        monkeypatch.setitem(sys.modules, "faster_whisper", mod)
        monkeypatch.setitem(sys.modules, "faster_whisper.audio", audio)
        with pytest.raises(EnvironmentIncompatibleError, match="av>=11,<19"):
            check_audio_decoding(require_pyav=True)

    def test_pyav_unavailable_is_reported_not_raised(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "faster_whisper", types.ModuleType("faster_whisper"))
        monkeypatch.setitem(sys.modules, "faster_whisper.audio", None)
        report = check_audio_decoding()
        assert report["pyav"].startswith("unavailable")

    def test_native_failure_is_fatal(self, monkeypatch):
        def boom(path, expected_sample_rate):
            raise RuntimeError("disk on fire")

        monkeypatch.setattr(env, "read_wav_float32", boom)
        with pytest.raises(EnvironmentIncompatibleError, match="Native WAV reading failed"):
            check_audio_decoding()
