"""Environment self-test and error triage (incident 2026-10-01, PyAV 19)."""

from __future__ import annotations

import sys
import types

import pytest

from speakerscribe import environment as env
from speakerscribe.environment import (
    EnvironmentIncompatibleError,
    check_audio_decoding,
    ensure_ctranslate2_cuda_libs,
    is_environment_error,
    is_environment_error_text,
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
            # 2026-10-03: CTranslate2 on Colab's CUDA 13 image (journaled as per-file before)
            RuntimeError("Library libcublas.so.12 is not found or cannot be loaded"),
            OSError("libcublasLt.so.12: cannot open shared object file: No such file"),
            EnvironmentIncompatibleError("any message"),
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


class TestIsEnvironmentErrorText:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("RuntimeError: Library libcublas.so.12 is not found or cannot be loaded", True),
            ("TypeError: open() got an unexpected keyword argument 'metadata_errors'", True),
            ("ModuleNotFoundError: No module named 'ctranslate2'", True),
            ("EnvironmentIncompatibleError: Native WAV reading failed", True),
            ("RuntimeError: CUDA failed to allocate: out of memory", False),
            ("OSError: [Errno 5] Input/output error", False),
            ("ValueError: bad segment", False),
            ("DiarizationFileError: diarization failed", False),
            ("", False),
            (None, False),
        ],
    )
    def test_matches_the_exception_classifier(self, text, expected):
        assert is_environment_error_text(text) is expected


class TestCtranslate2CudaLibs:
    """CTranslate2 dlopens cuBLAS mid-file; Colab's CUDA 13 image lacks libcublas.so.12."""

    @pytest.fixture(autouse=True)
    def _fresh(self, monkeypatch):
        real = env.ctranslate2_cublas_soname  # tests may replace it; clear the real cache
        real.cache_clear()
        monkeypatch.setattr(env, "_PRELOADED", [])
        yield
        real.cache_clear()

    def _fake_ct2(self, tmp_path, monkeypatch, payload: bytes | None):
        pkg = tmp_path / "site" / "ctranslate2"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("")
        if payload is not None:
            libs = tmp_path / "site" / "ctranslate2.libs"
            libs.mkdir()
            (libs / "libctranslate2-abc.so.4.9.0").write_bytes(b"\0junk" + payload + b"\0")
        spec = types.SimpleNamespace(origin=str(pkg / "__init__.py"))
        monkeypatch.setattr(env.importlib.util, "find_spec", lambda name: spec)

    def test_soname_is_read_from_the_ctranslate2_binary(self, tmp_path, monkeypatch):
        self._fake_ct2(tmp_path, monkeypatch, b"libcublas.so.13")
        assert env.ctranslate2_cublas_soname() == "libcublas.so.13"

    def test_soname_defaults_to_cuda12(self, tmp_path, monkeypatch):
        self._fake_ct2(tmp_path, monkeypatch, None)
        assert env.ctranslate2_cublas_soname() == "libcublas.so.12"
        env.ctranslate2_cublas_soname.cache_clear()
        monkeypatch.setattr(env.importlib.util, "find_spec", lambda name: None)
        assert env.ctranslate2_cublas_soname() == "libcublas.so.12"

    def _fake_loader(self, monkeypatch, *, resolvable_after_load: bool = True):
        loaded: list[str] = []

        def cdll(name):
            if "/" in name:
                loaded.append(name)
                return types.SimpleNamespace(name=name)
            if resolvable_after_load and any(p.endswith("/" + name) for p in loaded):
                return types.SimpleNamespace(name=name)
            raise OSError(f"{name}: cannot open shared object file")

        monkeypatch.setattr(env.ctypes, "CDLL", cdll)
        monkeypatch.setattr(env, "ctranslate2_cublas_soname", lambda: "libcublas.so.12")
        return loaded

    def test_system_library_needs_nothing(self, monkeypatch):
        monkeypatch.setattr(env.ctypes, "CDLL", lambda name: types.SimpleNamespace(name=name))
        monkeypatch.setattr(env, "ctranslate2_cublas_soname", lambda: "libcublas.so.12")
        assert ensure_ctranslate2_cuda_libs(library_dirs=[]) == {
            "soname": "libcublas.so.12",
            "source": "system",
        }

    def test_pip_wheel_is_preloaded_by_path_with_nvrtc_first(self, tmp_path, monkeypatch):
        loaded = self._fake_loader(monkeypatch)
        cublas_dir = tmp_path / "nvidia" / "cublas" / "lib"
        nvrtc_dir = tmp_path / "nvidia" / "cuda_nvrtc" / "lib"
        for d, name in ((cublas_dir, "libcublas.so.12"), (nvrtc_dir, "libnvrtc.so.12")):
            d.mkdir(parents=True)
            (d / name).write_bytes(b"")
        info = ensure_ctranslate2_cuda_libs(library_dirs=[tmp_path / "empty", cublas_dir])
        assert info == {"soname": "libcublas.so.12", "source": str(cublas_dir / "libcublas.so.12")}
        assert loaded == [str(nvrtc_dir / "libnvrtc.so.12"), str(cublas_dir / "libcublas.so.12")]
        assert len(env._PRELOADED) == 2, "handles are kept alive"

    def test_missing_everywhere_raises_with_the_fix(self, tmp_path, monkeypatch):
        self._fake_loader(monkeypatch)
        with pytest.raises(EnvironmentIncompatibleError, match="nvidia-cublas-cu12") as caught:
            ensure_ctranslate2_cuda_libs(library_dirs=[tmp_path])
        assert is_environment_error(caught.value)

    def test_preloaded_but_still_unresolvable_raises(self, tmp_path, monkeypatch):
        self._fake_loader(monkeypatch, resolvable_after_load=False)
        (tmp_path / "libcublas.so.12").write_bytes(b"")
        with pytest.raises(EnvironmentIncompatibleError, match="still does not resolve"):
            ensure_ctranslate2_cuda_libs(library_dirs=[tmp_path])


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
