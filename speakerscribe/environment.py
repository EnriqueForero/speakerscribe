"""Runtime environment checks: decoding self-test, stack versions, error triage.

Why this module exists:
    On 2026-10-01 a third-party release (PyAV 19.0.0) broke every
    transcription in Colab, and the batch kept going file after file —
    each one paying a full diarization before failing identically in ASR.
    Two cheap defenses prevent that class of incident:

    1. `check_audio_decoding()` exercises the real decoding path on one
       second of synthetic audio, in well under a second, BEFORE any model
       is loaded.
    2. `is_environment_error()` tells a broken environment (same failure
       for every file) apart from a problem with one file, so orchestrators
       stop the batch instead of burning GPU quota on every remaining file.

Public API:
    EnvironmentIncompatibleError — Raised when the stack cannot decode audio.
    check_audio_decoding         — Fast decoding self-test (no GPU, no model).
    package_versions             — Installed versions of the audio/ML stack.
    is_environment_error         — Classify an exception as environmental.
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import tempfile
import wave
from pathlib import Path
from typing import Any

from speakerscribe.audio import read_wav_float32
from speakerscribe.logging_config import logger

STACK_PACKAGES: tuple[str, ...] = (
    "speakerscribe",
    "faster-whisper",
    "av",
    "ctranslate2",
    "onnxruntime",
    "torch",
    "torchaudio",
    "torchcodec",
    "pyannote.audio",
    "numpy",
)
"""Distributions whose versions explain most environment failures."""

_SELFTEST_SECONDS = 1.0
_SELFTEST_AMPLITUDE = 0.25
_SELFTEST_TONE_HZ = 440.0

ENVIRONMENT_ERROR_TYPES: tuple[type[BaseException], ...] = (
    ImportError,  # includes ModuleNotFoundError
    AttributeError,
    TypeError,
    NameError,
)
"""Exception types that signal a broken/incompatible stack, not a bad file.

Deliberately excludes OSError: on Google Drive (FUSE) an OSError is usually
a transient per-file hiccup, not a reason to stop the batch."""

_ENVIRONMENT_MESSAGE_MARKERS: tuple[str, ...] = (
    "cuda error",
    "cudnn",
    "cublas_status",
    "libcudnn",
    "undefined symbol",
    "no kernel image is available",
)
"""RuntimeError messages that point at the CUDA/driver stack (not OOM)."""


class EnvironmentIncompatibleError(RuntimeError):
    """The installed stack cannot decode or transcribe audio at all."""


def package_versions(names: tuple[str, ...] = STACK_PACKAGES) -> dict[str, str | None]:
    """Installed version of each distribution (None when not installed).

    Args:
        names: Distribution names as published on PyPI.

    Returns:
        Mapping name -> version string or None.
    """
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def is_environment_error(exc: BaseException) -> bool:
    """True when `exc` signals a broken environment rather than a bad file.

    Examples: `TypeError: open() got an unexpected keyword argument
    'metadata_errors'` (PyAV 19 vs faster-whisper 1.2.1), an ImportError of a
    compiled extension, a CUDA driver error. CUDA *out of memory* is NOT
    environmental: the transcription layer already retries it with a smaller
    batch.

    Args:
        exc: The exception raised while processing one file.

    Returns:
        True if the same failure is expected for every remaining file.
    """
    if isinstance(exc, ENVIRONMENT_ERROR_TYPES):
        return True
    if isinstance(exc, RuntimeError):
        message = str(exc).lower()
        if "out of memory" in message:
            return False
        return any(marker in message for marker in _ENVIRONMENT_MESSAGE_MARKERS)
    return False


def _write_selftest_wav(path: Path, sample_rate: int) -> None:
    """Write a short PCM16 mono tone (stdlib + numpy only)."""
    import numpy as np

    n = int(sample_rate * _SELFTEST_SECONDS)
    t = np.arange(n, dtype=np.float64) / sample_rate
    tone = _SELFTEST_AMPLITUDE * np.sin(2.0 * np.pi * _SELFTEST_TONE_HZ * t)
    pcm = np.clip(np.round(tone * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def check_audio_decoding(
    *, sample_rate: int = 16_000, require_pyav: bool = False
) -> dict[str, Any]:
    """Self-test the audio decoding paths on one second of synthetic audio.

    Two paths are tested:
        * ``native_wav`` — `read_wav_float32`, the path the pipeline uses for
          the WAV it extracts. Must work; otherwise nothing can be transcribed.
        * ``pyav`` — `faster_whisper.audio.decode_audio`, used only as a
          fallback for non-PCM16 inputs. A failure here is reported (and
          logged as a warning) but only raises when `require_pyav=True`.

    Args:
        sample_rate: Sampling rate of the synthetic WAV (Hz).
        require_pyav: Raise if the PyAV path fails too.

    Returns:
        Dict with keys ``native_wav`` and ``pyav`` ("ok", "unavailable" or
        "error: <type>: <message>"), ``samples`` and ``versions``.

    Raises:
        EnvironmentIncompatibleError: If the native path fails, or the PyAV
            path fails while `require_pyav` is True.
    """
    report: dict[str, Any] = {"versions": package_versions()}
    with tempfile.TemporaryDirectory(prefix="speakerscribe_selftest_") as tmp:
        wav = Path(tmp) / "selftest.wav"
        _write_selftest_wav(wav, sample_rate)
        expected = int(sample_rate * _SELFTEST_SECONDS)

        try:
            samples = read_wav_float32(wav, expected_sample_rate=sample_rate)
            if samples.size != expected:
                raise RuntimeError(f"native reader returned {samples.size} of {expected} samples")
            report["native_wav"] = "ok"
            report["samples"] = int(samples.size)
        except Exception as e:  # any failure here is fatal by definition
            report["native_wav"] = f"error: {type(e).__name__}: {e}"
            raise EnvironmentIncompatibleError(
                f"Native WAV reading failed: {type(e).__name__}: {e}"
            ) from e

        try:
            from faster_whisper.audio import decode_audio
        except Exception as e:  # faster-whisper absent or faked in tests
            report["pyav"] = f"unavailable: {type(e).__name__}"
        else:
            try:
                decoded = decode_audio(str(wav), sampling_rate=sample_rate)
                report["pyav"] = (
                    "ok"
                    if len(decoded) == expected
                    else (f"error: decoded {len(decoded)} of {expected} samples")
                )
            except Exception as e:
                report["pyav"] = f"error: {type(e).__name__}: {e}"

    if report["pyav"].startswith("error"):
        versions = report["versions"]
        hint = (
            f"faster-whisper {versions.get('faster-whisper')} cannot decode with PyAV "
            f"{versions.get('av')} ({report['pyav']}). speakerscribe reads its own WAVs "
            'without PyAV, but non-WAV fallbacks would fail. Fix: pip install "av>=11,<19".'
        )
        if require_pyav:
            raise EnvironmentIncompatibleError(hint)
        logger.warning(hint)
    return report


__all__ = [
    "ENVIRONMENT_ERROR_TYPES",
    "STACK_PACKAGES",
    "EnvironmentIncompatibleError",
    "check_audio_decoding",
    "is_environment_error",
    "package_versions",
]
