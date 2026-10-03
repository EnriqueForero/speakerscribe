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

    On 2026-10-03 the next layer surfaced: Colab's image moved to CUDA 13
    (torch +cu130), which ships ``libcublas.so.13`` only, while every
    CTranslate2 4.x wheel dlopens ``libcublas.so.12`` on the first GPU
    matmul — after the model loaded, in the middle of the first file.
    `ensure_ctranslate2_cuda_libs()` makes that library loadable (or fails
    with the fix) BEFORE Whisper is loaded.

Public API:
    EnvironmentIncompatibleError — Raised when the stack cannot decode audio.
    check_audio_decoding         — Fast decoding self-test (no GPU, no model).
    package_versions             — Installed versions of the audio/ML stack.
    is_environment_error         — Classify an exception as environmental.
    is_environment_error_text    — Same, from a journaled ``Type: message``.
    ensure_ctranslate2_cuda_libs — Make CTranslate2's cuBLAS loadable.
"""

from __future__ import annotations

import contextlib
import ctypes
import functools
import glob
import importlib.metadata as importlib_metadata
import importlib.util
import mmap
import re
import site
import sys
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
    "cublas",  # CUBLAS_STATUS_*, libcublas.so.N, libcublasLt.so.N
    "undefined symbol",
    "no kernel image is available",
    "is not found or cannot be loaded",  # CTranslate2's dlopen failure
    "cannot open shared object file",  # glibc loader
)
"""Messages (RuntimeError/OSError) that point at the CUDA/driver stack (not OOM)."""

_ENVIRONMENT_TYPE_NAMES: frozenset[str] = frozenset(
    {
        "ImportError",
        "ModuleNotFoundError",
        "AttributeError",
        "TypeError",
        "NameError",
        "EnvironmentIncompatibleError",
    }
)
"""`ENVIRONMENT_ERROR_TYPES` by name, for errors known only as journaled text."""


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
    if isinstance(exc, (*ENVIRONMENT_ERROR_TYPES, EnvironmentIncompatibleError)):
        return True
    if isinstance(exc, RuntimeError | OSError):
        return _message_is_environmental(str(exc))
    return False


def _message_is_environmental(message: str) -> bool:
    lowered = message.lower()
    if "out of memory" in lowered:
        return False
    return any(marker in lowered for marker in _ENVIRONMENT_MESSAGE_MARKERS)


def is_environment_error_text(text: str | None) -> bool:
    """`is_environment_error` for an error known only as ``"Type: message"`` text.

    Lets a journal re-read old failures with today's rules: a failure that
    was recorded as a per-file error before a marker existed (e.g.
    ``RuntimeError: Library libcublas.so.12 is not found or cannot be
    loaded`` on 2026-10-03) stops counting as a consumed retry attempt.

    Args:
        text: ``sanitize_error`` output, e.g. ``"RuntimeError: ..."``.

    Returns:
        True if the text describes a broken environment.
    """
    if not text:
        return False
    type_name, sep, message = text.partition(":")
    if sep and type_name.strip() in _ENVIRONMENT_TYPE_NAMES:
        return True
    return _message_is_environmental(message if sep else text)


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


# ── CTranslate2 ↔ CUDA runtime libraries ──────────────────────────────────

_DEFAULT_CT2_CUBLAS = "libcublas.so.12"
_CUBLAS_SONAME = re.compile(rb"libcublas\.so\.(\d+)")
_PRELOADED: list[ctypes.CDLL] = []  # keep handles alive for the process lifetime


@functools.lru_cache(maxsize=1)
def ctranslate2_cublas_soname() -> str:
    """The cuBLAS soname the installed CTranslate2 dlopens (e.g. ``libcublas.so.12``).

    Read from the bundled ``libctranslate2`` binary (cheap: one mmap'd regex
    scan, cached), so a future CTranslate2 built for another CUDA major is
    handled without a code change. Falls back to ``libcublas.so.12``, the
    soname of every CTranslate2 4.x wheel up to 4.8.2.
    """
    spec = importlib.util.find_spec("ctranslate2")
    if spec is None or not spec.origin:
        return _DEFAULT_CT2_CUBLAS
    package = Path(spec.origin).parent
    for binary in sorted(
        [*package.parent.glob("ctranslate2.libs/libctranslate2*.so*"), *package.glob("*.so*")]
    ):
        try:
            with binary.open("rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                match = _CUBLAS_SONAME.search(mm)
                major = match.group(1).decode() if match else None  # read before unmapping
        except (OSError, ValueError, TypeError):
            continue
        if major:
            return f"libcublas.so.{major}"
    return _DEFAULT_CT2_CUBLAS


def _library_dirs() -> list[Path]:
    """Directories where pip's ``nvidia-*`` wheels and CUDA toolkits put libraries."""
    bases: list[str] = [*sys.path]
    with contextlib.suppress(AttributeError):  # virtualenvs without the site helpers
        bases += [*site.getsitepackages(), site.getusersitepackages()]
    dirs: list[Path] = []
    for base in dict.fromkeys(b for b in bases if b):
        nvidia = Path(base) / "nvidia"
        if nvidia.is_dir():
            dirs += sorted(p for p in nvidia.glob("*/lib") if p.is_dir())
    dirs += [Path(p) for p in sorted(glob.glob("/usr/local/cuda*/lib64"))]
    dirs += [Path(p) for p in sorted(glob.glob("/usr/local/cuda*/targets/x86_64-linux/lib"))]
    return dirs


def _loadable(soname: str) -> bool:
    try:
        ctypes.CDLL(soname)
    except OSError:
        return False
    return True


def ensure_ctranslate2_cuda_libs(*, library_dirs: list[Path] | None = None) -> dict[str, Any]:
    """Make the cuBLAS that CTranslate2 dlopens loadable in this process.

    CTranslate2 loads cuBLAS lazily (``dlopen("libcublas.so.12")``) on the
    first GPU matmul, so a missing library would only surface mid-file,
    after the model loaded. Call this before loading Whisper on CUDA.

    Strategy:
        1. The soname already resolves (system CUDA 12, or a torch built
           for CUDA 12 preloaded it) → nothing to do.
        2. A copy exists in a known directory (pip ``nvidia-cublas-cu12``,
           ``/usr/local/cuda-12*``) → preload it by absolute path. Its
           ``RUNPATH=$ORIGIN`` pulls in ``libcublasLt`` from the same
           directory; the matching ``libnvrtc`` is preloaded too when
           present. Loading is ``RTLD_LOCAL``: CTranslate2's later dlopen by
           soname reuses these handles, while torch's own cuBLAS (e.g. 13)
           keeps resolving its symbols — no interposition.
        3. Otherwise raise `EnvironmentIncompatibleError` with the fix.

    Args:
        library_dirs: Directories to search (tests); defaults to pip's
            ``nvidia/*/lib`` folders on ``sys.path`` and CUDA toolkits.

    Returns:
        ``{"soname": ..., "source": "system" | "<absolute path>"}``.

    Raises:
        EnvironmentIncompatibleError: The library cannot be found anywhere.
    """
    soname = ctranslate2_cublas_soname()
    if _loadable(soname):
        return {"soname": soname, "source": "system"}
    major = soname.rsplit(".", 1)[-1]
    for directory in library_dirs if library_dirs is not None else _library_dirs():
        candidate = directory / soname
        if not candidate.is_file():
            continue
        nvrtc = f"libnvrtc.so.{major}"
        for helper in [directory / nvrtc, *sorted(directory.parent.parent.glob(f"*/lib/{nvrtc}"))]:
            if not helper.is_file():
                continue
            with contextlib.suppress(OSError):  # optional: only runtime-compiled kernels use it
                _PRELOADED.append(ctypes.CDLL(str(helper)))
        try:
            _PRELOADED.append(ctypes.CDLL(str(candidate)))
        except OSError as e:
            raise EnvironmentIncompatibleError(
                f"{candidate} exists but cannot be loaded: {e}"
            ) from e
        if not _loadable(soname):  # exactly the lookup CTranslate2 will perform
            raise EnvironmentIncompatibleError(
                f"Preloaded {candidate}, but {soname} still does not resolve by name."
            )
        logger.info(f"CTranslate2 cuBLAS preloaded: {candidate}")
        return {"soname": soname, "source": str(candidate)}
    raise EnvironmentIncompatibleError(
        f"Library {soname} is not found or cannot be loaded: CTranslate2 (faster-whisper) "
        f"needs CUDA {major} cuBLAS on the GPU, and this runtime does not have it "
        f'(e.g. a CUDA 13 image). Fix: pip install "nvidia-cublas-cu{major}" '
        f'(or pip install "speakerscribe[cuda{major}]") and run again.'
    )


__all__ = [
    "ENVIRONMENT_ERROR_TYPES",
    "STACK_PACKAGES",
    "EnvironmentIncompatibleError",
    "check_audio_decoding",
    "ctranslate2_cublas_soname",
    "ensure_ctranslate2_cuda_libs",
    "is_environment_error",
    "is_environment_error_text",
    "package_versions",
]
