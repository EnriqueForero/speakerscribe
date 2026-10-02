"""Batch error taxonomy and safe error text.

Each class maps to one journal outcome, so the runner's handling is a table
rather than a chain of string checks:

==========================  ==================  ====================
Exception                   Journal event       Consumes an attempt?
==========================  ==================  ====================
FatalDiarizationSetupError  failed_environment  no (stops the batch)
environment error (*)       failed_environment  no (breaker stops)
DiarizationFileError        failed_retryable    yes
QualityRejectedError        quality_rejected    yes
SourceChangingError         source_changed      no
any other Exception         failed_retryable    yes
==========================  ==================  ====================

(*) ``speakerscribe.environment.EnvironmentIncompatibleError`` or any error
classified by ``is_environment_error`` (broken stack: ImportError,
signature TypeError, CUDA/cuDNN symbol errors).
"""

from __future__ import annotations

import re
import traceback

_HF_TOKEN = re.compile(r"hf_[A-Za-z0-9_-]{8,}")
MAX_ERROR_CHARS = 4000
MAX_TRACEBACK_CHARS = 8000


class FatalDiarizationSetupError(RuntimeError):
    """pyannote cannot load (token, gated terms, CUDA): every file would fail."""


class DiarizationFileError(RuntimeError):
    """Diarization failed for this file only (retryable)."""


class QualityRejectedError(RuntimeError):
    """The result failed the quality gate (retryable, consumes an attempt)."""


def redact(text: str) -> str:
    """Remove HuggingFace tokens from any text that may be persisted."""
    return _HF_TOKEN.sub("hf_***redactado***", text)


def sanitize_error(exc: BaseException) -> str:
    """``Type: message`` without tokens, bounded."""
    return redact(f"{type(exc).__name__}: {exc}")[:MAX_ERROR_CHARS]


def safe_traceback(exc: BaseException) -> str:
    """Tail of the formatted traceback, without tokens."""
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return redact(text)[-MAX_TRACEBACK_CHARS:]


__all__ = [
    "DiarizationFileError",
    "FatalDiarizationSetupError",
    "QualityRejectedError",
    "redact",
    "safe_traceback",
    "sanitize_error",
]
