"""speakerscribe — Speech-to-text with speaker diarization.

A production-oriented Python package that combines OpenAI Whisper (via
faster-whisper) with pyannote.audio for transcription with speaker labels.
Optimized for Google Colab Free Tier (T4 GPU) and local CUDA / CPU runs.

Quickstart:
    >>> from speakerscribe import TranscriptionConfig, WorkspacePaths, process_batch
    >>> config = TranscriptionConfig(model="large-v3-turbo", language="es")
    >>> paths = WorkspacePaths(workspace="/content/drive/MyDrive/MyAudios")
    >>> results = process_batch(paths, config)

Modules:
    config       — TranscriptionConfig + WorkspacePaths (Pydantic v2)
    audio        — WAV extraction with ffmpeg + time and hashing helpers
    diarization  — pyannote.audio 4.x + overlap-based speaker assignment
    transcription— faster-whisper streaming + per-segment speaker labeling
    output       — Markdown transcript generation + word-aware splits
    quality      — Heuristic post-transcription quality checker
    pipeline     — Orchestration: process_one + process_batch
    maintenance  — Helpers for cleaning, inspecting, renaming speakers
    persistence  — Optional SQLite history of runs (idempotency by file hash)
    logging_config — loguru setup (console + rotating file sinks)
"""

from importlib.metadata import PackageNotFoundError, version

# ─── Package version ───────────────────────────────────────────────
try:
    __version__: str = version("speakerscribe")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"

# ─── Public API ────────────────────────────────────────────────────
from speakerscribe.config import (
    SPK_NO_DIARIZATION,
    SPK_NO_OVERLAP,
    TranscriptionConfig,
    WorkspacePaths,
)
from speakerscribe.maintenance import (
    delete_all_outputs,
    delete_outputs_for,
    inspect_json,
    rename_speakers_in_outputs,
)
from speakerscribe.pipeline import (
    preflight_check,
    process_batch,
    process_one,
)
from speakerscribe.quality import (
    QualityFlag,
    QualityReport,
    Severity,
    evaluate_transcription_quality,
)

__all__ = [
    "QualityFlag",
    "QualityReport",
    "SPK_NO_DIARIZATION",
    "SPK_NO_OVERLAP",
    "Severity",
    "TranscriptionConfig",
    "WorkspacePaths",
    "__version__",
    "delete_all_outputs",
    "delete_outputs_for",
    "evaluate_transcription_quality",
    "inspect_json",
    "preflight_check",
    "process_batch",
    "process_one",
    "rename_speakers_in_outputs",
]
