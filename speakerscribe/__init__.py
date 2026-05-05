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
    audio        — WAV extraction, duration probe, long-audio splitting, hashing
    diarization  — pyannote.audio 4.x + overlap-based speaker assignment
    transcription— faster-whisper streaming + chunked transcription
    output       — Markdown transcript + word-aware splits + filler filter
    quality      — Heuristic post-transcription quality checker
    pipeline     — Orchestration: process_one + process_batch
    maintenance  — Helpers for cleaning, inspecting, renaming speakers
    persistence  — SQLite history of runs (idempotency by file hash)
    logging_config — loguru setup (console + rotating file sinks)
"""

from importlib.metadata import PackageNotFoundError, version

# ─── Package version ───────────────────────────────────────────────
try:
    __version__: str = version("speakerscribe")
except PackageNotFoundError:
    __version__ = "0.1.1"

# ─── Public API ────────────────────────────────────────────────────
from speakerscribe.audio import (
    AudioChunk,
    get_audio_duration_seconds,
    split_long_audio,
)
from speakerscribe.config import (
    FILLER_WORDS,
    SPK_NO_DIARIZATION,
    SPK_NO_OVERLAP,
    TranscriptionConfig,
    WorkspacePaths,
)
from speakerscribe.diarization import diarization_params_hash
from speakerscribe.maintenance import (
    delete_all_outputs,
    delete_outputs_for,
    inspect_json,
    rename_speakers_in_outputs,
)
from speakerscribe.output import (
    is_filler_only,
    write_unified_for_llm,
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
    "FILLER_WORDS",
    "SPK_NO_DIARIZATION",
    "SPK_NO_OVERLAP",
    "AudioChunk",
    "QualityFlag",
    "QualityReport",
    "Severity",
    "TranscriptionConfig",
    "WorkspacePaths",
    "__version__",
    "delete_all_outputs",
    "delete_outputs_for",
    "diarization_params_hash",
    "evaluate_transcription_quality",
    "get_audio_duration_seconds",
    "inspect_json",
    "is_filler_only",
    "preflight_check",
    "process_batch",
    "process_one",
    "rename_speakers_in_outputs",
    "split_long_audio",
    "write_unified_for_llm",
]
