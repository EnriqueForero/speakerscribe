"""speakerscribe — Speech-to-text with speaker diarization.

faster-whisper (CTranslate2) + pyannote.audio 4.x, designed for Google
Colab with outputs on Drive and high-churn temporaries on local scratch.

Quick start:
    >>> from speakerscribe import TranscriptionConfig, WorkspacePaths, process_batch
    >>> config = TranscriptionConfig(model="large-v3-turbo", language="es")
    >>> paths = WorkspacePaths(workspace="/content/drive/MyDrive/MyProject")
    >>> results = process_batch(paths, config)  # doctest: +SKIP
"""

from speakerscribe.config import (
    FILLERS_AGGRESSIVE,
    FILLERS_SAFE,
    SPK_NO_DIARIZATION,
    SPK_NO_OVERLAP,
    TranscriptionConfig,
    WorkspacePaths,
)
from speakerscribe.diarization import DiarizationEngine, diarize_audio
from speakerscribe.pipeline import preflight_check, process_batch, process_one
from speakerscribe.quality import evaluate_transcription_quality
from speakerscribe.transcription import (
    load_whisper_model,
    loaded_whisper,
    release_whisper_model,
)

__version__ = "0.3.0"

__all__ = [
    "FILLERS_AGGRESSIVE",
    "FILLERS_SAFE",
    "SPK_NO_DIARIZATION",
    "SPK_NO_OVERLAP",
    "DiarizationEngine",
    "TranscriptionConfig",
    "WorkspacePaths",
    "__version__",
    "diarize_audio",
    "evaluate_transcription_quality",
    "load_whisper_model",
    "loaded_whisper",
    "preflight_check",
    "process_batch",
    "process_one",
    "release_whisper_model",
]
