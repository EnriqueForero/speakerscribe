"""Pipeline package: orchestration of the speakerscribe transcription flow.

Public API:
    preflight_check  — Validate environment before loading any model.
    process_one      — Run the pipeline for a single media file.
    process_batch    — Run the pipeline over every file in the workspace.

Internal API (stable for tests; private to external users):
    RunContext, _maybe_extract_wav, _maybe_diarize, _should_chunk_audio,
    _cleanup_chunks, _purge_stale_chunks, report_speaker_distribution.

Stage modules (one per pipeline phase):
    pipeline.preflight             - environment validation
    pipeline.idempotency           - DB / output cache check
    pipeline.wav                   - ffmpeg WAV extraction + chunking decision
    pipeline.diarization_stage     - pyannote diarization wrapper
    pipeline.chunking_stage        - split_long_audio invocation
    pipeline.transcription_stage   - streaming/chunked transcription dispatcher
    pipeline.outputs_stage         - markdown transcript + word splits + LLM
    pipeline.cleanup_stage         - post-run temp file removal
    pipeline.quality_stage         - heuristic quality checker
    pipeline.persistence_stage     - SQLite history writer
    pipeline.orchestrator          - process_one + process_batch
    pipeline.context               - RunContext dataclass
    pipeline.cleanup               - chunk-WAV cleanup helpers
    pipeline.reporting             - speaker-distribution histogram
"""

from __future__ import annotations

# Public API
from speakerscribe.pipeline.cleanup import _cleanup_chunks, _purge_stale_chunks
from speakerscribe.pipeline.context import RunContext
from speakerscribe.pipeline.diarization_stage import _maybe_diarize
from speakerscribe.pipeline.orchestrator import process_batch, process_one
from speakerscribe.pipeline.preflight import preflight_check
from speakerscribe.pipeline.reporting import report_speaker_distribution
from speakerscribe.pipeline.wav import _maybe_extract_wav, _should_chunk_audio

__all__ = [
    "RunContext",
    "preflight_check",
    "process_batch",
    "process_one",
    "report_speaker_distribution",
]
