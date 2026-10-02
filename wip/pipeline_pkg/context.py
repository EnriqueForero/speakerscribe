"""Run-scoped state container for the speakerscribe pipeline.

A `RunContext` is the single mutable object passed between pipeline stages.
It carries everything a stage might need (inputs, intermediate artifacts,
output paths, telemetry) so stages do not need to share state through
parameter explosion or global side effects.

Stages mutate the context. The orchestrator (`process_one`) owns its
lifecycle. The context is NOT serialized to disk — only the `metadata`
dict inside it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from speakerscribe.audio import AudioChunk
from speakerscribe.config import TranscriptionConfig, WorkspacePaths
from speakerscribe.quality import QualityReport

if TYPE_CHECKING:
    from faster_whisper import WhisperModel


@dataclass
class RunContext:
    """All state for a single `process_one` invocation.

    Construct with the four required inputs; stages populate the rest.

    Args:
        media_path: Input media file the user wants transcribed.
        paths: Workspace layout (where to read/write).
        config: Pydantic-validated transcription config.
        model: Pre-loaded faster-whisper model (None until loaded).
    """

    # ── Inputs (immutable after construction) ──────────────────────
    media_path: Path
    paths: WorkspacePaths
    config: TranscriptionConfig
    model: WhisperModel | None = None

    # ── Audio stage outputs ────────────────────────────────────────
    audio_for_model: Path | None = None
    """Path that will be fed to Whisper. Equals `media_path` for already-audio
    inputs, or the extracted WAV otherwise."""

    wav_path: Path | None = None
    """Full WAV path (always populated for video inputs; equal to
    `audio_for_model` for audio inputs)."""

    chunks: list[AudioChunk] = field(default_factory=list)
    chunked: bool = False
    total_duration_s: float = 0.0

    # ── Diarization stage output ───────────────────────────────────
    diar_turns: list[dict[str, Any]] | None = None
    """List of {start, end, speaker} dicts (sorted by start), or None when
    diarization was skipped or failed."""

    # ── Transcription stage outputs ────────────────────────────────
    metadata: dict[str, Any] = field(default_factory=dict)
    """The metadata dict that will be serialized to `.json`. Includes
    segments, timings, package version, etc."""

    # ── Output paths (derived from media_path + workspace) ─────────
    output_txt: Path | None = None
    output_srt: Path | None = None
    output_json: Path | None = None
    output_md: Path | None = None

    # ── Quality stage output ───────────────────────────────────────
    quality_report: QualityReport | None = None

    # ── Persistence / idempotency ──────────────────────────────────
    file_hash: str | None = None
    skipped_due_to_cache: bool = False

    # ── Telemetry ──────────────────────────────────────────────────
    timings: dict[str, float] = field(default_factory=dict)
    t_start: float = 0.0

    def compute_output_paths(self) -> None:
        """Populate `output_txt/srt/json/md` from `media_path.stem` + model name."""
        stem = self.media_path.stem
        suffix = f"_{self.config.model}"
        self.output_txt = self.paths.transcripts / f"{stem}{suffix}.txt"
        self.output_srt = self.paths.transcripts / f"{stem}{suffix}.srt"
        self.output_json = self.paths.transcripts / f"{stem}{suffix}.json"
        self.output_md = self.paths.transcripts / f"{stem}{suffix}.transcript.md"
