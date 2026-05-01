"""Centralized configuration with Pydantic v2 validation.

Single source of truth for pipeline parameters. Zero magic numbers in
business logic — every tunable lives here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# ─── Public constants ──────────────────────────────────────────────
SPK_NO_DIARIZATION = "(no diarization)"
"""Label used when diarization is disabled or failed."""

SPK_NO_OVERLAP = "SPEAKER_NO_OVERLAP"
"""Label used when diarization ran but a Whisper segment did not overlap any turn."""

# Whisper models supported by faster-whisper
WhisperModelName = Literal[
    "tiny",
    "base",
    "small",
    "medium",
    "large-v2",
    "large-v3",
    "large-v3-turbo",
    "turbo",
]

DeviceType = Literal["auto", "cuda", "cpu"]
ComputeType = Literal["auto", "float16", "int8", "int8_float16", "float32"]

# Estimated minimum VRAM per model (GB) — empirical on NVIDIA T4
MIN_VRAM_BY_MODEL: dict[str, float] = {
    "tiny": 1.5,
    "base": 1.8,
    "small": 2.5,
    "medium": 3.5,
    "large-v2": 5.0,
    "large-v3": 5.0,
    "large-v3-turbo": 3.5,
    "turbo": 3.5,
}

# Processable media extensions
MEDIA_EXTENSIONS: tuple[str, ...] = (
    ".mp4",
    ".mp3",
    ".wav",
    ".m4a",
    ".mkv",
    ".aac",
    ".flac",
    ".ogg",
    ".webm",
)


# ─── TranscriptionConfig ───────────────────────────────────────────
class TranscriptionConfig(BaseModel):
    """Configuration for the transcription + diarization pipeline.

    All values are validated at construction time. If something is invalid,
    construction fails immediately rather than 30 minutes into a run.

    Attributes:
        model: Whisper model name. `large-v3-turbo` is recommended (~3x faster
            than large-v3 with comparable accuracy).
        device: "auto" detects CUDA/CPU; "cuda" forces GPU; "cpu" forces CPU.
        compute_type: "auto" -> float16 on GPU / int8 on CPU.
        beam_size: 1=greedy (fastest), 5=balanced, 10=highest quality.
        language: None=auto-detect; "en", "es", "fr", etc. force a language.
        initial_prompt: Glossary string to bias decoding (proper nouns, jargon).
        use_vad: Enable Silero Voice Activity Detection (skip silences).
        vad_min_silence_ms: Silences >= N ms split segments.
        word_timestamps: Emit per-word timestamps. Slows decoding ~10-15 percent.
        enable_diarization: If False, transcription only.
        hf_token: HuggingFace access token. None -> Colab Secrets, then env var.
        num_speakers: Pin exact speaker count if known. Mutually exclusive with
            min/max_speakers.
        min_speakers / max_speakers: Range when count is unknown.
        diarization_model: HuggingFace model id. Default is the recommended
            community-1 release.
        gap_max_s_transcript: A gap of more than N seconds within a single
            speaker opens a new turn block in the transcript.
        generate_transcript_md: If False, skip Markdown transcript generation.
        words_per_split: Soft cap of words per split file (for chunking large
            transcripts before downstream LLM processing).
        force_reprocess: If True, ignore existing outputs and re-run.
        evaluate_quality: If True, run heuristic quality checks post-run.
        enable_runs_db: If True, persist per-run metadata in a local SQLite
            database under the workspace. Off by default.
        sample_rate: 16000 Hz is standard for both Whisper and pyannote.
        disk_margin_factor: Required free space = total_input_size_mb * factor.
        disk_margin_min_mb: Hard floor for required free space (MB).
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        frozen=False,
    )

    # ── Whisper model ──────────────────────────────────────────────
    model: WhisperModelName = "large-v3"
    device: DeviceType = "auto"
    compute_type: ComputeType = "auto"

    # ── Whisper decoding ───────────────────────────────────────────
    beam_size: Annotated[int, Field(ge=1, le=10)] = 5
    language: str | None = None
    initial_prompt: Annotated[str | None, Field(max_length=500)] = None

    # ── VAD ────────────────────────────────────────────────────────
    use_vad: bool = True
    vad_min_silence_ms: Annotated[int, Field(ge=100, le=5000)] = 500
    word_timestamps: bool = False

    # ── Diarization ────────────────────────────────────────────────
    enable_diarization: bool = True
    hf_token: str | None = None
    num_speakers: Annotated[int | None, Field(ge=1, le=20)] = None
    min_speakers: Annotated[int | None, Field(ge=1, le=20)] = None
    max_speakers: Annotated[int | None, Field(ge=1, le=20)] = None
    diarization_model: str = "pyannote/speaker-diarization-community-1"

    # ── Transcript output ──────────────────────────────────────────
    gap_max_s_transcript: Annotated[float, Field(ge=0.0, le=60.0)] = 2.0
    generate_transcript_md: bool = True

    # ── Output / processing ────────────────────────────────────────
    words_per_split: Annotated[int, Field(ge=100, le=10_000)] = 1950
    extract_temp_wav: bool = True
    delete_temp_wav: bool = True
    sample_rate: Annotated[int, Field(ge=8_000, le=48_000)] = 16_000

    # ── Idempotency ────────────────────────────────────────────────
    force_reprocess: bool = False

    # ── Quality checks ─────────────────────────────────────────────
    evaluate_quality: bool = True

    # ── Persistence (opt-in) ───────────────────────────────────────
    enable_runs_db: bool = False

    # ── Pre-flight ─────────────────────────────────────────────────
    disk_margin_factor: Annotated[float, Field(ge=0.1, le=10.0)] = 0.5
    disk_margin_min_mb: Annotated[int, Field(ge=100)] = 500

    @field_validator("language")
    @classmethod
    def normalize_language(cls, v: str | None) -> str | None:
        """Lowercase and strip the language code."""
        if v is None:
            return None
        return v.lower().strip()

    @model_validator(mode="after")
    def validate_speakers(self) -> TranscriptionConfig:
        """num_speakers is mutually exclusive with min/max_speakers."""
        if self.num_speakers is not None and (
            self.min_speakers is not None or self.max_speakers is not None
        ):
            raise ValueError(
                "num_speakers is mutually exclusive with min_speakers/max_speakers. "
                "Use one or the other, not both."
            )
        if (
            self.min_speakers is not None
            and self.max_speakers is not None
            and self.min_speakers > self.max_speakers
        ):
            raise ValueError(
                f"min_speakers ({self.min_speakers}) > max_speakers ({self.max_speakers})"
            )
        return self

    def resolve_device(self) -> tuple[str, str]:
        """Resolve 'auto' device and compute_type to concrete runtime values.

        Returns:
            Tuple (device, compute_type) with concrete values.
        """
        # Lazy import: torch is heavy and may be missing at construction time
        try:
            import torch

            cuda_available = torch.cuda.is_available()
        except ImportError:
            cuda_available = False

        device = ("cuda" if cuda_available else "cpu") if self.device == "auto" else self.device
        compute_type = (
            ("float16" if device == "cuda" else "int8")
            if self.compute_type == "auto"
            else self.compute_type
        )
        return device, compute_type

    def resolve_hf_token(self) -> str | None:
        """Locate the HuggingFace token in this order:
        explicit config -> Colab Secrets -> HF_TOKEN env var.

        Returns:
            Token string or None if no source is available.
        """
        if self.hf_token:
            return self.hf_token
        # Colab Secrets
        try:
            from google.colab import userdata  # type: ignore[import-not-found]

            token = userdata.get("HF_TOKEN")
            if token:
                return str(token)
        except ImportError:
            pass
        except Exception:
            # google.colab.userdata may raise SecretNotFoundError or similar
            pass
        # Env var fallback
        return os.environ.get("HF_TOKEN")


# ─── WorkspacePaths ────────────────────────────────────────────────
class WorkspacePaths(BaseModel):
    """Single source of truth for all filesystem paths used by the pipeline.

    Every function that reads or writes files takes this object. Never use
    raw path strings inside functions.

    Folder layout under `workspace`:
        data/             — input audio/video files (you place them here)
        transcripts/      — .txt, .srt, .json, .transcript.md outputs
        splits/           — chunked .txt outputs for downstream LLM use
        _audio_temp/      — temporary 16 kHz mono WAVs (auto-deleted by default)
        _diar_cache/      — cached pyannote diarization results
        _logs/            — rotating loguru log files
        _runs.db          — optional SQLite history (only if enable_runs_db=True)

    Attributes:
        workspace: Root path of the project. Subdirectories are created on demand.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    workspace: str

    @property
    def base(self) -> Path:
        return Path(self.workspace)

    @property
    def data(self) -> Path:
        return self.base / "data"

    @property
    def transcripts(self) -> Path:
        return self.base / "transcripts"

    @property
    def splits(self) -> Path:
        return self.base / "splits"

    @property
    def audio_tmp(self) -> Path:
        return self.base / "_audio_temp"

    @property
    def diar_cache(self) -> Path:
        return self.base / "_diar_cache"

    @property
    def logs(self) -> Path:
        return self.base / "_logs"

    @property
    def db_path(self) -> Path:
        return self.base / "_runs.db"

    def create_directories(self) -> None:
        """Create all project subdirectories if they do not exist."""
        for path in [
            self.data,
            self.transcripts,
            self.splits,
            self.audio_tmp,
            self.diar_cache,
            self.logs,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def list_media_files(
        self,
        extensions: tuple[str, ...] = MEDIA_EXTENSIONS,
    ) -> list[Path]:
        """List processable files from the `data/` folder. Top-level only.

        Args:
            extensions: Accepted extensions (with leading dot, lowercase).

        Returns:
            Sorted list of absolute Paths.

        Raises:
            FileNotFoundError: If the `data/` folder does not exist.
        """
        if not self.data.exists():
            raise FileNotFoundError(
                f"Folder not found: {self.data}. Create it and place your audio/video files there."
            )
        return sorted(
            p for p in self.data.iterdir() if p.is_file() and p.suffix.lower() in extensions
        )
