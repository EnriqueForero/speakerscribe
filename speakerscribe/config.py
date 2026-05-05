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

# ─── Filler words by language ──────────────────────────────────────
# Segments whose stripped text matches EXACTLY one of these (case-insensitive,
# punctuation-stripped) are dropped from the readable .transcript.md when
# `remove_fillers=True`. The original .json keeps everything for traceability.
FILLER_WORDS: dict[str, frozenset[str]] = {
    "es": frozenset(
        {
            "eh",
            "ehm",
            "em",
            "mmm",
            "mhm",
            "ajá",
            "aja",
            "este",
            "pues",
            "o sea",
            "como que",
            "bueno",
            "claro",
            "sí",
            "no",
            "ok",
            "vale",
        }
    ),
    "en": frozenset(
        {
            "uh",
            "um",
            "uhm",
            "ah",
            "er",
            "erm",
            "hmm",
            "mhm",
            "yeah",
            "right",
            "ok",
            "okay",
            "you know",
            "i mean",
            "like",
        }
    ),
    "pt": frozenset({"é", "tipo", "né", "então", "bom", "sabe", "ahn"}),
    "fr": frozenset({"euh", "ben", "bah", "voilà", "quoi", "alors", "donc"}),
}


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
            Per-file overrides: a file `<audio_stem>.prompt.txt` next to the
            source media replaces this global prompt for that single file.
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
        remove_fillers: If True, the readable .transcript.md drops segments
            whose only content is filler words ("eh", "um", ...) for the
            detected language. The .json and .srt keep ALL segments (audit).
        words_per_split: Soft cap of words per split file (for chunking large
            transcripts before downstream LLM processing).
        produce_unified_for_llm: If True, additionally write a single
            `<base>_full_for_llm.txt` with the entire transcript (no chunking),
            convenient for LLMs with large context windows.
        long_audio_threshold_min: Audios longer than this are split into chunks
            for transcription. Set to 0 to disable splitting. Default 120 min.
        chunk_duration_min: Target duration of each chunk in minutes. Only
            applies when the audio exceeds `long_audio_threshold_min`.
        chunk_overlap_s: Overlap between consecutive chunks in seconds. Reduces
            risk of cutting words across boundaries.
        force_reprocess: If True, ignore existing outputs and re-run.
        evaluate_quality: If True, run heuristic quality checks post-run.
        enable_runs_db: If True, persist per-run metadata in a local SQLite
            database under the workspace. ON by default for robust idempotency
            by file hash; the user usually doesn't need to interact with it.
        streaming_jsonl: If True, additionally write each segment to
            `<base>.segments.jsonl` as it is produced. Useful for very long
            audios where loading the full .json into memory is undesirable.
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
    remove_fillers: bool = True

    # ── Output / processing ────────────────────────────────────────
    words_per_split: Annotated[int, Field(ge=100, le=10_000)] = 1950
    produce_unified_for_llm: bool = True
    extract_temp_wav: bool = True
    delete_temp_wav: bool = True
    sample_rate: Annotated[int, Field(ge=8_000, le=48_000)] = 16_000

    # ── Long-audio chunking ────────────────────────────────────────
    long_audio_threshold_min: Annotated[int, Field(ge=0, le=600)] = 120
    chunk_duration_min: Annotated[int, Field(ge=5, le=120)] = 30
    chunk_overlap_s: Annotated[int, Field(ge=0, le=120)] = 5
    delete_chunk_wavs: bool = True

    # ── Streaming output ───────────────────────────────────────────
    streaming_jsonl: bool = False

    # ── Idempotency ────────────────────────────────────────────────
    force_reprocess: bool = False

    # ── Quality checks ─────────────────────────────────────────────
    evaluate_quality: bool = True

    # ── Persistence (ON by default for robust idempotency) ─────────
    enable_runs_db: bool = True

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

    @model_validator(mode="after")
    def validate_chunking(self) -> TranscriptionConfig:
        """chunk_duration_min must be greater than chunk_overlap_s/60."""
        if self.long_audio_threshold_min > 0:
            chunk_s = self.chunk_duration_min * 60
            if chunk_s <= self.chunk_overlap_s:
                raise ValueError(
                    f"chunk_duration_min ({self.chunk_duration_min} min = {chunk_s}s) "
                    f"must be greater than chunk_overlap_s ({self.chunk_overlap_s}s)."
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

    def resolve_initial_prompt(self, media_path: Path) -> str | None:
        """Resolve the effective initial prompt for a given media file.

        Lookup order:
            1. A file `<media_stem>.prompt.txt` next to the source media.
            2. The global `initial_prompt` from this config.
            3. None.

        Per-file prompts allow having a tailored glossary for each meeting
        without re-configuring the pipeline between runs.

        Args:
            media_path: Path to the source media file.

        Returns:
            The effective prompt string, or None.
        """
        prompt_file = media_path.with_suffix(".prompt.txt")
        if prompt_file.exists() and prompt_file.is_file():
            content = prompt_file.read_text(encoding="utf-8").strip()
            if content:
                # Pydantic validates max_length=500 on construction; enforce
                # the same limit here for per-file prompts.
                return content[:500]
        return self.initial_prompt


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
        _audio_chunks/    — chunked WAVs for long-audio splitting
        _diar_cache/      — cached pyannote diarization results
        _logs/            — rotating loguru log files
        _runs.db          — SQLite history (created when enable_runs_db=True)

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
    def audio_chunks(self) -> Path:
        return self.base / "_audio_chunks"

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
            self.audio_chunks,
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
