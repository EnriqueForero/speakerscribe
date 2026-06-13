"""Centralized configuration with Pydantic v2 validation.

Single source of truth for pipeline parameters. Zero magic numbers in
business logic — every tunable lives here.
"""

from __future__ import annotations

import os
import tempfile
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
HashMode = Literal["fast", "full"]
SpeakerAssignment = Literal["segment", "word"]
FillerMode = Literal["off", "safe", "aggressive"]

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

# Whisper conditions decoding on (at most) the trailing ~224 tokens of the
# initial prompt. Anything beyond that is silently ignored by the model.
# Reference: openai/whisper decoding (n_ctx // 2 - 1 = 223 prompt tokens).
WHISPER_PROMPT_TOKEN_LIMIT = 224

# ─── Filler words by language ──────────────────────────────────────
# SAFE fillers carry no semantic load in any context ("eh", "um", "mmm").
# AGGRESSIVE fillers are discourse markers that CAN be real answers
# ("sí", "no", "claro", "ok", "right"): dropping them from an interview
# transcript can delete the answer to a yes/no question. The readable
# .transcript.md filter uses one of these sets depending on
# `TranscriptionConfig.remove_fillers`; the .json/.srt always keep ALL
# segments for traceability.
FILLERS_SAFE: dict[str, frozenset[str]] = {
    "es": frozenset({"eh", "ehm", "em", "mmm", "mhm", "ajá", "aja"}),
    "en": frozenset({"uh", "um", "uhm", "ah", "er", "erm", "hmm", "mhm"}),
    "pt": frozenset({"ahn", "é", "hum"}),
    "fr": frozenset({"euh", "ben", "bah"}),
}

FILLERS_AGGRESSIVE: dict[str, frozenset[str]] = {
    "es": FILLERS_SAFE["es"]
    | frozenset({"este", "pues", "o sea", "como que", "bueno", "claro", "sí", "no", "ok", "vale"}),
    "en": FILLERS_SAFE["en"]
    | frozenset({"yeah", "right", "ok", "okay", "you know", "i mean", "like"}),
    "pt": FILLERS_SAFE["pt"] | frozenset({"tipo", "né", "então", "bom", "sabe"}),
    "fr": FILLERS_SAFE["fr"] | frozenset({"voilà", "quoi", "alors", "donc"}),
}

# Backward-compatible alias (pre-0.3 behavior == today's "aggressive" set).
# Deprecated: import FILLERS_SAFE / FILLERS_AGGRESSIVE instead.
FILLER_WORDS: dict[str, frozenset[str]] = FILLERS_AGGRESSIVE


def fillers_for(language: str | None, mode: FillerMode) -> frozenset[str]:
    """Return the filler set for a language under the given filter mode.

    Args:
        language: Two-letter language code ("es", "en", ...) or None.
        mode: "off" returns an empty set; "safe" returns hesitation sounds
            only; "aggressive" additionally includes discourse markers.

    Returns:
        Frozenset of filler tokens (possibly empty).
    """
    if mode == "off" or not language:
        return frozenset()
    table = FILLERS_SAFE if mode == "safe" else FILLERS_AGGRESSIVE
    return table.get(language, frozenset())


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
        batch_size: Batched inference size for faster-whisper's
            BatchedInferencePipeline. 8 is the sweet spot from the official
            benchmark (~3.7x faster than sequential on large models). 1 =
            exact sequential path (bit-parity with pre-0.3 outputs). On CUDA
            OOM the pipeline automatically halves the batch size down to 1.
        beam_size: 1=greedy (fastest), 5=balanced, 10=highest quality.
        language: None=auto-detect; "en", "es", "fr", etc. force a language.
        initial_prompt: Glossary string to bias decoding (proper nouns, jargon).
            Whisper only conditions on the trailing ~224 tokens
            (`WHISPER_PROMPT_TOKEN_LIMIT`); a longer prompt is not an error
            but the excess head is ignored by the model — a warning is logged.
            Per-file overrides: a file `<audio_stem>.prompt.txt` next to the
            source media replaces this global prompt for that single file.
        use_vad: Enable Silero Voice Activity Detection (skip silences).
            Note: the batched path (batch_size > 1) always uses VAD-based
            segmentation internally regardless of this flag.
        vad_min_silence_ms: Silences >= N ms split segments.
        word_timestamps: Emit per-word timestamps in the .json. Slows
            decoding ~10-15 percent. Independently of this flag, word
            timestamps are enabled internally when `speaker_assignment="word"`
            (they are required for word-level attribution) or when
            `hallucination_silence_threshold` is set.
        condition_on_previous_text: Whisper conditions each window on the
            previous output. True = faster-whisper default. False is
            recommended for noisy multi-speaker meetings: it stops
            hallucination loops from propagating across windows.
        temperature: Decoding temperature or fallback schedule (faster-whisper
            default `(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)`).
        compression_ratio_threshold: Treat decoding as failed when gzip
            compression ratio exceeds this (default 2.4 — repetitive text
            compresses suspiciously well).
        log_prob_threshold: Treat decoding as failed below this average
            log-probability (default -1.0).
        no_speech_threshold: Probability above which a window is considered
            silence (default 0.6).
        hallucination_silence_threshold: When set (seconds), skip silent
            periods longer than this if a hallucination is detected. Requires
            word timestamps; they are auto-enabled when this is set.
        repetition_penalty: Penalty > 1.0 discourages token repetition
            (faster-whisper default 1.0 = off).
        no_repeat_ngram_size: Prevent exact n-gram repeats of this size
            (faster-whisper default 0 = off).
        enable_diarization: If False, transcription only.
        speaker_assignment: "word" (default) assigns a speaker per word and
            re-segments at speaker changes (WhisperX-style; correct when a
            Whisper segment spans a turn change). "segment" reproduces the
            pre-0.3 behavior: one speaker per Whisper segment by maximum
            overlap (faster; systematic attribution errors on fast turns).
        hf_token: HuggingFace access token. None -> Colab Secrets, then env var.
        num_speakers: Pin exact speaker count if known. Mutually exclusive with
            min/max_speakers.
        min_speakers / max_speakers: Range when count is unknown.
        diarization_model: HuggingFace model id. Default is the recommended
            community-1 release.
        gap_max_s_transcript: A gap of more than N seconds within a single
            speaker opens a new turn block in the transcript.
        generate_transcript_md: If False, skip Markdown transcript generation.
        remove_fillers: Filler filtering mode for the readable .transcript.md.
            "safe" (default) drops pure hesitation sounds ("eh", "um", "mmm").
            "aggressive" additionally drops discourse markers ("sí", "no",
            "claro", "ok") — WARNING: those can be real answers in interviews.
            "off" disables the filter. Booleans are accepted for backward
            compatibility (True -> "safe", False -> "off"). The .json and
            .srt always keep ALL segments.
        words_per_split: Soft cap of words per split file (for chunking large
            transcripts before downstream LLM processing).
        produce_unified_for_llm: If True, additionally write a single
            `<base>_full_for_llm.txt` with the entire transcript (no chunking),
            convenient for LLMs with large context windows.
        long_audio_threshold_min: DEPRECATED. Audios longer than this are
            split into chunks for transcription. Default 0 = disabled: the
            batched faster-whisper path handles long audio natively (its VAD
            partitions without cutting words), making external chunking both
            unnecessary and less accurate at boundaries. Set > 0 only if you
            explicitly need the legacy chunked path; a DeprecationWarning is
            emitted.
        chunk_duration_min: Target duration of each legacy chunk in minutes.
        chunk_overlap_s: Overlap between consecutive legacy chunks in seconds.
            Default raised 5 -> 30 in 0.3: a 5 s overlap truncated sentences
            at boundaries.
        force_reprocess: If True, ignore existing outputs and re-run.
        evaluate_quality: If True, run heuristic quality checks post-run.
        auto_retry_on_critical: If True (default), a run flagged CRITICAL for
            REPETITIONS or HIGH_WPM (hallucination loop signatures) is retried
            ONCE with anti-loop decoding (`condition_on_previous_text=False`,
            `repetition_penalty=1.15`, `no_repeat_ngram_size=3`). Both
            attempts are recorded in the runs ledger; the attempt with fewer
            critical flags is kept.
        hash_mode: Idempotency signature mode. "fast" (default) hashes
            size + first 8 MB + last 8 MB — O(16 MB) instead of reading
            multi-GB files through Drive/FUSE. "full" is the pre-0.3 complete
            SHA-256. Lookups transparently fall back to the full hash once
            per file to recognize runs recorded before 0.3.
        enable_runs_db: If True (default), persist per-run metadata in an
            append-only JSON-Lines ledger under the workspace
            (`_runs.jsonl`) for robust idempotency by file content. Legacy
            SQLite histories (`_runs.db`) remain readable.
        streaming_jsonl: If True, additionally write each segment to
            `<base>.segments.jsonl` as it is produced. Useful for very long
            audios where loading the full .json into memory is undesirable.
        sample_rate: 16000 Hz is standard for both Whisper and pyannote.
        ffmpeg_timeout_s: Hard timeout for ffmpeg/ffprobe subprocesses.
            0 (default) = auto: 120 s + 2x the media duration.
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
    batch_size: Annotated[int, Field(ge=1, le=32)] = 8

    # ── Whisper decoding ───────────────────────────────────────────
    beam_size: Annotated[int, Field(ge=1, le=10)] = 5
    language: str | None = None
    initial_prompt: Annotated[str | None, Field(max_length=4000)] = None

    # ── Anti-hallucination controls (defaults == faster-whisper) ───
    condition_on_previous_text: bool = True
    temperature: float | tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    compression_ratio_threshold: Annotated[float, Field(gt=0.0)] = 2.4
    log_prob_threshold: float = -1.0
    no_speech_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.6
    hallucination_silence_threshold: Annotated[float | None, Field(gt=0.0)] = None
    repetition_penalty: Annotated[float, Field(gt=0.0)] = 1.0
    no_repeat_ngram_size: Annotated[int, Field(ge=0)] = 0

    # ── VAD ────────────────────────────────────────────────────────
    use_vad: bool = True
    vad_min_silence_ms: Annotated[int, Field(ge=100, le=5000)] = 500
    word_timestamps: bool = False

    # ── Diarization ────────────────────────────────────────────────
    enable_diarization: bool = True
    speaker_assignment: SpeakerAssignment = "word"
    hf_token: str | None = None
    num_speakers: Annotated[int | None, Field(ge=1, le=20)] = None
    min_speakers: Annotated[int | None, Field(ge=1, le=20)] = None
    max_speakers: Annotated[int | None, Field(ge=1, le=20)] = None
    diarization_model: str = "pyannote/speaker-diarization-community-1"

    # ── Transcript output ──────────────────────────────────────────
    gap_max_s_transcript: Annotated[float, Field(ge=0.0, le=60.0)] = 2.0
    generate_transcript_md: bool = True
    remove_fillers: FillerMode = "safe"

    # ── Output / processing ────────────────────────────────────────
    words_per_split: Annotated[int, Field(ge=100, le=10_000)] = 1950
    produce_unified_for_llm: bool = True
    extract_temp_wav: bool = True
    delete_temp_wav: bool = True
    sample_rate: Annotated[int, Field(ge=8_000, le=48_000)] = 16_000
    ffmpeg_timeout_s: Annotated[int, Field(ge=0)] = 0

    # ── Long-audio chunking (DEPRECATED — see attribute docs) ──────
    long_audio_threshold_min: Annotated[int, Field(ge=0, le=600)] = 0
    chunk_duration_min: Annotated[int, Field(ge=5, le=120)] = 30
    chunk_overlap_s: Annotated[int, Field(ge=0, le=120)] = 30
    delete_chunk_wavs: bool = True

    # ── Streaming output ───────────────────────────────────────────
    streaming_jsonl: bool = False

    # ── Reprocessing ───────────────────────────────────────────────
    force_reprocess: bool = False

    # ── Quality checks ─────────────────────────────────────────────
    evaluate_quality: bool = True
    auto_retry_on_critical: bool = True

    # ── Persistence (ON by default for robust idempotency) ─────────
    enable_runs_db: bool = True
    hash_mode: HashMode = "fast"

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

    @field_validator("remove_fillers", mode="before")
    @classmethod
    def coerce_remove_fillers(cls, v: object) -> object:
        """Accept legacy booleans: True -> 'safe', False -> 'off'."""
        if isinstance(v, bool):
            return "safe" if v else "off"
        return v

    @field_validator("temperature", mode="before")
    @classmethod
    def coerce_temperature(cls, v: object) -> object:
        """Accept lists from JSON configs and coerce to tuple."""
        if isinstance(v, list):
            return tuple(v)
        return v

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

    def effective_word_timestamps(self, has_diarization: bool) -> bool:
        """Whether word timestamps must be requested from Whisper at runtime.

        True when the user asked for them, when word-level speaker assignment
        will run (it needs per-word times), or when
        `hallucination_silence_threshold` is set (faster-whisper requires
        word timestamps for that feature).

        Args:
            has_diarization: Whether diarization turns are actually available
                for the current file (not just enabled in config).

        Returns:
            Effective boolean to pass as ``word_timestamps``.
        """
        if self.word_timestamps:
            return True
        if self.hallucination_silence_threshold is not None:
            return True
        return has_diarization and self.speaker_assignment == "word"

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
            from google.colab import userdata

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
        without re-configuring the pipeline between runs. The content is NOT
        truncated here: Whisper itself conditions only on the trailing
        ~224 tokens (`WHISPER_PROMPT_TOKEN_LIMIT`); the transcription layer
        logs a warning when a prompt likely exceeds that budget.

        Args:
            media_path: Path to the source media file.

        Returns:
            The effective prompt string, or None.
        """
        prompt_file = media_path.with_suffix(".prompt.txt")
        if prompt_file.exists() and prompt_file.is_file():
            content = prompt_file.read_text(encoding="utf-8").strip()
            if content:
                return content
        return self.initial_prompt


# ─── WorkspacePaths ────────────────────────────────────────────────
class WorkspacePaths(BaseModel):
    """Single source of truth for all filesystem paths used by the pipeline.

    Every function that reads or writes files takes this object. Never use
    raw path strings inside functions.

    Durable vs scratch storage:
        `workspace` (typically Google Drive) keeps everything that must
        survive the session: inputs, outputs, diarization cache, logs and
        the runs ledger. `scratch` (local disk — `/content` on Colab) keeps
        high-churn temporaries: extracted WAVs and legacy chunk WAVs.
        Drive is mounted via FUSE with high latency and unreliable locking;
        writing a 115 MB/h WAV there and reading it back twice (pyannote +
        Whisper) costs minutes per file for zero durability benefit.

    Folder layout under `workspace`:
        data/             — input audio/video files (you place them here)
        transcripts/      — .txt, .srt, .json, .transcript.md outputs
        splits/           — chunked .txt outputs for downstream LLM use
        _diar_cache/      — cached pyannote diarization results (durable!)
        _logs/            — rotating loguru log files
        _runs.jsonl       — append-only runs ledger (idempotency)
        _runs.db          — legacy SQLite history (read-only since 0.3)

    Folder layout under `scratch` (auto-resolved when None):
        _audio_temp/      — temporary 16 kHz mono WAVs
        _audio_chunks/    — legacy chunked WAVs

    Attributes:
        workspace: Root path of the project (durable). Accepts str or Path.
        scratch: Root for temporaries (fast local disk). None -> auto:
            `/content/ss_scratch` when `/content` exists (Colab), else
            `<system tmp>/speakerscribe_scratch`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    workspace: str
    scratch: str | None = None

    @field_validator("workspace", "scratch", mode="before")
    @classmethod
    def coerce_path_to_str(cls, v: object) -> object:
        """Accept pathlib.Path transparently."""
        if isinstance(v, Path):
            return str(v)
        return v

    @property
    def base(self) -> Path:
        return Path(self.workspace)

    @property
    def scratch_base(self) -> Path:
        """Resolved scratch root (local disk). See class docstring."""
        if self.scratch:
            return Path(self.scratch)
        if Path("/content").exists():  # Google Colab local NVMe
            return Path("/content/ss_scratch")
        return Path(tempfile.gettempdir()) / "speakerscribe_scratch"

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
        return self.scratch_base / "_audio_temp"

    @property
    def audio_chunks(self) -> Path:
        return self.scratch_base / "_audio_chunks"

    @property
    def diar_cache(self) -> Path:
        return self.base / "_diar_cache"

    @property
    def logs(self) -> Path:
        return self.base / "_logs"

    @property
    def db_path(self) -> Path:
        """Legacy SQLite history (read-only fallback since 0.3)."""
        return self.base / "_runs.db"

    @property
    def ledger_path(self) -> Path:
        """Append-only JSON-Lines runs ledger (primary since 0.3)."""
        return self.base / "_runs.jsonl"

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
