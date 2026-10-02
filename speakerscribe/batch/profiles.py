"""Engine and presentation profiles: what costs GPU vs what is re-rendered on CPU.

* MOTOR profile — parameters that change the transcription itself (model,
  language, beam, batch, VAD, diarization, glossary). A change means a new
  GPU pass, so it is governed by ``profile_change_policy``.
* PRESENTATION profile — parameters that only change rendering (header,
  timestamps, active deliverables, speaker renames). A change is applied by
  re-rendering from the master JSON, never with GPU.

Change from notebook v5: the motor profile no longer embeds the library
version. v5 stored ``speakerscribe_version``, so a routine library upgrade
silently changed every profile id. It now stores `ENGINE_SEMANTICS`, bumped
by hand only when an upgrade changes transcription output on purpose.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

from speakerscribe.batch.fsio import canonical_hash, hash_file
from speakerscribe.batch.identity import SourceInfo
from speakerscribe.batch.renderers import DELIVERABLES, RENDER_SCHEMA, SPLITS_KEY, RenderOptions
from speakerscribe.batch.settings import BatchSettings

if TYPE_CHECKING:
    from speakerscribe.config import TranscriptionConfig

ENGINE_SEMANTICS = 1
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
DIAR_CACHE_SCHEMA = 1


def engine_config(settings: BatchSettings) -> TranscriptionConfig:
    """`TranscriptionConfig` for batch use (same engine settings as v5).

    The batch owns idempotency (journal), so the library's own ledger is
    disabled and ``force_reprocess`` is always on inside a job workspace.
    """
    from speakerscribe.config import TranscriptionConfig

    return TranscriptionConfig(
        model=settings.model,
        language=settings.language,
        device="cuda",
        compute_type="float16",
        beam_size=settings.beam_size,
        batch_size=settings.batch_size,
        initial_prompt=(settings.glossary.strip() or None),
        use_vad=True,
        vad_min_silence_ms=settings.vad_min_silence_ms,
        condition_on_previous_text=not settings.anti_hallucination,
        enable_diarization=True,
        speaker_assignment="word",
        diarization_model=DIARIZATION_MODEL,
        num_speakers=settings.none_if_zero(settings.num_speakers),
        min_speakers=settings.none_if_zero(settings.min_speakers),
        max_speakers=settings.none_if_zero(settings.max_speakers),
        word_timestamps=False,
        generate_transcript_md=False,
        remove_fillers="off",
        produce_unified_for_llm=False,
        streaming_jsonl=False,
        long_audio_threshold_min=0,
        extract_temp_wav=True,
        delete_temp_wav=True,
        delete_chunk_wavs=True,
        evaluate_quality=True,
        auto_retry_on_critical=True,
        enable_runs_db=False,
        force_reprocess=True,
        hash_mode="fast",
    )


def prompt_digest(info: SourceInfo, glossary: str) -> str:
    """Fingerprint of the effective glossary (per-file sidecar or global)."""
    if info.prompt_path is not None:
        return hash_file(info.prompt_path, "full")
    return hashlib.sha256(glossary.strip().encode()).hexdigest()


def motor_profile(config: TranscriptionConfig, prompt_sha256: str) -> dict[str, Any]:
    """Semantic identity of a transcription (what requires GPU if it changes).

    Excludes environment versions (torch, CUDA), presentation, and the
    library version (see module docstring).
    """
    profile = {
        "engine": "speakerscribe",
        "engine_semantics": ENGINE_SEMANTICS,
        "asr_model": config.model,
        "language": config.language,
        "device": config.device,
        "compute_type": config.compute_type,
        "beam_size": config.beam_size,
        "batch_size": config.batch_size,
        "use_vad": config.use_vad,
        "vad_min_silence_ms": config.vad_min_silence_ms,
        "condition_on_previous_text": config.condition_on_previous_text,
        "speaker_assignment": config.speaker_assignment,
        "diarization_model": config.diarization_model,
        "num_speakers": config.num_speakers,
        "min_speakers": config.min_speakers,
        "max_speakers": config.max_speakers,
        "prompt_sha256": prompt_sha256,
    }
    return {"id": canonical_hash(profile), **profile}


def diar_cache_name(config: TranscriptionConfig, signature: str) -> str:
    """Durable diarization-cache file name (identical to notebook v5).

    Keyed on content + diarization parameters only, so a retry after an ASR
    failure (or a different Whisper model) reuses the diarization.
    """
    params = canonical_hash(
        {
            "cache_schema": DIAR_CACHE_SCHEMA,
            "model": config.diarization_model,
            "num": config.num_speakers,
            "min": config.min_speakers,
            "max": config.max_speakers,
        }
    )
    return f"{signature}_{params}.diar.json"


def enabled_deliverables(settings: BatchSettings) -> tuple[str, ...]:
    """Keys of the active secondary deliverables, in registry order."""
    flags = {
        "md": settings.deliver_markdown,
        "srt": settings.deliver_srt,
        "json": settings.deliver_json,
        "plano": settings.deliver_plain,
        "full_llm": settings.deliver_full_llm,
    }
    keys = [key for key in DELIVERABLES if flags.get(key)]
    if settings.deliver_splits:
        keys.append(SPLITS_KEY)
    return tuple(keys)


def render_options(settings: BatchSettings, producer: str) -> RenderOptions:
    """Renderer options derived from settings."""
    return RenderOptions(
        include_header=settings.include_header,
        timestamp_ms=settings.timestamp_ms,
        monotonic_tolerance_s=settings.monotonic_tolerance_s,
        max_unlabeled_fraction=settings.max_unlabeled_fraction,
        accept_silent_audio=settings.accept_silent_audio,
        md_gap_s=settings.md_gap_s,
        md_fillers=settings.md_fillers,
        split_words=settings.split_words,
        model=settings.model,
        beam_size=settings.beam_size,
        batch_size=settings.batch_size,
        producer=producer,
    )


def presentation_global(settings: BatchSettings) -> dict[str, Any]:
    """Parameters that only affect rendering. Changing them never costs GPU.

    Folder locations and the name template are recorded but not hashed:
    names already published stay stable (new settings apply to new files).
    """
    core = {
        "schema": RENDER_SCHEMA,
        "header": settings.include_header,
        "timestamp_ms": settings.timestamp_ms,
        "md_fillers": settings.md_fillers,
        "md_gap_s": settings.md_gap_s,
        "split_words": settings.split_words,
        "deliverables": list(enabled_deliverables(settings)),
        "repair_orphans": settings.repair_orphans,
        "orphan_tolerance_s": settings.orphan_tolerance_s,
        "max_unlabeled_fraction": settings.max_unlabeled_fraction,
        "accept_silent_audio": settings.accept_silent_audio,
        "monotonic_tolerance_s": settings.monotonic_tolerance_s,
    }
    return {
        "id": canonical_hash(core),
        **core,
        "template": settings.name_template,
        "mirror": settings.mirror_subfolders,
    }


def presentation_for(global_id: str, rename_sha: str) -> dict[str, Any]:
    """Per-file presentation profile (global profile + its speaker renames)."""
    return {
        "id": canonical_hash({"global": global_id, "rename": rename_sha}),
        "global_id": global_id,
        "rename_sha": rename_sha,
    }


__all__ = [
    "DIARIZATION_MODEL",
    "DIAR_CACHE_SCHEMA",
    "ENGINE_SEMANTICS",
    "diar_cache_name",
    "enabled_deliverables",
    "engine_config",
    "motor_profile",
    "presentation_for",
    "presentation_global",
    "prompt_digest",
    "render_options",
]
