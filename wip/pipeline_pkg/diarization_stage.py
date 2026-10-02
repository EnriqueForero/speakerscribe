"""Pipeline stage wrapping `speakerscribe.diarization.diarize_audio`.

Adds caching path derivation from the workspace + per-stage telemetry +
typed error handling so the pipeline keeps running on diarization failures
(falling back to single-speaker output).
"""

from __future__ import annotations

import time
from pathlib import Path

from speakerscribe.config import TranscriptionConfig, WorkspacePaths
from speakerscribe.diarization import diarization_params_hash, diarize_audio
from speakerscribe.logging_config import logger


def _maybe_diarize(
    audio_for_model: Path,
    paths: WorkspacePaths,
    config: TranscriptionConfig,
    media_stem: str,
    timings: dict[str, float],
) -> list[dict] | None:
    """Run diarization on the full audio (with cache invalidated by params hash).

    Returns None when diarization is disabled, the HF token / model terms /
    VRAM are not OK, or any other recoverable error occurred. In every error
    branch the elapsed time is still recorded in `timings`.
    """
    if not config.enable_diarization:
        return None
    cache_path = paths.diar_cache / f"{media_stem}_{diarization_params_hash(config)}.diar.json"
    t = time.time()
    try:
        turns = diarize_audio(audio_for_model, config, cache_path)
        timings["diarization_s"] = round(time.time() - t, 2)
        return turns
    except RuntimeError as e:
        logger.error(f"Diarization failed (known): {e}")
        logger.error("Check: HF token, model terms, available VRAM")
    except (ImportError, AttributeError):
        logger.exception("Diarization failed (version mismatch). Verify pyannote.audio>=4.0.")
    except Exception:
        logger.exception("Diarization failed (uncategorized).")
    timings["diarization_s"] = round(time.time() - t, 2)
    return None
