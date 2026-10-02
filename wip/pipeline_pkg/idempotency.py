"""Idempotency stage: decide whether to skip a `process_one` invocation.

Two strategies:
    - DB-backed (when `enable_runs_db=True`, default): match by file content
      hash + ASR model + diar model. Robust to source renames.
    - Output-based fallback: check that .json + .txt outputs exist.

Returns a "skipped" result dict on hit, or None when the pipeline must run.
The hash is computed lazily — only when DB-backed idempotency is enabled.
"""

from __future__ import annotations

from typing import Any

from speakerscribe.audio import calculate_file_hash
from speakerscribe.logging_config import logger
from speakerscribe.persistence import find_run_by_hash
from speakerscribe.pipeline.cleanup import _purge_stale_chunks
from speakerscribe.pipeline.context import RunContext


def check_idempotency(ctx: RunContext) -> dict[str, Any] | None:
    """Decide if the run can be skipped. Returns a skip result, or None to proceed.

    Side effects:
        - On `force_reprocess=True`: purges stale chunk WAVs for this stem.
        - Populates `ctx.file_hash` when DB-backed idempotency is enabled.
        - Sets `ctx.skipped_due_to_cache = True` on hit.

    Args:
        ctx: RunContext with `media_path`, `paths`, `config`, and output paths
             already populated by `ctx.compute_output_paths()`.

    Returns:
        Skip result dict on cache hit, or None to proceed with the pipeline.
    """
    config = ctx.config
    media_path = ctx.media_path
    base_name = f"{media_path.stem}_{config.model}"

    if config.force_reprocess:
        logger.info(f"Force re-process for: {media_path.name}")
        _purge_stale_chunks(ctx.paths, media_path.stem)
        return None

    # Hash for content-based idempotency (only computed when DB is enabled)
    if config.enable_runs_db:
        ctx.file_hash = calculate_file_hash(media_path)
        diar_model = config.diarization_model if config.enable_diarization else None
        existing = find_run_by_hash(ctx.paths.db_path, ctx.file_hash, config.model, diar_model)
        if (
            existing
            and existing.get("status") == "ok"
            and ctx.output_json is not None
            and ctx.output_json.exists()
            and ctx.output_txt is not None
            and ctx.output_txt.exists()
        ):
            logger.info(
                f"Already processed: {media_path.name} (hash={ctx.file_hash[:8]}) — skipping"
            )
            ctx.skipped_due_to_cache = True
            return {
                "status": "skipped",
                "audio_file": media_path.name,
                "file_hash": ctx.file_hash,
                "previous_run_id": existing.get("id"),
                "base_name": base_name,
            }
    else:
        ctx.file_hash = ""
        if (
            ctx.output_json is not None
            and ctx.output_json.exists()
            and ctx.output_txt is not None
            and ctx.output_txt.exists()
        ):
            logger.info(f"Outputs already exist for: {media_path.name} — skipping")
            ctx.skipped_due_to_cache = True
            return {
                "status": "skipped",
                "audio_file": media_path.name,
                "base_name": base_name,
            }
    return None
