"""High-level orchestration: `process_one` and `process_batch`.

`process_one` is the canonical entry point for transcribing a single media
file. It composes the stages defined in sibling modules around a single
`RunContext` and returns the metadata dict that downstream code consumes.

`process_batch` loads the Whisper model once and iterates `process_one` over
every media file in the workspace, isolating per-file errors so one bad
file does not abort the batch.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from speakerscribe.audio import calculate_file_hash
from speakerscribe.config import TranscriptionConfig, WorkspacePaths
from speakerscribe.logging_config import logger
from speakerscribe.pipeline.chunking_stage import split_if_long
from speakerscribe.pipeline.cleanup_stage import cleanup_temp_files
from speakerscribe.pipeline.context import RunContext
from speakerscribe.pipeline.diarization_stage import _maybe_diarize
from speakerscribe.pipeline.idempotency import check_idempotency
from speakerscribe.pipeline.outputs_stage import generate_derived_outputs
from speakerscribe.pipeline.persistence_stage import register_if_enabled
from speakerscribe.pipeline.preflight import preflight_check
from speakerscribe.pipeline.quality_stage import evaluate_quality
from speakerscribe.pipeline.reporting import report_speaker_distribution
from speakerscribe.pipeline.transcription_stage import transcribe
from speakerscribe.pipeline.wav import _maybe_extract_wav
from speakerscribe.transcription import load_whisper_model, release_whisper_model

if TYPE_CHECKING:
    from pathlib import Path

    from faster_whisper import WhisperModel


def process_one(
    media_path: Path,
    paths: WorkspacePaths,
    model: WhisperModel,
    config: TranscriptionConfig,
) -> dict[str, Any]:
    """Run the full pipeline for a single media file.

    Idempotency strategy:
        - When `enable_runs_db=True` (default): skip if a previous run with the
          same file hash + ASR model + diarization model is recorded with
          status 'ok' AND the JSON output still exists. Robust to source renames.
        - When `enable_runs_db=False`: skip if both the JSON and TXT outputs
          exist and `force_reprocess` is False (filename-based; less robust).

    Long audios (> `config.long_audio_threshold_min` minutes) are split into
    overlapping chunks for transcription. Diarization always runs on the FULL
    audio for speaker consistency.

    The pipeline is implemented as a sequence of stages, each mutating a
    shared `RunContext`. See the `pipeline/*_stage.py` modules.

    Args:
        media_path: Path to the source audio/video file.
        paths: WorkspacePaths instance.
        model: A WhisperModel already loaded by `load_whisper_model`.
        config: Pipeline configuration.

    Returns:
        Dict with metadata + status: "ok" | "skipped" | "error".
    """
    ctx = RunContext(media_path=media_path, paths=paths, config=config, model=model)
    ctx.compute_output_paths()

    # ── 0. Idempotency: maybe skip (purges stale chunks on force_reprocess)
    skip = check_idempotency(ctx)
    if skip is not None:
        return skip

    ctx.t_start = time.time()

    # ── 1. Extract WAV (or pass media through)
    ctx.audio_for_model, ctx.wav_path = _maybe_extract_wav(media_path, paths, config, ctx.timings)

    # ── 2. Diarization (BEFORE Whisper to free VRAM) — always on FULL audio
    ctx.diar_turns = _maybe_diarize(
        ctx.audio_for_model, paths, config, media_path.stem, ctx.timings
    )

    # ── 3. Chunking decision (mutates ctx.chunks, ctx.chunked)
    split_if_long(ctx)

    # ── 4. Transcribe (chunked or streaming) -> ctx.metadata
    transcribe(ctx)

    # ── 5. Derived outputs (md transcript + word splits + unified-for-LLM)
    generate_derived_outputs(ctx)

    # ── 6. Cleanup chunk WAVs and temp WAV
    cleanup_temp_files(ctx)

    # ── 7. Quality check -> ctx.quality_report
    evaluate_quality(ctx)

    # ── 8. Persistence (DB row)
    register_if_enabled(ctx)

    # ── 9. Finalize metadata
    ctx.metadata.setdefault("timings", {})
    ctx.metadata["timings"]["total_s"] = round(time.time() - ctx.t_start, 2)
    ctx.metadata["status"] = "ok"
    ctx.metadata["base_name"] = f"{media_path.stem}_{config.model}"
    if config.enable_runs_db:
        ctx.metadata["file_hash"] = ctx.file_hash
    if ctx.quality_report:
        ctx.metadata["quality_ok"] = ctx.quality_report.quality_ok

    return ctx.metadata


def process_batch(
    paths: WorkspacePaths,
    config: TranscriptionConfig,
) -> list[dict[str, Any]]:
    """Process every media file in `data/`. The Whisper model is loaded ONCE.

    Errors are isolated per file: a failure in one media item is logged and
    optionally persisted, but the batch continues.

    Args:
        paths: WorkspacePaths instance.
        config: Pipeline configuration.

    Returns:
        List of metadata dicts, one per file (status: ok / skipped / error).
    """
    preflight_check(paths, config)

    media = paths.list_media_files()
    logger.info(f"{len(media)} file(s) detected:")
    for i, v in enumerate(media, 1):
        logger.info(f"   {i}. {v.name} ({v.stat().st_size / 1e6:.1f} MB)")

    model = load_whisper_model(config)

    results: list[dict[str, Any]] = []
    t_batch_start = time.time()
    try:
        for i, item in enumerate(media, 1):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"[{i}/{len(media)}] {item.name}")
            logger.info(f"{'=' * 60}")
            try:
                meta = process_one(item, paths, model, config)
                results.append(meta)
                if meta.get("status") == "ok" and meta.get("speakers_summary"):
                    report_speaker_distribution(meta["speakers_summary"])
            except Exception as e:
                logger.exception(f"ERROR on {item.name}: {type(e).__name__}: {e}")
                if config.enable_runs_db:
                    try:
                        from speakerscribe.persistence import register_run

                        file_hash = calculate_file_hash(item)
                        register_run(
                            paths.db_path,
                            {
                                "audio_file": item.name,
                                "model": config.model,
                                "diarization_model": (
                                    config.diarization_model if config.enable_diarization else None
                                ),
                                "processed_at": datetime.now(tz=timezone.utc).isoformat(),
                                "config": (
                                    config.model_dump() if hasattr(config, "model_dump") else {}
                                ),
                            },
                            file_hash=file_hash,
                            status="error",
                            error_message=f"{type(e).__name__}: {e}",
                        )
                    except Exception:
                        pass
                results.append(
                    {
                        "status": "error",
                        "audio_file": item.name,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
                continue
    finally:
        release_whisper_model(model)

    # ── Final report
    total_elapsed = time.time() - t_batch_start
    n_ok = sum(1 for r in results if r.get("status") == "ok")
    n_skip = sum(1 for r in results if r.get("status") == "skipped")
    n_err = sum(1 for r in results if r.get("status") == "error")
    n_words = sum(r.get("total_words", 0) for r in results if r.get("status") == "ok")
    n_speakers_total = sum(
        len(r.get("speakers_summary") or {}) for r in results if r.get("status") == "ok"
    )

    logger.info(
        "\n"
        "===============================================================\n"
        "                       BATCH FINAL REPORT                       \n"
        "===============================================================\n"
        f"  Total time         : {total_elapsed / 60:>8.1f} min\n"
        f"  Files OK           : {n_ok:>3d}\n"
        f"  Files skipped      : {n_skip:>3d}\n"
        f"  Files with errors  : {n_err:>3d}\n"
        f"  Total words        : {n_words:>10,}\n"
        f"  Speakers (sum)     : {n_speakers_total:>3d}\n"
        "==============================================================="
    )
    return results
