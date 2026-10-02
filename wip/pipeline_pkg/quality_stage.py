"""Pipeline stage: post-transcription heuristic quality check."""

from __future__ import annotations

import time

from speakerscribe.logging_config import logger
from speakerscribe.pipeline.context import RunContext
from speakerscribe.quality import evaluate_transcription_quality


def evaluate_quality(ctx: RunContext) -> None:
    """Run the heuristic quality checker and populate `ctx.quality_report`.

    Skipped silently when `config.evaluate_quality=False`. Adds telemetry to
    `ctx.metadata['timings']['quality_check_s']` and logs the summary on
    failures.
    """
    if not ctx.config.evaluate_quality:
        return
    t = time.time()
    ctx.quality_report = evaluate_transcription_quality(ctx.metadata)
    ctx.metadata.setdefault("timings", {})
    ctx.metadata["timings"]["quality_check_s"] = round(time.time() - t, 2)
    if ctx.quality_report.quality_ok:
        logger.success("Quality: OK")
    else:
        logger.warning(ctx.quality_report.summary())
