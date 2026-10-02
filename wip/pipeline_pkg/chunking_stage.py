"""Pipeline stage: decide whether to chunk long audio and produce the chunks."""

from __future__ import annotations

import time

from speakerscribe.audio import split_long_audio
from speakerscribe.logging_config import logger
from speakerscribe.pipeline.context import RunContext
from speakerscribe.pipeline.wav import _should_chunk_audio


def split_if_long(ctx: RunContext) -> None:
    """Populate `ctx.chunks` and `ctx.chunked` for long audios.

    For audios shorter than `config.long_audio_threshold_min`, leaves
    `ctx.chunks = []` and `ctx.chunked = False`. The transcription stage
    then takes the single-shot path. Mutates `ctx.timings`.
    """
    assert ctx.audio_for_model is not None, "audio_for_model must be set before chunking"

    should_chunk, duration_s = _should_chunk_audio(ctx.audio_for_model, ctx.config)
    ctx.total_duration_s = duration_s

    if not should_chunk:
        return

    logger.info(
        f"Long audio ({duration_s / 60:.1f} min > "
        f"{ctx.config.long_audio_threshold_min} min threshold) — chunking for transcription"
    )
    t = time.time()
    ctx.chunks = split_long_audio(
        input_wav=ctx.audio_for_model,
        output_dir=ctx.paths.audio_chunks,
        chunk_duration_s=ctx.config.chunk_duration_min * 60,
        overlap_s=ctx.config.chunk_overlap_s,
        sample_rate=ctx.config.sample_rate,
    )
    ctx.chunked = True
    ctx.timings["split_audio_s"] = round(time.time() - t, 2)
