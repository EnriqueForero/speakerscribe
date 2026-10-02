"""Pipeline stage: derived outputs (markdown transcript + word splits + unified-for-LLM)."""

from __future__ import annotations

import time

from speakerscribe.output import generate_transcript_md, split_text_by_words, write_unified_for_llm
from speakerscribe.pipeline.context import RunContext


def generate_derived_outputs(ctx: RunContext) -> None:
    """Generate `.transcript.md`, word-aware `*.split-*.txt`, and unified-for-LLM file.

    Requires `ctx.metadata` already populated by the transcription stage.
    Mutates `ctx.metadata['timings']`.
    """
    assert ctx.output_md is not None
    assert ctx.output_txt is not None
    metadata = ctx.metadata
    metadata.setdefault("timings", {})

    # Markdown transcript
    if ctx.config.generate_transcript_md and metadata.get("segments"):
        t = time.time()
        generate_transcript_md(
            metadata["segments"],
            ctx.output_md,
            metadata,
            gap_max_s=ctx.config.gap_max_s_transcript,
            remove_fillers=ctx.config.remove_fillers,
            aggressive_fillers=ctx.config.aggressive_fillers,
        )
        metadata["timings"]["transcript_md_s"] = round(time.time() - t, 2)

    # Splits + unified-for-LLM
    t = time.time()
    split_text_by_words(
        ctx.output_txt,
        ctx.paths.splits,
        ctx.config.words_per_split,
        has_speakers=metadata.get("diarization_enabled", False),
    )
    if ctx.config.produce_unified_for_llm:
        write_unified_for_llm(ctx.output_txt, ctx.paths.splits)
    metadata["timings"]["splits_s"] = round(time.time() - t, 2)
