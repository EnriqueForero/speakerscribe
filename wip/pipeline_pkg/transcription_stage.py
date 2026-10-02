"""Pipeline stage: invoke the appropriate transcription path (streaming or chunked).

Reads `ctx.chunked` to decide and writes the resulting metadata dict into
`ctx.metadata`. Merges accumulated external timings (`ctx.timings`) into the
metadata's `timings` sub-dict.
"""

from __future__ import annotations

from speakerscribe.pipeline.context import RunContext
from speakerscribe.transcription import transcribe_chunked, transcribe_streaming


def transcribe(ctx: RunContext) -> None:
    """Run streaming or chunked transcription based on `ctx.chunked`.

    Requires:
        ctx.model, ctx.audio_for_model, ctx.output_txt/_srt/_json must be set.
        ctx.chunks must be populated when ctx.chunked is True.

    Side effects:
        Mutates ctx.metadata. Merges ctx.timings into metadata['timings'].
    """
    assert ctx.model is not None, "model must be loaded before transcribe()"
    assert ctx.audio_for_model is not None
    assert ctx.output_txt is not None
    assert ctx.output_srt is not None
    assert ctx.output_json is not None

    output_jsonl = (
        ctx.paths.transcripts / f"{ctx.media_path.stem}_{ctx.config.model}.segments.jsonl"
        if ctx.config.streaming_jsonl
        else None
    )

    if ctx.chunked:
        metadata = transcribe_chunked(
            model=ctx.model,
            chunks=ctx.chunks,
            output_txt=ctx.output_txt,
            output_srt=ctx.output_srt,
            output_json=ctx.output_json,
            config=ctx.config,
            diar_turns=ctx.diar_turns,
            source_media=ctx.media_path,
            output_jsonl=output_jsonl,
        )
    else:
        metadata = transcribe_streaming(
            model=ctx.model,
            audio_path=ctx.audio_for_model,
            output_txt=ctx.output_txt,
            output_srt=ctx.output_srt,
            output_json=ctx.output_json,
            config=ctx.config,
            diar_turns=ctx.diar_turns,
            source_media=ctx.media_path,
            output_jsonl=output_jsonl,
        )

    # Merge external timings into metadata (transcription already added its own)
    metadata.setdefault("timings", {})
    for k, v in ctx.timings.items():
        metadata["timings"][k] = v

    ctx.metadata = metadata
