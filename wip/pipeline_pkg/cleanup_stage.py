"""Pipeline stage: post-run cleanup of temporary WAVs and chunk WAVs."""

from __future__ import annotations

from speakerscribe.logging_config import logger
from speakerscribe.pipeline.cleanup import _cleanup_chunks
from speakerscribe.pipeline.context import RunContext


def cleanup_temp_files(ctx: RunContext) -> None:
    """Delete chunk WAVs and the temp full WAV per config flags.

    Honors `config.delete_chunk_wavs` and `config.delete_temp_wav`. Never
    touches `ctx.media_path`. The chunk cleanup skips any chunk whose path
    equals `audio_for_model` (corner case for short audios where the single
    "chunk" is actually the full WAV).
    """
    assert ctx.audio_for_model is not None

    if ctx.config.delete_chunk_wavs and ctx.chunks:
        _cleanup_chunks(ctx.chunks, ctx.audio_for_model)
        logger.debug("Chunk WAVs deleted")
    if ctx.config.delete_temp_wav and ctx.wav_path and ctx.wav_path.exists():
        ctx.wav_path.unlink()
        logger.debug("Temporary WAV deleted")
