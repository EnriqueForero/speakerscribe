"""Chunk-WAV cleanup helpers."""

from __future__ import annotations

from pathlib import Path

from speakerscribe.audio import AudioChunk
from speakerscribe.config import WorkspacePaths
from speakerscribe.logging_config import logger


def _cleanup_chunks(chunks: list[AudioChunk], original_wav: Path) -> None:
    """Delete chunk WAVs (but never the original full WAV)."""
    for chunk in chunks:
        if chunk.path == original_wav:
            continue
        if chunk.path.exists():
            try:
                chunk.path.unlink()
            except OSError as e:
                logger.warning(f"Could not delete chunk {chunk.path.name}: {e}")


def _purge_stale_chunks(paths: WorkspacePaths, media_stem: str) -> None:
    """Delete every chunk WAV that belongs to `media_stem`.

    Called when `force_reprocess=True` so that mismatched chunk durations from
    a previous run with different `chunk_duration_min` cannot be silently
    reused by `split_long_audio` (which keys reuse on filename + duration).
    """
    if not paths.audio_chunks.exists():
        return
    n = 0
    for f in paths.audio_chunks.glob(f"{media_stem}_chunk*.wav"):
        try:
            f.unlink()
            n += 1
        except OSError as e:
            logger.warning(f"Could not purge stale chunk {f.name}: {e}")
    if n:
        logger.info(f"Purged {n} stale chunk(s) for {media_stem}")
