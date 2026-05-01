"""Speaker diarization with pyannote.audio 4.x + overlap-based assignment.

Pinned to pyannote.audio>=4.0,<5.0. Older versions (3.x) are not supported.
"""

from __future__ import annotations

import gc
import json
import time
from collections import defaultdict
from pathlib import Path

from speakerscribe.config import SPK_NO_OVERLAP, TranscriptionConfig
from speakerscribe.logging_config import logger


def diarize_audio(
    audio_path: Path,
    config: TranscriptionConfig,
    cache_path: Path | None = None,
) -> list[dict]:
    """Run speaker diarization on a WAV file and return a list of turns.

    Strategy on Colab Free (limited VRAM):
        1. Load the pyannote pipeline on GPU.
        2. Process the audio (in-memory).
        3. RELEASE the pipeline before returning so VRAM is free for Whisper.
        4. Optionally cache the result as JSON for future re-runs.

    Args:
        audio_path: 16 kHz mono WAV file.
        config: TranscriptionConfig with hf_token and num/min/max_speakers.
        cache_path: If set and the file exists, results are loaded from cache
            (diarization is skipped). If set and the file does not exist,
            results are written there after a successful run.

    Returns:
        List of turn dicts sorted ascending by start:
        `[{"start": float, "end": float, "speaker": "SPEAKER_00"}, ...]`

    Raises:
        RuntimeError: If the HuggingFace token is missing/invalid, the user
            has not accepted the model terms, or the model fails to load.
    """
    # ── Cache hit
    if cache_path and cache_path.exists():
        with cache_path.open(encoding="utf-8") as f:
            data = json.load(f)
        turns: list[dict] = data.get("turns", data)
        logger.info(f"Diarization loaded from cache: {len(turns)} turns ({cache_path.name})")
        return turns

    logger.info(f"Diarizing: {audio_path.name}")
    t0 = time.time()

    # Lazy imports
    import torch
    from pyannote.audio import Pipeline as PyannotePipeline

    token = config.resolve_hf_token()
    try:
        pipeline = PyannotePipeline.from_pretrained(
            config.diarization_model,
            token=token,
        )
    except Exception as e:
        raise RuntimeError(
            f"Could not load pyannote pipeline.\n"
            f"   Most likely cause: invalid HF token or unaccepted model terms.\n"
            f"   Steps to fix:\n"
            f"     1. Generate a 'Read' token (NOT fine-grained) at\n"
            f"        https://huggingface.co/settings/tokens\n"
            f"     2. Accept the terms at:\n"
            f"        https://huggingface.co/{config.diarization_model}\n"
            f"     3. Make the token available via Colab Secrets, the HF_TOKEN env var,\n"
            f"        or by passing hf_token=... to TranscriptionConfig.\n"
            f"   Original error: {e}"
        ) from e

    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
        vram_pre = torch.cuda.memory_allocated() / 1e9
        logger.debug(f"VRAM after loading pyannote: {vram_pre:.2f} GB")

    # ── Build pipeline kwargs
    diar_kwargs: dict = {}
    if config.num_speakers is not None:
        diar_kwargs["num_speakers"] = config.num_speakers
        logger.info(f"   num_speakers fixed = {config.num_speakers}")
    else:
        if config.min_speakers is not None:
            diar_kwargs["min_speakers"] = config.min_speakers
        if config.max_speakers is not None:
            diar_kwargs["max_speakers"] = config.max_speakers
        if diar_kwargs:
            logger.info(f"   Speakers: min={config.min_speakers} max={config.max_speakers}")
        else:
            logger.info("   Speakers: auto-detect")

    # ── Run diarization (pyannote 4.x returns DiarizeOutput)
    diarization_output = pipeline(str(audio_path), **diar_kwargs)

    # pyannote 4.x: DiarizeOutput exposes .exclusive_speaker_diarization (or
    # .speaker_diarization for some pipelines)
    if hasattr(diarization_output, "exclusive_speaker_diarization"):
        diarization = diarization_output.exclusive_speaker_diarization
        logger.debug("Using exclusive_speaker_diarization (pyannote 4.x community-1)")
    elif hasattr(diarization_output, "speaker_diarization"):
        diarization = diarization_output.speaker_diarization
        logger.debug("Using speaker_diarization (pyannote 4.x)")
    else:
        # Fallback: assume the object is iterable like an Annotation
        diarization = diarization_output
        logger.warning(
            "Unexpected pyannote output type; falling back to direct iteration. "
            "If diarization looks wrong, check that pyannote.audio is >=4.0."
        )

    # ── Extract turns
    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append(
            {
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
                "speaker": speaker,
            }
        )
    turns.sort(key=lambda t: t["start"])

    elapsed = time.time() - t0
    unique_speakers = sorted({t["speaker"] for t in turns})
    logger.success(
        f"Diarization: {len(turns)} turns | {len(unique_speakers)} speakers | "
        f"{elapsed / 60:.1f} min"
    )
    logger.info(f"   Speakers detected: {unique_speakers}")

    # ── Save cache
    if cache_path:
        from speakerscribe import __version__

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import pyannote.audio

            pyannote_version = pyannote.audio.__version__
        except Exception:
            pyannote_version = "unknown"

        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "audio_file": audio_path.name,
                    "model": config.diarization_model,
                    "n_speakers": len(unique_speakers),
                    "turns": turns,
                    "elapsed_seconds": round(elapsed, 2),
                    "pyannote_version": pyannote_version,
                    "package_version": __version__,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        logger.debug(f"Cache: {cache_path.name}")

    # ── Release pyannote pipeline (free VRAM for Whisper)
    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    logger.debug("pyannote pipeline released")

    return turns


def assign_speaker_to_segment(
    seg_start: float,
    seg_end: float,
    diar_turns: list[dict],
) -> tuple[str, float]:
    """Assign a speaker to a Whisper segment by maximum temporal overlap.

    For each diarization turn, compute its overlap with [seg_start, seg_end]
    and return the speaker that accumulates the most seconds. This is the
    standard algorithm to combine streaming ASR with offline diarization.

    Optimization: turns are pre-sorted by start ascending, allowing an early
    exit when a turn starts after the segment ends.

    Args:
        seg_start: Whisper segment start (seconds).
        seg_end: Whisper segment end (seconds).
        diar_turns: List of turns sorted ascending by start.

    Returns:
        Tuple (speaker, overlap_seconds). If no turn overlaps the segment,
        returns (SPK_NO_OVERLAP, 0.0).

    Examples:
        >>> assign_speaker_to_segment(10.0, 20.0, [])
        ('SPEAKER_NO_OVERLAP', 0.0)
        >>> turns = [{"start": 5, "end": 15, "speaker": "SPEAKER_00"},
        ...          {"start": 18, "end": 25, "speaker": "SPEAKER_01"}]
        >>> assign_speaker_to_segment(10, 20, turns)
        ('SPEAKER_00', 5.0)
    """
    if not diar_turns or seg_end <= seg_start:
        return (SPK_NO_OVERLAP, 0.0)

    overlap_by_speaker: dict[str, float] = defaultdict(float)
    for t in diar_turns:
        if t["start"] >= seg_end:
            break  # early exit: turns are sorted by start
        if t["end"] <= seg_start:
            continue
        overlap = min(seg_end, t["end"]) - max(seg_start, t["start"])
        if overlap > 0:
            overlap_by_speaker[t["speaker"]] += overlap

    if not overlap_by_speaker:
        return (SPK_NO_OVERLAP, 0.0)

    speaker, overlap = max(overlap_by_speaker.items(), key=lambda kv: kv[1])
    return (speaker, round(overlap, 3))


__all__ = ["assign_speaker_to_segment", "diarize_audio"]
