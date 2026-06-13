"""Speaker diarization with pyannote.audio 4.x + speaker assignment.

Pinned to pyannote.audio>=4.0,<5.0. Older versions (3.x) are not supported.

Engine lifecycle (new in 0.3):
    `DiarizationEngine` loads the pyannote pipeline ONCE and reuses it for
    every file in a batch. Before 0.3, `diarize_audio` re-ran
    `from_pretrained` + `.to(cuda)` for EVERY file — tens of seconds per
    file for zero benefit on a 16 GB T4, where Whisper turbo fp16 (~1.6 GB)
    and pyannote (~2-3 GB) comfortably coexist in VRAM.

Cache strategy:
    The diarization output is cached to disk keyed by `(audio_stem, params_hash)`,
    where `params_hash` is derived from the parameters that materially affect
    the output (`diarization_model`, `num_speakers`, `min_speakers`, `max_speakers`).
    Changing any of these invalidates the cache automatically.

Speaker assignment:
    `assign_speaker_to_segment` — one speaker per Whisper segment by maximum
        temporal overlap (legacy `speaker_assignment="segment"`).
    `assign_speakers_by_words` — WhisperX-style word-level attribution with
        re-segmentation at speaker changes and single-word island smoothing
        (default `speaker_assignment="word"`). Reference approach:
        https://github.com/m-bain/whisperX
"""

from __future__ import annotations

import gc
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from speakerscribe.config import SPK_NO_OVERLAP, TranscriptionConfig
from speakerscribe.io_utils import atomic_write_json
from speakerscribe.logging_config import logger


def diarization_params_hash(config: TranscriptionConfig) -> str:
    """Compute a short hash of the diarization parameters that affect the output.

    Used to key the diarization cache: changes in any of these parameters
    invalidate the cache so we never accidentally reuse stale results when
    the user tweaks `num_speakers` or switches diarization models.

    Args:
        config: TranscriptionConfig instance.

    Returns:
        Hex string of length 8.
    """
    payload = {
        "model": config.diarization_model,
        "num_speakers": config.num_speakers,
        "min_speakers": config.min_speakers,
        "max_speakers": config.max_speakers,
    }
    serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(serialized, usedforsecurity=False).hexdigest()[:8]


class DiarizationEngine:
    """Reusable pyannote pipeline holder: load once, diarize many files.

    Usage (the pipeline weights stay resident across calls):
        >>> with DiarizationEngine(config) as engine:
        ...     turns_a = engine.diarize(wav_a, cache_a)
        ...     turns_b = engine.diarize(wav_b, cache_b)

    The pyannote pipeline is loaded lazily on the first `diarize` call, so
    constructing the engine is free and a fully-cached batch never touches
    HuggingFace at all. `close()` (or exiting the context) releases the
    model and clears the CUDA cache.

    Args:
        config: TranscriptionConfig with hf_token and num/min/max_speakers.
    """

    def __init__(self, config: TranscriptionConfig) -> None:
        self._config = config
        self._pipeline: Any | None = None

    # ── Context manager ────────────────────────────────────────────
    def __enter__(self) -> DiarizationEngine:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ── Lifecycle ──────────────────────────────────────────────────
    def _ensure_loaded(self) -> Any:
        """Load the pyannote pipeline on first use (and move it to CUDA)."""
        if self._pipeline is not None:
            return self._pipeline

        import torch
        from pyannote.audio import Pipeline as PyannotePipeline

        token = self._config.resolve_hf_token()
        t0 = time.time()
        try:
            pipeline = PyannotePipeline.from_pretrained(
                self._config.diarization_model,
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
                f"        https://huggingface.co/{self._config.diarization_model}\n"
                f"     3. Make the token available via Colab Secrets, the HF_TOKEN env var,\n"
                f"        or by passing hf_token=... to TranscriptionConfig.\n"
                f"   Original error: {e}"
            ) from e

        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
            vram = torch.cuda.memory_allocated() / 1e9
            logger.debug(f"VRAM after loading pyannote: {vram:.2f} GB")
        logger.success(f"pyannote pipeline loaded in {time.time() - t0:.1f}s (stays resident)")
        self._pipeline = pipeline
        return pipeline

    def close(self) -> None:
        """Release the pyannote pipeline and clear the CUDA cache. Idempotent."""
        if self._pipeline is None:
            return
        self._pipeline = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:  # pragma: no cover - torch always present with pyannote
            pass
        logger.debug("pyannote pipeline released")

    # ── Work ───────────────────────────────────────────────────────
    def diarize(self, audio_path: Path, cache_path: Path | None = None) -> list[dict]:
        """Run speaker diarization on a WAV file and return a list of turns.

        Cache validation: the cache file's params_hash is compared against
        the current config. If they differ, the cache is ignored and
        diarization is re-run. Cache hits never load the model.

        Args:
            audio_path: 16 kHz mono WAV file.
            cache_path: If set, results are loaded from cache when available
                with matching params_hash, and saved there after a run.

        Returns:
            List of turn dicts sorted ascending by start:
            `[{"start": float, "end": float, "speaker": "SPEAKER_00"}, ...]`

        Raises:
            RuntimeError: If the HuggingFace token is missing/invalid, the
                user has not accepted the model terms, or the model fails
                to load.
        """
        config = self._config
        expected_hash = diarization_params_hash(config)

        cached = _read_cache(cache_path, expected_hash)
        if cached is not None:
            return cached

        logger.info(f"Diarizing: {audio_path.name}")
        t0 = time.time()
        pipeline = self._ensure_loaded()

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

        diarization_output = pipeline(str(audio_path), **diar_kwargs)

        # pyannote 4.x: DiarizeOutput exposes .exclusive_speaker_diarization
        # (community-1: non-overlapping turns, designed for reconciliation
        # with STT word timelines) or .speaker_diarization.
        if hasattr(diarization_output, "exclusive_speaker_diarization"):
            diarization = diarization_output.exclusive_speaker_diarization
            logger.debug("Using exclusive_speaker_diarization (pyannote 4.x community-1)")
        elif hasattr(diarization_output, "speaker_diarization"):
            diarization = diarization_output.speaker_diarization
            logger.debug("Using speaker_diarization (pyannote 4.x)")
        else:
            diarization = diarization_output
            logger.warning(
                "Unexpected pyannote output type; falling back to direct iteration. "
                "If diarization looks wrong, check that pyannote.audio is >=4.0."
            )

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

        _write_cache(cache_path, config, audio_path, expected_hash, turns, elapsed)
        return turns


def _read_cache(cache_path: Path | None, expected_hash: str) -> list[dict] | None:
    """Load cached turns when the params hash matches; None otherwise."""
    if not cache_path or not cache_path.exists():
        return None
    try:
        with cache_path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Could not read diarization cache ({e}); recomputing.")
        return None
    cached_hash = data.get("params_hash")
    if cached_hash != expected_hash:
        logger.info(
            f"Diarization cache invalidated: params changed ({cached_hash} -> {expected_hash})"
        )
        return None
    turns: list[dict] = data.get("turns", [])
    logger.info(
        f"Diarization loaded from cache: {len(turns)} turns "
        f"({cache_path.name}, params_hash={cached_hash})"
    )
    return turns


def _write_cache(
    cache_path: Path | None,
    config: TranscriptionConfig,
    audio_path: Path,
    params_hash: str,
    turns: list[dict],
    elapsed: float,
) -> None:
    """Persist the diarization result atomically (a torn cache poisons reruns)."""
    if not cache_path:
        return
    from speakerscribe import __version__

    try:
        import pyannote.audio

        pyannote_version = pyannote.audio.__version__
    except Exception:
        pyannote_version = "unknown"

    atomic_write_json(
        cache_path,
        {
            "audio_file": audio_path.name,
            "model": config.diarization_model,
            "params_hash": params_hash,
            "params": {
                "num_speakers": config.num_speakers,
                "min_speakers": config.min_speakers,
                "max_speakers": config.max_speakers,
            },
            "n_speakers": len({t["speaker"] for t in turns}),
            "turns": turns,
            "elapsed_seconds": round(elapsed, 2),
            "pyannote_version": pyannote_version,
            "package_version": __version__,
        },
    )
    logger.debug(f"Cache: {cache_path.name}")


def diarize_audio(
    audio_path: Path,
    config: TranscriptionConfig,
    cache_path: Path | None = None,
) -> list[dict]:
    """One-shot diarization: load pipeline, diarize, release.

    Backward-compatible wrapper around `DiarizationEngine` for single-file
    use. For batches, construct ONE `DiarizationEngine` and reuse it —
    this wrapper pays the full model load on every call.

    Args:
        audio_path: 16 kHz mono WAV file.
        config: TranscriptionConfig with hf_token and num/min/max_speakers.
        cache_path: If set, results are loaded from cache when available with
            matching params_hash, and saved there after a successful run.

    Returns:
        List of turn dicts sorted ascending by start.

    Raises:
        RuntimeError: If the HuggingFace token is missing/invalid, the user
            has not accepted the model terms, or the model fails to load.
    """
    with DiarizationEngine(config) as engine:
        return engine.diarize(audio_path, cache_path)


def assign_speaker_to_segment(
    seg_start: float,
    seg_end: float,
    diar_turns: list[dict],
) -> tuple[str, float]:
    """Assign a speaker to a Whisper segment by maximum temporal overlap.

    For each diarization turn, compute its overlap with [seg_start, seg_end]
    and return the speaker that accumulates the most seconds. This is the
    legacy `speaker_assignment="segment"` algorithm: a segment that spans a
    turn change is attributed ENTIRELY to one speaker. Prefer the default
    word-level assignment for conversational audio.

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


# Words shorter than this that disagree with BOTH neighbors are treated as
# diarization-timestamp jitter and reassigned to the surrounding speaker.
WORD_ISLAND_MAX_S = 0.4


def assign_speakers_by_words(
    words: list[dict[str, Any]],
    diar_turns: list[dict],
    *,
    min_island_s: float = WORD_ISLAND_MAX_S,
) -> list[dict[str, Any]]:
    """Word-level speaker attribution with re-segmentation at speaker changes.

    Why: a single Whisper segment frequently spans a turn change in fast
    conversation ("...claro que sí. ¿Y usted qué opina?"). Segment-level
    max-overlap attributes the WHOLE segment to one speaker — a systematic
    error. Here each word is attributed by overlap with the diarization
    timeline, then consecutive same-speaker words are merged into pure
    sub-segments (the WhisperX approach).

    Smoothing: a single word shorter than `min_island_s` whose speaker
    differs from BOTH identical neighbors is reassigned to the neighbor
    speaker — diarization boundaries jitter by ~100-300 ms and would
    otherwise produce "ping-pong" one-word turns.

    Args:
        words: Word dicts with keys `word`, `start`, `end` (seconds; absolute
            timeline matching `diar_turns`).
        diar_turns: Diarization turns sorted ascending by start.
        min_island_s: Maximum duration of a reassignable one-word island.

    Returns:
        List of sub-segment dicts, each with keys:
        `start`, `end`, `text`, `speaker`, `overlap` (seconds of diarization
        support for the sub-segment) and `words` (the member word dicts).
        Empty list when `words` is empty.
    """
    if not words:
        return []

    speakers = [assign_speaker_to_segment(w["start"], w["end"], diar_turns)[0] for w in words]

    # ── Island smoothing (single-word jitter) ──────────────────────
    for i in range(1, len(words) - 1):
        if (
            speakers[i] != speakers[i - 1]
            and speakers[i - 1] == speakers[i + 1]
            and (words[i]["end"] - words[i]["start"]) < min_island_s
        ):
            speakers[i] = speakers[i - 1]

    # ── Merge consecutive same-speaker words into sub-segments ─────
    pieces: list[dict[str, Any]] = []
    group_start = 0
    for i in range(1, len(words) + 1):
        if i == len(words) or speakers[i] != speakers[group_start]:
            members = words[group_start:i]
            speaker = speakers[group_start]
            start = members[0]["start"]
            end = members[-1]["end"]
            text = "".join(w["word"] for w in members).strip()
            pieces.append(
                {
                    "start": start,
                    "end": end,
                    "text": text,
                    "speaker": speaker,
                    "overlap": _overlap_with_speaker(start, end, diar_turns, speaker),
                    "words": members,
                }
            )
            group_start = i
    return pieces


def _overlap_with_speaker(
    start: float,
    end: float,
    diar_turns: list[dict],
    speaker: str,
) -> float:
    """Seconds of [start, end] covered by turns of `speaker`."""
    total = 0.0
    for t in diar_turns:
        if t["start"] >= end:
            break
        if t["speaker"] != speaker or t["end"] <= start:
            continue
        total += max(0.0, min(end, t["end"]) - max(start, t["start"]))
    return round(total, 3)


__all__ = [
    "DiarizationEngine",
    "assign_speaker_to_segment",
    "assign_speakers_by_words",
    "diarization_params_hash",
    "diarize_audio",
]
