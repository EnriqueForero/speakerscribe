"""Processing-time estimates for planning (Colab T4 reference numbers).

Single source of truth for the RTF table the notebook displays before a
batch (the table used to live duplicated inside notebook cells). These are
PLANNING estimates measured informally on a T4 with the sequential path;
your `_runs.jsonl` ledger contains your REAL per-run RTF — trust that over
this table once you have history. Batched inference (batch_size=8)
typically multiplies these RTFs by ~2-4x on large models
(https://github.com/SYSTRAN/faster-whisper#benchmark).
"""

from __future__ import annotations

RTF_ESTIMATE_T4: dict[str, float] = {
    # model -> minutes of audio processed per minute of wall time (sequential)
    "tiny": 30.0,
    "base": 25.0,
    "small": 15.0,
    "medium": 8.0,
    "large-v2": 4.0,
    "large-v3": 4.0,
    "large-v3-turbo": 10.0,
    "turbo": 10.0,
}

DIARIZATION_RTF_T4: float = 12.0
"""Minutes of audio diarized per minute of wall time on a T4 (pyannote 4.x)."""

_DEFAULT_RTF: float = 4.0  # unknown model: assume the slowest large


def estimate_processing_minutes(
    audio_minutes: float,
    model: str,
    *,
    with_diarization: bool = True,
    batch_speedup: float = 1.0,
) -> float:
    """Estimate wall-clock minutes to process `audio_minutes` of audio.

    Args:
        audio_minutes: Total audio duration in minutes.
        model: Whisper model name (see `RTF_ESTIMATE_T4`).
        with_diarization: Add the diarization pass estimate.
        batch_speedup: Multiplier on ASR throughput for batched inference
            (1.0 = sequential planning numbers; ~2-4 with batch_size=8 on
            large models — calibrate with your own ledger).

    Returns:
        Estimated minutes (>= 0.0). Planning aid, not a promise.

    Examples:
        >>> estimate_processing_minutes(60, "large-v3-turbo", with_diarization=False)
        6.0
    """
    if audio_minutes <= 0:
        return 0.0
    rtf = RTF_ESTIMATE_T4.get(model, _DEFAULT_RTF) * max(batch_speedup, 1e-9)
    total = audio_minutes / rtf
    if with_diarization:
        total += audio_minutes / DIARIZATION_RTF_T4
    return round(total, 1)


__all__ = ["DIARIZATION_RTF_T4", "RTF_ESTIMATE_T4", "estimate_processing_minutes"]
