"""Human-friendly reporting helpers for the pipeline."""

from __future__ import annotations

from speakerscribe.logging_config import logger


def report_speaker_distribution(speakers_summary: dict[str, int]) -> None:
    """Log a visual histogram of segment counts per speaker."""
    if not speakers_summary:
        return
    total = sum(speakers_summary.values())
    if total == 0:
        return
    items = sorted(speakers_summary.items(), key=lambda kv: -kv[1])
    max_label = max(len(k) for k, _ in items)
    logger.info("Segment distribution by speaker:")
    for spk, n in items:
        pct = n / total
        bar = "#" * int(pct * 30)
        logger.info(f"   {spk:<{max_label}}  {n:>5,}  {pct:>6.1%}  {bar}")
