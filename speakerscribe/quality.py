"""Heuristic post-transcription quality checker.

Detects problematic transcriptions WITHOUT requiring a human to read them.
Raises severity-tagged flags based on:

- Language detection confidence
- Real-time factor (RTF) anomalies
- Word density (words per minute)
- Speaker distribution anomalies
- Repetition loops (Whisper hallucinations)
- Empty or degenerate segments
- Word dominance (suspicious frequency of a single token)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    """Severity level for a quality flag."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class QualityFlag:
    """A single problem detected in a transcription."""

    severity: Severity
    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.severity.value}] {self.code}: {self.message}"


@dataclass(frozen=True)
class QualityReport:
    """Full quality report for a transcription run."""

    quality_ok: bool
    flags: list[QualityFlag]
    metrics: dict[str, Any]

    @property
    def n_critical(self) -> int:
        return sum(1 for f in self.flags if f.severity == Severity.CRITICAL)

    @property
    def n_warning(self) -> int:
        return sum(1 for f in self.flags if f.severity == Severity.WARNING)

    def summary(self) -> str:
        """Human-readable single-string summary."""
        if self.quality_ok:
            return "Quality OK — no issues detected"

        lines = [
            f"Quality: {self.n_critical} critical, {self.n_warning} warning(s)",
        ]
        for flag in sorted(
            self.flags,
            key=lambda f: ["CRITICAL", "WARNING", "INFO"].index(f.severity.value),
        ):
            lines.append(f"  - {flag}")
        return "\n".join(lines)


# ─── Thresholds ─────────────────────────────────────────────────────
class _Thresholds:
    """Calibrated thresholds. Tune for your domain if needed."""

    LANG_PROB_MIN: float = 0.85
    """Minimum language detection confidence."""

    RTF_MIN_GPU: float = 2.0
    """RTF below this on GPU likely indicates a problem (target is >5x)."""

    RTF_MAX_SUSPICIOUS: float = 100.0
    """RTF above this likely indicates the audio is mostly silence."""

    WPM_MIN: float = 60.0
    """Minimum words per minute (normal speech is 100-180 wpm)."""

    WPM_MAX: float = 250.0
    """Maximum words per minute. Above this points to hallucination."""

    SPEAKER_DOM_MAX: float = 0.95
    """If one speaker has >95% of segments, diarization is poor."""

    N_SPEAKERS_MAX: int = 8
    """More than N detected speakers suggests over-segmentation."""

    NGRAM_REPEAT_SIZE: int = 5
    """Size (in words) of n-grams used to detect Whisper loops."""

    NGRAM_REPEAT_TIMES: int = 3
    """Minimum consecutive repetitions to flag as a loop."""


def _detect_anomalous_repetitions(
    segments: list[dict],
    n: int = _Thresholds.NGRAM_REPEAT_SIZE,
    min_times: int = _Thresholds.NGRAM_REPEAT_TIMES,
) -> list[str]:
    """Detect n-grams that repeat consecutively more than `min_times` times.

    Whisper occasionally falls into loops, producing the same phrase repeatedly.

    Args:
        segments: List of dicts containing a `text` key.
        n: N-gram size in words.
        min_times: Minimum repetition count to flag.

    Returns:
        List of offending n-gram strings.
    """
    full_text = " ".join(s.get("text", "") for s in segments).lower().split()
    if len(full_text) < n * min_times:
        return []

    flagged: list[str] = []
    i = 0
    while i <= len(full_text) - n * min_times:
        ngram = tuple(full_text[i : i + n])
        # Count consecutive repetitions
        times = 1
        j = i + n
        while j <= len(full_text) - n and tuple(full_text[j : j + n]) == ngram:
            times += 1
            j += n
        if times >= min_times:
            flagged.append(" ".join(ngram))
            i = j  # skip past the repeated block
        else:
            i += 1
    return flagged


def evaluate_transcription_quality(metadata: dict[str, Any]) -> QualityReport:
    """Run heuristic checks against transcription metadata.

    All checks are optional: if a metric is missing in `metadata`, the check
    is skipped silently.

    Args:
        metadata: Run metadata, typically the dict returned by
            `transcribe_streaming`.

    Returns:
        QualityReport with the flags and computed metrics.
    """
    flags: list[QualityFlag] = []
    metrics: dict[str, Any] = {}

    # ── 0. Diarization failure (degraded run) ──────────────────────
    # The pipeline sets `diarization_failed=True` when diarization was
    # REQUESTED but raised: the user asked for speakers and got none.
    # That is a CRITICAL quality event, not a silent "ok".
    if metadata.get("diarization_failed"):
        flags.append(
            QualityFlag(
                severity=Severity.CRITICAL,
                code="DIARIZATION_FAILED",
                message=(
                    "Diarization was enabled but failed — transcript has NO "
                    "speaker labels. Run is degraded; it will be retried on "
                    "the next batch (not skipped)."
                ),
                context={"error": metadata.get("diarization_error")},
            )
        )

    # ── 1. Language confidence ─────────────────────────────────────
    lang_prob = metadata.get("language_probability")
    if lang_prob is not None:
        metrics["language_probability"] = lang_prob
        if lang_prob < _Thresholds.LANG_PROB_MIN:
            flags.append(
                QualityFlag(
                    severity=Severity.WARNING,
                    code="LOW_LANG_CONFIDENCE",
                    message=f"Low language detection confidence: {lang_prob:.0%}",
                    context={"language_probability": lang_prob},
                )
            )

    # ── 2. RTF ─────────────────────────────────────────────────────
    rtf = metadata.get("real_time_factor")
    if rtf is not None:
        metrics["real_time_factor"] = rtf
        if rtf < _Thresholds.RTF_MIN_GPU:
            flags.append(
                QualityFlag(
                    severity=Severity.WARNING,
                    code="LOW_RTF",
                    message=f"Very low RTF ({rtf:.1f}x) — possible GPU/CPU issue",
                    context={"rtf": rtf},
                )
            )
        if rtf > _Thresholds.RTF_MAX_SUSPICIOUS:
            flags.append(
                QualityFlag(
                    severity=Severity.INFO,
                    code="HIGH_RTF",
                    message=f"Very high RTF ({rtf:.1f}x) — audio mostly silent?",
                    context={"rtf": rtf},
                )
            )

    # ── 3. Word density ────────────────────────────────────────────
    words = metadata.get("total_words", 0)
    duration_min = metadata.get("duration_minutes", 0)
    if words > 0 and duration_min > 0:
        wpm = words / duration_min
        metrics["wpm"] = round(wpm, 1)
        if wpm < _Thresholds.WPM_MIN:
            flags.append(
                QualityFlag(
                    severity=Severity.WARNING,
                    code="LOW_WPM",
                    message=f"Low density ({wpm:.0f} wpm) — long silences or aggressive VAD",
                    context={"wpm": wpm},
                )
            )
        if wpm > _Thresholds.WPM_MAX:
            flags.append(
                QualityFlag(
                    severity=Severity.CRITICAL,
                    code="HIGH_WPM",
                    message=f"High density ({wpm:.0f} wpm) — possible Whisper hallucination",
                    context={"wpm": wpm},
                )
            )

    # ── 4. Speaker distribution ────────────────────────────────────
    speakers = metadata.get("speakers_summary")
    if speakers and isinstance(speakers, dict) and len(speakers) > 0:
        total = sum(speakers.values())
        if total > 0:
            max_count = max(speakers.values())
            max_pct = max_count / total
            n_spk = len(speakers)
            metrics["n_speakers"] = n_spk
            metrics["speaker_dom_pct"] = round(max_pct, 3)

            if max_pct > _Thresholds.SPEAKER_DOM_MAX and n_spk > 1:
                flags.append(
                    QualityFlag(
                        severity=Severity.WARNING,
                        code="SPEAKER_DOMINANCE",
                        message=(
                            f"One speaker dominates {max_pct:.0%} of segments — poor diarization"
                        ),
                        context={"max_pct": max_pct, "n_speakers": n_spk},
                    )
                )
            if n_spk > _Thresholds.N_SPEAKERS_MAX:
                flags.append(
                    QualityFlag(
                        severity=Severity.WARNING,
                        code="TOO_MANY_SPEAKERS",
                        message=(
                            f"Detected {n_spk} speakers — possible over-segmentation. "
                            f"Consider pinning `num_speakers`."
                        ),
                        context={"n_speakers": n_spk},
                    )
                )
            # Speakers with very few segments are likely false positives
            tiny_speakers = [k for k, v in speakers.items() if v <= 2]
            if len(tiny_speakers) >= 2:
                flags.append(
                    QualityFlag(
                        severity=Severity.INFO,
                        code="TINY_SPEAKERS",
                        message=(
                            f"{len(tiny_speakers)} speakers with <=2 segments: "
                            f"{tiny_speakers} — likely false positives"
                        ),
                        context={"tiny_speakers": tiny_speakers},
                    )
                )

    # ── 5. Repetition loops ────────────────────────────────────────
    segments = metadata.get("segments", [])
    if segments:
        repetitions = _detect_anomalous_repetitions(segments)
        if repetitions:
            flags.append(
                QualityFlag(
                    severity=Severity.CRITICAL,
                    code="REPETITIONS",
                    message=(
                        f"Detected {len(repetitions)} consecutive repeated sequences — "
                        f"Whisper hallucination"
                    ),
                    context={"examples": repetitions[:3]},
                )
            )

    # ── 6. Empty / degenerate segments ─────────────────────────────
    # Empty segments never reach metadata["segments"] (the writer discards
    # them), so the count travels in `empty_segments_discarded` — set by
    # the transcription layer. The denominator is what Whisper EMITTED.
    empty_count = metadata.get("empty_segments_discarded", 0)
    emitted = len(segments) + empty_count
    metrics["empty_segments_discarded"] = empty_count
    if emitted > 0 and empty_count / emitted > 0.1:
        flags.append(
            QualityFlag(
                severity=Severity.WARNING,
                code="EMPTY_SEGMENTS",
                message=(
                    f"{empty_count} empty segments discarded "
                    f"({empty_count / emitted:.0%} of emitted) — VAD too aggressive?"
                ),
                context={"empty_count": empty_count, "emitted": emitted},
            )
        )

        # Single-word dominance detection
        joined = " ".join((s.get("text") or "").lower() for s in segments)
        if joined:
            tokens = joined.split()
            if len(tokens) >= 50:
                most_common, count = Counter(tokens).most_common(1)[0]
                if count / len(tokens) > 0.15 and len(most_common) > 2:
                    flags.append(
                        QualityFlag(
                            severity=Severity.WARNING,
                            code="WORD_DOMINANCE",
                            message=(
                                f"Word '{most_common}' appears "
                                f"{count}/{len(tokens)} times "
                                f"({count / len(tokens):.0%}) — possible hallucination"
                            ),
                            context={"word": most_common, "frequency": count},
                        )
                    )

    quality_ok = not any(f.severity in (Severity.WARNING, Severity.CRITICAL) for f in flags)

    return QualityReport(
        quality_ok=quality_ok,
        flags=flags,
        metrics=metrics,
    )


__all__ = [
    "QualityFlag",
    "QualityReport",
    "Severity",
    "evaluate_transcription_quality",
]
