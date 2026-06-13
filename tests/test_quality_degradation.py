"""Quality flags introduced/repaired in 0.3: DIARIZATION_FAILED and EMPTY_SEGMENTS."""

from __future__ import annotations

from speakerscribe.quality import evaluate_transcription_quality


def _meta(**over):
    base = {
        "language_detected": "es",
        "language_probability": 0.99,
        "duration_seconds": 600.0,
        "duration_minutes": 10.0,
        "total_words": 1200,
        "total_segments": 8,
        "diarization_enabled": False,
        "segments": [
            {"id": i, "start": i * 70.0, "end": i * 70.0 + 60.0, "text": f"texto distinto {i}"}
            for i in range(8)
        ],
    }
    base.update(over)
    return base


class TestDiarizationFailedFlag:
    def test_failure_is_critical(self):
        report = evaluate_transcription_quality(
            _meta(diarization_failed=True, diarization_error="RuntimeError: token")
        )
        flags = {f.code: f for f in report.flags}
        assert "DIARIZATION_FAILED" in flags
        assert flags["DIARIZATION_FAILED"].severity.value == "CRITICAL"
        assert report.quality_ok is False

    def test_absent_when_not_failed(self):
        report = evaluate_transcription_quality(_meta())
        assert all(f.code != "DIARIZATION_FAILED" for f in report.flags)


class TestEmptySegmentsFlag:
    def test_fires_on_discarded_count(self):
        """Empties never reach `segments`; the count travels in metadata."""
        report = evaluate_transcription_quality(_meta(empty_segments_discarded=3))
        codes = [f.code for f in report.flags]
        assert "EMPTY_SEGMENTS" in codes  # 3 / (8 + 3) = 27% > 10%

    def test_silent_below_threshold(self):
        report = evaluate_transcription_quality(_meta(empty_segments_discarded=0))
        assert all(f.code != "EMPTY_SEGMENTS" for f in report.flags)
        # 2 / (20 + 2) = 9% <= 10% -> silent
        twenty = [
            {"id": i, "start": i * 30.0, "end": i * 30.0 + 25.0, "text": f"texto distinto {i}"}
            for i in range(20)
        ]
        report2 = evaluate_transcription_quality(
            _meta(segments=twenty, total_segments=20, empty_segments_discarded=2)
        )
        assert all(f.code != "EMPTY_SEGMENTS" for f in report2.flags)
