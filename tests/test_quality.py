"""Tests for speakerscribe.quality."""

from __future__ import annotations

import pytest

from speakerscribe.quality import (
    QualityFlag,
    Severity,
    evaluate_transcription_quality,
)


def _good_metadata() -> dict:
    """Metadata representing a clean, problem-free transcription."""
    return {
        "language_probability": 0.99,
        "real_time_factor": 8.5,
        "total_words": 1500,
        "duration_minutes": 10.0,  # 150 wpm — normal speech
        "speakers_summary": {"SPEAKER_00": 60, "SPEAKER_01": 40},
        "segments": [
            # Use varied text to avoid triggering WORD_DOMINANCE heuristic
            {"id": i, "text": f"The speaker said something meaningful on topic {i}.",
             "start": float(i), "end": float(i + 1)}
            for i in range(50)
        ],
    }


class TestQualityChecks:
    def test_good_metadata_passes(self):
        report = evaluate_transcription_quality(_good_metadata())
        assert report.quality_ok is True
        assert report.n_critical == 0
        assert report.n_warning == 0

    def test_low_language_confidence_warning(self):
        meta = _good_metadata()
        meta["language_probability"] = 0.5
        report = evaluate_transcription_quality(meta)
        assert not report.quality_ok
        assert any(f.code == "LOW_LANG_CONFIDENCE" for f in report.flags)

    def test_very_low_rtf_warning(self):
        meta = _good_metadata()
        meta["real_time_factor"] = 0.5
        report = evaluate_transcription_quality(meta)
        assert any(f.code == "LOW_RTF" for f in report.flags)

    def test_very_high_wpm_critical(self):
        meta = _good_metadata()
        # 5000 words in 10 min = 500 wpm → CRITICAL
        meta["total_words"] = 5000
        report = evaluate_transcription_quality(meta)
        critical = [f for f in report.flags if f.severity == Severity.CRITICAL]
        assert any(f.code == "HIGH_WPM" for f in critical)

    def test_very_low_wpm_warning(self):
        meta = _good_metadata()
        # 100 words in 10 min = 10 wpm
        meta["total_words"] = 100
        meta["duration_minutes"] = 10.0
        report = evaluate_transcription_quality(meta)
        assert any(f.code == "LOW_WPM" for f in report.flags)

    def test_dominant_speaker_warning(self):
        meta = _good_metadata()
        # SPEAKER_00 with 99% of segments
        meta["speakers_summary"] = {"SPEAKER_00": 99, "SPEAKER_01": 1}
        report = evaluate_transcription_quality(meta)
        assert any(f.code == "SPEAKER_DOMINANCE" for f in report.flags)

    def test_too_many_speakers_warning(self):
        meta = _good_metadata()
        meta["speakers_summary"] = {f"SPEAKER_{i:02d}": 10 for i in range(15)}
        report = evaluate_transcription_quality(meta)
        assert any(f.code == "TOO_MANY_SPEAKERS" for f in report.flags)

    def test_tiny_speakers_info(self):
        meta = _good_metadata()
        # Multiple speakers with very few segments (false positives)
        meta["speakers_summary"] = {
            "SPEAKER_00": 100,
            "SPEAKER_01": 1,
            "SPEAKER_02": 2,
        }
        report = evaluate_transcription_quality(meta)
        assert any(f.code == "TINY_SPEAKERS" for f in report.flags)

    def test_consecutive_repetitions_critical(self):
        """Whisper hallucination: same 5-word phrase repeated 4 times consecutively."""
        meta = _good_metadata()
        # Phrase must be exactly NGRAM_REPEAT_SIZE (5) words for the algorithm to detect
        repeated = "hello world this is repeated"
        meta["segments"] = [
            {"id": 1, "text": repeated, "start": 0.0, "end": 5.0},
            {"id": 2, "text": repeated, "start": 5.0, "end": 10.0},
            {"id": 3, "text": repeated, "start": 10.0, "end": 15.0},
            {"id": 4, "text": repeated, "start": 15.0, "end": 20.0},
        ]
        report = evaluate_transcription_quality(meta)
        assert any(f.code == "REPETITIONS" for f in report.flags)

    def test_minimal_metadata_does_not_crash(self):
        """With minimal metadata, quality checker must not raise."""
        report = evaluate_transcription_quality({})
        assert report.quality_ok is True

    def test_summary_is_readable(self):
        meta = _good_metadata()
        meta["language_probability"] = 0.5
        report = evaluate_transcription_quality(meta)
        summary = report.summary()
        assert "LOW_LANG_CONFIDENCE" in summary


class TestQualityFlag:
    def test_str_format(self):
        flag = QualityFlag(
            severity=Severity.WARNING,
            code="TEST",
            message="Test message",
        )
        assert "WARNING" in str(flag)
        assert "TEST" in str(flag)
        assert "Test message" in str(flag)
