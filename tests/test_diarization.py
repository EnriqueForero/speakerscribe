"""Tests for speakerscribe.diarization (v0.2.0)."""

from __future__ import annotations

from speakerscribe.config import SPK_NO_OVERLAP, TranscriptionConfig
from speakerscribe.diarization import (
    assign_speaker_to_segment,
    diarization_params_hash,
)


class TestAssignSpeakerToSegment:
    def test_no_turns_returns_no_overlap(self):
        spk, ov = assign_speaker_to_segment(0, 10, [])
        assert spk == SPK_NO_OVERLAP
        assert ov == 0.0

    def test_invalid_segment_returns_no_overlap(self):
        turns = [{"start": 0.0, "end": 100.0, "speaker": "SPEAKER_00"}]
        spk, ov = assign_speaker_to_segment(10, 5, turns)  # end < start
        assert spk == SPK_NO_OVERLAP

    def test_max_overlap_wins(self):
        turns = [
            {"start": 0.0, "end": 10.0, "speaker": "SPEAKER_00"},
            {"start": 8.0, "end": 20.0, "speaker": "SPEAKER_01"},
        ]
        spk, ov = assign_speaker_to_segment(5, 15, turns)
        assert spk == "SPEAKER_01"  # 8-15 = 7s vs SPEAKER_00 5-10 = 5s
        assert ov == 7.0

    def test_no_overlap_returns_no_overlap_label(self):
        turns = [{"start": 100.0, "end": 200.0, "speaker": "SPEAKER_00"}]
        spk, ov = assign_speaker_to_segment(0, 10, turns)
        assert spk == SPK_NO_OVERLAP
        assert ov == 0.0

    def test_early_exit_optimization_correctness(self):
        """Verify that the early-exit on sorted turns is correct."""
        turns = [
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
            {"start": 6.0, "end": 8.0, "speaker": "SPEAKER_01"},
            {"start": 100.0, "end": 200.0, "speaker": "SPEAKER_02"},
        ]
        spk, _ = assign_speaker_to_segment(2, 7, turns)
        assert spk == "SPEAKER_00"  # max overlap is 3s


class TestDiarizationParamsHash:
    """v0.2.0: cache invalidation by parameter hash."""

    def test_same_params_same_hash(self):
        h1 = diarization_params_hash(TranscriptionConfig(num_speakers=2))
        h2 = diarization_params_hash(TranscriptionConfig(num_speakers=2))
        assert h1 == h2

    def test_different_num_speakers_different_hash(self):
        h1 = diarization_params_hash(TranscriptionConfig(num_speakers=2))
        h2 = diarization_params_hash(TranscriptionConfig(num_speakers=4))
        assert h1 != h2

    def test_different_min_max_different_hash(self):
        h1 = diarization_params_hash(
            TranscriptionConfig(min_speakers=2, max_speakers=4)
        )
        h2 = diarization_params_hash(
            TranscriptionConfig(min_speakers=2, max_speakers=8)
        )
        assert h1 != h2

    def test_different_model_different_hash(self):
        h1 = diarization_params_hash(
            TranscriptionConfig(diarization_model="pyannote/speaker-diarization-community-1")
        )
        h2 = diarization_params_hash(
            TranscriptionConfig(diarization_model="pyannote/speaker-diarization-3.1")
        )
        assert h1 != h2

    def test_hash_length_8(self):
        h = diarization_params_hash(TranscriptionConfig())
        assert len(h) == 8

    def test_unrelated_params_dont_change_hash(self):
        """Changing model (Whisper) or beam_size must NOT invalidate the cache."""
        h1 = diarization_params_hash(
            TranscriptionConfig(model="large-v3", beam_size=5, num_speakers=3)
        )
        h2 = diarization_params_hash(
            TranscriptionConfig(model="large-v3-turbo", beam_size=1, num_speakers=3)
        )
        assert h1 == h2
