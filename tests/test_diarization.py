"""Tests for speakerscribe.diarization (overlap-based speaker assignment)."""

from __future__ import annotations

import pytest

from speakerscribe.config import SPK_NO_OVERLAP
from speakerscribe.diarization import assign_speaker_to_segment


class TestAssignSpeakerToSegment:
    def test_empty_turns_returns_no_overlap(self):
        assert assign_speaker_to_segment(0, 10, []) == (SPK_NO_OVERLAP, 0.0)

    def test_inverted_segment_returns_no_overlap(self):
        turns = [{"start": 0, "end": 30, "speaker": "A"}]
        assert assign_speaker_to_segment(20, 10, turns) == (SPK_NO_OVERLAP, 0.0)

    def test_zero_length_segment_returns_no_overlap(self):
        turns = [{"start": 0, "end": 30, "speaker": "A"}]
        assert assign_speaker_to_segment(15, 15, turns) == (SPK_NO_OVERLAP, 0.0)

    def test_full_overlap(self):
        turns = [{"start": 0, "end": 100, "speaker": "A"}]
        spk, ov = assign_speaker_to_segment(10, 20, turns)
        assert spk == "A"
        assert ov == 10.0

    def test_partial_overlap_max_wins(self):
        turns = [
            {"start": 0, "end": 8, "speaker": "A"},   # 8s overlap with [0,10]
            {"start": 8, "end": 12, "speaker": "B"},  # 2s overlap with [0,10]
        ]
        spk, ov = assign_speaker_to_segment(0, 10, turns)
        assert spk == "A"
        assert ov == 8.0

    def test_no_overlap_turn_before(self):
        turns = [{"start": 100, "end": 200, "speaker": "A"}]
        assert assign_speaker_to_segment(0, 50, turns) == (SPK_NO_OVERLAP, 0.0)

    def test_no_overlap_turn_after(self):
        turns = [{"start": 0, "end": 10, "speaker": "A"}]
        assert assign_speaker_to_segment(100, 200, turns) == (SPK_NO_OVERLAP, 0.0)

    def test_early_exit_works_with_sorted_turns(self):
        """Sorted turns allow early exit when a turn starts after the segment ends."""
        turns = [
            {"start": 0, "end": 5, "speaker": "A"},
            {"start": 100, "end": 105, "speaker": "B"},
            {"start": 200, "end": 205, "speaker": "C"},
        ]
        spk, ov = assign_speaker_to_segment(2, 4, turns)
        assert spk == "A"
        assert ov == 2.0

    def test_same_speaker_multiple_turns_accumulates(self):
        turns = [
            {"start": 0, "end": 5, "speaker": "A"},
            {"start": 7, "end": 9, "speaker": "B"},
            {"start": 10, "end": 15, "speaker": "A"},
        ]
        # Segment [0,15]: A has 5+5=10 seconds, B has 2 seconds → A wins
        spk, ov = assign_speaker_to_segment(0, 15, turns)
        assert spk == "A"
        assert ov == 10.0

    @pytest.mark.parametrize(
        "seg_start,seg_end,expected_spk,expected_ov",
        [
            (0, 5, "A", 5.0),
            (3, 8, "A", 5.0),   # full overlap with A only
            (8, 12, "B", 4.0),  # full overlap with B only
            (5, 8, "A", 3.0),   # only A in this range
        ],
    )
    def test_parametrized(self, seg_start, seg_end, expected_spk, expected_ov):
        turns = [
            {"start": 0, "end": 8, "speaker": "A"},
            {"start": 8, "end": 12, "speaker": "B"},
        ]
        spk, ov = assign_speaker_to_segment(seg_start, seg_end, turns)
        assert spk == expected_spk
        assert abs(ov - expected_ov) < 0.001
