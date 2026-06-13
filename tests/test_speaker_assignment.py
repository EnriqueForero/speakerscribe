"""Word-level speaker attribution: re-segmentation + island smoothing (F3.1)."""

from __future__ import annotations

from speakerscribe.config import SPK_NO_OVERLAP
from speakerscribe.diarization import assign_speaker_to_segment, assign_speakers_by_words

TURNS = [
    {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
    {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_01"},
]


def w(word: str, start: float, end: float) -> dict:
    return {"word": word, "start": start, "end": end, "probability": 0.9}


class TestResegmentation:
    def test_segment_crossing_turn_change_is_split(self):
        """The systematic error fixed by word mode: one Whisper segment, two turns."""
        words = [
            w(" claro", 3.8, 4.2),
            w(" que", 4.2, 4.5),
            w(" sí.", 4.5, 4.9),
            w(" ¿Y", 5.1, 5.4),
            w(" usted", 5.4, 5.8),
            w(" opina?", 5.8, 6.5),
        ]
        pieces = assign_speakers_by_words(words, TURNS)
        assert [p["speaker"] for p in pieces] == ["SPEAKER_00", "SPEAKER_01"]
        assert pieces[0]["text"] == "claro que sí."
        assert pieces[1]["text"] == "¿Y usted opina?"
        # Sub-segment times come from member words
        assert pieces[0]["start"] == 3.8 and pieces[0]["end"] == 4.9
        # Contrast: segment-level attributes EVERYTHING to one speaker
        seg_speaker, _ = assign_speaker_to_segment(3.8, 6.5, TURNS)
        assert seg_speaker in ("SPEAKER_00", "SPEAKER_01")
        assert len({p["speaker"] for p in pieces}) == 2

    def test_single_speaker_segment_stays_whole(self):
        words = [w(" hola", 1.0, 1.4), w(" mundo", 1.5, 2.0)]
        pieces = assign_speakers_by_words(words, TURNS)
        assert len(pieces) == 1
        assert pieces[0]["speaker"] == "SPEAKER_00"

    def test_empty_words_returns_empty(self):
        assert assign_speakers_by_words([], TURNS) == []

    def test_overlap_is_per_piece_and_positive(self):
        words = [w(" hola", 1.0, 1.4), w(" mundo", 1.5, 2.0)]
        pieces = assign_speakers_by_words(words, TURNS)
        assert pieces[0]["overlap"] == 1.0  # (1.0-1.4) + (1.5-2.0) within SPEAKER_00


class TestIslandSmoothing:
    def test_short_dissenting_word_between_equals_is_reassigned(self):
        """Diarization boundary jitter must not create one-word ping-pong turns."""
        words = [
            w(" hola", 4.0, 4.4),
            w(" x", 5.05, 5.15),  # 0.1 s inside SPEAKER_01 — jitter
            w(" mundo", 4.6, 4.9),
        ]
        pieces = assign_speakers_by_words(words, TURNS)
        assert len(pieces) == 1
        assert pieces[0]["speaker"] == "SPEAKER_00"

    def test_long_dissenting_word_is_kept(self):
        words = [
            w(" hola", 4.0, 4.4),
            w(" exactamente", 5.1, 6.0),  # 0.9 s — a real interjection
            w(" mundo", 4.6, 4.9),
        ]
        pieces = assign_speakers_by_words(words, TURNS)
        assert [p["speaker"] for p in pieces] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]


class TestNoOverlapWords:
    def test_words_outside_all_turns(self):
        words = [w(" tarde", 12.0, 12.5), w(" señores", 12.6, 13.0)]
        pieces = assign_speakers_by_words(words, TURNS)
        assert len(pieces) == 1
        assert pieces[0]["speaker"] == SPK_NO_OVERLAP
        assert pieces[0]["overlap"] == 0.0
