"""Tests for speakerscribe.output (v0.2.0)."""

from __future__ import annotations

import pytest

from speakerscribe.output import (
    generate_transcript_md,
    is_filler_only,
    split_text_by_words,
    write_unified_for_llm,
)


class TestIsFillerOnly:
    def test_basic_es(self):
        assert is_filler_only("eh", "es") is True
        assert is_filler_only("Eh.", "es") is True
        assert is_filler_only("EHM,", "es") is True
        assert is_filler_only("o sea", "es") is True

    def test_basic_en(self):
        assert is_filler_only("uh", "en") is True
        assert is_filler_only("Um.", "en") is True
        assert is_filler_only("you know", "en") is True

    def test_real_text_not_filler(self):
        assert is_filler_only("Hola, ¿cómo estás?", "es") is False
        assert is_filler_only("This is a real sentence.", "en") is False

    def test_empty_text(self):
        assert is_filler_only("", "es") is False
        assert is_filler_only("   ", "es") is True  # only whitespace becomes empty after strip

    def test_no_language(self):
        """No language -> no filtering."""
        assert is_filler_only("eh", None) is False

    def test_unsupported_language(self):
        """Language without filler list -> no filtering."""
        assert is_filler_only("eh", "zh") is False
        assert is_filler_only("eh", "ja") is False

    def test_punctuation_stripped(self):
        assert is_filler_only("¿eh?", "es") is True
        assert is_filler_only("...uhm...", "en") is True

    def test_partial_filler_not_marked(self):
        """A sentence containing fillers but with real content is NOT a filler-only segment."""
        assert is_filler_only("Eh, no estoy seguro de eso.", "es") is False


class TestSplitTextByWords:
    def test_no_speakers_word_split(self, tmp_path):
        src = tmp_path / "a.txt"
        src.write_text("uno dos tres cuatro cinco seis")
        out_dir = tmp_path / "out"
        n = split_text_by_words(src, out_dir, max_words=2)
        assert n == 3
        assert (out_dir / "a_1.txt").read_text() == "uno dos"
        assert (out_dir / "a_2.txt").read_text() == "tres cuatro"
        assert (out_dir / "a_3.txt").read_text() == "cinco seis"

    def test_with_speakers_preserves_turns(self, tmp_path):
        src = tmp_path / "b.txt"
        src.write_text(
            "[SPEAKER_00] Hola buenas tardes\n"
            "[SPEAKER_01] Cómo estás\n"
            "[SPEAKER_00] Bien y tú\n"
        )
        out_dir = tmp_path / "out"
        n = split_text_by_words(src, out_dir, max_words=4, has_speakers=True)
        # Turns are never split mid-sentence
        files = sorted(out_dir.glob("b_*.txt"))
        assert len(files) == n
        # Every file must contain whole [SPEAKER_XX] lines
        for f in files:
            for line in f.read_text().splitlines():
                if line.strip():
                    assert line.startswith("[SPEAKER_")

    def test_empty_file_returns_zero(self, tmp_path):
        src = tmp_path / "empty.txt"
        src.write_text("")
        n = split_text_by_words(src, tmp_path / "out", max_words=10)
        assert n == 0

    def test_missing_file_returns_zero(self, tmp_path):
        n = split_text_by_words(tmp_path / "no.txt", tmp_path / "out", max_words=10)
        assert n == 0


class TestWriteUnifiedForLlm:
    def test_writes_unified_file(self, tmp_path):
        src = tmp_path / "x.txt"
        src.write_text("[SPEAKER_00] Hola\n[SPEAKER_01] Mundo")
        out_dir = tmp_path / "out"
        target = write_unified_for_llm(src, out_dir)
        assert target is not None
        assert target.name == "x_full_for_llm.txt"
        assert target.read_text().startswith("[SPEAKER_00]")

    def test_missing_source_returns_none(self, tmp_path):
        target = write_unified_for_llm(tmp_path / "no.txt", tmp_path / "out")
        assert target is None

    def test_empty_source_returns_none(self, tmp_path):
        src = tmp_path / "empty.txt"
        src.write_text("")
        target = write_unified_for_llm(src, tmp_path / "out")
        assert target is None


class TestGenerateTranscriptMd:
    @pytest.fixture
    def sample_metadata(self):
        return {
            "audio_file": "test.wav",
            "processed_at": "2026-05-01T12:00:00+00:00",
            "package_version": "0.2.0",
            "pyannote_version": "4.0.4",
            "faster_whisper_version": "1.2.1",
            "model": "large-v3",
            "language_detected": "es",
            "duration_minutes": 10.0,
            "total_words": 6,
            "diarization_enabled": True,
            "diarization_model": "pyannote/speaker-diarization-community-1",
            "speakers_summary": {"SPEAKER_00": 2, "SPEAKER_01": 1},
        }

    def test_basic_generation(self, tmp_path, sample_metadata):
        segments = [
            {"id": 1, "start": 0.0, "end": 5.0, "text": "Hola.", "speaker": "SPEAKER_00"},
            {"id": 2, "start": 5.0, "end": 10.0, "text": "Cómo estás.", "speaker": "SPEAKER_00"},
            {"id": 3, "start": 10.0, "end": 15.0, "text": "Bien.", "speaker": "SPEAKER_01"},
        ]
        out = tmp_path / "out.md"
        n = generate_transcript_md(segments, out, sample_metadata, gap_max_s=2.0)
        assert n == 2  # two turns: SPEAKER_00 (continuous), SPEAKER_01
        content = out.read_text()
        assert "SPEAKER_00" in content
        assert "SPEAKER_01" in content
        assert "Hola." in content

    def test_filler_filter_removes_filler_only_segments(self, tmp_path, sample_metadata):
        # Fillers ES incluidos en config: eh, ehm, em, mmm, mhm, ajá, aja, este,
        # pues, o sea, como que, bueno, claro, sí, no, ok, vale
        segments = [
            {"id": 1, "start": 0.0, "end": 1.0, "text": "Eh.", "speaker": "SPEAKER_00"},
            {"id": 2, "start": 1.0, "end": 5.0, "text": "Hola buenas.", "speaker": "SPEAKER_00"},
            {"id": 3, "start": 5.0, "end": 6.0, "text": "Ehm.", "speaker": "SPEAKER_01"},
        ]
        out = tmp_path / "out.md"
        n = generate_transcript_md(
            segments, out, sample_metadata, gap_max_s=10.0, remove_fillers=True
        )
        content = out.read_text()
        assert "Hola buenas" in content
        # Filler-only segments removed: SPEAKER_01 had only "Ehm." -> filtered out
        assert "### SPEAKER_01" not in content
        assert n == 1  # solo queda el bloque de SPEAKER_00 con "Hola buenas."

    def test_filler_filter_disabled_keeps_all(self, tmp_path, sample_metadata):
        segments = [
            {"id": 1, "start": 0.0, "end": 1.0, "text": "Eh.", "speaker": "SPEAKER_00"},
            {"id": 2, "start": 5.0, "end": 6.0, "text": "Ehm.", "speaker": "SPEAKER_01"},
        ]
        out = tmp_path / "out.md"
        n = generate_transcript_md(
            segments, out, sample_metadata, gap_max_s=2.0, remove_fillers=False
        )
        assert n == 2

    def test_empty_segments(self, tmp_path, sample_metadata):
        out = tmp_path / "out.md"
        n = generate_transcript_md([], out, sample_metadata)
        assert n == 0
        assert out.exists()

    def test_no_diarization_header(self, tmp_path, sample_metadata):
        sample_metadata["diarization_enabled"] = False
        segments = [
            {"id": 1, "start": 0.0, "end": 5.0, "text": "Hola."},
        ]
        out = tmp_path / "out.md"
        generate_transcript_md(segments, out, sample_metadata)
        content = out.read_text()
        assert "Diarization not available" in content
