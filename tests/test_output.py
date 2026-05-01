"""Tests for speakerscribe.output (Markdown transcript + text splitting)."""

from __future__ import annotations

import pytest

from speakerscribe.output import generate_transcript_md, split_text_by_words


class TestGenerateTranscriptMd:
    def test_empty_segments(self, tmp_path, sample_metadata):
        out = tmp_path / "test.transcript.md"
        n = generate_transcript_md([], out, sample_metadata)
        assert n == 0
        assert "Empty transcript" in out.read_text()

    def test_single_speaker_single_block(self, tmp_path, sample_metadata):
        out = tmp_path / "test.transcript.md"
        segs = [
            {"id": 1, "start": 0, "end": 5, "text": "Hello.", "speaker": "SPEAKER_00"},
            {"id": 2, "start": 5, "end": 10, "text": "Yes.", "speaker": "SPEAKER_00"},
        ]
        n = generate_transcript_md(segs, out, sample_metadata)
        assert n == 1
        content = out.read_text()
        assert "SPEAKER_00" in content
        assert "Hello." in content

    def test_different_speakers_two_blocks(self, tmp_path, sample_metadata):
        out = tmp_path / "test.transcript.md"
        segs = [
            {"id": 1, "start": 0, "end": 5, "text": "Hello.", "speaker": "SPEAKER_00"},
            {"id": 2, "start": 5, "end": 10, "text": "Fine.", "speaker": "SPEAKER_01"},
        ]
        n = generate_transcript_md(segs, out, sample_metadata)
        assert n == 2

    def test_long_pause_opens_new_block(self, tmp_path, sample_metadata):
        out = tmp_path / "test.transcript.md"
        # Same speaker but 10s gap → 2 blocks
        segs = [
            {"id": 1, "start": 0, "end": 5, "text": "First.", "speaker": "SPEAKER_00"},
            {"id": 2, "start": 15, "end": 20, "text": "Second.", "speaker": "SPEAKER_00"},
        ]
        n = generate_transcript_md(segs, out, sample_metadata, gap_max_s=2.0)
        assert n == 2

    def test_diarization_disabled_shows_warning(self, tmp_path, sample_metadata):
        out = tmp_path / "test.transcript.md"
        meta = {**sample_metadata, "diarization_enabled": False}
        segs = [{"id": 1, "start": 0, "end": 5, "text": "Hello."}]
        generate_transcript_md(segs, out, meta)
        content = out.read_text()
        assert "Diarization not available" in content or "not available" in content.lower()

    def test_header_includes_versions(self, tmp_path, sample_metadata):
        out = tmp_path / "test.transcript.md"
        segs = [{"id": 1, "start": 0, "end": 5, "text": "Hello.", "speaker": "SPEAKER_00"}]
        generate_transcript_md(segs, out, sample_metadata)
        content = out.read_text()
        assert "faster-whisper" in content
        assert "pyannote" in content


class TestSplitTextByWords:
    def test_missing_file_returns_zero(self, tmp_path):
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        n = split_text_by_words(tmp_path / "does-not-exist.txt", out_dir, 100)
        assert n == 0

    def test_empty_file_returns_zero(self, tmp_path):
        empty = tmp_path / "empty.txt"
        empty.write_text("")
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        n = split_text_by_words(empty, out_dir, 100)
        assert n == 0

    def test_word_split_without_speakers(self, tmp_path):
        file = tmp_path / "no_spk.txt"
        # 50 words, max=20 → 3 files
        file.write_text(" ".join(["word"] * 50))
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        n = split_text_by_words(file, out_dir, 20, has_speakers=False)
        assert n == 3
        assert len(list(out_dir.glob("*.txt"))) == 3

    def test_speaker_split_preserves_turns(self, tmp_path):
        file = tmp_path / "with_spk.txt"
        file.write_text(
            "[SPEAKER_00] one two three four five six seven eight nine ten.\n"
            "[SPEAKER_01] eleven twelve thirteen fourteen fifteen sixteen seventeen.\n"
            "[SPEAKER_00] eighteen nineteen twenty.\n"
        )
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        n = split_text_by_words(file, out_dir, 15, has_speakers=True)
        assert n >= 2
        # No output file should break a speaker turn across files
        for split_file in out_dir.glob("*.txt"):
            for line in split_file.read_text().splitlines():
                if line.startswith("[SPEAKER_"):
                    # Line must be complete (starts and has content)
                    assert len(line.split()) > 1

    def test_auto_infer_speakers(self, tmp_path):
        file = tmp_path / "auto.txt"
        file.write_text("[SPEAKER_00] hello.\n[SPEAKER_01] world.\n")
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        # Without has_speakers, function infers it from content
        n = split_text_by_words(file, out_dir, 100, has_speakers=None)
        assert n >= 1
