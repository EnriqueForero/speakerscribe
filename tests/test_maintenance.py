"""Speaker renaming: single-pass (no chaining), all output formats covered."""

from __future__ import annotations

import json

import pytest

from speakerscribe.config import WorkspacePaths
from speakerscribe.maintenance import rename_speakers_in_outputs


@pytest.fixture
def run_outputs(tmp_path):
    paths = WorkspacePaths(workspace=tmp_path)
    paths.create_directories()
    base = "meeting_large-v3"
    (paths.transcripts / f"{base}.txt").write_text(
        "[SPEAKER_00] hola\n[SPEAKER_01] adiós\n", encoding="utf-8"
    )
    (paths.transcripts / f"{base}.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n[SPEAKER_00] hola\n\n", encoding="utf-8"
    )
    (paths.transcripts / f"{base}.transcript.md").write_text(
        "### SPEAKER_00 · 00:00\n\nhola\n", encoding="utf-8"
    )
    (paths.transcripts / f"{base}.segments.jsonl").write_text(
        json.dumps({"id": 1, "speaker": "SPEAKER_00", "text": "hola"})
        + "\n"
        + json.dumps({"id": 2, "speaker": "SPEAKER_01", "text": "adiós"})
        + "\n",
        encoding="utf-8",
    )
    (paths.transcripts / f"{base}.json").write_text(
        json.dumps(
            {
                "segments": [{"speaker": "SPEAKER_00"}, {"speaker": "SPEAKER_01"}],
                "speakers_summary": {"SPEAKER_00": 1, "SPEAKER_01": 1},
            }
        ),
        encoding="utf-8",
    )
    return paths, base


class TestRenameSpeakers:
    def test_swap_mapping_does_not_chain(self, run_outputs):
        """{00 -> 01, 01 -> Ana} must SWAP, not collapse everything into Ana."""
        paths, base = run_outputs
        rename_speakers_in_outputs(paths, base, {"SPEAKER_00": "SPEAKER_01", "SPEAKER_01": "Ana"})
        txt = (paths.transcripts / f"{base}.txt").read_text()
        assert "[SPEAKER_01] hola" in txt
        assert "[Ana] adiós" in txt
        assert txt.count("[Ana]") == 1

    def test_all_formats_renamed(self, run_outputs):
        paths, base = run_outputs
        stats = rename_speakers_in_outputs(paths, base, {"SPEAKER_00": "Alice"})
        assert "[Alice] hola" in (paths.transcripts / f"{base}.txt").read_text()
        assert "[Alice]" in (paths.transcripts / f"{base}.srt").read_text()
        assert "### Alice ·" in (paths.transcripts / f"{base}.transcript.md").read_text()
        rows = [
            json.loads(line)
            for line in (paths.transcripts / f"{base}.segments.jsonl").read_text().splitlines()
        ]
        assert rows[0]["speaker"] == "Alice" and rows[1]["speaker"] == "SPEAKER_01"
        data = json.loads((paths.transcripts / f"{base}.json").read_text())
        assert data["segments"][0]["speaker"] == "Alice"
        assert data["speakers_summary"] == {"Alice": 1, "SPEAKER_01": 1}
        assert stats[f"{base}.segments.jsonl"] == 1

    def test_invalid_mapping_key_raises(self, run_outputs):
        paths, base = run_outputs
        with pytest.raises(ValueError, match="SPEAKER_"):
            rename_speakers_in_outputs(paths, base, {"BOB": "Alice"})

    def test_empty_mapping_noop(self, run_outputs):
        paths, base = run_outputs
        assert rename_speakers_in_outputs(paths, base, {}) == {}


class TestDeleteAllOutputs:
    def test_sweeps_scratch_and_preserves_ledger(self, tmp_path):
        """Outputs + scratch temporaries go; the runs ledger is audit history
        and survives (reprocessing still happens because outputs are gone)."""
        from speakerscribe.maintenance import delete_all_outputs

        paths = WorkspacePaths(workspace=tmp_path, scratch=str(tmp_path / "scratch"))
        paths.create_directories()
        (paths.transcripts / "a.txt").write_text("x")
        (paths.splits / "a_1.txt").write_text("x")
        (paths.diar_cache / "a.diar.json").write_text("{}")
        (paths.audio_tmp / "a_1234567890.wav").write_bytes(b"w")
        (paths.audio_chunks / "a_chunk000.wav").write_bytes(b"w")
        paths.ledger_path.write_text('{"id": 1}\n')

        deleted = delete_all_outputs(paths, confirm="YES DELETE ALL")
        assert deleted == 5
        assert paths.ledger_path.exists(), "ledger is history, not an output"
        assert not list(paths.audio_chunks.iterdir())

    def test_requires_exact_confirmation(self, tmp_path):
        from speakerscribe.maintenance import delete_all_outputs

        paths = WorkspacePaths(workspace=tmp_path)
        paths.create_directories()
        with pytest.raises(ValueError, match="YES DELETE ALL"):
            delete_all_outputs(paths, confirm="yes")
