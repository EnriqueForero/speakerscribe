"""WER/DER benchmarking (extras-gated; pure-python parts always run)."""

from __future__ import annotations

import pytest

from speakerscribe.evaluate import normalize_for_wer, parse_rttm, write_rttm


class TestNormalize:
    def test_strips_speaker_labels_case_and_punct(self):
        raw = "[SPEAKER_00] Hola, ¿CÓMO estás?\n[Ana] Bien."
        assert normalize_for_wer(raw) == "hola cómo estás bien"

    def test_accents_kept(self):
        assert normalize_for_wer("Canción única") == "canción única"


class TestRttmRoundtrip:
    def test_write_then_parse(self, tmp_path):
        turns = [
            {"start": 0.0, "end": 2.5, "speaker": "A"},
            {"start": 2.5, "end": 4.0, "speaker": "B"},
        ]
        p = write_rttm(turns, uri="meet", path=tmp_path / "ref.rttm")
        parsed = parse_rttm(p)
        assert parsed == [
            {"start": 0.0, "end": 2.5, "speaker": "A"},
            {"start": 2.5, "end": 4.0, "speaker": "B"},
        ]

    def test_parse_missing_or_empty(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_rttm(tmp_path / "nope.rttm")
        bad = tmp_path / "bad.rttm"
        bad.write_text("not an rttm line\n")
        with pytest.raises(ValueError):
            parse_rttm(bad)


class TestWer:
    def test_perfect_match_is_zero(self):
        pytest.importorskip("jiwer")
        from speakerscribe.evaluate import compute_wer

        r = compute_wer("hola mundo cruel", "[SPEAKER_00] Hola mundo cruel")
        assert r["wer"] == 0.0
        assert r["reference_words"] == 3

    def test_known_error_rate(self):
        pytest.importorskip("jiwer")
        from speakerscribe.evaluate import compute_wer

        r = compute_wer("uno dos tres cuatro", "uno dos tres cinco")
        assert r["wer"] == 0.25

    def test_empty_reference_raises(self):
        pytest.importorskip("jiwer")
        from speakerscribe.evaluate import compute_wer

        with pytest.raises(ValueError):
            compute_wer("...", "algo")


class TestDer:
    def test_perfect_diarization_is_zero(self):
        pytest.importorskip("pyannote.metrics")
        from speakerscribe.evaluate import compute_der

        turns = [
            {"start": 0.0, "end": 5.0, "speaker": "A"},
            {"start": 5.0, "end": 10.0, "speaker": "B"},
        ]
        r = compute_der(turns, turns, collar=0.0)
        assert r["der"] == 0.0

    def test_label_permutation_is_still_zero(self):
        """DER uses optimal label mapping: swapped names are not errors."""
        pytest.importorskip("pyannote.metrics")
        from speakerscribe.evaluate import compute_der

        ref = [
            {"start": 0.0, "end": 5.0, "speaker": "A"},
            {"start": 5.0, "end": 10.0, "speaker": "B"},
        ]
        hyp = [
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_01"},
            {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_00"},
        ]
        assert compute_der(ref, hyp, collar=0.0)["der"] == 0.0

    def test_confusion_detected(self):
        pytest.importorskip("pyannote.metrics")
        from speakerscribe.evaluate import compute_der

        ref = [
            {"start": 0.0, "end": 5.0, "speaker": "A"},
            {"start": 5.0, "end": 10.0, "speaker": "B"},
        ]
        hyp = [{"start": 0.0, "end": 10.0, "speaker": "X"}]  # one speaker for all
        r = compute_der(ref, hyp, collar=0.0)
        assert r["der"] == pytest.approx(0.5, abs=1e-6)
        assert r["confusion"] == pytest.approx(5.0, abs=1e-6)


class TestBenchRun:
    def test_missing_outputs_raise(self, tmp_path):
        pytest.importorskip("jiwer")
        from speakerscribe.config import WorkspacePaths
        from speakerscribe.evaluate import bench_run

        paths = WorkspacePaths(workspace=tmp_path)
        paths.create_directories()
        ref = tmp_path / "ref.txt"
        ref.write_text("hola")
        with pytest.raises(FileNotFoundError):
            bench_run(paths, "missing_base", ref)

    def test_end_to_end_with_ledger_record(self, tmp_path):
        pytest.importorskip("jiwer")
        import json

        from speakerscribe.config import WorkspacePaths
        from speakerscribe.evaluate import bench_run

        paths = WorkspacePaths(workspace=tmp_path)
        paths.create_directories()
        base = "m_large-v3"
        (paths.transcripts / f"{base}.txt").write_text("[SPEAKER_00] hola mundo\n")
        ref = tmp_path / "ref.txt"
        ref.write_text("hola mundo")
        result = bench_run(paths, base, ref)
        assert result["wer"] == 0.0
        rows = [json.loads(line) for line in paths.ledger_path.read_text().splitlines()]
        assert rows[-1]["kind"] == "bench" and rows[-1]["base_name"] == base
