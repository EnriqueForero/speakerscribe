"""Transcription core via fakes: writer, batching, OOM ladder, word mode.

No GPU, no real faster-whisper: `make_fake_faster_whisper()` is injected
into sys.modules so the lazy imports resolve to CPU fakes (see fakes.py).
"""

from __future__ import annotations

import json
import sys

import pytest

from speakerscribe.config import TranscriptionConfig
from speakerscribe.transcription import (
    _batch_size_plan,
    _build_transcribe_kwargs,
    _is_cuda_oom,
    transcribe_streaming,
)
from tests.fakes import (
    TWO_SPEAKER_TURNS,
    FakeBatchedPipeline,
    FakeModel,
    FakeSegment,
    make_fake_faster_whisper,
)


@pytest.fixture(autouse=True)
def fake_fw(monkeypatch):
    """Every test in this module runs against the fake faster_whisper."""
    monkeypatch.setitem(sys.modules, "faster_whisper", make_fake_faster_whisper())


@pytest.fixture
def outs(tmp_path):
    return {
        "txt": tmp_path / "a.txt",
        "srt": tmp_path / "a.srt",
        "json": tmp_path / "a.json",
        "jsonl": tmp_path / "a.segments.jsonl",
    }


def _run(config: TranscriptionConfig, outs, model=None, turns=TWO_SPEAKER_TURNS):
    model = model or FakeModel()
    meta = transcribe_streaming(
        model,
        outs["txt"].with_suffix(".wav"),
        outs["txt"],
        outs["srt"],
        outs["json"],
        config,
        diar_turns=turns,
        output_jsonl=outs["jsonl"],
    )
    return model, meta


class TestHelpers:
    def test_batch_size_plan(self):
        assert _batch_size_plan(8) == [8, 4, 2, 1]
        assert _batch_size_plan(1) == [1]
        assert _batch_size_plan(0) == [1]

    def test_is_cuda_oom(self):
        assert _is_cuda_oom(RuntimeError("CUDA error: out of memory"))
        assert not _is_cuda_oom(RuntimeError("file not found"))

    def test_kwargs_forward_all_anti_hallucination_params(self):
        cfg = TranscriptionConfig(
            condition_on_previous_text=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            hallucination_silence_threshold=2.0,
        )
        kw = _build_transcribe_kwargs(cfg, "glosario", word_timestamps=True)
        assert kw["condition_on_previous_text"] is False
        assert kw["repetition_penalty"] == 1.2
        assert kw["no_repeat_ngram_size"] == 3
        assert kw["hallucination_silence_threshold"] == 2.0
        assert kw["initial_prompt"] == "glosario"
        assert kw["temperature"] == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


class TestStreaming:
    def test_word_mode_resegments_and_numbers_srt_consecutively(self, outs):
        cfg = TranscriptionConfig(batch_size=1, language="es")
        _, meta = _run(cfg, outs)
        # 3 emitted display segments: whole + 2 pieces of the crossing segment
        assert meta["total_segments"] == 3
        assert meta["empty_segments_discarded"] == 1
        speakers = [s["speaker"] for s in meta["segments"]]
        assert speakers == ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01"]
        ids = [int(line) for line in outs["srt"].read_text().splitlines() if line.strip().isdigit()]
        assert ids == [1, 2, 3], "SRT numbering must be consecutive after filtering"
        # txt labels every line
        assert all(line.startswith("[SPEAKER_") for line in outs["txt"].read_text().splitlines())

    def test_segment_mode_keeps_whole_segments(self, outs):
        cfg = TranscriptionConfig(batch_size=1, language="es", speaker_assignment="segment")
        _, meta = _run(cfg, outs)
        assert meta["total_segments"] == 2  # crossing segment NOT split
        assert meta["speaker_assignment"] == "segment"

    def test_word_timestamps_auto_enabled_for_word_mode_but_not_in_json(self, outs):
        cfg = TranscriptionConfig(batch_size=1, language="es", word_timestamps=False)
        model, meta = _run(cfg, outs)
        assert model.calls[0]["word_timestamps"] is True  # needed internally
        assert meta["word_timestamps_effective"] is True
        assert meta["word_timestamps"] is False
        assert all("words" not in s for s in meta["segments"])  # user did not ask

    def test_user_word_timestamps_included_in_json(self, outs):
        cfg = TranscriptionConfig(batch_size=1, language="es", word_timestamps=True)
        _, meta = _run(cfg, outs)
        assert any("words" in s for s in meta["segments"])

    def test_no_diarization_has_no_labels(self, outs):
        cfg = TranscriptionConfig(batch_size=1, language="es", enable_diarization=False)
        _, meta = _run(cfg, outs, turns=None)
        assert meta["diarization_enabled"] is False
        assert all("speaker" not in s for s in meta["segments"])
        assert not outs["txt"].read_text().startswith("[")

    def test_jsonl_mirrors_segments(self, outs):
        cfg = TranscriptionConfig(batch_size=1, language="es")
        _, meta = _run(cfg, outs)
        rows = [json.loads(line) for line in outs["jsonl"].read_text().splitlines()]
        assert [r["id"] for r in rows] == [s["id"] for s in meta["segments"]]

    def test_json_written_atomically_no_tmp_left(self, outs, tmp_path):
        cfg = TranscriptionConfig(batch_size=1, language="es")
        _run(cfg, outs)
        assert not list(tmp_path.glob("*.tmp"))
        assert json.loads(outs["json"].read_text())["package_version"]


class TestBatchingAndOom:
    def test_batched_path_used_when_batch_size_gt_1(self, outs):
        cfg = TranscriptionConfig(batch_size=8, language="es")
        model, meta = _run(cfg, outs)
        assert model.calls[0]["mode"] == "batched"
        assert model.calls[0]["batch_size"] == 8
        assert meta["batch_size_effective"] == 8

    def test_sequential_path_used_at_batch_size_1(self, outs):
        cfg = TranscriptionConfig(batch_size=1, language="es")
        model, meta = _run(cfg, outs)
        assert model.calls[0]["mode"] == "sequential"
        assert meta["batch_size_effective"] == 1

    def test_oom_ladder_halves_until_success(self, outs):
        FakeBatchedPipeline.oom_remaining = 2  # fail at 8 and at 4
        cfg = TranscriptionConfig(batch_size=8, language="es")
        model, meta = _run(cfg, outs)
        modes = [(c["mode"], c.get("batch_size")) for c in model.calls]
        assert modes[:3] == [("batched", 8), ("batched", 4), ("batched", 2)]
        assert meta["batch_size_requested"] == 8
        assert meta["batch_size_effective"] == 2

    def test_oom_all_the_way_down_reaches_sequential(self, outs):
        FakeBatchedPipeline.oom_remaining = 3  # 8, 4, 2 all fail -> sequential
        cfg = TranscriptionConfig(batch_size=8, language="es")
        model, meta = _run(cfg, outs)
        assert model.calls[-1]["mode"] == "sequential"
        assert meta["batch_size_effective"] == 1

    def test_non_oom_runtime_error_propagates_immediately(self, outs):
        class ExplodingModel(FakeModel):
            def transcribe(self, audio, **kw):
                raise RuntimeError("invalid audio header")

        cfg = TranscriptionConfig(batch_size=1, language="es")
        with pytest.raises(RuntimeError, match="invalid audio header"):
            _run(cfg, outs, model=ExplodingModel())

    def test_partial_output_from_failed_attempt_is_overwritten(self, outs):
        """OOM mid-iteration must not leak partial lines into the final txt."""

        class MidIterOomModel(FakeModel):
            def __init__(self):
                super().__init__()
                self.failed_once = False

            def transcribe(self, audio, **kw):
                self.calls.append({"mode": "sequential", **kw})

                def gen():
                    yield FakeSegment(0.0, 1.0, " basura parcial")
                    if not self.failed_once:
                        self.failed_once = True
                        raise RuntimeError("CUDA out of memory")
                    yield FakeSegment(1.0, 2.0, " final limpio")

                from tests.fakes import FakeInfo

                return gen(), FakeInfo()

        # batch_size=2 -> plan [2, 1]; fake ignores batching but raises once
        cfg = TranscriptionConfig(batch_size=2, language="es", enable_diarization=False)
        monkey_model = MidIterOomModel()
        # Route batched calls to the sequential fake transparently
        sys.modules["faster_whisper"].BatchedInferencePipeline = lambda model: type(
            "P",
            (),
            {
                "transcribe": staticmethod(
                    lambda audio, batch_size, **kw: model.transcribe(audio, **kw)
                )
            },
        )()
        meta = transcribe_streaming(
            monkey_model,
            outs["txt"].with_suffix(".wav"),
            outs["txt"],
            outs["srt"],
            outs["json"],
            cfg,
            diar_turns=None,
        )
        content = outs["txt"].read_text()
        assert content.count("basura parcial") == 1  # rewritten, not appended
        assert "final limpio" in content
        assert meta["total_segments"] == 2
