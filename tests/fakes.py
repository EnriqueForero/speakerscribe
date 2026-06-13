"""Shared fakes for unit tests — a CPU-only stand-in for faster_whisper.

The unit suite must run WITHOUT torch/faster-whisper/pyannote installed
(CI installs only light deps). Tests inject `make_fake_faster_whisper(...)`
into `sys.modules["faster_whisper"]` so the lazy imports inside
`speakerscribe.transcription` resolve to these fakes.
"""

from __future__ import annotations

import types
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeWord:
    word: str
    start: float
    end: float
    probability: float = 0.9


@dataclass
class FakeSegment:
    start: float
    end: float
    text: str
    words: list[FakeWord] = field(default_factory=list)


class FakeInfo:
    language = "es"
    language_probability = 0.97
    duration = 10.0


class FakeModel:
    """Sequential-path fake. Records every transcribe call and its kwargs."""

    def __init__(self, segments_per_call: list[list[FakeSegment]] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._segments_per_call = segments_per_call or [_default_segments()]
        self._call_index = 0

    def transcribe(self, audio: str, **kwargs: Any) -> tuple[Any, FakeInfo]:
        self.calls.append({"mode": "sequential", "audio": audio, **kwargs})
        segments = self._segments_per_call[min(self._call_index, len(self._segments_per_call) - 1)]
        self._call_index += 1
        return iter(segments), FakeInfo()


class FakeBatchedPipeline:
    """Batched-path fake; optionally raises CUDA OOM for the first N calls."""

    oom_remaining: int = 0  # class-level knob set by tests

    def __init__(self, model: FakeModel) -> None:
        self.model = model

    def transcribe(self, audio: str, batch_size: int, **kwargs: Any) -> tuple[Any, FakeInfo]:
        self.model.calls.append(
            {"mode": "batched", "batch_size": batch_size, "audio": audio, **kwargs}
        )
        if FakeBatchedPipeline.oom_remaining > 0:
            FakeBatchedPipeline.oom_remaining -= 1
            raise RuntimeError("CUDA failed to allocate: out of memory")
        segments = self.model._segments_per_call[
            min(self.model._call_index, len(self.model._segments_per_call) - 1)
        ]
        self.model._call_index += 1
        return iter(segments), FakeInfo()


def _default_segments() -> list[FakeSegment]:
    """3 Whisper segments: normal, empty (discardable), turn-crossing."""
    return [
        FakeSegment(
            0.0,
            2.0,
            " Hola a todos.",
            [FakeWord(" Hola", 0.0, 0.5), FakeWord(" a", 0.5, 0.8), FakeWord(" todos.", 0.8, 2.0)],
        ),
        FakeSegment(2.0, 3.0, "   "),
        FakeSegment(
            3.5,
            6.5,
            " claro que sí. ¿Y usted qué opina?",
            [
                FakeWord(" claro", 3.8, 4.2),
                FakeWord(" que", 4.2, 4.5),
                FakeWord(" sí.", 4.5, 4.9),
                FakeWord(" ¿Y", 5.1, 5.4),
                FakeWord(" usted", 5.4, 5.8),
                FakeWord(" qué", 5.8, 6.0),
                FakeWord(" opina?", 6.0, 6.5),
            ],
        ),
    ]


TWO_SPEAKER_TURNS = [
    {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
    {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_01"},
]


def make_fake_faster_whisper() -> types.ModuleType:
    """Build an injectable fake `faster_whisper` module (OOM knob reset)."""
    FakeBatchedPipeline.oom_remaining = 0
    mod = types.ModuleType("faster_whisper")
    mod.WhisperModel = FakeModel  # type: ignore[attr-defined]
    mod.BatchedInferencePipeline = FakeBatchedPipeline  # type: ignore[attr-defined]
    mod.__version__ = "0.0-fake"
    return mod
