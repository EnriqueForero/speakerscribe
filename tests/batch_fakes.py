"""Test doubles for `speakerscribe.batch`: a scripted engine and runner deps.

The fake engine keys its behavior on the CONTENT of the staged media (the
batch renames files while staging), so tests write each audio with a
distinct byte payload and script the outcome per payload.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from speakerscribe.batch import BatchSettings, RunnerDeps
from speakerscribe.batch.discovery import ProbeResult

GOOD_SEGMENTS = [
    {"start": 0.0, "end": 4.0, "text": "Buenos días a todos.", "speaker": "SPEAKER_00"},
    {"start": 4.5, "end": 9.0, "text": "Gracias por venir hoy.", "speaker": "SPEAKER_01"},
    {"start": 9.2, "end": 12.0, "text": "Empecemos con la agenda.", "speaker": "SPEAKER_00"},
]


def metadata_for(segments: list[dict[str, Any]] | None = None, **overrides: Any) -> dict[str, Any]:
    segs = GOOD_SEGMENTS if segments is None else segments
    md: dict[str, Any] = {
        "status": "ok",
        "segments": [dict(s) for s in segs],
        "diarization_enabled": True,
        "diarization_model": "pyannote/speaker-diarization-community-1",
        "language_detected": "es",
        "language_probability": 0.99,
        "duration_seconds": 12.0,
        "total_words": sum(len(str(s["text"]).split()) for s in segs),
        "speakers_summary": {s["speaker"]: 1 for s in segs if s.get("speaker")},
        "real_time_factor": 20.0,
        "quality_flags": [],
        "model": "large-v3",
    }
    md.update(overrides)
    return md


Behavior = Callable[[Path, Path, Path | None, Path], dict[str, Any]]


class FakeEngine:
    """`TranscriptionEngine` double with per-content scripted outcomes.

    Args:
        script: payload bytes -> metadata dict, an Exception instance to
            raise, or a callable(media, workdir, cache_in, cache_out).
        default: Outcome for unscripted payloads.
        load_error: Exception raised by `load`.
    """

    def __init__(
        self,
        script: dict[bytes, Any] | None = None,
        default: Any = None,
        load_error: Exception | None = None,
    ) -> None:
        self.script = script or {}
        self.default = default if default is not None else metadata_for()
        self.load_error = load_error
        self._loaded = False
        self.calls: list[dict[str, Any]] = []
        self.loads = 0
        self.unloads = 0

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self.load_error is not None:
            raise self.load_error
        self.loads += 1
        self._loaded = True

    def unload(self) -> None:
        self.unloads += 1
        self._loaded = False

    def transcribe(
        self, media: Path, workdir: Path, diar_cache_in: Path | None, diar_cache_out: Path
    ) -> dict[str, Any]:
        payload = media.read_bytes()
        prompt = media.with_suffix(".prompt.txt")
        self.calls.append(
            {
                "payload": payload,
                "media": media,
                "cache_in": diar_cache_in,
                "prompt": prompt.read_text(encoding="utf-8") if prompt.exists() else None,
            }
        )
        if diar_cache_in is None:  # "diarization" happens first, as in the real engine
            diar_cache_out.write_text(json.dumps({"turns": [], "payload": len(payload)}))
        outcome = self.script.get(payload, self.default)
        if isinstance(outcome, BaseException):
            raise outcome
        if callable(outcome):
            return outcome(media, workdir, diar_cache_in, diar_cache_out)
        return json.loads(json.dumps(outcome))  # deep copy


def fake_probe(durations: dict[bytes, float] | None = None, invalid: set[bytes] | None = None):
    durations = durations or {}
    invalid = invalid or set()

    def probe(path: Path) -> ProbeResult:
        payload = path.read_bytes()
        if payload in invalid:
            return ProbeResult("invalid", None, "moov atom not found")
        return ProbeResult("ok", durations.get(payload, 60.0), None)

    return probe


def make_settings(root: Path, **overrides: Any) -> BatchSettings:
    values: dict[str, Any] = {
        "root": root,
        "scratch_dir": root.parent / f"{root.name}_scratch",
        "stability_seconds": 0,
        "resource_monitor": False,
        "shutdown_at_end": False,
        "max_session_minutes": 0,
        "audits_per_session": 0,
    }
    values.update(overrides)
    return BatchSettings(**values)


def make_deps(
    engine: FakeEngine,
    *,
    probe: Callable[[Path], ProbeResult] | None = None,
    ram: Callable[[], float] = lambda: 10.0,
    monotonic: Callable[[], float] | None = None,
    emitted: list[str] | None = None,
    gpu_env: dict[str, Any] | None = None,
) -> RunnerDeps:
    sink = emitted if emitted is not None else []
    deps = RunnerDeps(
        engine_factory=lambda config: engine,
        gpu_preflight=lambda config, paths, largest: dict(
            gpu_env or {"gpu": "FakeGPU", "vram_gb": 16.0, "warnings": []}
        ),
        storage_preflight=lambda paths, in_colab: paths.ensure(),
        probe=probe or fake_probe(),
        ram=ram,
        in_colab=lambda: False,
        emit=sink.append,
        producer=lambda: "speakerscribe vTEST",
    )
    if monotonic is not None:
        deps.monotonic = monotonic
    return deps


def write_audio(root: Path, relative: str, payload: bytes) -> Path:
    path = root / "data" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def events(root: Path) -> list[dict[str, Any]]:
    path = root / "entregables" / ".speakerscribe_state" / "events.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def event_names(root: Path) -> list[str]:
    return [e["event"] for e in events(root)]
