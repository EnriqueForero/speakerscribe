"""Batch runner with the REAL ffprobe and staging (engine still scripted)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from speakerscribe.batch import BatchRunner
from speakerscribe.batch.discovery import probe_media
from speakerscribe.batch.runner import RunnerDeps
from tests.batch_fakes import FakeEngine, make_settings

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None or shutil.which("ffprobe") is None:
        pytest.fail("ffmpeg/ffprobe son obligatorios para las pruebas de integración")
    return exe


def make_tone(ffmpeg: str, target: Path, seconds: int = 3) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}",
            "-c:a",
            "aac",
            str(target),
        ],  # fmt: skip
        check=True,
    )
    return target


def test_probe_classifies_real_files(ffmpeg: str, tmp_path: Path):
    good = make_tone(ffmpeg, tmp_path / "tono.m4a")
    result = probe_media(good)
    assert result.status == "ok" and result.duration_s == pytest.approx(3.0, abs=0.2)
    broken = tmp_path / "roto.mp4"
    broken.write_bytes(b"\x00" * 2048)
    assert probe_media(broken).status == "invalid"


def test_runner_with_real_probe_and_staging(ffmpeg: str, tmp_path: Path):
    root = tmp_path / "proj"
    audio = make_tone(ffmpeg, root / "data" / "2026-10-02 *Prueba corta.m4a")
    payload = audio.read_bytes()
    engine = FakeEngine()
    deps = RunnerDeps(
        engine_factory=lambda config: engine,
        gpu_preflight=lambda config, paths, largest: {
            "gpu": "cpu-test",
            "vram_gb": 0,
            "warnings": [],
        },
        in_colab=lambda: False,
        emit=lambda _: None,
    )
    report = BatchRunner(make_settings(root), deps).run()
    assert report.end_reason == "completo" and report.ok == 1
    assert engine.calls[0]["payload"] == payload, "the staged copy is byte-identical"
    assert (root / "entregables" / "2026-10-02 *Prueba corta.txt").is_file()
    assert not any((tmp_path / "proj_scratch" / "staging").iterdir())
