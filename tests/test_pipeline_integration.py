"""End-to-end integration: REAL audio -> REAL faster-whisper (tiny, CPU).

Marked `integration`: excluded from the default unit run. CI executes this
in a scheduled job that installs the full stack and espeak-ng. It exists
because the unit suite, by design, never exercises the real decoder — this
is the test that would have caught a broken faster-whisper pin or a kwargs
mismatch against the real `transcribe()` signature.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.integration

faster_whisper = pytest.importorskip("faster_whisper")

from speakerscribe.config import TranscriptionConfig, WorkspacePaths  # noqa: E402
from speakerscribe.pipeline import process_one  # noqa: E402
from speakerscribe.transcription import loaded_whisper  # noqa: E402

SPOKEN_TEXT = (
    "La reunión de hoy trata sobre el presupuesto anual. "
    "Necesitamos aprobar las partidas antes del viernes. "
    "Gracias a todos por participar en esta sesión."
)


@pytest.fixture(scope="module")
def real_workspace(tmp_path_factory):
    """Workspace with ~10 s of synthesized Spanish speech."""
    if shutil.which("espeak-ng") is None or shutil.which("ffmpeg") is None:
        pytest.skip("espeak-ng/ffmpeg not available")
    ws = tmp_path_factory.mktemp("integration_ws")
    paths = WorkspacePaths(workspace=ws, scratch=str(ws / "scratch"))
    paths.create_directories()
    wav = paths.data / "reunion.wav"
    subprocess.run(
        ["espeak-ng", "-v", "es", "-s", "140", "-w", str(wav), SPOKEN_TEXT],
        check=True,
        capture_output=True,
        timeout=60,
    )
    assert wav.stat().st_size > 10_000
    return paths, wav


def test_tiny_model_end_to_end_without_diarization(real_workspace):
    paths, wav = real_workspace
    config = TranscriptionConfig(
        model="tiny",
        language="es",
        device="cpu",
        compute_type="int8",
        batch_size=1,
        beam_size=1,
        enable_diarization=False,
        extract_temp_wav=True,
        evaluate_quality=True,
        auto_retry_on_critical=False,
    )
    with loaded_whisper(config) as model:
        meta = process_one(wav, paths, model, config)

    assert meta["status"] == "ok"
    base = meta["base_name"]

    txt = (paths.transcripts / f"{base}.txt").read_text(encoding="utf-8")
    assert len(txt.split()) >= 5, "real decode should yield words"
    assert "presupuesto" in txt.lower() or "reunión" in txt.lower() or "viernes" in txt.lower()

    srt = (paths.transcripts / f"{base}.srt").read_text(encoding="utf-8")
    ids = [int(line) for line in srt.splitlines() if line.strip().isdigit()]
    assert ids == list(range(1, len(ids) + 1)), "SRT numbering must be consecutive"

    data = json.loads((paths.transcripts / f"{base}.json").read_text(encoding="utf-8"))
    for key in (
        "package_version",
        "faster_whisper_version",
        "batch_size_effective",
        "empty_segments_discarded",
        "segments",
        "timings",
    ):
        assert key in data, key
    assert data["language_detected"] == "es"

    assert paths.ledger_path.exists()
    again = process_one(wav, paths, object(), config)  # model unused on skip
    assert again["status"] == "skipped", "idempotency must hold end-to-end"


def test_batched_path_against_real_decoder(real_workspace):
    """batch_size>1 exercises BatchedInferencePipeline kwargs for real."""
    paths, wav = real_workspace
    config = TranscriptionConfig(
        model="tiny",
        language="es",
        device="cpu",
        compute_type="int8",
        batch_size=4,
        beam_size=1,
        enable_diarization=False,
        force_reprocess=True,
        evaluate_quality=False,
    )
    with loaded_whisper(config) as model:
        meta = process_one(wav, paths, model, config)
    assert meta["status"] == "ok"
    assert meta["batch_size_effective"] == 4
    assert meta["total_words"] >= 5
