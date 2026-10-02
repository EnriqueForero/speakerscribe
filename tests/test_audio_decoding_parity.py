"""Native WAV reader vs faster-whisper's PyAV decoder on REAL ffmpeg output.

Marked `integration` (needs ffmpeg and faster-whisper). Guards two claims:

1. Parity: for the 16 kHz mono PCM16 WAV produced by `extract_audio_wav`,
   `read_wav_float32` returns exactly the samples `decode_audio` returns,
   so switching faster-whisper's input from a path to an array changes no
   transcription.
2. The native path does not depend on PyAV's API (PyAV 19 broke
   `decode_audio` for faster-whisper 1.2.1 on 2026-09-29).
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.integration

np = pytest.importorskip("numpy")
faster_whisper_audio = pytest.importorskip("faster_whisper.audio")

from speakerscribe.audio import extract_audio_wav, read_wav_float32  # noqa: E402


@pytest.fixture(scope="module")
def ffmpeg_wav(tmp_path_factory):
    if shutil.which("ffmpeg") is None:
        if os.environ.get("CI"):
            pytest.fail("ffmpeg must be installed in CI for integration tests")
        pytest.skip("ffmpeg not available")
    d = tmp_path_factory.mktemp("decode_parity")
    source = d / "source.m4a"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=330:duration=3:sample_rate=44100",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=550:duration=3:sample_rate=44100",
            "-filter_complex",
            "[0:a][1:a]amerge=inputs=2[a]",
            "-map",
            "[a]",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
        timeout=60,
    )
    return extract_audio_wav(source, d / "extracted.wav")


def test_native_reader_matches_decode_audio_exactly(ffmpeg_wav):
    try:
        reference = faster_whisper_audio.decode_audio(str(ffmpeg_wav), sampling_rate=16_000)
    except TypeError as e:  # PyAV >= 19 with faster-whisper 1.2.1
        pytest.skip(f"PyAV decoder unusable in this environment: {e}")
    native = read_wav_float32(ffmpeg_wav)
    assert native.dtype == reference.dtype == np.float32
    np.testing.assert_array_equal(native, reference)


def test_native_reader_works_regardless_of_pyav(ffmpeg_wav):
    native = read_wav_float32(ffmpeg_wav)
    # AAC priming/padding adds up to one encoder frame (1024 @ 44.1 kHz).
    assert native.size == pytest.approx(3 * 16_000, abs=1_000)
    assert float(np.max(np.abs(native))) > 0.01
