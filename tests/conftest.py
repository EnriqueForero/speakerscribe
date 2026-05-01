"""Shared fixtures for the test suite."""

from __future__ import annotations

from pathlib import Path

import pytest
from speakerscribe.config import TranscriptionConfig, WorkspacePaths


@pytest.fixture
def workspace_tmp(tmp_path: Path) -> Path:
    """A temporary workspace path."""
    return tmp_path


@pytest.fixture
def paths(workspace_tmp: Path) -> WorkspacePaths:
    """A WorkspacePaths instance with a temporary workspace and directories created."""
    p = WorkspacePaths(workspace=str(workspace_tmp))
    p.create_directories()
    return p


@pytest.fixture
def basic_config() -> TranscriptionConfig:
    """A valid TranscriptionConfig for tests (no diarization, no GPU/HF token needed)."""
    return TranscriptionConfig(
        model="large-v3",
        language="en",
        beam_size=5,
        enable_diarization=False,
    )


@pytest.fixture
def sample_metadata() -> dict:
    """Example metadata, shaped like what `transcribe_streaming` returns."""
    return {
        "audio_file": "test.wav",
        "processed_at": "2026-05-01T12:00:00+00:00",
        "package_version": "0.1.0",
        "pyannote_version": "4.0.4",
        "faster_whisper_version": "1.2.1",
        "model": "large-v3",
        "language_detected": "en",
        "language_probability": 0.99,
        "duration_seconds": 600.0,
        "duration_minutes": 10.0,
        "elapsed_seconds": 60.0,
        "real_time_factor": 10.0,
        "total_segments": 100,
        "total_words": 1500,
        "diarization_enabled": True,
        "diarization_model": "pyannote/speaker-diarization-community-1",
        "speakers_summary": {"SPEAKER_00": 60, "SPEAKER_01": 40},
        "word_timestamps": False,
        "config": {"model": "large-v3"},
        "segments": [
            {"id": 1, "start": 0.0, "end": 5.0, "text": "Hello.", "speaker": "SPEAKER_00"},
            {"id": 2, "start": 5.0, "end": 10.0, "text": "How are you.", "speaker": "SPEAKER_00"},
            {"id": 3, "start": 10.0, "end": 15.0, "text": "Fine.", "speaker": "SPEAKER_01"},
        ],
    }
