"""Tests for speakerscribe.config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from speakerscribe.config import (
    MEDIA_EXTENSIONS,
    SPK_NO_DIARIZATION,
    SPK_NO_OVERLAP,
    TranscriptionConfig,
    WorkspacePaths,
)


class TestTranscriptionConfig:
    def test_defaults_valid(self):
        c = TranscriptionConfig()
        assert c.model == "large-v3"
        assert c.beam_size == 5
        assert c.enable_diarization is True
        assert c.force_reprocess is False
        assert c.enable_runs_db is False  # opt-in by default

    def test_invalid_model_fails(self):
        with pytest.raises(ValidationError):
            TranscriptionConfig(model="fake-model")  # type: ignore[arg-type]

    def test_beam_size_out_of_range(self):
        with pytest.raises(ValidationError):
            TranscriptionConfig(beam_size=99)
        with pytest.raises(ValidationError):
            TranscriptionConfig(beam_size=0)

    def test_num_speakers_excludes_min_max(self):
        with pytest.raises(ValidationError, match="exclusive"):
            TranscriptionConfig(num_speakers=3, min_speakers=2)
        with pytest.raises(ValidationError, match="exclusive"):
            TranscriptionConfig(num_speakers=3, max_speakers=5)

    def test_min_greater_than_max(self):
        with pytest.raises(ValidationError, match="min_speakers"):
            TranscriptionConfig(min_speakers=5, max_speakers=3)

    def test_min_max_valid(self):
        c = TranscriptionConfig(min_speakers=2, max_speakers=5)
        assert c.min_speakers == 2
        assert c.max_speakers == 5

    def test_num_speakers_alone(self):
        c = TranscriptionConfig(num_speakers=4)
        assert c.num_speakers == 4
        assert c.min_speakers is None
        assert c.max_speakers is None

    def test_initial_prompt_max_length(self):
        with pytest.raises(ValidationError):
            TranscriptionConfig(initial_prompt="x" * 1000)

    def test_language_normalized(self):
        c = TranscriptionConfig(language="EN")
        assert c.language == "en"
        c = TranscriptionConfig(language="  ES  ")
        assert c.language == "es"

    def test_extra_attribute_rejected(self):
        """Pydantic with extra='forbid' rejects undeclared attributes."""
        with pytest.raises(ValidationError):
            TranscriptionConfig(unknown_param=42)  # type: ignore[call-arg]

    def test_resolve_device_auto(self):
        c = TranscriptionConfig(device="auto")
        device, compute = c.resolve_device()
        # In CI without GPU: cpu/int8; on a GPU host: cuda/float16
        assert device in ("cpu", "cuda")
        assert compute in ("int8", "float16")

    def test_resolve_device_explicit_cpu(self):
        c = TranscriptionConfig(device="cpu", compute_type="int8")
        assert c.resolve_device() == ("cpu", "int8")

    def test_resolve_hf_token_explicit(self):
        c = TranscriptionConfig(hf_token="hf_explicit_token")
        assert c.resolve_hf_token() == "hf_explicit_token"

    def test_resolve_hf_token_env_var(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_from_env")
        c = TranscriptionConfig(hf_token=None)
        assert c.resolve_hf_token() == "hf_from_env"

    def test_resolve_hf_token_none(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        c = TranscriptionConfig(hf_token=None)
        assert c.resolve_hf_token() is None

    def test_serializable_to_json(self):
        c = TranscriptionConfig(model="large-v3-turbo", language="es")
        data = c.model_dump_json()
        c2 = TranscriptionConfig.model_validate_json(data)
        assert c2.model == "large-v3-turbo"
        assert c2.language == "es"

    def test_enable_runs_db_default_off(self):
        c = TranscriptionConfig()
        assert c.enable_runs_db is False

    def test_enable_runs_db_can_be_enabled(self):
        c = TranscriptionConfig(enable_runs_db=True)
        assert c.enable_runs_db is True


class TestWorkspacePaths:
    def test_basic_properties(self, tmp_path):
        p = WorkspacePaths(workspace=str(tmp_path))
        assert p.base == tmp_path
        assert p.data == tmp_path / "data"
        assert p.transcripts == tmp_path / "transcripts"
        assert p.splits == tmp_path / "splits"
        assert p.audio_tmp == tmp_path / "_audio_temp"
        assert p.diar_cache == tmp_path / "_diar_cache"
        assert p.logs == tmp_path / "_logs"
        assert p.db_path == tmp_path / "_runs.db"

    def test_create_directories(self, tmp_path):
        p = WorkspacePaths(workspace=str(tmp_path))
        p.create_directories()
        assert p.data.exists()
        assert p.transcripts.exists()
        assert p.splits.exists()
        assert p.audio_tmp.exists()
        assert p.diar_cache.exists()
        assert p.logs.exists()

    def test_list_media_files_no_data_folder(self, tmp_path):
        p = WorkspacePaths(workspace=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            p.list_media_files()

    def test_list_media_files_filters_extensions(self, tmp_path):
        p = WorkspacePaths(workspace=str(tmp_path))
        p.create_directories()
        (p.data / "audio1.mp4").touch()
        (p.data / "audio2.mp3").touch()
        (p.data / "document.txt").touch()
        (p.data / "image.png").touch()
        media = p.list_media_files()
        assert len(media) == 2
        assert all(v.suffix.lower() in MEDIA_EXTENSIONS for v in media)


class TestConstants:
    def test_speaker_labels(self):
        assert SPK_NO_DIARIZATION == "(no diarization)"
        assert SPK_NO_OVERLAP == "SPEAKER_NO_OVERLAP"

    def test_media_extensions(self):
        assert ".mp4" in MEDIA_EXTENSIONS
        assert ".mp3" in MEDIA_EXTENSIONS
        assert ".wav" in MEDIA_EXTENSIONS
        # All entries lowercase, leading-dot
        assert all(e == e.lower() for e in MEDIA_EXTENSIONS)
        assert all(e.startswith(".") for e in MEDIA_EXTENSIONS)
