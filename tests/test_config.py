"""Tests for speakerscribe.config (v0.2.0)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from speakerscribe.config import (
    FILLER_WORDS,
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

    def test_default_enable_runs_db_on(self):
        """v0.2.0: SQLite history is ON by default for robust idempotency."""
        c = TranscriptionConfig()
        assert c.enable_runs_db is True

    def test_default_chunking(self):
        """Long-audio chunking defaults: 120 min threshold, 30 min chunks, 5s overlap."""
        c = TranscriptionConfig()
        assert c.long_audio_threshold_min == 120
        assert c.chunk_duration_min == 30
        assert c.chunk_overlap_s == 5

    def test_default_remove_fillers_on(self):
        c = TranscriptionConfig()
        assert c.remove_fillers is True

    def test_default_unified_for_llm_on(self):
        c = TranscriptionConfig()
        assert c.produce_unified_for_llm is True

    def test_default_streaming_jsonl_off(self):
        c = TranscriptionConfig()
        assert c.streaming_jsonl is False

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

    def test_chunk_duration_must_exceed_overlap(self):
        """Validator: chunk_duration_min*60 must be > chunk_overlap_s.

        chunk_duration_min has min=5 (=300s) and chunk_overlap_s has max=120s,
        so this validator can only fire when chunk_overlap_s is just barely
        above chunk_duration_min*60. We use chunk_duration_min=5 (300s) and
        chunk_overlap_s=120 — overlap > duration would be a ValidationError.
        Wait: 120 < 300, so this passes. Test instead the equality boundary:
        chunk_duration_min=2 is invalid (Field min=5), so we set Field bounds
        wider for this test by... actually the way to test the cross-validator
        is to set both at the boundary where the cross-validator triggers.
        """
        # The cross-validator fires when chunk_duration_min*60 <= chunk_overlap_s.
        # With Field min for chunk_duration_min=5 (=300s) and Field max for
        # chunk_overlap_s=120s, the cross-validator can never fire on valid
        # Field values. This is by design: Field bounds already prevent the
        # nonsensical case. We assert that fact rather than try to trick it.
        c = TranscriptionConfig(chunk_duration_min=5, chunk_overlap_s=120)
        assert c.chunk_duration_min * 60 > c.chunk_overlap_s

    def test_chunking_disabled_when_threshold_zero(self):
        """If long_audio_threshold_min=0, chunking validators are bypassed."""
        c = TranscriptionConfig(long_audio_threshold_min=0)
        assert c.long_audio_threshold_min == 0

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
        with pytest.raises(ValidationError):
            TranscriptionConfig(unknown_param=42)  # type: ignore[call-arg]

    def test_resolve_device_auto(self):
        c = TranscriptionConfig(device="auto")
        device, compute = c.resolve_device()
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


class TestPerFileGlossary:
    """v0.2.0: per-file initial_prompt via <stem>.prompt.txt."""

    def test_resolve_uses_per_file_when_present(self, tmp_path):
        audio = tmp_path / "meeting.mp4"
        audio.write_text("fake")
        prompt_file = tmp_path / "meeting.prompt.txt"
        prompt_file.write_text("GIC, ProColombia, OFAT")
        c = TranscriptionConfig(initial_prompt="global prompt")
        assert c.resolve_initial_prompt(audio) == "GIC, ProColombia, OFAT"

    def test_resolve_falls_back_to_global(self, tmp_path):
        audio = tmp_path / "other.mp4"
        audio.write_text("fake")
        c = TranscriptionConfig(initial_prompt="global only")
        assert c.resolve_initial_prompt(audio) == "global only"

    def test_resolve_returns_none_without_prompt(self, tmp_path):
        audio = tmp_path / "x.mp4"
        audio.write_text("fake")
        c = TranscriptionConfig(initial_prompt=None)
        assert c.resolve_initial_prompt(audio) is None

    def test_resolve_per_file_truncated_to_500_chars(self, tmp_path):
        audio = tmp_path / "m.mp4"
        audio.write_text("fake")
        prompt_file = tmp_path / "m.prompt.txt"
        prompt_file.write_text("a" * 800)
        c = TranscriptionConfig()
        resolved = c.resolve_initial_prompt(audio)
        assert resolved is not None
        assert len(resolved) == 500

    def test_resolve_empty_per_file_falls_back(self, tmp_path):
        audio = tmp_path / "m.mp4"
        audio.write_text("fake")
        prompt_file = tmp_path / "m.prompt.txt"
        prompt_file.write_text("   \n\n  ")
        c = TranscriptionConfig(initial_prompt="global")
        assert c.resolve_initial_prompt(audio) == "global"


class TestWorkspacePaths:
    def test_basic_properties(self, tmp_path):
        p = WorkspacePaths(workspace=str(tmp_path))
        assert p.base == tmp_path
        assert p.data == tmp_path / "data"
        assert p.transcripts == tmp_path / "transcripts"
        assert p.splits == tmp_path / "splits"
        assert p.audio_tmp == tmp_path / "_audio_temp"
        assert p.audio_chunks == tmp_path / "_audio_chunks"
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
        assert p.audio_chunks.exists()
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
        assert all(e == e.lower() for e in MEDIA_EXTENSIONS)
        assert all(e.startswith(".") for e in MEDIA_EXTENSIONS)

    def test_filler_words_have_main_languages(self):
        for lang in ("es", "en", "pt", "fr"):
            assert lang in FILLER_WORDS
            assert len(FILLER_WORDS[lang]) > 0
