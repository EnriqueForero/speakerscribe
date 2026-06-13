"""Planning estimates (the table that used to live duplicated in the notebook)."""

from __future__ import annotations

from speakerscribe.estimates import RTF_ESTIMATE_T4, estimate_processing_minutes


class TestEstimates:
    def test_known_model_no_diar(self):
        # 60 min at RTF 10 -> 6.0 min
        assert estimate_processing_minutes(60, "large-v3-turbo", with_diarization=False) == 6.0

    def test_diarization_adds_time(self):
        with_diar = estimate_processing_minutes(60, "large-v3-turbo", with_diarization=True)
        without = estimate_processing_minutes(60, "large-v3-turbo", with_diarization=False)
        assert with_diar > without

    def test_unknown_model_uses_conservative_default(self):
        est = estimate_processing_minutes(60, "no-such-model", with_diarization=False)
        assert est == 60 / 4.0  # slowest-large assumption

    def test_zero_and_negative_audio(self):
        assert estimate_processing_minutes(0, "small") == 0.0
        assert estimate_processing_minutes(-5, "small") == 0.0

    def test_batch_speedup_scales(self):
        seq = estimate_processing_minutes(60, "large-v3", with_diarization=False)
        fast = estimate_processing_minutes(
            60, "large-v3", with_diarization=False, batch_speedup=3.0
        )
        assert fast < seq

    def test_table_covers_all_supported_models(self):
        for model in ("tiny", "small", "medium", "large-v3", "large-v3-turbo"):
            assert model in RTF_ESTIMATE_T4
