"""Pipeline orchestration: process_one, process_batch, preflight_check.

Orchestrates the full pipeline:
    1. Pre-flight environment validation
    2. WAV extraction (ffmpeg) to LOCAL scratch, named by content signature
    3. Diarization on the FULL audio (engine resident per batch; cached)
    4. Streaming transcription (batched; word-level speaker attribution)
    5. Quality check + ONE automatic anti-hallucination retry when critical
    6. Markdown transcript generation (with filler filter)
    7. Word-aware splits + unified-for-LLM file
    8. Runs ledger (JSONL append-only; idempotency by content signature)

Idempotency and degradation contracts (see `process_one` docstring):
    - A run whose diarization FAILED finishes as `status="ok_degraded"`
      with a CRITICAL `DIARIZATION_FAILED` quality flag, is ledgered with
      `diar_model=None`, and never blocks a future run that diarizes.
    - Skip lookups use the fast content signature, with a one-time
      transparent fallback to the full SHA-256 for histories recorded
      before 0.3 (a migration record is appended so the fallback runs once).
"""

from __future__ import annotations

import json
import os
import shutil
import time
import traceback
import warnings
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from speakerscribe.audio import (
    AudioChunk,
    extract_audio_wav,
    file_signature,
    get_audio_duration_seconds,
    split_long_audio,
)
from speakerscribe.config import (
    MIN_VRAM_BY_MODEL,
    TranscriptionConfig,
    WorkspacePaths,
)
from speakerscribe.diarization import DiarizationEngine, diarization_params_hash
from speakerscribe.logging_config import logger
from speakerscribe.output import (
    generate_transcript_md,
    split_text_by_words,
    write_unified_for_llm,
)
from speakerscribe.quality import QualityReport, Severity, evaluate_transcription_quality
from speakerscribe.transcription import (
    loaded_whisper,
    transcribe_chunked,
    transcribe_streaming,
)

if TYPE_CHECKING:
    from faster_whisper import WhisperModel


@dataclass(frozen=True)
class RunOutputs:
    """Output paths for one run. txt/srt/json are always written; jsonl is
    opt-in (`streaming_jsonl`). Typed so `json.exists()` needs no None
    dance anywhere downstream."""

    txt: Path
    srt: Path
    json: Path
    jsonl: Path | None

    def with_retry_suffix(self) -> RunOutputs:
        """Sibling `.retry` paths for the second attempt."""
        return replace(
            self,
            txt=self.txt.with_name(self.txt.name + ".retry"),
            srt=self.srt.with_name(self.srt.name + ".retry"),
            json=self.json.with_name(self.json.name + ".retry"),
            jsonl=(self.jsonl.with_name(self.jsonl.name + ".retry") if self.jsonl else None),
        )

    def pairs(self) -> list[tuple[Path | None, Path | None]]:
        """(final, retry-counterpart) iteration helper for promotion."""
        retry = self.with_retry_suffix()
        return [
            (self.txt, retry.txt),
            (self.srt, retry.srt),
            (self.json, retry.json),
            (self.jsonl, retry.jsonl),
        ]


# Quality codes that trigger the automatic anti-hallucination retry.
AUTO_RETRY_CODES: frozenset[str] = frozenset({"REPETITIONS", "HIGH_WPM"})

# Decoding overrides applied on the retry attempt: stop loop propagation
# across windows and penalize exact repetition.
AUTO_RETRY_OVERRIDES: dict[str, Any] = {
    "condition_on_previous_text": False,
    "repetition_penalty": 1.15,
    "no_repeat_ngram_size": 3,
}


def preflight_check(paths: WorkspacePaths, config: TranscriptionConfig) -> dict[str, Any]:
    """Validate the entire environment before loading any heavy model.

    Checks performed:
        1. The data/ folder exists and contains processable files.
        2. There is enough free disk space (workspace AND scratch).
        3. GPU is available if device='cuda' was requested.
        4. There is enough free VRAM for the chosen Whisper model.
        5. A HuggingFace token is reachable when diarization is enabled.

    Args:
        paths: WorkspacePaths with a valid workspace.
        config: Pipeline configuration.

    Returns:
        Dict with verified metrics.

    Raises:
        RuntimeError: If a blocking issue is found.
        FileNotFoundError: If the data/ folder does not exist.
    """
    logger.info("Pre-flight check...")
    paths.create_directories()

    # 1. Input files
    media = paths.list_media_files()
    if not media:
        raise RuntimeError(
            f"No media files found in {paths.data}.\n"
            f"   Place at least one .mp4/.mp3/.wav/.m4a/.mkv file there."
        )
    total_mb = sum(v.stat().st_size for v in media) / 1e6
    logger.info(f"   {len(media)} file(s) — {total_mb:.1f} MB total")

    # 2. Disk space (durable workspace AND local scratch — WAVs live there)
    free_mb = shutil.disk_usage(paths.base).free / 1e6
    required_mb = max(config.disk_margin_min_mb, total_mb * config.disk_margin_factor)
    if free_mb < required_mb:
        raise RuntimeError(
            f"Insufficient disk space: {free_mb:.0f} MB free, ~{required_mb:.0f} MB required."
        )
    logger.info(f"   Disk (workspace): {free_mb:,.0f} MB free (~{required_mb:.0f} MB required)")
    scratch_free_mb = shutil.disk_usage(paths.scratch_base).free / 1e6
    if scratch_free_mb < required_mb:
        raise RuntimeError(
            f"Insufficient scratch space at {paths.scratch_base}: "
            f"{scratch_free_mb:.0f} MB free, ~{required_mb:.0f} MB required."
        )
    logger.info(f"   Disk (scratch {paths.scratch_base}): {scratch_free_mb:,.0f} MB free")

    # 3-4. GPU/VRAM
    resolved_device, resolved_compute = config.resolve_device()
    vram_avail_gb = 0.0
    gpu_ok = False

    try:
        import torch

        gpu_ok = resolved_device == "cuda" and torch.cuda.is_available()
        if config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "device='cuda' requested but no GPU is available. "
                "On Colab: Runtime -> Change runtime type -> T4 GPU."
            )
        if gpu_ok:
            vram_total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            vram_alloc_gb = torch.cuda.memory_allocated() / 1e9
            vram_avail_gb = vram_total_gb - vram_alloc_gb
            min_vram = MIN_VRAM_BY_MODEL.get(config.model, 4.0)
            if vram_avail_gb < min_vram:
                warnings.warn(
                    f"Available VRAM ({vram_avail_gb:.1f} GB) may be insufficient "
                    f"for '{config.model}' (~{min_vram} GB) plus diarization.",
                    stacklevel=2,
                )
            logger.info(f"   VRAM: {vram_avail_gb:.1f} GB free / {vram_total_gb:.1f} GB total")
    except ImportError:
        logger.warning("torch not available — skipping GPU checks")

    logger.info(f"   Device: {resolved_device} ({resolved_compute})")

    # 5. HuggingFace token
    hf_token_ok = False
    if config.enable_diarization:
        token = config.resolve_hf_token()
        if not token:
            raise RuntimeError(
                "enable_diarization=True but no HF_TOKEN found.\n"
                "   1. Generate a 'Read' token (NOT fine-grained) at\n"
                "      https://huggingface.co/settings/tokens\n"
                "   2. Provide it via Colab Secrets, the HF_TOKEN env var,\n"
                "      or by passing hf_token=... to TranscriptionConfig.\n"
                "   3. Accept the model terms at:\n"
                f"      https://huggingface.co/{config.diarization_model}"
            )
        hf_token_ok = True
        logger.info("   HF token: detected")  # never log token material
    else:
        logger.info("   Diarization disabled — transcription only")

    return {
        "n_files": len(media),
        "total_mb": round(total_mb, 1),
        "free_mb": round(free_mb, 0),
        "scratch_free_mb": round(scratch_free_mb, 0),
        "gpu_ok": gpu_ok,
        "vram_available_gb": round(vram_avail_gb, 2),
        "hf_token_ok": hf_token_ok,
        "device": resolved_device,
        "compute_type": resolved_compute,
    }


def _maybe_extract_wav(
    media_path: Path,
    paths: WorkspacePaths,
    config: TranscriptionConfig,
    timings: dict[str, float],
    signature: str,
) -> tuple[Path, Path | None]:
    """Extract WAV if configured, return (audio_for_model, wav_to_cleanup_or_None).

    The WAV name embeds the source content signature
    (`{stem}_{signature[:10]}.wav`): replacing the source media with
    different content under the same filename produces a DIFFERENT wav
    name, so a stale temp WAV can never be transcribed by mistake. The WAV
    lives in local scratch (NVMe on Colab), not on Drive.
    """
    if config.extract_temp_wav:
        t = time.time()
        wav_path = paths.audio_tmp / f"{media_path.stem}_{signature[:10]}.wav"
        extract_audio_wav(
            media_path, wav_path, config.sample_rate, timeout_s=config.ffmpeg_timeout_s
        )
        timings["extract_wav_s"] = round(time.time() - t, 2)
        return wav_path, wav_path
    return media_path, None


def _maybe_diarize(
    audio_for_model: Path,
    paths: WorkspacePaths,
    config: TranscriptionConfig,
    media_stem: str,
    timings: dict[str, float],
    engine: DiarizationEngine | None,
) -> tuple[list[dict] | None, str | None]:
    """Run diarization on the full audio (cache invalidated by params hash).

    Returns:
        Tuple ``(turns, failure_reason)``:
            - diarization disabled       -> (None, None)
            - diarization succeeded      -> (turns, None)
            - diarization FAILED         -> (None, "<ExcType>: message")
        The caller MUST treat a non-None failure_reason as a degraded run
        — never as a silent "ok".
    """
    if not config.enable_diarization:
        return None, None
    cache_path = paths.diar_cache / f"{media_stem}_{diarization_params_hash(config)}.diar.json"
    t = time.time()
    failure: str | None = None
    try:
        own_engine = engine is None
        active = engine if engine is not None else DiarizationEngine(config)
        try:
            turns = active.diarize(audio_for_model, cache_path)
        finally:
            if own_engine:
                active.close()
        timings["diarization_s"] = round(time.time() - t, 2)
        return turns, None
    except RuntimeError as e:
        logger.error(f"Diarization failed (known): {e}")
        logger.error("Check: HF token, model terms, available VRAM")
        failure = f"{type(e).__name__}: {e}"
    except (ImportError, AttributeError) as e:
        logger.error(f"Diarization failed (version mismatch): {e}")
        logger.error("Verify pyannote.audio>=4.0 is installed")
        traceback.print_exc()
        failure = f"{type(e).__name__}: {e}"
    except Exception as e:
        logger.error(f"Diarization failed (uncategorized): {type(e).__name__}: {e}")
        traceback.print_exc()
        failure = f"{type(e).__name__}: {e}"
    timings["diarization_s"] = round(time.time() - t, 2)
    return None, failure


def _should_chunk_audio(audio_path: Path, config: TranscriptionConfig) -> tuple[bool, float]:
    """Decide whether to use the DEPRECATED external chunking path."""
    if config.long_audio_threshold_min <= 0:
        return False, 0.0
    duration_s = get_audio_duration_seconds(audio_path)
    threshold_s = config.long_audio_threshold_min * 60
    return duration_s > threshold_s, duration_s


def _cleanup_chunks(chunks: list[AudioChunk], original_wav: Path) -> None:
    """Delete chunk WAVs (but never the original full WAV)."""
    for chunk in chunks:
        if chunk.path == original_wav:
            continue
        if chunk.path.exists():
            try:
                chunk.path.unlink()
            except OSError as e:
                logger.warning(f"Could not delete chunk {chunk.path.name}: {e}")


def _find_previous_run(
    paths: WorkspacePaths,
    config: TranscriptionConfig,
    media_path: Path,
    signature: str,
    diar_model: str | None,
) -> dict[str, Any] | None:
    """Ledger lookup with one-time fallback to the legacy full SHA-256.

    Histories written before 0.3 store the FULL SHA-256. When the fast
    lookup misses and any history exists, we compute the full hash once
    and retry; on a hit we append a migration record under the fast
    signature so the expensive fallback never runs again for this file.
    """
    from speakerscribe.persistence import find_run_by_hash, register_run

    existing = find_run_by_hash(paths.ledger_path, signature, config.model, diar_model)
    if existing is not None or config.hash_mode != "fast":
        return existing
    if not (paths.ledger_path.exists() or paths.db_path.exists()):
        return None

    full_sig = file_signature(media_path, mode="full")
    legacy = find_run_by_hash(paths.ledger_path, full_sig, config.model, diar_model)
    if legacy is None:
        return None
    logger.info(
        f"Legacy full-hash record found for {media_path.name} — "
        f"migrating to fast signature ({signature[:10]})"
    )
    register_run(
        paths.ledger_path,
        {
            "audio_file": legacy.get("audio_file"),
            "processed_at": legacy.get("processed_at"),
            "duration_minutes": legacy.get("duration_min"),
            "elapsed_seconds": legacy.get("elapsed_sec"),
            "real_time_factor": legacy.get("rtf"),
            "model": legacy.get("asr_model"),
            "diarization_model": legacy.get("diar_model"),
            "total_segments": legacy.get("n_segments"),
            "total_words": legacy.get("n_words"),
            "config": {"migrated_from_full_hash": full_sig},
            "package_version": legacy.get("package_version"),
        },
        file_hash=signature,
        quality_ok=bool(legacy["quality_ok"]) if legacy.get("quality_ok") is not None else None,
        quality_flags_json=legacy.get("quality_flags"),
        status=str(legacy.get("status") or "ok"),
    )
    return legacy


def _attempt_quality(
    metadata: dict[str, Any],
    config: TranscriptionConfig,
) -> QualityReport | None:
    """Evaluate quality for one attempt (or None when disabled)."""
    if not config.evaluate_quality:
        return None
    t = time.time()
    report = evaluate_transcription_quality(metadata)
    metadata.setdefault("timings", {})["quality_check_s"] = round(time.time() - t, 2)
    return report


def _should_auto_retry(config: TranscriptionConfig, report: QualityReport | None) -> bool:
    """True when the run hit a critical hallucination signature and retry is on."""
    if not config.auto_retry_on_critical or report is None or report.quality_ok:
        return False
    return any(f.code in AUTO_RETRY_CODES and f.severity == Severity.CRITICAL for f in report.flags)


def _promote_or_discard_retry(outputs: RunOutputs, keep_retry: bool) -> None:
    """Atomically replace finals with retry outputs, or delete the retry set."""
    for final_path, retry_path in outputs.pairs():
        if retry_path is None or not retry_path.exists():
            continue
        if keep_retry and final_path is not None:
            os.replace(retry_path, final_path)
        else:
            retry_path.unlink()


def _transcribe_attempt(
    model: WhisperModel,
    audio_for_model: Path,
    chunks: list[AudioChunk],
    outputs: RunOutputs,
    config: TranscriptionConfig,
    diar_turns: list[dict] | None,
    media_path: Path,
    diar_failure: str | None,
) -> tuple[dict[str, Any], QualityReport | None]:
    """Run one transcription attempt to the given outputs + quality check."""
    if chunks:
        metadata = transcribe_chunked(
            model=model,
            chunks=chunks,
            output_txt=outputs.txt,
            output_srt=outputs.srt,
            output_json=outputs.json,
            config=config,
            diar_turns=diar_turns,
            source_media=media_path,
            output_jsonl=outputs.jsonl,
        )
    else:
        metadata = transcribe_streaming(
            model=model,
            audio_path=audio_for_model,
            output_txt=outputs.txt,
            output_srt=outputs.srt,
            output_json=outputs.json,
            config=config,
            diar_turns=diar_turns,
            source_media=media_path,
            output_jsonl=outputs.jsonl,
        )

    if diar_failure is not None:
        metadata["diarization_failed"] = True
        metadata["diarization_error"] = diar_failure
    report = _attempt_quality(metadata, config)
    return metadata, report


def process_one(
    media_path: Path,
    paths: WorkspacePaths,
    model: WhisperModel,
    config: TranscriptionConfig,
    *,
    diar_engine: DiarizationEngine | None = None,
) -> dict[str, Any]:
    """Run the full pipeline for a single media file.

    Idempotency strategy:
        - When `enable_runs_db=True` (default): skip if a previous run with
          the same content signature + ASR model + diarization model is
          ledgered with status 'ok' AND the JSON/TXT outputs still exist.
          Robust to renames and unrelated parameter tweaks. Signatures use
          `config.hash_mode` ("fast" by default), with a one-time fallback
          to the legacy full SHA-256 for pre-0.3 histories.
        - When `enable_runs_db=False`: skip if both the JSON and TXT outputs
          exist and `force_reprocess` is False (filename-based; less robust).

    Degradation contract:
        If diarization was ENABLED but failed, the run completes with
        `status="ok_degraded"`, a CRITICAL `DIARIZATION_FAILED` quality
        flag, and is ledgered with `diar_model=None` — so the skip lookup
        for a future run WITH diarization (keyed on the diarization model)
        misses it and retries. The user is never told "ok" for a transcript
        that silently lost its speaker labels.

    Auto-retry contract:
        When the quality check raises a CRITICAL `REPETITIONS` or `HIGH_WPM`
        flag (Whisper loop signatures) and `auto_retry_on_critical=True`,
        the file is re-transcribed ONCE with anti-loop decoding overrides
        into sibling `.retry` outputs. The attempt with fewer critical flags
        wins; both attempts are appended to the ledger for audit.

    Args:
        media_path: Path to the source audio/video file.
        paths: WorkspacePaths instance.
        model: A WhisperModel already loaded (see `loaded_whisper`).
        config: Pipeline configuration.
        diar_engine: Optional resident DiarizationEngine (batch reuse). When
            None and diarization is enabled, an ephemeral engine is created
            and released for this single file.

    Returns:
        Dict with metadata + status: "ok" | "ok_degraded" | "skipped" | "error".
    """
    base_name = f"{media_path.stem}_{config.model}"
    outputs = RunOutputs(
        txt=paths.transcripts / f"{base_name}.txt",
        srt=paths.transcripts / f"{base_name}.srt",
        json=paths.transcripts / f"{base_name}.json",
        jsonl=(
            paths.transcripts / f"{base_name}.segments.jsonl" if config.streaming_jsonl else None
        ),
    )
    output_md = paths.transcripts / f"{base_name}.transcript.md"

    # Content signature: always computed — it also names the temp WAV, which
    # is what makes WAV reuse content-correct (a replaced source can never
    # resurrect a stale WAV).
    signature = file_signature(media_path, mode=config.hash_mode)
    diar_model = config.diarization_model if config.enable_diarization else None

    # ── Idempotency
    if not config.force_reprocess:
        if config.enable_runs_db:
            existing = _find_previous_run(paths, config, media_path, signature, diar_model)
            if (
                existing
                and existing.get("status") == "ok"
                and outputs.json.exists()
                and outputs.txt.exists()
            ):
                logger.info(
                    f"Already processed: {media_path.name} (sig={signature[:8]}) — skipping"
                )
                return {
                    "status": "skipped",
                    "audio_file": media_path.name,
                    "file_hash": signature,
                    "previous_run_id": existing.get("id"),
                    "base_name": base_name,
                }
        elif outputs.json.exists() and outputs.txt.exists():
            logger.info(f"Outputs already exist for: {media_path.name} — skipping")
            return {
                "status": "skipped",
                "audio_file": media_path.name,
                "base_name": base_name,
            }
    else:
        logger.info(f"Force re-process for: {media_path.name}")

    # Per-stage telemetry
    timings: dict[str, float] = {}
    t_start = time.time()

    # 1. Extract audio to WAV (local scratch, content-signature name)
    audio_for_model, wav_path = _maybe_extract_wav(media_path, paths, config, timings, signature)

    # 2. Diarization — always on the FULL audio for speaker consistency.
    #    (With a resident engine, model load is paid once per batch. Whisper
    #    and pyannote coexist in VRAM on a 16 GB T4 with room to spare.)
    diar_turns, diar_failure = _maybe_diarize(
        audio_for_model, paths, config, media_path.stem, timings, diar_engine
    )

    # 3. DEPRECATED external chunking path (only when user opted in)
    should_chunk, duration_s = _should_chunk_audio(audio_for_model, config)
    chunks: list[AudioChunk] = []
    if should_chunk:
        warnings.warn(
            "External long-audio chunking is deprecated since 0.3: the batched "
            "transcription path handles long audio natively and more accurately. "
            "Set long_audio_threshold_min=0 (default) to use it.",
            DeprecationWarning,
            stacklevel=2,
        )
        logger.warning(
            f"DEPRECATED chunking path active ({duration_s / 60:.1f} min > "
            f"{config.long_audio_threshold_min} min threshold)"
        )
        t = time.time()
        chunks = split_long_audio(
            input_wav=audio_for_model,
            output_dir=paths.audio_chunks,
            chunk_duration_s=config.chunk_duration_min * 60,
            overlap_s=config.chunk_overlap_s,
            sample_rate=config.sample_rate,
            timeout_s=config.ffmpeg_timeout_s,
        )
        timings["split_audio_s"] = round(time.time() - t, 2)

    # 4. Transcribe (attempt 1) + quality
    metadata, report = _transcribe_attempt(
        model, audio_for_model, chunks, outputs, config, diar_turns, media_path, diar_failure
    )
    attempt_kept = 1
    attempts_audit: list[dict[str, Any]] = [
        {"attempt": 1, "n_critical": report.n_critical if report else 0}
    ]

    # 5. Auto-retry on critical hallucination signatures
    if report is not None and _should_auto_retry(config, report):
        logger.warning(
            f"CRITICAL quality flags {sorted(f.code for f in report.flags if f.severity == Severity.CRITICAL)} "
            f"— auto-retrying once with anti-loop decoding {AUTO_RETRY_OVERRIDES}"
        )
        retry_config = config.model_copy(update=AUTO_RETRY_OVERRIDES)
        retry_outputs = outputs.with_retry_suffix()
        metadata_2, report_2 = _transcribe_attempt(
            model,
            audio_for_model,
            chunks,
            retry_outputs,
            retry_config,
            diar_turns,
            media_path,
            diar_failure,
        )
        n_crit_1 = report.n_critical if report else 0
        n_crit_2 = report_2.n_critical if report_2 else 0
        keep_retry = n_crit_2 < n_crit_1
        attempts_audit.append({"attempt": 2, "n_critical": n_crit_2, "kept": keep_retry})
        attempts_audit[0]["kept"] = not keep_retry

        # Ledger the LOSING attempt for audit before promoting the winner.
        if config.enable_runs_db:
            from speakerscribe.persistence import register_run

            loser_meta, loser_report, loser_n = (
                (metadata, report, 1) if keep_retry else (metadata_2, report_2, 2)
            )
            register_run(
                paths.ledger_path,
                loser_meta,
                file_hash=signature,
                quality_ok=loser_report.quality_ok if loser_report else None,
                quality_flags_json=(
                    json.dumps([str(f) for f in loser_report.flags], ensure_ascii=False)
                    if loser_report
                    else None
                ),
                status="retried",
                attempt=loser_n,
            )

        _promote_or_discard_retry(outputs, keep_retry)
        if keep_retry:
            metadata, report, attempt_kept = metadata_2, report_2, 2
            logger.success(f"Auto-retry improved quality ({n_crit_1} -> {n_crit_2} critical)")
        else:
            logger.warning(
                f"Auto-retry did not improve quality ({n_crit_1} -> {n_crit_2} critical) — "
                f"keeping attempt 1"
            )
    metadata["attempts"] = attempts_audit
    metadata["attempt_kept"] = attempt_kept

    # Merge external timings into metadata (transcription already added its own)
    metadata.setdefault("timings", {})
    for k, v in timings.items():
        metadata["timings"][k] = v

    # 6. Markdown transcript
    if config.generate_transcript_md and metadata.get("segments"):
        t = time.time()
        generate_transcript_md(
            metadata["segments"],
            output_md,
            metadata,
            gap_max_s=config.gap_max_s_transcript,
            remove_fillers=config.remove_fillers,
        )
        metadata["timings"]["transcript_md_s"] = round(time.time() - t, 2)

    # 7. Splits + unified-for-LLM
    t = time.time()
    split_text_by_words(
        outputs.txt,
        paths.splits,
        config.words_per_split,
        has_speakers=metadata.get("diarization_enabled", False),
    )
    if config.produce_unified_for_llm:
        write_unified_for_llm(outputs.txt, paths.splits)
    metadata["timings"]["splits_s"] = round(time.time() - t, 2)

    # 8. Cleanup temp WAV(s) and chunks
    if config.delete_chunk_wavs and chunks:
        _cleanup_chunks(chunks, audio_for_model)
        logger.debug("Chunk WAVs deleted")
    if config.delete_temp_wav and wav_path and wav_path.exists():
        wav_path.unlink()
        logger.debug("Temporary WAV deleted")

    # 9. Final status (degraded when diarization was requested but failed)
    status = "ok_degraded" if diar_failure is not None else "ok"
    if report is not None:
        if report.quality_ok:
            logger.success("Quality: OK")
        else:
            logger.warning(report.summary())

    # 10. Ledger
    if config.enable_runs_db:
        from speakerscribe.persistence import register_run

        register_run(
            paths.ledger_path,
            metadata,
            file_hash=signature,
            quality_ok=report.quality_ok if report else None,
            quality_flags_json=(
                json.dumps([str(f) for f in report.flags], ensure_ascii=False) if report else None
            ),
            status=status,
            attempt=attempt_kept,
        )

    metadata["timings"]["total_s"] = round(time.time() - t_start, 2)
    metadata["status"] = status
    metadata["base_name"] = base_name
    if config.enable_runs_db:
        metadata["file_hash"] = signature
    if report:
        metadata["quality_ok"] = report.quality_ok
        metadata["quality_flags"] = [str(f) for f in report.flags]
    return metadata


def report_speaker_distribution(speakers_summary: dict[str, int]) -> None:
    """Log a visual histogram of segment counts per speaker."""
    if not speakers_summary:
        return
    total = sum(speakers_summary.values())
    if total == 0:
        return
    items = sorted(speakers_summary.items(), key=lambda kv: -kv[1])
    max_label = max(len(k) for k, _ in items)
    logger.info("Segment distribution by speaker:")
    for spk, n in items:
        pct = n / total
        bar = "#" * int(pct * 30)
        logger.info(f"   {spk:<{max_label}}  {n:>5,}  {pct:>6.1%}  {bar}")


def process_batch(
    paths: WorkspacePaths,
    config: TranscriptionConfig,
) -> list[dict[str, Any]]:
    """Process every media file in `data/`.

    Heavy models are loaded ONCE for the whole batch: the Whisper model via
    the `loaded_whisper` context manager and the pyannote pipeline via a
    resident `DiarizationEngine` (lazy — a fully cached batch never loads
    it). Both are guaranteed released at the end, even on exceptions.

    Args:
        paths: WorkspacePaths instance.
        config: Pipeline configuration.

    Returns:
        List of metadata dicts, one per file
        (status: ok / ok_degraded / skipped / error).
    """
    preflight_check(paths, config)

    media = paths.list_media_files()
    logger.info(f"{len(media)} file(s) detected:")
    for i, v in enumerate(media, 1):
        logger.info(f"   {i}. {v.name} ({v.stat().st_size / 1e6:.1f} MB)")

    results: list[dict[str, Any]] = []
    t_batch_start = time.time()
    engine_ctx = DiarizationEngine(config) if config.enable_diarization else nullcontext(None)
    with loaded_whisper(config) as model, engine_ctx as engine:
        for i, item in enumerate(media, 1):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"[{i}/{len(media)}] {item.name}")
            logger.info(f"{'=' * 60}")
            try:
                meta = process_one(item, paths, model, config, diar_engine=engine)
                results.append(meta)
                if meta.get("status") in ("ok", "ok_degraded") and meta.get("speakers_summary"):
                    report_speaker_distribution(meta["speakers_summary"])
            except Exception as e:
                logger.error(f"ERROR on {item.name}: {type(e).__name__}: {e}")
                traceback.print_exc()
                if config.enable_runs_db:
                    try:
                        from speakerscribe.persistence import register_run

                        register_run(
                            paths.ledger_path,
                            {
                                "audio_file": item.name,
                                "model": config.model,
                                "diarization_model": (
                                    config.diarization_model if config.enable_diarization else None
                                ),
                                "processed_at": datetime.now(tz=timezone.utc).isoformat(),
                                "config": (
                                    config.model_dump() if hasattr(config, "model_dump") else {}
                                ),
                            },
                            file_hash=file_signature(item, mode=config.hash_mode),
                            status="error",
                            error_message=f"{type(e).__name__}: {e}",
                        )
                    except Exception:
                        logger.debug("Could not ledger the error record", exc_info=True)
                results.append(
                    {
                        "status": "error",
                        "audio_file": item.name,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
                continue

    # ── Final report
    total_elapsed = time.time() - t_batch_start
    n_ok = sum(1 for r in results if r.get("status") == "ok")
    n_degraded = sum(1 for r in results if r.get("status") == "ok_degraded")
    n_skip = sum(1 for r in results if r.get("status") == "skipped")
    n_err = sum(1 for r in results if r.get("status") == "error")
    n_words = sum(
        r.get("total_words", 0) for r in results if r.get("status") in ("ok", "ok_degraded")
    )
    n_speakers_total = sum(
        len(r.get("speakers_summary") or {}) for r in results if r.get("status") == "ok"
    )

    logger.info(
        "\n"
        "===============================================================\n"
        "                       BATCH FINAL REPORT                       \n"
        "===============================================================\n"
        f"  Total time         : {total_elapsed / 60:>8.1f} min\n"
        f"  Files OK           : {n_ok:>3d}\n"
        f"  Files OK (degraded): {n_degraded:>3d}\n"
        f"  Files skipped      : {n_skip:>3d}\n"
        f"  Files with errors  : {n_err:>3d}\n"
        f"  Total words        : {n_words:>10,}\n"
        f"  Speakers (sum)     : {n_speakers_total:>3d}\n"
        "==============================================================="
    )
    return results


__all__ = [
    "AUTO_RETRY_CODES",
    "AUTO_RETRY_OVERRIDES",
    "preflight_check",
    "process_batch",
    "process_one",
    "report_speaker_distribution",
]
