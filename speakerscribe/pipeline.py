"""Pipeline orchestration: process_one, process_batch, preflight_check.

Orchestrates the full pipeline:
    1. Pre-flight environment validation
    2. WAV extraction (ffmpeg)
    3. Long-audio chunking (when applicable)
    4. Diarization on the FULL audio (cached by params hash)
    5. Streaming transcription (single-shot or chunked)
    6. Markdown transcript generation (with optional filler filter)
    7. Word-aware splits + unified-for-LLM file
    8. Quality check (heuristic)
    9. SQLite history (idempotency by file content hash)
"""

from __future__ import annotations

import json
import shutil
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from speakerscribe.audio import (
    AudioChunk,
    calculate_file_hash,
    extract_audio_wav,
    get_audio_duration_seconds,
    split_long_audio,
)
from speakerscribe.config import (
    MIN_VRAM_BY_MODEL,
    TranscriptionConfig,
    WorkspacePaths,
)
from speakerscribe.diarization import diarization_params_hash, diarize_audio
from speakerscribe.logging_config import logger
from speakerscribe.output import (
    generate_transcript_md,
    split_text_by_words,
    write_unified_for_llm,
)
from speakerscribe.quality import evaluate_transcription_quality
from speakerscribe.transcription import (
    load_whisper_model,
    release_whisper_model,
    transcribe_chunked,
    transcribe_streaming,
)

if TYPE_CHECKING:
    from faster_whisper import WhisperModel


def preflight_check(paths: WorkspacePaths, config: TranscriptionConfig) -> dict[str, Any]:
    """Validate the entire environment before loading any heavy model.

    Checks performed:
        1. The data/ folder exists and contains processable files.
        2. There is enough free disk space.
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

    # 2. Disk space
    free_mb = shutil.disk_usage(paths.base).free / 1e6
    required_mb = max(config.disk_margin_min_mb, total_mb * config.disk_margin_factor)
    if free_mb < required_mb:
        raise RuntimeError(
            f"Insufficient disk space: {free_mb:.0f} MB free, ~{required_mb:.0f} MB required."
        )
    logger.info(f"   Disk: {free_mb:,.0f} MB free (~{required_mb:.0f} MB required)")

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
        logger.info(f"   HF token: detected ({token[:8]}...)")
    else:
        logger.info("   Diarization disabled — transcription only")

    return {
        "n_files": len(media),
        "total_mb": round(total_mb, 1),
        "free_mb": round(free_mb, 0),
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
) -> tuple[Path, Path | None]:
    """Extract WAV if configured, return (audio_for_model, wav_path_to_cleanup_or_None)."""
    if config.extract_temp_wav:
        t = time.time()
        wav_path = paths.audio_tmp / f"{media_path.stem}.wav"
        extract_audio_wav(media_path, wav_path, config.sample_rate)
        timings["extract_wav_s"] = round(time.time() - t, 2)
        return wav_path, wav_path
    return media_path, None


def _maybe_diarize(
    audio_for_model: Path,
    paths: WorkspacePaths,
    config: TranscriptionConfig,
    media_stem: str,
    timings: dict[str, float],
) -> list[dict] | None:
    """Run diarization on the full audio (with cache invalidated by params hash)."""
    if not config.enable_diarization:
        return None
    cache_path = paths.diar_cache / f"{media_stem}_{diarization_params_hash(config)}.diar.json"
    t = time.time()
    try:
        turns = diarize_audio(audio_for_model, config, cache_path)
        timings["diarization_s"] = round(time.time() - t, 2)
        return turns
    except RuntimeError as e:
        logger.error(f"Diarization failed (known): {e}")
        logger.error("Check: HF token, model terms, available VRAM")
    except (ImportError, AttributeError) as e:
        logger.error(f"Diarization failed (version mismatch): {e}")
        logger.error("Verify pyannote.audio>=4.0 and restart the runtime after install")
        traceback.print_exc()
    except Exception as e:
        logger.error(f"Diarization failed (uncategorized): {type(e).__name__}: {e}")
        traceback.print_exc()
    timings["diarization_s"] = round(time.time() - t, 2)
    return None


def _should_chunk_audio(audio_path: Path, config: TranscriptionConfig) -> tuple[bool, float]:
    """Decide whether to chunk based on duration vs threshold."""
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


def process_one(
    media_path: Path,
    paths: WorkspacePaths,
    model: WhisperModel,
    config: TranscriptionConfig,
) -> dict[str, Any]:
    """Run the full pipeline for a single media file.

    Idempotency strategy:
        - When `enable_runs_db=True` (default): skip if a previous run with the
          same file hash + ASR model + diarization model is recorded with
          status 'ok' AND the JSON output still exists. This is robust to
          renaming the source file or tweaking unrelated parameters.
        - When `enable_runs_db=False`: skip if both the JSON and TXT outputs
          exist and `force_reprocess` is False (filename-based; less robust).

    Long audios (> `config.long_audio_threshold_min` minutes) are split into
    overlapping chunks for transcription. Diarization always runs on the full
    audio for speaker consistency.

    Args:
        media_path: Path to the source audio/video file.
        paths: WorkspacePaths instance.
        model: A WhisperModel already loaded by `load_whisper_model`.
        config: Pipeline configuration.

    Returns:
        Dict with metadata + status: "ok" | "skipped" | "error".
    """
    base_name = f"{media_path.stem}_{config.model}"
    output_txt = paths.transcripts / f"{base_name}.txt"
    output_srt = paths.transcripts / f"{base_name}.srt"
    output_json = paths.transcripts / f"{base_name}.json"
    output_md = paths.transcripts / f"{base_name}.transcript.md"
    output_jsonl = (
        paths.transcripts / f"{base_name}.segments.jsonl" if config.streaming_jsonl else None
    )

    # Hash for content-based idempotency (only computed when DB is enabled)
    file_hash = calculate_file_hash(media_path) if config.enable_runs_db else ""
    diar_model = config.diarization_model if config.enable_diarization else None

    # ── Idempotency
    if not config.force_reprocess:
        if config.enable_runs_db:
            from speakerscribe.persistence import find_run_by_hash

            existing = find_run_by_hash(paths.db_path, file_hash, config.model, diar_model)
            if (
                existing
                and existing.get("status") == "ok"
                and output_json.exists()
                and output_txt.exists()
            ):
                logger.info(
                    f"Already processed: {media_path.name} (hash={file_hash[:8]}) — skipping"
                )
                return {
                    "status": "skipped",
                    "audio_file": media_path.name,
                    "file_hash": file_hash,
                    "previous_run_id": existing.get("id"),
                    "base_name": base_name,
                }
        elif output_json.exists() and output_txt.exists():
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

    # 1. Extract audio to WAV
    audio_for_model, wav_path = _maybe_extract_wav(media_path, paths, config, timings)

    # 2. Diarization (BEFORE Whisper to free VRAM) — always on the FULL audio
    diar_turns = _maybe_diarize(audio_for_model, paths, config, media_path.stem, timings)

    # 3. Decide whether to chunk for transcription
    should_chunk, duration_s = _should_chunk_audio(audio_for_model, config)
    chunks: list[AudioChunk] = []

    if should_chunk:
        logger.info(
            f"Long audio ({duration_s / 60:.1f} min > "
            f"{config.long_audio_threshold_min} min threshold) — chunking for transcription"
        )
        t = time.time()
        chunks = split_long_audio(
            input_wav=audio_for_model,
            output_dir=paths.audio_chunks,
            chunk_duration_s=config.chunk_duration_min * 60,
            overlap_s=config.chunk_overlap_s,
            sample_rate=config.sample_rate,
        )
        timings["split_audio_s"] = round(time.time() - t, 2)

    # 4. Transcribe — chunked or single-shot
    if should_chunk:
        metadata = transcribe_chunked(
            model=model,
            chunks=chunks,
            output_txt=output_txt,
            output_srt=output_srt,
            output_json=output_json,
            config=config,
            diar_turns=diar_turns,
            source_media=media_path,
            output_jsonl=output_jsonl,
        )
    else:
        metadata = transcribe_streaming(
            model=model,
            audio_path=audio_for_model,
            output_txt=output_txt,
            output_srt=output_srt,
            output_json=output_json,
            config=config,
            diar_turns=diar_turns,
            source_media=media_path,
            output_jsonl=output_jsonl,
        )

    # Merge external timings into metadata (transcription already added its own)
    metadata.setdefault("timings", {})
    for k, v in timings.items():
        metadata["timings"][k] = v

    # 5. Markdown transcript
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

    # 6. Splits + unified-for-LLM
    t = time.time()
    split_text_by_words(
        output_txt,
        paths.splits,
        config.words_per_split,
        has_speakers=metadata.get("diarization_enabled", False),
    )
    if config.produce_unified_for_llm:
        write_unified_for_llm(output_txt, paths.splits)
    metadata["timings"]["splits_s"] = round(time.time() - t, 2)

    # 7. Cleanup temp WAV(s) and chunks
    if config.delete_chunk_wavs and chunks:
        _cleanup_chunks(chunks, audio_for_model)
        logger.debug("Chunk WAVs deleted")
    if config.delete_temp_wav and wav_path and wav_path.exists():
        wav_path.unlink()
        logger.debug("Temporary WAV deleted")

    # 8. Quality check
    quality_report = None
    if config.evaluate_quality:
        t = time.time()
        quality_report = evaluate_transcription_quality(metadata)
        metadata["timings"]["quality_check_s"] = round(time.time() - t, 2)
        if quality_report.quality_ok:
            logger.success("Quality: OK")
        else:
            logger.warning(quality_report.summary())

    # 9. Optional persistence
    if config.enable_runs_db:
        from speakerscribe.persistence import register_run

        quality_flags_json = (
            json.dumps([str(f) for f in quality_report.flags], ensure_ascii=False)
            if quality_report
            else None
        )
        register_run(
            paths.db_path,
            metadata,
            file_hash=file_hash,
            quality_ok=quality_report.quality_ok if quality_report else None,
            quality_flags_json=quality_flags_json,
            status="ok",
        )

    metadata["timings"]["total_s"] = round(time.time() - t_start, 2)
    metadata["status"] = "ok"
    metadata["base_name"] = base_name
    if config.enable_runs_db:
        metadata["file_hash"] = file_hash
    if quality_report:
        metadata["quality_ok"] = quality_report.quality_ok
        metadata["quality_flags"] = [str(f) for f in quality_report.flags]
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
    """Process every media file in `data/`. The Whisper model is loaded ONCE.

    Args:
        paths: WorkspacePaths instance.
        config: Pipeline configuration.

    Returns:
        List of metadata dicts, one per file (status: ok / skipped / error).
    """
    preflight_check(paths, config)

    media = paths.list_media_files()
    logger.info(f"{len(media)} file(s) detected:")
    for i, v in enumerate(media, 1):
        logger.info(f"   {i}. {v.name} ({v.stat().st_size / 1e6:.1f} MB)")

    model = load_whisper_model(config)

    results: list[dict[str, Any]] = []
    t_batch_start = time.time()
    try:
        for i, item in enumerate(media, 1):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"[{i}/{len(media)}] {item.name}")
            logger.info(f"{'=' * 60}")
            try:
                meta = process_one(item, paths, model, config)
                results.append(meta)
                if meta.get("status") == "ok" and meta.get("speakers_summary"):
                    report_speaker_distribution(meta["speakers_summary"])
            except Exception as e:
                logger.error(f"ERROR on {item.name}: {type(e).__name__}: {e}")
                traceback.print_exc()
                if config.enable_runs_db:
                    try:
                        from speakerscribe.persistence import register_run

                        file_hash = calculate_file_hash(item)
                        register_run(
                            paths.db_path,
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
                            file_hash=file_hash,
                            status="error",
                            error_message=f"{type(e).__name__}: {e}",
                        )
                    except Exception:
                        pass
                results.append(
                    {
                        "status": "error",
                        "audio_file": item.name,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
                continue
    finally:
        release_whisper_model(model)

    # ── Final report
    total_elapsed = time.time() - t_batch_start
    n_ok = sum(1 for r in results if r.get("status") == "ok")
    n_skip = sum(1 for r in results if r.get("status") == "skipped")
    n_err = sum(1 for r in results if r.get("status") == "error")
    n_words = sum(r.get("total_words", 0) for r in results if r.get("status") == "ok")
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
        f"  Files skipped      : {n_skip:>3d}\n"
        f"  Files with errors  : {n_err:>3d}\n"
        f"  Total words        : {n_words:>10,}\n"
        f"  Speakers (sum)     : {n_speakers_total:>3d}\n"
        "==============================================================="
    )
    return results


__all__ = [
    "preflight_check",
    "process_batch",
    "process_one",
    "report_speaker_distribution",
]
