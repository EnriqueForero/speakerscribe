"""Pre-flight environment validation.

Run before any heavy model is loaded so the user finds out about missing
inputs, low disk, GPU/VRAM issues or auth problems in seconds, not minutes.
"""

from __future__ import annotations

import shutil
import warnings
from typing import Any

from speakerscribe.config import (
    MIN_VRAM_BY_MODEL,
    TranscriptionConfig,
    WorkspacePaths,
)
from speakerscribe.logging_config import logger


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
