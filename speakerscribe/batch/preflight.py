"""Preflight checks, cheapest first, so failures surface before expensive work.

* `check_storage` — no GPU, no token: a session with nothing pending never
  needs either.
* `check_gpu_stack` — only when real GPU work exists: CUDA, VRAM, the audio
  decoding self-test (the 2026-10-01 PyAV break) and HuggingFace access.
"""

from __future__ import annotations

import importlib
import shutil
from typing import TYPE_CHECKING, Any

from speakerscribe.batch.errors import sanitize_error
from speakerscribe.batch.paths import BatchPaths

if TYPE_CHECKING:
    from speakerscribe.config import TranscriptionConfig

DIARIZATION_VRAM_MARGIN_GB = 1.5
MIN_SCRATCH_BYTES = 500_000_000
SCRATCH_SAFETY_FACTOR = 1.25
WAV_BYTES_PER_SECOND = 32_000  # 16 kHz mono int16


class PreflightError(RuntimeError):
    """The environment cannot run the batch; the message says how to fix it."""


def check_tools() -> None:
    """ffmpeg and ffprobe must be on PATH."""
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise PreflightError(
            f"No se encontró {', '.join(missing)}. En Colab: reinicie la sesión; "
            "en local: instale ffmpeg."
        )


def check_storage(paths: BatchPaths, in_colab: bool) -> None:
    """Input exists, durable folders are on Drive (Colab), folders exist.

    Raises:
        PreflightError: With an actionable message.
    """
    if not paths.input.is_dir():
        raise PreflightError(f"No existe la carpeta de entrada: {paths.input}")
    try:
        paths.check_durable_on_colab(in_colab)
    except ValueError as e:
        raise PreflightError(str(e)) from e
    check_tools()
    paths.ensure()


def check_hf_access(config: TranscriptionConfig) -> list[str]:
    """Fail BEFORE downloading models if the token is missing or rejected.

    Returns:
        Non-fatal warnings (inconclusive checks).

    Raises:
        PreflightError: Missing/rejected token or gated model not accepted.
    """
    token = config.resolve_hf_token()
    if not token:
        raise PreflightError(
            "No se encontró HF_TOKEN.\n"
            "  1. Token tipo Read (no fine-grained): https://huggingface.co/settings/tokens\n"
            f"  2. Acepte las condiciones: https://huggingface.co/{config.diarization_model}\n"
            "  3. Guárdelo en Secrets de Colab (🔑) como HF_TOKEN, con acceso al notebook."
        )
    warnings: list[str] = []
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    except ImportError as e:  # pragma: no cover - present with pyannote
        return [f"Pre-verificación de Hugging Face omitida ({sanitize_error(e)})"]
    api = HfApi(token=token)
    try:
        api.whoami()
    except Exception as e:
        raise PreflightError(
            f"Hugging Face rechaza HF_TOKEN (¿revocado o mal copiado?): {sanitize_error(e)}"
        ) from e
    try:
        api.model_info(config.diarization_model)
    except GatedRepoError as e:
        raise PreflightError(
            f"El token no tiene acceso a {config.diarization_model}: acepte sus "
            f"condiciones y reintente. {sanitize_error(e)}"
        ) from e
    except HfHubHTTPError as e:
        warnings.append(
            f"Pre-verificación del modelo no concluyente ({sanitize_error(e)}); "
            "se validará al cargar la diarización."
        )
    return warnings


def check_gpu_stack(
    config: TranscriptionConfig, paths: BatchPaths, largest_job: tuple[int, float | None]
) -> dict[str, Any]:
    """Validate everything a GPU job needs, adapting to the GPU present.

    Args:
        config: Engine configuration.
        paths: Batch paths (scratch free space is measured there).
        largest_job: (size in bytes, duration in seconds or None) of the
            largest pending job.

    Returns:
        Environment report: GPU, VRAM, free scratch, package versions,
        decoding self-test and HF warnings.

    Raises:
        PreflightError: With an actionable message.
    """
    from speakerscribe.config import MIN_VRAM_BY_MODEL
    from speakerscribe.environment import check_audio_decoding, package_versions

    try:
        import torch
    except ImportError as e:
        raise PreflightError(f"torch no está instalado: {e}") from e
    if not torch.cuda.is_available():
        raise PreflightError(
            "No hay GPU CUDA. Entorno de ejecución → Cambiar tipo de entorno → "
            "T4 GPU (o superior) y vuelva a ejecutar."
        )
    for module in ("faster_whisper", "ctranslate2", "pyannote.audio"):
        try:
            importlib.import_module(module)
        except Exception as e:
            raise PreflightError(f"El stack no puede importar {module}: {sanitize_error(e)}") from e
    try:
        decoding = check_audio_decoding()
    except Exception as e:
        raise PreflightError(
            f"La autoprueba de decodificación de audio falló: {sanitize_error(e)}"
        ) from e

    gpu = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    required = MIN_VRAM_BY_MODEL.get(config.model, 5.0) + DIARIZATION_VRAM_MARGIN_GB
    if vram_gb < required:
        raise PreflightError(
            f"VRAM insuficiente: {vram_gb:.1f} GB < ~{required:.1f} GB para "
            f"{config.model} + diarización. Use una GPU mayor o un modelo menor."
        )
    size_bytes, duration_s = largest_job
    wav_bytes = (duration_s or 0.0) * WAV_BYTES_PER_SECOND or size_bytes * 4
    needed = max(MIN_SCRATCH_BYTES, int((size_bytes + wav_bytes) * SCRATCH_SAFETY_FACTOR))
    free = shutil.disk_usage(paths.scratch).free
    if free < needed:
        raise PreflightError(
            f"Disco local insuficiente: {free / 1e9:.1f} GB libres, ~{needed / 1e9:.1f} GB requeridos."
        )
    warnings = check_hf_access(config)
    return {
        "gpu": gpu,
        "vram_gb": round(vram_gb, 2),
        "scratch_free_gb": round(free / 1e9, 2),
        "versions": package_versions(),
        "audio_decoding": decoding,
        "warnings": warnings,
    }


__all__ = ["PreflightError", "check_gpu_stack", "check_hf_access", "check_storage", "check_tools"]
