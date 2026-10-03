"""Google Colab specifics, isolated so the rest of the package runs anywhere."""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import subprocess
import sys
import time
from collections.abc import Callable
from typing import Any

COUNTDOWN_STEP_S = 5

CUBLAS_WHEELS: dict[str, str] = {"12": "nvidia-cublas-cu12>=12.4,<13"}
"""pip requirement providing ``libcublas.so.<major>`` for CTranslate2."""


def in_colab() -> bool:
    """True inside a Google Colab runtime."""
    try:
        return importlib.util.find_spec("google.colab") is not None
    except (ImportError, ValueError):
        return False


def provide_ctranslate2_cuda_libs(
    *,
    install: bool | None = None,
    run: Callable[..., Any] = subprocess.run,
    emit: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Make CTranslate2's cuBLAS loadable, installing it once per VM in Colab.

    Since its CUDA 13 images (torch ``+cu130``), Colab no longer ships the
    CUDA 12 cuBLAS that every CTranslate2 4.x wheel dlopens. Installing the
    ``nvidia-cublas-cu12`` wheel (~600 MB, once per VM) and preloading it
    fixes that without touching torch's own CUDA 13 libraries.

    Args:
        install: Install the wheel when missing. Defaults to `in_colab()`:
            outside Colab the environment is the user's to manage.
        run: ``subprocess.run`` (injected in tests).
        emit: Progress output.

    Returns:
        `ensure_ctranslate2_cuda_libs` report, plus ``installed`` (the pip
        requirement) when this call installed it.

    Raises:
        EnvironmentIncompatibleError: Missing and not installable here.
    """
    from speakerscribe.environment import (
        EnvironmentIncompatibleError,
        ctranslate2_cublas_soname,
        ensure_ctranslate2_cuda_libs,
    )

    try:
        return ensure_ctranslate2_cuda_libs()
    except EnvironmentIncompatibleError:
        if not (in_colab() if install is None else install):
            raise
    major = ctranslate2_cublas_soname().rsplit(".", 1)[-1]
    requirement = CUBLAS_WHEELS.get(major)
    if requirement is None:
        raise EnvironmentIncompatibleError(
            f"CTranslate2 necesita libcublas.so.{major} y no hay rueda conocida para instalarla."
        )
    emit(
        f"   📦 Instalando {requirement}: CTranslate2 necesita cuBLAS de CUDA {major} y "
        "este entorno no lo trae (~600 MB, una vez por máquina)…"
    )
    result = run(
        [sys.executable, "-m", "pip", "install", "-q", requirement],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "")[-800:]
        raise EnvironmentIncompatibleError(f"pip install {requirement} falló: {detail}")
    importlib.invalidate_caches()
    info = ensure_ctranslate2_cuda_libs()
    emit(f"   ✔ cuBLAS listo: {info['source']}")
    return {**info, "installed": requirement}


def shutdown_runtime(
    delay_s: int,
    reason: str,
    *,
    emit: Callable[[str], None] = print,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Flush Drive and release the Colab VM after a cancellable countdown.

    Press ⏹ (KeyboardInterrupt) during the countdown to cancel.

    Returns:
        True if the release was requested; False when cancelled or outside
        Colab.
    """
    if not in_colab():
        emit("ℹ️ Fuera de Colab: no hay máquina que apagar.")
        return False
    emit(f"🛑 Apagado en {delay_s} s (motivo: {reason}) — pulse ⏹ para cancelar.")
    try:
        remaining = int(delay_s)
        while remaining > 0:
            emit(f"   … {remaining:>3d} s")
            sleep(min(COUNTDOWN_STEP_S, remaining))
            remaining -= COUNTDOWN_STEP_S
    except KeyboardInterrupt:
        emit("✋ Apagado CANCELADO.")
        return False
    emit("💾 Sincronizando Drive (flush_and_unmount)…")
    with contextlib.suppress(Exception):
        from google.colab import drive

        drive.flush_and_unmount()
    from google.colab import runtime

    emit("🔌 Liberando la máquina…")
    runtime.unassign()
    return True


__all__ = ["in_colab", "shutdown_runtime"]
