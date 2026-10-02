"""Google Colab specifics, isolated so the rest of the package runs anywhere."""

from __future__ import annotations

import contextlib
import importlib.util
import time
from collections.abc import Callable

COUNTDOWN_STEP_S = 5


def in_colab() -> bool:
    """True inside a Google Colab runtime."""
    try:
        return importlib.util.find_spec("google.colab") is not None
    except (ImportError, ValueError):
        return False


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
