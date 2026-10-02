"""Session guards: time budget, RAM guard, circuit breaker, shutdown decision.

All decisions are pure functions of their inputs (clock and RAM readings
are injected), so they are unit-tested without a GPU or a real clock.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from speakerscribe.batch.journal import Event

DEFAULT_DURATION_MIN = 60.0
"""Assumed length of a file whose duration ffprobe could not read."""

RTF_MIN_SAMPLES = 5
RTF_HISTORY = 100


def learned_rtf(events: list[dict[str, Any]], model: str, floor: float) -> float:
    """10th percentile of this model's historical real-time factor.

    The p10 of observed speeds is already the slow tail; it is not cut again
    (v5 lesson: a second haircut deferred files that did fit). Uses the
    most recent `RTF_HISTORY` observations (v5 sorted first and kept the
    fastest 100, biasing estimates optimistic). Fewer than `RTF_MIN_SAMPLES`
    observations fall back to `floor`.
    """
    recent = [
        float(r["real_time_factor"])
        for r in events
        if r.get("event") == Event.JOB_METRICS.value
        and (r.get("profile") or {}).get("asr_model") == model
        and isinstance(r.get("real_time_factor"), int | float)
        and 0 < float(r["real_time_factor"]) < 100
    ][-RTF_HISTORY:]
    seen = sorted(recent)
    if len(seen) < RTF_MIN_SAMPLES:
        return float(floor)
    p10 = seen[max(0, int(math.floor(0.1 * (len(seen) - 1))))]
    return max(1.0, min(60.0, p10))


@dataclass
class SessionBudget:
    """Remaining session time and per-file cost estimates.

    Args:
        max_minutes: Session cap (0 = unlimited).
        margin_minutes: Reserve kept free at the end (writing, shutdown).
        rtf_asr: Expected ASR real-time factor.
        rtf_diar: Expected diarization real-time factor.
        overhead_min: Fixed cost per file (load, extract, publish).
        started: Monotonic start of the session.
        clock: Monotonic clock (injected in tests).
    """

    max_minutes: float
    margin_minutes: float
    rtf_asr: float
    rtf_diar: float
    overhead_min: float
    started: float
    clock: Callable[[], float] = time.monotonic

    @property
    def usable_minutes(self) -> float:
        return (self.max_minutes - self.margin_minutes) if self.max_minutes > 0 else math.inf

    def elapsed_minutes(self) -> float:
        return (self.clock() - self.started) / 60

    def remaining_minutes(self) -> float:
        return self.usable_minutes - self.elapsed_minutes()

    def exhausted(self) -> bool:
        return self.remaining_minutes() <= 0

    def estimate_minutes(self, duration_s: float | None, diarization_cached: bool = False) -> float:
        """Expected wall-clock minutes to process one file."""
        minutes = (max(0.0, duration_s or 0.0) / 60) or DEFAULT_DURATION_MIN
        diar = 0.0 if diarization_cached else minutes / self.rtf_diar
        return minutes / max(self.rtf_asr, 1e-6) + diar + self.overhead_min

    def fits(self, duration_s: float | None, diarization_cached: bool = False) -> bool:
        return self.estimate_minutes(duration_s, diarization_cached) <= self.remaining_minutes()


RamAction = Literal["continue", "recycle", "stop"]


@dataclass(frozen=True)
class RamGuard:
    """Recycle models when RAM climbs; stop cleanly before an OOM kill.

    Measured on Colab T4 (v5 telemetry): ~71 MB of host RAM leak per file
    inside the Python process; releasing the models is the only deep purge
    short of restarting the runtime.
    """

    recycle_pct: float
    stop_pct: float
    recycle_every_n: int = 0

    def decide(self, ram_pct: float, files_since_recycle: int, models_loaded: bool) -> RamAction:
        if ram_pct >= self.stop_pct:
            return "stop"
        due = self.recycle_every_n > 0 and files_since_recycle >= self.recycle_every_n
        if models_loaded and (ram_pct >= self.recycle_pct or due):
            return "recycle"
        return "continue"


@dataclass
class CircuitBreaker:
    """Trip after `threshold` consecutive failures with the same signature.

    Two identical environment errors in a row (e.g. the 2026-10-01
    ``TypeError: open() got an unexpected keyword argument``) mean every
    remaining file will fail the same way: stop instead of burning quota
    and retry attempts.
    """

    threshold: int
    _last: str | None = None
    _count: int = 0
    history: list[str] = field(default_factory=list)

    def record_failure(self, signature: str) -> bool:
        """Register a failure; True if the breaker is now tripped."""
        self._count = self._count + 1 if signature == self._last else 1
        self._last = signature
        self.history.append(signature)
        return self._count >= self.threshold

    def record_success(self) -> None:
        self._last, self._count = None, 0

    @property
    def tripped(self) -> bool:
        return self._count >= self.threshold


def error_signature(exc: BaseException) -> str:
    """Stable identity of an error: type + message without numbers or paths."""
    text = f"{type(exc).__name__}: {exc}"
    text = "".join("#" if ch.isdigit() else ch for ch in text)
    return " ".join(part for part in text.split() if "/" not in part)[:200]


EndReason = Literal[
    "completo",
    "limite_sesion",
    "limite_archivos",
    "limite_ram",
    "abort_fatal",
    "entorno_incompatible",
    "interrumpido",
    "error_orquestador",
]
SHUTDOWN_REASONS = frozenset({"completo", "limite_sesion", "limite_archivos"})


@dataclass(frozen=True)
class ShutdownDecision:
    shutdown: bool
    suspicious: bool
    reason: str


def shutdown_decision(
    *,
    end_reason: str,
    fatal_exception: bool,
    deliverables: int,
    already_done: int,
    failures: int,
    discovered: int,
    session_minutes: float,
    shutdown_at_end: bool,
    shutdown_on_fatal: bool,
    only_if_work: bool,
) -> ShutdownDecision:
    """Whether to release the Colab VM at the end of a run.

    Never shuts down when the machine is needed to read an error: RAM stop
    (restart the runtime instead), fatal/environment aborts, or a run that
    produced nothing while failing (v5 bug: 6 failures + shutdown hid the
    traceback on 2026-10-01). An empty input is the normal steady state
    (transcribed audio leaves ``data/``), so it does shut down.
    """
    suspicious = (
        only_if_work
        and deliverables == 0
        and (
            failures > 0
            or (
                end_reason == "completo"
                and session_minutes < 10
                and already_done == 0
                and discovered > 0  # files present yet nothing produced: needs a look
            )
        )
    )
    if suspicious:
        return ShutdownDecision(
            False, True, "sin entregables: la máquina queda viva para diagnosticar"
        )
    if not shutdown_at_end:
        return ShutdownDecision(False, False, "apagado automático desactivado")
    if end_reason in SHUTDOWN_REASONS:
        return ShutdownDecision(True, False, end_reason)
    if fatal_exception and shutdown_on_fatal:
        return ShutdownDecision(True, False, "error fatal (apagado configurado)")
    return ShutdownDecision(False, False, f"motivo {end_reason}: la máquina queda viva")


__all__ = [
    "DEFAULT_DURATION_MIN",
    "SHUTDOWN_REASONS",
    "CircuitBreaker",
    "EndReason",
    "RamAction",
    "RamGuard",
    "SessionBudget",
    "ShutdownDecision",
    "error_signature",
    "learned_rtf",
    "shutdown_decision",
]
