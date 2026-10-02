"""Resource probes and the "black box" monitor thread.

When a Colab VM disappears the notebook leaves no trace of why. A one-line
JSON sample per minute (RAM, VRAM, disk, current file and stage) answers it
afterwards: RAM climbing to the ceiling means OOM; a flat line cut short
means an external kill (quota, disconnect).
"""

from __future__ import annotations

import contextlib
import ctypes
import gc
import json
import os
import shutil
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from speakerscribe.batch.fsio import utc_now

_MEMINFO = Path("/proc/meminfo")
MIN_SAMPLE_INTERVAL_S = 15


def _meminfo_bytes() -> dict[str, float]:
    info: dict[str, float] = {}
    for line in _MEMINFO.read_text().splitlines():
        key, _, value = line.partition(":")
        info[key] = float(value.split()[0]) * 1024
    return info


def ram_pct() -> float:
    """System RAM in use, percent (0.0 when it cannot be measured)."""
    try:
        info = _meminfo_bytes()
        return round(100.0 * (info["MemTotal"] - info["MemAvailable"]) / info["MemTotal"], 1)
    except (OSError, KeyError, ValueError, ZeroDivisionError):
        return 0.0


def release_memory() -> None:
    """gc + empty the CUDA cache + ``malloc_trim(0)`` to return heap to the OS.

    Without malloc_trim, memory freed by Python stays in glibc arenas and
    the system-wide RAM percentage (what the guard measures) never drops.
    """
    gc.collect()
    with contextlib.suppress(Exception):
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    with contextlib.suppress(Exception):
        ctypes.CDLL("libc.so.6").malloc_trim(0)


def sample(run_id: str, started_monotonic: float, scratch: Path, where: dict[str, Any]) -> dict:
    """One JSON-serializable resource sample. Never raises."""
    data: dict[str, Any] = {
        "utc": utc_now(),
        "run_id": run_id,
        "min_sesion": round((time.monotonic() - started_monotonic) / 60, 2),
    }
    with contextlib.suppress(Exception):
        info = _meminfo_bytes()
        data["ram_total_gb"] = round(info["MemTotal"] / 1e9, 2)
        data["ram_usada_gb"] = round((info["MemTotal"] - info["MemAvailable"]) / 1e9, 2)
        data["ram_pct"] = round(100 * data["ram_usada_gb"] / data["ram_total_gb"], 1)
    with contextlib.suppress(Exception):
        import torch

        if torch.cuda.is_available():
            data["vram_asignada_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
            data["vram_reservada_gb"] = round(torch.cuda.memory_reserved() / 1e9, 2)
    with contextlib.suppress(Exception):
        base = scratch if scratch.exists() else Path("/")
        data["disco_libre_gb"] = round(shutil.disk_usage(base).free / 1e9, 2)
    data["archivo"] = where.get("archivo")
    data["etapa"] = where.get("etapa")
    return data


class ResourceMonitor(threading.Thread):
    """Append a resource sample to `path` every `interval_s` seconds.

    Args:
        path: ``telemetria.jsonl``.
        run_id: Current run id.
        started_monotonic: Session start (``time.monotonic()``).
        scratch: Local scratch folder (disk free is measured there).
        where: Mutable dict with keys ``archivo`` and ``etapa`` updated by
            the runner (read-only here).
        interval_s: Seconds between samples (min 15).
        heartbeat_min: Print a one-line heartbeat every N minutes (0 = off).
        emit: Printer for heartbeats (defaults to print).
    """

    def __init__(
        self,
        path: Path,
        *,
        run_id: str,
        started_monotonic: float,
        scratch: Path,
        where: dict[str, Any],
        interval_s: int = 60,
        heartbeat_min: int = 10,
        emit: Callable[[str], None] = print,
    ) -> None:
        super().__init__(daemon=True, name="resource-monitor")
        self.path = path
        self.run_id = run_id
        self.started = started_monotonic
        self.scratch = scratch
        self.where = where
        self.interval_s = max(MIN_SAMPLE_INTERVAL_S, int(interval_s))
        self.heartbeat_min = max(0, int(heartbeat_min))
        self.emit = emit
        self._halt = threading.Event()
        self.peak_ram_gb = 0.0
        self.last: dict[str, Any] = {}

    def _write(self, reason: str) -> dict[str, Any]:
        data = sample(self.run_id, self.started, self.scratch, self.where)
        data["motivo"] = reason
        self.last = data
        self.peak_ram_gb = max(self.peak_ram_gb, float(data.get("ram_usada_gb") or 0.0))
        with contextlib.suppress(OSError):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(data, ensure_ascii=False) + "\n")
                fh.flush()
                with contextlib.suppress(OSError):
                    os.fsync(fh.fileno())
        return data

    def start(self) -> None:
        """Write the first sample synchronously, then start sampling."""
        self._write("inicio")
        super().start()

    def run(self) -> None:
        next_beat = self.heartbeat_min
        while not self._halt.wait(self.interval_s):
            data = self._write("periodico")
            if self.heartbeat_min and data.get("min_sesion", 0) >= next_beat:
                next_beat += self.heartbeat_min
                self.emit(
                    f"   🩺 {data.get('min_sesion', 0):.0f} min · RAM "
                    f"{data.get('ram_usada_gb', '?')}/{data.get('ram_total_gb', '?')} GB "
                    f"({data.get('ram_pct', '?')}%) · VRAM {data.get('vram_reservada_gb', '?')} GB "
                    f"· disco {data.get('disco_libre_gb', '?')} GB · {data.get('etapa') or '—'}"
                )

    def stop(self, reason: str = "fin") -> None:
        """Stop sampling and write a final sample."""
        self._halt.set()
        self._write(reason)


__all__ = ["ResourceMonitor", "ram_pct", "release_memory", "sample"]
