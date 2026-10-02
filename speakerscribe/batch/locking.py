"""Exclusive state lock with heartbeat (one batch per state folder at a time).

Closed-by-default semantics (from notebook v5): an unreadable lock is never
stolen automatically; a stale one (no heartbeat for `stale_after_s`) is
archived and replaced; ownership is re-verified before every result
promotion so a run that lost its lock cannot overwrite another run's work.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any

from speakerscribe.batch.fsio import atomic_write_json, utc_now

HEARTBEAT_INTERVAL_S = 60
DEFAULT_STALE_AFTER_S = 600


class LockError(RuntimeError):
    """The state folder is locked by another run, or the lock was lost."""


class StateLock:
    """File lock at `path`, owned by `owner` (the run id).

    Args:
        path: Lock file location (``active.lock.json``).
        owner: Unique id of this run.
        force_take: Take over an unreadable or fresh lock (dangerous: only
            when you are sure no other session is running).
        stale_after_s: Heartbeat age after which a lock counts as orphaned.
    """

    def __init__(
        self,
        path: Path,
        owner: str,
        *,
        force_take: bool = False,
        stale_after_s: int = DEFAULT_STALE_AFTER_S,
    ) -> None:
        self.path = path
        self.owner = owner
        self.force_take = force_take
        self.stale_after_s = stale_after_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.lost_reason: str | None = None

    def _payload(self) -> dict[str, Any]:
        return {
            "owner": self.owner,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "heartbeat_epoch": time.time(),
            "heartbeat_utc": utc_now(),
        }

    def acquire(self) -> None:
        """Take the lock or raise `LockError`."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(2):
            try:
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                try:
                    previous = json.loads(self.path.read_text(encoding="utf-8"))
                    age = time.time() - float(previous.get("heartbeat_epoch", 0))
                except (OSError, ValueError) as e:
                    if not self.force_take:
                        raise LockError(
                            "El lock existente es ilegible; por seguridad no se toma. "
                            "Verifique que no haya otra sesión y use force_take_lock=True."
                        ) from e
                    previous, age = {}, float("inf")
                if not self.force_take and age <= self.stale_after_s:
                    raise LockError(
                        "Ya hay otra ejecución activa sobre este estado "
                        f"(owner={str(previous.get('owner', '?'))[:8]}, latido hace {age:.0f} s). "
                        "Si es un lock huérfano: force_take_lock=True."
                    ) from None
                if attempt == 0:
                    os.replace(
                        self.path, self.path.with_name(f"{self.path.name}.stale.{int(time.time())}")
                    )
                    continue
                raise LockError("No se pudo tomar el lock tras archivar el anterior") from None
            else:
                try:
                    os.write(fd, json.dumps(self._payload()).encode())
                    with contextlib.suppress(OSError):
                        os.fsync(fd)
                finally:
                    os.close(fd)
                break
        self._thread = threading.Thread(target=self._heartbeat, daemon=True, name="state-lock")
        self._thread.start()

    def _heartbeat(self) -> None:
        while not self._stop.wait(HEARTBEAT_INTERVAL_S):
            try:
                current = json.loads(self.path.read_text(encoding="utf-8"))
                if current.get("owner") != self.owner:
                    self.lost_reason = "el lock pasó a otro propietario"
                    return
                atomic_write_json(self.path, self._payload())
            except (OSError, ValueError):
                continue  # transient Drive hiccup: retry next minute

    def assert_owned(self) -> None:
        """Raise `LockError` unless this run still owns the lock."""
        if self.lost_reason:
            raise LockError(f"Se perdió el lock de estado: {self.lost_reason}")
        for attempt in (1, 2):
            try:
                current = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                if attempt == 2:
                    raise LockError("Se perdió el lock de estado: ilegible dos veces") from None
                time.sleep(2)
                continue
            if current.get("owner") != self.owner:
                self.lost_reason = "el propietario del lock no coincide"
                raise LockError(f"Se perdió el lock de estado: {self.lost_reason}")
            return

    def release(self) -> None:
        """Stop the heartbeat and delete the lock if still ours."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        with contextlib.suppress(OSError, ValueError):
            current = json.loads(self.path.read_text(encoding="utf-8"))
            if current.get("owner") == self.owner:
                self.path.unlink()

    def __enter__(self) -> StateLock:
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


__all__ = ["DEFAULT_STALE_AFTER_S", "LockError", "StateLock"]
