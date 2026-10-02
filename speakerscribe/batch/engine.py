"""The GPU engine behind a small interface (Dependency Inversion).

The runner depends on `TranscriptionEngine`, not on faster-whisper or
pyannote: tests drive the whole batch with a fake engine on a laptop, and
the real `SpeakerscribeEngine` is the only place that touches CUDA.

Model lifecycle: loaded lazily on the first real job (a session with nothing
to do never needs a GPU or a token); pyannote is loaded FIRST because its
failure modes (token, gated terms) are cheap to detect, while Whisper costs
~35 s to load; both are released together on `unload` (RAM-guard recycle).
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from speakerscribe.batch.fsio import fsync_directory
from speakerscribe.batch.telemetry import release_memory

if TYPE_CHECKING:
    from speakerscribe.config import TranscriptionConfig


class TranscriptionEngine(Protocol):
    """What the batch needs from an ASR + diarization engine."""

    @property
    def loaded(self) -> bool: ...

    def load(self) -> None:
        """Load models. Raises on setup failure (token, CUDA, stack)."""
        ...

    def unload(self) -> None:
        """Release every model and the memory they hold. Idempotent."""
        ...

    def transcribe(
        self, media: Path, workdir: Path, diar_cache_in: Path | None, diar_cache_out: Path
    ) -> dict[str, Any]:
        """Transcribe + diarize `media` inside `workdir`.

        Args:
            media: Local media file (staged copy).
            workdir: Private, disposable working folder for this job (logs
                go to ``workdir/_logs``).
            diar_cache_in: Diarization result to reuse (None = compute).
            diar_cache_out: Where the diarization result must be left as
                soon as it exists — EVEN if transcription fails afterwards,
                so a retry only pays for ASR.

        Returns:
            Engine metadata (``speakerscribe.process_one`` contract).
        """
        ...


class DiarizationCacheStore:
    """Durable diarization cache (``<state>/diar_cache``), keyed by content.

    Persisted even after a failed attempt: with identical parameters
    diarization is deterministic, so a retry only pays for ASR.
    """

    def __init__(self, folder: Path) -> None:
        self.folder = folder

    def path(self, name: str) -> Path:
        return self.folder / name

    def lookup(self, name: str) -> Path | None:
        """Valid cache file for `name`, or None (corrupt files are quarantined)."""
        durable = self.path(name)
        if not durable.is_file():
            return None
        try:
            json.loads(durable.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            with contextlib.suppress(OSError):
                os.replace(durable, durable.with_name(durable.name + ".invalid"))
            return None
        return durable

    def persist(self, local: Path | None, name: str) -> bool:
        """Copy a freshly computed local cache to the durable store."""
        if local is None or not local.is_file():
            return False
        try:
            json.loads(local.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        durable = self.path(name)
        if durable.is_file() and durable.stat().st_size == local.stat().st_size:
            return False  # already stored (cache hit round-trip)
        durable.parent.mkdir(parents=True, exist_ok=True)
        tmp = durable.with_name(durable.name + ".tmp")
        try:
            shutil.copy2(local, tmp)
            os.replace(tmp, durable)
            fsync_directory(durable.parent)
        finally:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
        return True


class SpeakerscribeEngine:
    """`TranscriptionEngine` backed by the speakerscribe library on CUDA."""

    def __init__(self, config: TranscriptionConfig) -> None:
        self.config = config
        self._stack: contextlib.ExitStack | None = None
        self._model: Any = None
        self._diarizer: Any = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        from speakerscribe.diarization import DiarizationEngine
        from speakerscribe.transcription import loaded_whisper

        if self.loaded:
            return
        stack = contextlib.ExitStack()
        try:
            diarizer = stack.enter_context(DiarizationEngine(self.config))
            diarizer.load()
            model = stack.enter_context(loaded_whisper(self.config))
        except BaseException:
            stack.close()
            release_memory()
            raise
        self._stack, self._diarizer, self._model = stack, diarizer, model

    def unload(self) -> None:
        stack, self._stack = self._stack, None
        self._model = None
        self._diarizer = None
        if stack is not None:
            stack.close()
        release_memory()

    def transcribe(
        self, media: Path, workdir: Path, diar_cache_in: Path | None, diar_cache_out: Path
    ) -> dict[str, Any]:
        from speakerscribe.config import WorkspacePaths
        from speakerscribe.diarization import diarization_params_hash
        from speakerscribe.pipeline import process_one

        if not self.loaded:
            self.load()
        paths = WorkspacePaths(workspace=str(workdir), scratch=str(workdir / "_scratch"))
        paths.create_directories()
        local_cache = (
            paths.diar_cache / f"{media.stem}_{diarization_params_hash(self.config)}.diar.json"
        )
        if diar_cache_in is not None:
            shutil.copy2(diar_cache_in, local_cache)
        try:
            return process_one(media, paths, self._model, self.config, diar_engine=self._diarizer)
        finally:
            if local_cache.is_file():
                with contextlib.suppress(OSError):
                    shutil.copy2(local_cache, diar_cache_out)


__all__ = ["DiarizationCacheStore", "SpeakerscribeEngine", "TranscriptionEngine"]
