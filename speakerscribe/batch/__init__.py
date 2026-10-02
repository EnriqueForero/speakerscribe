"""Resumable, idempotent batch transcription for Google Drive + Colab.

Typical use (one configuration object, one call)::

    from speakerscribe.batch import BatchSettings, run_batch

    settings = BatchSettings(root="/content/drive/MyDrive/…/Transcripcion-Diarizacion")
    report = run_batch(settings)

Operator tools (no GPU): `status`, `published`, `rename_speakers`,
`rebind_workspace`, `autopsy`.
"""

from speakerscribe.batch.paths import BatchPaths, PathLayoutError
from speakerscribe.batch.reporting import SessionReport
from speakerscribe.batch.runner import BatchRunner, RunnerDeps, run_batch
from speakerscribe.batch.settings import BatchSettings
from speakerscribe.batch.tools import (
    Census,
    autopsy,
    published,
    rebind_workspace,
    rename_speakers,
    status,
)

__all__ = [
    "BatchPaths",
    "BatchRunner",
    "BatchSettings",
    "Census",
    "PathLayoutError",
    "RunnerDeps",
    "SessionReport",
    "autopsy",
    "published",
    "rebind_workspace",
    "rename_speakers",
    "run_batch",
    "status",
]
