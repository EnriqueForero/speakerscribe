"""Every filesystem path of a batch run, derived once from `BatchSettings`.

DRY: no other module builds a path by concatenating names. The state layout
under ``state`` is byte-compatible with notebook v5, so an existing
``.speakerscribe_state`` (journal, diarization cache, master JSON, renames)
keeps working after the upgrade.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from speakerscribe.batch.fsio import is_within
from speakerscribe.batch.settings import BatchSettings

STATE_DIRNAME = ".speakerscribe_state"
COLAB_DRIVE_PREFIX = "/content/drive/"


class PathLayoutError(ValueError):
    """The configured folders overlap or point at ephemeral storage."""


@dataclass(frozen=True)
class BatchPaths:
    """Resolved paths for one batch run.

    Attributes:
        root: Project folder (durable, usually on Drive).
        input: Input media (scanned recursively).
        deliverables: Canonical ``.txt`` per recording and ``_resumen.md``.
        formats: Optional formats (``.transcript.md``, ``.srt``, ``.json``, ``.plano.txt``).
        llm: LLM-ready text (``.full_for_llm.txt``, ``.parte_NN.txt``).
        processed: Audio already transcribed, kept N days then purged.
        state: Journal and caches (v5-compatible layout).
        scratch: Local disk for high-churn temporaries (never Drive).
    """

    root: Path
    input: Path
    deliverables: Path
    formats: Path
    llm: Path
    processed: Path
    state: Path
    scratch: Path

    @classmethod
    def from_settings(cls, settings: BatchSettings) -> BatchPaths:
        """Derive and validate all paths.

        Raises:
            PathLayoutError: If folders overlap dangerously.
        """
        root = settings.root.expanduser().resolve()

        def pick(value: Path | None, default: Path) -> Path:
            return (value.expanduser() if value is not None else default).resolve()

        deliverables = pick(settings.deliverables_dir, root / "entregables")
        paths = cls(
            root=root,
            input=pick(settings.input_dir, root / "data"),
            deliverables=deliverables,
            formats=pick(settings.formats_dir, root / "transcripts"),
            llm=pick(settings.llm_dir, root / "splits"),
            processed=pick(settings.processed_dir, root / "_procesados"),
            state=pick(settings.state_dir, deliverables / STATE_DIRNAME),
            scratch=pick(settings.scratch_dir, settings.default_scratch()),
        )
        paths.validate_layout()
        return paths

    # ── State files (v5-compatible names) ────────────────────────────
    @property
    def events(self) -> Path:
        return self.state / "events.jsonl"

    @property
    def lock(self) -> Path:
        return self.state / "active.lock.json"

    @property
    def identity(self) -> Path:
        return self.state / "workspace_identity.json"

    @property
    def audit_cursor(self) -> Path:
        return self.state / "audit_cursor.json"

    @property
    def review(self) -> Path:
        return self.state / "pendientes_revision.json"

    @property
    def diar_cache(self) -> Path:
        return self.state / "diar_cache"

    @property
    def masters(self) -> Path:
        return self.state / "intermedios"

    @property
    def renames(self) -> Path:
        return self.state / "renombres"

    @property
    def failure_logs(self) -> Path:
        return self.state / "logs_fallos"

    @property
    def summary(self) -> Path:
        return self.state / "last_summary.json"

    @property
    def telemetry(self) -> Path:
        return self.state / "telemetria.jsonl"

    @property
    def resumen_md(self) -> Path:
        return self.deliverables / "_resumen.md"

    # ── Scratch (local, ephemeral) ───────────────────────────────────
    @property
    def jobs(self) -> Path:
        return self.scratch / "jobs"

    @property
    def staging(self) -> Path:
        return self.scratch / "staging"

    # ── Behavior ─────────────────────────────────────────────────────
    def output_dirs(self) -> tuple[Path, ...]:
        """Folders the batch writes durable results into."""
        return (self.deliverables, self.formats, self.llm, self.processed, self.state)

    def validate_layout(self) -> None:
        """Reject layouts that would make the batch read its own outputs.

        Raises:
            PathLayoutError: If the input contains an output folder (or vice
                versa), or the scratch sits inside a durable folder.
        """
        for out in self.output_dirs():
            if out == self.input or is_within(out, self.input):
                raise PathLayoutError(f"La carpeta de salida {out} está dentro de la entrada")
            if is_within(self.input, out):
                raise PathLayoutError(f"La entrada {self.input} está dentro de {out}")
        for durable in (self.root, *self.output_dirs()):
            if is_within(self.scratch, durable):
                raise PathLayoutError("scratch debe estar en disco local, fuera de Drive")

    def check_durable_on_colab(self, in_colab: bool) -> None:
        """On Colab, durable folders must live under /content/drive.

        Raises:
            PathLayoutError: If a durable folder is on the ephemeral VM disk.
        """
        if not in_colab:
            return
        for path in (self.root, *self.output_dirs()):
            text = str(path)
            if text.startswith("/content/") and not text.startswith(COLAB_DRIVE_PREFIX):
                raise PathLayoutError(
                    f"{path} está en el disco efímero de Colab; use /content/drive/..."
                )

    def ensure(self) -> None:
        """Create every durable and scratch folder (idempotent)."""
        for path in (
            *self.output_dirs(),
            self.diar_cache,
            self.masters,
            self.renames,
            self.failure_logs,
            self.jobs,
            self.staging,
        ):
            path.mkdir(parents=True, exist_ok=True)


__all__ = ["COLAB_DRIVE_PREFIX", "STATE_DIRNAME", "BatchPaths", "PathLayoutError"]
