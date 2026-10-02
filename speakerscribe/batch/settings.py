"""Batch configuration: one validated, immutable object (single source of truth).

Replaces the ~70 loose globals and ~25 ``assert`` statements of notebook v5.
Every value that changes a business result is a field here (User Control >
Automation); validation happens at construction time (Fail Fast), before any
file is touched or any model is loaded.

Folder layout derived from ``root`` (each folder is overridable)::

    <root>/
    ├── data/                     input media (recursive); emptied after success
    ├── entregables/              canonical .txt per recording + _resumen.md
    │   └── .speakerscribe_state/ journal, caches, master JSON (do not edit)
    ├── transcripts/              optional formats (.transcript.md .srt .json)
    ├── splits/                   LLM-ready text (.full_for_llm.txt, parts)
    └── _procesados/YYYY-MM-DD/   audio already transcribed, purged after N days
"""

from __future__ import annotations

import hashlib
import re
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

WhisperModelName = Literal["large-v3", "large-v3-turbo", "medium", "small", "base", "tiny"]

NAME_TEMPLATE_MARKERS = frozenset({"stem", "carpeta", "modelo", "fecha"})
"""Placeholders allowed in `name_template`."""

_TEMPLATE_FIELD = re.compile(r"{(\w+)}")


class BatchSettings(BaseModel):
    """Every parameter of a batch run. Construct it once per run.

    Only ``root`` is required. Paths left as None are derived from ``root``
    by `BatchPaths` (see module docstring).

    Raises:
        pydantic.ValidationError: On any invalid value or inconsistent
            combination (unknown field names included).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # ── 1 · Rutas ────────────────────────────────────────────────────
    root: Path = Field(description="Raíz del proyecto en Drive (carpeta de resultados).")
    input_dir: Path | None = Field(None, description="Entrada. Por defecto <root>/data.")
    deliverables_dir: Path | None = Field(None, description="Por defecto <root>/entregables.")
    formats_dir: Path | None = Field(None, description="Por defecto <root>/transcripts.")
    llm_dir: Path | None = Field(None, description="Por defecto <root>/splits.")
    processed_dir: Path | None = Field(None, description="Por defecto <root>/_procesados.")
    state_dir: Path | None = Field(
        None, description="Por defecto <entregables>/.speakerscribe_state (compatible con v5)."
    )
    scratch_dir: Path | None = Field(
        None, description="Disco LOCAL para temporales. Por defecto /content/ss_batch/<hash>."
    )

    # ── 2 · Motor de transcripción ───────────────────────────────────
    model: WhisperModelName = "large-v3"
    language: str | None = Field("es", description="'' o None = autodetección por archivo.")
    beam_size: int = Field(5, ge=1, le=20)
    batch_size: Literal[1, 2, 4, 8, 16] = 8
    vad_min_silence_ms: int = Field(1000, ge=100)
    anti_hallucination: bool = Field(
        True, description="condition_on_previous_text=False: corta bucles de repetición."
    )
    glossary: str = Field("", description="Nombres propios/jerga; lo más importante al final.")

    # ── 3 · Diarización y hablantes ──────────────────────────────────
    num_speakers: int = Field(0, ge=0, description="0 = desconocido.")
    min_speakers: int = Field(1, ge=0)
    max_speakers: int = Field(20, ge=0)
    repair_orphans: bool = True
    orphan_tolerance_s: float = Field(2.0, ge=0)
    max_unlabeled_fraction: float = Field(0.35, ge=0, le=1)
    accept_silent_audio: bool = True
    monotonic_tolerance_s: float = Field(
        1.0,
        ge=0,
        description="Retroceso de tiempo tolerado entre segmentos (deriva del alineamiento).",
    )

    # ── 4 · Descubrimiento ───────────────────────────────────────────
    recursive: bool = True
    include_glob: str = Field(
        "",
        description="Solo rutas que coincidan (p. ej. '*Tres ejes*' para una prueba). '' = todas.",
    )
    only_extensions: str = ""
    extra_extensions: str = ""
    exclude_extensions: str = ""
    probe_unknown_extensions: bool = True
    max_unknown_probes: int = Field(5000, ge=0)
    follow_symlinks: bool = False
    stability_seconds: int = Field(15, ge=0)

    # ── 5 · Entregables ──────────────────────────────────────────────
    deliver_markdown: bool = False
    deliver_srt: bool = False
    deliver_json: bool = False
    deliver_full_llm: bool = True
    deliver_splits: bool = False
    deliver_plain: bool = False
    mirror_subfolders: bool = True
    name_template: str = "{stem}"
    include_header: bool = True
    timestamp_ms: bool = False
    md_fillers: Literal["off", "safe", "aggressive"] = "safe"
    md_gap_s: float = Field(3.0, ge=0)
    split_words: int = Field(1950, ge=100)
    keep_master_json: bool = True

    # ── 6 · Identidad, reintentos y calidad ──────────────────────────
    hash_mode: Literal["full", "fast"] = "full"
    verify_output_hash: bool = True
    audits_per_session: int = Field(12, ge=0)
    max_audit_gb: float = Field(15.0, gt=0)
    profile_change_policy: Literal["keep_and_report", "reprocess", "stop"] = "keep_and_report"
    on_output_changed: Literal["report", "reprocess"] = Field(
        "report",
        description=(
            "Si usted movió, renombró o editó un .txt ya confirmado: 'report' lo respeta "
            "y lo informa; 'reprocess' vuelve a transcribir y lo regenera."
        ),
    )
    force_reprocess: bool = False
    max_attempts: int = Field(2, ge=1)
    reject_critical_quality: bool = True
    publish_degraded_after_attempts: bool = True
    publish_if_quality_wont_improve: bool = True

    # ── 7 · Retención ────────────────────────────────────────────────
    after_success: Literal["move_to_processed", "keep"] = Field(
        "move_to_processed",
        description="Qué hacer con el audio tras una transcripción OK verificada.",
    )
    processed_retention_days: int = Field(
        30, ge=0, description="Días en _procesados antes de borrar (0 = en la siguiente corrida)."
    )
    diar_cache_retention_days: int = Field(90, ge=0, description="0 = no podar.")

    # ── 8 · Sesión, recursos y apagado ───────────────────────────────
    max_session_minutes: int = Field(650, ge=0, description="0 = sin límite.")
    close_margin_minutes: int = Field(20, ge=0)
    max_files_per_session: int = Field(0, ge=0, description="0 = sin límite.")
    rtf_asr_floor: float = Field(8.0, gt=0)
    rtf_diar_floor: float = Field(10.0, gt=0)
    per_file_overhead_min: int = Field(5, ge=0)
    ram_pct_recycle: int = Field(70, gt=0, lt=100)
    ram_pct_stop: int = Field(88, gt=0, le=95)
    recycle_every_n_files: int = Field(0, ge=0)
    resource_monitor: bool = True
    heartbeat_minutes: int = Field(10, ge=0)
    breaker_threshold: int = Field(
        2, ge=1, description="Errores de entorno idénticos seguidos que detienen el lote."
    )
    shutdown_at_end: bool = True
    shutdown_on_fatal: bool = False
    shutdown_only_if_work: bool = True
    shutdown_delay_s: int = Field(180, ge=0)
    force_take_lock: bool = False
    save_failure_logs: bool = True

    # ── Validation ───────────────────────────────────────────────────
    @field_validator("language")
    @classmethod
    def _normalize_language(cls, value: str | None) -> str | None:
        value = (value or "").strip().lower()
        return value or None

    @field_validator("name_template")
    @classmethod
    def _check_template(cls, value: str) -> str:
        markers = set(_TEMPLATE_FIELD.findall(value))
        if "stem" not in markers or not markers <= NAME_TEMPLATE_MARKERS:
            raise ValueError(
                "name_template debe incluir {stem}; marcadores válidos: "
                + " ".join("{" + m + "}" for m in sorted(NAME_TEMPLATE_MARKERS))
            )
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> BatchSettings:
        if self.ram_pct_recycle >= self.ram_pct_stop:
            raise ValueError("ram_pct_recycle debe ser menor que ram_pct_stop")
        if self.max_session_minutes and self.close_margin_minutes >= self.max_session_minutes:
            raise ValueError("close_margin_minutes debe ser menor que max_session_minutes")
        if (
            self.num_speakers == 0
            and self.min_speakers
            and self.max_speakers
            and self.min_speakers > self.max_speakers
        ):
            raise ValueError("min_speakers no puede superar max_speakers")
        return self

    # ── Derived values (pure) ────────────────────────────────────────
    def none_if_zero(self, value: int) -> int | None:
        """Map the notebook convention 0 = 'unset' to None."""
        return None if value <= 0 else value

    def default_scratch(self) -> Path:
        """Local scratch: /content on Colab, else the system temp dir."""
        base = Path("/content") if Path("/content").is_dir() else Path(tempfile.gettempdir())
        tag = hashlib.sha256(str(self.root).encode()).hexdigest()[:12]
        return base / "ss_batch" / tag


__all__ = ["NAME_TEMPLATE_MARKERS", "BatchSettings", "WhisperModelName"]
