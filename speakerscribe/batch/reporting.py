"""Human-readable outputs: session report, ``_resumen.md``, review list, census, autopsy.

Everything here is derived from the journal and the session report; nothing
in this module changes batch state.
"""

from __future__ import annotations

import contextlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from speakerscribe.batch.fsio import atomic_write_json, atomic_write_text, utc_now
from speakerscribe.batch.journal import Event
from speakerscribe.batch.renderers import format_ts

MD_ROWS_LIMIT = 400
TEXT_CELL_CHARS = 90


@dataclass
class Row:
    """One line of the per-session table."""

    rel: str
    outcome: str
    duration_s: float = 0.0
    words: int = 0
    detail: str | None = None


@dataclass
class SessionReport:
    """Counters and details of one batch session (persisted as last_summary.json)."""

    run_id: str
    started_utc: str = field(default_factory=utc_now)
    finished_utc: str | None = None
    end_reason: str | None = None
    ok: int = 0
    flagged: int = 0
    reused: int = 0
    republished: int = 0
    failed: int = 0
    deferred_time: int = 0
    retired: int = 0
    audio_s: float = 0.0
    recovered_commits: int = 0
    session_minutes: float = 0.0
    rtf_estimate: float | None = None
    peak_ram_gb: float | None = None
    model_recycles: int = 0
    fatal_error: str | None = None
    plan: dict[str, int] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    rows: list[Row] = field(default_factory=list)
    flagged_items: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    housekeeping: dict[str, Any] = field(default_factory=dict)
    shutdown: dict[str, Any] = field(default_factory=dict)

    @property
    def deliverables(self) -> int:
        """Files with a fresh deliverable this session (GPU or CPU)."""
        return self.ok + self.flagged + self.reused + self.republished

    def count(self, key: str, n: int = 1) -> None:
        self.plan[key] = self.plan.get(key, 0) + n

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["rows"] = data["rows"][-MD_ROWS_LIMIT:]
        return data


def _cell(value: Any) -> str:
    return str(value or "—").replace("|", "/").replace("\n", " ")[:TEXT_CELL_CHARS]


def render_resumen_md(report: SessionReport, context: dict[str, str]) -> str:
    """Markdown summary written to ``<deliverables>/_resumen.md``."""
    plan = report.plan
    lines = [
        f"# Resumen SpeakerScribe — {utc_now()}",
        "",
        f"- **Run:** `{report.run_id[:12]}` · {context.get('producer', '')}",
        f"- **Entrada:** `{context.get('input', '')}`",
        f"- **Entregables:** `{context.get('deliverables', '')}`",
        f"- **Motor:** {context.get('engine', '')}",
        f"- **Sesión:** ✔ {report.ok} OK · 🧯 {report.flagged} marcados · "
        f"♻️ {report.reused} reusados · 📝 {report.republished} republicados (CPU) · "
        f"✖ {report.failed} fallidos · ⏭️ {report.deferred_time} diferidos · "
        f"📦 {report.retired} audios a _procesados",
        f"- **Plan:** {plan.get('ya_listos', 0)} ya listos · {plan.get('invalidos', 0)} inválidos · "
        f"{plan.get('bloqueados', 0)} bloqueados · {plan.get('perfil_cambiado', 0)} con otro perfil · "
        f"{plan.get('salida_cambiada', 0)} con salida movida/editada por usted",
        "",
        "## Archivos tocados en esta sesión",
        "",
        "| Archivo | Resultado | Duración | Palabras | Detalle |",
        "|---|---|---|---|---|",
    ]
    for row in report.rows[-MD_ROWS_LIMIT:]:
        lines.append(
            f"| `{_cell(row.rel)}` | {row.outcome} | {format_ts(row.duration_s)} "
            f"| {row.words:,} | {_cell(row.detail)} |"
        )
    lines += [
        "",
        "_Estado: `.speakerscribe_state/` · revisión: `pendientes_revision.json` · "
        "caja negra: `telemetria.jsonl`_",
    ]
    return "\n".join(lines) + "\n"


def write_session_files(
    report: SessionReport,
    *,
    summary_path: Path,
    review_path: Path,
    resumen_path: Path,
    context: dict[str, str],
) -> None:
    """Persist summary JSON, review list and ``_resumen.md`` (best effort)."""
    with contextlib.suppress(OSError):
        atomic_write_json(summary_path, report.to_dict())
    with contextlib.suppress(OSError):
        atomic_write_json(
            review_path,
            {
                "generado_utc": utc_now(),
                "run_id": report.run_id,
                "publicados_marcados": report.flagged_items,
                "fallidos_esta_sesion": report.errors,
                "bloqueados_por_intentos": report.plan.get("bloqueados", 0),
                "salidas_cambiadas_por_usted": report.plan.get("salida_cambiada", 0),
                "nota": "Los marcados SÍ tienen .txt (vea 'estado:' en su encabezado). Los "
                "bloqueados se liberan con force_reprocess=True o al cambiar el archivo.",
            },
        )
    with contextlib.suppress(OSError):
        atomic_write_text(resumen_path, render_resumen_md(report, context))


def console_summary(report: SessionReport) -> list[str]:
    """Lines printed at the end of a session."""
    lines = [
        "══════════ RESUMEN DE LA SESIÓN ══════════",
        f"  ✔ OK: {report.ok}   🧯 Marcados: {report.flagged}   ♻️ Reusados: {report.reused}"
        f"   📝 Republicados: {report.republished}   ✖ Fallidos: {report.failed}"
        f"   ⏭️ Diferidos: {report.deferred_time}   📦 A _procesados: {report.retired}",
        f"  🎧 Audio confirmado: {format_ts(report.audio_s)} · motivo fin: {report.end_reason}"
        f" · {report.session_minutes:.1f} min",
    ]
    if report.plan.get("salida_cambiada"):
        lines.append(
            f"  ℹ️ {report.plan['salida_cambiada']} audio(s) en data/ cuya salida usted movió o "
            "editó: no se tocan (vea _resumen.md)."
        )
    if report.plan.get("bloqueados"):
        lines.append(
            f"  ⛔ {report.plan['bloqueados']} archivo(s) con intentos agotados "
            "(force_reprocess=True los libera)."
        )
    if report.fatal_error:
        lines.append(f"  🛑 Error: {report.fatal_error[:300]}")
    if report.end_reason == "limite_ram":
        lines.append(
            "  🛡️ Detenido por RAM (no es un fallo). Reinicie la sesión y ejecute de nuevo: "
            "retoma donde quedó."
        )
    if report.peak_ram_gb:
        lines.append(f"  🩺 Pico de RAM: {report.peak_ram_gb:.1f} GB")
    return lines


# ── Autopsy (CPU-only, after a session died) ─────────────────────────
def read_jsonl(path: Path, last: int = 0) -> list[dict[str, Any]]:
    """Read a JSONL file tolerating a truncated last line."""
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            with contextlib.suppress(json.JSONDecodeError):
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows[-last:] if last else rows


def autopsy(summary_path: Path, telemetry_path: Path, events_path: Path) -> list[str]:
    """Explain how the previous session ended: orderly close, OOM or external kill."""
    out = ["═" * 70, "AUTOPSIA DE LA ÚLTIMA SESIÓN", "═" * 70]
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            summary = {}
        out.append(
            f"🧾 Último resumen: motivo_fin = {summary.get('end_reason') or summary.get('motivo_fin')}"
            f" · {summary.get('session_minutes') or summary.get('minutos_sesion')} min"
        )
        if summary.get("fatal_error") or summary.get("error_fatal"):
            out.append(f"   🛑 error: {summary.get('fatal_error') or summary.get('error_fatal')}")
    else:
        out.append("🧾 No hay last_summary.json: la sesión NUNCA llegó al cierre ordenado.")

    samples = read_jsonl(telemetry_path)
    if not samples:
        out.append("🩺 Sin telemetría (active resource_monitor=True).")
    else:
        last = samples[-1]
        total = float(last.get("ram_total_gb") or 0)
        peak = max((float(t.get("ram_usada_gb") or 0) for t in samples), default=0.0)
        out.append(
            f"🩺 {len(samples)} muestras · última a los {last.get('min_sesion')} min "
            f"(motivo: {last.get('motivo')}) · RAM pico {peak:.1f}/{total} GB"
        )
        out.append(f"   Estaba en: {last.get('archivo') or '—'} · etapa {last.get('etapa') or '—'}")
        if total and peak / total > 0.90:
            out.append("   ⇒ VEREDICTO: la RAM tocó el techo ⇒ el proceso se quedó sin memoria.")
        elif str(last.get("motivo", "")).startswith("fin:"):
            out.append(f"   ⇒ VEREDICTO: cierre ordenado ({last.get('motivo')}).")
        elif last.get("motivo") == "periodico":
            out.append(
                "   ⇒ VEREDICTO: la telemetría se cortó en seco con RAM holgada ⇒ corte "
                "EXTERNO (cuota de Colab, desconexión del navegador)."
            )

    events = read_jsonl(events_path)
    if events:
        kinds = Counter(str(e.get("event")) for e in events)
        out.append("📊 Eventos acumulados:")
        out += [f"   {name:<30} {n:>6}" for name, n in kinds.most_common(14)]
        causes = Counter(
            str(e.get("error"))[:90]
            for e in events
            if e.get("event")
            in (
                Event.QUALITY_REJECTED.value,
                Event.FAILED_RETRYABLE.value,
                Event.FAILED_ENVIRONMENT.value,
            )
        )
        if causes:
            out.append("🔍 Causas de fallo más frecuentes:")
            out += [f"   {n:>4}× {cause}" for cause, n in causes.most_common(8)]
        rtfs = sorted(
            float(e["real_time_factor"])
            for e in events
            if e.get("event") == Event.JOB_METRICS.value
            and isinstance(e.get("real_time_factor"), int | float)
        )
        if rtfs:
            median = rtfs[len(rtfs) // 2]
            out.append(
                f"⚡ RTF: n={len(rtfs)} · p10={rtfs[len(rtfs) // 10]:.1f}x · mediana={median:.1f}x"
                f" ⇒ 1 h de audio ≈ {60 / median:.1f} min de ASR"
            )
    out.append("═" * 70)
    return out


__all__ = [
    "Row",
    "SessionReport",
    "autopsy",
    "console_summary",
    "read_jsonl",
    "render_resumen_md",
    "write_session_files",
]
