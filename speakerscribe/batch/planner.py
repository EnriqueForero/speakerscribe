"""Per-source decisions: what (if anything) each discovered file needs.

The planner only DECIDES (and journals the facts it observes: probe
results, content changes). Executing a decision — re-rendering, GPU work,
retiring the audio — belongs to the runner (Single Responsibility), which
keeps every rule below unit-testable without models.

Decision order for one source (cheapest evidence first):

1. Modified seconds ago -> postpone (still being uploaded).
2. Confirmed by stat (fast path, no audio read): ready, re-render on CPU,
   or apply the profile-change policy.
3. Output of a confirmed source missing/renamed/edited by the user ->
   report and leave it alone (``on_output_changed="report"``). v5 silently
   re-ran the GPU and overwrote the user's file.
4. Known-invalid media with the same stat -> skip.
5. ffprobe -> invalid (journal once) / transient (retry next run).
6. Stable signature (+ local staging in full mode).
7. Same content already transcribed elsewhere -> reuse its master on CPU.
8. Attempts exhausted -> blocked until the file or settings change.
9. Otherwise -> GPU job.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from speakerscribe.batch.discovery import (
    Discovery,
    ProbeResult,
    SourceChangingError,
    StagedSource,
    nfc,
    probe_media,
    stage_source,
)
from speakerscribe.batch.errors import sanitize_error
from speakerscribe.batch.identity import SourceInfo, source_payload, source_stat_matches
from speakerscribe.batch.journal import Event, Journal, JournalIndex
from speakerscribe.batch.layout import DeliverableLayout, OutputNamer, disambiguate
from speakerscribe.batch.profiles import motor_profile, prompt_digest
from speakerscribe.batch.publisher import Job, OutputRecords, Publisher, job_id_for
from speakerscribe.batch.settings import BatchSettings

COLLISION_LADDER_STEPS = 5


class Action(str, Enum):
    READY = "ready"
    REPUBLISH = "republish"
    REUSE = "reuse"
    PROCESS = "process"
    SKIP = "skip"


class ProfileChangeStopError(RuntimeError):
    """``profile_change_policy="stop"`` and the motor profile changed."""


@dataclass
class Decision:
    """What to do with one source, and why (``counter`` feeds the report)."""

    action: Action
    source: SourceInfo
    counter: str
    detail: str | None = None
    prior: dict[str, Any] | None = None
    donor: dict[str, Any] | None = None
    job: Job | None = None
    staged: StagedSource | None = None


@dataclass
class PlanState:
    """Built once per session from discovery + journal."""

    output_map: dict[str, Path]
    ordered: list[SourceInfo]
    strict_ids: set[str]
    audit_gb: float = 0.0
    totals: dict[str, int] = field(default_factory=dict)


def _key(path: Path) -> str:
    return nfc(path.as_posix()).casefold()


class Planner:
    """Decides the action for each discovered source.

    Args:
        settings: Batch settings.
        journal: Event journal (facts observed while planning are recorded).
        index: Journal indices (updated live by the runner after commits).
        records: Output-record validator.
        publisher: Used read-only here (presentation ids, deliverables present).
        layout: Deliverable locations.
        namer: Canonical-name policy for new sources.
        config: Engine configuration (for the motor profile).
        input_root: Input folder.
        staging_dir: Local folder for staged copies (None = read from Drive).
        probe: ffprobe classifier (injected in tests).
        stage: Signature/staging function (injected in tests).
        clock: Wall clock (stability window).
    """

    def __init__(
        self,
        *,
        settings: BatchSettings,
        journal: Journal,
        index: JournalIndex,
        records: OutputRecords,
        publisher: Publisher,
        layout: DeliverableLayout,
        namer: OutputNamer,
        config: Any,
        input_root: Path,
        staging_dir: Path | None,
        audit_cursor: Path,
        probe: Callable[[Path], ProbeResult] = probe_media,
        stage: Callable[..., StagedSource] = stage_source,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.s = settings
        self.journal = journal
        self.index = index
        self.records = records
        self.publisher = publisher
        self.layout = layout
        self.namer = namer
        self.config = config
        self.input_root = input_root
        self.staging_dir = staging_dir
        self.audit_cursor = audit_cursor
        self.probe = probe
        self.stage = stage
        self.clock = clock

    # ── Session-level plan ───────────────────────────────────────────
    def prepare(self, discovery: Discovery) -> PlanState:
        """Stable output names, processing order and strict-audit schedule."""
        current = {s.source_id for s in discovery.sources}
        reserved: dict[str, str] = {}
        output_map: dict[str, Path] = {}
        # 1) Names already published are kept (even if the template changed).
        for sid, records in sorted(self.index.ok_by_source.items()):
            prior = records[-1] if records else None
            rel = ((prior or {}).get("output") or {}).get("relative_path")
            if not rel:
                continue
            k = _key(Path(rel))
            if reserved.setdefault(k, sid) == sid and sid in current:
                output_map[sid] = Path(rel)
        # 2) New sources: template + collision ladder + anti-clobber.
        for source in discovery.sources:
            if source.source_id in output_map:
                continue
            base = self.namer.relative_for(source, discovery.stem_collisions)
            candidate = base
            for attempt in range(COLLISION_LADDER_STEPS + 1):
                k = _key(candidate)
                taken = k in reserved and reserved[k] != source.source_id
                clobber = not taken and k not in reserved and self._exists(candidate)
                if not taken and not clobber:
                    break
                if attempt == COLLISION_LADDER_STEPS:
                    raise RuntimeError(f"Sin nombre de salida único para {source.relative_posix}")
                candidate = disambiguate(base, source.source_id, attempt)
            reserved[_key(candidate)] = source.source_id
            output_map[source.source_id] = candidate

        def confirmed(source: SourceInfo) -> bool:
            prior = self.index.latest_success(source.source_id)
            return bool(
                prior and source_stat_matches(prior, source) and self.records.stat_valid(prior)
            )

        changed = [s for s in discovery.sources if not confirmed(s)]
        stable = [s for s in discovery.sources if confirmed(s)]
        audits, audit_bytes = self._audit_schedule(stable)
        strict_ids = {s.source_id for s in audits}
        rest = [s for s in stable if s.source_id not in strict_ids]
        return PlanState(
            output_map=output_map,
            ordered=changed + audits + rest,
            strict_ids=strict_ids,
            audit_gb=round(audit_bytes / 1e9, 2),
        )

    def _exists(self, relative: Path) -> bool:
        try:
            return self.layout.canonical(relative).exists()
        except (OSError, RuntimeError):
            return True  # unsafe or unreadable: never write there

    def _audit_schedule(self, stable: list[SourceInfo]) -> tuple[list[SourceInfo], int]:
        """Rotating strict re-hash of confirmed sources (catches same-stat swaps)."""
        if not stable or self.s.audits_per_session <= 0:
            return [], 0
        cursor: dict[str, Any] = {}
        with contextlib.suppress(OSError, ValueError):
            cursor = json.loads(self.audit_cursor.read_text(encoding="utf-8"))
        ids = [s.source_id for s in stable]
        last = cursor.get("last_source_id")
        start = (ids.index(last) + 1) % len(ids) if last in ids else 0
        rotated = stable[start:] + stable[:start]
        picked: list[SourceInfo] = []
        total = 0
        cap = int(self.s.max_audit_gb * 1e9)
        for source in rotated:
            if len(picked) >= self.s.audits_per_session or (
                picked and total + source.size_bytes > cap
            ):
                break
            picked.append(source)
            total += source.size_bytes
        return picked, total

    # ── Per-source decision ──────────────────────────────────────────
    def _payload(self, info: SourceInfo, signature: str | None = None) -> dict[str, Any]:
        return source_payload(info, self.input_root, signature, self.s.hash_mode)

    def _skip(
        self, source: SourceInfo, counter: str, detail: str | None = None, **kw: Any
    ) -> Decision:
        return Decision(Action.SKIP, source, counter, detail, **kw)

    def _up_to_date(self, prior: dict[str, Any], source_id: str) -> bool:
        presentation = self.publisher.presentation(source_id)
        rel = ((prior.get("output") or {}).get("relative_path")) or ""
        return (prior.get("presentacion") or {}).get("id") == presentation["id"] and (
            self.publisher.extras_present(Path(rel))
        )

    def _ready_or_republish(self, source: SourceInfo, prior: dict[str, Any]) -> Decision:
        if self._up_to_date(prior, source.source_id):
            return Decision(Action.READY, source, "ya_listos", prior=prior)
        return Decision(Action.REPUBLISH, source, "republicados", prior=prior)

    def _profile_policy(
        self, source: SourceInfo, prior: dict[str, Any], profile: dict[str, Any]
    ) -> Decision | None:
        """None means: reprocess with the new profile."""
        policy = self.s.profile_change_policy
        if policy == "stop":
            raise ProfileChangeStopError(
                f"Cambió el perfil de MOTOR para {source.relative_posix}. Elija "
                "profile_change_policy='keep_and_report' o 'reprocess'."
            )
        if policy == "keep_and_report":
            self.journal.append(
                Event.PROFILE_CHANGE_DEFERRED,
                source=prior.get("source"),
                previous_profile=prior.get("profile"),
                requested_profile=profile,
                output=prior.get("output"),
            )
            return Decision(Action.READY, source, "perfil_cambiado", prior=prior)
        return None

    def _output_changed(self, source: SourceInfo, prior: dict[str, Any]) -> Decision | None:
        """The user moved, renamed or edited a confirmed deliverable."""
        if self.s.force_reprocess or self.s.on_output_changed == "reprocess":
            return None
        path = self.records.path(prior)
        state = "editada" if path is not None and path.exists() else "movida o renombrada"
        rel = (prior.get("output") or {}).get("relative_path")
        return self._skip(
            source,
            "salida_cambiada",
            f"la salida {rel} fue {state} por usted; no se regenera",
            prior=prior,
        )

    def decide(self, source: SourceInfo, state: PlanState) -> Decision:
        """Decide the action for one source.

        Raises:
            ProfileChangeStopError: Under ``profile_change_policy="stop"``.
        """
        s = self.s
        try:
            age = self.clock() - source.path.stat().st_mtime
        except OSError as e:
            self.journal.append(
                Event.PROBE_RETRYABLE,
                source=self._payload(source),
                error=sanitize_error(e),
                stage="stat",
            )
            return self._skip(source, "probe_reintentable", sanitize_error(e))
        if age < s.stability_seconds:
            self.journal.append(
                Event.SOURCE_CHANGED,
                source=self._payload(source),
                error=f"modificado hace {age:.1f} s",
            )
            return self._skip(source, "pospuestos", "archivo modificado hace segundos")

        sid = source.source_id
        strict = sid in state.strict_ids or sid in self.index.dirty

        # ── Fast path: confirmed by stat, no audio read ─────────────
        if not s.force_reprocess and not strict:
            prior = self.index.latest_success(sid)
            if prior and source_stat_matches(prior, source):
                try:
                    profile = motor_profile(self.config, prompt_digest(source, s.glossary))
                except OSError as e:
                    self.journal.append(
                        Event.PROBE_RETRYABLE,
                        source=self._payload(source),
                        error=sanitize_error(e),
                        stage="prompt",
                    )
                    return self._skip(source, "probe_reintentable", sanitize_error(e))
                if self.records.valid(prior):
                    if (prior.get("profile") or {}).get("id") == profile["id"]:
                        return self._ready_or_republish(source, prior)
                    decided = self._profile_policy(source, prior, profile)
                    if decided is not None:
                        return decided
                else:
                    changed = self._output_changed(source, prior)
                    if changed is not None:
                        return changed

        invalid = self.index.invalid.get(sid)
        if not s.force_reprocess and invalid and source_stat_matches(invalid, source):
            return self._skip(source, "invalidos", str(invalid.get("error") or "sin audio"))

        probe = self.probe(source.path)
        if probe.status != "ok":
            event = Event.INVALID_MEDIA if probe.status == "invalid" else Event.PROBE_RETRYABLE
            self.journal.append(
                event,
                source=self._payload(source),
                error=probe.detail or "ffprobe",
                stage="ffprobe",
            )
            counter = "invalidos" if probe.status == "invalid" else "probe_reintentable"
            return self._skip(source, counter, probe.detail)

        try:
            staged = self.stage(
                source,
                hash_mode=s.hash_mode,
                stability_seconds=s.stability_seconds,
                staging_dir=self.staging_dir,
                staged_name=f"src_{sid[:20]}{_safe_suffix(source.path)}",
            )
            info = staged.info
            profile = motor_profile(self.config, prompt_digest(info, s.glossary))
        except SourceChangingError as e:
            self.journal.append(
                Event.SOURCE_CHANGED, source=self._payload(source), error=sanitize_error(e)
            )
            return self._skip(source, "pospuestos", sanitize_error(e))
        except OSError as e:
            self.journal.append(
                Event.PROBE_RETRYABLE,
                source=self._payload(source),
                error=sanitize_error(e),
                stage="hash",
            )
            return self._skip(source, "probe_reintentable", sanitize_error(e))

        signature = staged.signature
        output_rel = state.output_map[sid]
        job = Job(
            source=info,
            signature=signature,
            profile=profile,
            job_id=job_id_for(signature, profile["id"], output_rel),
            output_relative=output_rel,
            duration_seconds=probe.duration_s,
            staged_path=staged.staged_path,
        )

        if not s.force_reprocess:
            records = self.index.ok_by_source.get(sid, [])
            prior = self.records.latest_valid(records)
            if prior:
                prior_sig = (prior.get("source") or {}).get("content_signature")
                prior_profile = (prior.get("profile") or {}).get("id")
                if prior_sig and prior_sig != signature:
                    self.journal.append(
                        Event.CONTENT_CHANGE_DETECTED,
                        job_id=job.job_id,
                        source=self._payload(info, signature),
                        previous_content_signature=prior_sig,
                        profile=profile,
                    )
                    self.index.dirty.add(sid)
                elif prior_sig == signature:
                    if sid in self.index.dirty:
                        self.journal.append(
                            Event.CONTENT_CHANGE_RESOLVED,
                            source=self._payload(info, signature),
                            profile=profile,
                        )
                        self.index.dirty.discard(sid)
                    if prior_profile == profile["id"]:
                        decided = self._ready_or_republish(info, prior)
                        decided.staged = staged
                        return decided
                    decided = self._profile_policy(info, prior, profile)
                    if decided is not None:
                        decided.staged = staged
                        return decided
            else:
                latest = self.index.latest_success(sid)
                if latest and (latest.get("source") or {}).get("content_signature") == signature:
                    changed = self._output_changed(info, latest)
                    if changed is not None:
                        changed.staged = staged
                        return changed

            donor = self.records.latest_valid(
                self.index.ok_by_content_profile.get((signature, profile["id"]), [])
            )
            if donor and (donor.get("source") or {}).get("id") != sid:
                return Decision(Action.REUSE, info, "reusados", donor=donor, job=job, staged=staged)

        if not s.force_reprocess and self.index.attempts.get(job.job_id, 0) >= s.max_attempts:
            return self._skip(
                info, "bloqueados", f"{s.max_attempts} intentos agotados", job=job, staged=staged
            )
        return Decision(Action.PROCESS, info, "procesar", job=job, staged=staged)


def _safe_suffix(path: Path) -> str:
    suffix = path.suffix.lower()
    if 2 <= len(suffix) <= 13 and suffix[1:].isalnum():
        return suffix
    return ".media"


__all__ = ["Action", "Decision", "PlanState", "Planner", "ProfileChangeStopError"]
