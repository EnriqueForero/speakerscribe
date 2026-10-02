"""Pipeline stage: persist the run record to SQLite (when enabled)."""

from __future__ import annotations

import json

from speakerscribe.pipeline.context import RunContext


def register_if_enabled(ctx: RunContext, status: str = "ok") -> None:
    """Write a row to `_runs.db` when `config.enable_runs_db=True`.

    The quality flags are serialized as a JSON string for the DB column. The
    `file_hash` must already be populated on the context by the idempotency
    stage; when DB is disabled this is a no-op.
    """
    if not ctx.config.enable_runs_db:
        return
    from speakerscribe.persistence import register_run

    quality_flags_json = (
        json.dumps([str(f) for f in ctx.quality_report.flags], ensure_ascii=False)
        if ctx.quality_report
        else None
    )
    register_run(
        ctx.paths.db_path,
        ctx.metadata,
        file_hash=ctx.file_hash or "",
        quality_ok=ctx.quality_report.quality_ok if ctx.quality_report else None,
        quality_flags_json=quality_flags_json,
        status=status,
    )
