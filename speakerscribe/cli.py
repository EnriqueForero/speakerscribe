"""speakerscribe CLI built on typer + rich.

Commands:
    process        Process every media file in <workspace>/data/
    smoke-test     Run a smoke test with the 'small' model on the first file
    inspect        Print a quick summary of a transcription JSON file
    stats          Show aggregate statistics from the runs ledger
    list-runs      List the most recent N runs
    bench          Compute WER/DER against user references (extras: [bench])
    rename         Replace SPEAKER_XX labels with real names in outputs
    clean          Delete outputs (selectively or all)
    version        Show package version
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.table import Table

from speakerscribe import __version__
from speakerscribe.config import TranscriptionConfig, WorkspacePaths
from speakerscribe.logging_config import configure_logging, logger
from speakerscribe.maintenance import (
    delete_all_outputs,
    delete_outputs_for,
    inspect_json,
    rename_speakers_in_outputs,
)
from speakerscribe.persistence import (
    global_stats,
    list_runs,
)
from speakerscribe.pipeline import process_batch

app = typer.Typer(
    name="speakerscribe",
    help="Speech-to-text with speaker diarization (Whisper + pyannote).",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"speakerscribe v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-V",
            callback=_version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable DEBUG logging.")] = False,
) -> None:
    """speakerscribe — command line interface."""
    level: Literal["DEBUG", "INFO"] = "DEBUG" if verbose else "INFO"
    configure_logging(console_level=level)


@app.command()
def process(
    workspace: Annotated[Path, typer.Option("--workspace", "-w", help="Project root path.")],
    model: Annotated[
        str, typer.Option("--model", "-m", help="Whisper model name.")
    ] = "large-v3-turbo",
    language: Annotated[str | None, typer.Option("--language", "-l")] = None,
    beam_size: Annotated[int, typer.Option("--beam-size", "-b", min=1, max=10)] = 5,
    no_diar: Annotated[bool, typer.Option("--no-diar", help="Disable diarization.")] = False,
    num_speakers: Annotated[int | None, typer.Option("--num-speakers", "-n")] = None,
    min_speakers: Annotated[int | None, typer.Option("--min-speakers")] = None,
    max_speakers: Annotated[int | None, typer.Option("--max-speakers")] = None,
    word_timestamps: Annotated[
        bool, typer.Option("--word-timestamps/--no-word-timestamps")
    ] = False,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", min=1, max=32, help="Batched inference size (1=sequential)."),
    ] = 8,
    speaker_assignment: Annotated[
        str,
        typer.Option(
            "--speaker-assignment",
            help="'word' (re-segment at speaker changes) or 'segment' (legacy).",
        ),
    ] = "word",
    remove_fillers: Annotated[
        str,
        typer.Option(
            "--remove-fillers", help="Filler filter for .transcript.md: off|safe|aggressive."
        ),
    ] = "safe",
    hash_mode: Annotated[
        str,
        typer.Option(
            "--hash-mode", help="Idempotency signature: fast (16 MB sample) | full (SHA-256)."
        ),
    ] = "fast",
    auto_retry: Annotated[
        bool,
        typer.Option(
            "--auto-retry/--no-auto-retry",
            help="Retry once with anti-loop decoding on critical hallucination flags.",
        ),
    ] = True,
    enable_db: Annotated[
        bool, typer.Option("--enable-db/--no-db", help="Enable the runs ledger (idempotency).")
    ] = True,
    force: Annotated[bool, typer.Option("--force/--no-force")] = False,
    config_file: Annotated[
        Path | None,
        typer.Option("--config-file", "-c", help="Path to a JSON config file."),
    ] = None,
) -> None:
    """Process every media file under <workspace>/data/."""
    configure_logging(workspace=workspace)

    if config_file:
        if not config_file.exists():
            console.print(f"[red]Config file not found:[/red] {config_file}")
            raise typer.Exit(1)
        config = TranscriptionConfig.model_validate_json(config_file.read_text())
        logger.info(f"Config loaded from {config_file}")
    else:
        config = TranscriptionConfig(
            model=model,  # type: ignore[arg-type]
            language=language,
            beam_size=beam_size,
            enable_diarization=not no_diar,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            word_timestamps=word_timestamps,
            batch_size=batch_size,
            speaker_assignment=speaker_assignment,  # type: ignore[arg-type]
            remove_fillers=remove_fillers,  # type: ignore[arg-type]
            hash_mode=hash_mode,  # type: ignore[arg-type]
            auto_retry_on_critical=auto_retry,
            enable_runs_db=enable_db,
            force_reprocess=force,
        )

    paths = WorkspacePaths(workspace=str(workspace))
    results = process_batch(paths, config)

    n_ok = sum(1 for r in results if r.get("status") in ("ok", "ok_degraded", "skipped"))
    if n_ok == 0:
        raise typer.Exit(1)


@app.command(name="smoke-test")
def smoke_test_cmd(
    workspace: Annotated[Path, typer.Option("--workspace", "-w")],
) -> None:
    """Run a fast smoke test using the 'small' model on the first media file.

    Outputs go to an isolated <workspace>/_smoke sub-workspace so they never
    mix with (or skip-shadow) real production outputs.
    """
    configure_logging(workspace=workspace)

    config = TranscriptionConfig(
        model="small",
        beam_size=1,
        enable_diarization=True,
        force_reprocess=True,
        enable_runs_db=False,
        evaluate_quality=False,
    )
    paths = WorkspacePaths(workspace=str(workspace))
    media = paths.list_media_files()
    if not media:
        console.print("[red]No media files in data/[/red]")
        raise typer.Exit(1)

    from speakerscribe.pipeline import process_one
    from speakerscribe.transcription import loaded_whisper

    smoke_paths = WorkspacePaths(workspace=str(Path(workspace) / "_smoke"))
    smoke_paths.create_directories()
    console.print(f"Smoke outputs -> {smoke_paths.base}")
    with loaded_whisper(config) as model:
        result = process_one(media[0], smoke_paths, model, config)
    if result.get("status") in ("ok", "ok_degraded"):
        console.print(f"[green]SMOKE TEST OK[/green] (status={result.get('status')})")
    else:
        console.print(f"[yellow]Smoke test status={result.get('status')}[/yellow]")
        raise typer.Exit(1)


@app.command()
def inspect(
    json_path: Annotated[Path, typer.Argument(help="Path to a transcription .json file.")],
) -> None:
    """Print a quick summary of a transcription JSON file."""
    info = inspect_json(json_path)
    if not info:
        raise typer.Exit(1)


@app.command()
def stats(
    workspace: Annotated[Path, typer.Option("--workspace", "-w")],
) -> None:
    """Show aggregate statistics from the runs database (if present)."""
    paths = WorkspacePaths(workspace=str(workspace))
    s = global_stats(paths.ledger_path)

    table = Table(title=f"Statistics — {workspace}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    for k, v in s.items():
        table.add_row(k, str(v if v is not None else "-"))
    console.print(table)


@app.command(name="list-runs")
def list_runs_cmd(
    workspace: Annotated[Path, typer.Option("--workspace", "-w")],
    limit: Annotated[int, typer.Option("--limit", "-n")] = 20,
    status: Annotated[str | None, typer.Option("--status", "-s")] = None,
) -> None:
    """List the most recent N runs from the database."""
    paths = WorkspacePaths(workspace=str(workspace))
    runs = list_runs(paths.ledger_path, limit=limit, status=status)
    if not runs:
        console.print("[yellow]No runs recorded.[/yellow]")
        return

    table = Table(title=f"Last {len(runs)} runs")
    table.add_column("ID", style="cyan", justify="right")
    table.add_column("Audio")
    table.add_column("Model")
    table.add_column("Speakers", justify="right")
    table.add_column("Words", justify="right")
    table.add_column("RTF", justify="right")
    table.add_column("Quality")
    table.add_column("Status")

    for r in runs:
        if r.get("quality_ok") == 1:
            quality = "OK"
        elif r.get("quality_ok") == 0:
            quality = "WARN"
        else:
            quality = "-"
        table.add_row(
            str(r["id"]),
            (r["audio_file"] or "")[:40],
            r.get("asr_model") or "",
            str(r.get("n_speakers") or ""),
            f"{r.get('n_words') or 0:,}",
            str(r.get("rtf") or ""),
            quality,
            r["status"],
        )
    console.print(table)


@app.command()
def rename(
    workspace: Annotated[Path, typer.Option("--workspace", "-w")],
    base_name: Annotated[str, typer.Option("--base-name", help="Output file prefix.")],
    mapping_json: Annotated[
        Path,
        typer.Option("--mapping", help='JSON: {"SPEAKER_00": "Name", ...}'),
    ],
) -> None:
    """Replace SPEAKER_XX labels with real names across all outputs of a run."""
    configure_logging(workspace=workspace)
    if not mapping_json.exists():
        console.print(f"[red]File not found:[/red] {mapping_json}")
        raise typer.Exit(1)
    mapping = json.loads(mapping_json.read_text())
    paths = WorkspacePaths(workspace=str(workspace))
    stats = rename_speakers_in_outputs(paths, base_name, mapping)
    console.print(f"[green]Total replacements:[/green] {sum(stats.values())}")


@app.command()
def clean(
    workspace: Annotated[Path, typer.Option("--workspace", "-w")],
    pattern: Annotated[
        str | None,
        typer.Option("--pattern", "-p", help="Substring to match in filenames."),
    ] = None,
    all_outputs: Annotated[
        bool,
        typer.Option(
            "--all",
            help="Delete EVERYTHING (requires --confirm='YES DELETE ALL').",
        ),
    ] = False,
    confirm: Annotated[
        str | None,
        typer.Option("--confirm", help="Textual confirmation."),
    ] = None,
    include_diar_cache: Annotated[bool, typer.Option("--include-diar-cache")] = False,
) -> None:
    """Delete outputs (selectively by pattern or everything)."""
    configure_logging(workspace=workspace)
    paths = WorkspacePaths(workspace=str(workspace))

    if all_outputs:
        if confirm != "YES DELETE ALL":
            console.print("[red]To delete everything, pass:[/red] --confirm='YES DELETE ALL'")
            raise typer.Exit(1)
        n = delete_all_outputs(paths, confirm=confirm)
        console.print(f"[green]{n} file(s) deleted[/green]")
    elif pattern:
        n = delete_outputs_for(paths, pattern=pattern, include_diar_cache=include_diar_cache)
        console.print(f"[green]{n} file(s) deleted[/green]")
    else:
        console.print("[yellow]Pass --pattern or --all[/yellow]")
        raise typer.Exit(1)


@app.command()
def bench(
    workspace: Annotated[Path, typer.Option("--workspace", "-w", help="Project root path.")],
    base_name: Annotated[
        str,
        typer.Option(
            "--base-name",
            help="Output base name, e.g. 'meeting_large-v3-turbo' (without extension).",
        ),
    ],
    ref: Annotated[
        Path, typer.Option("--ref", help="Reference transcript .txt (human ground truth).")
    ],
    rttm: Annotated[
        Path | None,
        typer.Option("--rttm", help="Reference diarization .rttm (optional, enables DER)."),
    ] = None,
    collar: Annotated[
        float, typer.Option("--collar", help="DER forgiveness collar in seconds.")
    ] = 0.25,
) -> None:
    """Compute WER (and DER with --rttm) for a finished run against references.

    Requires the bench extras: pip install 'speakerscribe[bench]'.
    Appends a 'bench' record to the runs ledger for traceability.
    """
    from speakerscribe.evaluate import bench_run

    paths = WorkspacePaths(workspace=str(workspace))
    try:
        result = bench_run(paths, base_name, ref, reference_rttm=rttm, collar=collar)
    except ImportError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e

    table = Table(title=f"Benchmark — {base_name}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("WER", f"{result['wer']:.2%}")
    table.add_row("MER", f"{result['mer']:.2%}")
    table.add_row("WIL", f"{result['wil']:.2%}")
    if result.get("der") is not None:
        table.add_row("DER", f"{result['der']:.2%}")
        table.add_row("DER collar (s)", f"{collar}")
    table.add_row("Ref words", str(result["reference_words"]))
    table.add_row("Hyp words", str(result["hypothesis_words"]))
    console.print(table)


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(f"speakerscribe v{__version__}")


if __name__ == "__main__":
    app()
