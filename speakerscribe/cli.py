"""speakerscribe CLI built on typer + rich.

Commands:
    process        Process every media file in <workspace>/data/
    smoke-test     Run a smoke test with the 'small' model on the first file
    inspect        Print a quick summary of a transcription JSON file
    stats          Show aggregate statistics from the runs database
    list-runs      List the most recent N runs
    rename         Replace SPEAKER_XX labels with real names in outputs
    clean          Delete outputs (selectively or all)
    version        Show package version
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

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
        Optional[bool],
        typer.Option(
            "--version", "-V", callback=_version_callback, is_eager=True,
            help="Show version and exit.",
        ),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable DEBUG logging.")] = False,
) -> None:
    """speakerscribe — command line interface."""
    level = "DEBUG" if verbose else "INFO"
    configure_logging(console_level=level)


@app.command()
def process(
    workspace: Annotated[Path, typer.Option("--workspace", "-w", help="Project root path.")],
    model: Annotated[
        str, typer.Option("--model", "-m", help="Whisper model name.")
    ] = "large-v3-turbo",
    language: Annotated[Optional[str], typer.Option("--language", "-l")] = None,
    beam_size: Annotated[int, typer.Option("--beam-size", "-b", min=1, max=10)] = 5,
    no_diar: Annotated[bool, typer.Option("--no-diar", help="Disable diarization.")] = False,
    num_speakers: Annotated[Optional[int], typer.Option("--num-speakers", "-n")] = None,
    min_speakers: Annotated[Optional[int], typer.Option("--min-speakers")] = None,
    max_speakers: Annotated[Optional[int], typer.Option("--max-speakers")] = None,
    word_timestamps: Annotated[
        bool, typer.Option("--word-timestamps/--no-word-timestamps")
    ] = False,
    enable_db: Annotated[
        bool, typer.Option("--enable-db/--no-db", help="Enable SQLite run history.")
    ] = False,
    force: Annotated[bool, typer.Option("--force/--no-force")] = False,
    config_file: Annotated[
        Optional[Path],
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
            enable_runs_db=enable_db,
            force_reprocess=force,
        )

    paths = WorkspacePaths(workspace=str(workspace))
    results = process_batch(paths, config)

    n_ok = sum(1 for r in results if r.get("status") == "ok")
    if n_ok == 0:
        raise typer.Exit(1)


@app.command(name="smoke-test")
def smoke_test_cmd(
    workspace: Annotated[Path, typer.Option("--workspace", "-w")],
) -> None:
    """Run a fast smoke test using the 'small' model on the first media file."""
    configure_logging(workspace=workspace)

    config = TranscriptionConfig(
        model="small",
        beam_size=1,
        enable_diarization=True,
        force_reprocess=True,
    )
    paths = WorkspacePaths(workspace=str(workspace))
    media = paths.list_media_files()
    if not media:
        console.print("[red]No media files in data/[/red]")
        raise typer.Exit(1)

    from speakerscribe.pipeline import preflight_check, process_one
    from speakerscribe.transcription import (
        load_whisper_model,
        release_whisper_model,
    )

    preflight_check(paths, config)
    model = load_whisper_model(config)
    try:
        result = process_one(media[0], paths, model, config)
        if result.get("status") == "ok":
            console.print("[green]SMOKE TEST OK[/green]")
        else:
            console.print(f"[yellow]Smoke test status={result.get('status')}[/yellow]")
    finally:
        release_whisper_model(model)


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
    s = global_stats(paths.db_path)

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
    status: Annotated[Optional[str], typer.Option("--status", "-s")] = None,
) -> None:
    """List the most recent N runs from the database."""
    paths = WorkspacePaths(workspace=str(workspace))
    runs = list_runs(paths.db_path, limit=limit, status=status)
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
        typer.Option("--mapping", help="JSON: {\"SPEAKER_00\": \"Name\", ...}"),
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
        Optional[str],
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
        Optional[str],
        typer.Option("--confirm", help="Textual confirmation."),
    ] = None,
    include_diar_cache: Annotated[bool, typer.Option("--include-diar-cache")] = False,
) -> None:
    """Delete outputs (selectively by pattern or everything)."""
    configure_logging(workspace=workspace)
    paths = WorkspacePaths(workspace=str(workspace))

    if all_outputs:
        if confirm != "YES DELETE ALL":
            console.print(
                "[red]To delete everything, pass:[/red] --confirm='YES DELETE ALL'"
            )
            raise typer.Exit(1)
        n = delete_all_outputs(paths, confirm=confirm)
        console.print(f"[green]{n} file(s) deleted[/green]")
    elif pattern:
        n = delete_outputs_for(
            paths, pattern=pattern, include_diar_cache=include_diar_cache
        )
        console.print(f"[green]{n} file(s) deleted[/green]")
    else:
        console.print("[yellow]Pass --pattern or --all[/yellow]")
        raise typer.Exit(1)


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(f"speakerscribe v{__version__}")


if __name__ == "__main__":
    app()
