"""CLI contract: entry points, new options surfaced, bench wiring."""

from __future__ import annotations

import subprocess
import sys

from typer.testing import CliRunner

from speakerscribe import __version__
from speakerscribe.cli import app

runner = CliRunner()


class TestEntryPoints:
    def test_version_command(self):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_version_flag(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_python_dash_m_works(self):
        """H1/H18 regression: the module entry point must exist."""
        out = subprocess.run(
            [sys.executable, "-m", "speakerscribe", "version"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert out.returncode == 0
        assert __version__ in out.stdout


class TestProcessOptions:
    def test_new_options_are_exposed(self):
        # Rich truncates long option names at narrow widths ("--speaker-assi…"),
        # so render the help on a wide virtual terminal.
        result = runner.invoke(app, ["process", "--help"], env={"COLUMNS": "200"})
        assert result.exit_code == 0
        for opt in (
            "--batch-size",
            "--speaker-assignment",
            "--remove-fillers",
            "--hash-mode",
            "--auto-retry",
        ):
            assert opt in result.output, opt

    def test_new_options_are_accepted(self):
        """Functional contract: typer must parse the option names."""
        result = runner.invoke(
            app,
            [
                "process",
                "--speaker-assignment",
                "word",
                "--hash-mode",
                "fast",
                "--remove-fillers",
                "safe",
                "--batch-size",
                "8",
            ],
        )
        # exit 2 ONLY because --workspace is missing — never "No such option"
        assert "No such option" not in result.output
        assert "Missing option" in result.output


class TestBenchCommand:
    def test_listed_in_help(self):
        result = runner.invoke(app, ["--help"])
        assert "bench" in result.output

    def test_missing_files_exit_1(self, tmp_path):
        result = runner.invoke(
            app,
            [
                "bench",
                "--workspace",
                str(tmp_path),
                "--base-name",
                "nope",
                "--ref",
                str(tmp_path / "ref.txt"),
            ],
        )
        assert result.exit_code == 1
