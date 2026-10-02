"""scripts/lint_notebooks.py — each rule with a passing and a failing case."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("lint_notebooks", ROOT / "scripts/lint_notebooks.py")
assert _spec and _spec.loader
lint = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lint)


def _nb(tmp_path: Path, *sources: str, outputs=None, execution_count=None) -> Path:
    cells = [
        {
            "cell_type": "code",
            "source": src,
            "metadata": {},
            "outputs": outputs or [],
            "execution_count": execution_count,
        }
        for src in sources
    ]
    path = tmp_path / "n.ipynb"
    path.write_text(
        json.dumps({"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": cells})
    )
    return path


def test_clean_notebook_passes(tmp_path):
    assert lint.lint_notebook(_nb(tmp_path, "print('hola')")) == []


def test_gated_shutdown_is_allowed(tmp_path):
    src = "if APAGAR_AHORA:\n    from google.colab import runtime\n    runtime.unassign()\n"
    assert lint.lint_notebook(_nb(tmp_path, src)) == []


def test_unconditional_shutdown_fails(tmp_path):
    for src in ("runtime.unassign()\n", "_gr.unassign()\n"):
        assert any("shutdown" in v for v in lint.lint_notebook(_nb(tmp_path, src)))


def test_pytz_fails(tmp_path):
    assert any("pytz" in v for v in lint.lint_notebook(_nb(tmp_path, "import pytz\n")))


def test_outputs_and_execution_count_fail(tmp_path):
    path = _nb(tmp_path, "1+1", outputs=[{"output_type": "stream", "text": "x"}], execution_count=3)
    violations = lint.lint_notebook(path)
    assert any("outputs" in v for v in violations)
    assert any("execution_count" in v for v in violations)


def test_versioned_notebooks_are_clean():
    for path in sorted(ROOT.glob(lint.NOTEBOOK_GLOB)):
        assert lint.lint_notebook(path) == [], path.name
