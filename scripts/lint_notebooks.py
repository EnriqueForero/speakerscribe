#!/usr/bin/env python3
"""Lint every versioned notebook under ``notebooks/``.

Replaces ``check_notebook_parity.py``, whose name promised a comparison it
never performed and whose regex rejected a *gated* shutdown call
(``if APAGAR_AHORA: runtime.unassign()``), keeping CI red since 2026-06-13.

Rules (each one exists because it bit this project once):

1. **No unconditional runtime shutdown.** A ``runtime.unassign()`` (or any
   ``<alias>.unassign()``) at column 0 would kill the user's Colab session
   on "Run all". Calls nested under a condition (indented) are allowed.
2. **No ``pytz``.** It is an undeclared dependency; use ``zoneinfo``.
3. **No outputs or execution counts.** Outputs of a real run contain meeting
   titles and transcripts; versioned notebooks must be stripped
   (``nbstripout`` in pre-commit does it automatically).
4. **Valid nbformat 4 JSON** with at least one code cell.

Usage:
    python scripts/lint_notebooks.py            # all notebooks/**/*.ipynb
    python scripts/lint_notebooks.py a.ipynb    # specific files

Exit status 1 on any violation.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_GLOB = "notebooks/**/*.ipynb"

_UNCONDITIONAL_UNASSIGN = re.compile(r"^(?:[A-Za-z_]\w*\.)?unassign\(\)")
_PYTZ = re.compile(r"^\s*(?:import\s+pytz\b|from\s+pytz\b)")


def _code_lines(cell: dict) -> list[str]:
    src = cell.get("source", [])
    text = src if isinstance(src, str) else "".join(src)
    return text.splitlines()


def lint_notebook(path: Path) -> list[str]:
    """Return the list of violations for one notebook (empty means clean)."""
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return [f"not readable as JSON: {e}"]
    if nb.get("nbformat") != 4:
        return [f"nbformat must be 4, got {nb.get('nbformat')!r}"]

    violations: list[str] = []
    code_cells = 0
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        code_cells += 1
        if cell.get("outputs"):
            violations.append(f"cell {idx}: has outputs (strip them before committing)")
        if cell.get("execution_count") is not None:
            violations.append(f"cell {idx}: has an execution_count (strip it)")
        for n, line in enumerate(_code_lines(cell), 1):
            if _UNCONDITIONAL_UNASSIGN.match(line):
                violations.append(f"cell {idx} line {n}: unconditional runtime shutdown")
            if _PYTZ.match(line):
                violations.append(f"cell {idx} line {n}: imports pytz (use zoneinfo)")
    if code_cells == 0:
        violations.append("no code cells")
    return violations


def main(argv: list[str]) -> int:
    targets = [Path(a) for a in argv] or sorted(ROOT.glob(NOTEBOOK_GLOB))
    if not targets:
        print("No notebooks found.")
        return 0
    failed = False
    for path in targets:
        violations = lint_notebook(path)
        rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        if violations:
            failed = True
            print(f"FAIL {rel}")
            for v in violations:
                print(f"     - {v}")
        else:
            print(f"ok   {rel}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
