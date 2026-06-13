#!/usr/bin/env python3
"""Check parity between the two speakerscribe notebooks.

The repo ships two notebook files that documented divergent behavior in
v0.2.0 (the standalone `speakerscribe_colab.ipynb` had `runtime.unassign()`
uncommented; the packaged `notebooks/notebook_speakerscribe.ipynb` did not).

To prevent that class of bug from recurring, this script enforces a set of
checks that BOTH notebooks must pass:

1. No notebook contains an uncommented `runtime.unassign()` line.
2. No notebook imports `pytz` (zoneinfo is the supported timezone library).
3. Each notebook has at least one cell mentioning "Whisper" or "speakerscribe"
   (sanity check that we are auditing the right files).

Exits with status 1 on any failure, printing the offending file and reason.
Designed to run in CI on every PR.

Usage:
    python scripts/check_notebook_parity.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Notebooks to audit. The standalone one may not be shipped in the repo
# (it lives outside the package) — skip if missing.
NOTEBOOKS = [
    ROOT / "notebooks" / "notebook_speakerscribe.ipynb",
    ROOT / "speakerscribe_colab.ipynb",  # standalone (optional)
]

# Lines that MUST NOT appear uncommented in any cell
FORBIDDEN_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"^\s*from\s+google\.colab\s+import\s+runtime\s*;\s*runtime\.unassign\(\)"),
        "uncommented runtime.unassign() — would silently shut down the user's Colab session",
    ),
    (
        re.compile(r"^\s*runtime\.unassign\(\)"),
        "uncommented runtime.unassign() — would silently shut down the user's Colab session",
    ),
    (
        re.compile(r"^\s*import\s+pytz\b"),
        "uses pytz (undeclared dependency); use zoneinfo from stdlib instead",
    ),
    (
        re.compile(r"^\s*from\s+pytz\b"),
        "uses pytz (undeclared dependency); use zoneinfo from stdlib instead",
    ),
]

# At least one code cell must contain ONE of these markers (sanity check)
EXPECTED_MARKERS = ("Whisper", "speakerscribe", "faster_whisper", "pyannote")


def _iter_code_lines(nb_path: Path):
    """Yield (cell_idx, line_idx, line) for every line of every code cell."""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for cell_idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        if isinstance(src, str):
            src = src.splitlines(keepends=True)
        for line_idx, line in enumerate(src):
            yield cell_idx, line_idx, line.rstrip("\n")


def check_notebook(nb_path: Path) -> list[str]:
    """Run all checks. Returns the list of violations (empty == pass)."""
    violations: list[str] = []

    if not nb_path.exists():
        # Standalone notebook is optional — caller decides if missing is fatal.
        return [f"NOT FOUND: {nb_path}"]

    try:
        nb_path.read_text(encoding="utf-8")
    except Exception as e:
        return [f"UNREADABLE: {nb_path}: {e}"]

    # 1. Forbidden patterns
    for cell_idx, line_idx, line in _iter_code_lines(nb_path):
        for pattern, reason in FORBIDDEN_PATTERNS:
            if pattern.match(line):
                violations.append(f"cell {cell_idx} line {line_idx}: {line.strip()!r} — {reason}")

    # 2. Sanity marker
    text = nb_path.read_text(encoding="utf-8")
    if not any(marker in text for marker in EXPECTED_MARKERS):
        violations.append(f"no recognized marker ({EXPECTED_MARKERS}) found — wrong notebook?")

    return violations


def main() -> int:
    any_fail = False
    print(f"Notebook parity check (from {ROOT})\n")
    for nb in NOTEBOOKS:
        rel = nb.relative_to(ROOT) if nb.is_relative_to(ROOT) else nb
        if not nb.exists():
            print(f"  SKIP   {rel}  (not present in this checkout)")
            continue
        v = check_notebook(nb)
        if v:
            any_fail = True
            print(f"  FAIL   {rel}")
            for issue in v:
                print(f"           - {issue}")
        else:
            print(f"  OK     {rel}")
    print()
    if any_fail:
        print("ERROR: at least one notebook failed the parity check.")
        return 1
    print("All notebooks pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
