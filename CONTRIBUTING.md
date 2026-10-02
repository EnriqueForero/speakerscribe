# Contributing

## Setup

```bash
git clone <repo> && cd speakerscribe
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # or: pip install --no-deps -e . + light deps (see CI)
```

## Rules of the road

- **Unit tests run WITHOUT the GPU stack.** `tests/fakes.py` provides the
  fake `faster_whisper`; keep all heavy imports (`torch`, `faster_whisper`,
  `pyannote`) lazy inside functions. If your change makes `import
  speakerscribe` pull torch, it will be rejected.
- **Every bugfix ships with the test that would have caught it.**
- Quality gates (CI enforces all):
  ```bash
  ruff format speakerscribe/ tests/ && ruff check speakerscribe/ tests/
  mypy speakerscribe/                              # must be 0 errors (py.typed package)
  pytest tests/ -m "not integration and not gpu"   # coverage gate: 45%
  ```
- Integration suite (real decoder, needs ffmpeg + espeak-ng):
  ```bash
  pip install -e . && pytest -m integration --no-cov
  ```
- Docstrings: Google style, with the WHY, not just the what.
- No magic numbers in business logic — constants live in `config.py`.
- Outputs that gate idempotency (`.json`, ledger) are written atomically
  (`io_utils`); keep it that way.

## Data never goes to git

This repository is public. Audio, transcripts, journals (`events.jsonl`),
ledgers (`_runs.jsonl`) and master JSON files carry names and content of
real meetings: `.gitignore` excludes them and notebooks are committed
**without outputs** (`scripts/lint_notebooks.py`, enforced in CI and by the
pre-commit hook). Tests build their own synthetic data.

## Releasing (single path: tag → GitHub Actions)

1. Bump `__version__` in `speakerscribe/__init__.py` — the only version
   source (`pyproject.toml` reads it dynamically) — and add the CHANGELOG
   section. Claims in the changelog must be true at the tag.
2. Merge to `main` with CI green.
3. `python -m build && twine check dist/*`, install the wheel in a clean
   venv and run `speakerscribe version`.
4. Push the tag `vX.Y.Z`. `release.yml` checks tag == version, builds,
   publishes to PyPI with Trusted Publishing (no tokens) and creates the
   GitHub Release. Do not upload by hand: GitHub and PyPI must always carry
   the same, latest version.

One-time setup (repository owner): on PyPI → project `speakerscribe` →
*Publishing* → add a trusted publisher with owner `EnriqueForero`,
repository `speakerscribe`, workflow `release.yml`, environment `pypi`; and
create the `pypi` environment under GitHub → Settings → Environments.
