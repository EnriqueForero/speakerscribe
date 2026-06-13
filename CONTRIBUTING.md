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

## Releasing

1. Bump `version` in `pyproject.toml` (single source of truth) and add the
   CHANGELOG section — claims in the changelog must be true at the tag.
2. `python -m build` locally; install the wheel in a clean venv and run
   `speakerscribe version` before tagging.
3. Tag `vX.Y.Z` → `release.yml` publishes via PyPI Trusted Publishing.
