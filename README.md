# speakerscribe

> **Speech-to-text with speaker diarization** — powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) + [pyannote.audio](https://github.com/pyannote/pyannote-audio), optimized for Google Colab Free Tier (T4 GPU).

[![PyPI version](https://img.shields.io/pypi/v/speakerscribe)](https://pypi.org/project/speakerscribe/)
[![Python](https://img.shields.io/pypi/pyversions/speakerscribe)](https://pypi.org/project/speakerscribe/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/EnriqueForero/speakerscribe/actions/workflows/ci.yml/badge.svg)](https://github.com/EnriqueForero/speakerscribe/actions)

---

## What it does

**speakerscribe** takes any audio or video file and produces:

| Output | Description |
|--------|-------------|
| `.txt` | Plain transcript with `[SPEAKER_XX]` labels per line |
| `.srt` | Subtitle file with timestamps and speaker labels |
| `.json` | Full structured metadata (segments, speakers, versions, RTF, timings) |
| `.transcript.md` | Readable Markdown grouped by speaker turns with a metadata header |
| `_1.txt`, `_2.txt` … | Text chunks of ~1950 words for downstream LLM processing |
| `_full_for_llm.txt` | Single-file full transcript for LLMs with large context windows |
| `.segments.jsonl` | Per-segment streaming output (opt-in via `streaming_jsonl=True`) |

---

## Quick start (Google Colab — recommended)

Open the companion notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnriqueForero/speakerscribe/blob/main/notebooks/notebook_speakerscribe.ipynb)

The notebook handles everything: installation, Google Drive mounting, HF token setup, and batch processing.

---

## Installation

```bash
pip install speakerscribe
```

> **Requires Python ≥ 3.10.** On Colab, restart the runtime after installation.

### HuggingFace token (required for diarization)

pyannote.audio requires a free HuggingFace token with access to the diarization model:

1. Create a **Read** token at <https://huggingface.co/settings/tokens> (do **not** use fine-grained tokens).
2. Accept the model terms at <https://huggingface.co/pyannote/speaker-diarization-community-1>.
3. Make the token available in one of three ways:
   - **Colab Secrets** (recommended): add `HF_TOKEN` in the Colab sidebar.
   - **Environment variable**: `export HF_TOKEN=hf_...`
   - **Config parameter**: `TranscriptionConfig(hf_token="hf_...")`

---

## Python API

```python
from speakerscribe import TranscriptionConfig, WorkspacePaths, process_batch

# 1. Configure the pipeline
config = TranscriptionConfig(
    model="large-v3-turbo",   # recommended: fast + accurate
    language="es",            # None = auto-detect
    beam_size=5,
    enable_diarization=True,
    # hf_token="hf_..."       # or set HF_TOKEN env var / Colab Secret
)

# 2. Set workspace (put your audio files in <workspace>/data/)
paths = WorkspacePaths(workspace="/content/drive/MyDrive/MyProject")

# 3. Run — processes all files in data/ and writes outputs to transcripts/
results = process_batch(paths, config)
```

### Process a single file

```python
from speakerscribe import TranscriptionConfig, WorkspacePaths
from speakerscribe.pipeline import preflight_check, process_one
from speakerscribe.transcription import load_whisper_model, release_whisper_model

config = TranscriptionConfig(model="large-v3-turbo", language="en")
paths  = WorkspacePaths(workspace="/tmp/my_project")

preflight_check(paths, config)
model = load_whisper_model(config)
try:
    result = process_one(paths.data / "interview.mp4", paths, model, config)
    print(result["total_words"], "words transcribed")
finally:
    release_whisper_model(model)
```

### Rename SPEAKER_XX labels after reviewing the transcript

```python
from speakerscribe import WorkspacePaths, rename_speakers_in_outputs

paths = WorkspacePaths(workspace="/tmp/my_project")
rename_speakers_in_outputs(
    paths,
    base_name="interview_large-v3-turbo",
    mapping={"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"},
)
```

---

## Command-line interface

```bash
# Process all audio files in a workspace
speakerscribe process --workspace /path/to/project --model large-v3-turbo --language es

# Enable debug logging
speakerscribe --verbose process --workspace /path/to/project

# Load config from a JSON file instead of CLI flags
speakerscribe process --workspace /path/to/project --config-file config.json

# Run a fast smoke test (small model, first file only)
speakerscribe smoke-test --workspace /path/to/project

# Inspect a transcription JSON file
speakerscribe inspect /path/to/project/transcripts/interview.json

# Show aggregate statistics from the runs database
speakerscribe stats --workspace /path/to/project

# List the most recent runs
speakerscribe list-runs --workspace /path/to/project --limit 20

# Rename speaker labels
speakerscribe rename --workspace /path/to/project \
  --base-name "interview_large-v3-turbo" \
  --mapping mapping.json

# Delete outputs for a specific file (keep source audio)
speakerscribe clean --workspace /path/to/project --pattern "interview"

# Delete everything (requires explicit confirmation)
speakerscribe clean --workspace /path/to/project --all --confirm "YES DELETE ALL"

# Show version
speakerscribe version
```

---

## Workspace layout

```
my_project/
├── data/                  ← place your .mp4 / .mp3 / .wav / .m4a files here
├── transcripts/           ← outputs: .txt, .srt, .json, .transcript.md
├── splits/                ← chunked .txt files and _full_for_llm.txt for LLM use
├── _audio_temp/           ← temporary 16 kHz WAVs (auto-deleted)
├── _audio_chunks/         ← chunked WAVs for long-audio splitting (auto-deleted)
├── _diar_cache/           ← cached pyannote diarization results (reuse on reruns)
├── _logs/                 ← rotating log files
└── _runs.db               ← SQLite run history (created when enable_runs_db=True)
```

---

## Supported models

| Model | VRAM (T4) | Speed | Quality |
|-------|-----------|-------|---------|
| `tiny` | ~1.5 GB | fastest | basic |
| `base` | ~1.8 GB | very fast | fair |
| `small` | ~2.5 GB | fast | good |
| `medium` | ~3.5 GB | moderate | very good |
| `large-v3-turbo` ⭐ | ~3.5 GB | fast | excellent |
| `large-v3` | ~5.0 GB | slower | best |

> `large-v3-turbo` is the recommended choice for most workflows: ~3× faster than `large-v3` with comparable accuracy. The default when no model is specified is `large-v3`.

---

## Supported audio/video formats

`.mp4` · `.mp3` · `.wav` · `.m4a` · `.mkv` · `.aac` · `.flac` · `.ogg` · `.webm`

Audio is automatically converted to 16 kHz mono WAV via `ffmpeg` before processing.

---

## Configuration reference

```python
TranscriptionConfig(
    # ── Model ──────────────────────────────────────────────────────
    model            = "large-v3",        # Whisper model name (default: large-v3)
    device           = "auto",            # "auto" | "cuda" | "cpu"
    compute_type     = "auto",            # "auto" | "float16" | "int8"

    # ── Decoding ───────────────────────────────────────────────────
    beam_size        = 5,                 # 1 (greedy) to 10 (highest quality)
    language         = None,              # None = auto-detect; "en", "es", etc.
    initial_prompt   = None,              # glossary string for proper nouns / jargon
                                          # per-file override: <stem>.prompt.txt next to audio

    # ── VAD ────────────────────────────────────────────────────────
    use_vad              = True,          # Silero VAD — skip silences
    vad_min_silence_ms   = 500,           # silences ≥ N ms split segments

    # ── Transcription ──────────────────────────────────────────────
    word_timestamps  = False,             # per-word timestamps (slower)

    # ── Diarization ────────────────────────────────────────────────
    enable_diarization   = True,          # set False for transcription only
    hf_token             = None,          # or set HF_TOKEN env var / Colab Secret
    num_speakers         = None,          # pin exact count if known
    min_speakers         = None,          # or use min/max range
    max_speakers         = None,
    diarization_model    = "pyannote/speaker-diarization-community-1",

    # ── Transcript output ──────────────────────────────────────────
    gap_max_s_transcript    = 2.0,        # seconds of silence to open a new speaker turn
    generate_transcript_md  = True,       # write the readable .transcript.md
    remove_fillers          = True,       # drop filler-only segments from .transcript.md
                                          # (.json and .srt keep ALL segments)

    # ── Splits and LLM output ──────────────────────────────────────
    words_per_split          = 1950,      # chunk size for split files
    produce_unified_for_llm  = True,      # also write a single _full_for_llm.txt

    # ── Long-audio chunking ────────────────────────────────────────
    long_audio_threshold_min = 120,       # audios longer than N min are chunked
    chunk_duration_min       = 30,        # target chunk length in minutes
    chunk_overlap_s          = 5,         # overlap between chunks in seconds

    # ── Streaming ─────────────────────────────────────────────────
    streaming_jsonl  = False,             # write segments to .segments.jsonl as produced

    # ── Run control ────────────────────────────────────────────────
    force_reprocess  = False,             # True = ignore existing outputs
    evaluate_quality = True,              # heuristic quality check after each file
    enable_runs_db   = True,              # SQLite run history for idempotency by hash

    # ── Disk safety ────────────────────────────────────────────────
    disk_margin_factor  = 0.5,            # required free space = input_size × factor
    disk_margin_min_mb  = 500,            # hard floor for required free space (MB)
)
```

---

## Quality checker

After each file, speakerscribe runs a heuristic quality check and logs any issues:

| Flag | Severity | Meaning |
|------|----------|---------|
| `LOW_LANG_CONFIDENCE` | WARNING | Language detection < 85% — check if audio is noisy |
| `LOW_RTF` | WARNING | Processing faster than 2× — possible pipeline issue |
| `HIGH_RTF` | INFO | Audio mostly silence |
| `LOW_WPM` | WARNING | < 60 words/min — aggressive VAD or very slow speech |
| `HIGH_WPM` | CRITICAL | > 250 words/min — likely Whisper hallucination |
| `SPEAKER_DOMINANCE` | WARNING | One speaker > 95% — poor diarization |
| `TOO_MANY_SPEAKERS` | WARNING | > 8 speakers detected — consider pinning `num_speakers` |
| `TINY_SPEAKERS` | WARNING | Speakers with ≤ 2 segments — likely false positives |
| `REPETITIONS` | CRITICAL | Consecutive repeated n-grams — Whisper hallucination loop |
| `WORD_DOMINANCE` | WARNING | One word dominates the transcript — possible hallucination |
| `EMPTY_SEGMENTS` | WARNING | > 10% empty segments — VAD too aggressive |

---

## Requirements

- Python ≥ 3.10
- `ffmpeg` installed and available in PATH
- NVIDIA GPU (T4 or better) recommended; CPU mode is also supported but slow

### On Google Colab

`ffmpeg` is pre-installed. GPU runtime: **Runtime → Change runtime type → T4 GPU**.

---

## Development

```bash
git clone https://github.com/EnriqueForero/speakerscribe
cd speakerscribe
pip install -e ".[dev]"
pre-commit install

# Run tests (no GPU needed for unit tests)
pytest tests/ -m "not integration and not gpu"

# Lint
ruff check speakerscribe/ tests/
ruff format speakerscribe/ tests/
```

---

## License

[MIT](LICENSE) — © 2026 Néstor Enrique Forero Herrera
