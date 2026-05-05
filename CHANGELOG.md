# Changelog

All notable changes to **speakerscribe** are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versioning: [Semantic Versioning](https://semver.org/)

---

## [0.2.0] — 2026-05-03

### Fixed
- **Critical**: `pyproject.toml` line 51 had a malformed string literal
  (`where = [".]`) that prevented `pip install -e .` from succeeding.
  Now reads `where = ["."]`.
- **Critical**: The `speakerscribe` CLI was implemented in `cli.py` but
  never exposed via an entry point in `pyproject.toml`. Tools like
  `speakerscribe process --workspace ...` documented in the README failed
  with "command not found". Added `[project.scripts]` mapping to
  `speakerscribe.cli:app`.

### Added
- **Long-audio chunking** (`split_long_audio` in `audio.py`): audios longer
  than `config.long_audio_threshold_min` (default 120 min) are split into
  overlapping chunks (default 30 min, 5 s overlap) for transcription.
  Diarization continues to run on the FULL audio for speaker consistency.
  New `transcribe_chunked` in `transcription.py` handles concatenation with
  timestamp adjustment and overlap deduplication.
- **`get_audio_duration_seconds`** (`audio.py`): probe duration via ffprobe
  without decoding.
- **`AudioChunk`** dataclass: structured info per chunk (path, index, start_s,
  end_s, is_last).
- **Diarization cache invalidation by parameter hash** (`diarization.py`):
  cache files are now keyed by `(stem, params_hash)`. Changing
  `num_speakers`, `min_speakers`, `max_speakers`, or `diarization_model`
  invalidates the cache automatically. Old caches without `params_hash` are
  ignored on read.
- **JSON Lines streaming output** (`config.streaming_jsonl`): when enabled,
  each segment is additionally written to `<base>.segments.jsonl` as it is
  produced. Useful for very long audios where loading the full `.json`
  into memory is undesirable.
- **Per-stage telemetry**: every run now records `extract_wav_s`,
  `diarization_s`, `split_audio_s`, `whisper_launch_s`, `transcribe_total_s`,
  `transcript_md_s`, `splits_s`, `quality_check_s`, and `total_s` in
  `metadata["timings"]`.
- **Filler-word filter** (`config.remove_fillers=True` by default):
  segments whose text is composed entirely of fillers (`"eh"`, `"um"`,
  `"o sea"`, ...) for the detected language are dropped from the readable
  `.transcript.md`. The `.json` and `.srt` keep ALL segments for traceability.
  Filler lists provided for `es`, `en`, `pt`, `fr`.
- **Per-file glossary**: a file `<media_stem>.prompt.txt` next to the source
  media overrides the global `initial_prompt` for that single audio. Allows
  having a tailored glossary per meeting without re-configuring the pipeline.
- **Unified-for-LLM output** (`config.produce_unified_for_llm=True` by default):
  in addition to the chunked splits, writes a single `<base>_full_for_llm.txt`
  with the entire transcript — convenient for modern LLMs with large context
  windows.
- **`enable_runs_db=True` is now the default**: idempotency by file content
  hash is more robust than filename-based detection. The SQLite DB lives at
  `<workspace>/_runs.db` and adds ~50 KB per workspace.

### Changed
- `WorkspacePaths` now exposes `audio_chunks` (`<workspace>/_audio_chunks/`)
  for chunked WAVs. Created on demand via `create_directories()`.
- The companion notebook (`speakerscribe_colab_v3.ipynb`) was rewritten:
  centralized configuration, smoke test, deliverables folder, consolidated
  per-batch summary, fixed VRAM-cleanup bug, fixed the runtime-shutdown bug
  (no longer apaga sin confirmación explícita).

### Notes on migration
- Existing diarization caches without `params_hash` will be ignored once and
  recomputed on the next run. New caches will reuse correctly.
- If you were relying on `enable_runs_db=False` behaviour, set it explicitly
  in your `TranscriptionConfig`.
- The `transcript.md` will now be shorter (filler-free) by default. Set
  `remove_fillers=False` to keep the previous behaviour.

---

## [0.1.0] — 2026-05-01

### Added
- **`TranscriptionConfig`** — Pydantic v2 configuration with full validation (model,
  device, beam size, speaker counts, HF token resolution, VAD parameters).
- **`WorkspacePaths`** — Single source of truth for all filesystem paths; auto-creates
  subdirectories on demand.
- **`process_batch`** — Orchestrates the full pipeline over a folder of audio/video
  files. Whisper model is loaded once and reused across all files.
- **`process_one`** — Full pipeline for a single file: WAV extraction → diarization
  → streaming transcription → Markdown transcript → text splits → quality check.
- **`preflight_check`** — Validates GPU, VRAM, disk space, and HF token availability
  before loading any heavy model.
- **`diarize_audio`** — Speaker diarization via pyannote.audio 4.x with optional
  JSON cache to avoid re-running on repeated runs.
- **`assign_speaker_to_segment`** — Maximum-overlap algorithm to label each Whisper
  segment with the dominant speaker from the diarization output.
- **`transcribe_streaming`** — Streaming transcription via faster-whisper that writes
  each segment to disk immediately, keeping RAM below ~500 MB for any audio length.
- **`generate_transcript_md`** — Generates a readable `.transcript.md` file grouped
  by speaker turns with metadata header.
- **`split_text_by_words`** — Splits large `.txt` transcripts into LLM-friendly
  chunks, respecting speaker-turn boundaries when labels are present.
- **`evaluate_transcription_quality`** — Heuristic quality checker that detects
  low language confidence, RTF anomalies, word density extremes, speaker dominance,
  Whisper hallucination loops, and empty segments.
- **`rename_speakers_in_outputs`** — Post-processing: replaces `SPEAKER_XX` labels
  with real names across all output files.
- **SQLite run history** (opt-in via `enable_runs_db=True`) — content-based
  idempotency by file hash, queryable history, aggregate statistics.
- **CLI** via `speakerscribe` command: `process`, `smoke-test`, `inspect`, `stats`,
  `list-runs`, `rename`, `clean`, `version`.
- Google Colab companion notebook (`speakerscribe_colab.ipynb`) for zero-setup
  usage on the free T4 tier.
