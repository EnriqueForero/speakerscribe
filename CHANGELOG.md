# Changelog

All notable changes to **speakerscribe** are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versioning: [Semantic Versioning](https://semver.org/)

---

## [0.3.0] — 2026-06-11

> Release auditado: cierra los hallazgos P0/P1 de la auditoría de junio 2026.

### Fixed
- **P0** — The `speakerscribe` CLI entry point described in the 0.2.0 notes
  was never actually added to `pyproject.toml`; `0.2.0` was also never
  published to PyPI. The entry point ships NOW (`[project.scripts]` +
  `python -m speakerscribe`), with a subprocess regression test.
- **P0** — A diarization failure no longer ends as a silent `"ok"`: the run
  finishes as `status="ok_degraded"` with a CRITICAL `DIARIZATION_FAILED`
  quality flag, is ledgered with `diar_model=None`, and is retried on the
  next batch instead of being skipped.
- **P0** — Temporary WAVs are now named by content signature
  (`{stem}_{sig[:10]}.wav`): replacing a source file under the same name can
  never resurrect a stale WAV.
- SRT numbering in the streaming path is consecutive again (the counter
  incremented before the empty-text filter; both paths now share one
  `_SegmentWriter`).
- `EMPTY_SEGMENTS` quality check now measures real data
  (`empty_segments_discarded` in metadata) instead of an always-zero count.
- `rename_speakers_in_outputs` is single-pass (regex alternation): swap
  mappings (`00→01, 01→Ana`) no longer chain; `.segments.jsonl` included.
- `delete_all_outputs` now also sweeps the scratch `_audio_chunks/` folder
  and documents that the runs ledger is intentionally preserved (history,
  not output).
- `release.yml` no longer installs `hatchling` (backend is setuptools).
- ffmpeg/ffprobe calls have hard timeouts (auto: 120 s + 2× duration) with
  actionable errors instead of hanging on stalled Drive mounts.

### Added
- **Batched inference** (`batch_size`, default 8) via faster-whisper's
  `BatchedInferencePipeline` (~3-4× on T4 per the upstream benchmark), with
  an automatic OOM ladder (8→4→2→1; partial outputs rewritten cleanly).
- **Word-level speaker attribution** (`speaker_assignment="word"`, default):
  WhisperX-style re-segmentation at speaker changes + single-word island
  smoothing. `"segment"` keeps the legacy behavior.
- **Anti-hallucination decoding controls** exposed in config
  (`condition_on_previous_text`, `temperature`, `compression_ratio_threshold`,
  `log_prob_threshold`, `no_speech_threshold`,
  `hallucination_silence_threshold`, `repetition_penalty`,
  `no_repeat_ngram_size`) — defaults equal faster-whisper's.
- **Auto-retry on critical** (`auto_retry_on_critical`, default True): one
  retry with anti-loop overrides on CRITICAL `REPETITIONS`/`HIGH_WPM`; both
  attempts ledgered; the better one kept (atomic promotion).
- **Runs ledger in JSON-Lines** (`_runs.jsonl`, append-only, Drive/FUSE-safe)
  replacing SQLite as default backend; legacy `_runs.db` remains readable
  (transparent fallback + merged stats). SQLite still fully supported for
  local `.db` paths.
- **Fast content signatures** (`hash_mode="fast"`: size + first/last 8 MB)
  with one-time transparent migration from pre-0.3 full-SHA-256 histories.
- **Local scratch** for temporary WAVs/chunks (`WorkspacePaths.scratch`,
  auto `/content/ss_scratch` on Colab) — no more multi-hundred-MB WAVs
  round-tripping through Drive FUSE.
- **`DiarizationEngine`**: pyannote pipeline loaded once per batch and
  reused (it reloaded per file before).
- **`loaded_whisper`** context manager guaranteeing model release
  (`release_whisper_model` docstring now states its real semantics).
- **Filler modes** `off|safe|aggressive` (default `safe`): "sí/no/claro/ok"
  are only dropped in aggressive mode — they can be real answers.
- **`speakerscribe bench`** + `speakerscribe.evaluate`: WER (jiwer) and
  end-to-end DER (pyannote.metrics) against user references; extras
  `pip install 'speakerscribe[bench]'`; results appended to the ledger.
- **`speakerscribe.estimates`**: planning RTF table (was duplicated in the
  notebook).
- Atomic writes (`io_utils`) for `.json`, `.transcript.md` and caches.
- Integration test suite (real espeak-ng audio + tiny model) on a weekly
  CI job; unit CI runs in seconds against fakes, without the GPU stack.
- Notebook (v9): fast direct install from the local wheel with ONE planned
  automatic kernel restart when already-loaded binaries (numpy/torch) get
  upgraded — pyannote.audio 4.0.x pins exact torch versions, so fully
  constraint-frozen installs are unsolvable by design; the post-restart
  fast-path re-verifies in seconds (import origin + version + stack-filtered
  `pip check`). Keeps the import-shadowing guards, the deliverables cell with
  `_resumen.md` + redacted run metadata, and the runtime shutdown cell. Merges the original notebook's clarity:
  fully documented option surface (model/VRAM table, speaker range,
  VAD/gap, filler modes, ≤4000-char glossary, deliverable selection),
  a capabilities overview, a per-run JSON inspector cell, and a
  flag-driven speaker rename that refreshes deliverables.
  Supersedes the v7 staged installs and the v4 cell list:
  constraints install (no restart), diagnostic cell, isolated
  `_smoke/` workspace, ledger-backed history, optional bench cell.
- **Typed package** (PEP 561): `py.typed` marker shipped; `mypy
  speakerscribe/` passes with 0 errors and is enforced in CI. Internal
  `RunOutputs` dataclass replaces the untyped output-paths dict.
- `docs/ARCHITECTURE.md`: module responsibilities and the four runtime
  contracts (idempotency, degradation, auto-retry, storage layout).

### Changed
- `initial_prompt` schema bound 500 → 4000 chars; the per-file glossary is
  **no longer silently truncated** — Whisper conditions on the trailing
  ~224 tokens and the library warns when a prompt exceeds that budget.
- `remove_fillers` is a mode string (booleans coerced: True→"safe").
- Coverage gate 25% → 45% (suite currently ~70%); pytest markers reduced
  to `integration` / `gpu` / `slow`.

### Deprecated
- External long-audio chunking (`long_audio_threshold_min`, now default 0):
  the batched path handles long audio natively and more accurately at
  boundaries. The legacy path still works when explicitly enabled
  (`chunk_overlap_s` default raised 5 → 30; word offsets now corrected),
  and emits a `DeprecationWarning`.

---

## [0.2.0] — 2026-05-03 — never published

> **Correction (2026-06-11):** this version was tagged in the changelog but
> never released to PyPI, and the `[project.scripts]` entry point described
> below was **not** actually present in `pyproject.toml` at this tag — that
> fix ships in 0.3.0. The features below DID land in the codebase.


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
