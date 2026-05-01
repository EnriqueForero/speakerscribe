# Changelog

All notable changes to **speakerscribe** are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versioning: [Semantic Versioning](https://semver.org/)

---

## [Unreleased]

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

### Notes on migration
- Initial release. No migration needed.
