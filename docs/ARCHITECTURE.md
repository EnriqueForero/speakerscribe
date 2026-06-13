# Architecture

One page on how the pieces fit and the contracts the pipeline guarantees.
If code and this document disagree, the code's tests win — then fix one of
the two.

## Module map

| Module | Responsibility | Heavy imports |
|---|---|---|
| `config.py` | Single source of truth for every tunable (Pydantic v2, fail-fast) + `WorkspacePaths` (durable vs scratch) | none |
| `audio.py` | ffmpeg/ffprobe wrappers with hard timeouts; content signatures (`fast`/`full`); legacy chunk splitting (deprecated) | none (subprocess) |
| `diarization.py` | `DiarizationEngine` (load once per batch), params-hashed cache, segment- and word-level speaker attribution | pyannote/torch, **lazy** |
| `transcription.py` | Streaming decode (sequential or batched), `_SegmentWriter` (single emission path), OOM ladder, prompt budget warning, `loaded_whisper` | faster-whisper/torch, **lazy** |
| `pipeline.py` | Orchestration: preflight → WAV → diarize → transcribe (+auto-retry) → MD/splits → ledger. Owns `RunOutputs` and all status semantics | none directly |
| `quality.py` | Post-run heuristics → `QualityReport` (INFO/WARNING/CRITICAL flags) | none |
| `output.py` | Readable `.transcript.md` (filler modes), word-aware splits, unified-for-LLM file | none |
| `persistence.py` | Runs ledger: JSONL primary, SQLite legacy read fallback, merged stats | sqlite3 (stdlib) |
| `io_utils.py` | Atomic writes (`tmp` + `os.replace`) and fsynced JSONL appends | none |
| `evaluate.py` | WER (jiwer) / end-to-end DER (pyannote.metrics) against user references | extras `[bench]`, **lazy** |
| `estimates.py` | Planning RTF table (T4) | none |
| `maintenance.py` | Selective deletion, JSON inspection, single-pass speaker renaming | none |
| `cli.py` / `__main__.py` | typer CLI (`speakerscribe`, `python -m speakerscribe`) | none at import |

**Invariant:** `import speakerscribe` never pulls torch/faster-whisper/
pyannote. The unit suite runs against `tests/fakes.py`; CI's unit job
installs only light deps. Break this and CI breaks, by design.

## Runtime contracts

### 1 · Idempotency (content, not filename)
Every run is keyed by `(file_signature, asr_model, diar_model)` in the
append-only `_runs.jsonl`. `hash_mode="fast"` reads ≤16 MB (size + head +
tail 8 MB); a miss falls back ONCE to the full SHA-256 to recognize pre-0.3
histories, then appends a migration record so the fallback never repeats.
Skip additionally requires the `.txt` and `.json` outputs to exist — a
ledger row alone never suppresses work whose artifacts are gone.

### 2 · Degradation (never silently "ok")
Diarization requested but failed ⇒ run completes as `status="ok_degraded"`,
quality gets a CRITICAL `DIARIZATION_FAILED`, and the ledger row carries
`diar_model=None` — which by key design can never satisfy a diarized
lookup, so the file is retried next batch.

### 3 · Auto-retry (hallucination loops)
CRITICAL `REPETITIONS`/`HIGH_WPM` ⇒ one re-decode with
`condition_on_previous_text=False, repetition_penalty=1.15,
no_repeat_ngram_size=3` into sibling `.retry` paths. Fewer critical flags
wins; promotion is `os.replace` per file; the losing attempt is ledgered
(`status="retried"`) for audit. CUDA OOM is a separate, inner ladder:
batch 8→4→2→1 with full rewrite of partial outputs between attempts.

### 4 · Storage layout (durable vs scratch)
Workspace (Drive): inputs, outputs, diar cache, logs, ledger. Scratch
(local NVMe; `/content/ss_scratch` on Colab): extracted WAVs — named
`{stem}_{signature[:10]}.wav` so reuse-by-name is reuse-by-content — and
legacy chunk WAVs. The `.json` and `.transcript.md` are written atomically;
`.txt`/`.srt`/`.segments.jsonl` stream by design (their value is partial
visibility) and are not the skip gate.

## Data flow (per file)

```
media ─ file_signature ─► ledger lookup ──hit+outputs──► skipped
   │                          │ miss
   ▼                          ▼
ffmpeg → scratch WAV → DiarizationEngine (cache) → transcribe (batched)
                                  │                       │
                            turns/None+reason       _SegmentWriter
                                  │                 txt/srt/json(l)
                                  ▼                       ▼
                          quality check ──critical──► auto-retry (once)
                                  │                       │
                                  ▼                       ▼
                        .transcript.md + splits  ◄── winner promoted
                                  │
                                  ▼
                        ledger append (status, attempt, flags)
```
