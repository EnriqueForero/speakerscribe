# Security policy

## Supported versions

Only the latest minor release receives fixes.

## Sensitive data handling

- The HuggingFace token is read from (in order) explicit config, Colab
  Secrets, or the `HF_TOKEN` env var. It is **never logged** — not even a
  prefix — and never written to outputs, the ledger, or caches.
- Run metadata (`.json`, `_runs.jsonl`) embeds the full config; the
  `hf_token` field is whatever you passed explicitly — prefer Secrets/env
  so it stays `null` in artifacts you might share.
- Media files and transcripts never leave your machine/Drive: the only
  network calls are model downloads from HuggingFace.

## Reporting a vulnerability

Open a private security advisory on the repository (Security → Advisories →
Report a vulnerability). Please do not open public issues for security
reports. Expected first response: 7 days.
