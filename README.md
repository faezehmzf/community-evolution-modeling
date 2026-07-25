# community-evolution-modeling

Read-only data-access verification and forensic discovery tooling for the TDMEC
project's **Dataset A** (frozen 16,736-account tweet extract) and **Dataset B**
(70 raw `statuses-*.xlsx` files).

## Layout

- `src/tdmec_discovery/` — reusable, platform-neutral discovery library
  (source adapters for local / public Google Drive / authenticated Drive API,
  streaming SHA-256, resumable download cache, fast xlsx inspection, field-role
  heuristics, run/manifest + atomic Drive publication, Phase-0 verifier, CLI).
- `scripts/run_discovery.py` — discovery driver (Phases 2–5).
- `tests/` — unit tests (local synthetic workbooks; no network needed).
- `docs/data/` — discovery reports (00–07).
- `docs/project/` — corrected canonical spec + readiness audit (12, 13).
- `config/discovery.example.env` — configuration template (no real ids/URLs).
- `artifacts/discovery/` — sanitized JSON summaries (large parquet + per-file
  logs stay in the git-ignored run directory).

## Quick start

```bash
pip install -r requirements.txt
cp config/discovery.example.env config/discovery.local.env   # fill in, never commit
python -m tdmec_discovery verify-access --out artifacts/discovery
python scripts/run_discovery.py
pytest -q
```

Dataset locations and the output folder are supplied via environment variables
(`DATASET_A_SOURCE`, `DATASET_B_SOURCE`, `DISCOVERY_OUTPUT_ROOT`,
`DISCOVERY_CACHE_ROOT`, `TDMEC_OUTPUT_DRIVE_FOLDER_ID`); no URLs are hardcoded.

Sources are treated as strictly read-only. See `docs/data/00_access_verification_report.md`.
