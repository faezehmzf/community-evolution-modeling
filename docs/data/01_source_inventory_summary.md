# 01 — Immutable Source Inventory Summary (Phase 2)

All source files are treated as **strictly read-only**. Nothing in either source
folder was modified, renamed, moved, deleted, re-saved, or added.

Full machine-readable inventories (git-ignored, in the run directory):
`dataset_a_inventory.parquet` (12 rows), `dataset_b_inventory.parquet` (70 rows).
Sanitized schema summaries committed under `artifacts/discovery/`.

## Dataset A — 12 workbooks + 1 metadata file

| Property | Value | Status |
|---|---|---|
| Data files | 12 × `.xlsx` (`core_army_pro_fans_tweets_part_001..012.xlsx`) | `Verified from real files` |
| Metadata file | `extraction_summary.json` | `Verified from real files` |
| Worksheet per file | `tweets` (single sheet) | `Verified from real files` |
| Columns per file | 31 (identical across all 12) | `Verified from real files` |
| Rows | 11 files × 1,048,575 + 1 file × 1,047,210 = **12,581,535** | `Verified from real files` |
| Downloaded size (12 files) | ≈ 3.37 GB (258–308 MB each) | `Verified from real files` |
| SHA-256 | Computed for all 12 (recorded in inventory parquet + run checksums) | `Verified from real files` |
| Row total cross-check | equals `extraction_summary.total_matched_tweets` (12,581,535) | `Verified from real files` |

Each Excel part is filled to the worksheet row limit (1,048,576 including
header), i.e. the split is purely mechanical by Excel's row cap, not semantic.

## Dataset B — 70 workbooks + 1 manifest

| Property | Value | Status |
|---|---|---|
| Data files | 70 × `.xlsx` (`statuses-0.xlsx` … `statuses-69.xlsx`) | `Verified from real files` |
| Metadata file | `download_manifest.json` | `Verified from real files` |
| Worksheet per file | `Sheet1` (single sheet) | `Verified from real files` (inspected subset) |
| Columns per file | 10 (single canonical schema) | `Verified from real files` (inspected subset) |
| Reported total size | **9,582,913,975 bytes ≈ 9.58 GB** (109–159 MB each) | `Verified from real files` (manifest) |
| Manifest validation | every file `validation_status = valid`, `range_supported = true` | `Verified from source code` (manifest) |
| Provenance | downloaded from `https://<redacted-external-host>/static/reports/statuses-*.xlsx` | `Verified from real files` (manifest) |
| Deep inspection | 12 representative files fully inspected; remaining 58 size-inventoried only | see note |

### Inspection-completeness note

Per the standing instruction *"do not begin full dataset download yet,"* the
full 9.58 GB Dataset B corpus was **not** downloaded. A distributed sample of
**12 files** (`statuses-0,1,2,10,20,30,34,35,49,60,68,69`) — spanning the index
range and the size range (smallest and largest included) — was downloaded **one
file at a time and evicted after inspection**. All 70 files are inventoried by
name + exact reported size via the manifest; per-file row/schema inspection of
the remaining 58 is a resumable step deferred to the controlled pilot. The
tooling (`scripts/run_discovery.py`, `--b-sample`) resumes to complete them.

## Immutability evidence

- Every source read is a copy to a local cache path outside the source; source
  handles are opened read-only.
- Adapters expose only `list_files` and `download` (to a caller-provided local
  destination); there is no code path that writes, renames, or deletes in a source.
- SHA-256 checksums recorded here establish a baseline for future integrity checks.

Fields captured per file (in the parquet inventories): dataset, relative id,
sanitized filename, extension, reported size, downloaded size (when downloaded),
SHA-256 (when computed), worksheet names, open/read success, error category, and
inspection-completion state.
