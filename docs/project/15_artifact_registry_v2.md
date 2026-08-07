# 15 — Artifact Registry v2

Every known artifact: path, format, key facts, mutability, presence on disk.
Cross-links: `docs/handoff/00`–`06`. Presence is relative to this repository /
the documented run directories.

## Raw source (immutable, read-only, NOT in git)

| Artifact | Format | Key facts | Mutability | Present |
|---|---|---|---|---|
| Dataset A `core_army_pro_fans_tweets_part_001..012.xlsx` | xlsx | 12 files, sheet `tweets`, 31 cols, 12,581,535 rows, ≈3.37 GB, 16,736 accounts | **immutable** | External source (verified read; not committed) |
| Dataset A `extraction_summary.json` | json | provenance: 41 `bts-*`; 12,581,535 matched; 1,168,525 dups (0 removed) | immutable | External source |
| Dataset B `statuses-0..69.xlsx` | xlsx | 70 files, sheet `Sheet1`, 10 cols, ≈9.58 GB | **immutable** | External source (12 verified, 58 inventoried) |
| Dataset B `download_manifest.json` | json | per-file size, all `valid` | immutable | External source |

## Committed discovery artifacts (`artifacts/discovery/`)

| Artifact | Format | Key facts | Mutability | Present |
|---|---|---|---|---|
| `dataset_a_schema_registry.json` | json | 31 A columns; sig `83e45e17ffc3` | regenerable | ✅ |
| `dataset_b_schema_registry.json` | json | 10 B columns; family `f903842ea287`; 12 inspected | regenerable | ✅ |
| `discovery_manifest.json` | json | run id, phases 2–5 state (a complete / b,recon sample-complete) | regenerable | ✅ |
| `access_verification.json` | json | A/B read confirmed; Drive write NOT confirmed | regenerable | ✅ |
| `output_write_access.json` | json | `OUTPUT_WRITE_ACCESS_NOT_CONFIRMED` | regenerable | ✅ |

## Git-ignored run-dir artifacts (documented, NOT committed)

| Artifact | Format | Key facts | Present |
|---|---|---|---|
| `dataset_a_inventory.parquet` (12 rows) | parquet | per-file size/sha256/sheets | run dir only |
| `dataset_b_inventory.parquet` (70 rows) | parquet | per-file size; 12 inspected | run dir only |
| `dataset_b_file_statistics.parquet` | parquet | per-column nulls/distincts (12 files) | run dir only |
| `account_reconciliation.json`, `checksums.json`, per-file logs | json | phases 3–5 | run dir only |

## Documentation (committed)

| Artifact | Present |
|---|---|
| `docs/data/00`–`08` (access, inventory, A forensic/contract, B inspection/contract, reconciliation, pilot plan/impl) | ✅ |
| `docs/project/12` (canonical decision register), `13` (readiness audit) | ✅ |
| `docs/project/14` (this session), `15` (this file) | ✅ (new) |
| `docs/handoff/00`–`06` (this session) | ✅ (new) |

## Code (committed)

| Artifact | Present | Notes |
|---|---|---|
| `src/tdmec_discovery/*` (11 modules) | ✅ | read-only discovery library |
| `src/tdmec_pilot/*` (11 modules) | ✅ | Dataset B pilot; PILOT_VALIDATED |
| `scripts/run_discovery.py`, `run_dataset_b_pilot.py`, `build_node_index_map.py` | ✅ | — |
| `configs/dataset_b_pilot.yaml` | ✅ | canonical + runtime |
| `notebooks/03_dataset_b_controlled_pilot.ipynb` | ✅ | Lightning AI Studio wrapper |
| `tests/test_pilot.py`, `test_tooling.py` | ✅ | 31 passing (22 + 9) |

## Model-ready artifacts (target)

**No certified model-ready artifacts were found in the current repository or
verified persistent storage.** Temporary Cloud-local pilot outputs may have
existed during runs, but they are not persistent or artifact-certified and are not
treated as existing artifacts here. "NOT FOUND" below means: not present in the
repo or in verified persistent storage.

| Artifact | Status |
|---|---|
| `manifests/node_index_map.parquet` (0..16,735) | **NOT FOUND** — buildable now (first task) |
| `graph/edges/snapshot=*/relation=*` | **NOT FOUND** — A-07 |
| `graph/snapshot_calendar.json` | **NOT FOUND** — A-06 |
| `X_struct[T,N,17]` float32 + `struct_active_mask[T,N]` bool | **NOT FOUND** — spec CANONICAL (Q-FEAT, F_struct=17; ordered 17-feature schema + versioned name list + relation/snapshot maps + node-map hash + Q-DEDUP provenance); build after edges/node-map |
| Dataset B normalized full-corpus parquet (70 files) | **NOT FOUND** — B scale-up |
| `X_node_text[T,N,D_text]` (B node text) | **NOT FOUND** — unit CANONICAL (Q-TEXT: per-tweet→mean-pool); missingness CANONICAL (Q-MISS M1: exact zero + mask); blocked on Q-EMB (`D_text`, encoder) |
| edge-text embeddings `E_{i→j}^{(t,r)}` (A-derived) | **NOT FOUND** — unit CANONICAL (Q-TEXT: per-event→mean-pool per canonical edge; required, built second); missingness CANONICAL (Q-MISS M1); blocked on Q-EMB |
| `node_text_available_mask[T,N]`, `node_valid_text_count[T,N]` | **NOT FOUND** — contract CANONICAL (Q-MISS M1); build with node-text embeddings |
| `edge_text_available_mask`, `edge_valid_text_count` (canonical edge order) | **NOT FOUND** — contract CANONICAL (Q-MISS M1); build with edge-text embeddings |
| TDMEC checkpoints / results | **BLOCKED** (no method/model) |

## Immutability & safety

- Raw sources: strictly read-only; SHA-256 baselines recorded (`docs/data/00,01`).
- Node-index map: **immutable once published** (`src/tdmec_pilot/node_map.py`).
- No credentials, tokens, or real Drive IDs are committed (`config/discovery.example.env`
  is a template; real ids kept git-ignored — `docs/data/00`).
- No raw tweet text is written to any committed artifact (`scripts/run_discovery.py`).
