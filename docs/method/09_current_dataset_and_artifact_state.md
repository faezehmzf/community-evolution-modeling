# 09 — Current Dataset and Artifact State

**Rule:** Code existing ≠ artifact certified. Temporary Cloud-local outputs ≠ persistent Drive artifacts.

## Status vocabulary
| Tag | Meaning |
|---|---|
| CODE_EXISTS | Implementation present |
| TESTS_EXIST | Automated tests present |
| PILOT_EXECUTED | Real controlled run completed |
| TEMP_LOCAL | Temporary/local output only |
| PERSISTENT_DRIVE | Written to verified Drive root |
| CERTIFIED | Validated + checksummed + accepted |

## Dataset A — verified physical state
| Fact | Value |
|---|---|
| Files | 12 `.xlsx` + `extraction_summary.json` |
| Rows | ≈ 12,581,535 |
| Columns | 31 |
| Distinct accounts | **16,736** (frozen universe) |
| Nature | Filtered/partially preprocessed tweet extract; **not** graph-ready |
| Relations | Derivable: mention/retweet/reply/quote |
| Tweet `id` | Float-lossy — unsafe exact key |
| Duplicates | ≈ 1,168,525 detected; 0 removed; policy uncertified |
| Null columns | ocr_text, quoted_count, bookmarks, views, engagement, sentiment, topic, copy_count (100% null) |
| Time span | ≈ 2017-10-28 → 2026-05-30 |
| Graph artifacts | **Not present / not certified** |

## Dataset B — verified physical state
| Fact | Value |
|---|---|
| Files | 70 `statuses-0…69.xlsx` + manifest |
| Size | ≈ 9.58 GB |
| Schema | 10 columns; RAW |
| Tweet ID | Exact string |
| User | Serialized structured object (`user.id`, …) |
| Edge fields | **Absent** |
| Language column | **Absent** |
| Role | Node text source only; must not expand N |
| Full deep inspect | 12/70; 58 inventory-only |

## Dataset B pilot — verified
| Fact | Value |
|---|---|
| Files | `statuses-2.xlsx`, `statuses-69.xlsx` |
| Code | `src/tdmec_pilot/`, scripts, config, notebook, tests |
| Results | in 2,004,845; retained 1,992,014; excluded 12,831; rejected 0; authors 496/496 matched; exact dup groups 747; conflicting 0 |
| Tests | 31 passed |
| Full 70-file | **Not run** |
| Embeddings | **None** |
| Persistent full corpus | **Not certified** |

## Artifact inventory (current)

| Artifact | Status |
|---|---|
| Discovery registries / manifests | CODE + executed discovery docs |
| Node index map | CODE_EXISTS (builder); **not CERTIFIED** |
| Edges / snapshots (TDMEC) | V3 can produce; **no certified repo/Drive artifact** |
| Dataset B full normalized parquet | **Not found** |
| Node text embeddings | **Not found** |
| Edge text embeddings | **Not found** |
| Masks / node_features.pt | **Not found** |
| Model checkpoints | **Not found** |
| Evaluation tables | **Not found** |

**Statement:** No certified model-ready artifact was found in the repository or verified persistent storage.

## Implementation maturity
| Area | Maturity |
|---|---|
| Read-only discovery | Ready |
| Dataset B 2-file pilot | Implemented + tested + real run |
| Full A graph construction (certified) | Not ready |
| Full B normalization (70 files) | Not ready |
| Embeddings | Blocked |
| TDMEC model/train/eval | Not present; blocked |

## Non-modeled counts
- Historical **10,040** node count — not the modeled universe under D2 (N = 16,736).
