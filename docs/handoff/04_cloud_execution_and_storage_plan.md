# 04 — Cloud Execution & Storage Plan

**Purpose.** Where each stage runs, what it needs (CPU/RAM/GPU/storage), how
persistence and resume work, and the environment prerequisites. Companion to
`03_cloud_implementation_spec.md`.

## 1. Environment inventory (verified)

| Item | State | Detail | Evidence |
|---|---|---|---|
| Python / libs | ✅ | 3.12; pandas, pyarrow, openpyxl, python-calamine, gdown | `docs/project/13`; `requirements.txt` |
| Embedding / DL libs | ❌ | **no torch/tensorflow/transformers** installed | `requirements.txt` (subagent-confirmed) |
| Network egress | ✅ | google.com / googleapis.com reachable | `docs/project/13` |
| Disk | ✅ | ~236 GB free (A ≈ 3.4 GB, B ≈ 9.58 GB) | `docs/project/13` |
| RAM | ✅ | ~14 GB free (one ~1 M-row workbook at a time) | `docs/project/13` |
| Dataset READ | ✅ | anonymous `gdown`, byte-verified A & B | `docs/data/00`; `docs/project/13` |
| Drive API (auth) | ❌ | 403 unauthenticated; **no credentials** | `docs/data/00`; `docs/project/13` |
| Env vars (`DATASET_*`, `TDMEC_OUTPUT_*`, `GOOGLE_APPLICATION_CREDENTIALS`) | ❌ | not injected | `docs/data/00` |
| Drive WRITE | ❌ | `OUTPUT_WRITE_ACCESS_NOT_CONFIRMED` | `docs/data/00`; `docs/project/13` |

## 2. Stage → environment / resources

| Stage | Env | GPU | RAM | Storage | Notes |
|---|---|---|---|---|---|
| A-00 source validation | Cloud/Colab | No | <2 GB | 3.4 GB transient | evict each file after hashing |
| A-01 normalize | Cloud/Colab | No | ≤4 GB | few GB parquet | one part at a time |
| A-02 node map | Cloud | No | ≤4 GB | <5 MB | deterministic; build once |
| A-03 timestamps | Cloud/Colab | No | ≤4 GB | — | reuse pilot module |
| A-04 relation extract | Cloud/Colab | No | ≤6 GB | several GB | per-part chunks |
| A-05 dedup report | Cloud | No | ≤8 GB | GB | annotate-only until D-3 |
| A-06 snapshot assign | Cloud | No | ≤4 GB | — | shared calendar module |
| A-07 aggregation | Cloud/Colab | No | ≤12 GB (DuckDB out-of-core) | GB | `GROUP BY (t,src,dst,r)` |
| A-08 features | — | — | — | — | **BLOCKED (D-6)** |
| A-09/A-10 validate+publish | Cloud/Colab | No | ≤4 GB | GB | atomic + checksums |
| B-00…B-06 (70 files) | **Colab + Drive** | No | ≤6 GB | ≤0.4 GB/file transient; parquet out | needs creds + hold lift |
| B-07 text norm | Colab/Cloud | No | ≤6 GB | GB | conservative only |
| B-08…B-11 | — | later GPU | — | — | **BLOCKED (D-7…D-10)** |

## 3. Persistent storage layout

See `03` §10. Key rules:
- **`manifests/node_index_map.parquet` is immutable** and shared by both pipelines.
- Graph and text runs live under `graph/<run_id>/` and `text/<run_id>/`.
- On Colab, `OUTPUT_ROOT` = mounted Drive (`/content/drive/MyDrive/TDMEC_PROJECT_OUTPUTS`);
  the code path is identical to a local FS root (validated in the pilot,
  `docs/data/08` §9).
- Large parquet + per-file logs stay in the git-ignored run directory; only
  sanitized JSON summaries are committed (`artifacts/discovery/` precedent).

## 4. Checkpoint & resume (both pipelines)

- Per-file checkpoint JSON records each chunk's row counts + output paths + SHA-256
  (`src/tdmec_pilot/pipeline.py`).
- Resume skips chunks whose parquet SHA-256 matches; **refuses** on canonical-config
  hash change or input checksum drift.
- Run id encodes git + config hash so a moved output root does not invalidate resume
  (paths/source excluded from the canonical hash — `configs/dataset_b_pilot.yaml`).

## 5. Prerequisites gating full runs

1. **Drive credentials + env vars** (readiness audit gap #1) — required to publish
   and verify outputs to Drive. Without them, outputs persist to Git + local run
   dir only (`docs/data/00`; `docs/project/13`).
2. **Lift the "no full Dataset B download" hold** before B-00 at 70-file scale
   (`docs/data/01`, `04`).
3. **Scientific decisions D-1…D-10** (`05`) for the corresponding stages.

## 6. Determinism & reproducibility

Node map (sorted int), aggregation (`GROUP BY`), canonical duplicate (min
`source_file,source_row`), and config hashing are all deterministic. Re-runs must
reproduce identical output SHA-256; this is itself a validation gate.
