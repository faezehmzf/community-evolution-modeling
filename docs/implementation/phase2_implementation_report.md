# Phase 2 Implementation Report — Privacy-Safe Data Diagnostics

**Language:** English only  
**Date:** 2026-07-28 (pre-commit audit correction pass)  
**Git policy:** implementation + audit corrections only — **no stage / commit / push / PR**

## 1. Repository and main verification

| Item | Value |
|---|---|
| Branch | `feat/phase-2-data-diagnostics` |
| Committed HEAD | `2eb7a49` (= `origin/main`) |
| Phase 1 commit ancestor | `ff354519a69fa84c32437a9b4968eb18e29289a6` |
| Commits ahead/behind `origin/main` | 0 / 0 (uncommitted Phase 2 work only) |
| Integration rewrite | **Not required** |

## 2. Adapter readiness (mandatory)

| Dataset | Adapter ID | Status |
|---|---|---|
| A | `dataset_a_xlsx_v1` | **Implemented + tested** (documented `tweets` sheet schema) |
| B | `dataset_b_xlsx_v1` | **Implemented + tested** (documented `Sheet1` 10-column schema) |

Adapters stream rows (openpyxl read_only preferred), reject unsupported schemas,
map users through frozen node-universe lookup, never mutate sources, never guess
columns. Dataset A tweet `id` is explicitly **untrusted** (float-lossy).  
`referenced_status_id` extraction remains an **UnresolvedValue** (not guessed).

## 3. Real-mode CLI readiness

Operational once authorized sources + `--node-index-map` are supplied:

```bash
python3 -m tdmec_diagnostics.cli \
  --mode real \
  --config configs/phase2_diagnostics.yaml \
  --output-root ./artifacts \
  --dataset-a-source local:/path/to/A \
  --dataset-b-source local:/path/to/B \
  --node-index-map ./manifests/node_index_map.parquet \
  --resume-mode resume \
  --chunk-size 10000
```

Fails clearly when sources or node map are absent. No credentials embedded.

## 4. Diagnostic framework vs real evidence

| Kind | Status |
|---|---|
| Generic diagnostic framework | Complete |
| Concrete Dataset A adapter | Complete (schema-compatible fixtures tested) |
| Concrete Dataset B adapter | Complete (schema-compatible fixtures tested) |
| Synthetic validation | Executed |
| Real-data execution in Cursor Cloud | **Pending** (no authorized private data access) |
| QCAL-B01 / QDEDUP-B01 / coverage thresholds / QEMB | **Unresolved** (evidence-dependent) |

## 5. Pre-commit audit corrections (summary)

- Added concrete A/B adapters + schema contracts + workbook streaming I/O
- Made real-mode CLI operational with full configuration surface
- Fixed unbounded timestamp/length retention (min/max; exact frequency counters)
- Implemented transactionally consistent, source-row-aligned chunk resume
- Separated scientific vs engineering config hash (`chunk_size` excluded)
- Documented exact quantiles; production N=16736 default coverage denominator
- Strengthened tests (38 Phase 2 tests)

## 6. Streaming / resume semantics

- Workbook adapters stream from the last transactionally committed source row
- Chunk boundaries never split the multiple diagnostic records emitted by one
  Dataset A source row
- Exact dedup occurrences are stored as privacy-safe hashes and provenance in
  a disk-backed SQLite working store, not Python grouping maps
- One SQLite transaction commits dedup changes, bounded accumulator aggregates,
  and checkpoint progress together
- `diagnostics_checkpoint.json` and `accumulator_state.json` are inspection
  mirrors; an unsealed resume uses the SQLite transaction as its authority
- The working SQLite file is removed only after all final reports and required
  JSON checkpoint artifacts are sealed successfully
- Completed files and committed source rows are not reprocessed
- Restart clears checkpoints, working transaction state, reports, and manifest
- Config-hash and source-checksum mismatches hard-fail
- Scientific outputs remain independent of engineering chunk size
- Exact text quantiles use bounded frequency counters; dedup memory does not
  grow with processed occurrence count
- Final JSON artifacts and canonical scientific hashes are streamed, avoiding
  an additional complete serialized copy in memory

## 7. Exact commands

```bash
python3 -m pip install -e ".[test]" -r requirements.txt
python3 -c "import tdmec, tdmec_diagnostics, tdmec_discovery, tdmec_pilot; print(tdmec.__version__, tdmec_diagnostics.__version__)"
python3 -m pytest -p no:cacheprovider -v --tb=short
python3 -m tdmec_diagnostics.cli --help
python3 -m tdmec_diagnostics.cli --mode synthetic --output-root ./artifacts --run-id smoke --resume-mode restart
```

## 8. Phase 3+ boundary

No embeddings, Qwen3, GraphSAGE, trainers, evaluators, baselines, ablations, or
certification implemented or executed.
