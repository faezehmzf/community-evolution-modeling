# 08 — Dataset B Controlled Pilot: Implementation & Approval

Implements the plan in `07_dataset_b_pilot_plan.md`. Reusable, platform-neutral
code processes **exactly** `statuses-2.xlsx` and `statuses-69.xlsx`. No embeddings,
no training, no full 70-file run, no source mutation.

## 1. Files created / modified

- `src/tdmec_pilot/` — reusable package: `config.py` (canonical config + hash),
  `user_blob.py` (safe parser, no `eval`), `identifiers.py`, `timestamps.py`,
  `snapshots.py` (2017-Q4…2026-Q2), `text_quality.py` (conservative), `node_map.py`,
  `dedup.py` (exact vs conflicting), `schema.py` (normalized + Parquet schema),
  `storage.py` (run id, atomic writes, manifest, checksums), `pipeline.py` (P00–P12).
- `scripts/run_dataset_b_pilot.py` — CLI runner (resume-aware).
- `scripts/build_node_index_map.py` — builds the immutable 16,736-node map from Dataset A.
- `configs/dataset_b_pilot.yaml` — canonical + runtime config.
- `notebooks/03_dataset_b_controlled_pilot.ipynb` — thin Lightning AI Studio wrapper.
- `tests/test_pilot.py` — 22 unit + integration tests.

## 2. Lightning AI Studio execution steps

1. Open `notebooks/03_dataset_b_controlled_pilot.ipynb` in Lightning AI Studio.
2. Run cell 1 → sets local Studio paths (`OUTPUT_ROOT = …/TDMEC_PROJECT_OUTPUTS`).
3. Run cell 2 → installs the editable package and puts `src/` on path.
4. Run cell 3 → sets `DATASET_B_SOURCE=local:…` and `NODE_INDEX_MAP_PATH`.
5. Run cell 4 → executes the pilot; prints `run_id`, gate results, accounting.
6. Run cell 5 → lists `pilot/<run_id>/` outputs + gate summary.
7. To **resume**, re-run cell 4 with `run_id='<printed run_id>'`.

## 3. CLI command

```bash
python scripts/run_dataset_b_pilot.py \
  --config configs/dataset_b_pilot.yaml \
  --dataset-b-source "local:/teamspace/studios/this_studio/Dataset B/statuses_data" \
  --node-index-map /teamspace/studios/this_studio/TDMEC_PROJECT_OUTPUTS/manifests/node_index_map.parquet \
  --output-root /teamspace/studios/this_studio/TDMEC_PROJECT_OUTPUTS
  # add: --run-id <id> --resume to resume
```

## 4. Tests run and results

`python -m pytest -q` → **31 passed** (9 discovery + 22 pilot). Pilot coverage:
user-blob parsing (literal/JSON/quotes/malformed/no-eval), exact large-int tweet
IDs, float-ID rejection, missing user, invalid timestamps, snapshot boundaries &
out-of-range, node-map build/load, matched/unmatched accounts, exact vs
conflicting duplicates, full-run row accounting, resume after interruption, and
config-incompatibility refusal.

## 5. Real 2-file run evidence (Linux, local output root)

Executed on the real `statuses-2.xlsx` + `statuses-69.xlsx` with the real
16,736-node map. **All 11 validation gates passed.**

| Metric | Value |
|---|---|
| rows_in | 2,004,845 |
| retained | 1,992,014 |
| excluded | 12,831 (all `outside_canonical_snapshot_range`, i.e. after 2026-Q2) |
| rejected | 0 |
| accounting | input == retained + excluded + rejected ✓ |
| unique authors | 496 — **all matched to frozen**, 0 unmatched |
| duplicates | 747 groups, all `exact_duplicate`, 0 conflicting |
| text quality | 2,004,820 ok, 25 empty |
| raw source unchanged | ✓ (input checksums identical before/after) |

## 6. Validation gates (must all pass)

both inputs available · both checksums recorded · schema matches contract ·
tweet IDs exact strings · user IDs parsed or rejected · every retained record has
a node index · no node index outside 0–16,735 · every retained record has a
snapshot 0–34 · row accounting balances · Parquet reopenable · output checksums
verified · raw source unchanged.

## 7. How to resume

Re-invoke with the same `--run-id` (CLI) or `run_id=` (notebook). Completed,
checksum-verified chunks are skipped; incomplete chunks are reprocessed to
deterministic part filenames (no duplication). Resume is **refused** if the
canonical config hash differs (`ConfigIncompatibleError`) or an input checksum
changed.

## 8. How to verify raw files were unchanged

Input SHA-256 is recorded in `manifest.json.source_checksums` at P01 and
re-verified at P11 (`raw_source_unchanged`). Source adapters are read-only (no
write/rename/delete path). Independent check: re-hash the source/cached inputs and
compare to `source_checksums`.

## 9. Parts still requiring real Colab execution

- Running with Drive **mounted** as the persistent `output_root` (validated here
  against a local FS root — identical code path).
- Publishing/`node_index_map.parquet` to `…/manifests/` on Drive.
- (Everything else — parsing, normalization, reconciliation, snapshotting,
  chunking, resume, gates — is already validated on real data here.)

## 10. Pilot approval checklist

- [ ] 31/31 tests pass.
- [ ] Node-index map has exactly 16,736 entries, indices 0–16,735.
- [ ] Real 2-file run: all 11 gates pass; accounting balances.
- [ ] 0 records assigned a new node index; unmatched authors only in
      `unmatched_accounts.parquet` / excluded outputs.
- [ ] Out-of-range records excluded (not deleted) with
      `outside_canonical_snapshot_range`.
- [ ] `tweet_id` remains an exact string end-to-end.
- [ ] Raw source checksums unchanged.
- [ ] Duplicates reported (exact vs conflicting), none silently dropped.
- [ ] Resume verified (no duplicated normalized rows); config-incompat refused.
- [ ] Reviewer approves scaling beyond the 2-file pilot (separate task).
