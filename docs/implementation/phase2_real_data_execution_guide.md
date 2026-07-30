# Phase 2 Real-Data Execution Guide

English only. Phase 2 produces **diagnostic evidence**, not certifications.

## What is ready

1. **Generic diagnostics framework** (calendar, dedup, text-length, coverage)
2. **Concrete Dataset A adapter** `dataset_a_xlsx_v1` (docs/data/02–03 schema)
3. **Concrete Dataset B adapter** `dataset_b_xlsx_v1` (docs/data/04–05 schema)
4. **Colab/CLI runner** that does not require editing package source

## What is not done in Cursor Cloud

Real Dataset A/B files were **not** downloaded or processed here. Do not treat
synthetic reports as real-data evidence for QCAL-B01 / QDEDUP-B01.

## Required configuration fields

| Field | How to set | Notes |
|---|---|---|
| Dataset A source | `--dataset-a-source` or `--dataset-a-file` | `local:…` / `gdrive-anon:…` / `gdrive-api:…` |
| Dataset B source | `--dataset-b-source` or `--dataset-b-file` | same schemes |
| Output root | `--output-root` | e.g. Drive publish root in Colab |
| Checkpoint root | `--checkpoint-root` (optional) | defaults under run dir |
| Chunk size | `--chunk-size` | engineering only |
| Provisional calendar | `--provisional-start/end` or YAML | diagnostic-only |
| Source format | `--source-format xlsx` | documented workbooks |
| Adapter A/B | `--dataset-a-adapter` / `--dataset-b-adapter` | defaults to v1 |
| Resume mode | `--resume-mode resume\|restart` | |
| Node index map | `--node-index-map` | required for real mode; N=16736 |
| Cache root | `--cache-root` | discovery download cache |

Never put credentials, tokens, or private absolute paths in git-tracked YAML.

## Exact Colab command template

```bash
python3 -m pip install -e ".[test]" -r requirements.txt

python3 -m tdmec_diagnostics.cli \
  --mode real \
  --config configs/phase2_diagnostics.yaml \
  --output-root /content/drive/MyDrive/TDMEC_PROJECT_OUTPUTS \
  --dataset-a-source local:/content/data/dataset_a \
  --dataset-b-source local:/content/data/dataset_b \
  --node-index-map /content/data/manifests/node_index_map.parquet \
  --resume-mode resume \
  --chunk-size 10000 \
  --cache-root /tmp/tdmec_cache
```

Synthetic dry-run (always safe):

```bash
python3 -m tdmec_diagnostics.cli \
  --mode synthetic \
  --output-root ./artifacts \
  --run-id synthetic-smoke \
  --resume-mode restart
```

## Outputs

Under `artifacts/diagnostics/<run_id>/`:

- `reports/*.json` — machine-readable, privacy-safe
- `human/*.md` — human summaries
- `execution_manifest.json`
- `checkpoints/` — file progress + accumulator state

While an unsealed run is active, the checkpoint root also contains a private
SQLite transaction file. It stores privacy-safe hashed dedup occurrence
provenance and the authoritative matching checkpoint/accumulator snapshot.
It is removed only after successful report sealing, so a completed run retains
the documented final artifact set. Resume verifies the scientific
configuration hash and input checksums, skips transactionally committed source
rows, and fails closed on incompatible legacy partial state.

Statuses: `UNVALIDATED` | `DIAGNOSTIC_COMPLETE` | `REVIEW_REQUIRED` only.

## Evidence review

After real-data reports exist, use
`docs/implementation/phase2_evidence_review_template.md` for QCAL-B01 and
QDEDUP-B01. Do not begin Phase 3 until those reviews are authorized.

## Unresolved decisions (remain open)

- QCAL-B01 bounds / T / leading-trailing policy
- QDEDUP-B01 exact signature + L2 thresholds
- Numeric coverage certification thresholds
- QEMB-X01..X07 and production `D_text`
- Dataset A referenced-status extraction rules
- Hardware batch / AMP / OOM
