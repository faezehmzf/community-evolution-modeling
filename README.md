# community-evolution-modeling

Read-only data-access verification, Dataset B pilot preprocessing, and Phase 2
diagnostics tooling for the TDMEC project's **Dataset A** (frozen 16,736-account
tweet extract) and **Dataset B** (`statuses-*.xlsx` files).

This project is configured to run on **Lightning AI Studio** using local
filesystem paths. Google Colab Drive mounts are not required.

## Layout

- `src/tdmec_discovery/` — reusable, platform-neutral discovery library
  (source adapters for local filesystem; optional Google Drive adapters remain
  available but are unused in the Studio default workflow)
- `src/tdmec_pilot/` — Dataset B controlled preprocessing pilot
- `src/tdmec_diagnostics/` — Phase 2 streaming diagnostics
- `src/tdmec/` — shared contracts, hashing, validation helpers
- `scripts/` — CLI drivers (`run_discovery.py`, `run_dataset_b_pilot.py`,
  `run_phase2_diagnostics.py`, `build_node_index_map.py`)
- `notebooks/` — Lightning AI Studio notebooks (thin wrappers around the library)
- `tests/` — unit tests (local synthetic workbooks; no network needed)
- `docs/` — method, data, handoff, and implementation documentation
- `config/discovery.example.env` — configuration template (no secrets)
- `configs/` — YAML configs for pilot and Phase 2 diagnostics

## Lightning AI Studio layout

Expected Studio root: `/teamspace/studios/this_studio`

| Role | Path |
|------|------|
| Repository | `community-evolution-modeling/` |
| Dataset A | `Dataset A/core_army_pro_fans_tweets/` |
| Dataset B | `Dataset B/statuses_data/` |
| Outputs / manifests / checkpoints | `TDMEC_PROJECT_OUTPUTS/` |

## Quick start (Lightning AI Studio)

```bash
cd /teamspace/studios/this_studio/community-evolution-modeling
pip install -r requirements.txt
pip install -e ".[test]"

# Optional: local discovery env (never commit real secrets)
cp config/discovery.example.env config/discovery.local.env

pytest -q

# Phase 2 diagnostics (synthetic fixtures by default)
python scripts/run_phase2_diagnostics.py --help

# Dataset B controlled pilot against local Studio data
python scripts/run_dataset_b_pilot.py \
  --output-root /teamspace/studios/this_studio/TDMEC_PROJECT_OUTPUTS \
  --dataset-b-source "local:/teamspace/studios/this_studio/Dataset B/statuses_data" \
  --node-index-map /teamspace/studios/this_studio/TDMEC_PROJECT_OUTPUTS/manifests/node_index_map.parquet
```

### Notebooks

1. `notebooks/phase2_real_data_smoke_and_validation.ipynb` — install, tests,
   canonical node-map checks, one-file-per-dataset real-data smoke, sealed resume.
2. `notebooks/03_dataset_b_controlled_pilot.ipynb` — Dataset B pilot on local paths.

Open either notebook in Lightning AI Studio and run cells top-to-bottom. Paths
are preconfigured for `/teamspace/studios/this_studio`.

### GPU / CUDA

Phase 2 diagnostics and the Dataset B pilot are CPU-bound. The Studio image may
include a CUDA-enabled PyTorch build; attach a GPU machine in Lightning only when
running later embedding or training stages. Verify with:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Source configuration

Dataset locations and output folders are supplied via environment variables or
CLI flags (`DATASET_A_SOURCE`, `DATASET_B_SOURCE`, `DISCOVERY_OUTPUT_ROOT`,
`DISCOVERY_CACHE_ROOT`). Studio defaults use `local:` paths.

Optional Google Drive adapters (`gdrive-anon:`, `gdrive-api:`) remain in the
library for portability but are not required when data is already imported into
Studio.

Sources are treated as strictly read-only. See
`docs/data/00_access_verification_report.md` and
`docs/handoff/09_lightning_studio_execution_runbook.md`.
