# TDMEC Embedding Pilot

**Model:** TDMEC (Temporal Dynamic Multiplex Evolutionary Community model)  
**Stage:** Bounded Qwen3 embedding pilot (typically 10 000 node + 10 000 event units)  
**Status labels:** `PROVISIONAL_SMOKE_ONLY` · `ENGINEERING_VALIDATION` · `NOT_FOR_FINAL_THESIS_CONCLUSIONS`

This document describes the reproducible pilot pipeline that prepares text
embeddings for TDMEC training. It does **not** authorize full-corpus embedding
or final thesis conclusions.

---

## Purpose

After the 64+64 Kaggle preflight proved encode → pool → mask mechanics on
`Qwen/Qwen3-Embedding-4B`, the next engineering gate is a **bounded pilot**:

1. Deterministic stratified sampling with **forced multiplex relation coverage**
   (mention, retweet, reply, quote when present in the eligible population).
2. Unit encoding with **MRL truncation → L2 normalize → hard validation**
   (norm bypass is forbidden).
3. Stage-B mean pooling into dense tensors aligned to Dataset A.
4. Graph–text alignment against `X_struct` / canonical edges.
5. Export of a **`TDMEC_INPUT`** package suitable for Lightning migration.

---

## Inputs

| Role | Typical artifact | Env var |
|---|---|---|
| Dataset A (events, edges, `X_struct`) | `smoke_a_pg_001` | `TDMEC_DATASET_A_ROOT` |
| Dataset B (node texts) | `smoke_b_pg_001` | `TDMEC_DATASET_B_ROOT` |
| Pilot output root | writable directory | `TDMEC_EMBEDDING_OUTPUT_ROOT` |
| Model / tokenizer revision | immutable SHA | `QWEN3_MODEL_REVISION`, `QWEN3_TOKENIZER_REVISION` |

Pinned revision used in the successful Kaggle preflight:

```text
5cf2132abc99cad020ac570b19d031efec650f2b
```

---

## Pipeline stages

```text
Raw graph + text artifacts
        ↓
Eligibility (Q-TEXT / Q-MISS)
        ↓
Deterministic stratified sampling (+ relation coverage)
        ↓
Qwen3 encode → last-token pool → MRL truncate → L2 normalize → validate
        ↓
Unit Parquet shards (resume / checksums)
        ↓
Stage-B mean pooling → [T,N,D] / [E,D] + masks/counts
        ↓
Graph–text alignment report
        ↓
TDMEC_INPUT export + package validation
```

Config: `configs/qwen3_tdmec_pilot.yaml`  
(also mirrored under `configs/embeddings/qwen3_tdmec_pilot.yaml`)

Default pilot settings:

- `D_text = 512` with `enable_provisional_mrl_truncation: true` (Kaggle disk-safe)
- `max_node_rows = max_event_rows = 10000`
- `force_relation_coverage: true`
- Stage-B `final_normalization: none` (N1 arithmetic mean; unit L2 applies to **unit** vectors)

---

## How to run locally (Lightning / CPU dry-run)

```bash
cd community-evolution-modeling
export PYTHONPATH=src
export TDMEC_DATASET_A_ROOT=/path/to/smoke_a_pg_001
export TDMEC_DATASET_B_ROOT=/path/to/smoke_b_pg_001
export TDMEC_EMBEDDING_OUTPUT_ROOT=/path/to/embeddings_out
export QWEN3_MODEL_REVISION=5cf2132abc99cad020ac570b19d031efec650f2b
export QWEN3_TOKENIZER_REVISION=5cf2132abc99cad020ac570b19d031efec650f2b

# Source identity / path check only
python -m tdmec_embeddings.run_pilot_embedding \
  --config configs/qwen3_tdmec_pilot.yaml \
  --dry-run
```

Real GPU pilot (explicit dual authorization required):

```bash
python -m tdmec_embeddings.run_pilot_embedding \
  --config configs/qwen3_tdmec_pilot.yaml \
  --authorize-real-model \
  --authorize-bounded-pilot
```

Validate an exported package:

```bash
python scripts/validate_tdmec_input.py \
  --package-root /path/to/TDMEC_INPUT_qwen3_tdmec_pilot_10k_001
```

---

## How to run on Kaggle

1. Upload **private** datasets: code transfer tarball, `smoke_a_pg_001`, `smoke_b_pg_001`.
2. Open `notebooks/kaggle_tdmec_embedding_pilot.ipynb` with a GPU accelerator.
3. Install deps from `requirements/embeddings-target-studio.txt` (or the notebook cell).
4. Set env vars to Kaggle dataset mount paths.
5. Run cells through export; download `TDMEC_INPUT_*` as a notebook output.

Detailed operational notes remain in `docs/handoff/11_kaggle_embedding_runbook.md`
(use **TDMEC** naming for new artifacts; avoid TM-PCC path names).

---

## Generated files

```text
<output_root>/<embedding_run_id>/
├── unit_embeddings/{node_text,event_text}/
├── pooled/
│   ├── node_snapshot_embeddings.npy          # [T,N,D]
│   ├── node_text_available_mask.npy
│   ├── node_valid_text_count.npy
│   ├── canonical_edge_embeddings.npy         # [E,D]
│   ├── edge_text_available_mask.npy
│   └── edge_valid_event_count.npy
├── reports/
│   ├── sampling_report.json
│   ├── runtime_memory.json
│   ├── embedding_validation.json
│   ├── graph_text_alignment_report.json
│   └── pilot_final_report.json
├── embedding_manifest.json
└── all_checksums.json

TDMEC_INPUT_<run_id>/
├── manifests/
├── checksums/
├── graph/                 # X_struct, masks, edges, …
├── text_embeddings/       # pooled tensors
├── configs/
└── validation_reports/
```

---

## Connection to TDMEC training

| TDMEC input | Source in package |
|---|---|
| `X_struct`, `struct_active_mask`, edges, weights | `graph/` |
| `X_node_text`, `node_text_available_mask` | `text_embeddings/` |
| `edge_text_features`, `edge_text_available_mask` | `text_embeddings/` |

Fusion MLP / Edge Gate still need masks explicitly (Q-MISS M1).  
`D_text` remains a **provisional** scientific decision until pilot review closes Q-EMB.

---

## Remaining scientific decisions

- Final `D_text` (512 vs 1024 vs native 2560)
- Whether Stage-B should optionally re-L2 pooled vectors (N2) vs N1 none
- Calendar / certification still `PROVISIONAL` on smoke artifacts
- Full eligible-population embedding requires a **separate** authorization beyond this pilot
