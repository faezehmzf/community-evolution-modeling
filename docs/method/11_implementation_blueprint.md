# 11 — Implementation Blueprint

**This task does not write production code.** This blueprint defines what a later Cursor task should author after open decisions are resolved.

## 1. Package structure (target)

```
src/tdmec/
  __init__.py
  config/          # YAML schemas, loaders, validators
  ids/             # exact user/tweet ID utilities (no float)
  parse/           # safe user/status blob parsing
  nodes/           # immutable node_index_map builder/validator (N=16736)
  snapshots/       # calendar + assignment
  dataset_a/       # ingest, events, aggregate, features, edge_text prep
  dataset_b/       # ingest, reconcile, normalize, text units
  align/           # graph–text alignment + provenance
  embeddings/      # offline encoder interface (blocked until O-EMB)
  graph/           # tensor packaging, masks
  model/           # encoder, fusion, gru, head (blocked until tensors)
  losses/          # struct/sem/cluster/reg/temp
  train/           # staged trainer, checkpoints
  eval/            # metrics, baselines hooks
  evolution/       # trajectories, events
  io/              # parquet/safetensors/manifests/checksums
  runtime/         # logging, seeds, env probes

scripts/
  build_node_index_map.py          # exists — harden to D2
  run_dataset_a_graph.py           # new
  run_dataset_b_normalize.py       # new (full 70)
  run_embeddings.py                # blocked
  run_pack_tensors.py              # blocked until upstream
  run_train_tdmec.py               # blocked
  run_eval_tdmec.py                # blocked

configs/
  dataset_a_graph.yaml
  dataset_b_full.yaml
  embeddings.yaml                  # after O-EMB
  model_tdmec.yaml                 # after O-HPARAM
  train.yaml

tests/
  test_ids.py
  test_node_map_d2.py
  test_events_relations.py
  test_snapshots.py
  test_alignment.py
  test_masks.py
  test_model_shapes.py             # later
  test_losses.py                   # later

notebooks/                         # thin Colab wrappers only
  10_build_node_map.ipynb
  11_dataset_a_graph.ipynb
  12_dataset_b_full.ipynb
  13_embeddings.ipynb
  14_train_eval.ipynb
```

Keep existing:
- `src/tdmec_discovery/` (read-only discovery)
- `src/tdmec_pilot/` (2-file pilot reference; do not treat as full B)

## 2. Module responsibilities & public interfaces

| Module | Responsibility | Public API (sketch) |
|---|---|---|
| `nodes` | Build/validate immutable map N=16736 | `build_node_map(paths) -> NodeMap`; `assert_n(map, 16736)` |
| `dataset_a.events` | Extract 4 relations | `iter_events(xlsx) -> Event` |
| `dataset_a.aggregate` | Aggregate + weights | `aggregate(events, snaps) -> Edges` |
| `dataset_b.reconcile` | Map authors; drop unmatched | `filter_to_map(rows, map)` |
| `embeddings` | Frozen encode | `embed_texts(batch) -> ndarray` |
| `model` | TDMEC forward | `TDMEC.forward(batch) -> Q, aux` |
| `train` | Staged loop | `Trainer.fit(stages)` |
| `io` | Manifests/checksums | `publish(artifact, root)` |

## 3. Configuration schemas (fields)
- Paths: source A/B, output root, Drive mount
- `n_nodes: 16736` (invariant)
- `snapshot_freq`, `snapshot_start`, `snapshot_end` (after O-CAL)
- `relations: [mention, retweet, reply, quote]` + explicit id map (QREL-01: 0/1/2/3)
- `include_self_loops: false` (QSELF-01)
- `dedup_policy` (after O-DEDUP)
- `features` list (after O-FEAT)
- `encoder_name`, `d_text`, pooling, max_len (after O-EMB)
- Model/train: `d_h`, `K`, `lambdas`, seeds, epochs

## 4. Artifact schemas (targets)
- `node_index_map.parquet`: `user_id:str`, `node_idx:int32`, provenance cols
- `snapshots.parquet`: `snapshot_id`, `period`, `start_time_utc`, `end_time_utc`
- `edges.parquet`: `snapshot_id`, `relation_id`, `src_idx`, `dst_idx`, counts, weights, edge_text_id?, masks
- `dataset_b_normalized/`: partitioned parquet
- Embedding stores + meta JSON
- Tensors: node_features, node_text, edge_text, masks
- Manifests: SHA256, git SHA, config hash, row counts, exclusion counts

## 5. CLI surface
`python -m tdmec <command> --config ... --run-mode resume|fresh`  
Commands: `build-node-map`, `build-graph`, `normalize-b`, `embed`, `pack`, `train`, `eval`, `validate`.

## 6. Tests required
- Exact ID round-trips; refuse float reconstruction
- Node map length == 16736; deterministic indices
- Relation extraction unit tests (synthetic rows)
- Self-loop exclusion
- Snapshot assignment boundaries
- B unmatched authors dropped; N unchanged
- Implement only modules defined in TDMEC contracts (`03`–`07`)
- Leakage tests: future tweets not in past T_i^(t)
- Shape tests once tensors exist

## 7. Dependencies (future pin)
- Already: pandas, openpyxl, pyarrow, pyyaml, pytest, …
- Later: torch, torch-geometric (or equivalent), encoder package (Qwen3 Embedding per Q-EMB `docs/method/16`) after O-EMB, duckdb (if adopting V3 aggregation)

## 8. Execution environments
| Stage | Env |
|---|---|
| Unit tests | Local/Cloud CPU |
| A graph / B normalize | Colab/Cloud CPU, Drive I/O |
| Embeddings | Colab GPU |
| Train/eval | Colab GPU |

## 9. Implementation order & unblock status

| Order | Module | Unblocked now? |
|---|---|---|
| 1 | Harden node map to D2 + certify | **Yes** (after confirm output root) |
| 2 | Dataset A event/aggregate (align V3 + D2) | **Mostly yes**; O-DEDUP/O-CAL recommended; QREL-01/QSELF-01 frozen |
| 3 | Dataset B full normalize (scale pilot) | **Mostly yes**; O-CAL shared |
| 4 | Features/masks | Blocked on O-FEAT, O-CAL |
| 5 | Embeddings | **Blocked** on O-EMB (+ O-TEXT) |
| 6 | Edge text embed | Blocked on O-EMB, O-EDGE |
| 7 | Tensor pack | Blocked on 2–6 |
| 8 | Model/losses/train | **Blocked** on tensors + O-HPARAM |
| 9 | Eval/evolution | Blocked on trained outputs |

## 10. Scope for next implementation task
- Implement only modules defined in TDMEC contracts (`03`–`07`)
- Do not expand node universe beyond N = 16,736 (D2)
- Do not finalize the exact Qwen3 checkpoint/config before the Q-EMB pilot (family is user-confirmed Qwen3-only; config `PENDING_PILOT`)
- Do not claim certified artifacts without checksums
- Do not write embedding/model code before open decisions are closed for those stages
