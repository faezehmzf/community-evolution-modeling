# 02 — Model Input Contract (per snapshot)

**Purpose.** Define the per-snapshot tensor inputs TDMEC would consume, with an
explicit producer, dtype, shape, missing-value behavior, validation rule,
consuming module, and required variants for each. **Shapes are given only where
supported by evidence or confirmed decisions.** Every dimension that depends on an
unresolved scientific setting uses a **named symbolic dimension** and is marked
accordingly.

> **Contracts.** Structural dims follow Q-FEAT / Q-WGT (`docs/method/12`, `14`).
> Text dims follow Q-TEXT; `D_text` is `PENDING_PILOT` under Q-EMB (`docs/method/16`).
> Missing-text behavior is **Q-MISS M1** (`USER_CONFIRMED_CANONICAL`, `docs/method/17`):
> exact all-zero vector + boolean availability mask. Primary `d_h`/`K` are
> **USER_CONFIRMED** (Batch 4: QHP-01/QHP-03); `D_text` remains POST_PILOT.

## Symbolic dimensions

| Symbol | Meaning | Value | Status |
|---|---|---|---|
| `N` | frozen node count | **16,736** | VERIFIED_FROM_REAL_DATA / D2 |
| `T` | snapshot count | **35** provisional (quarterly, 2017-Q4…2026-Q2) | Q-CAL frequency confirmed; bounds O-CAL review-pending |
| `R` | relation types | **4** (mention, retweet, reply, quote) | VERIFIED (types) / codes **USER_CONFIRMED_CANONICAL (QREL-01)** |
| `E_t` | directed edges in snapshot `t` | data-dependent | not yet computed |
| `F` / `F_struct` | structural node-feature dim | **17** | Q-FEAT `USER_CONFIRMED_CANONICAL` |
| `D_text` | text-embedding dim | **PENDING_PILOT** (Qwen3 Embedding family USER_CONFIRMED; config per Q-EMB `docs/method/16`) | family `USER_CONFIRMED_QWEN3_ONLY`; config `PENDING_PILOT` |
| `H` / `d_h` | model hidden dim | **64** primary (QHP-01); sensitivity `{32,128}` | `USER_CONFIRMED` (Batch 4) |
| `K` | community count | **10** primary (QHP-03); sensitivity `{5,10,15,20,30}` | `USER_CONFIRMED` (Batch 4) |

## Per-snapshot inputs

| Input | Producer dataset | Producer stage | Meaning | dtype | Shape | Missing-value behavior | Validation rule | Consuming module | Required variants |
|---|---|---|---|---|---|---|---|---|---|
| `node_index` | A (node map) | A-06 build map | contiguous account id | int32 | `[N]` (fixed) | none (universe frozen) | contiguous 0..N-1; count = 16,736 | all | G, NT, ET |
| `struct_active_mask` | A | A-08 aggregate | node structurally active in `t` (Q-FEAT) | bool | `[N]` | inactive ⇒ False | True iff tweet_count_raw>0 OR ≥1 in OR ≥1 out edge | temporal / gating | G, NT, ET |
| `edge_index` | A | A-07 edge extract | directed `(src,dst)` in `t` | int64 | `[2, E_t]` | empty snapshot ⇒ `E_t=0` | endpoints ∈ 0..N-1; no NaN; self-loops excluded | message passing | G, NT, ET |
| `edge_relation` | A | A-07 | relation code per edge | int8 | `[E_t]` | — | ∈ {0,1,2,3}; codes **QREL-01** (`0=mention…3=quote`) | relation fusion | G, NT, ET |
| `count_raw` / `edge_weight_raw` | A | A-08 | distinct-event count per `(src,dst,rel)` after Q-DEDUP | int64 | `[E_t]` | — | ≥ 1 | audit / ablations | G, NT, ET |
| `weight_log1p` / `edge_weight_transformed` | A | A-08 | canonical weight `log(1 + count_raw)` (Q-WGT) | float32 | `[E_t]` | — | `weight_log1p = log1p(count_raw)`; finite | message passing | G, NT, ET |
| `node_structural_features` / `X_struct` | A | A-08 | per-node structural features in `t` | float32 | `[N, F_struct]` — **F_struct = 17** | inactive nodes remain in `V` with zeros per schema; activity via `struct_active_mask` | finite; schema order per `docs/method/14` | initial node encoder | G, NT, ET |
| `node_text_features` / `X_node_text` | B | B-text embed | node-snapshot mean-pooled text embedding (Q-TEXT) | float32 | `[N, D_text]` — **D_text PENDING_PILOT** | no-text-in-`t` ⇒ **exact all-zero** (Q-MISS M1); do not L2-norm missing zeros | encoder config per Q-EMB; mask↔vector alignment | Fusion MLP (with mask) | **NT, ET** |
| `text_availability_mask` / `node_text_available_mask` | B | B-text embed | node has valid text in `t` | bool | `[N]` | False ⇒ zero vector | True iff `node_valid_text_count>0`; separate from `struct_active_mask` | Fusion MLP / `L_sem` gating | NT, ET |
| `node_valid_text_count` | B | B-text embed | # valid cleaned tweets in mean | int | `[N]` | 0 when missing | ≥0; metadata only (not a model feature) | coverage / mean verification | NT, ET |
| `edge_text_features` | **A** (event tweet text) | edge-text build | per-edge mean-pooled embedding (Q-TEXT) | float32 | `[E_t, D_text]` — **D_text PENDING_PILOT** | no edge text ⇒ **exact all-zero** (Q-MISS M1); edge stays; `count_raw`/`weight` unchanged | A-derived only; no A↔B tweet join; aligned to canonical edge order | Edge Gate (with mask) | ET |
| `edge_text_available_mask` | A | edge-text build | edge has valid text | bool | `[E_t]` | False ⇒ zero vector; structural path preserved | True iff `edge_valid_text_count>0`; separate from struct masks | Edge Gate / `L_sem` | ET |
| `edge_valid_text_count` | A | edge-text build | # valid event texts in mean | int | `[E_t]` | 0 when missing | ≥0; metadata only (not a model feature) | coverage / mean verification | ET |
| `temporal_hidden_state` | model | recurrent | carried node state `h_{t-1}` | float32 | `[N, d_h]` — **`d_h=64` primary (QHP-01)** | `t=0` ⇒ zeros / standard GRU init | finite | temporal encoder | G, NT, ET |
| snapshot metadata | A/B | A-05/B-09 | `snapshot_id`, quarter label, counts | int/str | scalar/`[T]` | — | `snapshot_id` in provisional 0..34 until O-CAL freezes | bookkeeping | all |

## Notes and hard constraints (VERIFIED / CONFIRMED)

- **`node_index` is immutable** once the map is built; the node universe must
  **never** be expanded or renumbered (`configs/dataset_b_pilot.yaml` →
  `node_universe_expansion: forbidden`; `src/tdmec_pilot/node_map.py`). **D2:**
  N = 16,736; 10,040 is not the modeled universe.
- **Edges come only from Dataset A** (Dataset B has no edge fields —
  `docs/data/04`,`05`). Any edge tensor sourced from B is invalid.
- **Q-WGT:** compute `weight_log1p`; retain `count_raw`. **Q-FEAT:** `F_struct = 17`
  + `struct_active_mask`. **Q-TEXT:** A = edge text, B = node text; mean-pool.
  **Q-MISS M1:** exact zero + boolean availability masks; Fusion MLP / Edge Gate
  receive masks explicitly; valid-text counts = required metadata not features;
  `L_sem` only where mask=True (`docs/method/17`).
- **Still open (evidence-only; do not invent):** Q-EMB X01–X07 (`D_text`, final
  instruction, …); O-CAL exact calendar bounds; coverage hard thresholds;
  runtime batch/AMP/OOM.
- `relation_id` codes are **USER_CONFIRMED_CANONICAL (QREL-01)**:
  `0=mention,1=retweet,2=reply,3=quote` (must be stored explicitly & validated;
  register G4; Q-FEAT channel order matches).

## Variant applicability summary (**QVAR-01**)

- **TDMEC-G:** needs A-derived structural tensors
  (`node_index`, `struct_active_mask`, `edge_index`, `edge_relation`,
  `count_raw`, `weight_log1p`, `X_struct`, `temporal_hidden_state`). Structural
  contracts are confirmed (Q-WGT, Q-FEAT); implementation + certification remain.
- **TDMEC-NT:** additionally needs `node_text_features` + `node_text_available_mask`
  (+ `node_valid_text_count` metadata) → blocked on Q-EMB pilot only (Q-MISS closed).
- **TDMEC / TDMEC-Full (primary):** needs TDMEC-NT inputs plus `edge_text_features` +
  `edge_text_available_mask` (+ `edge_valid_text_count` metadata). Edge text is
  **in scope (Q-TEXT)** and A-derived; blocked on Q-EMB pilot only.
  Tweet-level A↔B join is prohibited.
- **TDMEC-ET:** reserved name for a graph+edge-text-only ablation **if implemented**;
  do **not** use as the full-method label.
