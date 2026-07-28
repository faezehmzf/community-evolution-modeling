# 17 — Q-MISS: Missing-Text Contract

**Status:** `USER_CONFIRMED_CANONICAL` (2026-07-26).  
**Policy:** **M1** — exact all-zero embedding vector + explicit boolean availability mask.  
**Scope:** missingness for Dataset B node-snapshot text and Dataset A edge text after Q-TEXT / Q-DEDUP. Encoder dimension/`D_text` and normalization of **available** vectors remain under **Q-EMB**.  
**Does not redefine:** encoder/`D_text` (Q-EMB). **`model_active_mask` / GRU update-carry** are defined under **QACT-01 / QGRU-01** (`docs/method/12`); they remain distinct from text masks.

---

## Canonical status labels
| Item | Status |
|---|---|
| Q-MISS | `USER_CONFIRMED_CANONICAL` |
| Missing node text | `EXACT_ZERO_VECTOR_PLUS_BOOLEAN_MASK` |
| Missing edge text | `EXACT_ZERO_VECTOR_PLUS_BOOLEAN_MASK` |
| Learned missing embedding | `EXCLUDED_FROM_PRIMARY_METHOD` |
| Dropping nodes/edges for missing text | `PROHIBITED` |
| Sparse-only text tensors | `NOT_PRIMARY` |
| Text carry-forward across snapshots | `PROHIBITED` |
| `node_valid_text_count` / `edge_valid_text_count` | `REQUIRED_ARTIFACT_METADATA_NOT_MODEL_FEATURE` |
| Semantic-loss eligibility | `MASK_TRUE_ONLY` |
| Coverage reporting | `REQUIRED` |

---

## 1. Node-text missingness
For node `i` and snapshot `t`:

`node_text_available_mask[t, i] = True` **iff** there exists at least one retained distinct Dataset B tweet that:
- is authored by node `i`;
- has `created_at` inside snapshot `t`;
- survives canonical deduplication;
- has valid non-empty `cleaned_text`;
- is successfully eligible for embedding.

Let `node_valid_text_count[t, i]` = number of valid cleaned tweets that contribute to the node-snapshot mean.

| Condition | Mask | Vector |
|---|---|---|
| `node_valid_text_count[t,i] > 0` | `True` | `T_i^(t) = (sum of valid tweet embeddings) / node_valid_text_count[t,i]` |
| `node_valid_text_count[t,i] = 0` | `False` | `T_i^(t)` = exact all-zero vector of dim `D_text` |

---

## 2. Edge-text missingness
For each existing canonical edge `(snapshot_id, relation_id, source_idx, target_idx)` with index `edge_idx`:

`edge_valid_text_count[edge_idx]` = number of valid cleaned Dataset A event texts contributing to the edge-text mean.

| Condition | Mask | Vector |
|---|---|---|
| `edge_valid_text_count > 0` | `edge_text_available_mask = True` | arithmetic mean of valid event-tweet embeddings |
| `edge_valid_text_count = 0` | `False` | exact all-zero vector of dim `D_text` |

**Structural edge remains present.** Missing edge text must not delete the edge, alter `count_raw`, alter edge weight, or disable the structural message path.

---

## 3. Invalid text → missing
Treat as missing (no embedding, no sum contribution, no count increment, no mean contribution; counted in missingness diagnostics):
- null text; empty string; whitespace-only;
- text that becomes empty after canonical cleaning;
- invalid text rejected by the canonical preprocessing contract.

Do not fabricate replacement text.

---

## 4. Exact-zero invariant
`availability_mask == False` ⇒ embedding vector is **exactly all zeros**, after load, dtype conversion, batching, serialization, and model-input prep.  
**Do not** L2-normalize missing zero vectors. Q-EMB normalization applies only to available non-missing vectors.

---

## 5. Model input
Fusion MLP must receive the node-text availability mask explicitly:

`[ X_struct, X_node_text, node_text_available_mask ]`

Edge Gate must receive the edge-text availability mask explicitly:

`[ edge structural inputs, edge_text_vector, edge_text_available_mask ]`

A zero vector alone does **not** communicate missingness.

---

## 6. Structural fallback (missing edge text)
When `edge_text_available_mask = False`: retain a valid structural message-passing path. Meaning = “no semantic edge evidence,” **not** “edge absent” and **not** “structural message must be zero.” Exact gate architecture may condition the semantic branch on the mask later; structural path must remain.

---

## 7. Semantic-loss eligibility
`L_sem` and text-dependent diagnostics: only where the relevant availability mask is `True`. Use mask-aware denominators. If a batch/snapshot has zero eligible text entries: skip that masked semantic term; return defined zero contribution; record the condition; no division by zero; no NaN/Inf. Missing entries are not semantic negatives or neutral-text examples.

---

## 8. Mask separation
Keep logically and physically separate:
- `struct_active_mask`
- `node_text_available_mask`
- `edge_text_available_mask`

Do not infer one from another. A node may be structurally active without node text, inactive with valid B text, both, or neither. **`model_active_mask = struct_active_mask OR node_text_available_mask` (QACT-01)**; GRU update/carry per **QGRU-01**. Edge-text availability does not define model activity.

---

## 9. No text carry-forward
Do not replace missing `T_i^(t)` with `T_i^(t-1)`, temporal averages, cumulative history, or adjacent/future snapshot embeddings. Text is strictly in-snapshot. GRU hidden-state carry (if any, under its own mask contract) does not change the current snapshot’s text mask.

---

## 10. Valid-text counts (metadata, not features)
Store nonnegative integers:
- `node_valid_text_count[t, i]`
- `edge_valid_text_count[edge_idx]`

Required for mean-pool verification, coverage, resume validation, group-size diagnostics, single- vs multi-text distinction. **Not** primary model features — do not concatenate into Fusion MLP or Edge Gate without a new explicit user decision. Does not alter confirmed `tweet_count_log1p` (Q-FEAT).

---

## 11. Coverage reporting (privacy-safe, required)
**Node:** total node-snapshot pairs; structurally active pairs; pairs with valid text; fraction with text (all / among structurally active); coverage by snapshot; valid-text-count distribution; empty/invalid cleaned-text counts.  
**Edge:** total canonical edges; edges with valid text; fraction with text; by snapshot; by relation; by snapshot×relation; valid-event-text-count distribution; edges with structure but no valid text.  
**Never expose:** raw text, tweet IDs, account IDs, source–target pairs, usernames.

---

## 12. Artifact contract
| Artifact | Shape / alignment | Dtype |
|---|---|---|
| `X_node_text` | `[T, N, D_text]` | per Q-EMB; missing rows exact zeros |
| `node_text_available_mask` | `[T, N]` | boolean |
| `node_valid_text_count` | `[T, N]` | nonnegative integer |
| edge-text vectors | aligned with canonical edge ordering | per Q-EMB; missing = exact zeros |
| `edge_text_available_mask` | same edge ordering | boolean |
| `edge_valid_text_count` | same edge ordering | nonnegative integer |

---

## 13. Required future validations
Missing node/edge → `mask=False` + exact zero; invalid cleaned text excluded from pooling; valid-text count = mean denominator; one valid text → pooled = that embedding; multiple → arithmetic mean; missing edge text does not remove/disable structural edge; `L_sem` excludes missing; zero-eligible batches safe; masks separate; no cross-snapshot text carry; deterministic artifact ordering; count/mask/embedding alignment; privacy-safe coverage reports.
