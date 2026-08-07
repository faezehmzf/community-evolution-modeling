# 14 — Implementation Status Matrix

Per-component evidence status + implementation maturity. Cross-links:
`docs/handoff/00`–`06`.

**Evidence status:** `VERIFIED_FROM_REAL_DATA` · `VERIFIED_FROM_CURRENT_METHOD` ·
`VERIFIED_FROM_IMPLEMENTATION` · `INCORRECT_PRIOR_CLAIM` · `PILOT_ONLY` · `INFERRED` · `PROPOSED`
· `USER_CONFIRMATION_REQUIRED` · `METHOD_SPEC_REQUIRED` ·
`BLOCKED_BY_DATA_LIMITATION`. `METHOD_SPEC_REQUIRED` = undecidable until the
method spec is imported (not a data limitation); `BLOCKED_BY_DATA_LIMITATION` is
reserved for items the datasets genuinely cannot support.

**Maturity:** `DEFINED` · `IMPLEMENTED` · `UNIT_TESTED` · `SMOKE_TESTED` ·
`PILOT_VALIDATED` · `FULL_ARTIFACT_CERTIFIED` · `SCIENTIFICALLY_EVALUATED` ·
`BLOCKED`.

| Component | Evidence status | Maturity | Repository evidence |
|---|---|---|---|
| Dataset A discovery | VERIFIED_FROM_REAL_DATA | FULL_ARTIFACT_CERTIFIED (discovery) | `docs/data/00,01,02`; `artifacts/discovery/dataset_a_schema_registry.json` |
| Dataset A normalization | PROPOSED | DEFINED | `docs/data/03`; reuse `src/tdmec_pilot/{user_blob,identifiers,schema}.py` (not yet applied to A) |
| Node mapping | VERIFIED_FROM_IMPLEMENTATION (builder) / PROPOSED (artifact) | IMPLEMENTED + UNIT_TESTED (builder); artifact NOT_CERTIFIED | `scripts/build_node_index_map.py`; `src/tdmec_pilot/node_map.py`; `tests/test_pilot.py` |
| Relation extraction | VERIFIED_FROM_REAL_DATA (fields) / PROPOSED (build) | DEFINED | `docs/data/02,03` (fields + measured counts); no builder |
| Dataset A dedup | USER_CONFIRMED_CANONICAL (policy, Q-DEDUP) / numbers REVIEW_REQUIRED | DEFINED | `docs/data/02`; `docs/method/12` (Q-DEDUP); `src/tdmec_pilot/dedup.py` |
| Snapshot construction | USER_CONFIRMED_CANONICAL (quarterly, Q-CAL) / boundaries REVIEW_REQUIRED | IMPLEMENTED + UNIT_TESTED (pilot) | `docs/data/03,06`; `docs/method/12` (Q-CAL); `src/tdmec_pilot/snapshots.py` |
| Graph aggregation & edge weight | USER_CONFIRMED_CANONICAL (Q-WGT: `weight_log1p=log(1+count_raw)`) | DEFINED | `docs/method/12` (Q-WGT); no builder |
| Structural node features `X_struct[T,N,17]` + `struct_active_mask[T,N]` | USER_CONFIRMED_CANONICAL (Q-FEAT, F_struct=17) | DEFINED (build after edges+node-map) | `docs/method/12` (Q-FEAT); `docs/method/14`; no builder |
| Graph model-ready publication | PROPOSED | DEFINED | `docs/handoff/03` A-10; nothing on disk (register A9) |
| Dataset B discovery | VERIFIED_FROM_REAL_DATA (12/70) / BLOCKED_BY_DATA_LIMITATION (58/70) | PILOT_VALIDATED (sample) | `docs/data/04,05`; `artifacts/discovery/dataset_b_schema_registry.json` |
| Dataset B normalization | VERIFIED_FROM_IMPLEMENTATION | PILOT_VALIDATED (2 files) | `src/tdmec_pilot/pipeline.py`; `docs/data/08` |
| Dataset B dedup | VERIFIED_FROM_IMPLEMENTATION (annotate) | PILOT_VALIDATED (2 files) | `src/tdmec_pilot/dedup.py`; `docs/data/08` |
| Cross-dataset reconciliation | VERIFIED_FROM_REAL_DATA (sample) / UNRESOLVED (full 70) | PILOT_VALIDATED (sample) | `docs/data/06`; `docs/data/08` |
| Text-unit construction | USER_CONFIRMED_CANONICAL (Q-TEXT) | DEFINED (no builder) | node=B distinct tweet→mean-pool `T_i^(t)`; edge=A event tweet→mean-pool `E_{i→j}^{(t,r)}`; cleaned_text; in-snapshot; `docs/method/15` |
| Text sampling | not required by Q-TEXT | OPEN if later needed | do not invent sampling |
| Text embedding | family USER_CONFIRMED_QWEN3_ONLY; config PENDING_PILOT (Q-EMB) | BLOCKED (no pilot auth / libs) | `docs/method/16`; `requirements.txt` |
| Text aggregation | USER_CONFIRMED_CANONICAL (Q-TEXT: mean-pool) | DEFINED (with embeddings) | cross-tweet/event mean-pool; encoder pooling → Q-EMB |
| Missing-text / availability masks | USER_CONFIRMED_CANONICAL (Q-MISS M1) | DEFINED (build with embeddings) | exact zero + boolean mask; counts=metadata; `docs/method/17` |
| Node-text publication | Q-TEXT/Q-MISS confirmed; Q-EMB open | BLOCKED on Q-EMB pilot | `docs/method/15`, `16`, `17` |
| Edge-text construction | USER_CONFIRMED_CANONICAL (Q-TEXT: A-derived, required) | DEFINED (build second, after node-text) | `E_{i→j}^{(t,r)}` = mean of per-event embeddings from A event `cleaned_text`; no A↔B tweet join; Q-MISS M1 for missing; `docs/method/15`, `17` |
| TDMEC-G | Features CANONICAL (Q-FEAT F_struct=17); model code not implemented | BLOCKED (model code; after Phase 1+) | structural input fixed; **QVAR-01** ablation name |
| TDMEC-NT | Q-TEXT/Q-MISS confirmed; Q-EMB open | BLOCKED | **QVAR-01** ablation name; needs embedding pilot |
| TDMEC / TDMEC-Full | Primary method (QVAR-01); Q-TEXT/Q-MISS confirmed; Q-EMB open | BLOCKED | primary names; edge+node text required |
| TDMEC-ET | Reserved graph+edge-text-only ablation **if implemented** (QVAR-01) | BLOCKED / optional | never name the full method ET |
| Baselines | Registry CLOSED (`docs/method/19`); code deferred to **Phase 10** | BLOCKED until Phase 9 complete | no baseline implementation started |
| Ablations | Registry CLOSED (`docs/method/20`); code deferred to **Phase 11** | BLOCKED until Phase 9–10 | no ablation implementation started |
| Evaluation | Contract CLOSED (`docs/method/18`); code deferred to Phase 8 | BLOCKED (evaluator not implemented) | selection = val relation-macro AP |

## Summary

- **Certified now:** read-only discovery of both datasets; Dataset B 2-file pilot.
- **Next (Cursor Cloud Phase 1 only):** typed configuration contracts, artifact/tensor schemas, validation utilities, synthetic fixtures, invariant unit tests (`docs/method/21`). **No Phase 1 implementation in the local documentation session.**
- **Later Cloud phases:** diagnostics → Q-EMB pilot → certified artifacts → TDMEC-Full → losses/trainer → evaluator → primary results → baselines (10) → ablations (11) → multi-seed (12).
- **Blocked (evidence-dependent):** text-embedding final config (QEMB-X01…X07); calendar certification (O-CAL); dedup signature/L2; coverage hard thresholds.
- **Blocked (data):** *B-sourced* per-edge tweet text (tweet-level A↔B join
  prohibited); full 70-file corpus statistics.
