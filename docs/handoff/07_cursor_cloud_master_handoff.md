# 07 — Cursor Cloud Master Handoff (Self-Contained)

**Audience:** Cursor Cloud agent.  
**Language:** English.  
**Do not access** `C:\Users\mrsmo\Documents\UT\Semester 6\Thesis\Proposed`. All method content needed for implementation decisions is summarized here and in `docs/method/03`–`11`.

---

## 1. Current canonical method (TDMEC)

**TDMEC** = Temporal Directed Multiplex Edge-Conditioned Community Detection.

Discover and track soft communities among a **fixed** set of Core ARMY accounts using:
1. Directed multiplex interaction structure (retweet, reply, quote, mention).
2. Leakage-safe temporal node text (snapshot-local).
3. Lightweight edge-text conditioning of message passing.

Pipeline: Dataset A → directed multiplex graph + edge text; Dataset B → node text; Edge-gated directed GraphSAGE per relation → masked relation fusion → GRU → prototype Student-t soft assignments → hierarchical losses → trajectories.

## 2. Authoritative-source hierarchy

1. User-confirmed decisions (D1, D2 below).  
2. Canonical docs in this repo: `docs/method/03_current_tdmec_canonical_spec.md` and siblings.  
3. Verified data contracts: `docs/data/*`, `docs/handoff/00`–`06`, `artifacts/discovery/*`.  
4. Current code: `src/tdmec_pilot/`, `src/tdmec_discovery/`, scripts, tests.  
5. External Proposed sources are **out of Cloud scope**; do not require them.

## 3. Implementation scope

Implement only modules defined in the TDMEC contracts (`docs/method/03`–`07`). Modeled universe remains **N = 16,736** under D2.

## 4. User-confirmed decisions

| ID | Binding |
|---|---|
| **D1** | Primary scientific authority is TDMEC Methodology (P-001); Complete Technical Explanation (P-002) is supporting only; data-construction ideas may be used when compatible with method + verified data + D2 |
| **D2** | **N = 16,736**; indices **0…16735**; immutable map; Dataset B must not create/append/renumber nodes; 10,040 is not the modeled universe |
| **Q-CAL** (partial + PROC) | Temporal frequency = **quarterly** (canonical); monthly = sensitivity variant only. Provisional pilot calendar = **35 bins, 2017-Q4→2026-Q2**, **diagnostic-only**; empty **internal** snapshots kept; calendar **config-driven, not hard-coded**. Out-of-range records excluded with explicit reason codes + reported; invalid/corrupt/epoch-outlier timestamps classified separately. **QCAL-B01-PROC confirmed (2026-07-28):** leading/trailing empty + final start/end/`T` require user approval after coverage report; **no full-data artifact may be `calendar-certified` before that approval.** Exact numerical boundaries remain `REVIEW_REQUIRED_AFTER_REAL_DATA_DIAGNOSTICS`. |
| **Q-WGT** | Edge identity `(snapshot_id, relation_id, source_idx, target_idx)`; `count_raw` = distinct events after Q-DEDUP; **`weight_log1p = log(1 + count_raw)`** (stable natural-log `log1p`) is the primary model edge weight; keep `count_raw`. Directed; relations separate; self-loops excluded+reported; multi-target mention events expand to one event per valid target. Edge fields: `snapshot_id, relation_id, source_idx, target_idx, count_raw, weight_log1p` + provenance/validation. Ablations: binary, raw count, `log1p` (main). Diagnostics validate distributions/hub concentration/correctness only. |
| **Q-TEXT** | **Node text = Dataset B; edge text = Dataset A event text** (not interchangeable; tweet-level A↔B join **prohibited**). **Node:** atomic unit = one distinct authored tweet; `z_node_q=TextEncoder(cleaned_text_q)`; `T_i^(t)=mean_{q∈Q_i^(t)} z_node_q` over B tweets by i in snapshot t → `X_node_text[T,N,D_text]`. **Edge:** atomic unit = one distinct interaction-event tweet; `E_{i→j}^{(t,r)}=mean_{q∈Q_{i→j}^{(t,r)}} z_edge_q` per canonical edge (relations/directions/snapshots separate; single-event edge = that embedding; multi-target mention shares cleaned text across target events with single-tweet provenance + caching). Input = **`cleaned_text` only** (Q-DEDUP L2); dedup before aggregation; cache keyed by content+provenance (not float A id). **Strict in-snapshot** temporal rule. Masks `node_text_available_mask`, `edge_text_available_mask` — exact missingness semantics under **Q-MISS**. Edge gate **learns** how edge text modifies structural messages (no hard-coded semantic rules). Per-tweet embed→mean-pool canonical (other pooling = future ablation only). **Node-text path first, edge-text second — both required.** `D_text`/encoder → Q-EMB. Full contract `docs/method/15`. |
| **Q-MISS** | **M1** (`USER_CONFIRMED_CANONICAL`, `docs/method/17`): exact all-zero `D_text` vector + boolean availability mask for node (B) and edge (A) text; no learned missing embedding; no drop nodes/edges for missing text; no text carry-forward; Fusion MLP and Edge Gate receive masks explicitly; missing edge text preserves structural path (`count_raw`/`weight` unchanged); `L_sem` only where mask=True (zero-eligible → defined zero); separate `struct_active_mask` / `node_text_available_mask` / `edge_text_available_mask`; `node_valid_text_count` / `edge_valid_text_count` = required metadata not model features; privacy-safe coverage reports required. Artifacts: `X_node_text[T,N,D_text]`, `node_text_available_mask[T,N]`, `node_valid_text_count[T,N]`; edge vectors/masks/counts aligned to canonical edge order. |
| **Q-FEAT** | **`F_struct = 17`.** `X_struct[t,node_idx,feature_idx] ∈ ℝ^{T×N×17}` float32, all 16,736 frozen nodes every snapshot. Relation order `0=mention,1=retweet,2=reply,3=quote`. Ordered schema: per relation {`out_degree`, `in_degree`, `out_strength_log1p`, `in_strength_log1p`} (0–15) + `16 tweet_count_log1p`. **Degree** = distinct-neighbor count (`|{j:c>0}|`, no log1p); **strength** = `log1p(Σ_j count_raw)` (NOT `Σ log1p`, NOT sum of edge weights); **tweet_count** = `log1p(#distinct authored tweets after Q-DEDUP)`, counted once regardless of edges (originals/out-of-universe/multi-target all count once). `c_{i→j}^{(t,r)}`=count_raw; self-loops excluded before degree/strength; multi-target mention = one event per valid target. `active` and `n_active_relations` are **NOT** features. Separate boolean **`struct_active_mask[t,node] ∈ {0,1}^{T×N}`** = True iff `tweet_count_raw>0` OR ≥1 incoming OR ≥1 outgoing canonical edge (Dataset-A structural activity only; relation to text/model/GRU masks decided later). Excluded: engagement, follower/account metadata, identifiers, duplicate flags, centrality (PageRank/HITS/k-core/clustering/common-neighbors), relation entropy/shares, recency, balance, reciprocity, active-days, media, hashtags. Ordering versioned + in artifact contract; never silently reordered. Additional normalization → **QHP-02** train-time robust scaling (train-only; do not standardize canonical artifacts). Artifacts: `X_struct`, `struct_active_mask`, ordered 17-name list, schema version, relation+snapshot maps, node-map hash, Q-DEDUP provenance, dtype/shape, validation summary. |
| **Q-DEDUP** (policy + PROC) | Two layers. **Records:** conflict-aware collapse; Dataset A via deterministic composite signature (**float tweet id forbidden as key**), Dataset B via exact string IDs; duplicate = same user+timestamp+content+compatible relation/target; **preserve** identical text across different users and (normally) same-user different-timestamp; concordant→one distinct event + provenance + copy count; discordant→retained+flagged; **graph event count = distinct events, not raw rows**. **In-field text:** conservatively remove only extraction-duplicated blocks (exact full-text/long consecutive spans); keep original text **private** + separate **cleaned-text** field + hashes; **embed cleaned text**; never remove meaningful/campaign repetition; never log full text in Git-visible manifests. **QDEDUP-B01-PROC confirmed (2026-07-28):** no full-data edge/text `CERTIFIED` before privacy-safe duplicate diagnostic + explicit user approval of A signature and L2 thresholds; aggressive fuzzy prohibited; raw unchanged; effects on counts/coverage must be reported. Exact signature/thresholds/final numbers remain `REVIEW_REQUIRED_AFTER_REAL_DATA_DIAGNOSTICS`. |
| **QEMB-I1** | Shared pilot instruction wording `USER_APPROVED_PILOT_CANDIDATE` (2026-07-28): “Represent the topic, stance, sentiment, and social meaning of this social-media post for temporal community analysis.” Pilot comparison vs no-instruction only; **not** final production instruction (QEMB-X03 remains POST_PILOT). |
| **QART-01-FRAME** | Certification framework `USER_CONFIRMED_CANONICAL` (2026-07-28): contract-correctness = mandatory hard gates; text/activity coverage always reported and normally warning until numeric thresholds approved; complete absence of primary-method required artifact/semantic path = hard fail (graph-only ablation may be certified separately); no `CERTIFIED` without schema/shape/dtype/order/relation-map/self-loop/alignment/exact-zero/`model_active`/NaN-Inf/checksum/manifest/provenance/deterministic ordering/resume/double-count checks. Numeric coverage thresholds remain open. |
| **Batch 4 architecture** | **QFUS-01:** `z=h^(0)` if no relations; never overwrite `struct_active_mask`. **QENC-01:** directed mean agg (immutable); fanout `[15]` primary training default. **QENC-02:** `L=1` primary; `L=2` sensitivity. **QPROJ-01:** `d_rel=16`, `d_sem=d_h`, `P_z`/`P_T_node`; edge text own MLP_e projection. **QHP-01:** `d_h=64`; sens `{32,128}`. **QHP-02:** train-time robust scaling (train-only). **QHP-03:** `K=10`; sens `{5,10,15,20,30}`. **QHP-04:** KMeans++ on training active `s`; `n_init=20`. **QVAR-01:** TDMEC-G / TDMEC-NT / TDMEC\|TDMEC-Full; TDMEC-ET reserved. Full detail `docs/method/05`, `12`. |
| **Batch 5 loss/training** | **QLOSS-01:** 15% mask, 3 negatives. **QLOSS-02:** `m=1.0` on ‖μ‖². **QLOSS-03:** `L_temp` iff both `model_active`. **QLOSS-04:** λ 1/1/1/0.1/0.1 + staged + 20% ramp. **QTR-01:** AdamW 5e-4 / wd 1e-4 / clip 1.0; 100/200 epochs; patience 20. **QTR-02:** ~70/15/15 chronological (exact bounds POST_CAL). **QTR-03:** BPTT=3. **QTR-04:** 5-seed contract. Full detail `docs/method/06`, `12`. |
| **Batch 6–7 (architect-resolved)** | QEVAL-01 metrics+selection (`18`); QEVAL-02 baseline registry (`19`, implement Phase 10); QEVAL-03 ablation registry (`20`, Phase 11); QMLP/QDEC/QCLU/QBATCH/QPHASE defaults in `05`/`06`. Consolidated contract `21`. **No further conceptual user questions.** |

## 5. Verified Dataset A physical state

- 12 Excel + `extraction_summary.json`; ≈12,581,535 rows; 31 columns.  
- Exactly **16,736** distinct `user.id`.  
- Tweet extract (filtered/partially preprocessed), **not** graph-ready.  
- Relations derivable from structured fields.  
- Tweet `id` **float-lossy** — never exact cross-dataset key.  
- ~1.17M duplicates flagged, 0 removed in raw; **Q-DEDUP policy now confirmed** (conflict-aware collapse; distinct-event counts; final numbers pending diagnostics).  
- Several columns 100% null.  
- Graph artifacts not certified.

## 6. Verified Dataset B physical state

- 70 `statuses-*.xlsx` + manifest; ≈9.58 GB; 10-column RAW schema.  
- Exact string tweet IDs; authored `text`; timestamps; engagement; serialized `user` with `user.id`.  
- Parse `user` as JSON or Python-literal — **never `eval`**.  
- **No** mention/reply/retweet/quote edge target fields → **not** the multiplex graph source.  
- Must not expand N.

## 7. Current pilot status

- Two files only: `statuses-2.xlsx`, `statuses-69.xlsx`.  
- Code under `src/tdmec_pilot/`, `scripts/run_dataset_b_pilot.py`, `scripts/build_node_index_map.py`, `configs/dataset_b_pilot.yaml`, tests.  
- Real run: in 2,004,845; retained 1,992,014; excluded 12,831; rejected 0; authors 496 matched / 0 unmatched; exact dup groups 747; conflicting 0.  
- Tests: 31 passed.  
- **Not** the full 70-file pipeline; no embeddings; no certified full B corpus.

## 8. Current implementation status

| Area | Status |
|---|---|
| Discovery tooling | Implemented + used |
| Dataset B 2-file pilot | Implemented + tested + executed |
| Certified node map | **Missing** |
| Certified A edges/snapshots | **Missing** |
| Full B normalize | **Not run** |
| Embeddings | **Missing / blocked** |
| TDMEC model/train/eval | **Not in repo / blocked** |

## 9. Current artifact status

**No certified model-ready artifact was found in the repository or verified persistent storage.**  
Distinguish: code exists ≠ pilot ran ≠ temp local ≠ persistent Drive ≠ certified.

## 10–11. Full Dataset A / B pipelines

See `docs/method/04_complete_data_to_model_pipeline.md` for stage IDs A00–A10 and B00–B09.

Cloud must treat:
- A: extract 4 directed relations → aggregate → log1p weights → features/masks → A-derived edge text.  
- B: reconcile to frozen map → snapshots → normalize → text units → embed → pool → masks.  
- Pilot ≠ full B.

## 12. Graph–text alignment

- Reliable: account-level via exact `user.id` + `node_idx` + `snapshot_id`.  
- Not proven: tweet-level A↔B join.  
- Do not reconstruct A tweet ID as exact integer.  
- B-sourced per-edge text requiring A↔B tweet join: **unsupported**.  
- A-derived edge text: method-supported.

## 13. Complete model architecture (summary)

`h^(0)=MLP_x([X,T,masks])` → per-relation edge-gated directed GraphSAGE → masked fusion → GRU (freeze inactive) → Student-t prototypes (α=1) → Q, hard, conf, entropy.

Details/shapes: `docs/method/05_architecture_and_tensor_contract.md`.  
**Confirmed:** N=16736, R=4, **F_struct=17 (Q-FEAT)**, **primary `d_h=64` (QHP-01)**, **primary `K=10` (QHP-03)**, **`d_rel=16`**, **`L=1`**, fanout `[15]`. Still symbolic/evidence-dependent: exact `T` (calendar bounds), `D_text` (POST_PILOT).

## 14. Tensor contracts

Symbolic shapes in `docs/method/05`. Never allocate dense `[T,R,N,N,D]` edge-text tensors.

## 15. Losses

`L_struct + L_sem + L_cluster + L_reg + L_temp` (hierarchical). See `docs/method/06`. Primary λ **USER_CONFIRMED** (Batch 5): 1 / 1 / 1 / 0.1 / 0.1 with staged phases + temporal ramp (QLOSS-04, QPHASE-01).

## 16. Training stages

Precompute embeddings → pretrain struct+sem (≤100) → KMeans++ init prototypes → joint (80, `λ_temp=0`) → temporal (120 with ramp over first 24) → 5 seeds → save outputs. Defaults: AdamW 5e-4, BPTT=3, ~70/15/15 chronological (exact quarter counts POST_CAL).

## 17. Inference outputs

Per-snapshot soft Q, hard labels, confidence, entropy, trajectories, optional β attention.

## 18. Evaluation strategy

Convergent validity across structural/temporal/semantic/robustness; baselines and ablation ladder in `docs/method/07`. No ground-truth required.

## 19–20. Risks

See `docs/method/08`. Critical: temporal leakage, masked-edge leakage, float tweet_id misuse, claiming uncertified artifacts.

## 21. Unresolved decisions

See `docs/method/10` and `21`. **All conceptual decisions closed.** Remaining: diagnostics (calendar/dedup/coverage), Q-EMB pilot, runtime probes. Next engineering step after operational authorization: **Phase 1** schemas/config/invariant tests. Baselines/ablations only after Phase 9.

**Q-DEDUP policy resolved.** Two-layer conflict-aware dedup (records via composite signature / exact-ID; conservative in-field text cleaning). Only exact signature, span thresholds, and final counts remain `REVIEW_REQUIRED_AFTER_REAL_DATA_DIAGNOSTICS`. Implement two text fields (original private + cleaned) and provenance/hashes; never emit full text to Git-visible manifests.

**Q-CAL frequency resolved (quarterly).** Remaining **O-CAL** = exact boundaries + tail policy only, gated on the real-data coverage report (`REVIEW_REQUIRED_AFTER_REAL_DATA_DIAGNOSTICS`). Implement the calendar as a runtime config (never hard-code 2017-Q4/2026-Q2). Diagnostic runs may use the provisional 35-bin calendar; do not stamp any full-data artifact `calendar-certified` before user boundary confirmation.

**Q-EMB — embedding family resolved; config pilot-pending.** Encoder family = **Qwen3 Embedding only** (`USER_CONFIRMED`); preferred checkpoint `Qwen/Qwen3-Embedding-4B` (`PROVISIONAL_PENDING_PILOT`). `D_text`, instruction, token-level pooling, per-tweet/final L2 normalization (N1 vs N2), and `max_length` are `PENDING_PILOT_AND_USER_CONFIRMATION`. Stage-B cross-tweet mean-pool is canonical (Q-TEXT). Cloud must **not** name/introduce any non-Qwen embedding model, must **not** download the model or run embeddings, and must not propagate any provisional value into a final tensor contract. Full contract + bounded pilot spec: `docs/method/16`.

**Q-MISS resolved (M1).** Exact all-zero embedding + boolean availability mask for node and edge text; Fusion MLP / Edge Gate receive masks; structural path preserved on missing edge text; `L_sem` mask=True only; valid-text counts = metadata. Full contract: `docs/method/17`.

## 22. Tasks Cloud may implement now

- Review/harden node-map builder to enforce D2 (N=16736).  
- Unit tests for IDs, self-loops, relation parsing (synthetic).  
- Dry-run validation of configs/paths.  
- Documentation/checklist updates.  
- Controlled pilots **after** local code lands and open decisions for those stages are closed.

## 23. Tasks Cloud must not implement yet

- Full embeddings (O-EMB open).  
- TDMEC model training.  
- Claiming certified Drive artifacts without checksum gates.  
- Expanding node universe.  
- Modules outside the TDMEC contracts.  
- Speculative modules for unresolved scientific choices.

## 24. Exact modules that should eventually be created

See `docs/method/11_implementation_blueprint.md` (`src/tdmec/...`).

## 25. Exact tests required

See blueprint §6: ID safety, N=16736, relations, snapshots, B reconcile, TDMEC-contract module scope, leakage tests, later shape/loss tests.

## 26. Exact Colab execution sequence

See `docs/handoff/09_colab_execution_runbook.md`.

## 27. Persistent storage layout

Default root: `/content/drive/MyDrive/TDMEC_PROJECT_OUTPUTS/`

```
TDMEC_PROJECT_OUTPUTS/
  runs/<run_id>/
    config/          # frozen yaml + hashes
    manifests/       # checksums, git SHA
    nodes/           # node_index_map.parquet
    snapshots/
    graph/           # edges, features, masks
    text_b/          # normalized + embeddings
    edge_text/
    tensors/
    model/
    eval/
    logs/
```

## 28. Validation and acceptance gates

Fail-closed if:
- node count ≠ 16736  
- schema drift  
- float ID reconstruction attempted  
- unmatched B authors added as nodes  
- checksum mismatch  
- missing eligibility/mask fields for text stages  

## 29. Status vocabulary

`CODE_EXISTS` | `TESTS_EXIST` | `PILOT_EXECUTED` | `TEMP_LOCAL` | `PERSISTENT_DRIVE` | `CERTIFIED` | `BLOCKED` | `USER_CONFIRMATION_REQUIRED` | `METHOD_SPEC_REQUIRED` | `INCORRECT_PRIOR_CLAIM`

## 30. Provenance requirements

Every published artifact must record: Git SHA, config hash, source path URIs, row in/out/excluded counts, exclusion reasons, timestamp UTC, random seeds (if any), checksum SHA256.
