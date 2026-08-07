# 03 — Current TDMEC Canonical Method Specification

**Status:** Canonical reconstruction under user-confirmed **D1** and **D2**.  
**Language:** English.  
**Scope:** Scientific method end-to-end. Does not authorize implementation of unresolved decisions.

## Authority hierarchy (D1 — user-confirmed)

1. User-confirmed decisions (this document’s D1/D2 bindings).
2. Primary scientific authority: `Proposed/TDMEC_Methodology.md` (P-001).
3. Supporting technical authority: `Proposed/TDMEC_Complete_Technical_Explanation.md` (P-002) — *Source-stated* content only; *Interpretation/Derived* is non-binding and must not override P-001.
4. Verified physical Dataset A/B contracts in `docs/data/`, `docs/handoff/`, `artifacts/discovery/`.
5. Compatible data-construction behavior in `Proposed/temporal_graph_pipeline/` (V3).
6. Current repository implementation/tests (`src/tdmec_pilot/`, `src/tdmec_discovery/`).
7. Recent TDMEC presentation sources (explanatory only).

## User-confirmed decisions

| ID | Decision | Binding |
|---|---|---|
| **D1** | P-001 primary scientific authority; P-002 supporting technical; `temporal_graph_pipeline` data-construction authority when compatible with P-001 + verified data + D2 | Confirmed |
| **D2** | Canonical node universe **N = 16,736**; indices **0 … 16,735**; immutable; Dataset B must not create/append/renumber nodes; historical **10,040** is not the modeled universe | Confirmed |

## Symbolic dimensions

| Symbol | Meaning | Status |
|---|---|---|
| `N` | Node count | **User-confirmed: 16,736** |
| `T` | Number of snapshots | Frequency **USER_CONFIRMED = quarterly** (Q-CAL); exact count/boundaries `REVIEW_REQUIRED_AFTER_REAL_DATA_DIAGNOSTICS`; provisional pilot value = 35 bins (2017-Q4→2026-Q2) |
| `R` | Number of relations | **4** — QREL-01: `0=mention,1=retweet,2=reply,3=quote` (immutable) |
| `F_struct` | Structural feature dim | **User-confirmed: 17** (Q-FEAT) |
| `D_text` | Text embedding dim | Qwen3 Embedding family (user-confirmed); exact value `PROVISIONAL_PENDING_PILOT` (Q-EMB) |
| `H_graph` / `d_h` | Graph / fused / GRU hidden dim | **Primary experimental default `d_h=64` (QHP-01)**; required sensitivity `{32,128}` |
| `H_time` | Temporal state dim | Same as `d_h` |
| `K` | Communities | Fixed per run; **primary `K=10` (QHP-03)**; sensitivity `{5,10,15,20,30}`; never select on test |

---

## S01 — Research problem and scientific objective

**A. Plain language.** Discover and track subcommunities among a fixed set of Core ARMY X/Twitter accounts using directed multiplex interactions, leakage-safe temporal node semantics, and lightweight edge-text context — without ground-truth labels.

**B. Formal.** Given fixed `V` with `|V|=N=16736`, timestamped directed events `(u,v,r,t)`, and tweet histories, produce soft assignments `Q^(t) ∈ [0,1]^{N×K}` per snapshot plus confidence, entropy, and trajectories.

**C. Primary:** P-001 preamble, §1–§2.  
**D. Supporting:** P-002 §1–§2.  
**E/F Dataset roles:** A = interactions/graph/edge text; B = node text only.  
**G/H Artifacts:** none yet certified.  
**I Tensors:** `Q^(t)`, hard labels, confidence, entropy.  
**J Masks:** active / text / relation (defined later).  
**K Assumptions:** fixed universe; relations parseable; unsupervised convergent validity.  
**L Engineering:** Colab/GPU-feasible staged pipeline.  
**M Conflicts:** none for objective after D1.  
**N Data compatibility:** supported.  
**O Impl status:** method documented; model not implemented in repo.  
**P Artifact status:** no certified model-ready artifacts.  
**Q Blockers:** method open decisions (embedding, calendar, features).  
**R User confirm:** none for objective itself.

---

## S02 — Frozen node universe, identifiers, node-index map, provenance

**A.** Exactly 16,736 accounts form the immutable node set. Indices are dense `0…16735`. Dataset B cannot expand the set.

**B.** `node_idx ∈ {0,…,N−1}`; join key = exact digit-string `user.id`. Never reconstruct IDs from float.

**C.** User D2; `docs/data/02`, `03`; P-001 §13.  
**D.** V3 `node_map.py` — **compatible only if** the produced map equals exactly the 16,736 frozen set (V3’s optional non-canonical author union is **not** allowed under D2).  
**E.** Dataset A defines/verifies the 16,736 universe.  
**F.** Dataset B reconciles authors to the map; unmatched dropped; no new nodes.  
**G.** Dataset A workbooks + account inventory.  
**H.** Immutable `node_index_map.parquet` (or equivalent) with checksum/manifest.  
**I.** Map length `N=16736`.  
**J.** Inactive nodes remain in `V` with masks.  
**K.** Population frozen for the study period.  
**L.** Deterministic sort of `user.id` (digit-string) for index assignment; provenance of source file.  
**M.** V3 union-of-noncanonical vs D2 — **implementation must follow D2**. Historical 10,040 is not the modeled universe under D2.  
**N.** DATA_SUPPORTS_METHOD_DIRECTLY for N=16736.  
**O.** Builder exists (`scripts/build_node_index_map.py`); artifact **not certified**.  
**P.** No certified persistent node map.  
**Q.** Persistent certification + V3 alignment with D2.  
**R.** None (D2 confirmed).

---

## S03 — Temporal calendar, snapshots, activity, inactive handling

> **Q-CAL (USER_CONFIRMED, partial — 2026-07-26; PROC 2026-07-28):** Temporal frequency = **quarterly** (canonical). Monthly = sensitivity-analysis variant only. Provisional pilot/diagnostic calendar = **35 quarterly snapshots, 2017-Q4→2026-Q2** — **diagnostic-only**, not the certified final calendar. **Internal empty quarterly snapshots always preserved (QCAL-B01-PROC).** Leading/trailing empty + final start/end/`T` require explicit user approval after the coverage report. Calendar bounds must be **runtime-configurable, never hard-coded** in reusable source. Out-of-range records are **excluded with explicit reason codes and reported** (never silently dropped); invalid/corrupted/epoch-outlier timestamps are classified **separately** from valid-out-of-range records. **Exact numerical boundaries = `REVIEW_REQUIRED_AFTER_REAL_DATA_DIAGNOSTICS`**: a 10-item coverage report must be produced and reviewed before boundaries freeze. No full-data artifact may be labeled `calendar-certified` before that approval.

**A.** Calendar-aligned quarterly snapshots are the main design (confirmed). Empty periods inside the observation range are kept. GRU updates when `model_active` (QACT/QGRU); otherwise exact hidden-state carry.

**B.** Ordered intervals `[start,end)`; assign by `created_at`. Main: quarterly (confirmed). Sensitivity: monthly.

**C.** P-001 §12, §13, §17.  
**D.** P-002 §7.7–7.8; V3 `snapshots.py` (`freq=Q`, empty kept); pilot 35-bin calendar.  
**E/F.** Both A and B events/tweets assigned by timestamp.  
**G.** Timestamp fields.  
**H.** `snapshots.parquet` / `snapshot_calendar.json`.  
**I.** `T` symbolic until calendar frozen; masks `[T,N]`.  
**J.** `struct_active_mask`, `node_text_available_mask`, `edge_text_available_mask`, `relation_available_mask` (kept separate — Q-MISS); `model_active_mask = struct_active OR node_text_available` (**QACT-01**); if not model-active → `s_i^(t)=s_i^(t-1)` (**QGRU-01**; not a text carry).  
**K.** Quarters capture meaningful structure; boundaries fixed before training.  
**L.** Deterministic UTC parsing; out-of-range → exclude/null snap with eligibility flags.  
**M.** Tail beyond 2026-Q2; B early (2012+) / late (2026-07) tweets — boundaries `REVIEW_REQUIRED_AFTER_REAL_DATA_DIAGNOSTICS` (frequency already confirmed quarterly).  
**N.** 35 bins 2017-Q4→2026-Q2 confirmed as **provisional** pilot calendar; exact boundaries pending coverage report.  
**O.** Pilot snapshots implemented; full certified calendar not published.  
**P.** No certified full calendar artifact.  
**Q.** Tail policy (open decision O-CAL).  
**R.** Snapshot end/tail (essential).

---

## S04 — Multiplex relations, direction, self-loops

> **QREL-01 (USER_CONFIRMED_CANONICAL — 2026-07-28).** Immutable relation IDs: `mention=0`, `retweet=1`, `reply=2`, `quote=3`. `relation_order=["mention","retweet","reply","quote"]`. Must be stored explicitly and validated everywhere (edges, features, masks, embeddings, GNN, fusion, edge-text, eval, configs, schemas, tests). No alphabetical, dynamic, or library remapping.

> **QSELF-01 (USER_CONFIRMED_CANONICAL — 2026-07-28).** Exclude all author-to-self events (`source_idx==target_idx`) before edge aggregation. Excluded from `count_raw`, weights, degree/strength, relation agg, edge-text. Multi-target mentions: drop only self targets. Report aggregate exclusion counts. No exceptions. Do not reify self-loops as another relation/feature/special observed edge.

**A.** Four directed relations with frozen IDs (QREL-01). Self-loops excluded before aggregation (QSELF-01). External (non-universe) targets are summaries, not nodes.

**B.** `relation_to_id={mention:0,retweet:1,reply:2,quote:3}`; edges `(src_idx→dst_idx)` under relation `r`.

**C.** P-001 §5, §7, §14–§15; QREL-01; QSELF-01.  
**D.** V3-compatible extraction; IDs now user-frozen.  
**E.** Relations derived from Dataset A structured fields.  
**F.** No relation fields in B.  
**G.** A blobs: `retweeted_status`, `reply_status`, `quoted_status`, `user_mentions`.  
**H.** Directed event logs → filter self-loops → aggregated `edges`.  
**I.** Per `(t,r)` edge lists; `R=4`.  
**J.** Relation availability mask `[T,N,R]` with R ordered by QREL-01.  
**K.** Multiplex signal must not be collapsed.  
**L.** Mentions: pairwise author→mentioned Core accounts; unique targets per tweet; self targets dropped (QSELF-01).  
**M.** Relation integer codes **USER_CONFIRMED_CANONICAL** (QREL-01).  
**N.** Fully A-derivable.  
**O.** V3 implements extraction; repo certified edges absent.  
**P.** No certified edges.  
**Q.** Persistent edges + explicit ID map validation.  
**R.** `USER_CONFIRMED_CANONICAL` (QREL-01 + QSELF-01).

---

## S05 — Dataset A duplicates, events, aggregation, weights

> **Q-DEDUP (USER_CONFIRMED, policy — 2026-07-26; PROC 2026-07-28).** Two layers. **Layer 1 (records):** conflict-aware collapse. Dataset A duplicate candidates detected by a deterministic **composite signature** (source user id + normalized timestamp + normalized text/hash + relation + target id/set + referenced-status when safe + source-file/row provenance); **float tweet id is forbidden as an exact key**. Dataset B may use **exact string tweet IDs** for detection. Duplicate = same user + same timestamp + same content + compatible relation/target. **Preserve** identical text from different users (coordinated-behavior signal) and normally identical text from the same user at different timestamps. **Concordant** groups collapse to **one distinct event** with provenance + collapsed-copy count (raw files unchanged). **Discordant** candidates (same user+timestamp, different text/relation/target) are retained, classified, and reported — never silently collapsed. **Graph event count = distinct interaction events after reconciliation.** **Layer 2 (in-field text):** conservatively remove extraction/parse/concatenation-duplicated blocks (exact repeated full-text or long consecutive repeated spans) only on strong evidence; never remove meaningful/campaign/linguistic repetition; keep original text in a private read-only artifact and a **separate cleaned-text field** with hashes of both; embeddings use the cleaned text. **QDEDUP-B01-PROC:** no full-data edge/text `CERTIFIED` before privacy-safe duplicate diagnostic + explicit user approval of A signature and L2 thresholds; aggressive fuzzy prohibited; effects on counts/coverage must be reported. `REVIEW_REQUIRED_AFTER_REAL_DATA_DIAGNOSTICS`: exact signature, span thresholds, final collapsed/modified counts, effect on weights & text coverage. See `docs/method/12`.

> **Q-WGT (USER_CONFIRMED_CANONICAL — 2026-07-26).** Edge identity `(snapshot_id, relation_id, source_idx, target_idx)` = one directed edge, one snapshot, one relation, one source, one target. `count_raw` = number of distinct interaction events after Q-DEDUP reconciliation. Canonical weight `weight_log1p = log(1 + count_raw)` using a numerically stable natural-log `log1p`. Model uses `weight_log1p` as the primary edge weight; `count_raw` is retained for auditability. Edges are directed; relations kept separate; **self-loops excluded before aggregation and reported (QSELF-01)**; multi-target mention events expand to one event per valid non-self target. Required edge fields: `snapshot_id, relation_id, source_idx, target_idx, count_raw, weight_log1p`, plus provenance/validation metadata. Permitted weight ablations: binary, raw count, and `log1p` count (canonical main). Real-data diagnostics may validate count distributions, hub concentration, and implementation correctness but do not reopen the weight definition.

**A.** Extract directed events, aggregate per `(snapshot, relation, src, dst)`, weight `w_{i→j}^{(t,r)} = log(1 + c_{i→j}^{(t,r)})` where `c` = distinct directed interaction events from `i` to `j` within snapshot `t` and relation `r` (after Q-DEDUP reconciliation). Dataset A tweet `id` is float-lossy — must not be used as an exact cross-dataset key or dedup key.

**B.** Aggregation key: `(snapshot_id, relation_id, source_idx, target_idx)`; `c` = event count; `w=log(1+c)`.

**C.** P-001 §5, §7.  
**D.** V3 aggregation SQL; verified A forensics (`docs/data/02`).  
**E.** Sole graph source.  
**F.** None for edges.  
**G.** A rows + node map + snapshots.  
**H.** `edges.parquet` with counts and `weight_log1p`.  
**I.** Sparse edge lists, not dense `N×N×R`.  
**J.** Missing edge text → zero + mask 0.  
**K.** Core↔Core only in main graph.  
**L.** Provenance IDs (basename+row) for event identity when tweet_id unsafe.  
**M.** Spec dedup-by-tweet_id outdated vs float `id`; corpus dedup policy uncertified (~1.17M dups flagged).  
**N.** DATA_SUPPORTS_AFTER_TRANSFORMATION; dedup policy USER_CONFIRMATION.  
**O.** V3 implements aggregation; A dedup policy not certified.  
**P.** No certified edge artifacts.  
**Q.** Dedup policy (O-DEDUP).  
**R.** Dedup essential.

---

## S06 — Structural node features and initial representation

> **Q-FEAT (USER_CONFIRMED_CANONICAL — 2026-07-26). `F_struct = 17`.** Tensor `X_struct[t, node_idx, feature_idx] ∈ ℝ^{T×N×17}`, N=16,736, all frozen nodes present every snapshot, dtype float32. Relation order `0=mention,1=retweet,2=reply,3=quote` (never merged). **Exact ordered schema:** `0 mention_out_degree, 1 mention_in_degree, 2 mention_out_strength_log1p, 3 mention_in_strength_log1p, 4 retweet_out_degree, 5 retweet_in_degree, 6 retweet_out_strength_log1p, 7 retweet_in_strength_log1p, 8 reply_out_degree, 9 reply_in_degree, 10 reply_out_strength_log1p, 11 reply_in_strength_log1p, 12 quote_out_degree, 13 quote_in_degree, 14 quote_out_strength_log1p, 15 quote_in_strength_log1p, 16 tweet_count_log1p`. Ordering explicit, versioned, part of the artifact contract; never silently reordered.
> **Degree:** `out_degree_i^(t,r)=|{j:c_{i→j}^{(t,r)}>0}|`, `in_degree_i^(t,r)=|{j:c_{j→i}^{(t,r)}>0}|` (distinct neighbors; repeats don't increase; **no log1p on degree**; nonnegative ints→float32). **Strength:** `out_strength_log1p=log1p(Σ_j c_{i→j}^{(t,r)})`, `in_strength_log1p=log1p(Σ_j c_{j→i}^{(t,r)})` = **log1p(sum of count_raw)**, NOT `Σ log1p(c)`, NOT sum of edge weights (natural-log log1p). **`tweet_count_log1p=log1p(#distinct authored tweets after Q-DEDUP)`** (counts once regardless of edges produced; dedup prevents inflation; float tweet id not sole dedup key). `c_{i→j}^{(t,r)}` = `count_raw` after Q-DEDUP; directed; self-loops excluded before degree/strength; multi-target mention = one event per valid target.
> **`active` and `n_active_relations` are NOT features** (excluded). **Separate mask** `struct_active_mask[t,node_idx] ∈ {0,1}^{T×N}` = True iff `tweet_count_raw>0` OR ≥1 incoming OR ≥1 outgoing canonical edge (only-incoming and tweets-no-edges are both active; inactive→17 zeros+False; empty snapshot→all zeros+all False). This is **Dataset-A structural activity only**. Per **Q-MISS**, do not infer `struct_active_mask` from text masks or vice versa. **`model_active_mask = struct_active_mask OR node_text_available_mask` (QACT-01)**; GRU update/carry per **QGRU-01**.
> **Excluded from primary:** `total_in/out_events_log1p`, `n_active_relations`, numeric `active`, engagement-derived, follower/following, account metadata, identifiers, duplicate flags, PageRank/HITS/k-core/clustering/common-neighbors, relation entropy, relation-share, recency, activity balance, reciprocity, active-days fraction, media, hashtag counts (diagnostics/ablation only; must not change F_struct). **Additional train-time scaling = QHP-02** (robust per-feature; fit on training structurally-active rows only; raw `X_struct` artifact unchanged; winsorization diagnostic-conditional). See `docs/method/14` (audit) and `docs/method/12`.

**A.** Confirmed 17-dim leakage-safe A-derived structural vector fused with temporal text via MLP into the initial node state; `active` carried as a separate mask.

**B.** `h_i^(0,t) = MLP_x([X_struct_i^(t), T_i^(t), node_text_available_mask_i^(t)])` where `X_struct_i^(t) ∈ ℝ^{17}` per the schema above (Q-MISS; do not append `node_valid_text_count` as a feature).

**C.** P-001 §8, §15.  
**D.** P-002 §9.1.  
**E/F.** All 17 features are Dataset-A-derived (edges + authored tweet counts); Dataset B not used for structural features; follower/account metadata rejected (leaky+static).  
**G.** Edges + authored-tweet rows (post Q-DEDUP).  
**H.** `X_struct` tensor `[T,N,17]` float32 + `struct_active_mask [T,N]` bool.  
**I.** `F_struct = 17` (confirmed).  
**J.** No feature has zero=missing ambiguity (frozen-node zeros are real inactivity, marked by `struct_active_mask`).  
**K.** Avoid future-aware global normalization in predictive settings (Q-HPARAM).  
**L.** Additional normalization deferred to Q-HPARAM (train-fit only).  
**M.** Engagement columns rejected (cumulative/leaky or 100% null); only the confirmed 17-dim structural set is authoritative.  
**N.** Confirmed canonical; feature list frozen at 17 (any change needs a new explicit user decision).  
**O.** Not implemented as certified tensors.  
**P.** Absent.  
**Q.** Feature list (O-FEAT).  
**R.** Essential.

---

## S07 — Text sources, node text, edge text, atomic units

> **Q-TEXT (USER_CONFIRMED_CANONICAL — 2026-07-26).** Full contract: `docs/method/15_q_text_text_unit_contract.md`. **Node text = Dataset B** (node-snapshot semantics); **edge text = Dataset A event text** (interaction semantics); roles not interchangeable; **tweet-level A↔B join prohibited**. **Node:** atomic unit = one distinct authored tweet; `z_node_q=TextEncoder(cleaned_text_q)`; `T_i^(t)=mean_{q∈Q_i^(t)} z_node_q` over retained distinct tweets authored by i with `created_at` in t; tensor `X_node_text[t,node,:]=[T,N,D_text]`; per-tweet embed→mean-pool, never concatenate. **Edge:** atomic unit = one distinct interaction-event tweet; `z_edge_q=TextEncoder(cleaned_text_q)`; `E_{i→j}^{(t,r)}=mean_{q∈Q_{i→j}^{(t,r)}} z_edge_q`; relations/directions/snapshots separate; single-event edge = that embedding; multi-target mention shares cleaned text across target events with single-tweet provenance + caching. **Input = cleaned_text only** (Q-DEDUP L2); dedup before aggregation; cache keyed by content+provenance (not float A id). **Strict in-snapshot** temporal rule. **Missing-text (Q-MISS):** exact zero + boolean masks — `docs/method/17`. Edge gate **learns** how edge text modifies structural messages (no hard-coded semantic rules). Node-text path implemented **first**, edge-text **second** — both required. `D_text`/encoder/tokenizer/max-len/truncation/dtype → Q-EMB.

**A.** Node text = what a user talks about in snapshot `t` (Dataset B). Edge text = interaction semantics (Dataset A event text). Strict leakage rule: only in-snapshot evidence.

**B.**  
- Node: `T_i^(t) = mean_{q∈Q_i^(t)} TextEncoder(cleaned_text_q)`, `Q_i^(t)` = distinct B tweets by i in t.  
- Edge: `E_{i→j}^{(t,r)} = mean_{q∈Q_{i→j}^{(t,r)}} TextEncoder(cleaned_text_q)`, `Q_{i→j}^{(t,r)}` = distinct A events i→j in (t,r).  

**C.** P-001 §6, §9–§11.  
**D.** P-002 §7.4–7.6; Q-TEXT contract `docs/method/15`.  
**E.** Edge text from A event records only (no A↔B tweet join).  
**F.** Node text from B only.  
**G.** A cleaned event text + B cleaned tweet text + maps.  
**H.** `X_node_text [T,N,D_text]`, edge-text vectors aligned to canonical edges, `node_text_available_mask`, `edge_text_available_mask`, `node_valid_text_count`, `edge_valid_text_count` (Q-MISS).  
**I.** `[T,N,D_text]`; edge embeddings by edge order; `D_text` via Q-EMB.  
**J.** Missing → exact zero + boolean mask False (Q-MISS `docs/method/17`).  
**K.** Node ≠ edge semantics; artifacts not collapsed.  
**L.** No tweet-level A↔B join; A float tweet_id never an exact key or sole cache key.  
**M.** Atomic units **confirmed** (per-tweet→mean-pool both paths); mean-pool canonical (other pooling = future ablation only).  
**N.** A-derived edge text + B node text, both required.  
**O.** Pilot normalizes B text path only (2 files); no embeddings yet.  
**P.** No certified text artifacts.  
**Q.** Encoder/dimension (Q-EMB); missing-text **resolved** (Q-MISS).  
**R.** `USER_CONFIRMED_CANONICAL`.

---

## S07b — Missing-text policy (Q-MISS)

> **Q-MISS (USER_CONFIRMED_CANONICAL — 2026-07-26).** Full contract: `docs/method/17_q_miss_missing_text_contract.md`. Policy **M1**: exact all-zero `D_text` vector + boolean availability mask for node and edge text. No learned missing embedding; no drop for missing text; no text carry-forward. Fusion MLP and Edge Gate receive masks explicitly. Missing edge text preserves structural message path. `L_sem` only where mask=True. Masks separate from `struct_active_mask`. Valid-text counts are required metadata, not model features. Privacy-safe coverage reports required.

**A.** Missingness = no valid cleaned embedding-eligible text in-snapshot (or for the edge).  
**B.** Available ⇒ arithmetic mean; unavailable ⇒ exact zero + mask False.  
**C.** P-001 zero+mask; user Q-MISS confirmation.  
**D.** `docs/method/17`.  
**E/F.** Symmetric node (B) and edge (A) rules.  
**G.** Boolean masks + integer counts + embedding tensors.  
**H.** Shapes: node `[T,N,*]`; edge aligned to canonical edge order.  
**I.** Exact-zero invariant after load/dtype/batch/serialize; no L2-norm of missing zeros.  
**J–R.** `USER_CONFIRMED_CANONICAL`.

---

## S08 — Text embedding model

> **Q-EMB (USER_CONFIRMED family; pilot design confirmed; I1 pilot wording approved; config POST_PILOT).** The embedding-model family is the **Qwen3 Embedding family only**. Preferred checkpoint `Qwen/Qwen3-Embedding-4B` is `PROVISIONAL_PENDING_PILOT`. **QEMB-I1** shared pilot instruction wording is `USER_APPROVED_PILOT_CANDIDATE` (2026-07-28) for comparison against no-instruction only — **not** the final production instruction (QEMB-X03 remains POST_PILOT). `D_text`, final instruction, `max_length`, token-level pooling, and final normalization remain pilot/verification-pending. Cross-tweet aggregation stays arithmetic mean (Q-TEXT). No non-Qwen embedding model may be named or compared. Full contract + bounded pilot spec: `docs/method/16_q_emb_embedding_contract_and_pilot_spec.md`.

**A.** Encoder = **frozen Qwen3 Embedding** (preferred `Qwen/Qwen3-Embedding-4B`, provisional pending pilot). Per-tweet encode → Stage-B mean-pool (Q-TEXT).

**B.** Frozen encoder; per-tweet encode; mean-pool; store masks/counts. `D_text` = Qwen3 output dimension, exact value pilot-selected.

**C.** P-001 §9 (per-tweet encode → mean-pool).  
**D.** Q-EMB contract `docs/method/16`.  
**E/F.** Same encoder (single Qwen3 checkpoint) for A edge texts and B node texts.  
**G.** Cleaned text units only (Q-DEDUP L2).  
**H.** Embedding tensors + meta (revision/tokenizer/instruction/dimension/max-length hashes).  
**I.** `D_text` `PROVISIONAL_PENDING_PILOT`.  
**J.** Empty sets → exact zero + mask False (Q-MISS `docs/method/17`); available vectors normalized per Q-EMB only.  
**K.** Single Qwen3 checkpoint fixed across all ablations.  
**L.** Offline precompute; runtime-detected Colab/Kaggle GPU; resumable.  
**M.** Family fixed (Qwen3 only); checkpoint/dimension/instruction pilot-pending.  
**N.** Text exists; encoder family confirmed; exact config pending pilot.  
**O.** No embedding code/libs in repo requirements yet (added at implementation).  
**P.** No embeddings computed.  
**Q.** Q-EMB pilot + user confirmation blocks final text tensors and text-aware training.  
**R.** `OPEN_PENDING_PILOT_AND_USER_CONFIRMATION`.

---

## S09–S13 — Architecture (summary; details in `05_architecture_and_tensor_contract.md`)

| Component | Definition | Status |
|---|---|---|
| S09 Edge-gated directed GraphSAGE | Per-relation in/out **mean** aggregation (**QENC-01**); primary `L=1`, sensitivity `L=2` (**QENC-02**); training fanout `[15]` primary default | Spec + Batch-4 defaults |
| S10 Edge gate | `γ=σ(MLP_g([h_j,h_i,g]))`; `g=MLP_e([e_r, weight_log1p, E, mask])`; `d_rel=16` primary (**QPROJ-01**) | Spec + Batch-4 defaults |
| S11 Masked relation fusion | Availability-aware β → `z`; if no relations: `z=h^(0)` without rewriting `struct_active_mask` (**QFUS-01**) | Spec + Batch-4 contract |
| S12 GRU | **QACT/QGRU:** update iff `model_active=struct OR node_text`; else exact `s` carry; dim `d_h=64` primary | Spec + Batch-4 defaults |
| S13 Student-t prototypes | Soft `q_ik` with α=1; primary `K=10`; KMeans++ on training active `s` (**QHP-03/04**) | Spec + Batch-4 defaults |

Primary dims: `d_h=64`, `d_sem=d_h`, `d_rel=16`. Variants: **TDMEC-G / TDMEC-NT / TDMEC|TDMEC-Full** (**QVAR-01**); TDMEC-ET reserved for graph+edge-text-only ablation.

TDMEC implements only the modules defined in this specification and `05`/`06`.

---

## S14–S15 — Losses and training (summary; details in `06_…`)

`L_total = λ_struct L_struct + λ_sem L_sem + λ_cluster L_cluster + λ_reg L_reg + λ_temp L_temp`  
**Batch 5 confirmed:** mask 15% / 3 negatives (**QLOSS-01**); `m=1.0` on ‖μ‖² (**QLOSS-02**); `L_temp` iff both endpoints `model_active` (**QLOSS-03**); λ targets 1/1/1/0.1/0.1 with staged activation + 20% `λ_temp` ramp (**QLOSS-04**); AdamW lr 5e-4 (**QTR-01**); ~70/15/15 chronological split (**QTR-02**, exact bounds POST_CAL); BPTT=3 (**QTR-03**); 5-seed policy (**QTR-04**).  
`L_sem` uses `P_z` and `P_T_node` (**QPROJ-01**).  
Stages: freeze embeddings → pretrain (struct+sem) → KMeans++ init → joint → temporal ramp → multi-seed.  
Joint early-stop criterion finalized in **Batch 6** (provisional: weighted val total loss).

---

## S16 — Community evolution

Track hard labels across snapshots; birth/death/merge/split summaries; event-window case studies. Soft assignments + entropy support transition analysis. Alignment of community IDs over time is analysis post-processing (label permutation risk — see risk register).

---

## S17 — Baselines, ablations, evaluation

Full contracts: `docs/method/18` (metrics+selection), `19` (baselines registry — Phase 10), `20` (ablations — Phase 11), `07` (summary).  
Primary predictive metric: future-link relation-macro AP. Primary temporal similarity: AMI (AMIsum).  
Model selection: validation macro-AP with collapse guards; never test.

---

## S18 — Artifacts, environments, reproducibility

Model-ready files named in P-001 §23: `node_map`, `snapshots`, `edges`, `edge_text`, `node_features`, `node_text`, masks. Formats: Parquet + `.pt`/`.npy`.  
Environments: local tests; Colab/Drive for data/GPU; persistent root `/content/drive/MyDrive/TDMEC_PROJECT_OUTPUTS/`.  
Reproducibility: Git SHA, config hashes, 5 seeds, stage checkpoints.  
**Current:** discovery + Dataset B 2-file pilot implemented/tested; **no certified model-ready artifacts**.

---

## Implementation scope

TDMEC implements only the modules in this specification (and `05`/`06`/`07`). Modeled universe remains **N = 16,736** under D2 (no non-canonical author expansion).

## Document map

| Doc | Content |
|---|---|
| `04_complete_data_to_model_pipeline.md` | A/B stage contracts |
| `05_architecture_and_tensor_contract.md` | Full architecture + tensors |
| `06_loss_training_and_inference_contract.md` | Losses/training/inference |
| `07_evaluation_baseline_and_ablation_contract.md` | Eval |
| `08_scientific_and_engineering_risk_register.md` | Risks |
| `09_current_dataset_and_artifact_state.md` | Current state |
| `10_open_decisions_before_implementation.md` | Open decisions |
| `11_implementation_blueprint.md` | Future code structure |
| `docs/handoff/07_cursor_cloud_master_handoff.md` | Cloud self-contained handoff |
