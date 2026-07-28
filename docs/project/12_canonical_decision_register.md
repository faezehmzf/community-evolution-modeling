# 12 — Canonical Decision Register

Authoritative, corrected project facts after the read-only discovery pass. Every
statement is classified: `Verified from real files` · `Verified from source code`
· `Inferred` · `Proposed` · `Unresolved` · `Incorrect prior claim`.

> Repository note: this repository previously contained **no** `docs/project/`
> specifications (only a stub `README.md`). Any incorrect prior claim that
> called Dataset B a single "canonical processed Parquet corpus," a "pilot
> corpus," or a "smaller processed dataset" is recorded below as
> **`Incorrect prior claim`** — no such lineage is supported by the real files.

## Dataset A

| # | Statement | Classification |
|---|---|---|
| A1 | Dataset A = 12 `.xlsx` tweet parts + `extraction_summary.json` (13 entries). | `Verified from real files` |
| A2 | It is a **preprocessed, account-filtered tweet extract**, not a graph/model-ready artifact set. | `Verified from real files` |
| A3 | Frozen population = **16,736** distinct accounts (`user.id`). | `Verified from real files` |
| A4 | 12,581,535 matched tweets; built from 41 raw `bts-*` files. | `Verified from source code` (extraction_summary) |
| A5 | 1,168,525 duplicates detected, **0 removed**; dedup still required. | `Verified from source code` |
| A6 | Directed relation edges (mention/retweet/reply/quote) are embedded in A blobs and derivable from A alone. | `Verified from real files` |
| A7 | Tweet `id` is stored as **float** → precision loss; unreliable as an exact key. | `Verified from real files` |
| A8 | Several columns are 100% null (ocr_text, quoted_count, bookmarks, views, engagement, sentiment, topic, copy_count). | `Verified from real files` |
| A9 | No node-index map, edge lists, snapshots, or tensors exist yet. | `Verified from real files` |
| A10 | "10,040 Core ARMY" ⊂ 16,736 (Core ARMY + Pro-fans); exact split not in files. Under **D2**, 10,040 is **not** the modeled universe (N = 16,736). | `Inferred` (subset) / D2 `USER_CONFIRMED_CANONICAL` |

## Dataset B (corrected canonical specification)

| # | Statement | Classification |
|---|---|---|
| B1 | **Dataset B = ~70 (exactly 70) RAW `statuses-*.xlsx` files** + `download_manifest.json`. | `Verified from real files` |
| B2 | **No official preprocessing has been performed** on Dataset B. | `Verified from real files` |
| B3 | Total reported size ≈ **9.58 GB** (109–159 MB/file). | `Verified from real files` (manifest) |
| B4 | Single canonical schema: 10 columns `id, created_at, user, text, likes, retweets, reply_count, quoted_count, bookmarks, views`. | `Verified from real files` (12-file sample) |
| B5 | Tweet `id` stored as **string** (exact); `created_at` = epoch-seconds string. | `Verified from real files` |
| B6 | Dataset B has **no relational edge fields** — only engagement counts. | `Verified from real files` |
| B7 | Files are per-account history chunks (~120–290 authors/file; min-time rises with index). | `Verified from real files` (sample) |
| B8 | Provenance = `https://<redacted-external-host>/static/reports/statuses-*.xlsx`. | `Verified from real files` (manifest) |
| B9 | A previously "canonical processed Dataset B Parquet corpus / pilot corpus / smaller processed dataset". | **`Incorrect prior claim`** — no lineage from the 70 raw files is proven |
| B10 | Full-corpus distinct tweet/author counts (all 70 files). | `Unresolved` (deep-inspection deferred) |

## Cross-dataset

| # | Statement | Classification |
|---|---|---|
| X1 | Canonical join key = numeric **`user.id`** (precise in both datasets). | `Verified from real files` |
| X2 | 100% of sampled Dataset B authors (2,498) are in the frozen 16,736; 0 outside. | `Verified from real files` (sample) |
| X3 | Whether all 70 B files cover the full 16,736 or a subset. | `Unresolved` |
| X4 | Temporal overlap A(2017–2026) ∩ B(2012–2026) is substantial; B extends earlier. | `Verified from real files` (sample) |
| X5 | Do **not** join on A's tweet `id` (float-lossy); use B's string IDs for tweet-level linkage. | `Verified from real files` |

## Graph contract

| # | Statement | Classification |
|---|---|---|
| G1 | `node_count = 16,736`. | `Verified from real files` |
| G2 | `directed_edges = true` (by construction from A). | `Verified` (structure) |
| G3 | relation types mention/retweet/reply/quote all present. | `Verified from real files` |
| G4 | relation codes `{mention:0,retweet:1,reply:2,quote:3}`. | **`USER_CONFIRMED_CANONICAL` (QREL-01)** — must be stored explicitly & validated |
| G5 | node indices `0..16,735`. | `Proposed` (map must be created) |
| G6 | `candidate_snapshot_count = 35` = quarterly bins 2017-Q4 … 2026-Q2. | `Inferred` (consistent with A's range) |

## Decisions

1. **Dataset B is raw**; treat the 70 Excel files as the sole canonical source.
2. **Join on `user.id`**; keep tweet IDs as strings from Dataset B.
3. **Build the graph from Dataset A**; enrich text/timestamps from Dataset B.
4. **Create (once) an immutable node-index map** over the 16,736 accounts.
5. **Snapshots are quarterly** — `USER_CONFIRMED_CANONICAL` (Q-CAL, 2026-07-26). Monthly = sensitivity-analysis variant only. Provisional pilot calendar = 35 bins (2017-Q4→2026-Q2), empty snapshots preserved, calendar config-driven (not hard-coded). Out-of-range records excluded with explicit reason codes + reported; invalid/corrupted/epoch-outlier timestamps classified separately. **Exact boundaries + tail policy = `REVIEW_REQUIRED_AFTER_REAL_DATA_DIAGNOSTICS`** (10-item coverage report must be reviewed before freeze; no full-data artifact `calendar-certified` before then). See `docs/method/12_interactive_canonical_decision_log.md`.
6. **Duplicate policy (Q-DEDUP, 2026-07-26, `USER_CONFIRMED_CANONICAL` for policy).** Two layers. **Records:** conflict-aware collapse; A via deterministic composite signature (float id forbidden as key), B via exact string IDs; duplicate = same user+timestamp+content+compatible relation/target; identical text across different users preserved; same-user different-timestamp normally preserved; concordant→one distinct event + provenance + copy count; discordant→retained+flagged; **graph count = distinct events**. **In-field text:** conservative removal of extraction-duplicated blocks only; original text kept private + separate cleaned field + hashes; embeddings use cleaned text; meaningful/campaign repetition preserved. Exact signature/thresholds/final numbers `REVIEW_REQUIRED_AFTER_REAL_DATA_DIAGNOSTICS`; raw source files unchanged. See `docs/method/12`.
7. **Edge weight (Q-WGT, 2026-07-26, `USER_CONFIRMED_CANONICAL`).** Edge identity `(snapshot_id, relation_id, source_idx, target_idx)`. `count_raw` = distinct interaction events after Q-DEDUP. Canonical weight `weight_log1p = log(1 + count_raw)` (stable natural-log `log1p`), used as the primary model edge weight; `count_raw` retained. Directed; relations separate; self-loops excluded+reported; multi-target mention events expanded to one event per valid target. Required edge fields: `snapshot_id, relation_id, source_idx, target_idx, count_raw, weight_log1p` + provenance/validation metadata. Ablations: binary, raw count, `log1p` (main). Diagnostics validate distributions/hub concentration/correctness only; they do not reopen the definition. See `docs/method/12`.
8. **Structural node features (Q-FEAT, 2026-07-26, `USER_CONFIRMED_CANONICAL`). `F_struct = 17`.** `X_struct[t,node_idx,feature_idx] ∈ ℝ^{T×N×17}` float32, all 16,736 frozen nodes present every snapshot. Relation order `0=mention,1=retweet,2=reply,3=quote`. Ordered schema: per relation {out_degree, in_degree, out_strength_log1p, in_strength_log1p} (indices 0–15) + `16 tweet_count_log1p`. Degree = distinct-neighbor count (no log1p); strength = `log1p(Σ_j count_raw)` (NOT Σlog1p, NOT sum of edge weights); tweet_count = `log1p(#distinct authored tweets after Q-DEDUP, counted once regardless of edges)`. `active` and `n_active_relations` are **excluded** from X_struct. Separate boolean `struct_active_mask[t,node] ∈ {0,1}^{T×N}` (True iff tweet_count_raw>0 OR ≥1 incoming OR ≥1 outgoing canonical edge; Dataset-A structural activity only — relation to text/model masks decided later). Excluded: engagement, follower/account metadata, identifiers, duplicate flags, centrality (PageRank/HITS/k-core/clustering/common-neighbors), relation entropy/shares, recency, balance, reciprocity, active-days, media, hashtags. Additional normalization **QHP-02 USER_CONFIRMED** (train-time robust scaling; train-fit only; do not standardize canonical artifacts). Schema ordering is versioned and part of the artifact contract. See `docs/method/12`, `docs/method/14`.
9. **Node/edge text units (Q-TEXT, 2026-07-26, `USER_CONFIRMED_CANONICAL`).** Dataset B = node-snapshot semantics; Dataset A event text = edge/interaction semantics; not interchangeable; **tweet-level A↔B join prohibited**. Node text: atomic unit = one distinct authored tweet; `T_i^(t) = mean_{q∈Q_i^(t)} TextEncoder(cleaned_text_q)` over B tweets by i in snapshot t → `X_node_text[T,N,D_text]`. Edge text: atomic unit = one distinct interaction-event tweet; `E_{i→j}^{(t,r)} = mean_{q∈Q_{i→j}^{(t,r)}} TextEncoder(cleaned_text_q)` per canonical edge (relations/directions/snapshots separate). Embedding input = `cleaned_text` only (Q-DEDUP L2); dedup before aggregation; cache keyed by content+provenance (not float A id). Strict in-snapshot temporal rule. Masks `node_text_available_mask`, `edge_text_available_mask` (exact missingness → **Q-MISS**). Edge gate learns how edge text modifies structural messages. Per-tweet embed→mean-pool (other pooling = future ablation only). Node-text path first, edge-text second — both required. `D_text`/encoder → Q-EMB. See `docs/method/15`, `docs/method/12`.
10. **Missing-text (Q-MISS, 2026-07-26, `USER_CONFIRMED_CANONICAL`). Policy M1.** Exact all-zero `D_text` vector + boolean availability mask for node (B) and edge (A) text. No learned missing embedding; no dropping nodes/edges for missing text; no text carry-forward. Fusion MLP and Edge Gate receive masks explicitly. Missing edge text preserves structural path (`count_raw`/`weight` unchanged). `L_sem` only where mask=True; zero-eligible → defined zero contribution. Separate `struct_active_mask`, `node_text_available_mask`, `edge_text_available_mask`. `node_valid_text_count` / `edge_valid_text_count` = required artifact metadata, not model features. Privacy-safe coverage reports required. Artifacts: `X_node_text[T,N,D_text]`, `node_text_available_mask[T,N]`, `node_valid_text_count[T,N]`; edge vectors/masks/counts aligned to canonical edge order. See `docs/method/17`.

## Update — session 2 (handoff specification)

- The full **handoff specification** for Cursor Cloud Agent is in
  `docs/handoff/00`–`06`; per-component status in `docs/project/14`; artifact
  registry in `docs/project/15`. Open decisions with the smallest confirming
  question are in `docs/handoff/05`. Canonical method contracts live in
  `docs/method/03`–`17` and `docs/method/12`.
- **Confirmed contracts (do not reopen):** D1/D2; Q-CAL (quarterly; bounds
  review-pending); Q-DEDUP (policy; signature review-pending); Q-WGT
  (`weight_log1p`); Q-FEAT (`F_struct=17` + `struct_active_mask`); Q-TEXT
  (A=edge text, B=node text; per-tweet/event → mean-pool; edge text in scope);
  Q-MISS (M1 exact zero + boolean mask — `docs/method/17`); Q-EMB family =
  Qwen3 Embedding only (X01–X07 `PENDING_PILOT` — `docs/method/16`);
  QREL/QSELF/QACT/QGRU/QGATE; Batch 4 architecture (incl. QHP-01…04, QVAR-01);
  Batch 5 loss/training; Batch 6–7 eval/baselines/ablations/MLP/decoder/phases
  (`docs/method/18`–`21`).
- **Still open / evidence-dependent only:** O-CAL boundaries; QDEDUP signature/L2;
  coverage hard thresholds; Q-EMB X01–X07; runtime batch/AMP/OOM; post-training
  baseline feasibility. See `docs/handoff/05`, `docs/method/10`, `21`.
- **Variant names:** **TDMEC / TDMEC-Full** = primary; **TDMEC-G / TDMEC-NT** =
  ablations; **TDMEC-ET** reserved for graph+edge-text-only ablation if
  implemented (**QVAR-01**). Never name the full method TDMEC-ET.
- **Next operational action (Cursor Cloud only):** Phase 1 — typed configs,
  schemas, validation utilities, synthetic fixtures, invariant unit tests
  (`docs/method/21`). No local implementation in the documentation session.
