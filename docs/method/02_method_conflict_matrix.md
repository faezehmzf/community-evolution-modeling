# 02 — TDMEC Internal Conflict / Open-Item Matrix

Every substantive **internal** conflict or open item among TDMEC sources, verified
data facts, and open decisions. Conflicts with non-TDMEC designs are out of scope.
No final method is invented here; status labels record evidence strength only.

Conflict types: `DETAIL_MISSING`, `PRESENTATION_SIMPLIFICATION`,
`IMPLEMENTATION_DIVERGENCE`, `DATA_INCOMPATIBILITY`, `OPEN_POLICY`.

Resolution status: `RESOLVED_FROM_EVIDENCE`, `LIKELY_CURRENT_BUT_NOT_FINAL`,
`USER_CONFIRMATION_REQUIRED`, `BLOCKED_BY_MISSING_SOURCE`, `USER_CONFIRMED`.

---

## Summary

| Bucket | Count (approx.) | Examples |
|---|---|---|
| Spec-internal (compatible expansions) | few | P-001 ↔ P-002 detail level |
| Spec vs data pipeline | several | edge-text stage not built; node-universe union vs D2 |
| Spec vs verified data | several | float-lossy A tweet id; null feature columns |
| Open policies / pilots | several | calendar tail, D_text, relation ids |

---

## A. Spec hierarchy & presentation

### C-01 — P-001 vs P-002 detail level
- **A:** P-001 base methodology.
- **B:** P-002 expanded technical companion (α=1, d_h examples, tensor tables).
- **Type:** `DETAIL_MISSING` (expansion, not contradiction).
- **Status:** `RESOLVED_FROM_EVIDENCE` — compatible; D1: P-001 primary, P-002 supporting (*Source-stated* binding).

### C-02 — Presentation simplifications
- **A:** Formal specs (P-001/P-002).
- **B:** P-003/P-004/P-005 decks (acknowledged simplifications, e.g. slide-level equations).
- **Type:** `PRESENTATION_SIMPLIFICATION`.
- **Status:** `RESOLVED_FROM_EVIDENCE` — decks explanatory only; must not override P-001.

---

## B. Architecture & community (TDMEC-positive; mostly settled)

### C-03 — Graph encoder / direction / temporal / head / K / losses
TDMEC uses: **edge-gated directed GraphSAGE**; **directed** (separate in/out) message
passing; **single GRU**; **prototype Student-t** soft head (α=1) with KMeans init;
**fixed K** with sensitivity sweep {5,10,15,20,30}; hierarchical
`L_struct + L_sem + L_cluster + L_reg + L_temp` (JS on adjacent Q).
- **Status:** architecture family `RESOLVED_FROM_EVIDENCE` / D1; exact hyperparameters
  and chosen K remain `LIKELY_CURRENT_BUT_NOT_FINAL`.

### C-04 — Loss weights λ
- Spec starting values λ_struct=λ_sem=λ_cluster=1.0, λ_reg=λ_temp=0.1 — "to be tuned".
- **Status:** `LIKELY_CURRENT_BUT_NOT_FINAL`.

---

## C. Text & embedding

### C-05 — Embedding family / config (Q-EMB)
- Family: **USER_CONFIRMED Qwen3 Embedding only**.
- Preferred checkpoint `Qwen/Qwen3-Embedding-4B` (`PROVISIONAL_PENDING_PILOT`);
  `D_text`, instruction, pooling, normalization, `max_length` = **PENDING_PILOT**
  (`docs/method/16`).
- **Status:** family resolved; config `PENDING_PILOT`.

### C-06 — Text granularity / pooling
- Spec default: per-tweet encode → mean-pool to snapshot-local T_i^(t); recency-pooling
  as ablation.
- Atomic text unit also tracked in DATA/handoff as confirmation-sensitive.
- **Status:** `LIKELY_CURRENT_BUT_NOT_FINAL` (confirm if departing from spec default).

### C-07 — Node text vs edge text
- TDMEC defines both T_i^(t) and E_ij^(t,r).
- `temporal_graph_pipeline` emits structural edges + raw event text; **no edge-text
  embedding column / stage yet**.
- **Type:** `IMPLEMENTATION_DIVERGENCE` (spec ahead of data pipeline).
- **Status:** intended by P-001; edge text A-derivable; embedding stage open.

---

## D. Graph construction, time, features, universe

### C-08 — Relations
- TDMEC + pipeline: **4** relations; **QREL-01 frozen IDs** `mention=0, retweet=1, reply=2, quote=3`.
- Integer code ordering: **`USER_CONFIRMED_CANONICAL` (QREL-01)**.
- Self-loops: **`USER_CONFIRMED_CANONICAL` (QSELF-01)** — exclude before aggregation.
- **Status:** relation set `RESOLVED_FROM_EVIDENCE`; ids `OPEN_POLICY`.

### C-09 — Snapshot frequency & calendar
- TDMEC: quarterly main, monthly sensitivity; pipeline default `freq=Q`; empty bins kept.
- DATA: A span 2017-Q4→2026-Q2 ≈ **35** quarterly bins; late timestamps / outliers;
  B has pre-2018 and mid-2026 tweets.
- **Status:** quarterly main `RESOLVED_FROM_EVIDENCE`; **tail / bounds**
  `USER_CONFIRMATION_REQUIRED` (calendar policy).

### C-10 — Self-loops
- Spec + pipeline default: exclude as graph edges.
- **Status:** `RESOLVED_FROM_EVIDENCE`.

### C-11 — Node universe (D2)
- **USER_CONFIRMED:** modeled universe exactly **N = 16,736**, indices `0…16735`.
- Pipeline historically allowed canonical CSV ∪ non-canonical Dataset A authors —
  **not allowed under D2** for the modeled set.
- Historical **10,040** figure is not the modeled universe.
- **Status:** `USER_CONFIRMED` (D2).

### C-12 — Deduplication vs float-lossy A tweet id
- Spec mentions tweet_id-style keys; Dataset A top-level `id` is float-lossy;
  pipeline aggregates by (snapshot, src, dst, relation) with provenance IDs.
- **Status:** implementation approach `RESOLVED_FROM_EVIDENCE` (provenance/aggregation);
  corpus policy tied to **Q-DEDUP** (`USER_CONFIRMED_CANONICAL` for policy; diagnostics
  for exact signatures/thresholds).

### C-13 — Structural features / dim F
- Spec recommends tweet_count, has_text, is_active, relation degrees, followers if
  reliable; many A engagement columns 100% null; followers only in B `user` blob.
- **Status:** `USER_CONFIRMATION_REQUIRED` / `METHOD_SPEC_REQUIRED` (Q-FEAT).

### C-14 — Missing-text / availability masks
- **Q-MISS M1** (`USER_CONFIRMED_CANONICAL`, `docs/method/17`): exact all-zero
  `D_text` vector + boolean availability mask (node+edge); no learned missing;
  no drop; no text carry-forward; Fusion MLP / Edge Gate receive masks;
  missing edge text preserves structural path; `L_sem` mask=True only; separate
  `struct_active_mask` / text masks; valid-text counts = metadata not features.
- Masks/counts/embeddings **not yet produced** (blocked on Q-EMB materialization).
- **Status:** `USER_CONFIRMED` (policy); build pending embeddings.

---

## E. Blocked-by-missing-source

### C-M1 — Stage-1 fine detail
- P-002 cites absent `sections/Stage1_TemporalMultiplexGraph.md`.
- Equivalent graph detail exists in P-001 + pipeline.
- **Status:** `BLOCKED_BY_MISSING_SOURCE` only for Stage-1-*only* nuances;
  does not block D1 canonicalization.

---

## Status rollup

- **USER_CONFIRMED:** D1, D2; Q-EMB family; Q-DEDUP policy (details pending diagnostics);
  Q-MISS M1 (exact zero + boolean mask).
- **RESOLVED_FROM_EVIDENCE (TDMEC design):** encoder family, direction, single GRU,
  Student-t head, fixed-K strategy, 4 relations, hierarchical losses, quarterly main,
  self-loop exclusion, P-001↔P-002 compatibility.
- **PENDING / OPEN:** calendar tail, `D_text` + embedding config, relation id
  ordering, edge-text embedding stage, λ values, chosen K.
