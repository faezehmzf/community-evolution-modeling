# 02d — Phase-2 Open Questions (TDMEC)

Smallest set of questions still open (or recently settled) for TDMEC
canonicalization / implementation. Items already fixed by D1/D2/Q-* are summarized
positively; contrastive framing is omitted.

---

### Q1 — Embedding model: Qwen3 Embedding only; config pending pilot
- **Topic:** text encoder (C-05 / Q-EMB).
- **Status:** **RESOLVED — family `USER_CONFIRMED_QWEN3_ONLY`.** Exact checkpoint
  (`Qwen/Qwen3-Embedding-4B` preferred, `PROVISIONAL_PENDING_PILOT`), `D_text`,
  instruction, pooling, normalization, and `max_length` are **`PENDING_PILOT`**
  (`docs/method/16`).
- **Why it matters:** sets tensor dims (`D_t`, `D_e`), sequence length, prompting,
  normalization, and precompute budget; must be held constant across ablations.
- **Forensic note:** Proposed TDMEC files named a multilingual encoder (specific name
  omitted); the Qwen3 family entered via the user's canonical decision (Q-EMB).
- **Remaining:** exact Qwen3 checkpoint/config via the pilot only.

### Q2 — Modeled node universe
- **Status: USER-CONFIRMED (D2).** Canonical universe exactly **N = 16,736**,
  indices `0…16735`, immutable. Dataset B must not create/append/renumber nodes.
  Data construction must not expand beyond the frozen 16,736 set. Historical
  **10,040** is not the modeled universe.

### Q3 — Snapshot calendar bounds & tail policy
- **Topic:** time (C-09).
- **Why it matters:** determines T and which records are eligible.
- **Evidence:** DATA A span 2017-Q4→2026-Q2 ≈ **35** quarterly bins; late timestamps
  to 2026-05-30 and possible outliers; Dataset B has pre-2018 (2012+) and mid-2026
  tweets. TDMEC: quarterly, fixed boundaries, empty bins kept.
- **Options:** (a) clip to 2017-Q4→2026-Q2 (35 bins), drop out-of-range;
  (b) extend calendar to cover B's range; (c) treat outliers as errors.
- **User confirmation:** **Yes** (DATA D-4 / Q-CAL).

### Q4 — Atomic text unit & missing-text handling
- **Topic:** text granularity (C-06 / Q-TEXT) + missing-text (C-14 / Q-MISS).
- **Status:** **RESOLVED.** Atomic units + mean-pool = Q-TEXT
  (`USER_CONFIRMED_CANONICAL`, `docs/method/15`). Missing-text = **Q-MISS M1**
  (`USER_CONFIRMED_CANONICAL`, `docs/method/17`): exact all-zero vector + boolean
  availability mask (node+edge); no learned missing; no drop; no text carry-forward;
  Fusion MLP / Edge Gate receive masks; missing edge text preserves structural path;
  `L_sem` mask=True only; separate struct/text masks; valid-text counts = metadata.
- **Remaining:** none for this topic (encoder/`D_text` → Q-EMB).

### Q5 — Structural node-feature set and dimension F
- **Topic:** node features (C-13 / Q-FEAT).
- Spec recommends tweet_count / has_text / is_active / in-out counts / followers if
  reliable. DATA: several A engagement columns 100% null; `followers` only in B
  `user` blob; F unresolved.
- **Options:** (a) minimal populated set (counts + activity + relation degrees);
  (b) add followers from B.
- **User confirmation:** **Yes** (Q-FEAT).

### Q6 — Dataset A / B duplicate policy
- **Topic:** dedup (C-12 / Q-DEDUP).
- **Status:** policy **`USER_CONFIRMED_CANONICAL`** (two-layer records + in-field text
  cleaning — see `docs/method/12`). Exact composite signature, span thresholds, and
  final numbers `REVIEW_REQUIRED_AFTER_REAL_DATA_DIAGNOSTICS`.
- Float A tweet id is not an exact key; B may use exact string IDs.

### Q7 — Authoritative source hierarchy
- **Status: USER-CONFIRMED (D1).** P-001 primary scientific authority; P-002 supporting
  technical (must not override P-001); `temporal_graph_pipeline` data-construction
  authority only when compatible with method + verified data + D2. Absent referenced
  files remain documentation-completeness items and do not block canonicalization.

---

## Settled TDMEC design (reference)

TDMEC uses: edge-gated directed GraphSAGE; single GRU; prototype Student-t community
head (α=1); fixed-K with sensitivity sweep; **4** relations; directed edges;
hierarchical losses (struct / sem / cluster / reg / temp); quarterly main snapshots;
A-derived edge text (no tweet-level A↔B join required). These do **not** need a
user question unless reopening a contract.
