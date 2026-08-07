# 02c — Method-vs-Data Compatibility Audit (TDMEC)

Checks each TDMEC method requirement against **verified** Dataset A/B facts in
`docs/data/`, `docs/handoff/`, `docs/project/`, `artifacts/discovery/`. Mismatches
are recorded, not fixed.

Classification per row:
- `DATA_SUPPORTS_METHOD_DIRECTLY`
- `DATA_SUPPORTS_AFTER_TRANSFORMATION`
- `METHOD_OUTDATED_RELATIVE_TO_DATA`
- `METHOD_SPEC_REQUIRED`
- `USER_CONFIRMATION_REQUIRED`
- `BLOCKED_BY_DATA_LIMITATION`

Verified facts: node count 16,736; Dataset A = tweet extract, edges derivable from A
alone; A `id` float-lossy; Dataset B RAW, 70 files, 10 cols, no edge fields, no
language column; join key `user.id`; A span 2017-Q4→2026-Q2 ≈ 35 quarterly bins;
embedding config not pinned in-repo (Q-EMB PENDING_PILOT).

---

## 1. Requirement-by-requirement audit

| # | TDMEC requirement | Required data fields | Actual A/B support | Transformation needed | Classification |
|---|---|---|---|---|---|
| R1 | Fixed Core node universe, indexed | canonical account IDs | **16,736** frozen distinct `user.id` verified | Freeze index; enforce D2 (no non-canonical expansion) | `USER_CONFIRMED` (D2) + `DATA_SUPPORTS_AFTER_TRANSFORMATION` |
| R2 | Directed multiplex graph, 4 relations | retweet/reply/quote/mention target IDs | A has structured relation fields; **graph derivable from A alone** | Parse blobs → directed edges; aggregate | `DATA_SUPPORTS_AFTER_TRANSFORMATION` |
| R3 | Dataset A becomes a graph | edges | A is a **preprocessed tweet extract, NOT a graph** | Build edges (pipeline for structural edges) | `DATA_SUPPORTS_AFTER_TRANSFORMATION` |
| R4 | Reliable dedup / event identity | stable event key | A top-level `id` **float-lossy**; 1.17M dups detected historically | Provenance IDs / aggregation; Q-DEDUP policy | `METHOD_OUTDATED_RELATIVE_TO_DATA` (tweet_id key) → `DATA_SUPPORTS_AFTER_TRANSFORMATION` + Q-DEDUP |
| R5 | Tweet-level A↔B join | shared exact tweet id | **Not reliably possible** | Method does **not** require it: node text from B; edge text from A | `DATA_SUPPORTS_METHOD_DIRECTLY` (no join needed) |
| R6 | Edge text E_ij^(t,r) | interaction tweet `text` + relation | A carries text + relation fields → **A-derivable** | Attach authoring tweet text to edge; embed (stage not built) | `DATA_SUPPORTS_AFTER_TRANSFORMATION`; embedding = `METHOD_SPEC_REQUIRED` |
| R7 | Node text T_i^(t) | author id, created_at, tweet text | B has `user.id`, `created_at`, `text` | Filter to Core, snapshot-bin, per-tweet embed, mean-pool | `DATA_SUPPORTS_AFTER_TRANSFORMS` |
| R8 | B does not expand node set | — | Sampled B authors ∈ frozen 16,736; pipeline filters to node_map | Keep fixed universe (D2) | `DATA_SUPPORTS_METHOD_DIRECTLY` |
| R9 | B has no relation/edge fields | — | Confirmed | Use B for node text only | `DATA_SUPPORTS_METHOD_DIRECTLY` |
| R10 | Quarterly snapshots fit timestamps | created_at | A ≈ **35** quarterly bins 2017-Q4→2026-Q2 | Build calendar (`freq=Q`) | `DATA_SUPPORTS_METHOD_DIRECTLY` |
| R11 | Snapshot boundaries / tail | created_at extremes | Late timestamps; B pre-2018 / mid-2026; possible outliers | Decide tail policy (Q-CAL) | `USER_CONFIRMATION_REQUIRED` |
| R12 | Text unit = per-tweet then pool | tweet `text` rows | Per-tweet text in A (edge) and B (node) | Confirm atomic unit if departing from default | mechanically supported; choice open (Q-TEXT) |
| R13 | Structural node features | counts + optional followers | Counts derivable; many A engagement cols **100% null**; followers in B blob | Populate subset; set F (Q-FEAT) | `METHOD_SPEC_REQUIRED` + `USER_CONFIRMATION_REQUIRED` |
| R14 | Text encoder | text | Family **USER_CONFIRMED Qwen3 Embedding only**; config **PENDING_PILOT** | Pin checkpoint + `D_text` (Q-EMB) | family confirmed; config `PENDING_PILOT` + `METHOD_SPEC_REQUIRED` |
| R15 | Language handling | lang | B no language column; A lang population unknown | Multilingual encoder → no filter mandated | `DATA_SUPPORTS_METHOD_DIRECTLY` / lang stats limited |
| R16 | log(1+count) edge weights | event counts | Derivable; pipeline emits weight_log1p | — | `DATA_SUPPORTS_METHOD_DIRECTLY` |
| R17 | Model-ready tensors / masks | all above | **None exist** as `.pt`; pipeline stops at parquet | Build tensor-export + embedding stages | `METHOD_SPEC_REQUIRED` |
| R18 | Masks (active / text / relation) | activity per node-snapshot | Derivable; not implemented | Build masks | `DATA_SUPPORTS_AFTER_TRANSFORMS` |
| R19 | External-account summaries (not nodes) | non-Core targets | Pipeline emits `external_targets.parquet` | — | `DATA_SUPPORTS_METHOD_DIRECTLY` |

## 2. Identifier risks

- Dataset A tweet `id` is float-lossy — unusable as an exact key (R4/R5). Use exact
  digit-string user IDs + provenance / composite signatures (Q-DEDUP).
- `user.id` is exact in both datasets → safe join/index key.

## 3. Temporal compatibility

Quarterly calendar (~35 bins) matches TDMEC quarterly main and pipeline `freq=Q`.
Tail beyond 2026-Q2 and outliers need Q-CAL. B tweets outside A's graph range need an
eligibility rule (`embedding_temporal_eligible`).

## 4. Data limitations

- Full 70-file Dataset B corpus stats may be incomplete → author coverage across all
  16,736 accounts may be `BLOCKED_BY_DATA_LIMITATION` until computed.
- Tweet-level A↔B linkage not demonstrated (TDMEC does not need it).
- Language distribution unquantified.

## 5. Adjustments to apply in implementation (not invented science)

1. Use provenance / composite-signature dedup (not float tweet_id) per Q-DEDUP.
2. Restrict structural features to **populated** fields; decide followers (Q-FEAT).
3. Enforce modeled universe **N = 16,736** (D2).
4. Decide snapshot tail / B out-of-range embedding (Q-CAL).
5. Pin Qwen3 checkpoint / `D_text` via pilot (Q-EMB).

## 6. Compatibility summary

TDMEC is structurally well-matched to the data: directed 4-relation multiplex, edge
text, node text, quarterly snapshots, fixed universe, and log1p weights are
**derivable from A + B** without a tweet-level join. Gaps are the model / embedding /
tensor stack and open policies (calendar, features, embedding config, text unit,
missing-text). No verified data fact contradicts the TDMEC design; the unusable
float tweet_id key is already handled by provenance/aggregation patterns.
