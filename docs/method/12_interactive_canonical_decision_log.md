# 12 — Interactive Canonical Decision Log

Chronological record of the interactive canonicalization session. Language: English.

## Status vocabulary
`PROPOSED_TO_USER` · `USER_ANSWER_RECEIVED` · `CLARIFICATION_REQUIRED` · `USER_CONFIRMED_CANONICAL` · `BLOCKED`

## Confirmed before this session
- **D1** — Method authority (P-001 primary scientific; P-002 supporting technical; `temporal_graph_pipeline` data-construction when compatible with P-001 + verified data + D2). Status: `USER_CONFIRMED_CANONICAL`.
- **D2** — Frozen node universe N = 16,736 (indices 0…16,735; immutable; B cannot expand; 10,040 not modeled). Status: `USER_CONFIRMED_CANONICAL`.

## Ordered decision queue
| Order | ID | Title | Status |
|---|---|---|---|
| 1 | Q-CAL | Snapshot calendar & tail policy | `USER_CONFIRMED_CANONICAL` (partial; **PROC confirmed 2026-07-28**; boundaries `REVIEW_REQUIRED`) |
| 1b | QCAL-B01-PROC | Calendar certification procedure | `USER_CONFIRMED_CANONICAL` (2026-07-28) |
| 2 | Q-DEDUP | Dataset A duplicate policy | `USER_CONFIRMED_CANONICAL` (policy; **PROC confirmed 2026-07-28**; numbers/thresholds `REVIEW_REQUIRED`) |
| 2b | QDEDUP-B01-PROC | Dedup certification procedure | `USER_CONFIRMED_CANONICAL` (2026-07-28) |
| 3 | Q-WGT | Edge aggregation & weight transform | `USER_CONFIRMED_CANONICAL` |
| 4 | Q-FEAT | Structural node features | `USER_CONFIRMED_CANONICAL` (F_struct=17; train-time scaling **QHP-02**) |
| 5 | Q-TEXT | Atomic node-text & edge-text units | `USER_CONFIRMED_CANONICAL` |
| 6 | Q-EMB | Embedding model & full embedding contract | `OPEN_PENDING_PILOT` — family confirmed; **P01–P03 USER_CONFIRMED**; **I1 pilot wording USER_APPROVED**; X* POST_PILOT |
| 6b | QEMB-I1 | Shared pilot instruction wording | `USER_APPROVED_PILOT_CANDIDATE` (2026-07-28; not final production) |
| 7 | Q-MISS | Missing-text behavior | `USER_CONFIRMED_CANONICAL` — M1 zero+boolean mask |
| 8 | QREL-01 | Canonical relation-ID mapping | `USER_CONFIRMED_CANONICAL` — 0=mention…3=quote |
| 8b | QSELF-01 | Self-loop exclusion | `USER_CONFIRMED_CANONICAL` — exclude before aggregation |
| 9 | QACT-01 | model_active_mask | `USER_CONFIRMED_CANONICAL` — struct OR node_text |
| 10 | QGRU-01 | GRU update/carry | `USER_CONFIRMED_CANONICAL` — update if active else carry |
| 11 | QGATE-01 | Edge Gate / MLP_e inputs | `USER_CONFIRMED_CANONICAL` — A; counts metadata only |
| 11b | QART-01-FRAME | Artifact certification hard vs warning framework | `USER_CONFIRMED_CANONICAL` (2026-07-28; numeric coverage thresholds open) |
| 12 | Q-HPARAM / Batch 4 | Architecture hyperparameters | `USER_CONFIRMED` (2026-07-28) — see QFUS/QENC/QPROJ/QHP/QVAR |
| 12a | QFUS-01 | No-relation fusion fallback | `USER_CONFIRMED_CANONICAL` |
| 12b | QENC-01 | Directed mean agg + fanout | `USER_CONFIRMED` (mean immutable; fanout primary default) |
| 12c | QENC-02 | GNN depth L | `USER_CONFIRMED` (L=1 primary; L=2 sensitivity) |
| 12d | QPROJ-01 | Relation / semantic projection dims | `USER_CONFIRMED` (contract + primary defaults) |
| 12e | QHP-01 | `d_h` | `USER_CONFIRMED` (64 primary; {32,128} sensitivity) |
| 12f | QHP-02 | Train-time structural feature scaling | `USER_CONFIRMED` |
| 12g | QHP-03 | Primary K | `USER_CONFIRMED` (K=10; sensitivity set) |
| 12h | QHP-04 | Prototype KMeans++ init | `USER_CONFIRMED` |
| 12i | QVAR-01 | Variant reporting names | `USER_CONFIRMED_CANONICAL` (reporting contract) |
| 13 | Q-LOSS / Batch 5 | Loss weights & training schedule | `USER_CONFIRMED` (2026-07-28) |
| 14 | Q-EVAL / Batch 6 | Evaluation, baselines, ablations | `RESOLVED_BY_AUTHORIZED_ARCHITECT` (2026-07-28) — see 18/19/20 |
| 15 | Batch 7 ops | MLP/decoder/DEC/batch/phases | `RESOLVED_BY_AUTHORIZED_ARCHITECT` (2026-07-28) |

---

## Q-CAL — Snapshot calendar and tail policy
- **Status:** `USER_CONFIRMED_CANONICAL` (partial) + `REVIEW_REQUIRED_AFTER_REAL_DATA_DIAGNOSTICS` (boundaries/tail) + **QCAL-B01-PROC confirmed 2026-07-28**
- **Date proposed:** 2026-07-26 · **Date answered:** 2026-07-26 · **PROC answered:** 2026-07-28
- **Question:** Fix the canonical temporal calendar (frequency, start, end) and the policy for records outside the calendar range (tail / outliers).
- **User response (verbatim intent):** Quarterly is the confirmed main resolution. Monthly is a sensitivity-analysis variant only. Use the provisional 35-quarter calendar 2017-Q4 → 2026-Q2 for pilots/diagnostics/tests. Keep empty snapshots. Never silently drop out-of-range records — exclude with explicit reason codes and report in validation/coverage. Classify invalid/corrupted/implausible timestamps separately from valid-but-out-of-range records. Do NOT yet freeze exact start/end/tail policy; require a real-data coverage diagnostic report first. Calendar must be config-driven, not hard-coded. No full-data artifact may be labeled calendar-certified until boundary review is confirmed.
- **Interpreted canonical decision:**
  - **Temporal frequency:** `USER_CONFIRMED_CANONICAL` — **quarterly**.
  - **Provisional pilot calendar:** `2017-Q4 … 2026-Q2` (**35** snapshots) — **diagnostic-only**; not the certified final calendar.
  - **Exact canonical boundaries + final leading/trailing-empty policy + final T:** `REVIEW_REQUIRED_AFTER_REAL_DATA_DIAGNOSTICS` (not finalized).
  - **Monthly calendar:** sensitivity-analysis variant only (never main).
  - **Out-of-range handling:** exclude with explicit reason codes; report counts/percentages/tail distribution; never silent-drop.
  - **Timestamp validity:** invalid/corrupted/epoch-outlier records classified in a **separate** category from valid-out-of-range records.
  - **Engineering constraint:** calendar bounds must be an explicit runtime config; provisional start/end must NOT be hard-coded in reusable source. No `calendar-certified` label on full-data artifacts before boundary review.

## QCAL-B01-PROC — Calendar certification procedure
- **Status:** `USER_CONFIRMED_CANONICAL` (2026-07-28). Does **not** finalize QCAL-B01 numerical boundaries.
- **Canonical procedural policy:**
  1. Internal empty quarterly snapshots must **always** be preserved.
  2. Leading and trailing empty quarters must **not** be finalized until the real-data calendar coverage diagnostic is reviewed.
  3. Final start quarter, final end quarter, and resulting `T` require **explicit user approval** after the diagnostic report.
  4. No full-data time-indexed artifact may be labeled `calendar-certified` before that approval.
  5. Provisional `2017-Q4` through `2026-Q2` remains **diagnostic-only**.
- **Still open (do not finalize now):** numerical start/end, final `T`, final leading/trailing-empty policy.
- **Required diagnostic report (gate before boundary freeze), for both A and B:** (1) row/event counts per quarter; (2) unique active-node counts per quarter; (3) A edge counts per relation×quarter; (4) B valid-text counts per quarter; (5) counts & % before/after provisional range; (6) timestamp distribution of excluded tail records; (7) empty & extremely sparse snapshots; (8) graph-only / text-only / graph+text node coverage per snapshot; (9) evidence of timestamp corruption / epoch outliers; (10) compute & modeling consequences of extending/shortening the calendar.
- **Source evidence:** P-001 §12 (quarterly main, monthly sensitivity, empty bins kept, boundaries fixed before processing); `docs/method/03` S03; `docs/method/10` O-CAL; `docs/data/03` (candidate 35 quarterly bins 2017-Q4→2026-Q2); `docs/data/02` (A span 2017-10-28→2026-05-30, possible epoch outliers); `docs/data/08` (pilot excluded 12,831 post-2026-Q2 rows).
- **Superseded alternatives:** monthly-as-main (rejected → sensitivity only); silent tail-drop (rejected → explicit reason codes); immediate permanent boundary freeze (deferred → review gate).
- **Scientific consequence:** fixes quarterly `T` semantics and reproducible sequence; keeps final coverage empirically justified rather than assumed.
- **Engineering consequence:** calendar is a config parameter; pilots may use the provisional 35-bin calendar; full-data certification blocked on the coverage report + user boundary confirmation.
- **Downstream effects:** snapshot assignment (A06/B03), edge aggregation, masks, GRU sequence length, `L_temp`, evolution analysis, Colab run sizing, artifact certification gating.

---

## Q-DEDUP — Dataset A (and B) duplicate policy
- **Status:** `USER_CONFIRMED_CANONICAL` (policy) + `REVIEW_REQUIRED_AFTER_REAL_DATA_DIAGNOSTICS` (exact signature, thresholds, final numbers) + **QDEDUP-B01-PROC confirmed 2026-07-28**
- **Date proposed:** 2026-07-26 · **Date answered:** 2026-07-26 · **PROC answered:** 2026-07-28
- **Question:** Fix how Dataset A/B duplicate rows and duplicated in-field text are handled before event extraction, edge aggregation, and embedding.
- **User decision — two separate layers:**
  - **Layer 1 — Duplicate tweet RECORDS:**
    - Preserve identical text from **different users** (possible coordinated/campaign behavior; never a deletion rule).
    - Preserve identical text from the **same user at different timestamps** (normally genuine repeated posts) unless strong evidence proves it is the same duplicated source record.
    - Duplicate candidate = **same user + same timestamp + same underlying content + compatible relation/target**.
    - **Dataset B:** exact string tweet IDs MAY be used for detection; repeated same-ID rows are duplicate candidates subject to a conflict check on author/timestamp/text; concordant→collapse to one canonical record; conflicting→retain for review, never silent-discard.
    - **Dataset A:** float tweet ID **forbidden** as exact key. Detect via deterministic **composite signature** including ≥ {exact source user ID, normalized timestamp, normalized text or text hash, relation type, target user ID/set where relevant, referenced-status info when safely available, source-file+row provenance}.
    - Same author+timestamp with **different** text/relation/target = **conflict** (classify, retain, report; do NOT auto-collapse).
    - Concordant duplicate groups → collapse to **one distinct tweet event**, preserve all source-file/row provenance, store collapsed-copy count, keep raw input files unchanged.
    - **Graph event count = distinct interaction events after reconciliation**, not raw repeated rows.
  - **Layer 2 — Duplicated content INSIDE one tweet text field:**
    - Detect extraction/parsing/concatenation artifacts (full text duplicated 2×; long sentence/paragraph/block repeated consecutively; accidentally re-appended quoted content).
    - **Conservative** cleaning: never remove meaningful repetition (campaign slogans, repeated hashtags, "very very", repeated emojis, lyrics/chants/rhetorical repetition, similar-but-non-identical sentences).
    - Operate only on **strong evidence** of a substantial duplicated segment (exact repeated full-text blocks / long consecutive repeated spans); **no aggressive semantic or fuzzy deletion**.
    - Preserve original text in a **read-only private provenance field/artifact**; create a **separate cleaned-text field**; record whether cleaning applied, number/length of removed spans, and hashes of both original and cleaned text; **never** log/place full tweet text in Git-visible/public manifests.
    - **Embeddings use the cleaned text**; original remains recoverable privately for audit.
- **`USER_CONFIRMED_CANONICAL` sub-rules:** identical text across different users preserved; identical text same-user different-timestamp normally preserved; same-user same-timestamp concordant copies collapsed; conflicting candidates retained+flagged; A float IDs forbidden as exact key; B exact string IDs allowed for detection; raw data unchanged; cleaned text stored separately; strong internal duplicated blocks removed pre-embedding; meaningful/linguistic/campaign repetition preserved.
- **`REVIEW_REQUIRED_AFTER_REAL_DATA_DIAGNOSTICS`:** exact A composite event signature; exact threshold for repeated internal-text-span detection; final collapsed-record count; final modified-text count; effect of dedup on graph weights and text coverage. **No aggressive fuzzy-text dedup may be frozen before real-data diagnostics.**
- **Required diagnostics (separate A and B reports before certification):** duplicate candidate groups; concordant groups; discordant groups; rows collapsed; records retained; duplicate rates by source file / author / snapshot / relation; identical-text posts across different users; identical-text same-user different-timestamp; same-user same-timestamp conflicts; #texts with repeated internal blocks; #texts modified by internal cleaning; distribution of removed text lengths; hash/sanitized-only examples in Git-visible reports; effect of dedup on edge counts & text-unit counts. Must explicitly distinguish: duplicated source records vs legitimate repeated tweets vs cross-user coordinated text vs internal text-field corruption.
- **Source evidence:** P-001 §5,§7 (events; aggregate per `(snapshot,relation,src,dst)`; `w=log(1+count)`); `docs/data/02` (≈1.17M A duplicates flagged, 0 removed; tweet `id` float-lossy); `docs/data/08` (B pilot: 747 concordant, 0 conflicting); V3 `temporal_graph_pipeline` provenance aggregation; `docs/method/10` O-DEDUP.
- **Superseded alternatives:** annotate-only counting of raw rows (rejected); using A float tweet id as exact key (forbidden); cross-user text-similarity deletion (forbidden); aggressive fuzzy/semantic text dedup (forbidden as canonical; not frozen).
- **Scientific consequence:** edge counts represent genuine distinct interactions; coordinated-behavior signal preserved; embeddings not distorted by extraction artifacts.
- **Engineering consequence:** two-stage dedup (record-level composite/exact-ID collapse + conservative in-field text cleaning); dual text fields (original private + cleaned); provenance & hashes retained; diagnostics gate certification.
- **Downstream effects:** edge counts & weights (Q-WGT), structural activity features (Q-FEAT), text units & embeddings (Q-TEXT/Q-EMB), coherence metrics, artifact certification.

## QDEDUP-B01-PROC — Dedup certification procedure
- **Status:** `USER_CONFIRMED_CANONICAL` (2026-07-28). Does **not** finalize exact A signature or numeric L2 thresholds.
- **Canonical procedural policy:**
  1. Full-data edge and text artifacts must **not** be labeled `CERTIFIED` before the privacy-safe duplicate diagnostic is completed.
  2. Exact Dataset A composite signature requires **explicit user approval** after diagnostic evidence.
  3. Conservative L2 repeated-text-span thresholds require **explicit user approval** after diagnostic evidence.
  4. Aggressive fuzzy deduplication remains **prohibited**.
  5. Concordant duplicates may collapse with provenance and copy counts preserved.
  6. Discordant records must remain retained and flagged.
  7. Raw source data must remain unchanged.
  8. Effects on graph counts, text counts, pooling groups, relations, targets, and coverage must be reported before certification.
- **Still open (do not finalize now):** exact A composite signature; numeric L2 span thresholds; certified collapsed/modified counts.

---

## Q-WGT — Edge aggregation and edge-weight transformation
- **Status:** `USER_CONFIRMED_CANONICAL`
- **Date proposed:** 2026-07-26 · **Date answered:** 2026-07-26
- **Question:** Fix the edge aggregation key and the count→weight transformation used by the graph encoder.
- **Canonical decision:**
  - **Edge identity:** `(snapshot_id, relation_id, source_idx, target_idx)` — one directed edge, one snapshot, one relation, one source node, one target node.
  - **Count:** `count_raw` = number of distinct interaction events after Q-DEDUP reconciliation.
  - **Weight:** `w_{i→j}^{(t,r)} = log(1 + c_{i→j}^{(t,r)})`, i.e. `weight_log1p = log(1 + count_raw)`, via a numerically stable natural-log `log1p`.
  - **Graph rules:** edges directed; relations kept separate; self-loops excluded and reported; multi-target mention events expanded to one event per valid target; `count_raw` retained for auditability; `weight_log1p` is the primary model edge weight.
  - **Required edge fields:** `snapshot_id, relation_id, source_idx, target_idx, count_raw, weight_log1p`, plus provenance/validation metadata.
  - **Permitted weight ablations:** binary edge weight; raw count edge weight; `log1p` count edge weight (canonical main).
- **Diagnostics scope:** real-data diagnostics may validate count distributions, hub concentration, and implementation correctness only; they do **not** reopen the scientific weight definition.
- **Source evidence:** P-001 §7 (`w=log(1+count)`, store raw count); V3 aggregation; `docs/method/03` S05; `docs/method/10` O-WGT.
- **Scientific consequence:** edge weight represents leakage-safe, in-snapshot directed interaction intensity, compressed for heavy tails; comparable across relations/snapshots and baselines.
- **Engineering consequence:** sparse per-`(snapshot,relation)` edge lists with `count_raw` + `weight_log1p`; trivial compute.
- **Downstream effects:** directed GraphSAGE messages, edge gate, structural metrics, `weight_1` vs `log1p` ablation.

---

## Q-FEAT — Structural node features
- **Status:** `USER_CONFIRMED_CANONICAL` (2026-07-26). Additional normalization `UNRESOLVED_UNDER_Q_HPARAM`.
- **Date proposed:** 2026-07-26 · **Audit added:** 2026-07-26 · **Confirmed:** 2026-07-26
- **Audit document:** `docs/method/14_q_feat_structural_feature_audit.md` (full 31-column assessment).
- **Canonical decision:**
  - **`F_struct = 17`.** Tensor `X_struct[t, node_idx, feature_idx] ∈ ℝ^{T×N×17}`, N=16,736, all frozen nodes present every snapshot, dtype **float32**.
  - **Relation order (canonical):** `0=mention, 1=retweet, 2=reply, 3=quote`; relations never merged.
  - **Exact ordered schema (17):** per relation {`out_degree`, `in_degree`, `out_strength_log1p`, `in_strength_log1p`} for mention(0–3), retweet(4–7), reply(8–11), quote(12–15); then `16 = tweet_count_log1p`. Ordering is explicit, versioned, and part of the artifact contract; never silently reordered.
  - **Degree:** `out_degree_i^(t,r)=|{j : c_{i→j}^{(t,r)}>0}|`, `in_degree_i^(t,r)=|{j : c_{j→i}^{(t,r)}>0}|`. Distinct neighbors; repeated interactions do not increase degree; **no log1p on degree** (any later scaling → Q-HPARAM); nonnegative integers before float32.
  - **Strength (disambiguated):** `out_strength_log1p_i^(t,r)=log1p(Σ_j c_{i→j}^{(t,r)})`, `in_strength_log1p_i^(t,r)=log1p(Σ_j c_{j→i}^{(t,r)})`. This is **log1p(sum of count_raw)**, NOT `Σ_j log1p(c)` and NOT the sum of edge weights. Natural-log `log1p`.
  - **`tweet_count_log1p_i^(t)=log1p(#distinct authored tweets after Q-DEDUP)`** — counts a tweet once regardless of edges produced (originals, out-of-universe targets, multi-target all count once); duplicate records must not inflate; float tweet id not sole dedup key.
  - **`c_{i→j}^{(t,r)}`** = `count_raw` for aggregated edge `(t,r,i,j)` after Q-DEDUP; directed, snapshot- & relation-specific; self-loops excluded before degree/strength; multi-target mentions = one event per valid target.
  - **`active` is NOT a feature** and **`n_active_relations` is NOT a feature** — both excluded from `X_struct`.
- **Separate mask:** `struct_active_mask[t,node_idx] ∈ {0,1}^{T×N}` (boolean). True iff `tweet_count_raw>0` OR ≥1 incoming canonical edge OR ≥1 outgoing canonical edge. Inactive → all-17 zeros + mask False; tweets-but-no-edges → 16 zeros, positive tweet_count, mask True; only-incoming → in features positive, mask True; empty snapshot → all zeros, mask all False. Dataset-A structural activity only. **`model_active_mask = struct_active OR node_text_available` (QACT-01)**; GRU per **QGRU-01**.
- **Excluded from primary:** `total_in/out_events_log1p`, `n_active_relations`, `active` (numeric), engagement-derived, follower/following, account metadata, identifiers, duplicate flags, PageRank, HITS, k-core, clustering, common-neighbor stats, relation entropy, relation-share proportions, recency, activity balance, reciprocity, active-days fraction, media indicators, hashtag counts. Non-primary items may remain diagnostics/ablation only and must not alter F_struct=17.
- **Normalization:** feature semantics fixed (degree raw; strength log1p; tweet_count log1p). Additional train-time robust scaling = **QHP-02** (`USER_CONFIRMED`); do not standardize the canonical raw `X_struct` artifact; store training-only scaler metadata (val/test never influence fitted stats).
- **Required artifacts (eventual):** `X_struct`, `struct_active_mask`, ordered 17-name list, feature-schema version, relation map+order, snapshot map, node-map hash, Q-DEDUP contract hash/provenance, dtype/shape metadata, validation summary.
- **Required validations (future tests):** shapes `(T,16736,17)`/`(T,16736)`; finite; nonnegative degree/strength/tweet_count; inactive→zero row + mask False; only-incoming→active; tweets-no-edges→active; multi-target changes edge features not tweet_count; dedup prevents inflation; self-loops don't affect degree/strength; deterministic ordering; identical inputs→identical artifact hashes.
- **Scientific consequence:** compact, fully leakage-safe, directed multiplex activity/role signal; no engineered centrality (learned by GraphSAGE); no leaky engagement/account fields.
- **Engineering consequence:** dense `float32` `[≈35,16736,17]` (~40 MB) + boolean mask; cheap edge/row aggregations; explicit versioned schema.
- **Source evidence:** P-001 §6/§8; `docs/method/03` S06/S08; `docs/data/02`; `docs/data/03`; `docs/method/10` O-FEAT; audit `docs/method/14`.

---

## Q-TEXT — Atomic node-text and edge-text units
- **Status:** `USER_CONFIRMED_CANONICAL` (2026-07-26).
- **Contract document:** `docs/method/15_q_text_text_unit_contract.md`.
- **Canonical decision:**
  - **Role separation:** Dataset B text = **node-snapshot** semantics; Dataset A event text = **edge/interaction** semantics. Not interchangeable; **no tweet-level A↔B join** (prohibited).
  - **Node text (Dataset B):** atomic unit = **one distinct authored tweet**. `z_node_q = TextEncoder(cleaned_text_q)`. `T_i^(t) = mean_{q ∈ Q_i^(t)} z_node_q`, where `Q_i^(t)` = retained distinct tweets authored by node i with `created_at` in snapshot t. Tensor `X_node_text[t,node_idx,:] ∈ ℝ^{T×N×D_text}`, N=16,736, `D_text` from Q-EMB. **Per-tweet embed → mean-pool**; never concatenate long strings; equal weights.
  - **Edge text (Dataset A):** atomic unit = **one distinct interaction-event tweet**. `z_edge_q = TextEncoder(cleaned_text_q)`. `E_{i→j}^{(t,r)} = mean_{q ∈ Q_{i→j}^{(t,r)}} z_edge_q`, where `Q_{i→j}^{(t,r)}` = distinct A events from i→j in snapshot t, relation r. Relations/directions/snapshots separate; single-event edge = that event's embedding. Multi-target mention: same cleaned text may map to each target-specific event; provenance shows one original tweet; caching avoids duplicate embedding work.
  - **Embedding input:** `cleaned_text` only (Q-DEDUP Layer-2); original text private/read-only, never default embed field.
  - **Temporal rule:** strict **in-snapshot only** — no earlier/later/cumulative/future text; no full multi-year history per snapshot.
  - **Dedup before aggregation:** A-event and B-tweet dedup precede embedding/pooling; duplicates must not inflate embedding calls, pooling, tweet/event counts. Embedding cache keyed by **stable content + preprocessing provenance**, NOT float A tweet id alone.
  - **Masks:** see **Q-MISS** (`docs/method/17`) — exact zero + boolean `node_text_available_mask` / `edge_text_available_mask` (edge masks aligned to canonical edge order); missing ≠ fabricated semantics.
  - **Downstream:** `T_i^(t)` → node-text variant, initial representation, `L_sem`, temporal node state. `E_{i→j}^{(t,r)}` → learned edge gate / edge-aware message passing (gate may see relation id, edge weight, source/target reps, edge-text vector, edge-text availability). Model does NOT hard-code semantic rules (e.g. "hostile→different community"); the gate **learns** how content modifies structural messages.
  - **Implementation order:** node-text path **first** (certify), edge-text path **second** — both **required**; edge text is not optional and must not be indefinitely deferred. Staging is engineering-only.
- **Deferred from Q-TEXT:** `D_text`, encoder family/id, tokenizer, max length, truncation, normalization, pooling dtype → **Q-EMB**. Missing-text → **Q-MISS** (now `USER_CONFIRMED_CANONICAL`).
- **Source evidence:** P-001 §9/§10; `docs/method/03` S07; `docs/method/10` O-TEXT/O-EDGE; `docs/data/04,05,06`; Q-DEDUP Layer-2.
- **Scientific consequence:** clean separation of node vs edge semantics; leakage-safe temporal text; edge text can refine/correct structural evidence via a learned gate.
- **Engineering consequence:** per-tweet embedding + mean-pool for both paths; tweet/event-level caching; dedup-before-embed; two masks.

---

## Q-EMB — Embedding model and full embedding contract
- **Status:** `OPEN_PENDING_PILOT_AND_USER_CONFIRMATION`. Full spec: `docs/method/16_q_emb_embedding_contract_and_pilot_spec.md`.
- **Date proposed:** 2026-07-26 · **User direction received:** 2026-07-26
- **User decision boundaries:**
  - **Embedding family = Qwen3 Embedding ONLY** (`USER_CONFIRMED`). No non-Qwen embedding model may be named/introduced/compared anywhere in current TDMEC docs (method, decisions, handoff, comparison tables, alternatives, rejected-options, historical notes).
  - **Preferred checkpoint `Qwen/Qwen3-Embedding-4B`** = `PROVISIONAL_PENDING_PILOT`. If it fails acceptance gates, report the limiting factor — **no auto-substitution**; any change within the Qwen3 family needs a new explicit user decision.
  - **`D_text`** = `PROVISIONAL_PENDING_PILOT` (pilot native full dim + one practical reduced + one storage-efficient, only officially supported dims; do not fabricate).
  - **Instruction:** shared instruction = `PRIMARY_PILOT_CANDIDATE`; no-instruction = `PILOT_COMPARATOR`; separate node/edge = `OPTIONAL_PILOT_COMPARATOR`; exact wording = `PROVISIONAL_PENDING_PILOT`. Must request semantic representation only (no community rules).
  - **Stage-A token pooling** = `PENDING_OFFICIAL_VERIFICATION` (verify last-token pooling, padding side, mask handling, official per-tweet L2 norm). **Stage-B cross-tweet mean pooling** = `USER_CONFIRMED_CANONICAL_UNDER_Q_TEXT` (unchanged).
  - **Per-tweet L2 normalization** = `PENDING_OFFICIAL_VERIFICATION`. **Final pooled-vector L2 normalization (N2)** = `PRIMARY_PILOT_HYPOTHESIS` (test vs N1; store pre-norm pooled norm as diagnostic only, not a feature).
  - **`max_length`** = `PROVISIONAL_PENDING_DIAGNOSTICS_AND_PILOT` (token-length diagnostics per dataset; atomic unit = one tweet, so context length ≠ max_length).
  - **GPU** = `RUNTIME_DETECTED_COLAB_OR_KAGGLE` (hardware-agnostic; probe batch size; safe OOM backoff; fp16/bf16 only where supported; resumable; portable).
  - **Storage/resume:** pilot evaluates **C streaming sums+counts** vs **D hybrid resumable** quantitatively (QEMB-P03); permanent/temp full caches = estimates unless reduced dim makes permanent plausible; production policy POST_PILOT.
- **Batch-2 pilot design (USER_CONFIRMED 2026-07-28):** see QEMB-P01/P02/P03 below. Full gates/matrix in `docs/method/16` §§14–14c.
- **QEMB-I1 (USER_APPROVED_PILOT_CANDIDATE 2026-07-28):** shared instruction wording approved for the bounded pilot comparison against no-instruction only. Exact final production instruction remains **QEMB-X03 POST_PILOT**.
- **Execution constraints:** pilot notebook **spec only**; **no execution/download/full-data/commit** without explicit authorization; I1 wording is approved for pilot comparison but provisional values must not enter any final tensor contract.
- **Source evidence:** P-001 §9 (per-tweet encode → mean-pool; frozen encoder); Dataset B largely **Persian** (multilingual required); Q-TEXT (per-tweet unit + mean-pool); official Qwen3-Embedding model card (to verify in pilot).
- **Scientific consequence:** semantic space determined empirically for Persian/multilingual social text; N1/N2 decision affects whether magnitude (coherence) enters the model.
- **Engineering consequence:** hardware-agnostic, resumable embedding harness; `D_text` drives all text-tensor shapes, storage, and training memory.

## QEMB-P01 — Pilot acceptance gates
- **Status:** `USER_CONFIRMED_PILOT_DESIGN` (Option C, 2026-07-28). Does **not** finalize Q-EMB.
- **Hard gates:** load OK; zero NaN/Inf; determinism (cosine abs Δ ≤ 1e-4); stable batch / no unrecoverable OOM; exact pooling recompute; resume no double-count; Q-MISS mask/zero invariants; no semantic collapse; expected norms; Persian checks OK; reduced-dim mean Overlap@k ≥ 0.9.
- **Neighbor-overlap (explicit):** k=10; Overlap@k = \|NN_k(native)∩NN_k(reduced)\|/k; aggregate = mean over queries; sample ≤ 1000 stratified A/B queries (or all if fewer).
- **Truncation:** ≤ 1% primary **target**; report overall + by A/B, language, length strata (not sole hard fail of the pilot).
- **Report-only:** full-corpus runtime and storage estimates (GPU/batch/dtype/length/storage dependent).

## QEMB-P02 — Pilot comparator matrix
- **Status:** `USER_CONFIRMED_PILOT_DESIGN` (Option A, 2026-07-28).
- **Mandatory:** no-instruction vs shared **I1**; 4B only; native + up to two official reduced dims; N1 vs N2; max_length after token-length diagnostics; stratified A/B sample; sequential comparisons preferred over full Cartesian product.
- **Optional after mandatory:** separate node/edge instructions if compute remains.
- **Not in mandatory first pilot:** I2, I3.
- **I1 wording:** `USER_APPROVED_PILOT_CANDIDATE` (2026-07-28) — see QEMB-I1. Exact final instruction remains POST_PILOT (QEMB-X03).

## QEMB-P03 — Storage/resume evaluation plan
- **Status:** `USER_CONFIRMED_PILOT_DESIGN` (Option B, 2026-07-28).
- **Quantitative:** C streaming sums+counts vs D hybrid resumable (correctness, peak/final storage, write overhead, interruption recovery, resume, double-count protection, manifest integrity, aggregate equivalence).
- **Estimates only:** permanent per-tweet and temporary full sharded caches (unless reduced dim makes permanent plausible).
- **Production policy:** POST_PILOT. Hybrid = hypothesis only.

## Q-MISS — Missing-text behavior
- **Status:** `USER_CONFIRMED_CANONICAL`. Full contract: `docs/method/17_q_miss_missing_text_contract.md`.
- **Date proposed:** 2026-07-26 · **Date answered:** 2026-07-26
- **Question:** Fix missing-text storage, masks, model input, loss gating, and coverage when node-snapshot or edge text is absent/invalid after Q-TEXT/Q-DEDUP.
- **User response (verbatim intent):** Canonical policy **M1** — exact zero vector + explicit boolean availability mask; apply symmetrically to B node-snapshot text and A edge text; no learned missing embedding; no dropping nodes/edges for missing text; no replacing missing current-snapshot text with previous-snapshot text; Fusion MLP and Edge Gate must receive masks explicitly; missing edge text preserves structural path; `L_sem` only on mask=True with safe zero-eligible batches; keep `struct_active_mask` / `node_text_available_mask` / `edge_text_available_mask` separate; store `node_valid_text_count` and `edge_valid_text_count` as required metadata (not model features unless new decision); privacy-safe coverage reports required; exact-zero invariant after load/dtype/batch/serialize; no L2-norm of missing zeros (Q-EMB norms only available vectors).
- **Interpreted canonical decision:**
  - **Policy:** `EXACT_ZERO_VECTOR_PLUS_BOOLEAN_MASK` for both paths.
  - **Node:** `node_text_available_mask[t,i]=True` iff `node_valid_text_count[t,i]>0` (retained B tweets by i in t, deduped, valid non-empty cleaned_text, embedding-eligible); else mask=False and `T_i^(t)=exact zero ∈ ℝ^{D_text}`; when available, `T` = sum/count mean.
  - **Edge:** per canonical edge index; `edge_text_available_mask=True` iff `edge_valid_text_count>0`; else False + exact zero; structural edge/`count_raw`/weight/message path unchanged.
  - **Invalid text** (null/empty/whitespace/empty-after-clean/rejected) → missing; no fabricated text.
  - **Model:** Fusion input `[X_struct, X_node_text, node_text_available_mask]`; Edge Gate `[structural…, edge_text_vector, edge_text_available_mask]`.
  - **`L_sem`:** mask-True only; mask-aware denominators; zero-eligible → defined zero contribution, no NaN/Inf.
  - **Prohibited:** learned missing embedding; drop for missing text; sparse-only as primary; text carry-forward; inferring masks from each other; concatenating valid-text counts as features without new decision.
  - **Artifacts:** `X_node_text[T,N,D_text]`, `node_text_available_mask[T,N]` bool, `node_valid_text_count[T,N]` int; edge vectors/masks/counts aligned to canonical edge order.
- **Source evidence:** P-001 (zero+mask; keep masks); Q-TEXT §7 principles; Q-FEAT separate `struct_active_mask`.
- **Deferred:** Q-EMB normalization of available vectors. **QACT-01 / QGRU-01 / QGATE-01** now canonical (see below).
- **Scientific consequence:** missingness ≠ neutral semantics ≠ structural inactivity; semantic objectives not biased by fabricating zeros as content.
- **Engineering consequence:** dense primary tensors with exact-zero rows; coverage + count metadata for verification/resume.

## QREL-01 — Canonical relation-ID mapping
- **Status:** `USER_CONFIRMED_CANONICAL` (2026-07-28)
- **User confirmation:** Option A — permanent immutable IDs.
- **Canonical maps:**
  - `relation_to_id = {mention:0, retweet:1, reply:2, quote:3}`
  - `id_to_relation = {0:mention, 1:retweet, 2:reply, 3:quote}`
  - `relation_count = 4`
  - `relation_order = [mention, retweet, reply, quote]`
- **Binding scope:** edge artifacts, feature channels, relation masks/embeddings/GNN modules, in/out aggregation, fusion, edge-text, eval-by-relation, configs, schemas, manifests, validation, docs, handoffs, tests, future code.
- **Forbidden:** alternate orders; dynamic/alphabetical/dict-insertion remapping; silent library remaps. Mapping must be stored explicitly and validated.
- **Note:** Q-FEAT feature blocks already assumed this order; QREL-01 freezes it project-wide.

## QSELF-01 — Self-loop exclusion
- **Status:** `USER_CONFIRMED_CANONICAL` (2026-07-28)
- **User confirmation:** Option A — exclude all author-to-self events before edge aggregation.
- **Rule:** for every expanded interaction event, if `source_idx == target_idx` → exclude from canonical edges; do not enter `count_raw`, `weight_log1p`, degree, strength, relation aggregation, or edge-text aggregates; report aggregate exclusion counts.
- **Multi-target mentions:** exclude only expanded targets where source=target; keep other valid targets from the same original event.
- **Prohibited transforms:** do not convert self-loops into another relation, a node feature, a special edge type, or an observed GNN self-loop artifact. Internal neural self-transforms (if any) are conceptually separate from observed graph edges.
- **Exceptions:** `NONE`.
- **Reporting:** self-loop exclusion counts `REQUIRED_IN_AGGREGATE_VALIDATION_REPORTS`.

## QACT-01 — model_active_mask
- **Status:** `USER_CONFIRMED_CANONICAL` (2026-07-28)
- **Formula:** `model_active_mask[t,i] = struct_active_mask[t,i] OR node_text_available_mask[t,i]`
- **Reason (user):** Temporal evolution should incorporate semantic-only changes, while keeping structural and text masks separate.
- **Constraints:** `struct_active_mask`, `node_text_available_mask`, and `edge_text_available_mask` remain separate stored artifacts; none is inferred from another. Edge-text availability does **not** enter this formula (edge text exists only on existing edges, which already imply structural incidence).
- **Controls:** GRU update eligibility (QGRU-01) and related temporal-loss eligibility that depends on model activity. Does **not** control graph topology, `X_struct` construction, or text tensor values.

## QGRU-01 — GRU update vs carry
- **Status:** `USER_CONFIRMED_CANONICAL` (2026-07-28)
- **Rule:** If `model_active_mask[t,i]`: `s_i^(t) = GRU(z_i^(t), s_i^(t-1))`; else `s_i^(t) = s_i^(t-1)` (exact hidden-state carry; no GRU step). `s_i^(0) = 0`.
- **Reason (user):** Preserves temporal memory and avoids artificial updates or resets.
- **Clarifications:** Carry copies the **hidden state**, not previous tweet embeddings (text carry-forward remains prohibited under Q-MISS). BPTT truncation length remains a later training decision.

## QGATE-01 — Edge-context and Edge-Gate inputs
- **Status:** `USER_CONFIRMED_CANONICAL` (2026-07-28)
- **MLP_e input:** `[e_r, weight_log1p, E_edge, edge_text_available_mask]` → `g`
- **MLP_g input:** `[h_source, h_target, g]` → `γ = σ(MLP_g(...))`
- **`edge_valid_text_count`:** metadata only — not a model feature.
- **Reason (user):** Interaction intensity, semantic content, and missingness are already represented without introducing coverage shortcuts.
- **Consistency:** Aligns with Q-MISS (counts not features; structural path preserved when mask=False and `E=0`).

## QEMB-I1 — Shared pilot instruction wording
- **Status:** `USER_APPROVED_PILOT_CANDIDATE` (2026-07-28). **Not** the final production instruction.
- **Approved pilot candidate (verbatim):**
  > Represent the topic, stance, sentiment, and social meaning of this social-media post for temporal community analysis.
- **Scope:** shared instruction candidate for the bounded pilot comparison against the **no-instruction** condition only.
- **Still open:** QEMB-X03 final production instruction policy + wording (`POST_PILOT`).

## QART-01-FRAME — Artifact certification hard vs warning framework
- **Status:** `USER_CONFIRMED_CANONICAL` (2026-07-28). Does **not** finalize numeric text/activity coverage thresholds. Full-run production certification remains evidence-gated (`QART-01` numeric thresholds / full-run = open).
- **Canonical certification framework:**
  1. Contract-correctness checks are **mandatory hard gates**.
  2. Text and activity coverage must **always** be reported.
  3. Coverage normally remains a **data-quality warning** until evidence-based numeric thresholds are explicitly approved.
  4. Complete absence of an artifact or semantic path required by the **primary TDMEC method** is a **hard failure**.
     - Example: zero usable edge-text coverage cannot be certified as the full primary TDMEC configuration.
     - The same graph artifacts may still be separately certified for an **explicitly defined graph-only ablation**.
  5. No artifact may be labeled `CERTIFIED` unless all applicable checks pass, including:
     - schema validation
     - shape validation
     - dtype validation
     - node count and node-order validation
     - edge-order validation
     - canonical relation-map validation
     - canonical self-loop invariant
     - edge / edge-text / mask / count alignment
     - exact-zero-when-mask-false invariants
     - `model_active_mask` consistency
     - NaN and Inf checks
     - checksum validation
     - manifest validation
     - configuration and version provenance
     - deterministic ordering
     - resume integrity
     - double-count protection
- **Still open:** numeric coverage thresholds for hard-fail (await real-data evidence); full-run production stamp.

## QFUS-01 — No-relation fusion fallback
- **Status:** `USER_CONFIRMED_CANONICAL` (2026-07-28). Classification: `IMMUTABLE_METHOD_CONTRACT`.
- **Rule:** If no relation-specific representation is available for a node-snapshot (∑_r a_i^(t,r)=0): `z_i^(t) = h_i^(0,t)`.
- **Mask constraint:** `struct_active_mask` is **not** redefined or overwritten by relation availability. It remains exactly:
  `struct_active_mask[t,i] = (tweet_count[t,i] > 0) OR (any canonical incoming or outgoing edge exists)`.
- **Consequence:** fallback supports structural-only, text-only, and fully inactive node-snapshots without conflating masks (`struct_active`, `node_text_available`, `model_active` stay separate).

## QENC-01 — Directed mean aggregation and fanout
- **Status:** `USER_CONFIRMED` (2026-07-28).
- **Immutable method contract:** relation-specific and direction-specific **mean** aggregation over gated, weighted neighbor messages; incoming and outgoing neighborhoods remain separate for every relation.
- **Primary experimental default:** training fanout `[15]` for the primary one-layer encoder, applied **per relation and direction**; if fewer neighbors available, use all.
- **Classification:** fanout = `PRIMARY_EXPERIMENTAL_DEFAULT` (configurable scalability setting, not an immutable method claim).
- **Val/inference:** full-neighbor aggregation when feasible; otherwise deterministic seeded sampling and report it.
- **Ablation:** sum aggregation = `OPTIONAL_ABLATION`.

## QENC-02 — GNN depth L
- **Status:** `USER_CONFIRMED` (2026-07-28).
- **Primary experimental default:** `L = 1`.
- **Required sensitivity:** `L = 2`.
- Deeper than two is not part of the primary sensitivity set unless later evidence justifies it.

## QPROJ-01 — Relation embedding and semantic projections
- **Status:** `USER_CONFIRMED` (2026-07-28).
- **Immutable method contract:** common semantic projection space for `L_sem` via learned `P_z` and `P_T_node`.
- **Primary experimental defaults:** `d_rel = 16`; `d_sem = d_h` (with primary `d_h=64` → `d_sem=64`).
- **Maps:** `P_z: ℝ^{d_h} → ℝ^{d_sem}`; `P_T_node: ℝ^{D_text} → ℝ^{d_sem}`.
- **Edge text:** uses its own learned projection inside `MLP_e`; does **not** share `P_T_node`.
- **`D_text`:** remains configurable and POST_PILOT; changing `D_text` changes only text-projection input dims, not the architecture contract.
- **Optional sensitivity:** `d_rel = 32`.

## QHP-01 — Hidden dimension d_h
- **Status:** `USER_CONFIRMED` (2026-07-28).
- **Primary experimental default:** `d_h = 64`.
- **Required sensitivity:** `d_h ∈ {32, 128}`.
- Shared across graph/fused/GRU/prototype dims per architecture contract.

## QHP-02 — Train-time structural feature scaling
- **Status:** `USER_CONFIRMED` (2026-07-28).
- **Immutable method contract:** training-only fit; no leakage from val/test/inference into scaler stats.
- **Primary experimental default:** train-time **robust per-feature scaling**.
- **Rules:**
  - fit only on training snapshots
  - fit only on structurally active training node-snapshot rows (`struct_active_mask=True`)
  - freeze and reuse the same statistics for validation, test, and inference
  - store scaler parameters and training-period provenance
  - force `struct_active_mask=False` rows back to the **exact zero vector** after transformation
  - preserve the canonical raw `X_struct` artifact unchanged
  - implement scaling only as a training-pipeline transformation
- **Winsorization:** not automatic; enable only after training-data diagnostics show extreme outliers; thresholds fitted on training data only (`DIAGNOSTIC_CONDITIONAL`).
- **Ablation:** no-scaling = `OPTIONAL_ABLATION`.

## QHP-03 — Community count K
- **Status:** `USER_CONFIRMED` (2026-07-28).
- **Immutable method contract:** `K` fixed within each run; do **not** select `K` using test snapshots.
- **Primary experimental default:** `K = 10`.
- **Required sensitivity:** `K ∈ {5, 10, 15, 20, 30}`.
- Any data-driven selection must use training or validation evidence only and must be reported separately from the preregistered `K=10` result.

## QHP-04 — Prototype initialization
- **Status:** `USER_CONFIRMED` (2026-07-28).
- **Immutable method contract:** training-only states; no temporal leakage from val/test.
- **Primary experimental default:** KMeans++ on pretrained temporal states `s`.
- **Eligible states:** training snapshots only; node-snapshot states where `model_active_mask=True`.
- **Sampling:** snapshot-balanced so dense snapshots do not dominate initialization.
- **Primary settings:** `n_init = 20`; fixed and recorded seed; KMeans++; deterministic preprocessing.
- **Scalability:** MiniBatchKMeans only as a documented scalability implementation when necessary.
- **After init:** prototypes remain trainable.
- **Empty cluster:** reinitialize deterministically using a high-distance training state; report the event.
- **Ablation:** random prototype initialization = `OPTIONAL_ABLATION`.

## QVAR-01 — Variant reporting names
- **Status:** `USER_CONFIRMED_CANONICAL` (2026-07-28). Classification: `IMMUTABLE_REPORTING_CONTRACT`.
- **Frozen names:**
  - **TDMEC-G** — graph-only ablation
  - **TDMEC-NT** — graph + node text, without edge text
  - **TDMEC** or **TDMEC-Full** — complete primary method (graph + node text + edge text)
  - **TDMEC-ET** — reserved for a graph + edge-text-only ablation **if** that variant is actually implemented; **must not** name the full method
- **Reason:** `TDMEC-ET` as full-method label is ambiguous about node-text inclusion.

## QLOSS-01 — L_struct mask rate and negatives
- **Status:** `USER_CONFIRMED` (2026-07-28).
- **Immutable:** masked positive edges excluded from encoder input.
- **Primary experimental default:** mask **15%** of observed canonical edges; **3** negatives per masked positive; uniform without replacement; same snapshot and relation; directed; exclude self-loops and all observed canonical edges.
- **Optional ablations:** degree-aware / hard negatives; `L_relation_type` auxiliary loss (not primary).

## QLOSS-02 — L_reg margin
- **Status:** `USER_CONFIRMED` (2026-07-28).
- **Immutable:** hinge separation without equal-size forcing.
- **Primary:** squared Euclidean `‖μ_k−μ_l‖²`; `m=1.0`.
- **Required sensitivity:** `m ∈ {0.5, 1.0, 2.0}`.

## QLOSS-03 — L_temp eligibility
- **Status:** `USER_CONFIRMED_CANONICAL` (2026-07-28). Classification: `IMMUTABLE_METHOD_CONTRACT`.
- **Rule:** apply `L_temp` for node `i` between `t−1` and `t` iff `model_active_mask[t−1,i]` AND `model_active_mask[t,i]`.
- Do **not** use `struct_active_mask` as eligibility; do **not** penalize transitions with an inactive endpoint.

## QLOSS-04 — λ coefficients and staged activation
- **Status:** `USER_CONFIRMED` (2026-07-28).
- **Immutable:** hierarchical loss composition.
- **Primary targets:** `λ_struct=λ_sem=λ_cluster=1.0`, `λ_reg=λ_temp=0.1`.
- **Stages:** (1) pretrain: cluster/reg/temp = 0; (2) joint warmup: cluster=1.0, reg=0.1, temp=0; (3) temporal: linear ramp `λ_temp` 0→0.1 over first **20%** of temporal epochs (no abrupt step).
- Report unweighted magnitude of every loss component.

## QTR-01 — Optimizer, epochs, early stopping
- **Status:** `USER_CONFIRMED` (2026-07-28). Classification: `PRIMARY_EXPERIMENTAL_DEFAULT`.
- AdamW; lr `5e-4`; weight decay `1e-4`; grad clip norm `1.0`.
- Max epochs: pretrain **100**; joint+temporal **200**.
- Early stop: per epoch; patience **20**; min_delta `1e-4`; restore best val checkpoint; never use test for stopping/selection.
- Pretrain monitors val representation loss; joint uses Batch-6 criterion when ready (until then weighted val total loss provisional).

## QTR-02 — Chronological train/val/test split
- **Status:** `USER_CONFIRMED` (2026-07-28).
- **Immutable:** contiguous chronological split; no random temporal split; no future leakage; no val/test fitting.
- **Primary allocation:** ~70% earliest train / ~15% val / ~15% latest test.
- **Exact quarter boundaries:** `POST_CALENDAR_CERTIFICATION` (after QCAL-B01).
- Scaling + KMeans train-only; val for early stop/hparams only; chronological carry of hidden states allowed; rolling-origin = optional later sensitivity.

## QTR-03 — BPTT truncation
- **Status:** `USER_CONFIRMED` (2026-07-28).
- **Immutable:** chronological hidden-state carry (no reset between windows).
- **Primary:** length **3**; no detach inside window; detach only at window boundaries; carry final state into next window.
- **Required sensitivity:** lengths **2** and **4** (if memory permits).
- Detach-every-snapshot = optional computational ablation.

## QTR-04 — Seed policy
- **Status:** `USER_CONFIRMED_CANONICAL` (2026-07-28). Classification: `IMMUTABLE_EVALUATION_AND_REPRODUCIBILITY_CONTRACT`.
- 5 seeds for TDMEC-Full, principal ablations, and stochastic learned baselines when feasible; ≥3 for one-factor sensitivity; 1 seed only for smoke/diagnostics (never final claims).
- Control/record: model init, KMeans, neighbor sampling, edge masking, negative sampling, batching/workers.
- Report mean, std, and individual-seed results; shared predefined seed set across comparable methods when possible.

## QEVAL-01 — Metrics and model selection
- **Status:** `RESOLVED_BY_AUTHORIZED_ARCHITECT` (2026-07-28). Full text: `docs/method/18`.
- Mandatory dimensions: structural (dir. weighted modularity + symmetrized conductance), temporal (AMI/NVI), semantic (cosine silhouette on shared frozen eval encoder), predictive (future-link relation-macro AP primary), stability, efficiency; ARI ground-truth-only.
- Early stop: smoothed phase-appropriate val loss. Select: val relation-macro AP + hard non-collapse + deterministic tie-break. No test selection.
- Formula attributions recorded in `18` (published exact / equivalent / project extension).

## QEVAL-02 — Baseline registry
- **Status:** `RESOLVED_BY_AUTHORIZED_ARCHITECT` (2026-07-28). Full registry: `docs/method/19`.
- No candidates deleted; extended with Infomap, multislice, Peixoto SBM, DySAT, attributed-multiplex, etc.
- **Implement only Phase 10** after TDMEC-Full primary success.

## QEVAL-03 — Ablation registry
- **Status:** `RESOLVED_BY_AUTHORIZED_ARCHITECT` (2026-07-28). Full registry: `docs/method/20`.
- Separates ABL-REL-AGG vs ABL-FUS-MEAN; retains G/NT/ET/Full ladder and Batch 4–5 sensitivities.
- **Implement only Phase 11**.

## QMLP-01 — MLP defaults
- **Status:** `RESOLVED_BY_AUTHORIZED_ARCHITECT` (2026-07-28).
- Primary: one hidden layer width `d_h`, ReLU, dropout 0, no LN/BN, no residual, Kaiming init.
- MLP_x / MLP_e / MLP_g / MLP_r / MLP_dec as in `docs/method/05`.
- Classification: architecture pattern `PRIMARY_EXPERIMENTAL_DEFAULT`; relation-specific params `IMMUTABLE` with encoder contract.
- Optional ablation: dropout 0.1.

## QDEC-01 — Structural decoder
- **Status:** `RESOLVED_BY_AUTHORIZED_ARCHITECT` (2026-07-28).
- `MLP_dec([s_i,s_j,e_r,E',m_e,w̃])` + BCE-with-logits; negatives use zero text/mask/weight.
- Classification: decoder I/O contract `IMMUTABLE_METHOD_CONTRACT` relative to Q-MISS/QLOSS-01; MLP width `PRIMARY_EXPERIMENTAL_DEFAULT`.

## QCLU-01 — DEC target update
- **Status:** `RESOLVED_BY_AUTHORIZED_ARCHITECT` (2026-07-28).
- Update P once per epoch (primary). Optional every-N-batch ablation.
- Classification: `PRIMARY_EXPERIMENTAL_DEFAULT`.

## QBATCH-01 — Mini-batch and epoch
- **Status:** `RESOLVED_BY_AUTHORIZED_ARCHITECT` (2026-07-28).
- Node mini-batches covering all N per snapshot; epoch = chronological pass over train snapshots; `B_nodes` runtime-probed (start 512).
- Classification: unit/epoch `PRIMARY_EXPERIMENTAL_DEFAULT`; batch size `RUNTIME_AUTOMATIC_SETTING`.

## QPHASE-01 — Joint vs temporal epoch split
- **Status:** `RESOLVED_BY_AUTHORIZED_ARCHITECT` (2026-07-28).
- Within 200: joint 80 (λ_temp=0) + temporal 120 with λ_temp ramp over first 24.
- Classification: `PRIMARY_EXPERIMENTAL_DEFAULT`.

