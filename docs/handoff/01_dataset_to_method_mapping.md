# 01 — Dataset → Method Mapping

**Purpose.** Map each dataset to its scientific role and answer Primary Questions
**A** (Dataset A / graph) and **B** (Dataset B / text) from repository evidence
plus **current TDMEC canonical decisions** (`docs/method/12`, `03`, `14`, `15`, `16`).
Where an item still depends on an open decision (e.g. Q-EMB pilot config),
it is marked accordingly and deferred to
`05_open_decisions_before_implementation.md`.

> **Authoritative contracts.** Current-TDMEC decisions for D1/D2, Q-CAL (frequency),
> Q-DEDUP (policy), Q-WGT, Q-FEAT, Q-TEXT, Q-MISS (M1), and Q-EMB (family) are
> `USER_CONFIRMED_CANONICAL` or family-confirmed as recorded in `docs/method/12`.
> Statements below use those contracts. Items still open: exact calendar boundaries
> (O-CAL), Q-EMB pilot config, relation-code freeze, hyperparameters.

---

## A. Dataset A and graph construction

| # | Question | Answer | Status | Evidence |
|---|---|---|---|---|
| A1 | Scientific role of Dataset A | Source of the **directed multiplex interaction graph**, **frozen node universe**, and **edge/interaction text** (Q-TEXT). It is the *only* source of account→account edges. | VERIFIED_FROM_REAL_DATA + Current-TDMEC | `docs/data/02`,`03`,`06`; register A6; `docs/method/15` |
| A2 | Frozen node universe definition | The **16,736 distinct `user.id`** across all 12 A workbooks (= `extraction_summary.unique_matched_accounts`). | VERIFIED_FROM_REAL_DATA / D2 | `docs/data/02`,`03`; `docs/method/12` |
| A3 | 10,040 vs 16,736 | Under **D2**, the modeled universe is **all 16,736**. 10,040 "Core ARMY" is an inferred subset (Core ARMY + Pro-fans); the split is not recorded in files and **is not the modeled universe**. | D2 `USER_CONFIRMED_CANONICAL` | `docs/data/02`; `docs/method/03`, `12` |
| A4 | Event extraction (mention/retweet/reply/quote) | From A blob fields: `user_mentions[].id`, `retweeted_status.user.id`, `reply_status.user.id`, `quoted_status.user.id`. | VERIFIED_FROM_REAL_DATA | `docs/data/02`,`03` |
| A5 | Source/target per relation | src = tweet author `user.id`; dst = the referenced account id in the corresponding blob field. Direction is intrinsic (author → target). | VERIFIED (structure) | `docs/data/03` |
| A6 | Self-loops removed? | **Q-WGT:** self-loops excluded and reported. Confirm freeze in config (O-SELF) if not yet set. No edge-build code exists yet. | Q-WGT confirmed; impl pending / O-SELF | `docs/method/12` |
| A7 | Graph symmetrized? | **No** — `directed_edges = true` by construction. Symmetrization is not indicated. | VERIFIED (structure) | register G2; `docs/data/03`; Q-WGT |
| A8 | Events individual or aggregated? | Aggregate to edge identity `(snapshot_id, relation_id, source_idx, target_idx)`; `count_raw` = distinct events after Q-DEDUP. Builder not yet implemented. | Q-WGT / Q-DEDUP `USER_CONFIRMED_CANONICAL` | `docs/method/12` |
| A9 | Edge weight definition/transform | **`count_raw`** retained; canonical model weight **`weight_log1p = log(1 + count_raw)`** (Q-WGT). Ablations: binary, raw count, log1p. | Q-WGT `USER_CONFIRMED_CANONICAL` | `docs/method/12`, `03` |
| A10 | 35 snapshots | Quarterly bins **2017-Q4 → 2026-Q2** (35) are the **provisional** pilot/diagnostic calendar (Q-CAL). Exact boundaries + tail **review-pending** (O-CAL). | Q-CAL partial / O-CAL open | `docs/method/12`; `snapshots.py` |
| A11 | Node activity representation | Separate boolean **`struct_active_mask[t,node]`** (Q-FEAT): True iff tweet_count_raw>0 OR ≥1 in OR ≥1 out canonical edge. Not a feature channel. Artifact not built yet. | Q-FEAT `USER_CONFIRMED_CANONICAL` | `docs/method/14`, `12` |
| A12 | Graph artifacts to produce | node-index map (0..16,735), directed per-relation edge lists per snapshot (`count_raw`, `weight_log1p`), snapshot calendar, `X_struct` + `struct_active_mask`. **None certified yet**. | PROPOSED (artifacts) / contracts confirmed | `docs/data/03`; `docs/method/14` |
| A13 | Graph tensors per model variant | Structural tensors: `X_struct ∈ ℝ^{T×N×17}` (Q-FEAT). Text tensors: Q-MISS mask/zero contract confirmed; `D_text`/encoder depend on Q-EMB pilot. Variant module/loss tables follow Current-TDMEC method docs. | Q-FEAT/Q-MISS resolved; Q-EMB config open | `docs/method/03`, `14`, `15`, `17` |
| A14 | Which graph artifacts exist vs must be built | **No certified model-ready artifacts** (node-map / edges / snapshots / tensors) **were found in the current repository or verified persistent storage.** All must be built + certified. | VERIFIED_FROM_REAL_DATA (repo/persistent absence) | `docs/data/02`,`03`; register A9 |
| A15 | Are earlier graph statistics certified? | Uncertified out-of-repo edge-count (and similar) figures are **not certified in this repo; do not use as contract**. In-repo, only relation *presence* and node count are verified. Structural dim is **`F_struct = 17` (Q-FEAT)**, not any uncertified external figure. | NOT CERTIFIED (stats) / Q-FEAT for F | `docs/data/03`; `docs/method/14` |

**Dataset A one-line role:** *graph + edges + node universe + edge text (structural backbone).*

## B. Dataset B and textual representation

| # | Question | Answer | Status |
|---|---|---|---|
| B1 | Scientific purpose of Dataset B | Source of **exact tweet text + timestamps + engagement + exact tweet IDs** for the frozen accounts (**node text**, Q-TEXT). It carries **no edges**. | VERIFIED_FROM_REAL_DATA (`docs/data/05`,`06`); Q-TEXT |
| B2 | Required for main model / extension / ablation? | Node-text path is **required** under Q-TEXT (implemented first; edge-text second). Both paths required for full Current-TDMEC. | Q-TEXT `USER_CONFIRMED_CANONICAL` |
| B3 | Atomic text unit (tweet / user / edge / …) | **Node:** one distinct authored B tweet in-snapshot. **Edge:** one distinct A interaction-event tweet (not from B). Per-tweet/event embed → mean-pool (Q-TEXT). | Q-TEXT `USER_CONFIRMED_CANONICAL` |
| B4 | Tweets embedded individually? | **Yes** — embed each distinct cleaned tweet/event, then mean-pool (Q-TEXT). Encoder family Qwen3 only (Q-EMB); do not run embeddings until pilot authorized. | Q-TEXT resolved; Q-EMB config `PENDING_PILOT` |
| B5 | Texts grouped/concatenated before embedding? | **No** — embed per tweet/event; aggregate by mean-pool after embedding (Q-TEXT). | Q-TEXT `USER_CONFIRMED_CANONICAL` |
| B6 | Static or snapshot-specific? | **Snapshot-specific**, leakage-safe: only texts with timestamps in snapshot `t` (strict in-snapshot rule, Q-TEXT). | Q-TEXT `USER_CONFIRMED_CANONICAL` |
| B7 | Every valid tweet used? | All distinct in-snapshot authored tweets after Q-DEDUP (node path); sampling policy not separately confirmed — default is use all retained valid texts unless a later sampling decision is made. | Q-TEXT / Q-DEDUP; sampling not separately frozen |
| B8 | Sampling required? | **Not required by Q-TEXT.** Any sampling policy would be a later decision; do not invent. | OPEN if needed later |
| B9 | Aggregation of multiple texts/embeddings | **Mean-pool** of per-tweet/event embeddings (canonical). Other pooling = future ablation only (Q-TEXT). Token-level pooling inside the encoder = Q-EMB pilot. | Q-TEXT resolved; encoder pooling `PENDING_PILOT` |
| B10 | Users without text in a snapshot | **Q-MISS M1:** `node_text_available_mask=False` + exact all-zero `D_text` vector; no learned missing embedding; no drop; no text carry-forward. | Q-MISS `USER_CONFIRMED_CANONICAL` |
| B11 | Text-availability mask used? | **Yes** — separate `node_text_available_mask` / `edge_text_available_mask` (plus `struct_active_mask`); Fusion MLP / Edge Gate receive masks explicitly (`docs/method/17`). | Q-MISS `USER_CONFIRMED_CANONICAL` |
| B12 | Exact embedding model | Family **USER_CONFIRMED: Qwen3 Embedding only** (Q-EMB `docs/method/16`); preferred checkpoint `Qwen/Qwen3-Embedding-4B` (`PROVISIONAL_PENDING_PILOT`), exact `D_text`/config `PENDING_PILOT`. | family `USER_CONFIRMED_QWEN3_ONLY`; config `PENDING_PILOT` |
| B13 | Instruction template / pooling / normalization / max length / dtype / output dim | Family fixed; config **PENDING_PILOT** (Q-EMB). Stage-B cross-tweet mean-pool fixed (Q-TEXT). | Q-EMB `PENDING_PILOT` |
| B14 | How text enters TDMEC | Node path: `X_node_text[T,N,D_text]` + `node_text_available_mask` + `node_valid_text_count` (metadata). Edge path: per-edge mean-pooled A event embeddings + `edge_text_available_mask` + `edge_valid_text_count` into the edge gate (Q-TEXT/Q-MISS). `D_text` from Q-EMB pilot. | Q-TEXT/Q-MISS confirmed; Q-EMB config open |
| B15 | Initial node feature / late fusion / auxiliary loss? | Structural initial features = `X_struct` (Q-FEAT). Node text is a separate text pathway (Q-TEXT); exact fusion/loss wiring follows Current-TDMEC method docs (`docs/method/03`+). Do not treat uncertified external designs as contract. | Current-TDMEC / method docs |
| B16 | Losses depending on text | Text-dependent losses follow Current-TDMEC method contracts; λ / schedule **USER_CONFIRMED** (Batch 5 / QLOSS-04, QPHASE-01; `docs/method/06`). **Q-MISS:** `L_sem` only where availability mask=True; zero-eligible → defined zero contribution. | method docs; Batch 5 confirmed; Q-MISS confirmed |
| B17 | Is edge text part of current TDMEC? | **Yes (Q-TEXT).** A-derived; required; after node-text path. | Q-TEXT `USER_CONFIRMED_CANONICAL` |
| B18 | Can the datasets support edge text? | **Yes, from Dataset A directly.** Dataset A carries tweet `text` and relation fields. Tweet-level A↔B join is **prohibited** (float-lossy A id); B is not the edge-text source. | Q-TEXT; A↔B tweet join prohibited |
| B19 | Verified vs inferred vs unresolved | Verified: A/B content + pilot normalization. Confirmed contracts: D2, QREL-01, QSELF-01, Q-WGT, Q-FEAT, Q-TEXT, Q-MISS, Q-EMB family, Batch 4–7 architecture/loss/eval. Open (evidence-only): O-CAL boundaries, Q-EMB X01–X07, coverage numerics, runtime probes. | mixed |

**Dataset B one-line role:** *exact tweet text + timestamps + engagement for the
same frozen accounts (node-text signal source); no edges.*

## C. Model variants (QVAR-01)

Variant names are **`USER_CONFIRMED` reporting contract (QVAR-01)**. Conceptual ladder matches Q-TEXT (both text paths required for the full model; node-text first, edge-text second).

| Variant | Repo status | Required A artifacts | Required B artifacts | Blocking issue |
|---|---|---|---|---|
| **TDMEC-G** (graph only) | `USER_CONFIRMED` ablation name | node map + directed edges (`count_raw`,`weight_log1p`) + snapshots + `X_struct` + `struct_active_mask` | none | O-CAL review for certification |
| **TDMEC-NT** (graph + node text) | `USER_CONFIRMED` ablation name | as above | node-snapshot mean-pooled embeddings + masks/counts (Q-MISS) | Q-EMB pilot config |
| **TDMEC** / **TDMEC-Full** (primary) | `USER_CONFIRMED` primary-method names | as above + A-derived edge-text embeddings + masks/counts (Q-MISS) | node text (B) | Q-EMB pilot; edge path after node-text |
| **TDMEC-ET** | reserved for graph+edge-text-only ablation **if implemented**; never name the full method | as G + edge text | — | optional |

See `02_model_input_contract.md` and `docs/method/05`.

## D. End-to-end identifier alignment

| Identifier | Source | Exactness | Provenance-preserving artifact |
|---|---|---|---|
| `user.id` | A `user` object; B `user` object (serialized structured user objects, parsed as JSON/Python-literal, no eval) | **EXACT** (int, both) — canonical join key | node-index map; pilot `author_account_id` |
| `node_index` (0..16,735) | derived over frozen `user.id` | EXACT once map is built & frozen (immutable) | `node_index_map.parquet` (`author_account_id,node_index`) |
| `tweet_id` | A `id` = **float (UNSAFE)**; B `id` = **string (EXACT)** | Use **B** for exact tweet IDs; never use A's; **no tweet-level A↔B join** | pilot `tweet_id` (string), `duplicate_records.parquet` |
| `snapshot_id` (0..34 provisional) | derived from `created_at` | EXACT given the fixed quarterly calendar once O-CAL freezes bounds | pilot `snapshot_id`; `snapshot_statistics.parquet` |
| `relation_id` | derived from which A blob field the edge came from | codes `{mention:0,retweet:1,reply:2,quote:3}` **USER_CONFIRMED (QREL-01)** | (edge lists — not built yet) |
| edge source / target | A author → A blob target account | EXACT at account level (both are `user.id`) | (edge lists — not built yet) |
| `source_file` / `source_row` | ingestion provenance | EXACT | pilot normalized rows carry `source_file`, `source_row_number` |

**Exact:** `user.id`, `node_index`, B `tweet_id`, `snapshot_id` (under frozen calendar), account-level
edge endpoints, `source_file`/`source_row`. **Unsafe:** A `tweet_id` (float).
**Confirmed mapping (not yet materialized as edge artifacts):** `relation_id` codes **QREL-01**; edge lists still to be built in Cloud Phase 5+.

## E. Method version separation (comparison table)

`Method version` ∈ {Current-TDMEC, Pilot-only, Presentation-only, Proposed,
Supporting (P-002)}. Uncertified external claims are **not** a method version.

| Source file | Section / symbol | Method version | A role | B role | Text unit | Embed model | Aggregation | Model integration | Related loss | Impl status | Relevance to current TDMEC | Confidence | Conflicting evidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `docs/method/12` | D1/D2, Q-* | Current-TDMEC | graph+nodes+edge text | node text | per-tweet/event → mean-pool | Qwen3 family | mean-pool | per method/03 | per method | DEFINED | High | — |
| `docs/project/12` | Decisions 1–9 | Current-TDMEC (in-repo) | graph+nodes | text/time enrich | Q-TEXT | Q-EMB family | mean-pool | — | — | DEFINED | High | — |
| `docs/data/03` | verified contract | Current (data) | graph/edges | — | — | — | — | — | — | DEFINED (design steps) | Verified data | uncertified out-of-repo edge stats (not used) |
| `docs/data/05` | verified raw contract | Current (data) | — | text/time/engagement | — | — | — | — | — | VERIFIED_FROM_REAL_DATA | High | — |
| `src/tdmec_pilot/*` | P00–P12 | Pilot-only | consumes node map (from A) | normalizes B text rows | (row-level, no embed) | none | none | none | none | PILOT_VALIDATED | Feeds later text pipeline | High | pilot ≠ final text input |
| `configs/dataset_b_pilot.yaml` | `canonical` | Pilot-only | — | conservative norm | — | — | — | — | — | IMPLEMENTED | High | — |
| Task prompt / canonical decision | "Qwen3", variants | Current (QVAR-01 + Q-EMB family) | — | — | Q-TEXT | Qwen3 Embedding (family USER_CONFIRMED; config pilot) | mean-pool | ? | ? | family + variant names confirmed | High | — |
| P-002 Complete Technical Explanation (supporting) | supporting detail | Supporting (P-002) | directed multiplex graph | node/edge text detail | per Q-TEXT when compatible | Qwen3 family (Q-EMB overrides any older encoder name) | mean-pool (Q-TEXT) | supporting only under D1 | supporting | not the primary authority | Medium | use only when compatible with P-001 + D2 |
