# 03 — Cloud Implementation Specification

**Audience.** Cursor Cloud Agent. **Self-contained.** Do not reinterpret the
scientific method; where a decision is not fixed below, it is marked
**DO-NOT-START** and must be resolved by the user first.

> **Supersession (2026-07-28).** Binding Phase 1–12 sequence and closed conceptual
> decisions live in `docs/method/21`, `10`, `12`, and `docs/handoff/05`/`07`.
> **QREL-01**, **QSELF-01**, **Q-DEDUP policy**, **Batch 4 `d_h`/`K`/λ**, and
> **Batch 5–7** are closed. Stale `READY-AFTER-DECISION` / `O-RELID` / `O-HPARAM`
> wording below must not reopen those contracts. Exact calendar bounds, dedup
> signature/L2, coverage thresholds, and QEMB-X01…X07 remain evidence-dependent.
> **First Cloud work after this documentation push = Phase 1** (schemas/config/
> invariant tests), not full A-00…A-10 data builds.

---

## 1. Exact objective

Turn the two **verified** raw datasets into **certified, model-ready data
artifacts** for TDMEC, by implementing two deterministic, resumable, provenance-
preserving pipelines:

1. **Dataset A → directed multiplex temporal graph artifacts** (node-index map,
   per-relation directed edge lists per snapshot, snapshot calendar, structural
   node features).
2. **Dataset B → normalized, snapshot-aligned, frozen-node-reconciled text
   corpus** over all 70 files (scale-up of the validated 2-file pilot).

Producing these two artifact sets — **and nothing model-related** — is the whole
objective of this specification.

## 2. Scope (what MAY be implemented in Cursor Cloud)

- Dataset A graph pipeline stages **A-00 … A-10** (below).
- Dataset B full-corpus pipeline stages **B-00 … B-09** (scale the pilot).
- Node-index map build + certification (`scripts/build_node_index_map.py` exists).
- All validation gates, manifests, checksums, checkpointing, resume, logging.
- Unit + integration tests for every new stage.

## 3. Excluded tasks (DO-NOT-START — hard stops)

The following **must not be started** until the named open items close
(see `05_open_decisions_before_implementation.md` and `docs/method/12`):

- ❌ Text **embedding** generation (embedding family is **USER_CONFIRMED: Qwen3
  Embedding only**; exact checkpoint (`Qwen/Qwen3-Embedding-4B` preferred)/config
  `PENDING_PILOT` via Q-EMB `docs/method/16`). Do not download models or run
  embeddings without authorization.
- ❌ **Embedding materialization** until Q-EMB pilot config is confirmed.
  **Q-MISS** is **resolved** (M1: exact zero + boolean mask — `docs/method/17`);
  follow that contract; do not invent alternatives. Text **units** and **mean-pool
  aggregation** are already confirmed (Q-TEXT); do not reopen them.
- ❌ **Edge-text embedding** materialization until Q-EMB pilot config is confirmed
  (edge text is **in scope** under Q-TEXT: A-derived, required, after node-text
  path; tweet-level A↔B join prohibited; missing edge text preserves structural
  path per Q-MISS).
- ❌ TDMEC **model / GNN / GRU / community head / losses / training** until
  Phase 1 schemas are complete and later phases authorize them (primary pins
  already confirmed: `d_h=64`, `K=10`, λ / phases — `docs/method/05`,`06`,`21`).
- ❌ Inventing alternate feature dims or weight transforms that contradict
  **Q-FEAT (`F_struct=17`)** or **Q-WGT (`weight_log1p`)**.
- ❌ Downloading all 70 Dataset B files or generating embeddings **before** the
  user lifts the "no full download / no embedding" hold and supplies Drive
  credentials.

## 4. Ordered implementation stages

Legend for **current status**: `READY` (implement now), `READY-AFTER-DECISION`
(needs one small user confirmation, named), `BLOCKED` (do not start).

### 4.1 Dataset A pipeline (graph)

Reference contracts: `docs/data/02`, `docs/data/03`, `docs/project/12`.

---

**A-00 — Source validation**
- Purpose: enumerate + checksum the 12 A workbooks; confirm read-only access.
- Input artifact: Dataset A source folder (12 `.xlsx` + `extraction_summary.json`).
- Input schema: n/a (file listing).
- Transformation: list files; stream SHA-256; verify ZIP magic; record sizes.
- Output artifact: `source_validation_a.json` (path→sha256, size, sheet).
- Output schema: `{file, sha256, bytes, sheet, rows?}`.
- Invariants: 12 data files present; each opens; sheet = `tweets`.
- Identifier policy: none yet.
- Time-alignment: none.
- Validation gates: 12 files; all ZIP; checksums recorded.
- Checkpoint granularity: per file.
- Resume: skip files with recorded sha256.
- Resources: CPU-only; RAM < 2 GB; storage ≈ 3.4 GB transient (evict after read).
- Downstream consumer: A-01, A-02.
- Current status: **READY** (reuse `src/tdmec_discovery` adapters + hashing).
- Evidence: `docs/data/00`,`01`; `src/tdmec_discovery/sources.py`,`hashing.py`.

**A-01 — Schema & identifier normalization**
- Purpose: validate 31-column schema; extract `author_id=user.id` (int, exact),
  keep top-level `id` **flagged unsafe** (float).
- Input: A workbooks. Input schema: 31 columns (`dataset_a_schema_registry.json`).
- Transformation: exact column-order check; safe `user`-blob parse (reuse
  `src/tdmec_pilot/user_blob.py` pattern; **no eval**); cast `author_id`→int64.
- Output artifact: per-part normalized parquet (author_id, created_at, blob refs,
  provenance `source_file`,`source_row`). Output schema: documented in A-05 join.
- Invariants: schema sig = `83e45e17ffc3`; author_id non-null ≥ 99.9%.
- Identifier policy: `user.id` EXACT; A `id` **never** used as a join key.
- Time-alignment: none yet.
- Validation gates: column order exact; author parse rate.
- Checkpoint: per part, per chunk (row-cap ≈ 1.05 M/part → chunk 200 k).
- Resume: checksum-guarded chunk skip (pilot pattern).
- Resources: CPU; RAM ≤ 4 GB (one part at a time); storage ≈ few GB parquet.
- Downstream: A-02, A-04, A-05.
- Current status: **READY**.
- Evidence: `docs/data/02`; `src/tdmec_pilot/{user_blob,identifiers,schema}.py`.

**A-02 — Frozen node mapping**
- Purpose: build the **immutable** node-index map over the 16,736 `user.id`.
- Input: distinct `user.id` across all 12 parts. Input schema: int64 ids.
- Transformation: `build_node_map_from_ids` — unique ids **sorted ascending by
  integer** → `node_index = 0..16,735` (`src/tdmec_pilot/node_map.py`).
- Output artifact: `node_index_map.parquet` (`author_account_id:int64`,
  `node_index:int32`). **Immutable once published.**
- Output schema: exactly 16,736 rows; indices contiguous 0..16,735.
- Invariants: count = 16,736; contiguous; bijective; sorted determinism.
- Identifier policy: universe **frozen**; expansion forbidden.
- Time-alignment: none.
- Validation gates: count == 16,736; min 0; max 16,735; no gaps/dupes.
- Checkpoint: single artifact (atomic write).
- Resume: idempotent (recompute is deterministic; sha256 must match if re-run).
- Resources: CPU; RAM ≤ 4 GB; storage < 5 MB.
- Downstream: A-05, A-07, A-08; Dataset B pipeline B-05.
- Current status: **READY** (`scripts/build_node_index_map.py` exists; must run on
  all 12 A files and be certified present + checksummed).
- Evidence: `docs/data/03`; `scripts/build_node_index_map.py`; `node_map.py`.

**A-03 — Timestamp validation**
- Purpose: parse `created_at` (epoch) to UTC; validate range.
- Input: normalized A rows. Transformation: reuse `src/tdmec_pilot/timestamps.py`
  (ms-guard, min 2006-03-21, max now+1d).
- Output: `created_at_utc` column added; violations quarantined (flagged).
- Invariants: quarantine rate ≈ 0 (as observed). Time-alignment: UTC.
- Validation gates: invalid-ts count reported; re-check 2026 tail (A max 2026-05-30).
- Checkpoint/resume/resources: as A-01. Downstream: A-06.
- Current status: **READY**. Evidence: `docs/data/02` (quality #5); `timestamps.py`.

**A-04 — Relation-event extraction**
- Purpose: extract directed account→account events for 4 relations.
- Input: A blob fields `user_mentions[].id`, `retweeted_status.user.id`,
  `reply_status.user.id`, `quoted_status.user.id`.
- Transformation: per tweet, emit `(src=author_id, dst=target_id, relation)` for
  each populated field; map both endpoints via node map; drop endpoints ∉ frozen.
- Output artifact: raw directed event list parquet (`src_index, dst_index,
  relation_id, created_at_utc, source_file, source_row`).
- Output schema: int32 indices; `relation_id ∈ {0,1,2,3}` (**QREL-01
  USER_CONFIRMED_CANONICAL**: `0=mention,1=retweet,2=reply,3=quote`).
- Invariants: endpoints ∈ 0..16,735; direction author→target preserved.
- Identifier policy: account-level EXACT via `user.id`.
- Time-alignment: each event carries `created_at_utc`.
- Validation gates: per-relation non-null counts in expected order of magnitude
  (mention > retweet > reply > quote, per `docs/data/02`).
- Self-loops: **QSELF-01 / Q-WGT** — exclude before aggregation; report counts;
  do not silently drop without accounting.
- Checkpoint: per part/chunk. Resume: checksum-guarded.
- Resources: CPU; RAM ≤ 6 GB; storage ≈ several GB.
- Downstream: A-06, A-07.
- Current status: **READY** (relation codes + self-loop policy confirmed).
- Evidence: `docs/data/02`,`03`; `docs/method/12` (QREL-01, QSELF-01).

**A-05 — Safe duplicate handling**
- Purpose: resolve the 1,168,525 flagged A duplicates before edge aggregation.
- Input: normalized A rows. Transformation: apply **Q-DEDUP** policy (conflict-
  aware collapse; A via deterministic composite signature — float tweet id
  forbidden as key). Exact signature/L2 thresholds remain POST_DIAGNOSTIC
  (`QDEDUP-B01-PROC`); until user approves diagnostic outcomes, prefer annotate
  + report and do not stamp edge/text artifacts `dedup-certified`.
- Output: `dataset_a_duplicate_report.parquet`.
- Invariants: nothing silently dropped without accounting; row accounting balances.
- Validation gates: duplicate counts reconcile with `extraction_summary`
  (1,168,525 detected).
- Current status: **READY** for privacy-safe diagnostic report; certification
  gated on QDEDUP-B01 user approval.
- Evidence: `docs/data/02` (quality #2); register A5; `docs/method/12` (Q-DEDUP).

**A-06 — Snapshot assignment**
- Purpose: assign each event to a quarterly snapshot.
- Input: events + `created_at_utc`. Transformation: reuse
  `src/tdmec_pilot/snapshots.py` (2017-Q4=0 … 2026-Q2=34; start-inclusive/end-
  exclusive). Out-of-range → `snapshot_id = -1` (flag, not drop).
- Output: events with `snapshot_id ∈ {-1,0..34}`.
- Invariants: identical calendar to Dataset B pipeline (shared module).
- Validation gates: retained events have `snapshot_id ∈ 0..34`.
- Decision: whether A's 2026 tail beyond Q2 is truncated or the calendar extends
  (D-4). Current status: **READY** (calendar) / tail policy READY-AFTER-DECISION.
- Evidence: `docs/data/03`,`06`; `snapshots.py`.

**A-07 — Directed relation-specific aggregation**
- Purpose: aggregate events into weighted directed edges per `(snapshot, relation)`.
- Input: snapshot-assigned events. Transformation: `GROUP BY (snapshot_id,
  src_index, dst_index, relation_id)` → `count_raw = distinct-event count` after
  Q-DEDUP; **`weight_log1p = log(1 + count_raw)`** (Q-WGT; stable natural-log
  `log1p`). Emit per `(snapshot, relation)` directed edge list. Exclude self-loops
  and report counts (Q-WGT).
- Output artifact: `edges/snapshot={t}/relation={r}/part-*.parquet`
  (`src_index, dst_index, relation_id, count_raw, weight_log1p` + provenance).
- Output schema: int32 endpoints, int8 relation, int64 `count_raw`, float32
  `weight_log1p`.
- Invariants: no NaN; endpoints in range; directed (no symmetrization —
  `docs/data/03`); `weight_log1p = log1p(count_raw)`.
- Decisions: **Q-WGT RESOLVED** — **do compute** `weight_log1p` as the canonical
  model edge weight; retain `count_raw`. Ablations (binary / raw / log1p) may be
  derived later; do not omit `weight_log1p`.
- Validation gates: Σ `count_raw` == retained distinct-event count per `(t,r)`.
- Checkpoint: per `(snapshot,relation)`. Resume: skip completed partitions.
- Resources: CPU; consider DuckDB out-of-core (readiness audit ~14 GB RAM).
- Downstream: A-08, A-09, A-10.
- Current status: **READY** for aggregation + log1p (QREL/QSELF/Q-DEDUP policy
  confirmed; signature thresholds POST_DIAGNOSTIC). Evidence: `docs/method/12` (Q-WGT); `docs/data/03`.

**A-08 — Structural node features** — **READY under Q-FEAT**
- Purpose: per-`(node,snapshot)` structural feature matrix
  `X_struct[t,node,:] ∈ ℝ^{N×17}` plus boolean `struct_active_mask[t,node]`.
- Contract: **`F_struct = 17`** (Q-FEAT `USER_CONFIRMED_CANONICAL`). Ordered
  schema: per relation {`out_degree`, `in_degree`, `out_strength_log1p`,
  `in_strength_log1p`} for mention(0–3)/retweet(4–7)/reply(8–11)/quote(12–15) +
  `16 tweet_count_log1p`. Degree = distinct-neighbor count (no log1p); strength =
  `log1p(Σ count_raw)`; tweet_count = `log1p(#distinct authored tweets)`.
  `active` / `n_active_relations` are **not** feature channels. See
  `docs/method/14`, `docs/method/12`.
- Additional feature normalization → **QHP-02** train-time robust scaling
  (train-only; do not apply to canonical artifacts here).
- Current status: **READY** once A-07 edges exist (QREL-01 already confirmed).

**A-09 — Graph validation**
- Purpose: validate the edge artifacts before publication.
- Gates: every endpoint ∈ 0..16,735; relation ∈ {0..3}; snapshot ∈ 0..34
  (provisional until O-CAL); `count_raw` ≥ 1; `weight_log1p = log1p(count_raw)`;
  per-`(t,r)` counts reconcile with A-07; deterministic re-run sha256.
- Current status: **READY** (for A-07 outputs).
- Evidence: pattern from `src/tdmec_pilot/pipeline.py` `p11_validate`.

**A-10 — Model-ready graph artifacts (publication)**
- Purpose: publish node map + edge lists + snapshot calendar + structural
  features/masks + manifest/checksums.
- Output: `graph/manifest.json`, `graph/checksums.json`, `node_index_map.parquet`,
  `edges/…`, `snapshot_calendar.json`, `X_struct` / `struct_active_mask` artifacts.
- Invariants: atomic publish; immutable node map; provenance recorded;
  `F_struct=17`.
- Current status: **READY** for structural graph under Q-WGT/Q-FEAT (calendar
  certification waits on O-CAL review).

### 4.2 Dataset B pipeline (text, full corpus)

Reference: `docs/data/07` (plan), `docs/data/08` (pilot), `configs/dataset_b_pilot.yaml`.
The pilot **already implements B-00 … B-09 for 2 files** and is PILOT_VALIDATED.
The Cloud task is to **scale to all 70 files** (after credentials + hold lift).

---

**B-00 — Source validation** — reuse pilot `p01_source_verification`. Scale to 70
files. Status: **READY-AFTER-HOLD-LIFT** (needs "no full download" lifted + creds).
Evidence: `src/tdmec_pilot/pipeline.py`; `docs/data/08`.

**B-01 — Safe `user` parsing** — reuse `user_blob.parse_user_blob` (no eval).
Status: **READY** (validated). Evidence: `tests/test_pilot.py`.

**B-02 — Exact identifier preservation** — `tweet_id` exact string; float rejected
(`identifiers.normalize_tweet_id`). Status: **READY** (validated).

**B-03 — Timestamp normalization** — `timestamps.parse_created_at`. Status:
**READY** (validated).

**B-04 — Duplicate classification** — exact vs conflicting; annotate only
(`dedup.py`; `duplicate_records.parquet`). **Full-corpus cross-file dedup** is a
larger stage than the 2-file pilot; keep annotate-only until a corpus dedup policy
(D-3) is set. Status: **READY** (report) / dedup-drop READY-AFTER-DECISION.

**B-05 — Frozen-node reconciliation** — left-join on `author_id` to the immutable
node map; never create indices; unmatched → `node_index=null`. Status: **READY**
(validated; A-02 map required).

**B-06 — Snapshot assignment** — shared `snapshots.py`; out-of-range excluded
(flagged, not dropped). Status: **READY** (validated).

**B-07 — Conservative text normalization** — NFC + newline + BOM only; **all
destructive cleaning disabled** (`text_quality.py`; config all-false). `text_raw`
preserved verbatim. Status: **READY** (validated). Do **not** enable any removal.

**B-08 — Text-unit construction (Q-TEXT)** — contract **RESOLVED**; materialization
of embed-ready units may proceed as row grouping only **without** calling an
encoder. Atomic units: node = distinct B authored tweet in-snapshot; edge =
distinct A event tweet. Do **not** concatenate texts before embedding. Pilot may
still stop at normalized rows until embedding is authorized.

**B-09 — Method-defined embedding** — **DO-NOT-START (BLOCKED).** Embedding family
**USER_CONFIRMED: Qwen3 Embedding only**; exact checkpoint/config `PENDING_PILOT`
(Q-EMB `docs/method/16`). No embedding libs in `requirements.txt`. Do not download
models.

**B-10 — Aggregation + missing-text** — Aggregation = **mean-pool RESOLVED**
(Q-TEXT). **Q-MISS M1 RESOLVED** (`USER_CONFIRMED_CANONICAL`, `docs/method/17`):
exact all-zero vector + boolean availability mask; no learned missing embedding;
no drop; no text carry-forward; Fusion MLP / Edge Gate receive masks; missing
edge text preserves structural path; `L_sem` mask=True only; separate
`struct_active_mask` / `node_text_available_mask` / `edge_text_available_mask`;
`node_valid_text_count` / `edge_valid_text_count` = required metadata (not
features); privacy-safe coverage reports required. Status: mean-pool + mask
contract READY to implement with embeddings; blocked only on Q-EMB pilot.

**B-11 — Model-ready text artifacts / TDMEC textual tensors** — **DO-NOT-START
(BLOCKED)** until Q-EMB pilot config is confirmed (Q-MISS closed; follow M1).

## 5. Modules/files to create or modify

- **New package `src/tdmec_graph/`** (mirror `tdmec_pilot` structure) for A-00…A-10:
  `source.py`, `schema_a.py`, `relations.py`, `aggregate.py` (DuckDB),
  `graph_validate.py`, `graph_storage.py`, `pipeline_a.py`.
- **New script** `scripts/run_dataset_a_graph.py` (resume-aware, mirrors
  `run_dataset_b_pilot.py`).
- **Reuse** `src/tdmec_pilot/{user_blob,identifiers,timestamps,snapshots,dedup,
  node_map,storage}.py` and `src/tdmec_discovery/{sources,hashing,cache}.py`.
- **New config** `configs/dataset_a_graph.yaml` (canonical block: node count,
  relation codes [QREL-01 confirmed], snapshot calendar [Q-CAL; bounds
  O-CAL], dedup policy [Q-DEDUP; signature review-pending],
  `weight_log1p` [Q-WGT confirmed], `F_struct=17` [Q-FEAT confirmed]).
- **Scale runner** for B: extend `configs/dataset_b_pilot.yaml` `input_files` to
  all 70 (only after hold lift + creds) — do **not** edit the canonical block.
- **Do NOT create** any model/embedding module.

## 6. Contracts, schemas, tensor shapes

- Input schemas: A = 31 cols (`dataset_a_schema_registry.json`); B = 10 cols
  (`dataset_b_schema_registry.json`).
- Output schemas: node map (`author_account_id,node_index`); edges
  (`src_index,dst_index,relation_id,count_raw,weight_log1p` per `snapshot/relation`);
  B normalized rows (22-col `NORMALIZED_COLUMNS`, `src/tdmec_pilot/schema.py`).
- Tensor shapes: see `02_model_input_contract.md`. `F_struct=17`, primary
  `d_h=64`, primary `K=10` are confirmed. Do **not** materialize text tensors
  whose `D_text` is still `PENDING_PILOT`.

## 7. Configuration fields

Canonical (hash-affecting) A config must fix only VERIFIED / confirmed values: `frozen_node_count:16736`,
`valid_node_index_min/max:0/16735`, `account_join_key:user.id`,
`tweet_id_policy: A.id UNSAFE (do not join)`, `snapshot_frequency:quarterly`,
`snapshot_start:2017-Q4`, `snapshot_count:35`, `node_universe_expansion:forbidden`,
`raw_source_modification:forbidden`,
`edge_weight_transform: log1p` (Q-WGT), `F_struct: 17` (Q-FEAT),
`relation_codes: QREL-01`. Leave exact dedup signature (Q-DEDUP review) and calendar
bounds (O-CAL) as explicit **review-pending** fields that fail-closed
if treated as certified before confirmation.

## 8. Validation gates (must pass before publication)

Graph (A): 12 files validated; node map 16,736 contiguous; endpoints in range;
relation ∈ {0..3}; snapshot ∈ 0..34; weights ≥ 1; aggregation reconciles; raw
source checksums unchanged; deterministic re-run sha256 identical.
Text (B): the **11 pilot gates** (`docs/data/08` §6) applied to every file at
scale; row accounting balances; `tweet_id` exact strings end-to-end; every
retained row has node_index + snapshot; raw source unchanged.

## 9. Tests

- Unit: relation extraction per field; self-loop flagging; node-map determinism;
  snapshot boundaries; A duplicate classification; DuckDB aggregation reconciliation.
- Integration: synthetic A workbook → full A pipeline → gates pass; resume after
  interruption yields identical checksums; config-incompat refused. Mirror
  `tests/test_pilot.py` structure. Keep tests network-free (synthetic workbooks).

## 10. Persistent storage structure

```
{OUTPUT_ROOT}/
  manifests/node_index_map.parquet          # immutable, built once (A-02)
  graph/<run_id>/
    manifest.json  checksums.json  snapshot_calendar.json
    dataset_a_duplicate_report.parquet
    edges/snapshot={t}/relation={r}/part-*.parquet
    logs/…
  text/<run_id>/                             # B scale-up (pilot layout)
    normalized_records/source_file={name}/part-*.parquet
    excluded_records/  duplicate_records.parquet  account_reconciliation.parquet
    snapshot_statistics.parquet  manifest.json  checksums.json  validation_report.json
```
On Colab/Drive, `OUTPUT_ROOT = /content/drive/MyDrive/TDMEC_PROJECT_OUTPUTS`.

## 11. Manifests, checksums, checkpointing, resume, determinism, logging, provenance

- **Manifests + SHA-256** per output (reuse `src/tdmec_pilot/storage.py`,
  `src/tdmec_discovery/hashing.py`). **Run id** = timestamp + git hash + config
  hash (`storage.py`).
- **Checkpoint granularity:** per file → per 200 k-row chunk (A and B).
- **Resume:** skip chunks whose parquet sha256 matches the checkpoint; refuse
  resume on canonical-config-hash change (`ConfigIncompatibleError`) or input
  checksum drift.
- **Determinism:** sorted-integer node map; `GROUP BY` aggregation; canonical
  duplicate = min `(source_file,source_row)`; config hash over sorted JSON.
- **Logging:** per-run `logs/…log`; structured gate results in `validation_report.json`.
- **Provenance:** every normalized/event row carries `source_file` + `source_row`.

## 12. Raw-data immutability

Sources are strictly read-only (`src/tdmec_discovery/sources.py`; copies to a
local cache; no write/rename/delete path). Input SHA-256 recorded at ingest and
re-verified at publication (`raw_source_unchanged` gate). **Never** modify source.

## 13. Pilot inputs & expected pilot outputs (regression anchor)

Before any 70-file B run, re-run the 2-file pilot (`statuses-2.xlsx`,
`statuses-69.xlsx`) and reproduce: rows_in 2,004,845 / retained 1,992,014 /
excluded 12,831 / rejected 0 / 496 authors all matched / 747 exact-dup groups /
all 11 gates pass (`docs/data/08`). This is the acceptance anchor for scale-up.

## 14. Approval gates & full-run prerequisites

- **Before A full run:** QREL-01 / QSELF-01 already confirmed; complete Q-DEDUP
  diagnostic + user approval of signature/L2 before `dedup-certified` stamps.
  **Q-WGT** and **Q-FEAT** are already confirmed — compute `weight_log1p` and
  `F_struct=17` features.
- **Before B full run:** user lifts "no full download", supplies **Drive
  credentials** (`GOOGLE_APPLICATION_CREDENTIALS`) + `DATASET_B_SOURCE` /
  `TDMEC_OUTPUT_DRIVE_FOLDER_ID` (readiness audit gap #1).
- **Before any embedding/model work:** complete Q-EMB pilot (X01–X07); follow
  Q-MISS M1; Batch 4–7 primary pins already confirmed (`docs/method/21`).

## 15. Environment placement

| Stage | GPU? | Where |
|---|---|---|
| A-00…A-10 (graph) | No | Cursor Cloud (CPU) or Colab; DuckDB out-of-core |
| B-00…B-09 (text norm, 70 files) | No | **Colab with Drive mounted** (persistent output + publish) or Cloud if Drive creds injected |
| Node-map build (A-02) | No | Cursor Cloud (CPU) |
| Embedding (B-09) | **Yes (later)** | Colab/GPU — **DO-NOT-START** |
| TDMEC training | **Yes (later)** | Colab/GPU — **DO-NOT-START** |

## 16. Tasks the Cloud Agent must NOT decide

Embedding checkpoint/config (QEMB-X01…X07) · inventing a non-Qwen encoder ·
reopening Q-MISS / Q-WGT / Q-FEAT / Q-TEXT / QREL / QSELF / Batch 4–7 · inventing
alternate `D_text` before pilot · exact dedup signature thresholds before
diagnostics review · calendar boundary freeze before O-CAL review. Confirmed
contracts to **follow** (not re-decide): `weight_log1p` (Q-WGT), `F_struct=17`
(Q-FEAT), mean-pool text units (Q-TEXT), Q-MISS M1 (`docs/method/17`), N=16736
(D2), QREL-01 codes, primary `d_h=64`/`K=10`/λ/phases. See
`05_open_decisions_before_implementation.md` and `docs/method/21`.

## 17. First task to assign

**Certify the immutable node-index map (A-02):** run
`scripts/build_node_index_map.py` over all 12 Dataset A workbooks, verify exactly
16,736 contiguous indices (0..16,735), publish `manifests/node_index_map.parquet`
with a recorded SHA-256, and add a regression test. It is fully specified,
unblocked, deterministic, CPU-only, and unblocks both A-04/A-05 and B-05.
