# 00 — Verified Dataset Profile

**Purpose.** A single, evidence-cited profile of the two physical datasets and the
controlled pilot, for handoff to Cursor Cloud Agent. Everything here is
**VERIFIED_FROM_REAL_DATA** or **VERIFIED_FROM_IMPLEMENTATION** unless marked
otherwise. Sources are repository files only.

> **Scope note (read first).** This repository is a **data discovery + Dataset B
> pilot** codebase. It contains **no TDMEC model, architecture, loss, embedding,
> or training specification or code** (verified: 0 matches for `Qwen`, `GraphSAGE`,
> `GRU`, `Student-t`, `L_struct`/`L_sem`/`L_temp`/`L_cluster`, or the variant names
> `TDMEC-G/NT/ET` anywhere in the repo). Method-level questions are therefore
> answered as *absent / requires the method spec to be imported* — see
> `05_open_decisions_before_implementation.md`. (The 0-matches finding above is the
> **pre-decision** repo state; the embedding family is now **user-confirmed Qwen3
> Embedding only** by canonical decision — exact checkpoint/config `PENDING_PILOT`,
> see Q-EMB `docs/method/16`.)

## Evidence-status legend

`VERIFIED_FROM_REAL_DATA` · `VERIFIED_FROM_CURRENT_METHOD` ·
`VERIFIED_FROM_IMPLEMENTATION` · `INCORRECT_PRIOR_CLAIM` · `PILOT_ONLY` ·
`INFERRED` · `PROPOSED` · `USER_CONFIRMATION_REQUIRED` · `METHOD_SPEC_REQUIRED` ·
`BLOCKED_BY_DATA_LIMITATION`.

`METHOD_SPEC_REQUIRED` = the item cannot be decided until the authoritative TDMEC
method specification is imported; it is **not** a data limitation.
`BLOCKED_BY_DATA_LIMITATION` is reserved for items the physical datasets genuinely
cannot support (e.g. cross-dataset tweet-level linkage, full-corpus statistics).

---

## 1. Dataset A — exact verified definition

| Property | Value | Status | Evidence |
|---|---|---|---|
| Composition | **12 × `.xlsx`** (`core_army_pro_fans_tweets_part_001..012.xlsx`) **+ `extraction_summary.json`** (13 entries) | VERIFIED_FROM_REAL_DATA | `docs/data/00`, `01`, `02` |
| Nature | Preprocessed, **account-filtered tweet extract** (cleaned/intermediate). Not raw, not a graph artifact, not model-ready. | VERIFIED_FROM_REAL_DATA | `docs/data/02` (file-role table) |
| Worksheet | `tweets` (single sheet) | VERIFIED_FROM_REAL_DATA | `docs/data/01` |
| Columns | **31**, identical across all 12; schema sig `83e45e17ffc3` | VERIFIED_FROM_REAL_DATA | `artifacts/discovery/dataset_a_schema_registry.json`; `docs/data/02` |
| Rows | **12,581,535** (11×1,048,575 + 1×1,047,210) | VERIFIED_FROM_REAL_DATA | `docs/data/01`, `02` |
| Downloaded size | ≈ 3.37 GB (258–308 MB/file) | VERIFIED_FROM_REAL_DATA | `docs/data/01` |
| Frozen node universe | **16,736 distinct `user.id`** (cumulative distinct reaches 16,736 by part_006) | VERIFIED_FROM_REAL_DATA | `docs/data/02`, `03`; `extraction_summary` |
| Provenance | built from **41** raw `bts-*`/`bts-fa-*` files; `total_source_rows_scanned=14,437,781`, `total_matched_tweets=12,581,535` | VERIFIED_FROM_IMPLEMENTATION (extraction_summary) | `docs/data/02` |
| Duplicates | **1,168,525 detected, 0 removed** (mostly cross-file; within-file dup-id rows = 196) | VERIFIED_FROM_REAL_DATA | `docs/data/02`; register A5 |
| Tweet `id` | **float / scientific notation → precision-lossy**; NOT an exact tweet key | VERIFIED_FROM_REAL_DATA | `docs/data/02` (quality #1); register A7 |
| Fully-null columns (8) | `ocr_text, quoted_count, bookmarks, views, engagement, sentiment, topic, copy_count` (+ `media` ~81% null, `text_emojis` ~99% null) | VERIFIED_FROM_REAL_DATA | `docs/data/02` (quality #3); register A8 |
| Temporal range | `created_at` epoch **2017-10-28 → 2026-05-30**, concentrated ~2020–2022 | VERIFIED_FROM_REAL_DATA | `docs/data/02` (quality #5) |
| 10,040 vs 16,736 | folder = `core_army_pro_fans`; **10,040 "Core ARMY" ⊂ 16,736** (Core ARMY + Pro-fans); exact split not in files. Under **D2**, **10,040 is not the modeled universe** (N = 16,736). | D2 `USER_CONFIRMED_CANONICAL` / subset INFERRED | `docs/data/02`; `docs/method/12`; register A10 |

**Full 31-column list** (`artifacts/discovery/dataset_a_schema_registry.json`):
`timestamp, is_removed, id, created_at, is_quote_status, user, text, ocr_text,
text_lang, lang, text_tags, text_hashtags, text_emojis, user_mentions, media,
likes, retweets, reply_count, quoted_count, reply_status, quoted_status,
retweeted_status, location_tags, bookmarks, views, place, impression, engagement,
sentiment, topic, copy_count`.

**Relation-bearing fields (edges derivable from Dataset A alone):**

| Relation | Field | Direction |
|---|---|---|
| mention | `user_mentions[].id` | author → mentioned |
| retweet | `retweeted_status.user.id` | author → retweeted |
| reply | `reply_status.user.id` | author → replied-to |
| quote | `quoted_status.user.id` | author → quoted |

Measured non-null targets in a single 1.05 M-row part (`part_001`): mention ≈ 912,306,
retweet ≈ 670,055, reply ≈ 149,842, quote ≈ 102,893 (`docs/data/02`). Status:
VERIFIED_FROM_REAL_DATA.

## 2. Dataset B — exact verified definition

| Property | Value | Status | Evidence |
|---|---|---|---|
| Composition | **70 × `statuses-{0..69}.xlsx` + `download_manifest.json`** (71 visible entries) | VERIFIED_FROM_REAL_DATA | `docs/data/00`, `01`, `05` |
| Nature | **RAW** w.r.t. this project (no official preprocessing performed) | VERIFIED_FROM_REAL_DATA | register B2; `docs/data/05` |
| Reported total size | **9,582,913,975 bytes ≈ 9.58 GB** (109–159 MB/file) | VERIFIED_FROM_REAL_DATA (manifest) | `docs/data/01`, `05` |
| Worksheet | `Sheet1` (single) | VERIFIED_FROM_REAL_DATA (12-file sample) | `docs/data/04` |
| Columns | **10, fixed order**; single schema family `f903842ea287`; 0 order-variants | VERIFIED_FROM_REAL_DATA (12-file sample) | `artifacts/discovery/dataset_b_schema_registry.json`; `docs/data/04` |
| Inspection coverage | **12 files deeply inspected**; 58 inventoried by manifest only | VERIFIED_FROM_REAL_DATA / rest BLOCKED_BY_DATA_LIMITATION | `docs/data/04`; register B10 |
| Tweet `id` | **exact integer-as-string** (precision preserved) | VERIFIED_FROM_REAL_DATA | `docs/data/04`, `05`; register B5 |
| Author id | `user.id` inside a **serialized structured `user` object** (parsed safely as JSON or Python-literal syntax, never `eval`) with fields `{id, followers, username, title, political_category}` | VERIFIED_FROM_REAL_DATA | `docs/data/04`, `05`; `src/tdmec_pilot/user_blob.py` |
| Edge fields | **ABSENT** — only engagement counts (`retweets, reply_count, quoted_count` are counts, not targets) | VERIFIED_FROM_REAL_DATA | `docs/data/04`, `05`; register B6 |
| File structure | per-account history chunks (~120–290 authors/file); `created_at.min` rises with index | VERIFIED_FROM_REAL_DATA (sample) | `docs/data/04` |
| Temporal range (sample) | ~**2012-08 → 2026-07** | VERIFIED_FROM_REAL_DATA (sample) | `docs/data/04`, `05` |
| Quality | invalid timestamps 0; empty text ~30/file; within-file dup-id 1–746 | VERIFIED_FROM_REAL_DATA (sample) | `docs/data/04` |
| Language field | **none** | VERIFIED_FROM_REAL_DATA | `docs/data/04` |
| Full-corpus distinct counts | not computed (58 files not deeply inspected) | BLOCKED_BY_DATA_LIMITATION | register B10 |

**10-column schema:** `id, created_at, user, text, likes, retweets, reply_count,
quoted_count, bookmarks, views`.

## 3. Cross-dataset facts

| # | Fact | Status | Evidence |
|---|---|---|---|
| X1 | Canonical join key = numeric **`user.id`** (precise in both) | VERIFIED_FROM_REAL_DATA | register X1; `docs/data/06` |
| X2 | 100% of sampled B authors (2,498) ∈ frozen 16,736; 0 outside | VERIFIED_FROM_REAL_DATA (12-file sample) | `docs/data/06` |
| X3 | Whether all 70 B files cover the full 16,736 or a subset | UNRESOLVED | register X3; `docs/data/06` |
| X4 | Temporal overlap ~2017→2026; B extends earlier (2012) and slightly later | VERIFIED_FROM_REAL_DATA (sample) | `docs/data/06` |
| X5 | **Do NOT join on A's tweet `id`** (float-lossy); use B's string IDs for tweet-level linkage | VERIFIED_FROM_REAL_DATA | register X5; `docs/data/06` |
| — | Reliable tweet-level/edge-level linkage A↔B not yet demonstrated | UNRESOLVED | `docs/data/06` |

## 4. Controlled Dataset B pilot — verified status

Implemented under `src/tdmec_pilot/`, `scripts/run_dataset_b_pilot.py`,
`scripts/build_node_index_map.py`, `configs/dataset_b_pilot.yaml`,
`notebooks/03_dataset_b_controlled_pilot.ipynb`, `tests/test_pilot.py`. Processes
**exactly** `statuses-2.xlsx` and `statuses-69.xlsx`.

Real 2-file run (`docs/data/08`): rows_in **2,004,845**; retained **1,992,014**;
excluded **12,831** (all `outside_canonical_snapshot_range`, i.e. after 2026-Q2);
rejected **0**; unique authors **496** (all matched to frozen, 0 unmatched);
duplicate groups **747** (all `exact_duplicate`, 0 conflicting); text quality
2,004,820 ok / 25 empty; raw source checksums unchanged. **All 11 gates passed.**
Status: **PILOT_VALIDATED**.

The pilot **outputs normalized text rows** (parquet, 22-col `NORMALIZED_COLUMNS`),
**not embeddings** (`src/tdmec_pilot/__init__.py`, `pipeline.py` docstrings;
`src/tdmec_pilot/schema.py`). It performs safe `user`-blob parsing (no `eval`),
identifier normalization (tweet id as exact string; float rejected), epoch
timestamp parsing, **conservative** text normalization (NFC + newline + BOM only;
all destructive cleaning `false` in config), exact-vs-conflicting duplicate
*reporting* (nothing dropped), quarterly snapshot assignment (35 bins 2017-Q4…
2026-Q2), frozen-node reconciliation (never creates node indices), atomic writes,
manifests, SHA-256 checksums, and checksum-guarded resume.

**Whole test suite:** `31 passed` (9 discovery + 22 pilot) — confirmed in the
session terminal log and `docs/data/08`.

## 5. Project-contract cross-reference

The task references `docs/project/02..10` contract numbers. In **this** repository
those authoritative contracts already exist under `docs/data/` and
`docs/project/12`; to avoid duplicate documents they are **not** recreated. Mapping:

| Requested contract | Authoritative document in this repo |
|---|---|
| `02_dataset_a_contract` | `docs/data/03_dataset_a_verified_contract.md` |
| `03_dataset_b_contract` | `docs/data/05_dataset_b_verified_raw_contract.md` |
| `04_text_embedding_contract` | **absent** — see `05_open_decisions_before_implementation.md` (blocked) |
| `05_temporal_multiplex_graph_contract` | `docs/data/03` (graph fields) + `docs/project/12` (graph contract) |
| `06_model_architecture_contract` | **absent** — no method spec in repo (blocked) |
| `08_end_to_end_pipeline` | `docs/handoff/03` + `04` (this handoff set) |
| `10_open_questions_and_conflicts` | `docs/handoff/05` + `docs/project/12` |
| `12_canonical_decision_register` | `docs/project/12` (updated this session) |
| `14_implementation_status_matrix` | `docs/project/14` (created this session) |
| `15_artifact_registry_v2` | `docs/project/15` (created this session) |
