# 02 — Dataset A Forensic Report (Phase 3)

Scope: all 12 Dataset A workbooks were fully read (one at a time, evicted after
inspection) plus `extraction_summary.json`. No file was modified; no mapping was
altered; no node was renumbered.

## File-role classification

| File(s) | Role | Basis |
|---|---|---|
| `core_army_pro_fans_tweets_part_001..012.xlsx` | **Cleaned / intermediate source** (account-filtered, enriched tweet extract) | `Verified from real files` |
| `extraction_summary.json` | **Validation report / provenance manifest** | `Verified from real files` |
| *(none present)* | Raw source | Dataset A's raw inputs (`bts-*.xlsx`) are **not** in this folder |
| *(none present)* | Mapping artifact (node-index map) | Not present |
| *(none present)* | Graph artifact (edge lists / snapshots) | Not present |
| *(none present)* | Model-ready artifact (tensors) | Not present |

**Conclusion:** Dataset A is a *preprocessed but not graph-ready* tweet extract.
Preprocessing already performed = account filtering/matching to the 16,736 target
accounts and column enrichment. Preprocessing **not** performed = deduplication,
node-index assignment, edge-list construction, snapshot binning, tensor export.

## What `extraction_summary.json` establishes (`Verified from source code`/`real files`)

- `target_accounts_total = 16736`; `unique_matched_accounts = 16736`;
  `target_accounts_with_at_least_one_tweet = 16736`;
  `target_accounts_with_no_tweets_found = 0`.
- Built by scanning **41** raw source files (`bts-*.xlsx`, `bts-fa-*.xlsx`),
  `total_source_rows_scanned = 14,437,781`, `total_matched_tweets = 12,581,535`.
- `duplicate_tweets_detected = 1,168,525`, **`duplicate_tweets_removed = 0`**
  → duplicates are flagged but retained.
- Author identity comes from a `user` blob (`author_strategy = "user_blob"`),
  `tweet_id_column = "id"`, target keys `user_id` / `username`.

## Per-file measured facts (`Verified from real files`)

- **Sheet**: `tweets`; **columns**: 31, identical across all 12 files.
- **Rows**: 12,581,535 total (matches the summary exactly).
- **Distinct author accounts** (union of `user.id` over all 12 files): **16,736**
  — the cumulative distinct count reaches 16,736 by `part_006` and never grows
  again. This **directly verifies the node universe**.
- **Columns present**: `timestamp, is_removed, id, created_at, is_quote_status,
  user, text, ocr_text, text_lang, lang, text_tags, text_hashtags, text_emojis,
  user_mentions, media, likes, retweets, reply_count, quoted_count, reply_status,
  quoted_status, retweeted_status, location_tags, bookmarks, views, place,
  impression, engagement, sentiment, topic, copy_count`.

### Identifier / graph fields

| Concept | Field | Notes |
|---|---|---|
| Account identifier | `user.id` (int inside `user` blob) | stable, immutable → node identity |
| Username | `user.username` | mutable, secondary |
| Tweet identifier | `id` (top-level) | **stored as float / scientific notation** — see data quality |
| Snapshot field | *(none)* | must be derived from `created_at` |
| Node-index field | *(none)* | must be assigned; not present |
| Relation: mention | `user_mentions` = list of `{id}` | directed: author → mentioned account |
| Relation: retweet | `retweeted_status.user.id` | directed: author → retweeted account |
| Relation: reply | `reply_status.user.id` | directed: author → replied-to account |
| Relation: quote | `quoted_status.user.id` | directed: author → quoted account |

Edge availability is abundant (measured on `part_001`): `mention ≈ 912,306`,
`retweet ≈ 670,055`, `reply ≈ 149,842`, `quote ≈ 102,893` non-null targets in a
single 1.05 M-row part. **The full directed relation graph is derivable from
Dataset A alone.**

## Data-quality findings

1. **Tweet-id precision loss (high severity).** The top-level `id` column is a
   float (e.g. `1.548256971999941e+18`). 19-digit snowflake tweet IDs exceed
   float64 integer precision, so `id` is **not reliable as an exact tweet key**
   in Dataset A. Account IDs inside the `user` blob are preserved as integers and
   are unaffected. (`Verified from real files`)
2. **Duplicates retained.** 1,168,525 duplicate tweets detected corpus-wide,
   0 removed. Within-file duplicate `id` rows are small (196 across 12 files);
   the large figure is cross-file. Deduplication is still required. (`Verified from real files`)
3. **Fully-empty columns.** `ocr_text, quoted_count, bookmarks, views,
   engagement, sentiment, topic, copy_count` are **100% null**; `media` ~81% null;
   `text_emojis` ~99% null. These are schema placeholders with no data. (`Verified from real files`)
4. **Sparse text nulls / empties.** `text` has a handful of nulls/empties per
   file (27 empty texts total). (`Verified from real files`)
5. **Temporal spread.** Observed `created_at` (epoch) spans **2017-10-28 →
   2026-05-30** across the 12 files, concentrated ~2020–2022. The late maximum
   should be re-checked during preprocessing (possible epoch outliers). (`Verified from real files`)

## Dependencies between files

- The 12 `.xlsx` parts are **peers** (mechanical row-cap split); order is by part
  number only. Global dedup and node/edge/snapshot derivation must operate
  **across** all 12.
- `extraction_summary.json` documents lineage from the 41 `bts-*` raw files (which
  are **not** in this folder) and is the authority for corpus totals.

## The 10,040 vs 16,736 question

- `extraction_summary.json` fixes the frozen population at **16,736** and this is
  verified from the data (distinct `user.id` = 16,736).
- The folder is named `core_army_pro_fans` (Core ARMY **+** Pro-fans).
- Most consistent reading: **10,040 "Core ARMY" ⊂ 16,736 frozen population**
  (Core ARMY plus Pro-fans). No file in Dataset A records the 10,040/6,696 split,
  so the exact partition is **`Unresolved`** from these files; the total (16,736)
  is `Verified from real files`.
