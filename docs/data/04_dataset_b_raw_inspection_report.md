# 04 — Dataset B Raw Schema Inspection Report (Phase 4)

Method: each inspected workbook was downloaded **one at a time**, read with a
streaming/whole-sheet pass, and **evicted** before the next. No workbook was
modified. No cross-corpus dedup, text cleaning, or embedding was performed.
No raw tweet text is reproduced here — only column names, counts, and aggregates.

**Coverage:** all 70 files inventoried by exact size (manifest); **12
representative files fully inspected** (`statuses-0,1,2,10,20,30,34,35,49,60,68,69`),
spanning the whole index range and the size range. Deep inspection of the
remaining 58 is deferred (see report 01, "no full download" constraint) and is
resumable with the same tooling.

## Canonical schema (`Verified from real files`, all inspected files identical)

Single worksheet `Sheet1`, **10 columns, fixed order**:

| # | Column | Candidate type | Role |
|---|---|---|---|
| 1 | `id` | integer-as-**string** | **tweet/post ID** (precision preserved) |
| 2 | `created_at` | epoch-seconds-as-string | **timestamp** |
| 3 | `user` | JSON/dict blob | **author** → `{id, followers, username, title, political_category}` |
| 4 | `text` | string | **tweet text** |
| 5 | `likes` | numeric | engagement count |
| 6 | `retweets` | numeric | engagement **count** (not an edge target) |
| 7 | `reply_count` | numeric | engagement **count** (not an edge target) |
| 8 | `quoted_count` | numeric | engagement **count** (not an edge target) |
| 9 | `bookmarks` | numeric | engagement count |
| 10 | `views` | numeric | engagement count |

Schema-family signature (order-insensitive): `f903842ea287` — **one family** across
all 12 inspected files; **zero** column-order variants observed.

### Candidate field mapping

- tweet ID → `id` · author account ID → `user.id` · username → `user.username`
- timestamp → `created_at` · text → `text` · language → **none** (no `lang` column)
- reply/retweet/quote/mention **edge targets → NONE**. Dataset B has only
  engagement *counts* (`reply_count`, `retweets`, `quoted_count`); it does **not**
  record replied-to / retweeted / quoted / mentioned **account IDs**. Therefore
  **Dataset B cannot contribute directed account-account edges** — that structure
  exists only in Dataset A. (`Verified from real files`)

## Per-file measured statistics (12 inspected files)

| File | Rows | Distinct tweet IDs | Dup-ID rows | Unique authors | created_at min | created_at max | Invalid ts |
|---|---|---|---|---|---|---|---|
| statuses-0 | 1,001,628 | 1,001,604 | 24 | 123 | 2012-08-27 | 2026-07-15 | 0 |
| statuses-1 | 1,000,044 | — | 12 | 206 | 2016-09-12 | 2026-07-15 | 0 |
| statuses-2 | 1,001,768 | — | 1 | 286 | 2018-02-06 | 2026-07-15 | 0 |
| statuses-10 | 1,001,870 | — | 26 | 205 | 2020-03-21 | 2026-07-15 | 0 |
| statuses-20 | 1,002,981 | — | 24 | 198 | 2020-09-23 | 2026-07-15 | 0 |
| statuses-30 | 1,000,044 | — | 51 | 169 | 2021-02-15 | 2026-07-15 | 0 |
| statuses-34 | 1,008,661 | — | 33 | 239 | 2021-03-30 | 2026-07-16 | 0 |
| statuses-35 | 1,000,593 | — | 5 | 240 | 2021-04-12 | 2026-07-16 | 0 |
| statuses-49 | 1,004,133 | — | 69 | 213 | 2021-08-31 | 2026-07-16 | 0 |
| statuses-60 | 1,005,329 | — | 39 | 233 | 2022-03-26 | 2026-07-17 | 0 |
| statuses-68 | 1,002,786 | — | 395 | 177 | 2023-05-28 | 2026-07-17 | 0 |
| statuses-69 | 1,003,077 | — | 746 | 210 | 2023-12-02 | 2026-07-17 | 0 |

(Full per-column null counts and distinct counts for every inspected file live in
`dataset_b_file_statistics.parquet`.)

### Salient patterns

1. **Files are per-account history chunks.** Each ~1 M-row file contains only
   **~120–290 distinct authors**. `created_at` **min rises monotonically with the
   file index** (file 0 → 2012, file 69 → 2023) while **max ≈ the scrape window
   (mid-July 2026)**. This is consistent with files packed by account, accounts
   ordered by age, each account's full history ending at scrape time. (`Verified from real files`)
2. **Tweet IDs are clean strings** — no float precision loss (unlike Dataset A).
   Within-file duplicate-ID rows are small (1–746). (`Verified from real files`)
3. **Sparse engagement metrics.** e.g. `statuses-0`: `quoted_count` ~93% null,
   `bookmarks` ~92% null, `views` ~93% null, `reply_count` ~0.7% null. (`Verified from real files`)
4. **Empty text is rare** (~30 rows/file). No invalid timestamps found
   (`ts_invalid = 0` everywhere). (`Verified from real files`)
5. **No language field** — language must be detected during preprocessing if needed.

## Workbook classification

| Class | Files |
|---|---|
| **Canonical schema** | all 12 inspected files (single family `f903842ea287`) |
| Compatible schema variant | none observed |
| Type-only variant | none observed |
| Requires explicit adapter | none observed |
| Malformed / unreadable | none observed; manifest marks all 70 `valid` |

Expectation for the remaining 58 files: **canonical** (identical source pipeline,
identical pattern, all manifest-`valid`). To be confirmed during the pilot.
