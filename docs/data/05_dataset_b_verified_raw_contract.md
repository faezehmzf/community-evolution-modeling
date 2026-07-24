# 05 — Dataset B Verified Raw Contract (Phase 4)

The canonical raw contract for Dataset B, verified against 12 representative
files (all identical). No text cleaning, cross-file dedup, or embedding done.

```
dataset_b:
  status: RAW (no official preprocessing)          # VERIFIED
  files: 70 x statuses-{0..69}.xlsx + download_manifest.json   # VERIFIED
  total_reported_size_bytes: 9582913975 (~9.58 GB) # VERIFIED (manifest)
  provenance: https://<redacted-external-host>/static/reports/statuses-*.xlsx  # VERIFIED (manifest)

  workbook:
    sheet: Sheet1                                  # VERIFIED (single sheet)
    columns: 10 (fixed order)                      # VERIFIED
    schema_family: f903842ea287 (single family)    # VERIFIED (0 variants observed)

  columns:
    id:           int-as-string   # tweet id (EXACT, precision-safe)   VERIFIED
    created_at:   epoch-seconds-as-string # timestamp                  VERIFIED
    user:         json blob {id, followers, username, title, political_category}  VERIFIED
    text:         string          # raw tweet text (uncleaned)         VERIFIED
    likes:        numeric                                              VERIFIED
    retweets:     numeric  (COUNT, not an edge target)                 VERIFIED
    reply_count:  numeric  (COUNT, not an edge target)                 VERIFIED
    quoted_count: numeric  (COUNT, not an edge target)                 VERIFIED
    bookmarks:    numeric                                              VERIFIED
    views:        numeric                                              VERIFIED

  keys:
    tweet_id:  id            # exact string key                        VERIFIED
    author_id: user.id       # stable numeric account id               VERIFIED
    username:  user.username # mutable, secondary                      VERIFIED

  edges:
    account_account_edges: ABSENT   # no replied-to/retweeted/quoted/mentioned account ids  VERIFIED
    only engagement counts present

  quality:
    rows_per_file: ~1,000,000 (Excel row-cap chunks)                  VERIFIED (sample)
    distinct_authors_per_file: ~120-290 (per-account history chunks)  VERIFIED (sample)
    within_file_duplicate_id_rows: small (1-746)                      VERIFIED (sample)
    invalid_timestamps: 0                                             VERIFIED (sample)
    empty_text_rows: ~30/file                                        VERIFIED (sample)
    language_field: none                                             VERIFIED

  temporal:
    per_file: created_at.min rises with index (2012..2023); max ~= scrape (2026-07)  VERIFIED (sample)
    corpus_span: ~2012 .. 2026-07                                    VERIFIED (sample)

  deferred (resumable):
    deep inspection of the 58 non-sampled files                      PROPOSED
    exact corpus-wide distinct tweet & author counts                 PROPOSED
```

## Contract implications

- Dataset B is a **text + engagement** corpus for individual accounts. It is the
  right source for **exact tweet IDs** and **raw text**, but it carries **no
  relational edges** — the graph must come from Dataset A.
- Because files are per-account chunks, **cross-file dedup** and **account→file
  indexing** are required before any per-account aggregation.
- `user.id` is the join key to the frozen population (see report 06).
