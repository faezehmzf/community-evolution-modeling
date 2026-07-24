# 07 — Dataset B Preprocessing Pilot Plan (Phase 7, design only)

Design only. **Not executed** across all 70 files during discovery. The pilot
processes a tiny, controlled set to validate the pipeline before any full run.

## Pilot input files

- **Representative normal workbook:** `statuses-2.xlsx` (canonical schema, ~1.00 M
  rows, mid-range size, low within-file duplicates).
- **Quality-difference workbook:** `statuses-69.xlsx` (same canonical schema but
  the **highest** within-file duplicate-ID count observed — 746 — and the latest
  `created_at.min`; exercises dedup + late-window snapshot assignment).

No third file is needed: only **one** schema family exists in Dataset B, so a
"schema variant" file is unavailable; the quality-difference file substitutes.

## Normalized target schema (Parquet)

| Column | Type | Source | Notes |
|---|---|---|---|
| `tweet_id` | `string` (or uint64) | B.`id` | exact key; keep as string to avoid precision loss |
| `author_id` | `int64` | B.`user.id` | join key to frozen population |
| `author_username` | `string` | B.`user.username` | secondary |
| `created_at` | `timestamp[us, UTC]` | B.`created_at` (epoch s) | parsed |
| `text_raw` | `string` | B.`text` | preserved verbatim |
| `text_clean` | `string` | derived | light normalization layer (see below) |
| `likes,retweets,reply_count,quoted_count,bookmarks,views` | `int64` (nullable) | B.* | counts (may be null) |
| `frozen_node` | `bool` | reconciliation | `author_id ∈ 16,736` |
| `node_index` | `int32` (nullable) | mapping | from the immutable 0..16,735 map (join only) |
| `snapshot_id` | `int16` | derived | quarterly bin index (see calendar) |
| `source_file` | `string` | provenance | e.g. `statuses-2.xlsx` |

## Identifier normalization

- `tweet_id`: strip, keep as string; validate `^\d{15,20}$`; reject/flag others.
- `author_id`: parse from the `user` blob via safe literal-eval (fallback regex
  `'id'\s*:\s*(\d+)`); cast to int64.
- `username`: from blob; lowercase copy kept only for optional matching.

## Timestamp parsing

- `created_at` is epoch **seconds as string** → `to_datetime(unit="s", utc=True)`.
- Guard rails: reject < 2006-03-21 (pre-Twitter) and > run-time + 1 day; count and
  quarantine violations (none seen in the sample).

## Text preservation & cleaning layers

1. **Preserve** `text_raw` byte-for-byte (never overwrite).
2. **Clean layer** (`text_clean`) is additive: Unicode NFC normalization,
   whitespace collapse, optional URL/mention placeholdering. **No stemming, no
   language filtering, no embedding in the pilot.**

## Duplicate handling

- **Within-file**: drop exact duplicate `tweet_id` (keep first).
- **Cross-file (pilot = 2 files)**: dedup on `tweet_id` across the two files;
  record counts. Full-corpus dedup is a separate, later stage.

## Frozen-node reconciliation

- Load the immutable node-index map (0..16,735 over the 16,736 `user.id`).
- Left-join on `author_id`; set `frozen_node` and `node_index`.
- **Never** create new node indices; rows with `author_id ∉ frozen` keep
  `node_index = null` and `frozen_node = false` (expected: ~0 in Dataset B).

## Snapshot assignment

- Quarterly calendar, 35 bins, **2017-Q4 → 2026-Q2** (report 03).
- `snapshot_id = quarter_index(created_at)`; out-of-range (e.g. B's pre-2017
  tweets) → `snapshot_id = -1` (flagged, not dropped) pending a calendar decision.

## Parquet output schema

- Partition by `snapshot_id` then `source_file`; `zstd` compression; row-group ~128 MB.
- One dataset directory per stage under the run's `processed/` output.

## Checkpoint & resume behavior

- Per-file checkpoint record `{file, sha256, rows_in, rows_out, status}` written
  atomically after each file (reuses `cache._cache_index.json` pattern).
- Resume = skip files whose `sha256` + `status=done` already recorded.
- Atomic publication: write temp → validate → upload → verify size/checksum →
  add to manifest → mark stage done (only then).

## Validation gates (must all pass before scaling to 70 files)

1. Row conservation: `rows_out == rows_in − within_file_dups − quarantined`.
2. Schema exactly matches the normalized target schema; dtypes correct.
3. `author_id` non-null rate ≥ 99.9%; `tweet_id` valid-pattern ≥ 99.9%.
4. `frozen_node` true-rate consistent with report 06 (~100% for B sample).
5. Timestamp quarantine rate == 0 (as observed) — else stop and investigate.
6. No `text_raw` mutation (hash preserved vs. source rows).

## Expected disk use (pilot)

- 2 input files ≈ 143 MB + 110 MB ≈ **0.25 GB** transient (evicted after).
- Parquet output for ~2 M rows (10 cols + derived) ≈ **0.2–0.4 GB**.
- Peak working set well under 2 GB.

## Proposed pilot command

```bash
python scripts/run_pilot.py \
  --dataset-b-source "$DATASET_B_SOURCE" \
  --files statuses-2.xlsx,statuses-69.xlsx \
  --node-index-map artifacts/discovery/runs/<run_id>/node_index_map.parquet \
  --out artifacts/pilot \
  --snapshots quarterly --snapshot-start 2017Q4 --snapshot-count 35 \
  --dry-run-validate
```

(`run_pilot.py` is intentionally **not** implemented yet; this document is the
spec it must satisfy. `--dry-run-validate` runs all gates without publishing.)
