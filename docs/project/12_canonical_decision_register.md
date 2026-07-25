# 12 — Canonical Decision Register

Authoritative, corrected project facts after the read-only discovery pass. Every
statement is classified: `Verified from real files` · `Verified from source code`
· `Inferred` · `Proposed` · `Unresolved` · `Legacy/incorrect`.

> Repository note: this repository previously contained **no** `docs/project/`
> specifications (only a stub `README.md`). Any external/legacy description that
> called Dataset B a single "canonical processed Parquet corpus," a "pilot
> corpus," or a "smaller processed dataset" is recorded below as
> **`Legacy/incorrect`** — no such lineage is supported by the real files.

## Dataset A

| # | Statement | Classification |
|---|---|---|
| A1 | Dataset A = 12 `.xlsx` tweet parts + `extraction_summary.json` (13 entries). | `Verified from real files` |
| A2 | It is a **preprocessed, account-filtered tweet extract**, not a graph/model-ready artifact set. | `Verified from real files` |
| A3 | Frozen population = **16,736** distinct accounts (`user.id`). | `Verified from real files` |
| A4 | 12,581,535 matched tweets; built from 41 raw `bts-*` files. | `Verified from source code` (extraction_summary) |
| A5 | 1,168,525 duplicates detected, **0 removed**; dedup still required. | `Verified from source code` |
| A6 | Directed relation edges (mention/retweet/reply/quote) are embedded in A blobs and derivable from A alone. | `Verified from real files` |
| A7 | Tweet `id` is stored as **float** → precision loss; unreliable as an exact key. | `Verified from real files` |
| A8 | Several columns are 100% null (ocr_text, quoted_count, bookmarks, views, engagement, sentiment, topic, copy_count). | `Verified from real files` |
| A9 | No node-index map, edge lists, snapshots, or tensors exist yet. | `Verified from real files` |
| A10 | "10,040 Core ARMY" ⊂ 16,736 (Core ARMY + Pro-fans); exact split not in files. | `Inferred` / split `Unresolved` |

## Dataset B (corrected canonical specification)

| # | Statement | Classification |
|---|---|---|
| B1 | **Dataset B = ~70 (exactly 70) RAW `statuses-*.xlsx` files** + `download_manifest.json`. | `Verified from real files` |
| B2 | **No official preprocessing has been performed** on Dataset B. | `Verified from real files` |
| B3 | Total reported size ≈ **9.58 GB** (109–159 MB/file). | `Verified from real files` (manifest) |
| B4 | Single canonical schema: 10 columns `id, created_at, user, text, likes, retweets, reply_count, quoted_count, bookmarks, views`. | `Verified from real files` (12-file sample) |
| B5 | Tweet `id` stored as **string** (exact); `created_at` = epoch-seconds string. | `Verified from real files` |
| B6 | Dataset B has **no relational edge fields** — only engagement counts. | `Verified from real files` |
| B7 | Files are per-account history chunks (~120–290 authors/file; min-time rises with index). | `Verified from real files` (sample) |
| B8 | Provenance = `https://<redacted-external-host>/static/reports/statuses-*.xlsx`. | `Verified from real files` (manifest) |
| B9 | A previously "canonical processed Dataset B Parquet corpus / pilot corpus / smaller processed dataset". | **`Legacy/incorrect`** — no lineage from the 70 raw files is proven |
| B10 | Full-corpus distinct tweet/author counts (all 70 files). | `Unresolved` (deep-inspection deferred) |

## Cross-dataset

| # | Statement | Classification |
|---|---|---|
| X1 | Canonical join key = numeric **`user.id`** (precise in both datasets). | `Verified from real files` |
| X2 | 100% of sampled Dataset B authors (2,498) are in the frozen 16,736; 0 outside. | `Verified from real files` (sample) |
| X3 | Whether all 70 B files cover the full 16,736 or a subset. | `Unresolved` |
| X4 | Temporal overlap A(2017–2026) ∩ B(2012–2026) is substantial; B extends earlier. | `Verified from real files` (sample) |
| X5 | Do **not** join on A's tweet `id` (float-lossy); use B's string IDs for tweet-level linkage. | `Verified from real files` |

## Graph contract

| # | Statement | Classification |
|---|---|---|
| G1 | `node_count = 16,736`. | `Verified from real files` |
| G2 | `directed_edges = true` (by construction from A). | `Verified` (structure) |
| G3 | relation types mention/retweet/reply/quote all present. | `Verified from real files` |
| G4 | relation codes `{mention:0,retweet:1,reply:2,quote:3}`. | `Proposed` (not stored) |
| G5 | node indices `0..16,735`. | `Proposed` (map must be created) |
| G6 | `candidate_snapshot_count = 35` = quarterly bins 2017-Q4 … 2026-Q2. | `Inferred` (consistent with A's range) |

## Decisions

1. **Dataset B is raw**; treat the 70 Excel files as the sole canonical source.
2. **Join on `user.id`**; keep tweet IDs as strings from Dataset B.
3. **Build the graph from Dataset A**; enrich text/timestamps from Dataset B.
4. **Create (once) an immutable node-index map** over the 16,736 accounts.
5. **Snapshots are quarterly** (35 bins) pending confirmation on cleaned data.
