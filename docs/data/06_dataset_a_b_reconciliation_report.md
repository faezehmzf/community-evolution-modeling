# 06 — Dataset A ↔ Dataset B Reconciliation Report (Phase 5)

Uses only verified schemas/keys. The 16,736-node universe is **not** expanded and
**no** new node indices are assigned.

## Candidate join keys

| Candidate | In A | In B | Suitability |
|---|---|---|---|
| **Numeric account ID (`user.id`)** | ✅ int (precise) | ✅ int (precise) | **Best** — stable, immutable, precise in both |
| Username / screen name (`user.username`) | ✅ | ✅ | Secondary — mutable, not unique over time |
| Tweet ID (`id`) | ⚠️ **float (lossy)** | ✅ string (exact) | Usable A→B only after re-sourcing exact A IDs |
| Replied-to / retweeted / quoted / mentioned account ID | ✅ (A blobs) | ❌ absent in B | Edge keys exist only in A |

**Chosen join key: `user.id`** (numeric author account id embedded in the `user`
blob), present and precise in both datasets. (`Verified from real files`)

## Account reconciliation (measured)

Frozen population from Dataset A (all 12 files): **16,736** distinct `user.id`
(`Verified from real files`; equals `extraction_summary`).

Dataset B **sample** = 12 files (`statuses-0,1,2,10,20,30,34,35,49,60,68,69`):

| Metric | Value |
|---|---|
| Unique Dataset B author accounts (sample) | **2,498** |
| Sample authors that are frozen accounts | **2,498 (100%)** |
| Sample authors **outside** the frozen population | **0** |
| Ambiguous account matches | 0 (exact integer key) |
| Frozen accounts seen in the B sample | 2,498 |
| Frozen accounts not seen in the B sample | 14,238 (expected: only 12/70 files sampled) |

**Key finding:** every Dataset B author in the sample is a member of the frozen
16,736 population, and none fall outside it. This strongly indicates Dataset B is
a **per-account status corpus restricted to (a subset of) the same frozen
accounts** as Dataset A — i.e. the two datasets are **account-aligned** even
though they came from different scrape pipelines (A ← `bts-*`, B ← `statuses-*`).
(`Verified from real files` for the sample; full-corpus confirmation `Proposed`.)

### Extrapolation (to verify in the pilot)

12 files → 2,498 unique accounts, ~120–290/file with heavy cross-file overlap.
Whether the full 70-file corpus covers **all** 16,736 or a strict subset is
**`Unresolved`** until the remaining files are inspected. The measured direction
("B ⊆ frozen") is expected to hold.

## Temporal overlap

| Dataset | Observed `created_at` span |
|---|---|
| A | 2017-10-28 → 2026-05-30 (concentrated ~2020–2022) |
| B (sample) | ~2012-08 → ~2026-07 |

Overlap is **substantial** (roughly 2017 → 2026). Dataset B extends **earlier**
(back to 2012) and slightly **later** than A. For quarterly snapshots, both
datasets populate the candidate 2017-Q4 → 2026-Q2 window; B additionally covers
pre-2017 quarters that A does not.

## Candidate records per quarterly snapshot

- The 35-quarter calendar (report 03) is derived from A's `created_at`.
- Per-snapshot record counts require the cleaned, deduplicated, snapshot-binned
  data and are therefore **`Proposed`** for the pilot, not computed here (no full
  preprocessing yet). The tooling captures per-file `created_at` min/max needed to
  seed this.

## Reconciliation summary

- **Join on `user.id`.** Do **not** join on Dataset A's tweet `id` (float-lossy);
  if tweet-level linkage is needed, take exact IDs from Dataset B (strings).
- Dataset A = **graph/edge + node universe**; Dataset B = **exact tweet text +
  timestamps + engagement** for the same accounts.
- No account expansion and no re-indexing performed.
