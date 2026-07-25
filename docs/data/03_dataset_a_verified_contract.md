# 03 — Dataset A Verified Graph Contract (Phase 3)

Each element of the *candidate* contract is verified against the real files
rather than assumed. No mapping was modified and no node was renumbered.

## Candidate contract vs. evidence

| Candidate claim | Verdict | Evidence |
|---|---|---|
| `node_count = 16,736` | **Verified from real files** | Distinct `user.id` across all 12 workbooks = 16,736; equals `extraction_summary.unique_matched_accounts`. |
| `valid_node_indices = 0 … 16,735` | **Proposed / Unresolved** | No node-index mapping artifact exists in Dataset A. A contiguous 0…16,735 index is a *valid design* but must still be **created**; it is not yet present. |
| `directed_edges = true` | **Verified (structure)** | Edges are author → target-account derived from `user_mentions`, `retweeted_status.user.id`, `reply_status.user.id`, `quoted_status.user.id`; direction is intrinsic. |
| `relation_mapping {mention:0, retweet:1, reply:2, quote:3}` | **Types Verified; codes Proposed** | All four relation types are present as populated columns in Dataset A. The specific integer codes are a naming convention **not stored** in any file. |
| `candidate_snapshot_count = 35` | **Inferred (consistent)** | No snapshot field exists. 35 = number of **quarterly** bins spanning Dataset A's observed `created_at` range (2017-Q4 → 2026-Q2 inclusive = 35 quarters). Consistent, but the binning must be defined and applied. |

## Verified contract (what the files actually support today)

```
population:
  frozen_node_count: 16736            # VERIFIED (distinct user.id)
  node_identity_key: user.id          # VERIFIED (int, precision-safe)
  node_index_map: NOT PRESENT         # must be created (proposed 0..16735)

edges:
  directed: true                      # VERIFIED (by construction)
  derivable_from: Dataset A only      # VERIFIED (edge blobs populated)
  relation_types:                     # VERIFIED present
    - mention   (user_mentions[].id)
    - retweet   (retweeted_status.user.id)
    - reply     (reply_status.user.id)
    - quote     (quoted_status.user.id)
  relation_codes: PROPOSED            # mention=0,retweet=1,reply=2,quote=3 (not stored)
  edge_weight: PROPOSED               # e.g. interaction count per (src,dst,rel,snapshot)

time:
  snapshot_field: NOT PRESENT         # derive from created_at (epoch)
  observed_created_at_range: 2017-10-28 .. 2026-05-30   # VERIFIED (see caveat)
  candidate_snapshots: 35 quarterly bins (2017Q4..2026Q2)  # INFERRED, consistent

tweet_key:
  field: id
  status: UNRELIABLE in Dataset A     # float precision loss; re-source exact IDs
```

## Required actions before the contract is fully realized

1. **Assign node indices** `0..16,735` over the verified 16,736 `user.id` set and
   persist the mapping (immutable thereafter). *(Do not renumber later.)*
2. **Deduplicate** tweets globally (1.17 M duplicates flagged, 0 removed).
3. **Define the snapshot calendar** (quarterly bins) and validate the 35-count
   against the cleaned `created_at` distribution (re-check the 2026 tail).
4. **Build directed edge lists** per relation type + snapshot from the blob fields.
5. Treat tweet `id` as **untrusted** in Dataset A (float); prefer exact IDs from a
   string-typed source (Dataset B stores IDs as strings — see reports 04/06).

Everything marked `Proposed`/`Inferred`/`Unresolved` above is a **design
decision or a build step**, not a fact readable from today's Dataset A files.
