# 06 — Cloud Acceptance Checklist

**Purpose.** A binary, evidence-checkable checklist the Cloud Agent must satisfy
per stage before an artifact is considered done. Nothing here authorizes a
DO-NOT-START stage from `03`/`05`.

## Preconditions

- [ ] `python -m pytest -q` → all pass (baseline: **31 passed**; must not regress).
- [ ] Drive credentials + `DATASET_*` / `TDMEC_OUTPUT_*` env vars injected (for any
      Drive publication) — else outputs stay Git + local run dir (readiness audit).
- [ ] Relevant decisions from `05` confirmed for the stage being run.

## Node-index map (A-02) — FIRST TASK

- [ ] Built from all 12 Dataset A workbooks.
- [ ] Exactly **16,736** rows; indices contiguous **0..16,735**; no gaps/dupes.
- [ ] Sorted-integer determinism; re-run SHA-256 identical.
- [ ] Published to `manifests/node_index_map.parquet` with recorded checksum.
- [ ] Regression test added; suite still green.

## Dataset A graph (A-00…A-10, excl. A-08)

- [ ] 12 source files validated; ZIP magic; checksums recorded; source unchanged.
- [ ] Schema exactly 31 columns (sig `83e45e17ffc3`); `user.id` parsed (no eval).
- [ ] A `id` never used as a join key (float-lossy).
- [ ] Relations extracted from the 4 blob fields; endpoints mapped to node indices;
      out-of-universe endpoints dropped; self-loop policy (D-1) applied + logged.
- [ ] Relation codes match confirmed D-2 ordering.
- [ ] Duplicate report produced; nothing silently dropped (D-3 policy applied if set).
- [ ] Every retained event has `snapshot_id ∈ 0..34`; out-of-range flagged not dropped.
- [ ] Edges aggregated `GROUP BY (snapshot,src,dst,relation)`; `count_raw`≥1;
      Σ `count_raw` == retained distinct events per `(t,r)`; **`weight_log1p =
      log1p(count_raw)` computed** (Q-WGT).
- [ ] Graph is **directed** (no symmetrization); endpoints in range; relation ∈ {0..3}.
- [ ] Atomic publish; manifest + checksums; deterministic re-run SHA-256 identical.
- [ ] A-08 structural features produced with **`F_struct=17`** + `struct_active_mask`
      (Q-FEAT); schema order per `docs/method/14`.

## Dataset B text (B-00…B-07 at scale) — after hold lift + creds

- [ ] 2-file pilot regression reproduced exactly (rows_in 2,004,845 / retained
      1,992,014 / excluded 12,831 / 0 rejected / 496 matched / 747 exact-dup / 11 gates).
- [ ] All 70 files processed; per-file the 11 pilot gates pass.
- [ ] `tweet_id` exact string end-to-end; float rejected.
- [ ] Every retained row has `node_index` (0..16,735) and `snapshot_id` (0..34).
- [ ] Unmatched authors only in `unmatched_accounts.parquet`; 0 new node indices.
- [ ] Text normalization conservative only (NFC+newline+BOM); `text_raw` unchanged;
      all destructive-cleaning flags remain false.
- [ ] Duplicates classified (exact vs conflicting), none silently dropped.
- [ ] Resume verified (no duplicated rows); config-incompat refused.
- [ ] Raw source checksums unchanged before/after.
- [ ] B-08…B-11 (text unit / embedding / aggregation / tensors) **not** started (BLOCKED).

## Global negative checks (must all be TRUE)

- [ ] No embeddings generated.
- [ ] No TDMEC model / training code added.
- [ ] No edge-text artifacts produced.
- [ ] No raw source file modified, renamed, or deleted.
- [ ] No credentials, tokens, or real Drive IDs committed to Git.
- [ ] No UNRESOLVED tensor dimension (`F`,`D_text`,`H`,`K`) materialized.

## Sign-off

- [ ] Reviewer confirms artifacts + checksums match this checklist.
- [ ] Reviewer approves proceeding to the next stage (separate approval per scale-up).
