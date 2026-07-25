# 13 — Cloud Agent Readiness Audit

Assessment of whether the environment and datasets are ready for the next stages.

## Environment

| Item | State | Detail |
|---|---|---|
| Python / libs | ✅ | Python 3.12; `pandas`, `pyarrow`, `openpyxl`, `python-calamine`, `gdown` installed. |
| Network egress | ✅ | `google.com` / `googleapis.com` reachable (HTTP 200). |
| Disk | ✅ | ~236 GB free (source B ≈ 9.58 GB; A ≈ 3.4 GB). |
| RAM | ✅ | ~14 GB free — sufficient for one ~1 M-row workbook at a time. |
| Dataset READ access | ✅ | Anonymous public download (`gdown`); byte-level verified for A & B. |
| Drive API (authenticated) | ❌ | 403 unauthenticated; **no credentials** in environment. |
| Env vars (`DATASET_*`, `TDMEC_OUTPUT_*`, `GOOGLE_APPLICATION_CREDENTIALS`) | ❌ | **Not injected**; folder ids taken from the task and kept git-ignored. |
| Output Drive WRITE access | ❌ | `OUTPUT_WRITE_ACCESS_NOT_CONFIRMED` (no credentials). |

## Readiness gaps & required actions

1. **Provide Google Drive credentials** (service-account JSON via
   `GOOGLE_APPLICATION_CREDENTIALS`, shared to the source + output folders) and set
   `DATASET_A_SOURCE`, `DATASET_B_SOURCE`, `TDMEC_OUTPUT_DRIVE_FOLDER_ID` as Cloud
   Agent secrets. Without these, outputs cannot be **published/verified** to Drive
   (they currently persist to Git + a local run dir only).
2. **Full Dataset B inspection** (58 remaining files) — deferred by the "no full
   download" rule; resumable via `scripts/run_discovery.py --b-sample …`.
3. **Immutable node-index map** (0..16,735) must be created before edge/tensor build.
4. **Global dedup** (A: 1.17 M flagged; B: cross-file) not yet done.

## What is ready now

- Reusable, tested, platform-neutral discovery tooling (local / gdrive-anon /
  gdrive-api); runs unchanged on Colab/Kaggle/Lightning/Linux via env config.
- Verified byte-level read access + repeatable enumeration for both datasets.
- Verified node universe (16,736), verified schemas, verified join key,
  reconciliation on a 12-file B sample, and a written pilot spec.

## Environment-setup recommendation

The dependency set (`pandas`, `pyarrow`, `openpyxl`, `python-calamine`, `gdown`,
`google-api-python-client`, `google-auth*`) and, critically, the **Drive
credentials + `DATASET_*`/`TDMEC_OUTPUT_*` secrets** should be baked into the
Cloud Agent environment so future agents don't repeat setup and can publish to
Drive. See `requirements.txt` and `config/discovery.example.env`.

## Verdict

- **Read-only discovery:** ✅ ready and largely complete.
- **Controlled pilot (local outputs / Git):** ✅ safe to begin.
- **Drive-published pipeline & full 70-file run:** ⛔ blocked on credentials +
  the explicit "no full download yet" hold.
