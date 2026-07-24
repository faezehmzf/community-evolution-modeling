# 00 — Data-Access Verification Report (Phase 0)

Status legend: `Verified from real files` · `Verified from source code` · `Inferred` · `Proposed` · `Unresolved` · `Legacy/incorrect`

## Summary

| Question | Result |
|---|---|
| Byte-level READ access to Dataset A | **Confirmed** (`Verified from real files`) |
| Byte-level READ access to Dataset B | **Confirmed** (`Verified from real files`) |
| Output Google Drive WRITE access | **NOT confirmed** — `OUTPUT_WRITE_ACCESS_NOT_CONFIRMED` |

Byte-level access to both source datasets is **reliable and repeatable** through
the anonymous ("anyone with the link") public-download path (`gdown`). The
authenticated Google Drive API path is **not** available in this environment
because no credentials were injected (see below).

## Access mechanism actually observed

- The two source folders are **publicly visible** (folder web pages return HTTP `200`).
- The Google Drive **REST API rejects unauthenticated callers** (`HTTP 403`,
  "Method doesn't allow unregistered callers"), so the authenticated API path
  requires credentials.
- **No Google Drive credentials and none of the documented environment
  variables** (`DATASET_A_SOURCE`, `DATASET_B_SOURCE`, `DATASET_A_DRIVE_FOLDER_ID`,
  `DATASET_B_DRIVE_FOLDER_ID`, `TDMEC_OUTPUT_DRIVE_FOLDER_ID`,
  `GOOGLE_APPLICATION_CREDENTIALS`) were present in the Cloud Agent environment
  at run time. Folder identifiers were taken from the task description and placed
  in a **git-ignored** local config file; they are never committed. (`Verified from real files`)
- Because the folders are shared publicly, read-only enumeration and per-file
  download succeed **anonymously** via `gdown`, which is sufficient for the
  read-only discovery pass.

## Phase 0 checklist

### Dataset A

1. Files listed: **13 visible entries** (`Verified from real files`).
2. Count: 12 × `.xlsx` + 1 × `extraction_summary.json`.
3. Names/extensions: `core_army_pro_fans_tweets_part_001.xlsx` … `_012.xlsx`, `extraction_summary.json`.
4. Representative file downloaded: `core_army_pro_fans_tweets_part_012.xlsx` (274,061,016 bytes).
5. Content verification: begins with the ZIP magic `PK\x03\x04`; **not** an HTML
   login/warning page, not empty, not truncated, not a permission error.
6. Parser: opened with `openpyxl`/`calamine`; single worksheet `tweets`,
   1,047,210 data rows × 31 columns — matches `extraction_summary.json`.
7. Repeatable: re-enumeration and re-download are deterministic and cached.

### Dataset B

1. Files listed: **71 visible entries** (`Verified from real files`).
2. Count: 70 × `.xlsx` (`statuses-0.xlsx` … `statuses-69.xlsx`) + 1 × `download_manifest.json`.
3. Names/extensions and **exact reported sizes** are available from
   `download_manifest.json` (per-file `expected_file_size`, 109–159 MB, total ≈ 9.58 GB).
4. Representative file downloaded: `statuses-68.xlsx` (109,030,187 bytes; the smallest).
5. Content verification: ZIP magic present; not HTML/empty/truncated/permission page.
6. Parser: single worksheet `Sheet1`, 1,002,786 data rows × 10 columns.
7. Repeatable: deterministic, cached.

## Discrepancies vs. the authoritative brief

| Expectation | Observation | Note |
|---|---|---|
| Dataset A "12 files" | 12 `.xlsx` **+ 1 `extraction_summary.json`** (13 entries) | Count of data files matches; a metadata JSON is also present. |
| Dataset A = "frozen population of 16,736 accounts" (graph-like) | 12 **tweet** workbooks (account-filtered tweet extract); **no** graph/node-index/edge/snapshot files present | Dataset A is a *preprocessed tweet extract*, not a graph artifact set. See report 02/03. |
| Dataset B "~70 raw Excel files" | Exactly **70** `statuses-*.xlsx` (+ 1 manifest) | Matches. Confirmed raw. |

## Output write access

`TDMEC_OUTPUT_DRIVE_FOLDER_ID` is documented but was **not** present as an
environment variable, and **no Drive credentials** were injected. The
write-access self-test therefore could not run:

```
credentials_available = false
write_access_confirmed = false
reason = "no Google Drive credentials in environment"
```

Consequence: the `_cursor_write_access_test.json` create/read/delete cycle and
publication of outputs to the Drive output folder are **not possible** in this
environment. Per policy this is reported as `OUTPUT_WRITE_ACCESS_NOT_CONFIRMED`.
All discovery outputs are instead persisted to **Git** (sanitized reports +
small JSON summaries) and to the **git-ignored local run directory** (full
parquet inventories + per-file logs), and the tooling (`DrivePublisher`) is
ready to publish + verify to Drive as soon as credentials are supplied.

## Reproduction

```
pip install -r requirements.txt
# configure DATASET_A_SOURCE / DATASET_B_SOURCE (never commit real ids)
python -m tdmec_discovery verify-access --out artifacts/discovery
python -m tdmec_discovery check-drive-write
```

Machine-readable summary: `artifacts/discovery/access_verification.json`.
