# 09 — Lightning AI Studio Execution Runbook

Future code executes here without relying on Google Colab or Google Drive mounts.

**Default Studio root:** `/teamspace/studios/this_studio`  
**Default persistent output root:** `/teamspace/studios/this_studio/TDMEC_PROJECT_OUTPUTS/`

## 1. Session bootstrap

1. Open the Lightning AI Studio project (CPU or GPU as required by the task).
2. Confirm local data is present:
   - `Dataset A/core_army_pro_fans_tweets/`
   - `Dataset B/statuses_data/`
   - `TDMEC_PROJECT_OUTPUTS/manifests/node_index_map.parquet`
3. Use the existing repository checkout at
   `community-evolution-modeling/` (do not wipe/reclone casually).
4. Record SHA: `git rev-parse HEAD` → write into run manifests when publishing.
5. Set environment variables:

```text
TDMEC_REPO=/teamspace/studios/this_studio/community-evolution-modeling
TDMEC_OUT=/teamspace/studios/this_studio/TDMEC_PROJECT_OUTPUTS
TDMEC_A=/teamspace/studios/this_studio/Dataset A/core_army_pro_fans_tweets
TDMEC_B=/teamspace/studios/this_studio/Dataset B/statuses_data
TDMEC_RUN_ID=<yyyyymmdd_HHMMSS_shortsha>
```

6. Install dependencies from the repo:

```bash
cd "$TDMEC_REPO"
pip install -r requirements.txt
pip install -e ".[test]"
```

7. Do **not** process data from external Windows Proposed directories or ephemeral
   Colab `/content` paths.

## 2. Preflight checks

- Repo tests: `pytest -q`
- Source path existence and read probe for Dataset A/B
- Confirm node-map policy: N must be 16736 when loaded
- Confirm Dataset B inventory (expected 70 files; report any missing names)
- Refuse to continue if O-EMB still open and the task is embedding/training

## 3. Standard stage sequence

| Step | Command pattern | Machine |
|---|---|---|
| Build node map | `python scripts/build_node_index_map.py --config ...` | CPU |
| Validate map | assert N==16736 + checksum publish | CPU |
| Dataset A graph | `python -m tdmec build-graph ...` (future) | CPU |
| Dataset B normalize | full 70-file job with resume | CPU |
| Controlled pilots | 1–2 shards first | CPU |
| Approve full runs | human gate | — |
| Embeddings | GPU Studio machine / script | GPU |
| Pack tensors | CPU/GPU | CPU |
| Graph-only smoke | GPU | GPU |
| Text-aware smoke | GPU | GPU |
| Train staged | GPU | GPU |
| Eval | GPU/CPU | either |

Use resume mode for long jobs; restart only when intentional.

## 4. Resume interrupted runs

1. Read checkpoint/manifest under the run directory.
2. Re-run the same command with the same config hash and commit.
3. Do not mix artifacts across different SHAs/configs without a new `run_id`.

## 5. Publishing artifacts

For each output:

- Write under `TDMEC_OUT/...`
- Compute SHA256
- Append to manifests with row counts and exclusion reasons
- Only mark `CERTIFIED` after validation gates pass

## 6. Logs

Preserve stdout/stderr, env probe (GPU name if any, CUDA, package versions),
timestamps UTC.

## 7. Forbidden actions

- Training before certified tensors
- Embedding before O-EMB confirmation
- Expanding node universe from Dataset B
- Using float Dataset A tweet IDs as exact keys
- Implementing modules outside TDMEC contracts
- Declaring success without checksums
- Mounting Google Drive or writing to `/content/drive` paths

## 8. Minimal first Studio task (recommended)

**CPU only:** verify local paths → install deps → run unit tests → open
`notebooks/phase2_real_data_smoke_and_validation.ipynb` and complete the
one-file smoke + sealed resume.  
Do **not** start full 12+70 or GPU embedding in the first session.
