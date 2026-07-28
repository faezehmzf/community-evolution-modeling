# 09 — Colab Execution Runbook

Future code executes here without relying on local workstation data processing.  
**Default persistent root:** `/content/drive/MyDrive/TDMEC_PROJECT_OUTPUTS/`

## 1. Session bootstrap

1. Open a new Colab notebook (CPU or GPU as required by the task).  
2. Mount Google Drive.  
3. Clone or `git pull` the repository; checkout the **approved commit SHA**.  
4. Record SHA: `git rev-parse HEAD` → write into `runs/<run_id>/manifests/git_sha.txt`.  
5. Set environment variables:

```text
TDMEC_REPO=/content/community-evolution-modeling
TDMEC_OUT=/content/drive/MyDrive/TDMEC_PROJECT_OUTPUTS
TDMEC_A=<Drive path to Dataset A>
TDMEC_B=<Drive path to Dataset B>
TDMEC_RUN_ID=<yyyyymmdd_HHMMSS_shortsha>
```

6. Check free Drive storage before large writes.  
7. Install **pinned** dependencies from the repo lock/requirements for that SHA.  
8. Do **not** process data from the local Windows Proposed directory.

## 2. Preflight checks

- Repo tests: `pytest -q` (or project’s documented subset).  
- Config hash: hash frozen YAML → `config_hash.txt`.  
- Source path existence and read probe (byte-level) for A/B.  
- Confirm node-map policy: N must be 16736 when built.  
- Refuse to continue if O-EMB still open and the task is embedding/training.

## 3. Standard stage sequence

| Step | Command pattern | Machine |
|---|---|---|
| Build node map | `python scripts/build_node_index_map.py --config ...` | CPU |
| Validate map | assert N==16736 + checksum publish | CPU |
| Dataset A graph | `python -m tdmec build-graph ...` (future) | CPU |
| Dataset B normalize | full 70-file job with resume | CPU |
| Controlled pilots | 1–2 shards first | CPU |
| Approve full runs | human gate | — |
| Embeddings | GPU notebook / script | GPU |
| Pack tensors | CPU/GPU | CPU |
| Graph-only smoke | GPU | GPU |
| Text-aware smoke | GPU | GPU |
| Train staged | GPU | GPU |
| Eval | GPU/CPU | either |

Use `--run-mode resume` for long jobs; `--fresh` only when intentional.

## 4. Resume interrupted runs

1. Read checkpoint/manifest under `runs/<run_id>/`.  
2. Re-run same command with same config hash and SHA.  
3. Do not mix artifacts across different SHAs/configs without a new `run_id`.

## 5. Publishing artifacts

For each output:
- Write under `TDMEC_OUT/runs/<run_id>/...`  
- Compute SHA256  
- Append to `manifests/artifacts.jsonl` with row counts and exclusion reasons  
- Only mark `CERTIFIED` after validation gates pass

## 6. Logs

Preserve stdout/stderr, `train_log.json`, env probe (GPU name, CUDA, package versions), timestamps UTC.

## 7. Forbidden Colab actions

- Training before certified tensors  
- Embedding before O-EMB confirmation  
- Expanding node universe from Dataset B  
- Using float Dataset A tweet IDs as exact keys  
- Implementing modules outside TDMEC contracts  
- Declaring success without checksums

## 8. Minimal first Colab task (recommended)

**CPU only:** mount Drive → checkout SHA → install deps → run unit tests → dry-run node-map builder on Dataset A metadata / small probe → write manifests.  
Do **not** start full 70-file or GPU embedding in the first session.
