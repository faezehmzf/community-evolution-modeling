# Kaggle Qwen embedding recovery runbook

Status: implementation ready; model-only and tiny real-model reruns intentionally not executed here.

This supersedes the monolithic execution cells in `11_kaggle_embedding_runbook.md`.
It preserves Qwen3-Embedding-4B revision
`5cf2132abc99cad020ac570b19d031efec650f2b`, D=512, last-token pooling,
MRL truncation, float32 unit re-normalization, canonical sampling/order,
Stage-B arithmetic means, masks, and graph/edge alignment.

## Evidence-based diagnosis

Confirmed:

- The failed notebook used `transformers 5.0.0` although the repository contract was
  `transformers>=4.51,<5`.
- It stopped inside weight materialization (`31–128/398` in the available captures),
  before any text batch, shard, pooling, alignment, checksum, or export operation.
- The notebook parent imported Torch/Transformers, but the model was instantiated only
  by one child process. There is no evidence of two model instances.
- The archived successful 64+64 Kaggle artifact used Transformers 4.55.4 and
  Torch 2.6.0+cu124. It encoded 128 texts in 7.339 s (17.44 texts/s), with zero OOM
  retries, 8,321,754,112 peak allocated GPU bytes, and 3,734,228,992 peak RSS bytes.

Likely, but not proven until Stage A runs on the current Kaggle image:

- Transformers 5's new dynamic weight-materialization path is the primary regression.
  Restoring 4.55.4 removes that untested major-version change. The loader now downloads
  or resolves a snapshot first, then materializes once in FP16 directly onto one explicit
  GPU with no auto placement and no CPU/disk offload.

Unverified hypotheses now measured by Stage A:

- Torch 2.10.0+cu128 interaction with Transformers 4.55.4; cold-cache/filesystem read
  behavior; CPU RAM or disk pressure; and stale/duplicated progress rendering.
- A progress line alone cannot distinguish a genuinely stuck tensor from buffered
  carriage-return output. JSON-line heartbeats now report RAM/VRAM/disk every minute.

## Exact dependency and loading contract

The non-Torch stack is pinned in `requirements/embeddings-target-studio.txt`:

```text
torch==2.10.0+cu128       # Kaggle preinstalled; verify only, never install here
transformers==4.55.4
accelerate==1.10.1
tokenizers==0.21.4
safetensors==0.5.3
huggingface_hub==0.34.4
numpy==2.2.6
pandas==2.2.3
pyarrow==19.0.1
PyYAML==6.0.2
psutil==6.1.1
pytest==8.4.1
```

Loading is FP16, `low_cpu_mem_usage=True`, `device_map={"": "cuda:0"}`,
`attn_implementation="sdpa"`, no quantization, no CPU/disk offload, no model
duplication, `eval()`, `use_cache=False`, and `torch.inference_mode()`. Each download
and materialization stage has a 900-second deadline. Parameters must all end on
`cuda:0`; meta, CPU, and disk placement fail the preflight.

## Cell 1 — clean repository and dependency setup

Set `EXPECTED_GIT_SHA` to the commit containing this implementation. Do not use a
moving branch tip for an embedding run.

```bash
set -euo pipefail
REPO=/kaggle/working/community-evolution-modeling
EXPECTED_GIT_SHA="${EXPECTED_GIT_SHA:?set the reviewed implementation commit SHA}"

git clone --branch feat/tdmec-g-smoke --single-branch \
  https://github.com/faezehmzf/community-evolution-modeling.git "$REPO"
git -C "$REPO" checkout --detach "$EXPECTED_GIT_SHA"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_GIT_SHA"

python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); assert torch.__version__ == "2.10.0+cu128"; assert torch.cuda.is_available()'
python -m pip install --disable-pip-version-check -r "$REPO/requirements/embeddings-target-studio.txt"
python -m pip install --disable-pip-version-check --no-deps -e "$REPO"
python -m pip check
python -c 'import torch; assert torch.__version__ == "2.10.0+cu128"; print(torch.__version__, torch.version.cuda)'
python -c 'import transformers,accelerate,tokenizers,safetensors,huggingface_hub; print(transformers.__version__,accelerate.__version__,tokenizers.__version__,safetensors.__version__,huggingface_hub.__version__)'
```

Run package installation only in subprocesses/cells that have not imported Transformers.
Do not delete modules from `sys.modules`, and do not reinstall Torch.

## Cell 2 — paths, secrets, and subprocess environment

```python
import json, os, subprocess, sys
from pathlib import Path
from kaggle_secrets import UserSecretsClient

REPO = Path("/kaggle/working/community-evolution-modeling")

def find_run(run_id: str) -> Path:
    for manifest in Path("/kaggle/input").rglob("manifest.json"):
        try:
            if json.loads(manifest.read_text()).get("run_id") == run_id:
                return manifest.parent
        except Exception:
            pass
    raise FileNotFoundError(run_id)

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
os.environ["TDMEC_DATASET_A_ROOT"] = str(find_run("smoke_a_pg_001"))
os.environ["TDMEC_DATASET_B_ROOT"] = str(find_run("smoke_b_pg_001"))
os.environ["TDMEC_EMBEDDING_OUTPUT_ROOT"] = "/kaggle/working/tdmec_embeddings"
os.environ["HF_HOME"] = "/kaggle/working/hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TDMEC_MODEL_LOAD_TIMEOUT_SECONDS"] = "900"
os.environ["PYTHONPATH"] = str(REPO / "src")
Path(os.environ["TDMEC_EMBEDDING_OUTPUT_ROOT"]).mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
```

Optional attached model input: point `TDMEC_MODEL_SNAPSHOT_PATH` at a complete,
read-only snapshot containing `config.json`, tokenizer files, safetensors files, and
`tdmec_model_revision.txt` whose sole line is the pinned revision. The loader verifies
the marker and does not contact the Hub. Creating or uploading that private Kaggle
dataset is a separate manual operation; this code performs no credential-dependent upload.

## Stage A — model-only preflight

```python
PREFLIGHT = "/kaggle/working/model_preflight.json"
subprocess.run([
    sys.executable, "-u", "-m", "tdmec_embeddings.model_preflight",
    "--config", "configs/qwen3_tdmec_pilot.yaml",
    "--output", PREFLIGHT,
    "--timeout-seconds", "900",
], cwd=REPO, env=env, check=True, timeout=960)
print(json.loads(Path(PREFLIGHT).read_text())["status"])
```

Success requires eight multilingual texts, `[8,512]`, finite unit vectors, FP16,
evaluation/inference mode, one `cuda:0` placement, no meta/offload parameters, exact
dependency versions, and completion under 15 minutes. Failure or timeout blocks all
later commands.

Expected outputs: `model_preflight.json` and `model_preflight.json.sha256.json`; no Dataset A/B files are opened.

## Stage B — tiny 64+64 smoke

Use a new run ID. Do not use `--replace-incomplete` for a resumable run.

```python
subprocess.run([
    sys.executable, "-u", "-m", "tdmec_embeddings.run_pilot_embedding",
    "--config", "configs/qwen3_tdmec_smoke_64.yaml",
    "--embedding-run-id", "qwen3_tdmec_smoke_64_recovery_001",
    "--preflight-report", PREFLIGHT,
    "--authorize-real-model",
], cwd=REPO, env=env, check=True)
print("TINY SMOKE DONE")
```

Success requires 64 node and 64 event unit rows, finite `[*,512]` unit shards,
validated SHA-256 sidecars, exact mask/count/zero invariants, canonical edge-order
fingerprint, passed graph-text alignment, and package checksums. Expected files include
`execution_manifest.json`, `embedding_manifest.json`, `all_checksums.json`,
`unit_embeddings/{node_text,event_text}/`, `pooled/*.npy`, pooling manifests,
alignment/validation reports, and the exported `TDMEC_INPUT_*` package.

Historical baseline (not a substitute for this new Stage A/B): the archived 64+64
run completed 128 embeddings in 7.339 seconds, with shapes `[35,16736,512]` and
`[794637,512]`, 62 available node-snapshot groups, 64 available edges, and no
truncation or OOM. It predates the new provenance/preflight gates.

## Stage C — bounded representative benchmark

```python
BENCHMARK = "/kaggle/working/qwen_benchmark_256x2.json"
subprocess.run([
    sys.executable, "-u", "-m", "tdmec_embeddings.run_embedding_job",
    "--config", "configs/qwen3_tdmec_pilot.yaml",
    "--preflight-report", PREFLIGHT,
    "--authorize-real-model", "--authorize-bounded-pilot",
    "benchmark", "--rows-per-modality", "256", "--output", BENCHMARK,
], cwd=REPO, env=env, check=True)
print(json.loads(Path(BENCHMARK).read_text())["performance"])
```

Choose batch size and `num_shards` from measured rows/s, tokens/s, VRAM, RAM, and
estimated pilot inference seconds. The benchmark writes only its report.

## Stage D — resumability and corruption test

The first command stops cleanly after one 16-row atomic output batch. The second
resumes the same deterministic range; the committed batch must not be encoded again.

```python
base = [
    sys.executable, "-u", "-m", "tdmec_embeddings.run_embedding_job",
    "--config", "configs/qwen3_tdmec_smoke_64.yaml",
    "--preflight-report", PREFLIGHT,
    "--embedding-run-id", "qwen3_resume_test_001",
    "--output-shard-size", "16",
    "--authorize-real-model",
]
subprocess.run(base + [
    "embed-shard", "--resume", "--modality", "node_text", "--shard-id", "0",
    "--num-shards", "1", "--wall-clock-hours", "1",
    "--graceful-stop-minutes", "10", "--max-output-batches", "1",
], cwd=REPO, env=env, check=True)
subprocess.run(base + [
    "embed-shard", "--resume", "--modality", "node_text", "--shard-id", "0",
    "--num-shards", "1", "--wall-clock-hours", "1",
    "--graceful-stop-minutes", "10",
], cwd=REPO, env=env, check=True)
```

For the isolated test run only, append bytes to one copied/committed Parquet shard,
then rerun the same command with `--repair-corrupt-shards`. The invalid shard and
sidecar must move under `quarantine/`; all other checksummed shards must remain byte
identical; only the missing batch is recomputed. The automated test suite performs
this exact corruption scenario without touching real artifacts.

Run the event modality similarly, then run `finalize`. Success requires exact canonical
batch coverage and the same graph-text alignment fingerprint as uninterrupted output.

## Resumable bounded pilot — one successful Kaggle version per command

Do not run until Stages A–D pass. With `num_shards=2`, publish four embedding versions:

```bash
python -u -m tdmec_embeddings.run_embedding_job \
  --config configs/qwen3_tdmec_pilot.yaml \
  --preflight-report /kaggle/input/tdmec-preflight/model_preflight.json \
  --authorize-real-model --authorize-bounded-pilot \
  embed-shard --resume --modality node_text --shard-id 0 --num-shards 2 \
  --wall-clock-hours 9 --graceful-stop-minutes 20
```

Repeat with `(node_text,1)`, `(event_text,0)`, and `(event_text,1)`. Before each fresh
version, attach the preceding successful notebook output and restore only the run state:

```bash
mkdir -p /kaggle/working/tdmec_embeddings
cp -a /kaggle/input/PREVIOUS_PRIVATE_OUTPUT/tdmec_embeddings/. \
  /kaggle/working/tdmec_embeddings/
```

Never run two writers concurrently against the same run directory. A killed Kaggle
version cannot publish `/kaggle/working`; therefore each submitted version must finish
comfortably and become the next version's private input. External mid-run persistence
would require Kaggle API credentials or an object store and is intentionally not added.

After all four jobs, run a fifth short version:

```bash
python -u -m tdmec_embeddings.run_embedding_job \
  --config configs/qwen3_tdmec_pilot.yaml \
  --preflight-report /kaggle/input/tdmec-preflight/model_preflight.json \
  --authorize-real-model --authorize-bounded-pilot \
  finalize --package-root /kaggle/working/TDMEC_INPUT_qwen3_tdmec_pilot_10k_001
```

Finalization verifies every canonical batch/checksum before sealing modalities. It
refuses gaps and does not load the model. It then pools, validates masks/counts/finite
values, checks canonical graph/edge order, exports, and writes final checksums.

## Runtime and version estimate

The only measured baseline is 17.44 texts/s on the archived Kaggle GPU run. At that
rate, 5,000 texts per logical shard take about 287 seconds of inference; all 20,000
take about 19.1 inference minutes. Until Stage C measures the P100/current stack,
budget each 5,000-row version at **under 30 minutes if model loading passes within
15 minutes**, plus source-selection/I/O variance. The bounded pilot requires four
embedding versions plus one finalization version. Preflight, smoke, benchmark, and
resumability are separate gates and are not counted in those five pilot versions.

`--replace-incomplete` retains its old, intentionally destructive meaning: it removes
the entire non-COMPLETED run directory after classification. It never removes a
COMPLETED run. Do not use it for normal resume. Use `--resume`/staged jobs to preserve
validated shards, and `--repair-corrupt-shards` only to quarantine checksum-invalid
incomplete shards and recompute them.
