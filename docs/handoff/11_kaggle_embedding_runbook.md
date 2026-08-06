# Kaggle GPU Runbook — TDMEC Embedding Stage (Preflight + Bounded Pilot)

**Status:** `TARGET_KAGGLE_RUNBOOK_READY` · provisional engineering path  
**Compute:** Kaggle Notebook GPU (free accelerator), **not** a personal laptop, **not** a paid external embedding API  
**Model:** `Qwen/Qwen3-Embedding-4B` weights downloaded into the Kaggle session / HF cache and inferred on the Kaggle GPU

This runbook migrates the transferable embedding package from Lightning Teamspace storage to Kaggle, then runs:

1. Qwen Preflight (**64 + 64**)
2. Bounded Pilot (**10,000 + 10,000**) only after preflight review

---

## 0. Critical Kaggle constraints (read first)

### Disk math at provisional native `D_text=2560`

Even a **64-unit** preflight still materializes **full** pooled tensors aligned to the smoke graph:

| Tensor | Shape | Approx size |
|---|---|---|
| `node_snapshot_embeddings.npy` | `[35, 16736, 2560]` | **~6.0 GB** |
| `canonical_edge_embeddings.npy` | `[794637, 2560]` | **~8.1 GB** |
| Qwen3-4B weights (fp16/bf16) | model cache | **~8 GB** |
| Sources A+B | input datasets | **~0.66 GB** |

Kaggle `/kaggle/working` is typically ~20 GB. Native-2560 pooling + model + working files can **exhaust disk**.

**Recommended Kaggle engineering strategy:**

1. Run **preflight first** with a **provisional reduced dimension** via MRL truncation (officially supported range 32–2560 on the model card), e.g. `512` or `1024`, using `enable_provisional_mrl_truncation: true`.
2. Treat reduced-dim outputs as **engineering validation**, not final thesis `D_text`.
3. Only attempt native `2560` if you have confirmed free disk ≥ ~25–30 GB after install (often not available on free Kaggle).

| `D_text` | Approx pooled vectors only |
|---|---|
| 2560 | ~14.1 GB |
| 1024 | ~5.7 GB |
| 512 | ~2.8 GB |
| 256 | ~1.4 GB |

T4 GPUs (common on Kaggle) lack bf16; the encoder `precision: auto` path falls back to **fp16**.

### Privacy

`smoke_a_pg_001` and `smoke_b_pg_001` contain **private / sensitive text**. Upload them only as **Private** Kaggle Datasets. Do not make them public. Do not commit them to public GitHub.

---

## 1. Data & code transfer to Kaggle

### 1.1 Rebuild the transfer package on Lightning (includes latest patches)

On the Lightning Studio (CPU is fine):

```bash
cd /teamspace/studios/this_studio/community-evolution-modeling
TRANSFER_DATE=20260804 bash scripts/package_embedding_transfer.sh
ls -lh transfer/artifacts/tdmec_embedding_code_transfer_20260804.tar.gz
cat transfer/artifacts/ARCHIVE_SHA256.json
```

Download that archive to your machine (or any staging location you use to reach Kaggle).

### 1.2 Prepare three **Private** Kaggle Datasets

Create datasets in the Kaggle UI (**Private**):

#### Dataset A — code package

Suggested slug: `tdmec-embedding-code-20260804`

Upload:

```text
tdmec_embedding_code_transfer_20260804.tar.gz
```

Optional: also upload `ARCHIVE_SHA256.json`.

#### Dataset B — Dataset A smoke artifacts

Suggested slug: `tdmec-smoke-a-pg-001`

Upload the **entire directory contents** of:

```text
/teamspace/studios/this_studio/TDMEC_PROJECT_OUTPUTS/graph/smoke_a_pg_001/
```

Preserve relative structure so these exist under the dataset root:

```text
manifest.json
checksums.json
validation_report.json
snapshot_calendar.json
events/canonical_events.parquet
edges/snapshot=*/relation=*/part-*.parquet
... (all checksummed files)
```

Easiest reliable method: zip the folder while preserving paths, upload the zip, and extract once in the notebook into `/kaggle/working/sources/` **or** upload an already-unzipped folder tree if the Kaggle dataset UI accepts it.

Approx size: **~194 MB**.

#### Dataset C — Dataset B smoke artifacts

Suggested slug: `tdmec-smoke-b-pg-001`

Upload the entire contents of:

```text
/teamspace/studios/this_studio/TDMEC_PROJECT_OUTPUTS/text_b/smoke_b_pg_001/
```

Must include:

```text
manifest.json
checksums.json
validation_report.json
normalized_records/**/*.parquet
duplicate_records.parquet
... (all checksummed files)
```

Approx size: **~466 MB**.

### 1.3 Optional packaging commands on Lightning before upload

```bash
# Code archive already built above.

# Dataset A zip (private transfer only)
cd /teamspace/studios/this_studio/TDMEC_PROJECT_OUTPUTS/graph
zip -r /tmp/tdmec_smoke_a_pg_001.zip smoke_a_pg_001

# Dataset B zip (private transfer only)
cd /teamspace/studios/this_studio/TDMEC_PROJECT_OUTPUTS/text_b
zip -r /tmp/tdmec_smoke_b_pg_001.zip smoke_b_pg_001

ls -lh /tmp/tdmec_smoke_a_pg_001.zip /tmp/tdmec_smoke_b_pg_001.zip
```

Upload those zips into the corresponding private Kaggle datasets if you prefer zip-based transfer.

### 1.4 Create the Kaggle Notebook

1. New Notebook → **GPU** accelerator (T4/P100).
2. Add the three private datasets as inputs.
3. Internet **on** (needed for first model download unless you pre-vendor weights into a dataset).
4. Persistence: save outputs often; `/kaggle/working` is session-scoped unless you write a Kaggle Dataset version.

---

## 2. Path mappings & environment variables

Kaggle mounts datasets under `/kaggle/input/<dataset-slug>/`.  
Writable space is `/kaggle/working/`.

Adjust slug names to match what you created.

```bash
# --- inputs (read-only) ---
export TDMEC_CODE_ARCHIVE=/kaggle/input/tdmec-embedding-code-20260804/tdmec_embedding_code_transfer_20260804.tar.gz
export TDMEC_DATASET_A_ROOT=/kaggle/working/sources/smoke_a_pg_001
export TDMEC_DATASET_B_ROOT=/kaggle/working/sources/smoke_b_pg_001

# If you uploaded already-extracted trees instead of zips, point directly at input paths, e.g.:
# export TDMEC_DATASET_A_ROOT=/kaggle/input/tdmec-smoke-a-pg-001/smoke_a_pg_001
# export TDMEC_DATASET_B_ROOT=/kaggle/input/tdmec-smoke-b-pg-001/smoke_b_pg_001

# --- outputs / caches (writable) ---
export TDMEC_EMBEDDING_OUTPUT_ROOT=/kaggle/working/embeddings
export HF_HOME=/kaggle/working/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export TOKENIZERS_PARALLELISM=false

mkdir -p "$TDMEC_EMBEDDING_OUTPUT_ROOT" "$HF_HOME" /kaggle/working/repo /kaggle/working/sources
```

### Immutable Qwen revision pins

Do **not** use `main` / `latest`. Resolve a commit SHA from:

https://huggingface.co/Qwen/Qwen3-Embedding-4B/commits/main

Then:

```bash
export QWEN3_MODEL_REVISION=<immutable-40-char-or-full-commit-sha>
export QWEN3_TOKENIZER_REVISION=$QWEN3_MODEL_REVISION
```

Optional HF auth (only if the model gate / rate limits require it):

```bash
# Prefer Kaggle Secrets: Add-ons → Secrets → HF_TOKEN
from kaggle_secrets import UserSecretsClient
import os
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

Do not hardcode tokens in the notebook if you will share it.

---

## 3. Kaggle notebook setup cells

### Cell 1 — accelerator check

```python
!nvidia-smi
import torch
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
```

### Cell 2 — extract code + sources

```bash
set -euo pipefail

# Adjust dataset slug/file names to match your Kaggle Datasets
CODE_DS=/kaggle/input/tdmec-embedding-code-20260804
A_DS=/kaggle/input/tdmec-smoke-a-pg-001
B_DS=/kaggle/input/tdmec-smoke-b-pg-001

mkdir -p /kaggle/working/repo /kaggle/working/sources

# Code
ARCHIVE=$(ls "$CODE_DS"/*.tar.gz | head -1)
tar -xzf "$ARCHIVE" -C /kaggle/working/repo
ls /kaggle/working/repo/src/tdmec_embeddings | head

# Sources: support either extracted tree or zip upload
prepare_source () {
  local src="$1"
  local dst="$2"
  mkdir -p "$dst"
  if [ -f "$src"/manifest.json ]; then
    # dataset root already is the artifact root
    cp -a "$src"/. "$dst"/
  elif [ -d "$src"/smoke_* ]; then
    cp -a "$src"/smoke_*/* "$dst"/ 2>/dev/null || cp -a "$src"/*/ "$dst"/
  elif ls "$src"/*.zip >/dev/null 2>&1; then
    unzip -q "$(ls "$src"/*.zip | head -1)" -d /kaggle/working/sources_unpack
    # find manifest.json and copy its parent
    MANIFEST=$(find /kaggle/working/sources_unpack -name manifest.json | head -1)
    cp -a "$(dirname "$MANIFEST")"/. "$dst"/
  else
    echo "Cannot locate artifact root under $src" >&2
    ls -la "$src" >&2
    exit 1
  fi
  test -f "$dst/manifest.json"
  test -f "$dst/checksums.json"
}

prepare_source "$A_DS" /kaggle/working/sources/smoke_a_pg_001
prepare_source "$B_DS" /kaggle/working/sources/smoke_b_pg_001

export TDMEC_DATASET_A_ROOT=/kaggle/working/sources/smoke_a_pg_001
export TDMEC_DATASET_B_ROOT=/kaggle/working/sources/smoke_b_pg_001
echo "A rows check:"; ls "$TDMEC_DATASET_A_ROOT" | head
echo "B rows check:"; ls "$TDMEC_DATASET_B_ROOT" | head
df -h /kaggle/working
```

### Cell 3 — Python env + dependencies

```bash
set -euo pipefail
cd /kaggle/working/repo

python -m pip install -U pip setuptools wheel

# Prefer CUDA torch already present in Kaggle; then install project deps
python - <<'PY'
import torch
print("preinstalled torch", torch.__version__, "cuda", torch.cuda.is_available())
PY

# Install package + embedding extras without forcing a CPU torch overwrite if possible
pip install -e ".[test]"
pip install -r requirements/embeddings-target-studio.txt

# If transformers/torch conflict, install embedding stack explicitly after checking versions:
# pip install "transformers>=4.51,<5" "accelerate>=0.33,<2" "tokenizers>=0.19,<0.22" \
#   "safetensors>=0.4,<0.6" "huggingface_hub>=0.24,<0.30" "psutil>=5.9,<7"

python - <<'PY'
import torch, transformers
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

### Cell 4 — export env + pin revisions

```bash
export TDMEC_DATASET_A_ROOT=/kaggle/working/sources/smoke_a_pg_001
export TDMEC_DATASET_B_ROOT=/kaggle/working/sources/smoke_b_pg_001
export TDMEC_EMBEDDING_OUTPUT_ROOT=/kaggle/working/embeddings
export HF_HOME=/kaggle/working/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export TOKENIZERS_PARALLELISM=false

# REPLACE with the immutable SHA you verified on Hugging Face
export QWEN3_MODEL_REVISION=REPLACE_WITH_IMMUTABLE_SHA
export QWEN3_TOKENIZER_REVISION=$QWEN3_MODEL_REVISION

mkdir -p "$TDMEC_EMBEDDING_OUTPUT_ROOT" "$HF_HOME"
cd /kaggle/working/repo
export PYTHONPATH=/kaggle/working/repo/src:${PYTHONPATH:-}

# quick source identity/checksum verification
python - <<'PY'
import os
from tdmec_embeddings.file_sources import load_file_source_identity
a = load_file_source_identity(os.environ["TDMEC_DATASET_A_ROOT"], source_kind="dataset_a", expected_run_id="smoke_a_pg_001", verify_checksums=True)
b = load_file_source_identity(os.environ["TDMEC_DATASET_B_ROOT"], source_kind="dataset_b", expected_run_id="smoke_b_pg_001", verify_checksums=True)
print("A", a.provenance()["run_id"], a.n_nodes, a.n_snapshots)
print("B", b.provenance()["run_id"], b.n_nodes, b.n_snapshots)
PY
```

### Cell 5 — write a Kaggle-safe preflight config (reduced D recommended)

Native 2560 often will not fit. Create a Kaggle overlay config:

```bash
cat > /kaggle/working/repo/configs/embeddings/qwen3_preflight_64_kaggle.yaml <<'EOF'
schema_version: tdmec-embedding-run-config-v1
execution_mode: qwen_preflight
embedding_run_id: qwen3_preflight_64_kaggle_001
output_root: ${TDMEC_EMBEDDING_OUTPUT_ROOT}
force: false
resume: false
dry_run: false
input_batch_size: 64
output_shard_size: 64
max_node_rows: 64
max_event_rows: 64
node_source:
  artifact_root: ${TDMEC_DATASET_B_ROOT}
  run_id: smoke_b_pg_001
event_source:
  artifact_root: ${TDMEC_DATASET_A_ROOT}
  run_id: smoke_a_pg_001
encoder:
  backend: qwen3
  model_name: Qwen/Qwen3-Embedding-4B
  model_revision: ${QWEN3_MODEL_REVISION}
  tokenizer_revision: ${QWEN3_TOKENIZER_REVISION}
  instruction: >-
    Represent the topic, stance, sentiment, and social meaning of this
    social-media post for temporal community analysis.
  # PROVISIONAL reduced dim for Kaggle disk/VRAM engineering validation.
  # Native 2560 remains the scientific hypothesis pending later confirmation.
  output_dimension: 512
  max_length: 512
  precision: auto
  device: cuda:0
  batch_size: 4
  max_oom_retries: 4
  local_files_only: false
  allow_cpu: false
  attn_implementation: null
  enable_provisional_mrl_truncation: true
sampling:
  strategy: deterministic_stratified_hash
  seed: 20260804
  node_hash_buckets: 64
  short_text_max_chars: 64
  medium_text_max_chars: 256
pooling:
  final_normalization: none
  delta_batch_rows: 4096
EOF
```

For a 10k pilot overlay later, copy and set:

- `execution_mode: qwen_bounded_pilot`
- `embedding_run_id: qwen3_pilot_10k_kaggle_001`
- `max_node_rows: 10000`
- `max_event_rows: 10000`
- `batch_size: 2` or `4` on T4
- keep `enable_provisional_mrl_truncation: true` and the same provisional `output_dimension` unless disk allows more

---

## 4. CLI execution on Kaggle

### 4.1 Dry-run

```bash
cd /kaggle/working/repo
export PYTHONPATH=/kaggle/working/repo/src:${PYTHONPATH:-}

python -m tdmec_embeddings.file_cli \
  --config configs/embeddings/qwen3_preflight_64_kaggle.yaml \
  --dry-run
```

### 4.2 Qwen Preflight (64+64)

```bash
python -m tdmec_embeddings.file_cli \
  --config configs/embeddings/qwen3_preflight_64_kaggle.yaml \
  --authorize-real-model
```

Equivalent if entry point is installed:

```bash
tdmec-embedding-files \
  --config configs/embeddings/qwen3_preflight_64_kaggle.yaml \
  --authorize-real-model
```

Expected run directory:

```text
/kaggle/working/embeddings/qwen3_preflight_64_kaggle_001/
```

Inspect:

```bash
RUN=/kaggle/working/embeddings/qwen3_preflight_64_kaggle_001
ls -lah "$RUN"
ls -lah "$RUN/pooled"
python - <<PY
import json, numpy as np
from pathlib import Path
run=Path("$RUN")
print(json.loads((run/"embedding_manifest.json").read_text())["status"])
print("node", np.load(run/"pooled"/"node_snapshot_embeddings.npy", mmap_mode="r").shape)
print("edge", np.load(run/"pooled"/"canonical_edge_embeddings.npy", mmap_mode="r").shape)
print((run/"reports"/"runtime_memory.json").read_text()[:500])
PY
df -h /kaggle/working
```

### 4.3 Bounded Pilot (10k+10k) — only after preflight success

```bash
python -m tdmec_embeddings.file_cli \
  --config configs/embeddings/qwen3_pilot_10k_kaggle.yaml \
  --authorize-real-model \
  --authorize-bounded-pilot
```

### 4.4 Resume after interruption

```bash
python -m tdmec_embeddings.file_cli \
  --config configs/embeddings/qwen3_preflight_64_kaggle.yaml \
  --resume \
  --authorize-real-model
```

Do **not** use `--force`.

---

## 5. Retrieve outputs from Kaggle

### 5.1 Package a portable results archive

```bash
RUN=qwen3_preflight_64_kaggle_001   # or pilot run id
OUT=/kaggle/working/embeddings/$RUN
ARCHIVE=/kaggle/working/tdmec_${RUN}_outputs.tar.gz

# Prefer excluding giant HF caches
tar -czf "$ARCHIVE" \
  -C /kaggle/working/embeddings \
  "$RUN"

ls -lh "$ARCHIVE"
python - <<PY
import hashlib, json
from pathlib import Path
p=Path("$ARCHIVE")
print({"path": str(p), "bytes": p.stat().st_size,
       "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
PY
```

Minimum files to keep if you need a smaller download:

```text
embedding_manifest.json
all_checksums.json
pooled/
reports/
manifests/
checksums/
pooling/*_manifest.json
pooling/*_checkpoint.json
```

Unit shards under `unit_embeddings/` can be large for the 10k pilot; include them if you need resume/re-pool capability.

### 5.2 Download options

1. **Notebook output download:** right-click `/kaggle/working/tdmec_*_outputs.tar.gz` → Download.  
2. **Save to a Private Kaggle Dataset version** from the notebook (best for ≥1–2 GB artifacts).  
3. Copy later into Lightning Teamspace persistent storage, e.g.:

```text
/teamspace/studios/this_studio/TDMEC_PROJECT_OUTPUTS/embeddings/from_kaggle/
```

### 5.3 Do not download

- `$HF_HOME` model caches (re-downloadable)
- Temporary unzip directories
- Any notebook cell outputs that print raw text (your CLI/reports should already be privacy-safe)

---

## 6. Suggested end-to-end order on Kaggle

1. Rebuild Lightning transfer archive.  
2. Upload private code + A + B datasets.  
3. GPU notebook → extract → install → pin SHAs.  
4. Verify checksums.  
5. Preflight 64+64 with provisional `D_text=512` (or 1024).  
6. Inspect `reports/runtime_memory.json`, `scale_estimates.json`, masks/shapes.  
7. Only then run 10k+10k with the same provisional dim.  
8. Package outputs → download / save private dataset → optionally restore into Lightning Teamspace.  
9. Keep native-2560 / final Q-EMB decisions for a later higher-disk environment.

---

## 7. Authorization flags reminder

| Run | Required CLI flags |
|---|---|
| Preflight | `--authorize-real-model` |
| Bounded pilot | `--authorize-real-model` **and** `--authorize-bounded-pilot` |

Chat authorization example for your records:

```text
AUTHORIZE KAGGLE GPU EXPERIMENTAL RUNS:
- Not on my personal computer
- No external paid embedding API
- Weights + inference on Kaggle free GPU only
- Preflight 64+64 with --authorize-real-model
- Pilot 10k+10k only after preflight, with both authorization flags
- Provisional reduced D_text allowed for Kaggle disk limits
- Private datasets only for smoke_a_pg_001 / smoke_b_pg_001
```

---

## 8. What success looks like

Preflight run directory contains:

```text
embedding_manifest.json          status=COMPLETED
all_checksums.json
pooled/node_snapshot_embeddings.npy
pooled/node_text_available_mask.npy
pooled/node_valid_text_count.npy
pooled/canonical_edge_embeddings.npy
pooled/edge_text_available_mask.npy
pooled/edge_valid_event_count.npy
reports/runtime_memory.json
reports/scale_estimates.json
```

These are the **text-side** TDMEC inputs (plus masks). Structural graph tensors (`X_struct`, edges/weights, `struct_active_mask`) still come from Dataset A graph artifacts, not from this embedding stage alone.

Label Kaggle reduced-dim outputs:

```text
ENGINEERING_VALIDATION
PROVISIONAL_D_TEXT
NOT_FOR_FINAL_THESIS_CONCLUSIONS
```
