# Target Lightning Studio Runbook — TDMEC Embedding Stage

**Status:** `TARGET_STUDIO_RUNBOOK_READY` · `IMPLEMENTED_NOT_EXECUTED` · `TRANSFER_PENDING_VALIDATION`

This runbook is for the **second** Lightning AI Studio that will execute the
embedding stage. Do **not** run these commands in the authoring Studio that
produced the transfer bundle under the no-execution restriction.

Commands below are examples only.

---

## 0. Labels

Every embedding component transferred here is:

```text
IMPLEMENTED_NOT_EXECUTED
TRANSFER_PENDING_VALIDATION
```

Preserve the last verified authoring baseline for older tests:

```text
165 passed
4 skipped
9 warnings
```

Newly added pooling / Qwen / pipeline / sampling tests are `NOT_EXECUTED` until
you run them here.

---

## 1. Create or select the new Studio

1. In Lightning AI, create a new Studio in the **same Teamspace** as
   `this_studio` if possible (recommended), or a new Teamspace Studio with
   access to the same persistent `/teamspace` storage.
2. Prefer a Studio with persistent Teamspace storage mounted at `/teamspace`.
3. Do not rely only on ephemeral Docker/VM disk for outputs or HF caches.

## 2. Recommended GPU type and VRAM

| Workload | Recommendation |
|---|---|
| Mock end-to-end | CPU is enough |
| Qwen3 64+64 preflight | NVIDIA GPU, **≥24 GB** VRAM preferred; 16 GB may work with batch size 1–2 and fp16 |
| Qwen3 10k+10k pilot | **≥24 GB** VRAM (L4/A10/A100-class); 32–64 GB system RAM |

Approximate model weight footprint for `Qwen/Qwen3-Embedding-4B` in fp16/bf16:
about **8 GB** (verify after download).

## 3. Obtain the repository code

Preferred (same GitHub remote):

```bash
cd /teamspace/studios/<your_studio>
git clone https://github.com/faezehmzf/community-evolution-modeling.git
cd community-evolution-modeling
git fetch origin
git checkout migrate/lightning-ai-studio   # or the branch containing the embedding work
```

If the embedding work is not yet pushed, use the transfer archive (step 4).

## 4. Apply the local embedding implementation bundle

```bash
# Example archive name; use the exact file produced in the authoring Studio.
ARCHIVE=/path/to/tdmec_embedding_code_transfer_20260804.tar.gz
cd /teamspace/studios/<your_studio>
mkdir -p community-evolution-modeling
cd community-evolution-modeling
tar -tzf "$ARCHIVE" | head
tar -xzf "$ARCHIVE"
# Verify transfer manifest checksums (see section 16)
```

## 5. Create the Python environment

```bash
cd /teamspace/studios/<your_studio>/community-evolution-modeling
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## 6. Install dependencies

```bash
# Base package + embedding extras + tests
pip install -e ".[embeddings,test]"
# Optional overlay of the constrained target-Studio file
pip install -r requirements/embeddings-target-studio.txt
```

Install a CUDA-matched PyTorch wheel from https://pytorch.org when the default
CPU wheel is insufficient.

Environment variables (no secrets in source files):

```bash
export HF_HOME=/teamspace/studios/<your_studio>/caches/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
mkdir -p "$HF_HOME"
# Prefer Studio secret store or: huggingface-cli login
# export HF_TOKEN=...   # do not write into the repository
```

## 7. Make smoke source artifacts available

If this Studio shares the same Teamspace persistent storage as the authoring
Studio, **reuse in place** (do not duplicate):

```text
Dataset A: /teamspace/studios/this_studio/TDMEC_PROJECT_OUTPUTS/graph/smoke_a_pg_001/
Dataset B: /teamspace/studios/this_studio/TDMEC_PROJECT_OUTPUTS/text_b/smoke_b_pg_001/
Node map:  /teamspace/studios/this_studio/TDMEC_PROJECT_OUTPUTS/manifests/node_index_map.parquet
```

```bash
export TDMEC_DATASET_A_ROOT=/teamspace/studios/this_studio/TDMEC_PROJECT_OUTPUTS/graph/smoke_a_pg_001
export TDMEC_DATASET_B_ROOT=/teamspace/studios/this_studio/TDMEC_PROJECT_OUTPUTS/text_b/smoke_b_pg_001
export TDMEC_EMBEDDING_OUTPUT_ROOT=/teamspace/studios/<your_studio>/TDMEC_PROJECT_OUTPUTS/embeddings
mkdir -p "$TDMEC_EMBEDDING_OUTPUT_ROOT"
```

If the paths are **not** visible, transfer Dataset A and Dataset B **separately**
from the code archive (encrypted private channel). Preserve directory structure
and `checksums.json`. Do **not** upload private text artifacts to public GitHub.

## 8. Verify source checksums

```bash
python - <<'PY'
from tdmec_embeddings.file_sources import load_file_source_identity
a = load_file_source_identity("$TDMEC_DATASET_A_ROOT", source_kind="dataset_a", expected_run_id="smoke_a_pg_001", verify_checksums=True)
b = load_file_source_identity("$TDMEC_DATASET_B_ROOT", source_kind="dataset_b", expected_run_id="smoke_b_pg_001", verify_checksums=True)
print(a.provenance())
print(b.provenance())
PY
```

Expand the environment variables in the Python snippet or pass resolved paths.

## 9. Run focused tests in the target Studio

```bash
source .venv/bin/activate
pytest -q tests/test_embeddings_file_sources.py tests/test_embeddings_eligibility.py tests/test_embeddings_file_writer.py
pytest -q tests/test_embeddings_pooling.py tests/test_embeddings_sampling.py tests/test_embeddings_qwen_encoder.py tests/test_embeddings_pipeline.py
# Optional broader suite
pytest -q
```

## 10. Run the mock end-to-end pipeline

```bash
tdmec-embedding-files \
  --config configs/embeddings/mock_end_to_end.yaml \
  --embedding-run-id mock_e2e_transfer_001 \
  --output-root "$TDMEC_EMBEDDING_OUTPUT_ROOT" \
  --node-source-root "$TDMEC_DATASET_B_ROOT" \
  --event-source-root "$TDMEC_DATASET_A_ROOT" \
  --max-node-rows 128 \
  --max-event-rows 128 \
  --output-dimension 16
```

Dry-run first:

```bash
tdmec-embedding-files --config configs/embeddings/mock_end_to_end.yaml --dry-run
```

## 11. Inspect mock outputs

```bash
RUN=$TDMEC_EMBEDDING_OUTPUT_ROOT/mock_e2e_transfer_001
ls -la "$RUN"
python - <<PY
import json, numpy as np
from pathlib import Path
run = Path("$RUN")
print(json.loads((run/"embedding_manifest.json").read_text())["status"])
print(np.load(run/"pooled"/"node_snapshot_embeddings.npy").shape)
print(np.load(run/"pooled"/"node_text_available_mask.npy").mean())
print(np.load(run/"pooled"/"canonical_edge_embeddings.npy").shape)
PY
```

## 12. Run the 64+64 real-Qwen preflight

Pin immutable revisions first (resolve from the Hugging Face model card; do not
use `main` / `latest`):

```bash
export QWEN3_MODEL_REVISION=<immutable-sha>
export QWEN3_TOKENIZER_REVISION=<immutable-sha>
tdmec-embedding-files \
  --config configs/embeddings/qwen3_preflight_64.yaml \
  --authorize-real-model
```

Limits above 64 per modality are refused unless the execution mode is the
bounded pilot.

## 13. Inspect preflight reports

```bash
RUN=$TDMEC_EMBEDDING_OUTPUT_ROOT/qwen3_preflight_64_001
ls "$RUN/reports"
python - <<PY
import json
from pathlib import Path
run = Path("$RUN")
for name in ["runtime_memory.json", "scale_estimates.json", "node_sampling.json", "event_sampling.json"]:
    print(name, json.loads((run/"reports"/name).read_text()).keys())
PY
```

Confirm reports contain **no raw text**.

## 14. Run the 10,000+10,000 pilot only after approval

```bash
export QWEN3_MODEL_REVISION=<immutable-sha>
export QWEN3_TOKENIZER_REVISION=<immutable-sha>
tdmec-embedding-files \
  --config configs/embeddings/qwen3_bounded_pilot_10k.yaml \
  --authorize-real-model \
  --authorize-bounded-pilot
```

This command never continues automatically to all eligible source texts.

## 15. Resume an interrupted run

```bash
tdmec-embedding-files \
  --config configs/embeddings/<same-config>.yaml \
  --resume \
  --authorize-real-model \
  --authorize-bounded-pilot   # only if resuming a pilot
```

Compatibility hashes must match. Do not use `--force`.

## 16. Verify manifests and checksums

```bash
RUN=$TDMEC_EMBEDDING_OUTPUT_ROOT/<embedding_run_id>
python - <<PY
import json
from pathlib import Path
from tdmec.hashing import sha256_file
run = Path("$RUN")
checksums = json.loads((run/"all_checksums.json").read_text())
bad = []
for rel, expected in checksums.items():
    path = run/rel
    if not path.is_file() or sha256_file(path) != expected:
        bad.append(rel)
print("ok" if not bad else bad[:20])
PY
```

## 17. Monitor GPU memory, system RAM, storage, and runtime

```bash
nvidia-smi -l 5
watch -n 5 free -h
df -h /teamspace
# Runtime aggregates are also written to reports/runtime_memory.json
```

## 18. Preserve outputs in persistent Lightning storage

Write under `/teamspace/...` (Teamspace persistent storage), not only under
`/tmp` or container-local disk.

Recommended layout:

```text
/teamspace/studios/<your_studio>/TDMEC_PROJECT_OUTPUTS/embeddings/<embedding_run_id>/
```

## 19. Avoid ephemeral Docker or VM-only storage

- Keep HF caches on `/teamspace`
- Keep embedding outputs on `/teamspace`
- Treat Studio local scratch as disposable

## 20. Create a portable backup of code, configs, manifests, and embedding outputs

```bash
DATE=$(date -u +%Y%m%d)
BACKUP=/teamspace/studios/<your_studio>/TDMEC_PROJECT_OUTPUTS/backups/tdmec_embedding_backup_${DATE}.tar.gz
tar -czf "$BACKUP" \
  community-evolution-modeling/src/tdmec_embeddings \
  community-evolution-modeling/configs/embeddings \
  community-evolution-modeling/requirements \
  community-evolution-modeling/docs/implementation/embedding_stage_*.md \
  community-evolution-modeling/docs/handoff/10_target_studio_embedding_runbook.md \
  "$TDMEC_EMBEDDING_OUTPUT_ROOT"
ls -lh "$BACKUP"
```

Keep Dataset A/B private text artifacts in a **separate** encrypted backup if
needed. Never place them in a public archive.
