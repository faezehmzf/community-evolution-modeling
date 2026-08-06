# Extract and install the embedding code transfer bundle

**Status:** `TRANSFER_BUNDLE_READY` · `IMPLEMENTED_NOT_EXECUTED` · `TRANSFER_PENDING_VALIDATION`

## Extract

```bash
ARCHIVE=tdmec_embedding_code_transfer_20260804.tar.gz
mkdir -p /teamspace/studios/<your_studio>/community-evolution-modeling
cd /teamspace/studios/<your_studio>/community-evolution-modeling
tar -xzf /path/to/$ARCHIVE
```

Repository-relative paths are preserved (for example `src/tdmec_embeddings/...`).

## Verify the transfer manifest

```bash
python - <<'PY'
import hashlib, json
from pathlib import Path
manifest = json.loads(Path("transfer/TRANSFER_MANIFEST.json").read_text())
bad = []
for entry in manifest["files"]:
    path = Path(entry["repository_relative_path"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry["sha256"] or path.stat().st_size != entry["size_bytes"]:
        bad.append(entry["repository_relative_path"])
print("TRANSFER_OK" if not bad else bad)
print("file_count", len(manifest["files"]))
PY
```

## Install

If this archive was applied onto a full clone, use the repository `pyproject.toml`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -e ".[embeddings,test]"
pip install -r requirements/embeddings-target-studio.txt
```

If only the transfer bundle contents are present:

```bash
cp transfer/pyproject.embeddings-only.toml pyproject.toml
pip install -e ".[embeddings,test]"
pip install -r requirements/embeddings-target-studio.txt
```

## What is excluded

- `.git`
- `.env` / secrets / HF tokens
- Dataset A/B private artifacts
- Generated embedding outputs
- Model caches / weights
- `__pycache__` / `.pytest_cache`
- Unrelated uncommitted project outputs

Private Dataset A/B artifacts must be mounted or transferred separately.
