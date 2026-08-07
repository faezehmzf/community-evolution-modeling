#!/usr/bin/env bash
# Build a code-only embedding transfer archive.
# Packaging only — does not run embedding pipelines, tests, or model downloads.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATE="${TRANSFER_DATE:-$(date -u +%Y%m%d)}"
OUT_DIR="${TRANSFER_OUT_DIR:-$ROOT/transfer/artifacts}"
ARCHIVE_NAME="tdmec_embedding_code_transfer_${DATE}.tar.gz"
ARCHIVE_PATH="$OUT_DIR/$ARCHIVE_NAME"
MANIFEST="$ROOT/transfer/TRANSFER_MANIFEST.json"
INVENTORY="$ROOT/transfer/TRANSFER_INVENTORY.md"

mkdir -p "$OUT_DIR"

mapfile -t FILES < <(
  cd "$ROOT"
  {
    find src/tdmec_embeddings -type f -name '*.py'
    find src/tdmec -type f \( -name '__init__.py' -o -name 'constants.py' -o -name 'hashing.py' -o -name 'unresolved.py' \)
    find tests -type f -name 'test_embeddings_*.py'
    find configs/embeddings -type f -name '*.yaml'
    find requirements -type f -name 'embeddings-target-studio.txt'
    find docs/implementation -type f -name 'embedding_stage_implementation_notes.md'
    find docs/handoff -type f -name '10_target_studio_embedding_runbook.md'
    find docs/method -type f -name '16_q_emb_embedding_contract_and_pilot_spec.md'
    printf '%s\n' \
      pyproject.toml \
      transfer/EXTRACT_AND_INSTALL.md \
      transfer/pyproject.embeddings-only.toml \
      transfer/TRANSFER_INVENTORY.md
  } | sed '/^$/d' | sort -u
)

python3 - <<'PY' "$ROOT" "$MANIFEST" "$INVENTORY" "${FILES[@]}"
import hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
inventory_path = Path(sys.argv[3])
rel_paths = sys.argv[4:]

files = []
for rel in rel_paths:
    path = root / rel
    if not path.is_file():
        raise SystemExit(f"missing transfer file: {rel}")
    data = path.read_bytes()
    files.append(
        {
            "repository_relative_path": rel,
            "absolute_path": str(path.resolve()),
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
            "required": True,
        }
    )

manifest = {
    "schema_version": "tdmec-embedding-code-transfer-manifest-v1",
    "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "status_labels": [
        "IMPLEMENTED_NOT_EXECUTED",
        "TRANSFER_PENDING_VALIDATION",
        "TRANSFER_BUNDLE_READY",
    ],
    "excludes": [
        ".git",
        ".env",
        "secrets",
        "caches",
        "source datasets",
        "preprocessing parquet artifacts",
        "generated embedding outputs",
        "model caches",
        "model weights",
        "__pycache__",
        ".pytest_cache",
    ],
    "file_count": len(files),
    "files": files,
}
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

lines = [
    "# Embedding Code Transfer Inventory",
    "",
    "**Status:** `TRANSFER_BUNDLE_READY` / `IMPLEMENTED_NOT_EXECUTED` / `TRANSFER_PENDING_VALIDATION`",
    "",
    f"Generated (UTC): `{manifest['created_at_utc']}`",
    "",
    "## A. Required code files",
    "",
    "| Repository-relative path | Size (bytes) | SHA-256 | Required | Purpose |",
    "|---|---:|---|---|---|",
]
purpose = {
    "src/tdmec_embeddings/": "Embedding stage implementation",
    "src/tdmec/": "Shared hashing/constants contracts",
    "tests/": "Focused embedding tests (NOT_EXECUTED in authoring Studio)",
    "configs/embeddings/": "Mock / preflight / pilot configuration templates",
    "requirements/": "Target-Studio dependency specification",
    "docs/": "Runbook, implementation notes, Q-EMB contract",
    "pyproject.toml": "Package metadata and optional embedding extras",
    "transfer/": "Transfer packaging docs and embed-only pyproject",
}
for entry in files:
    rel = entry["repository_relative_path"]
    why = "Embedding transfer artifact"
    for prefix, label in purpose.items():
        if rel.startswith(prefix) or rel == prefix.rstrip("/"):
            why = label
            break
    lines.append(
        f"| `{rel}` | {entry['size_bytes']} | `{entry['sha256']}` | required | {why} |"
    )
inventory_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {manifest_path}")
print(f"wrote {inventory_path}")
print(f"files {len(files)}")
PY

# Rebuild manifest list including newly written inventory/manifest for the archive
mapfile -t FILES < <(
  cd "$ROOT"
  {
    find src/tdmec_embeddings -type f -name '*.py'
    find src/tdmec -type f \( -name '__init__.py' -o -name 'constants.py' -o -name 'hashing.py' -o -name 'unresolved.py' \)
    find tests -type f -name 'test_embeddings_*.py'
    find configs/embeddings -type f -name '*.yaml'
    find requirements -type f -name 'embeddings-target-studio.txt'
    find docs/implementation -type f -name 'embedding_stage_implementation_notes.md'
    find docs/handoff -type f -name '10_target_studio_embedding_runbook.md'
    find docs/method -type f -name '16_q_emb_embedding_contract_and_pilot_spec.md'
    printf '%s\n' \
      pyproject.toml \
      transfer/EXTRACT_AND_INSTALL.md \
      transfer/pyproject.embeddings-only.toml \
      transfer/TRANSFER_INVENTORY.md \
      transfer/TRANSFER_MANIFEST.json
  } | sed '/^$/d' | sort -u
)

# Refresh manifest hashes to include inventory/manifest themselves
python3 - <<'PY' "$ROOT" "$MANIFEST" "$INVENTORY" "${FILES[@]}"
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
root = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
inventory_path = Path(sys.argv[3])
rel_paths = sys.argv[4:]
files = []
for rel in rel_paths:
    path = root / rel
    data = path.read_bytes()
    files.append({
        "repository_relative_path": rel,
        "absolute_path": str(path.resolve()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "required": True,
    })
manifest = {
    "schema_version": "tdmec-embedding-code-transfer-manifest-v1",
    "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "status_labels": [
        "IMPLEMENTED_NOT_EXECUTED",
        "TRANSFER_PENDING_VALIDATION",
        "TRANSFER_BUNDLE_READY",
    ],
    "excludes": [
        ".git", ".env", "secrets", "caches", "source datasets",
        "preprocessing parquet artifacts", "generated embedding outputs",
        "model caches", "model weights", "__pycache__", ".pytest_cache",
    ],
    "file_count": len(files),
    "files": files,
}
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
# rewrite inventory table with final hashes
lines = [
    "# Embedding Code Transfer Inventory",
    "",
    "**Status:** `TRANSFER_BUNDLE_READY` / `IMPLEMENTED_NOT_EXECUTED` / `TRANSFER_PENDING_VALIDATION`",
    "",
    f"Generated (UTC): `{manifest['created_at_utc']}`",
    "",
    "## A. Required code files",
    "",
    "| Repository-relative path | Size (bytes) | SHA-256 | Required | Purpose |",
    "|---|---:|---|---|---|",
]
for entry in files:
    rel = entry["repository_relative_path"]
    lines.append(
        f"| `{rel}` | {entry['size_bytes']} | `{entry['sha256']}` | required | transfer member |"
    )
inventory_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
# One more hash refresh for inventory after rewrite would create churn; archive uses
# the manifest file written above and regenerates one last time below.
PY

# Final stable manifest after inventory rewrite
python3 - <<'PY' "$ROOT" "$MANIFEST" "${FILES[@]}"
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
root = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
rel_paths = sys.argv[3:]
files = []
for rel in rel_paths:
    path = root / rel
    data = path.read_bytes()
    files.append({
        "repository_relative_path": rel,
        "absolute_path": str(path.resolve()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "required": True,
    })
manifest = {
    "schema_version": "tdmec-embedding-code-transfer-manifest-v1",
    "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "status_labels": [
        "IMPLEMENTED_NOT_EXECUTED",
        "TRANSFER_PENDING_VALIDATION",
        "TRANSFER_BUNDLE_READY",
    ],
    "excludes": [
        ".git", ".env", "secrets", "caches", "source datasets",
        "preprocessing parquet artifacts", "generated embedding outputs",
        "model caches", "model weights", "__pycache__", ".pytest_cache",
    ],
    "file_count": len(files),
    "files": files,
}
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"file_count": len(files), "manifest": str(manifest_path)}, indent=2))
PY

cd "$ROOT"
tar -czf "$ARCHIVE_PATH" \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.pytest_cache' \
  "${FILES[@]}"

python3 - <<PY
import hashlib, json
from pathlib import Path
archive = Path("$ARCHIVE_PATH")
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
meta = {
    "archive_path": str(archive.resolve()),
    "archive_name": archive.name,
    "size_bytes": archive.stat().st_size,
    "sha256": digest,
    "status_labels": [
        "TRANSFER_BUNDLE_READY",
        "IMPLEMENTED_NOT_EXECUTED",
        "TRANSFER_PENDING_VALIDATION",
    ],
}
Path("$OUT_DIR/ARCHIVE_SHA256.json").write_text(json.dumps(meta, indent=2) + "\n")
print(json.dumps(meta, indent=2))
PY

echo "ARCHIVE_READY $ARCHIVE_PATH"
