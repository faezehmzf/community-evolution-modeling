"""Run identity, atomic writes, checksums, and manifest management."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from tdmec_discovery.hashing import sha256_file


def git_short_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      stderr=subprocess.DEVNULL)
        return out.decode().strip() or "nogit"
    except Exception:
        return "nogit"


def make_run_id(config_hash: str, now: Optional[datetime] = None) -> str:
    now = now or datetime.now(timezone.utc)
    ts = now.strftime("%Y-%m-%dT%H%M%SZ")
    return f"{ts}_git-{git_short_hash()}_cfg-{config_hash}"


def runtime_environment() -> dict:
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_short_hash(),
        "cwd": os.getcwd(),
    }


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, default=str))


def write_parquet_atomic(df, path: Path, schema=None) -> str:
    """Write a DataFrame to parquet atomically; return the file's sha256."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)
    return sha256_file(path)


class RunLayout:
    """Filesystem layout for a single pilot run directory."""

    def __init__(self, output_root: str | Path, run_id: str):
        self.root = Path(output_root) / "pilot" / run_id
        self.run_id = run_id

    def ensure(self) -> "RunLayout":
        for sub in ["normalized_records", "excluded_records", "checkpoints", "logs"]:
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        return self

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def checksums_path(self) -> Path:
        return self.root / "checksums.json"

    def normalized_dir(self, source_file: str) -> Path:
        return self.root / "normalized_records" / f"source_file={source_file}"

    def excluded_dir(self) -> Path:
        return self.root / "excluded_records"

    def checkpoint_path(self, source_file: str) -> Path:
        return self.root / "checkpoints" / f"{source_file}.json"


class Manifest:
    def __init__(self, layout: RunLayout):
        self.layout = layout
        self.data: Dict[str, Any] = {}

    def load_or_init(self, run_id: str, config_hash: str, git_commit: str,
                     canonical: dict) -> "Manifest":
        if self.layout.manifest_path.is_file():
            self.data = json.loads(self.layout.manifest_path.read_text())
        else:
            self.data = {
                "run_id": run_id,
                "config_hash": config_hash,
                "git_commit": git_commit,
                "canonical": canonical,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "runtime_environment": runtime_environment(),
                "stages": {},
                "outputs": {},
                "source_checksums": {},
            }
        return self

    def compatibility_check(self, config_hash: str, git_commit: str) -> Dict[str, Any]:
        result = {"config_hash_match": True, "git_commit_match": True, "code_version_changed": False}
        if self.data:
            result["config_hash_match"] = self.data.get("config_hash") == config_hash
            result["git_commit_match"] = self.data.get("git_commit") == git_commit
            result["code_version_changed"] = not result["git_commit_match"]
        return result

    def set_stage(self, stage: str, state: str, detail: Optional[dict] = None) -> None:
        self.data.setdefault("stages", {})[stage] = {
            "state": state, "detail": detail or {},
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def add_output(self, rel: str, size: int, sha256: str, rows: Optional[int] = None) -> None:
        self.data.setdefault("outputs", {})[rel] = {"size": size, "sha256": sha256, "rows": rows}

    def flush(self) -> None:
        atomic_write_json(self.layout.manifest_path, self.data)

    def write_checksums(self) -> None:
        checks = {k: v["sha256"] for k, v in self.data.get("outputs", {}).items()}
        atomic_write_json(self.layout.checksums_path, checks)
