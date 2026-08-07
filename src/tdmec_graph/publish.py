"""Filesystem layout, atomic writes, and manifests for graph runs."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from tdmec_discovery.hashing import sha256_file
from tdmec_pilot.storage import (
    atomic_write_json,
    git_short_hash,
    make_run_id,
    runtime_environment,
    write_parquet_atomic,
)


class GraphRunLayout:
    def __init__(self, output_root: str | Path, run_id: str):
        self.root = Path(output_root) / "graph" / run_id
        self.run_id = run_id

    def ensure(self) -> "GraphRunLayout":
        for sub in [
            "edges",
            "events",
            "checkpoints",
            "logs",
            "work",
        ]:
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        return self

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def checksums_path(self) -> Path:
        return self.root / "checksums.json"

    @property
    def work_db(self) -> Path:
        """Deprecated SQLite path (no longer used; Postgres is SoR)."""
        return self.root / "work" / "signatures.sqlite"

    def edge_part(self, snapshot_id: int, relation_id: int) -> Path:
        return (
            self.root
            / "edges"
            / f"snapshot={snapshot_id}"
            / f"relation={relation_id}"
            / "part-00000.parquet"
        )

    def events_part(self, source_file: str, chunk_idx: int) -> Path:
        stem = Path(source_file).stem
        return self.root / "events" / f"source_file={stem}" / f"part-{chunk_idx:05d}.parquet"

    def checkpoint_path(self, source_file: str) -> Path:
        return self.root / "checkpoints" / f"{Path(source_file).name}.json"


class GraphManifest:
    def __init__(self, layout: GraphRunLayout):
        self.layout = layout
        self.data: Dict[str, Any] = {}

    def load_or_init(self, run_id: str, config_hash: str, canonical: dict) -> "GraphManifest":
        if self.layout.manifest_path.is_file():
            self.data = json.loads(self.layout.manifest_path.read_text(encoding="utf-8"))
        else:
            self.data = {
                "run_id": run_id,
                "config_hash": config_hash,
                "git_commit": git_short_hash(),
                "canonical": canonical,
                "artifact_status": "PROVISIONAL",
                "certification_status": "UNVALIDATED",
                "calendar_status": canonical.get("calendar_status", "PROVISIONAL"),
                "dedup_status": canonical.get("dedup_status", "PROVISIONAL"),
                "cleaned_text_policy": canonical.get(
                    "cleaned_text_policy", "nfc_normalized_v1"
                ),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "runtime_environment": runtime_environment(),
                "stages": {},
                "outputs": {},
                "source_checksums": {},
                "accounting": {},
            }
        return self

    def set_stage(self, stage: str, state: str, detail: Optional[dict] = None) -> None:
        self.data.setdefault("stages", {})[stage] = {
            "state": state,
            "detail": detail or {},
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def add_output(self, rel: str, size: int, sha256: str, rows: Optional[int] = None) -> None:
        self.data.setdefault("outputs", {})[rel] = {
            "size": size,
            "sha256": sha256,
            "rows": rows,
        }

    def flush(self) -> None:
        atomic_write_json(self.layout.manifest_path, self.data)

    def write_checksums(self) -> None:
        checks = {k: v["sha256"] for k, v in self.data.get("outputs", {}).items()}
        atomic_write_json(self.layout.checksums_path, checks)


__all__ = [
    "GraphRunLayout",
    "GraphManifest",
    "atomic_write_json",
    "make_run_id",
    "git_short_hash",
    "write_parquet_atomic",
    "sha256_file",
]
