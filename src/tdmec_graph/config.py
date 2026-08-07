"""Graph builder configuration loading + canonical hashing."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class GraphConfig:
    raw: Dict[str, Any]
    path: Optional[str] = None

    @property
    def canonical(self) -> Dict[str, Any]:
        return self.raw["canonical"]

    @property
    def runtime(self) -> Dict[str, Any]:
        return self.raw.get("runtime", {})

    @property
    def expected_sheet(self) -> str:
        return self.raw.get("expected_sheet", "tweets")

    @property
    def chunk_size(self) -> int:
        return int(self.runtime.get("chunk_size", 100000))

    def config_hash(self) -> str:
        blob = json.dumps(self.canonical, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]


def load_graph_config(path: str | os.PathLike) -> GraphConfig:
    import yaml

    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    _validate(raw)
    return GraphConfig(raw=raw, path=str(p))


def _validate(raw: Dict[str, Any]) -> None:
    if "canonical" not in raw:
        raise ValueError("config missing 'canonical' block")
    c = raw["canonical"]
    required = [
        "frozen_node_count",
        "valid_node_index_min",
        "valid_node_index_max",
        "snapshot_start",
        "snapshot_end",
        "snapshot_count",
        "f_struct",
    ]
    missing = [k for k in required if k not in c]
    if missing:
        raise ValueError(f"canonical config missing keys: {missing}")
    if c["frozen_node_count"] != (
        c["valid_node_index_max"] - c["valid_node_index_min"] + 1
    ):
        raise ValueError("frozen_node_count inconsistent with index range")
    if int(c["snapshot_count"]) != 35:
        raise ValueError("provisional builder expects snapshot_count == 35")
    if int(c["f_struct"]) != 17:
        raise ValueError("f_struct must be 17 (Q-FEAT)")
