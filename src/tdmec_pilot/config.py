"""Pilot configuration loading + canonical config hashing."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PilotConfig:
    raw: Dict[str, Any]
    path: Optional[str] = None

    # ---- convenience accessors ----
    @property
    def canonical(self) -> Dict[str, Any]:
        return self.raw["canonical"]

    @property
    def runtime(self) -> Dict[str, Any]:
        return self.raw.get("runtime", {})

    @property
    def input_files(self) -> List[str]:
        return list(self.raw["input_files"])

    @property
    def expected_sheet(self) -> str:
        return self.raw.get("expected_sheet", "Sheet1")

    @property
    def expected_columns(self) -> List[str]:
        return list(self.raw.get("expected_columns", []))

    @property
    def chunk_size(self) -> int:
        return int(self.runtime.get("chunk_size", 200000))

    def config_hash(self) -> str:
        """Stable hash over the CANONICAL block only (path-independent)."""
        blob = json.dumps(self.canonical, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]

    def resolved(self, key: str, env: Optional[str] = None,
                 default: Any = None) -> Any:
        """Resolve a runtime value with environment-variable override."""
        if env and os.environ.get(env):
            return os.environ[env]
        val = self.runtime.get(key, default)
        return val


def load_pilot_config(path: str | os.PathLike) -> PilotConfig:
    import yaml

    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    _validate_config(raw)
    return PilotConfig(raw=raw, path=str(p))


def _validate_config(raw: Dict[str, Any]) -> None:
    if "canonical" not in raw:
        raise ValueError("config missing 'canonical' block")
    c = raw["canonical"]
    required = [
        "frozen_node_count", "valid_node_index_min", "valid_node_index_max",
        "snapshot_start", "snapshot_end", "snapshot_count",
    ]
    missing = [k for k in required if k not in c]
    if missing:
        raise ValueError(f"canonical config missing keys: {missing}")
    if c["frozen_node_count"] != (c["valid_node_index_max"] - c["valid_node_index_min"] + 1):
        raise ValueError("frozen_node_count inconsistent with valid node index range")
    if c["snapshot_count"] != 35:
        raise ValueError("this pilot expects snapshot_count == 35")
