"""Frozen node-index map: author_account_id (string) -> node_index (int)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class NodeMap:
    mapping: Dict[str, int]
    min_index: int
    max_index: int

    def __len__(self) -> int:
        return len(self.mapping)

    def get(self, account_id: Optional[str]) -> Optional[int]:
        if account_id is None:
            return None
        return self.mapping.get(account_id)


def load_node_map(path: str | Path,
                  expected_count: Optional[int] = None,
                  index_min: int = 0, index_max: Optional[int] = None) -> NodeMap:
    """Load and validate a node-index map from a parquet file.

    Parquet must contain columns ``author_account_id`` and ``node_index``.
    """
    import pandas as pd

    df = pd.read_parquet(path)
    if "author_account_id" not in df.columns or "node_index" not in df.columns:
        raise ValueError("node map parquet must have author_account_id + node_index")
    df = df[["author_account_id", "node_index"]].copy()
    df["author_account_id"] = df["author_account_id"].astype(str)
    df["node_index"] = df["node_index"].astype(int)
    mapping = dict(zip(df["author_account_id"], df["node_index"]))
    if len(mapping) != len(df):
        raise ValueError("duplicate author_account_id in node map")
    mn = int(df["node_index"].min())
    mx = int(df["node_index"].max())
    if expected_count is not None and len(mapping) != expected_count:
        raise ValueError(f"node map has {len(mapping)} entries, expected {expected_count}")
    if mn != index_min:
        raise ValueError(f"node index min {mn} != {index_min}")
    if index_max is not None and mx != index_max:
        raise ValueError(f"node index max {mx} != {index_max}")
    # contiguity check
    if expected_count is not None and (mx - mn + 1) != expected_count:
        raise ValueError("node indices are not contiguous")
    return NodeMap(mapping=mapping, min_index=mn, max_index=mx)


def build_node_map_from_ids(account_ids, index_min: int = 0) -> NodeMap:
    """Deterministically assign indices to a set of account ids (sorted ascending).

    This is the canonical map-generation rule: unique account ids sorted as
    integers ascending -> node_index index_min, index_min+1, …
    """
    uniq = sorted({str(a) for a in account_ids if a is not None}, key=lambda x: int(x))
    mapping = {a: index_min + i for i, a in enumerate(uniq)}
    if not mapping:
        return NodeMap({}, index_min, index_min)
    return NodeMap(mapping, index_min, index_min + len(mapping) - 1)


def save_node_map(nm: NodeMap, path: str | Path) -> None:
    import pandas as pd

    df = pd.DataFrame(
        {"author_account_id": list(nm.mapping.keys()),
         "node_index": list(nm.mapping.values())}
    )
    df = df.sort_values("node_index").reset_index(drop=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
