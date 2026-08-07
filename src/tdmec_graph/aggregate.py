"""Aggregate distinct events into weighted directed edges (Q-WGT)."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from tdmec import constants as C

from .events import GraphEvent

EdgeKey = Tuple[int, int, int, int]  # snapshot, relation, src, dst


def accumulate_edges(
    events: Iterable[GraphEvent],
    counts: Dict[EdgeKey, int],
) -> None:
    for ev in events:
        key = (ev.snapshot_id, ev.relation_id, ev.source_idx, ev.target_idx)
        counts[key] += 1


def edges_to_records(counts: Dict[EdgeKey, int]) -> List[dict]:
    rows: List[dict] = []
    for (sid, rid, src, dst), count in sorted(counts.items()):
        if src == dst:
            continue
        if not (C.NODE_INDEX_MIN <= src <= C.NODE_INDEX_MAX):
            continue
        if not (C.NODE_INDEX_MIN <= dst <= C.NODE_INDEX_MAX):
            continue
        if not (C.RELATION_ID_MIN <= rid <= C.RELATION_ID_MAX):
            continue
        if count <= 0:
            continue
        rows.append(
            {
                "snapshot_id": int(sid),
                "relation_id": int(rid),
                "src_index": int(src),
                "dst_index": int(dst),
                "count_raw": int(count),
                "weight_log1p": float(math.log1p(int(count))),
            }
        )
    return rows


def empty_edge_counts() -> Dict[EdgeKey, int]:
    return defaultdict(int)
