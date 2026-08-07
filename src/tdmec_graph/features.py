"""Build X_struct[T,N,17] and struct_active_mask[T,N] (Q-FEAT)."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Mapping, Tuple

import numpy as np

from tdmec import constants as C

EdgeKey = Tuple[int, int, int, int]


def build_structural_tensors(
    edge_counts: Mapping[EdgeKey, int],
    tweet_counts: Mapping[Tuple[int, int], int],
    *,
    t_snapshots: int = C.PROVISIONAL_SNAPSHOT_COUNT,
    n_nodes: int = C.N_NODES,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X_struct float32 [T,N,17], struct_active_mask bool [T,N])."""
    if t_snapshots != C.PROVISIONAL_SNAPSHOT_COUNT:
        raise ValueError("provisional builder expects T=35")
    if n_nodes != C.N_NODES:
        raise ValueError(f"N must be {C.N_NODES}")

    out_neighbors: Dict[Tuple[int, int, int], set] = defaultdict(set)
    in_neighbors: Dict[Tuple[int, int, int], set] = defaultdict(set)
    out_strength: Dict[Tuple[int, int, int], int] = defaultdict(int)
    in_strength: Dict[Tuple[int, int, int], int] = defaultdict(int)

    for (sid, rid, src, dst), count in edge_counts.items():
        if src == dst or count <= 0:
            continue
        if not (0 <= sid < t_snapshots):
            continue
        out_neighbors[(sid, rid, src)].add(dst)
        in_neighbors[(sid, rid, dst)].add(src)
        out_strength[(sid, rid, src)] += int(count)
        in_strength[(sid, rid, dst)] += int(count)

    x = np.zeros((t_snapshots, n_nodes, C.F_STRUCT), dtype=np.float32)
    for (sid, rid, node), neigh in out_neighbors.items():
        if not (0 <= node < n_nodes):
            continue
        base = int(rid) * 4
        x[sid, node, base + 0] = float(len(neigh))
        x[sid, node, base + 2] = float(np.log1p(out_strength[(sid, rid, node)]))
    for (sid, rid, node), neigh in in_neighbors.items():
        if not (0 <= node < n_nodes):
            continue
        base = int(rid) * 4
        x[sid, node, base + 1] = float(len(neigh))
        x[sid, node, base + 3] = float(np.log1p(in_strength[(sid, rid, node)]))

    for (sid, node), tc in tweet_counts.items():
        if 0 <= sid < t_snapshots and 0 <= node < n_nodes and tc > 0:
            x[sid, node, 16] = float(np.log1p(int(tc)))

    mask = np.zeros((t_snapshots, n_nodes), dtype=bool)
    for (sid, node), tc in tweet_counts.items():
        if tc > 0 and 0 <= sid < t_snapshots and 0 <= node < n_nodes:
            mask[sid, node] = True
    for (sid, rid, src, dst), count in edge_counts.items():
        if count <= 0 or src == dst:
            continue
        if 0 <= sid < t_snapshots:
            if 0 <= src < n_nodes:
                mask[sid, src] = True
            if 0 <= dst < n_nodes:
                mask[sid, dst] = True

    x[~mask] = 0.0
    return x, mask
