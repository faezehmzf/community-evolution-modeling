"""Shared dataclasses for TDMEC model I/O."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch

from tdmec import constants as C


@dataclass(frozen=True)
class ModelConfig:
    """Primary experimental defaults for TDMEC (Batch 4/7)."""

    d_h: int = 64
    k_communities: int = 10
    d_rel: int = 16
    d_text: int = 16  # smoke mock default; set from package for Full
    d_sem: int = 64  # primary: d_sem = d_h
    alpha: float = 1.0
    num_layers: int = 1
    fanout: Tuple[int, ...] = (15,)
    f_struct: int = 17
    n_nodes: int = C.N_NODES
    num_relations: int = len(C.RELATION_ORDER)
    relation_order: Tuple[str, ...] = C.RELATION_ORDER
    dropout: float = 0.0
    bptt_window: int = 3
    status_labels: Tuple[str, ...] = (
        "PROVISIONAL_SMOKE_ONLY",
        "ENGINEERING_VALIDATION",
        "NOT_FOR_FINAL_THESIS_CONCLUSIONS",
    )


@dataclass
class SnapshotBatch:
    """Per-snapshot tensors for TDMEC-G / TDMEC-Full."""

    snapshot_id: int
    time_index: int
    x_struct: torch.Tensor  # [N, F]
    struct_active_mask: torch.Tensor  # [N] bool
    edge_index: torch.Tensor  # [2, E]
    relation_id: torch.Tensor  # [E] int64
    count_raw: torch.Tensor  # [E] int64
    weight_log1p: torch.Tensor  # [E] float32
    quarter_label: Optional[str] = None
    node_text: Optional[torch.Tensor] = None  # [N, D_text]
    node_text_available_mask: Optional[torch.Tensor] = None  # [N] bool
    edge_text: Optional[torch.Tensor] = None  # [E, D_text]
    edge_text_available_mask: Optional[torch.Tensor] = None  # [E] bool
    edge_canonical_idx: Optional[torch.Tensor] = None  # [E] int64
    status_labels: Tuple[str, ...] = field(
        default_factory=lambda: (
            "PROVISIONAL_SMOKE_ONLY",
            "ENGINEERING_VALIDATION",
            "NOT_FOR_FINAL_THESIS_CONCLUSIONS",
        )
    )

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def num_nodes(self) -> int:
        return int(self.x_struct.shape[0])

    @property
    def has_text(self) -> bool:
        return self.node_text is not None and self.edge_text is not None


@dataclass(frozen=True)
class PackageMeta:
    package_root: str
    package_id: str
    graph_run_id: str
    embedding_run_id: str
    d_text: int
    t_snapshots: int
    n_nodes: int
    e_canonical: int
    relation_ids: Tuple[int, ...]
    status_labels: Tuple[str, ...]
    snapshot_ids: Tuple[int, ...]
    quarter_labels: Tuple[str, ...]
