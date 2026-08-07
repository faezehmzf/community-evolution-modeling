"""Semantic projections and ``L_sem`` (QPROJ-01 / Q-MISS)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .types import ModelConfig


class SemanticProjections(nn.Module):
    """``P_z: d_h → d_sem`` and ``P_T_node: D_text → d_sem`` with ``d_sem = d_h`` primary."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        cfg = config or ModelConfig()
        d_sem = int(cfg.d_sem or cfg.d_h)
        self.config = cfg
        self.d_sem = d_sem
        self.p_z = nn.Linear(cfg.d_h, d_sem)
        self.p_t_node = nn.Linear(cfg.d_text, d_sem)
        nn.init.xavier_uniform_(self.p_z.weight)
        nn.init.zeros_(self.p_z.bias)
        nn.init.xavier_uniform_(self.p_t_node.weight)
        nn.init.zeros_(self.p_t_node.bias)

    def forward_z(self, z: torch.Tensor) -> torch.Tensor:
        return self.p_z(z)

    def forward_node_text(self, node_text: torch.Tensor) -> torch.Tensor:
        return self.p_t_node(node_text)


def semantic_cosine_loss(
    z: torch.Tensor,
    node_text: torch.Tensor,
    node_text_available_mask: torch.Tensor,
    projections: SemanticProjections,
) -> torch.Tensor:
    """Mean ``1 - cos(P_z(z), P_T(T))`` over mask-True nodes; else exact 0."""
    mask = node_text_available_mask.to(dtype=torch.bool).reshape(-1)
    if not mask.any():
        return z.new_zeros(())
    pz = F.normalize(projections.forward_z(z[mask]), dim=-1, eps=1e-8)
    pt = F.normalize(projections.forward_node_text(node_text[mask]), dim=-1, eps=1e-8)
    return (1.0 - (pz * pt).sum(dim=-1)).mean()


__all__ = ["SemanticProjections", "semantic_cosine_loss"]
