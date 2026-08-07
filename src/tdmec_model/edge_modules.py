"""Edge context and gate modules (MLP_e / MLP_g) for TDMEC.

TDMEC-G uses graph-only edge context: edge-text projection is exact zero and
``edge_text_available_mask`` is False (Q-MISS / QVAR-01).
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from .encoders import MLPBlock
from .types import ModelConfig


class RelationEmbedding(nn.Module):
    """Learned relation vectors ``e_r`` with shape ``[R, d_rel]``."""

    def __init__(self, num_relations: int, d_rel: int) -> None:
        super().__init__()
        self.num_relations = int(num_relations)
        self.d_rel = int(d_rel)
        self.emb = nn.Embedding(self.num_relations, self.d_rel)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)

    def forward(self, relation_id: torch.Tensor) -> torch.Tensor:
        return self.emb(relation_id)


class EdgeContextG(nn.Module):
    """TDMEC-G edge context: ``g = MLP_e([e_r, weight_log1p, 0_{d_h}, 0])``."""

    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        relation_emb: RelationEmbedding | None = None,
    ) -> None:
        super().__init__()
        cfg = config or ModelConfig()
        self.config = cfg
        self.relation_emb = relation_emb or RelationEmbedding(cfg.num_relations, cfg.d_rel)
        in_dim = cfg.d_rel + 1 + cfg.d_h + 1
        self.mlp_e = MLPBlock(in_dim, cfg.d_h, cfg.d_h, dropout=cfg.dropout)

    def forward(
        self,
        relation_id: torch.Tensor,
        weight_log1p: torch.Tensor,
        edge_text: torch.Tensor | None = None,
        edge_text_available_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Graph-only: text args ignored (exact zeros)."""
        if relation_id.ndim != 1:
            raise ValueError(f"relation_id must be [E], got {tuple(relation_id.shape)}")
        e = self.relation_emb(relation_id)
        w = weight_log1p.reshape(-1, 1).to(dtype=e.dtype)
        zeros_text = e.new_zeros((e.shape[0], self.config.d_h))
        zeros_mask = e.new_zeros((e.shape[0], 1))
        x = torch.cat([e, w, zeros_text, zeros_mask], dim=-1)
        return self.mlp_e(x)


class EdgeContextFull(nn.Module):
    """TDMEC-Full edge context with edge-text projection ``E' = Linear(D_text, d_h)``."""

    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        relation_emb: RelationEmbedding | None = None,
        edge_text_proj: nn.Linear | None = None,
    ) -> None:
        super().__init__()
        cfg = config or ModelConfig()
        if cfg.d_text <= 0:
            raise ValueError(f"d_text must be positive for Full edge context, got {cfg.d_text}")
        self.config = cfg
        self.relation_emb = relation_emb or RelationEmbedding(cfg.num_relations, cfg.d_rel)
        self.edge_text_proj = edge_text_proj or nn.Linear(cfg.d_text, cfg.d_h)
        if edge_text_proj is None:
            nn.init.xavier_uniform_(self.edge_text_proj.weight)
            nn.init.zeros_(self.edge_text_proj.bias)
        in_dim = cfg.d_rel + 1 + cfg.d_h + 1
        self.mlp_e = MLPBlock(in_dim, cfg.d_h, cfg.d_h, dropout=cfg.dropout)

    def project_edge_text(
        self,
        edge_text: torch.Tensor,
        edge_text_available_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(E', m_e)`` with unavailable → exact zero projection."""
        mask = edge_text_available_mask.to(dtype=torch.bool).reshape(-1)
        e_prime = self.edge_text_proj(edge_text)
        e_prime = torch.where(mask.unsqueeze(-1), e_prime, torch.zeros_like(e_prime))
        m = mask.to(dtype=e_prime.dtype).reshape(-1, 1)
        return e_prime, m

    def forward(
        self,
        relation_id: torch.Tensor,
        weight_log1p: torch.Tensor,
        edge_text: torch.Tensor | None = None,
        edge_text_available_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if relation_id.ndim != 1:
            raise ValueError(f"relation_id must be [E], got {tuple(relation_id.shape)}")
        if edge_text is None or edge_text_available_mask is None:
            raise ValueError("EdgeContextFull requires edge_text and edge_text_available_mask")
        e = self.relation_emb(relation_id)
        w = weight_log1p.reshape(-1, 1).to(dtype=e.dtype)
        e_prime, m_e = self.project_edge_text(edge_text, edge_text_available_mask)
        x = torch.cat([e, w, e_prime, m_e], dim=-1)
        return self.mlp_e(x)


class EdgeGate(nn.Module):
    """Scalar edge gate ``γ = σ(MLP_g([h_src, h_dst, g]))`` ∈ (0, 1)."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        cfg = config or ModelConfig()
        self.config = cfg
        self.mlp_g = MLPBlock(3 * cfg.d_h, cfg.d_h, 1, dropout=cfg.dropout)

    def forward(
        self,
        h_src: torch.Tensor,
        h_dst: torch.Tensor,
        g: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([h_src, h_dst, g], dim=-1)
        return torch.sigmoid(self.mlp_g(x))


__all__ = [
    "RelationEmbedding",
    "EdgeContextG",
    "EdgeContextFull",
    "EdgeGate",
]
