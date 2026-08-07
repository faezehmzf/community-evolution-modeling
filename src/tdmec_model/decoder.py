"""Structural edge decoder for ``L_struct`` (QDEC-01)."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .edge_modules import RelationEmbedding
from .encoders import MLPBlock
from .types import ModelConfig


class StructuralDecoder(nn.Module):
    """Score directed edges from post-GRU states.

    Positives may carry edge-text projection ``E'`` + mask (Full). Negatives use
    ``E'=0``, ``m_e=False``, ``weight_log1p=0``. TDMEC-G passes zeros for text.
    """

    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        relation_emb: RelationEmbedding | None = None,
        edge_text_proj: Optional[nn.Linear] = None,
    ) -> None:
        super().__init__()
        cfg = config or ModelConfig()
        self.config = cfg
        self.relation_emb = relation_emb or RelationEmbedding(cfg.num_relations, cfg.d_rel)
        self.edge_text_proj = edge_text_proj
        in_dim = 2 * cfg.d_h + cfg.d_rel + cfg.d_h + 1 + 1
        self.mlp_dec = MLPBlock(in_dim, cfg.d_h, 1, dropout=cfg.dropout)

    def forward(
        self,
        s: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        relation_id: torch.Tensor,
        weight_log1p: torch.Tensor,
        *,
        edge_text: Optional[torch.Tensor] = None,
        edge_text_available_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns logits ``[E]``."""
        if src.numel() == 0:
            return s.new_zeros((0,))
        e_r = self.relation_emb(relation_id)
        if (
            edge_text is not None
            and edge_text_available_mask is not None
            and self.edge_text_proj is not None
        ):
            mask = edge_text_available_mask.to(dtype=torch.bool).reshape(-1)
            e_prime = self.edge_text_proj(edge_text)
            e_prime = torch.where(mask.unsqueeze(-1), e_prime, torch.zeros_like(e_prime))
            m_e = mask.to(dtype=s.dtype).reshape(-1, 1)
        else:
            e_prime = s.new_zeros((src.shape[0], self.config.d_h))
            m_e = s.new_zeros((src.shape[0], 1))
        w = weight_log1p.reshape(-1, 1).to(dtype=s.dtype)
        x = torch.cat([s[src], s[dst], e_r, e_prime, m_e, w], dim=-1)
        return self.mlp_dec(x).squeeze(-1)


__all__ = ["StructuralDecoder"]
