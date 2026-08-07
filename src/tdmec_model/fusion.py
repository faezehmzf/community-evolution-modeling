"""Masked relation fusion (QFUS-01)."""
from __future__ import annotations

import torch
from torch import nn

from .types import ModelConfig


class MaskedRelationFusion(nn.Module):
    """Attention over available relations; fallback ``z = h0`` when none available."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        cfg = config or ModelConfig()
        self.config = cfg
        self.w_f = nn.Linear(cfg.d_h, cfg.d_h)
        self.q = nn.Parameter(torch.empty(cfg.d_h))
        nn.init.xavier_uniform_(self.w_f.weight)
        nn.init.zeros_(self.w_f.bias)
        nn.init.normal_(self.q, mean=0.0, std=0.02)

    def forward(
        self,
        h_rel: torch.Tensor,
        avail: torch.Tensor,
        h0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_rel: ``[R, N, d_h]``
            avail: ``[N, R]`` bool — relation availability per node
            h0: ``[N, d_h]`` initial node states
        Returns:
            ``z`` ``[N, d_h]``, ``beta`` ``[N, R]`` (0 on unavailable; rows sum to 1
            when any relation is available, else all-zero with ``z=h0``)
        """
        if h_rel.ndim != 3:
            raise ValueError(f"h_rel must be [R,N,d_h], got {tuple(h_rel.shape)}")
        r, n, d_h = h_rel.shape
        if avail.shape != (n, r):
            raise ValueError(f"avail expected {(n, r)}, got {tuple(avail.shape)}")
        if h0.shape != (n, d_h):
            raise ValueError(f"h0 expected {(n, d_h)}, got {tuple(h0.shape)}")

        # scores: u_{i,r} = q^T tanh(W_f h_{i,r} + b)
        # h_rel is [R,N,d] → process as [N,R,d]
        h = h_rel.transpose(0, 1).contiguous()  # [N, R, d_h]
        u = torch.tanh(self.w_f(h))  # [N, R, d_h]
        scores = torch.einsum("nrd,d->nr", u, self.q)  # [N, R]

        mask = avail.to(dtype=torch.bool)
        neg = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~mask, neg)

        any_avail = mask.any(dim=-1)  # [N]
        beta = torch.softmax(scores, dim=-1)
        beta = beta * mask.to(dtype=beta.dtype)
        # renorm for numerical safety when some (but not all) relations available
        denom = beta.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        beta = beta / denom
        beta = torch.where(any_avail.unsqueeze(-1), beta, torch.zeros_like(beta))

        z_attn = torch.einsum("nr,nrd->nd", beta, h)
        z = torch.where(any_avail.unsqueeze(-1), z_attn, h0)
        return z, beta


__all__ = ["MaskedRelationFusion"]
