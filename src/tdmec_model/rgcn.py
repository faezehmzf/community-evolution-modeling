"""Directed relation-specific GraphSAGE encoder (QENC-01/02) — pure PyTorch.

No torch-geometric dependency. Mean aggregation over in/out neighborhoods;
empty neighborhoods → exact zero. Training fanout defaults to ``[15]``.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from .edge_modules import EdgeContextG, EdgeGate
from .encoders import MLPBlock
from .types import ModelConfig


def scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    *,
    dim_size: int,
) -> torch.Tensor:
    """Mean-aggregate ``src[e]`` into rows ``index[e]``; untouched rows stay 0."""
    if src.numel() == 0:
        return src.new_zeros((dim_size, src.shape[-1] if src.ndim == 2 else 0))
    out = src.new_zeros((dim_size, src.shape[-1]))
    counts = src.new_zeros((dim_size, 1))
    out.index_add_(0, index, src)
    counts.index_add_(0, index, src.new_ones((src.shape[0], 1)))
    return out / counts.clamp(min=1.0)


def _fanout_keep_mask(
    group_ids: torch.Tensor,
    *,
    fanout: int,
    num_nodes: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Boolean mask over edges: keep ≤ ``fanout`` edges per ``group_ids`` value."""
    e = int(group_ids.numel())
    if e == 0 or fanout <= 0:
        return torch.ones(e, dtype=torch.bool, device=group_ids.device)
    # Random permutation then keep first `fanout` occurrences per node.
    perm = torch.randperm(e, device=group_ids.device, generator=generator)
    gid = group_ids[perm]
    # running count per node along permuted order
    order = torch.arange(e, device=group_ids.device)
    # stable sort by gid to assign ranks within each group
    sorted_gid, sort_idx = torch.sort(gid, stable=True)
    # rank within group
    first = torch.ones_like(sorted_gid, dtype=torch.bool)
    first[1:] = sorted_gid[1:] != sorted_gid[:-1]
    rank = torch.arange(e, device=group_ids.device)
    group_start = torch.zeros_like(rank)
    group_start[first] = rank[first]
    group_start = group_start.cummax(dim=0).values
    within = rank - group_start
    keep_sorted = within < fanout
    keep_perm = torch.zeros(e, dtype=torch.bool, device=group_ids.device)
    keep_perm[sort_idx] = keep_sorted
    keep = torch.zeros(e, dtype=torch.bool, device=group_ids.device)
    keep[perm] = keep_perm
    return keep


class RelationLayer(nn.Module):
    """One directed GraphSAGE block for a single relation."""

    def __init__(self, d_h: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w_in = nn.Linear(d_h, d_h, bias=False)
        self.w_out = nn.Linear(d_h, d_h, bias=False)
        self.mlp_r = MLPBlock(3 * d_h, d_h, d_h, dropout=dropout)
        nn.init.xavier_uniform_(self.w_in.weight)
        nn.init.xavier_uniform_(self.w_out.weight)

    def forward(
        self,
        h0: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        gamma: torch.Tensor,
        *,
        fanout: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Args:
            h0: ``[N, d_h]``
            src, dst: ``[E_r]`` endpoints for this relation
            gamma: ``[E_r, 1]``
        Returns:
            ``h_r`` with shape ``[N, d_h]``
        """
        n, d_h = h0.shape
        device = h0.device
        if src.numel() == 0:
            zeros = h0.new_zeros((n, d_h))
            return self.mlp_r(torch.cat([h0, zeros, zeros], dim=-1))

        # Optional training fanout (independent for in / out)
        if fanout is not None and fanout > 0 and self.training:
            keep_in = _fanout_keep_mask(dst, fanout=fanout, num_nodes=n, generator=generator)
            keep_out = _fanout_keep_mask(src, fanout=fanout, num_nodes=n, generator=generator)
        else:
            keep_in = torch.ones(src.shape[0], dtype=torch.bool, device=device)
            keep_out = torch.ones(src.shape[0], dtype=torch.bool, device=device)

        # In-messages to dst: γ · W_in h_src
        if keep_in.any():
            msg_in = gamma[keep_in] * self.w_in(h0[src[keep_in]])
            m_in = scatter_mean(msg_in, dst[keep_in], dim_size=n)
        else:
            m_in = h0.new_zeros((n, d_h))

        # Out-messages from src: γ · W_out h_dst
        if keep_out.any():
            msg_out = gamma[keep_out] * self.w_out(h0[dst[keep_out]])
            m_out = scatter_mean(msg_out, src[keep_out], dim_size=n)
        else:
            m_out = h0.new_zeros((n, d_h))

        return self.mlp_r(torch.cat([h0, m_in, m_out], dim=-1))


class DirectedRelationEncoder(nn.Module):
    """Per-relation directed GraphSAGE stack (primary ``L=1``).

    Returns relation states ``h_r`` for ``r=0..R-1`` and boolean relation
    availability ``a[N, R]`` (node has ≥1 in or out edge in that relation).
    """

    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        edge_context: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        cfg = config or ModelConfig()
        self.config = cfg
        self.edge_context = edge_context or EdgeContextG(cfg)
        self.edge_gate = EdgeGate(cfg)
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [RelationLayer(cfg.d_h, dropout=cfg.dropout) for _ in range(cfg.num_layers)]
                )
                for _ in range(cfg.num_relations)
            ]
        )
        self.fanout = int(cfg.fanout[0]) if cfg.fanout else 15

    def forward(
        self,
        h0: torch.Tensor,
        edge_index: torch.Tensor,
        relation_id: torch.Tensor,
        weight_log1p: torch.Tensor,
        *,
        edge_text: Optional[torch.Tensor] = None,
        edge_text_available_mask: Optional[torch.Tensor] = None,
        use_fanout: Optional[bool] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            h0: ``[N, d_h]``
            edge_index: ``[2, E]``
            relation_id: ``[E]``
            weight_log1p: ``[E]``
            edge_text / mask: required by ``EdgeContextFull``; ignored by G
        Returns:
            ``h_rel`` ``[R, N, d_h]``, ``avail`` ``[N, R]`` bool, aux dict
        """
        n = h0.shape[0]
        r_count = self.config.num_relations
        src = edge_index[0] if edge_index.numel() else h0.new_empty((0,), dtype=torch.long)
        dst = edge_index[1] if edge_index.numel() else h0.new_empty((0,), dtype=torch.long)

        if edge_index.numel() == 0:
            g = h0.new_zeros((0, self.config.d_h))
            gamma = h0.new_zeros((0, 1))
        else:
            g = self.edge_context(
                relation_id,
                weight_log1p,
                edge_text=edge_text,
                edge_text_available_mask=edge_text_available_mask,
            )
            gamma = self.edge_gate(h0[src], h0[dst], g)

        apply_fanout = self.training if use_fanout is None else bool(use_fanout)
        fanout = self.fanout if apply_fanout else None

        h_list: List[torch.Tensor] = []
        avail = torch.zeros((n, r_count), dtype=torch.bool, device=h0.device)
        for r in range(r_count):
            mask = relation_id == r if relation_id.numel() else None
            if mask is not None and mask.any():
                s_r = src[mask]
                d_r = dst[mask]
                g_r = gamma[mask]
                avail[:, r].scatter_(0, s_r, True)
                avail[:, r].scatter_(0, d_r, True)
            else:
                s_r = src.new_empty((0,), dtype=torch.long)
                d_r = dst.new_empty((0,), dtype=torch.long)
                g_r = gamma.new_empty((0, 1))

            h = h0
            for layer in self.layers[r]:  # type: ignore[index]
                h = layer(
                    h,
                    s_r,
                    d_r,
                    g_r,
                    fanout=fanout,
                    generator=generator,
                )
            h_list.append(h)

        h_rel = torch.stack(h_list, dim=0)
        return h_rel, avail, {"gamma": gamma, "g": g}


__all__ = ["DirectedRelationEncoder", "RelationLayer", "scatter_mean"]
