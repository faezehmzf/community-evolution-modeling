"""GRU temporal encoder with exact inactive-state carry (QGRU-01 / QACT-01)."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .types import ModelConfig


def model_active_mask_g(
    struct_active_mask: torch.Tensor,
    node_text_available_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build ``model_active`` for TDMEC-G.

    Full method uses ``struct OR node_text``. Graph-only ablation ignores text
    (or treats missing text mask as all-False).
    """
    active = struct_active_mask.to(dtype=torch.bool)
    if node_text_available_mask is not None:
        active = active | node_text_available_mask.to(dtype=torch.bool)
    return active


class TemporalGRU(nn.Module):
    """Single GRU cell over nodes; inactive nodes exactly carry ``s``."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        cfg = config or ModelConfig()
        self.config = cfg
        self.gru = nn.GRUCell(cfg.d_h, cfg.d_h)
        # GRUCell uses its own init; keep default (orthogonal-ish via PyTorch)

    def initial_state(
        self,
        num_nodes: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        return torch.zeros(
            (num_nodes, self.config.d_h),
            device=device,
            dtype=dtype or torch.float32,
        )

    def forward(
        self,
        z: torch.Tensor,
        s_prev: torch.Tensor,
        model_active: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z: ``[N, d_h]`` fused node state at ``t``
            s_prev: ``[N, d_h]`` previous temporal state
            model_active: ``[N]`` bool
        Returns:
            ``s_t`` ``[N, d_h]`` — GRU update where active, else exact ``s_prev``
        """
        if z.shape != s_prev.shape:
            raise ValueError(f"z/s_prev shape mismatch: {tuple(z.shape)} vs {tuple(s_prev.shape)}")
        active = model_active.to(dtype=torch.bool).reshape(-1)
        if active.numel() != z.shape[0]:
            raise ValueError(f"model_active length {active.numel()} != N={z.shape[0]}")

        s_new = self.gru(z, s_prev)
        # Exact carry for inactive (no mixing / no dtype cast drift beyond copy)
        s_t = torch.where(active.unsqueeze(-1), s_new, s_prev)
        return s_t


__all__ = ["TemporalGRU", "model_active_mask_g"]
