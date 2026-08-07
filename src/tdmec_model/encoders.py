"""Node input encoders (MLP_x) — TDMEC-G and TDMEC-Full paths."""
from __future__ import annotations

import torch
from torch import nn

from tdmec import constants as C

from .types import ModelConfig


def _kaiming_linear(layer: nn.Linear) -> None:
    nn.init.kaiming_uniform_(layer.weight, a=0.0, nonlinearity="relu")
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class MLPBlock(nn.Module):
    """Linear → ReLU → Linear with Kaiming init (QMLP-01 primary)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        _kaiming_linear(self.fc1)
        _kaiming_linear(self.fc2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.act(self.fc1(x))))


class NodeEncoderG(nn.Module):
    """TDMEC-G initial node encoder: ``MLP_x(X_struct_scaled) → h^(0)``."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        cfg = config or ModelConfig()
        if cfg.f_struct != C.F_STRUCT:
            raise ValueError(f"f_struct must be {C.F_STRUCT}, got {cfg.f_struct}")
        self.config = cfg
        self.mlp_x = MLPBlock(
            in_dim=cfg.f_struct,
            hidden_dim=cfg.d_h,
            out_dim=cfg.d_h,
            dropout=cfg.dropout,
        )

    def forward(self, x_struct_scaled: torch.Tensor) -> torch.Tensor:
        if x_struct_scaled.ndim != 2 or x_struct_scaled.shape[-1] != self.config.f_struct:
            raise ValueError(
                f"expected [N,{self.config.f_struct}], got {tuple(x_struct_scaled.shape)}"
            )
        return self.mlp_x(x_struct_scaled)


class NodeEncoderFull(nn.Module):
    """TDMEC-Full: ``MLP_x([X_struct, X_node_text, mask]) → h^(0)``."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        cfg = config or ModelConfig()
        if cfg.f_struct != C.F_STRUCT:
            raise ValueError(f"f_struct must be {C.F_STRUCT}, got {cfg.f_struct}")
        if cfg.d_text <= 0:
            raise ValueError(f"d_text must be positive for Full encoder, got {cfg.d_text}")
        self.config = cfg
        in_dim = cfg.f_struct + cfg.d_text + 1
        self.mlp_x = MLPBlock(
            in_dim=in_dim,
            hidden_dim=cfg.d_h,
            out_dim=cfg.d_h,
            dropout=cfg.dropout,
        )

    def forward(
        self,
        x_struct_scaled: torch.Tensor,
        node_text: torch.Tensor,
        node_text_available_mask: torch.Tensor,
    ) -> torch.Tensor:
        if x_struct_scaled.ndim != 2 or x_struct_scaled.shape[-1] != self.config.f_struct:
            raise ValueError(
                f"expected struct [N,{self.config.f_struct}], got {tuple(x_struct_scaled.shape)}"
            )
        if node_text.shape != (x_struct_scaled.shape[0], self.config.d_text):
            raise ValueError(
                f"expected node_text [N,{self.config.d_text}], got {tuple(node_text.shape)}"
            )
        mask = node_text_available_mask.to(dtype=x_struct_scaled.dtype).reshape(-1, 1)
        x = torch.cat([x_struct_scaled, node_text, mask], dim=-1)
        return self.mlp_x(x)


__all__ = ["MLPBlock", "NodeEncoderG", "NodeEncoderFull", "_kaiming_linear"]
