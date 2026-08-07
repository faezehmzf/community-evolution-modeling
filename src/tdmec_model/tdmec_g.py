"""TDMEC-G graph-only model assembly (QVAR-01 ablation)."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn

from .community import StudentTCommunityHead
from .decoder import StructuralDecoder
from .encoders import NodeEncoderG
from .fusion import MaskedRelationFusion
from .rgcn import DirectedRelationEncoder
from .temporal import TemporalGRU, model_active_mask_g
from .types import ModelConfig, SnapshotBatch


class TDMECG(nn.Module):
    """Graph-only TDMEC forward: struct → encode → fuse → GRU → community."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.node_encoder = NodeEncoderG(self.config)
        self.relation_encoder = DirectedRelationEncoder(self.config)
        self.fusion = MaskedRelationFusion(self.config)
        self.temporal = TemporalGRU(self.config)
        self.community = StudentTCommunityHead(self.config)
        self.decoder = StructuralDecoder(
            self.config,
            relation_emb=self.relation_encoder.edge_context.relation_emb,
        )

    def encode_snapshot(
        self,
        x_struct_scaled: torch.Tensor,
        edge_index: torch.Tensor,
        relation_id: torch.Tensor,
        weight_log1p: torch.Tensor,
        struct_active_mask: torch.Tensor,
        s_prev: torch.Tensor,
        *,
        use_fanout: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        h0 = self.node_encoder(x_struct_scaled)
        h_rel, avail, aux_e = self.relation_encoder(
            h0,
            edge_index,
            relation_id,
            weight_log1p,
            use_fanout=use_fanout,
        )
        z, beta = self.fusion(h_rel, avail, h0)
        active = model_active_mask_g(struct_active_mask)
        s = self.temporal(z, s_prev, active)
        q, aux_c = self.community(s)
        return {
            "h0": h0,
            "h_rel": h_rel,
            "avail": avail,
            "beta": beta,
            "z": z,
            "s": s,
            "q": q,
            "model_active": active,
            "gamma": aux_e["gamma"],
            "hard": aux_c["hard"],
            "confidence": aux_c["confidence"],
            "entropy": aux_c["entropy"],
        }

    def forward_batch(
        self,
        batch: SnapshotBatch,
        x_struct_scaled: torch.Tensor,
        s_prev: torch.Tensor,
        *,
        enc_edge_index: Optional[torch.Tensor] = None,
        enc_relation_id: Optional[torch.Tensor] = None,
        enc_weight_log1p: Optional[torch.Tensor] = None,
        use_fanout: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        edge_index = batch.edge_index if enc_edge_index is None else enc_edge_index
        relation_id = batch.relation_id if enc_relation_id is None else enc_relation_id
        weight = batch.weight_log1p if enc_weight_log1p is None else enc_weight_log1p
        out = self.encode_snapshot(
            x_struct_scaled,
            edge_index,
            relation_id,
            weight,
            batch.struct_active_mask,
            s_prev,
            use_fanout=use_fanout,
        )
        out["snapshot_id"] = torch.tensor(batch.snapshot_id, device=s_prev.device)
        return out


__all__ = ["TDMECG"]
