"""TDMEC-Full model assembly (graph + node text + edge text)."""
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from .community import StudentTCommunityHead
from .decoder import StructuralDecoder
from .edge_modules import EdgeContextFull, RelationEmbedding
from .encoders import NodeEncoderFull
from .fusion import MaskedRelationFusion
from .rgcn import DirectedRelationEncoder
from .semantic import SemanticProjections
from .temporal import TemporalGRU, model_active_mask_g
from .types import ModelConfig, SnapshotBatch


class TDMECFull(nn.Module):
    """Primary TDMEC method: structural + node-text + edge-text paths."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        cfg = config or ModelConfig()
        if cfg.d_text <= 0:
            raise ValueError("TDMECFull requires config.d_text > 0")
        self.config = cfg
        self.relation_emb = RelationEmbedding(cfg.num_relations, cfg.d_rel)
        self.edge_text_proj = nn.Linear(cfg.d_text, cfg.d_h)
        nn.init.xavier_uniform_(self.edge_text_proj.weight)
        nn.init.zeros_(self.edge_text_proj.bias)

        self.node_encoder = NodeEncoderFull(cfg)
        self.edge_context = EdgeContextFull(
            cfg, relation_emb=self.relation_emb, edge_text_proj=self.edge_text_proj
        )
        self.relation_encoder = DirectedRelationEncoder(cfg, edge_context=self.edge_context)
        self.fusion = MaskedRelationFusion(cfg)
        self.temporal = TemporalGRU(cfg)
        self.community = StudentTCommunityHead(cfg)
        self.semantic = SemanticProjections(cfg)
        self.decoder = StructuralDecoder(
            cfg,
            relation_emb=self.relation_emb,
            edge_text_proj=self.edge_text_proj,
        )

    def encode_snapshot(
        self,
        x_struct_scaled: torch.Tensor,
        node_text: torch.Tensor,
        node_text_available_mask: torch.Tensor,
        edge_index: torch.Tensor,
        relation_id: torch.Tensor,
        weight_log1p: torch.Tensor,
        edge_text: torch.Tensor,
        edge_text_available_mask: torch.Tensor,
        struct_active_mask: torch.Tensor,
        s_prev: torch.Tensor,
        *,
        use_fanout: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        h0 = self.node_encoder(x_struct_scaled, node_text, node_text_available_mask)
        h_rel, avail, aux_e = self.relation_encoder(
            h0,
            edge_index,
            relation_id,
            weight_log1p,
            edge_text=edge_text,
            edge_text_available_mask=edge_text_available_mask,
            use_fanout=use_fanout,
        )
        z, beta = self.fusion(h_rel, avail, h0)
        active = model_active_mask_g(struct_active_mask, node_text_available_mask)
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
            "node_text": node_text,
            "node_text_available_mask": node_text_available_mask,
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
        enc_edge_text: Optional[torch.Tensor] = None,
        enc_edge_text_mask: Optional[torch.Tensor] = None,
        use_fanout: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        if batch.node_text is None or batch.edge_text is None:
            raise ValueError("TDMECFull requires node_text and edge_text on SnapshotBatch")
        edge_index = batch.edge_index if enc_edge_index is None else enc_edge_index
        relation_id = batch.relation_id if enc_relation_id is None else enc_relation_id
        weight = batch.weight_log1p if enc_weight_log1p is None else enc_weight_log1p
        edge_text = batch.edge_text if enc_edge_text is None else enc_edge_text
        edge_mask = (
            batch.edge_text_available_mask
            if enc_edge_text_mask is None
            else enc_edge_text_mask
        )
        out = self.encode_snapshot(
            x_struct_scaled,
            batch.node_text,
            batch.node_text_available_mask,  # type: ignore[arg-type]
            edge_index,
            relation_id,
            weight,
            edge_text,
            edge_mask,  # type: ignore[arg-type]
            batch.struct_active_mask,
            s_prev,
            use_fanout=use_fanout,
        )
        out["snapshot_id"] = torch.tensor(batch.snapshot_id, device=s_prev.device)
        return out


__all__ = ["TDMECFull"]
