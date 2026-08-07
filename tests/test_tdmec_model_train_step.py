"""Decoder / loss unit gates + one-step TDMEC-G trainability check."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tdmec import constants as C
from tdmec_model.dataset import TDMECInputDataset
from tdmec_model.losses import (
    LossWeights,
    cluster_kl_loss,
    combine_losses,
    prototype_separation_loss,
    sample_uniform_negatives,
    split_edges_for_struct_loss,
    structural_bce_loss,
)
from tdmec_model.scaling import RobustFeatureScaler
from tdmec_model.tdmec_g import TDMECG
from tdmec_model.types import ModelConfig

SMOKE_PKG = Path(
    "/teamspace/studios/this_studio/TDMEC_PROJECT_OUTPUTS/tdmec_input/TDMEC_INPUT_smoke_e2e"
)


def test_split_and_negatives_tiny():
    edge_index = torch.tensor([[0, 0, 1], [1, 2, 0]], dtype=torch.long)
    rel = torch.tensor([0, 0, 1], dtype=torch.long)
    w = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    split = split_edges_for_struct_loss(edge_index, rel, w, mask_rate=0.5)
    assert split.enc_edge_index.shape[1] + split.pos_src.numel() == 3
    neg_s, neg_d, neg_r = sample_uniform_negatives(
        split.pos_src,
        split.pos_relation_id,
        edge_index,
        rel,
        num_nodes=4,
        num_neg=3,
    )
    assert neg_s.numel() == split.pos_src.numel() * 3


def test_prototype_and_cluster_losses():
    mu = torch.randn(5, 8)
    l_reg = prototype_separation_loss(mu, margin=1.0)
    assert torch.isfinite(l_reg)
    q = torch.softmax(torch.randn(20, 5), dim=-1)
    l_clu, p = cluster_kl_loss(q)
    assert p.shape == q.shape
    assert torch.isfinite(l_clu)


@pytest.mark.skipif(not SMOKE_PKG.is_dir(), reason="TDMEC_INPUT_smoke_e2e not on disk")
def test_one_train_step_smoke10():
    ds = TDMECInputDataset(SMOKE_PKG)
    scaler = RobustFeatureScaler().fit(
        ds._x_struct,
        ds._struct_mask,
        train_snapshot_ids=list(range(24)),
        time_index_for=ds._snap_to_t,
    )
    b = ds.get_snapshot(10)
    x = scaler.transform(b.x_struct, b.struct_active_mask, as_torch=True)
    assert isinstance(x, torch.Tensor)
    cfg = ModelConfig()
    model = TDMECG(cfg)
    optim = torch.optim.AdamW(model.parameters(), lr=5e-4)
    s0 = model.temporal.initial_state(C.N_NODES)
    split = split_edges_for_struct_loss(
        b.edge_index, b.relation_id, b.weight_log1p, mask_rate=0.15
    )
    out = model.encode_snapshot(
        x,
        split.enc_edge_index,
        split.enc_relation_id,
        split.enc_weight_log1p,
        b.struct_active_mask,
        s0,
        use_fanout=True,
    )
    neg_s, neg_d, neg_r = sample_uniform_negatives(
        split.pos_src,
        split.pos_relation_id,
        b.edge_index,
        b.relation_id,
        num_nodes=C.N_NODES,
        num_neg=3,
    )
    l_struct = structural_bce_loss(
        model.decoder,
        out["s"],
        split.pos_src,
        split.pos_dst,
        split.pos_relation_id,
        split.pos_weight_log1p,
        neg_s,
        neg_d,
        neg_r,
    )
    active = out["model_active"]
    l_cluster, _ = cluster_kl_loss(out["q"][active])
    l_reg = prototype_separation_loss(model.community.mu)
    total, comps = combine_losses(
        l_struct=l_struct,
        l_sem=out["s"].new_zeros(()),
        l_cluster=l_cluster,
        l_reg=l_reg,
        l_temp=out["s"].new_zeros(()),
        weights=LossWeights(lambda_sem=0.0, lambda_temp=0.0),
    )
    assert torch.isfinite(total)
    total.backward()
    optim.step()
    assert comps["L_total"] == comps["L_total"]  # finite logged
