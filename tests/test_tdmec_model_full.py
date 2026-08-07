"""TDMEC-Full gates: text batch load, semantic path, one train step."""
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
from tdmec_model.semantic import semantic_cosine_loss
from tdmec_model.tdmec_full import TDMECFull
from tdmec_model.types import ModelConfig

SMOKE_PKG = Path(
    "/teamspace/studios/this_studio/TDMEC_PROJECT_OUTPUTS/tdmec_input/TDMEC_INPUT_smoke_e2e"
)


@pytest.mark.skipif(not SMOKE_PKG.is_dir(), reason="TDMEC_INPUT_smoke_e2e not on disk")
def test_dataset_loads_aligned_text():
    ds = TDMECInputDataset(SMOKE_PKG, load_text=True)
    b = ds.get_snapshot(10)
    assert b.has_text
    assert b.node_text is not None and b.edge_text is not None
    assert b.node_text.shape == (C.N_NODES, ds.d_text)
    assert b.edge_text.shape == (b.num_edges, ds.d_text)
    assert b.edge_canonical_idx is not None
    assert b.edge_canonical_idx.shape[0] == b.num_edges
    # unavailable exact zeros
    assert torch.all(b.node_text[~b.node_text_available_mask] == 0)  # type: ignore[index]
    assert torch.all(b.edge_text[~b.edge_text_available_mask] == 0)  # type: ignore[index]


@pytest.mark.skipif(not SMOKE_PKG.is_dir(), reason="TDMEC_INPUT_smoke_e2e not on disk")
def test_full_one_train_step_smoke10():
    ds = TDMECInputDataset(SMOKE_PKG, load_text=True)
    scaler = RobustFeatureScaler().fit(
        ds._x_struct,
        ds._struct_mask,
        train_snapshot_ids=list(range(24)),
        time_index_for=ds._snap_to_t,
    )
    b = ds.get_snapshot(10)
    x = scaler.transform(b.x_struct, b.struct_active_mask, as_torch=True)
    assert isinstance(x, torch.Tensor)
    cfg = ModelConfig(d_text=ds.d_text, d_sem=64)
    model = TDMECFull(cfg)
    optim = torch.optim.AdamW(model.parameters(), lr=5e-4)
    s0 = model.temporal.initial_state(C.N_NODES)
    split = split_edges_for_struct_loss(
        b.edge_index,
        b.relation_id,
        b.weight_log1p,
        mask_rate=0.15,
        edge_text=b.edge_text,
        edge_text_available_mask=b.edge_text_available_mask,
    )
    out = model.encode_snapshot(
        x,
        b.node_text,  # type: ignore[arg-type]
        b.node_text_available_mask,  # type: ignore[arg-type]
        split.enc_edge_index,
        split.enc_relation_id,
        split.enc_weight_log1p,
        split.enc_edge_text,  # type: ignore[arg-type]
        split.enc_edge_text_mask,  # type: ignore[arg-type]
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
        pos_edge_text=split.pos_edge_text,
        pos_edge_text_mask=split.pos_edge_text_mask,
    )
    l_sem = semantic_cosine_loss(
        out["z"],
        b.node_text,  # type: ignore[arg-type]
        b.node_text_available_mask,  # type: ignore[arg-type]
        model.semantic,
    )
    active = out["model_active"]
    l_cluster, _ = cluster_kl_loss(out["q"][active])
    l_reg = prototype_separation_loss(model.community.mu)
    total, comps = combine_losses(
        l_struct=l_struct,
        l_sem=l_sem,
        l_cluster=l_cluster,
        l_reg=l_reg,
        l_temp=out["s"].new_zeros(()),
        weights=LossWeights(lambda_sem=1.0, lambda_temp=0.0),
    )
    assert torch.isfinite(total)
    assert comps["L_sem"] >= 0.0
    total.backward()
    optim.step()
