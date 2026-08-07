"""Gates for edge modules + directed relation GraphSAGE."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tdmec import constants as C
from tdmec_model.dataset import TDMECInputDataset
from tdmec_model.edge_modules import EdgeContextG, EdgeGate
from tdmec_model.encoders import NodeEncoderG
from tdmec_model.rgcn import DirectedRelationEncoder, scatter_mean
from tdmec_model.scaling import RobustFeatureScaler
from tdmec_model.types import ModelConfig

SMOKE_PKG = Path(
    "/teamspace/studios/this_studio/TDMEC_PROJECT_OUTPUTS/tdmec_input/TDMEC_INPUT_smoke_e2e"
)
TRAIN_IDS = list(range(0, 24))


def test_scatter_mean_empty_and_basic():
    src = torch.tensor([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]])
    index = torch.tensor([0, 0, 2])
    out = scatter_mean(src, index, dim_size=4)
    assert out.shape == (4, 2)
    assert torch.allclose(out[0], torch.tensor([2.0, 2.0]))
    assert torch.allclose(out[1], torch.zeros(2))
    assert torch.allclose(out[2], torch.tensor([5.0, 5.0]))
    empty = scatter_mean(src[:0], index[:0], dim_size=4)
    assert empty.shape == (4, 2)
    assert torch.all(empty == 0)


def test_edge_modules_tiny():
    cfg = ModelConfig(d_h=64, d_rel=16)
    ctx = EdgeContextG(cfg)
    gate = EdgeGate(cfg)
    rel = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    w = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
    g = ctx(rel, w)
    assert g.shape == (4, 64)
    assert torch.isfinite(g).all()
    h_src = torch.randn(4, 64)
    h_dst = torch.randn(4, 64)
    gamma = gate(h_src, h_dst, g)
    assert gamma.shape == (4, 1)
    assert torch.isfinite(gamma).all()
    assert torch.all((gamma >= 0) & (gamma <= 1))


def test_rgcn_empty_edges():
    cfg = ModelConfig()
    enc = DirectedRelationEncoder(cfg)
    h0 = torch.randn(16, cfg.d_h)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    rel = torch.zeros((0,), dtype=torch.long)
    w = torch.zeros((0,), dtype=torch.float32)
    h_rel, avail, aux = enc(h0, edge_index, rel, w)
    assert h_rel.shape == (4, 16, cfg.d_h)
    assert avail.shape == (16, 4)
    assert not avail.any()
    assert torch.isfinite(h_rel).all()


@pytest.mark.skipif(not SMOKE_PKG.is_dir(), reason="TDMEC_INPUT_smoke_e2e not on disk")
def test_rgcn_smoke_snapshot_10():
    ds = TDMECInputDataset(SMOKE_PKG)
    scaler = RobustFeatureScaler().fit(
        ds._x_struct,
        ds._struct_mask,
        train_snapshot_ids=TRAIN_IDS,
        time_index_for=ds._snap_to_t,
    )
    b = ds.get_snapshot(10)
    x = scaler.transform(b.x_struct, b.struct_active_mask, as_torch=True)
    cfg = ModelConfig()
    h0 = NodeEncoderG(cfg)(x)
    enc = DirectedRelationEncoder(cfg)
    enc.train()
    h_rel, avail, aux = enc(
        h0, b.edge_index, b.relation_id, b.weight_log1p, use_fanout=True
    )
    assert h_rel.shape == (4, C.N_NODES, 64)
    assert avail.shape == (C.N_NODES, 4)
    assert torch.isfinite(h_rel).all()
    assert aux["gamma"].shape[0] == b.num_edges
    assert torch.isfinite(aux["gamma"]).all()
    assert torch.all((aux["gamma"] >= 0) & (aux["gamma"] <= 1))
    # at least one relation available somewhere in a dense quarter
    assert avail.any()
    # nodes with no edges in a relation stay False there
    assert (~avail).any()
