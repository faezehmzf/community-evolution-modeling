"""Gates for fusion, temporal GRU, and community head."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tdmec import constants as C
from tdmec_model.community import StudentTCommunityHead
from tdmec_model.dataset import TDMECInputDataset
from tdmec_model.encoders import NodeEncoderG
from tdmec_model.fusion import MaskedRelationFusion
from tdmec_model.rgcn import DirectedRelationEncoder
from tdmec_model.scaling import RobustFeatureScaler
from tdmec_model.temporal import TemporalGRU, model_active_mask_g
from tdmec_model.types import ModelConfig

SMOKE_PKG = Path(
    "/teamspace/studios/this_studio/TDMEC_PROJECT_OUTPUTS/tdmec_input/TDMEC_INPUT_smoke_e2e"
)
TRAIN_IDS = list(range(0, 24))


def test_fusion_fallback_to_h0():
    cfg = ModelConfig()
    fuse = MaskedRelationFusion(cfg)
    n, r, d = 5, 4, cfg.d_h
    h0 = torch.randn(n, d)
    h_rel = torch.randn(r, n, d)
    avail = torch.zeros(n, r, dtype=torch.bool)
    z, beta = fuse(h_rel, avail, h0)
    assert torch.allclose(z, h0)
    assert torch.all(beta == 0)


def test_fusion_masked_softmax_sums():
    cfg = ModelConfig()
    fuse = MaskedRelationFusion(cfg)
    n, r, d = 6, 4, cfg.d_h
    h0 = torch.randn(n, d)
    h_rel = torch.randn(r, n, d)
    avail = torch.zeros(n, r, dtype=torch.bool)
    avail[:, 0] = True
    avail[:3, 2] = True
    z, beta = fuse(h_rel, avail, h0)
    assert z.shape == (n, d)
    assert torch.isfinite(z).all()
    # rows with any avail sum ≈ 1
    any_a = avail.any(dim=-1)
    assert torch.allclose(beta[any_a].sum(dim=-1), torch.ones(int(any_a.sum())), atol=1e-5)
    assert torch.all(beta[~avail] == 0)


def test_temporal_exact_carry():
    cfg = ModelConfig()
    gru = TemporalGRU(cfg)
    n = 10
    z = torch.randn(n, cfg.d_h)
    s_prev = torch.randn(n, cfg.d_h)
    active = torch.zeros(n, dtype=torch.bool)
    active[:4] = True
    s_t = gru(z, s_prev, active)
    assert torch.allclose(s_t[~active], s_prev[~active])
    assert not torch.allclose(s_t[active], s_prev[active])  # almost surely updated
    assert torch.isfinite(s_t).all()


def test_community_q_shapes():
    cfg = ModelConfig()
    head = StudentTCommunityHead(cfg)
    s = torch.randn(32, cfg.d_h)
    q, aux = head(s)
    assert q.shape == (32, cfg.k_communities)
    assert torch.allclose(q.sum(dim=-1), torch.ones(32), atol=1e-5)
    assert aux["hard"].shape == (32,)
    assert torch.isfinite(q).all()


def test_community_kmeans_init():
    cfg = ModelConfig(k_communities=5)
    head = StudentTCommunityHead(cfg)
    s = torch.randn(200, cfg.d_h)
    active = torch.ones(200, dtype=torch.bool)
    stats = head.init_kmeans_plus_plus(s, active, seed=0, n_init=3, max_iter=10)
    assert head._initialized_from_kmeans
    assert stats["n_active"] == 200.0
    q, _ = head(s)
    assert torch.isfinite(q).all()


@pytest.mark.skipif(not SMOKE_PKG.is_dir(), reason="TDMEC_INPUT_smoke_e2e not on disk")
def test_stack_through_community_smoke10():
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
    h_rel, avail, _ = DirectedRelationEncoder(cfg)(
        h0, b.edge_index, b.relation_id, b.weight_log1p, use_fanout=False
    )
    z, beta = MaskedRelationFusion(cfg)(h_rel, avail, h0)
    # nodes with no relations → z == h0
    none = ~avail.any(dim=-1)
    assert torch.allclose(z[none], h0[none])

    active = model_active_mask_g(b.struct_active_mask)
    gru = TemporalGRU(cfg)
    s0 = gru.initial_state(C.N_NODES)
    s1 = gru(z, s0, active)
    assert torch.allclose(s1[~active], s0[~active])

    q, aux = StudentTCommunityHead(cfg)(s1)
    assert q.shape == (C.N_NODES, 10)
    assert torch.isfinite(q).all()
    assert aux["hard"].shape == (C.N_NODES,)
