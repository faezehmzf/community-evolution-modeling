"""Gates for QHP-02 scaler and TDMEC-G MLP_x encoder."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tdmec import constants as C
from tdmec_model.dataset import TDMECInputDataset
from tdmec_model.encoders import NodeEncoderG
from tdmec_model.scaling import RobustFeatureScaler, ScalingError
from tdmec_model.types import ModelConfig

SMOKE_PKG = Path(
    "/teamspace/studios/this_studio/TDMEC_PROJECT_OUTPUTS/tdmec_input/TDMEC_INPUT_smoke_e2e"
)
TRAIN_IDS = list(range(0, 24))


@pytest.mark.skipif(not SMOKE_PKG.is_dir(), reason="TDMEC_INPUT_smoke_e2e not on disk")
def test_scaler_fit_transform_smoke():
    ds = TDMECInputDataset(SMOKE_PKG)
    scaler = RobustFeatureScaler()
    scaler.fit(
        ds._x_struct,
        ds._struct_mask,
        train_snapshot_ids=TRAIN_IDS,
        time_index_for=ds._snap_to_t,
    )
    assert scaler.is_fitted
    assert scaler.state is not None
    assert scaler.state.n_active_rows > 0
    assert scaler.state.median.shape == (C.F_STRUCT,)

    b = ds.get_snapshot(10)
    x_scaled = scaler.transform(b.x_struct, b.struct_active_mask, as_torch=True)
    assert isinstance(x_scaled, torch.Tensor)
    assert x_scaled.shape == (C.N_NODES, C.F_STRUCT)
    assert torch.isfinite(x_scaled).all()
    inactive = ~b.struct_active_mask
    assert torch.all(x_scaled[inactive] == 0.0)

    # no leakage: val snapshot uses frozen train stats
    b_val = ds.get_snapshot(25)
    x_val = scaler.transform(b_val.x_struct, b_val.struct_active_mask, as_torch=True)
    assert torch.isfinite(x_val).all()
    assert torch.all(x_val[~b_val.struct_active_mask] == 0.0)


def test_scaler_rejects_empty_train():
    x = np.zeros((2, 8, C.F_STRUCT), dtype=np.float32)
    m = np.zeros((2, 8), dtype=bool)
    with pytest.raises(ScalingError):
        RobustFeatureScaler().fit(x, m, train_snapshot_ids=[0, 1])


def test_encoder_g_shapes_finite():
    cfg = ModelConfig(d_h=64, f_struct=C.F_STRUCT)
    enc = NodeEncoderG(cfg)
    x = torch.randn(32, C.F_STRUCT)
    h0 = enc(x)
    assert h0.shape == (32, 64)
    assert torch.isfinite(h0).all()


@pytest.mark.skipif(not SMOKE_PKG.is_dir(), reason="TDMEC_INPUT_smoke_e2e not on disk")
def test_encoder_g_on_scaled_smoke_slice():
    ds = TDMECInputDataset(SMOKE_PKG)
    scaler = RobustFeatureScaler().fit(
        ds._x_struct,
        ds._struct_mask,
        train_snapshot_ids=TRAIN_IDS,
        time_index_for=ds._snap_to_t,
    )
    b = ds.get_snapshot(10)
    x_scaled = scaler.transform(b.x_struct, b.struct_active_mask, as_torch=True)
    enc = NodeEncoderG(ModelConfig())
    h0 = enc(x_scaled)
    assert h0.shape == (C.N_NODES, 64)
    assert torch.isfinite(h0).all()
