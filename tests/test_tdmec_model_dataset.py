"""DataLoader gate tests against TDMEC_INPUT_smoke_e2e (if present)."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tdmec import constants as C
from tdmec_model.dataset import DatasetError, TDMECInputDataset

SMOKE_PKG = Path(
    "/teamspace/studios/this_studio/TDMEC_PROJECT_OUTPUTS/tdmec_input/TDMEC_INPUT_smoke_e2e"
)


@pytest.mark.skipif(not SMOKE_PKG.is_dir(), reason="TDMEC_INPUT_smoke_e2e not on disk")
def test_get_snapshot_10_shapes():
    ds = TDMECInputDataset(SMOKE_PKG)
    b = ds.get_snapshot(10)
    assert b.x_struct.shape == (C.N_NODES, 17)
    assert b.struct_active_mask.shape == (C.N_NODES,)
    assert b.edge_index.ndim == 2 and b.edge_index.shape[0] == 2
    assert b.relation_id.shape[0] == b.edge_index.shape[1]
    assert b.weight_log1p.dtype == torch.float32
    uniq = sorted(set(b.relation_id.tolist()))
    assert set(uniq).issubset({0, 1, 2, 3})
    if b.num_edges > 0:
        assert int(b.edge_index.min()) >= 0
        assert int(b.edge_index.max()) < C.N_NODES


@pytest.mark.skipif(not SMOKE_PKG.is_dir(), reason="TDMEC_INPUT_smoke_e2e not on disk")
def test_empty_snapshot_edges_ok():
    ds = TDMECInputDataset(SMOKE_PKG)
    # snapshot 0 has no edge partitions in smoke_a_pg_001
    b = ds.get_snapshot(0)
    assert b.x_struct.shape == (C.N_NODES, 17)
    assert b.num_edges == 0
    assert b.relation_id.numel() == 0


@pytest.mark.skipif(not SMOKE_PKG.is_dir(), reason="TDMEC_INPUT_smoke_e2e not on disk")
def test_unknown_snapshot_raises():
    ds = TDMECInputDataset(SMOKE_PKG)
    with pytest.raises(DatasetError):
        ds.get_snapshot(999)
