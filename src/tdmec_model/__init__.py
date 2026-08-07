"""TDMEC neural model package (Step 1a: TDMEC-G scaffolding).

Status labels for smoke work:
  PROVISIONAL_SMOKE_ONLY
  ENGINEERING_VALIDATION
  NOT_FOR_FINAL_THESIS_CONCLUSIONS
"""
from __future__ import annotations

__version__ = "0.1.0"

from .community import StudentTCommunityHead
from .dataset import TDMECInputDataset
from .decoder import StructuralDecoder
from .edge_modules import EdgeContextFull, EdgeContextG, EdgeGate
from .encoders import NodeEncoderFull, NodeEncoderG
from .fusion import MaskedRelationFusion
from .rgcn import DirectedRelationEncoder
from .scaling import RobustFeatureScaler
from .semantic import SemanticProjections
from .tdmec_full import TDMECFull
from .tdmec_g import TDMECG
from .temporal import TemporalGRU, model_active_mask_g
from .types import ModelConfig, SnapshotBatch

__all__ = [
    "TDMECInputDataset",
    "ModelConfig",
    "SnapshotBatch",
    "RobustFeatureScaler",
    "NodeEncoderG",
    "NodeEncoderFull",
    "EdgeContextG",
    "EdgeContextFull",
    "EdgeGate",
    "DirectedRelationEncoder",
    "MaskedRelationFusion",
    "TemporalGRU",
    "model_active_mask_g",
    "StudentTCommunityHead",
    "StructuralDecoder",
    "SemanticProjections",
    "TDMECG",
    "TDMECFull",
    "__version__",
]
