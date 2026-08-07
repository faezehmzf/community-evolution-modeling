"""Logical tensor schema descriptors (shapes only; no model ops)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from tdmec import constants as C
from tdmec.unresolved import ResolutionGate, UnresolvedValue


@dataclass(frozen=True)
class TensorSchema:
    name: str
    logical_shape: Tuple[Optional[int], ...]
    dtype: str
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "logical_shape": list(self.logical_shape),
            "logical_shape_symbols": [
                "T"
                if (i == 0 and d is None and self.name != "edge_text_embeddings")
                else (
                    "E"
                    if (i == 0 and d is None)
                    else ("D_text" if d is None else d)
                )
                for i, d in enumerate(self.logical_shape)
            ],
            "dtype": self.dtype,
            "notes": self.notes,
        }


def primary_tensor_schemas(
    *,
    n: int = C.N_NODES,
    f_struct: int = C.F_STRUCT,
    t: Optional[int] = None,
    d_text: Optional[int] = None,
    d_h: int = C.DEFAULT_D_H,
    k: int = C.DEFAULT_K,
) -> Dict[str, TensorSchema]:
    """Return logical primary tensor schemas.

    ``t`` and ``d_text`` may be None to mark unresolved dimensions symbolically.
    """
    return {
        "X_struct": TensorSchema(
            "X_struct", (t, n, f_struct), C.STRUCT_FEATURE_DTYPE, "Q-FEAT"
        ),
        "struct_active_mask": TensorSchema(
            "struct_active_mask", (t, n), "bool", "Q-FEAT"
        ),
        "X_node_text": TensorSchema(
            "X_node_text", (t, n, d_text), "float32", "Q-TEXT/Q-MISS; D_text unresolved"
        ),
        "node_text_available_mask": TensorSchema(
            "node_text_available_mask", (t, n), "bool", "Q-MISS"
        ),
        "node_valid_text_count": TensorSchema(
            "node_valid_text_count", (t, n), "int64", "metadata only"
        ),
        "edge_text_embeddings": TensorSchema(
            "edge_text_embeddings", (None, d_text), "float32", "sparse edge-aligned"
        ),
        "edge_text_available_mask": TensorSchema(
            "edge_text_available_mask", (None,), "bool", "edge-aligned"
        ),
        "edge_valid_text_count": TensorSchema(
            "edge_valid_text_count", (None,), "int64", "metadata only"
        ),
        "model_active_mask": TensorSchema(
            "model_active_mask", (t, n), "bool", "QACT-01"
        ),
        "h": TensorSchema("h", (n, d_h), "float32", "config-only dim; no forward"),
        "z": TensorSchema("z", (n, d_h), "float32", "config-only dim; no forward"),
        "s": TensorSchema("s", (n, d_h), "float32", "config-only dim; no forward"),
        "mu": TensorSchema("mu", (k, d_h), "float32", "config-only dim; no forward"),
        "Q": TensorSchema("Q", (n, k), "float32", "config-only dim; no forward"),
    }


D_TEXT_SYMBOLIC = UnresolvedValue[int](
    name="D_text",
    gate=ResolutionGate.POST_QEMB_PILOT,
    notes="Symbolic tensor dim until Q-EMB pilot",
)

T_SYMBOLIC = UnresolvedValue[int](
    name="T",
    gate=ResolutionGate.POST_DIAGNOSTIC,
    provisional=C.PROVISIONAL_SNAPSHOT_COUNT,
    notes="Symbolic calendar length until QCAL-B01",
)
