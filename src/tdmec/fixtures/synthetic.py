"""Small synthetic fixtures for Phase 1 invariant tests only.

No Dataset A/B, Drive, or private storage access.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from tdmec import constants as C
from tdmec.config.schemas import (
    ArtifactCertificationState,
    CalendarCertificationStatus,
    CertificationConfig,
    DatasetContractConfig,
    RelationConfig,
)
from tdmec.hashing import hash_edge_order, sha256_bytes
from tdmec.schemas.artifacts import (
    CanonicalEdgeArtifactSchema,
    CanonicalEdgeRecord,
    ManifestSchema,
    ShardRef,
    SnapshotRecord,
    SnapshotRegistrySchema,
)


# Tiny synthetic universe (not production N) for mask/tensor tests
SYN_N = 4
SYN_T = 3
SYN_D_TEXT = 8  # fixture-local only; not a finalized D_text


def valid_relation_map() -> Dict[str, int]:
    return dict(C.RELATION_TO_ID)


def invalid_reordered_relation_map() -> Dict[str, int]:
    # Same IDs but wrong assignment / order semantics
    return {"mention": 1, "retweet": 0, "reply": 2, "quote": 3}


def invalid_altered_relation_ids() -> Dict[str, int]:
    return {"mention": 0, "retweet": 1, "reply": 2, "quote": 99}


def valid_node_map_indices(*, n: int = C.N_NODES) -> List[int]:
    return list(range(n))


def valid_snapshot_registry() -> SnapshotRegistrySchema:
    snaps = (
        SnapshotRecord(
            0, "2017-Q4", "2017-10-01T00:00:00+00:00", "2018-01-01T00:00:00+00:00",
            is_empty=False, status="provisional",
        ),
        SnapshotRecord(
            1, "2018-Q1", "2018-01-01T00:00:00+00:00", "2018-04-01T00:00:00+00:00",
            is_empty=True, status="provisional",  # internal empty
        ),
        SnapshotRecord(
            2, "2018-Q2", "2018-04-01T00:00:00+00:00", "2018-07-01T00:00:00+00:00",
            is_empty=False, status="provisional",
        ),
    )
    return SnapshotRegistrySchema(
        snapshots=snaps,
        calendar_certification_status="PROVISIONAL_DIAGNOSTIC_ONLY",
    )


def _edge(
    sid: int, rid: int, src: int, tgt: int, count: int
) -> CanonicalEdgeRecord:
    return CanonicalEdgeRecord(
        snapshot_id=sid,
        relation_id=rid,
        source_idx=src,
        target_idx=tgt,
        count_raw=count,
        weight_log1p=math.log1p(count),
    )


def valid_edges() -> List[CanonicalEdgeRecord]:
    """Directed edges covering all relations, opposite directions, no self-loops."""
    return [
        _edge(0, 0, 0, 1, 3),   # mention
        _edge(0, 0, 1, 0, 1),   # opposite direction
        _edge(0, 1, 0, 2, 2),   # retweet
        _edge(0, 2, 2, 3, 5),   # reply
        _edge(0, 3, 3, 1, 4),   # quote
        _edge(2, 0, 0, 3, 1),   # non-empty later snapshot
        # snapshot 1 intentionally has no edges (internal empty)
    ]


def valid_edge_artifact() -> CanonicalEdgeArtifactSchema:
    edges = tuple(valid_edges())
    return CanonicalEdgeArtifactSchema(
        edges=edges,
        relation_map_hash=RelationConfig().mapping_hash(),
        edge_order_hash=hash_edge_order(e.key() for e in edges),
    )


def invalid_self_loop_edge() -> CanonicalEdgeRecord:
    return _edge(0, 0, 5, 5, 1)


def invalid_duplicate_edges() -> List[CanonicalEdgeRecord]:
    e = _edge(0, 0, 0, 1, 2)
    return [e, CanonicalEdgeRecord(0, 0, 0, 1, 9, math.log1p(9))]


def invalid_nonpositive_count() -> CanonicalEdgeRecord:
    return CanonicalEdgeRecord(0, 0, 0, 1, 0, 0.0)


def invalid_log1p_edge() -> CanonicalEdgeRecord:
    return CanonicalEdgeRecord(0, 0, 0, 1, 3, 1.0)  # wrong; log1p(3)≈1.386


@dataclass
class SyntheticActivityBundle:
    """Tiny [T,N] masks + embeddings for QACT/Q-MISS tests."""

    struct_active: np.ndarray
    node_text_available: np.ndarray
    edge_text_available_nodes: np.ndarray  # per-node view for negative test only
    model_active: np.ndarray
    x_struct: np.ndarray
    node_text: np.ndarray
    node_valid_text_count: np.ndarray


def valid_activity_bundle() -> SyntheticActivityBundle:
    """
    Node roles at t=0:
      0: structure-only active
      1: text-only active
      2: fully inactive
      3: both active
    """
    struct = np.zeros((SYN_T, SYN_N), dtype=bool)
    node_text = np.zeros((SYN_T, SYN_N), dtype=bool)
    edge_only = np.zeros((SYN_T, SYN_N), dtype=bool)

    struct[0, 0] = True
    node_text[0, 1] = True
    struct[0, 3] = True
    node_text[0, 3] = True
    # node 2 inactive
    # edge-only node would be node 2 with edge text — must NOT activate
    edge_only[0, 2] = True

    model = struct | node_text
    x = np.zeros((SYN_T, SYN_N, C.F_STRUCT), dtype=np.float32)
    # structure-active rows get nonzero feature 0
    x[struct, 0] = 1.0
    # inactive rows remain exact zero (already)

    emb = np.zeros((SYN_T, SYN_N, SYN_D_TEXT), dtype=np.float32)
    emb[node_text] = 0.1
    counts = node_text.astype(np.int64)

    return SyntheticActivityBundle(
        struct_active=struct,
        node_text_available=node_text,
        edge_text_available_nodes=edge_only,
        model_active=model,
        x_struct=x,
        node_text=emb,
        node_valid_text_count=counts,
    )


def invalid_nonzero_unavailable_node_text() -> Tuple[np.ndarray, np.ndarray]:
    emb = np.zeros((1, 2, SYN_D_TEXT), dtype=np.float32)
    mask = np.array([[True, False]], dtype=bool)
    emb[0, 1] = 0.5  # unavailable but nonzero
    return emb, mask


def valid_edge_text(
    edges: List[CanonicalEdgeRecord],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    e = len(edges)
    emb = np.zeros((e, SYN_D_TEXT), dtype=np.float32)
    mask = np.zeros((e,), dtype=bool)
    counts = np.zeros((e,), dtype=np.int64)
    # first edge available, second unavailable (exact zero)
    if e > 0:
        mask[0] = True
        emb[0] = 0.25
        counts[0] = 2
    order_hash = hash_edge_order(ed.key() for ed in edges)
    return emb, mask, counts, order_hash


def invalid_edge_text_order_hash(edges: List[CanonicalEdgeRecord]) -> str:
    # hash of reversed order
    rev = list(reversed(edges))
    return hash_edge_order(ed.key() for ed in rev)


def valid_manifest(*, artifact_bytes: bytes = b"phase1-synthetic-artifact") -> ManifestSchema:
    checksum = sha256_bytes(artifact_bytes)
    shard = ShardRef(
        shard_id="edges-000",
        relative_path="graph/edges/part-000.parquet",
        checksum_sha256=checksum,
        byte_size=len(artifact_bytes),
    )
    cfg = DatasetContractConfig()
    return ManifestSchema(
        artifact_type="canonical_edges",
        artifact_version="v1",
        logical_shapes={"edges": [len(valid_edges()), 6]},
        physical_shards=(shard,),
        dtypes={"count_raw": "int64", "weight_log1p": "float32"},
        ordering_rules={
            "edges": "snapshot_id,relation_id,source_idx,target_idx ascending",
            "nodes": "node_idx ascending 0..N-1",
        },
        checksums={"edges-000": checksum},
        config_hash=cfg.config_hash(),
        source_provenance={"synthetic": True, "dataset": "none"},
        certification_status=C.CERT_UNVALIDATED,
        unresolved_fields=(
            {"name": "D_text", "gate": "POST_QEMB_PILOT", "resolved": False},
            {"name": "certified_T", "gate": "POST_DIAGNOSTIC", "resolved": False},
        ),
        node_order_hash="synthetic",
    )


def incomplete_manifest_dict() -> Dict[str, Any]:
    return {
        "artifact_type": "canonical_edges",
        # missing checksums, shards, etc.
    }


def nan_array() -> np.ndarray:
    a = np.zeros((2, 2), dtype=np.float32)
    a[0, 0] = np.nan
    return a


def inf_array() -> np.ndarray:
    a = np.zeros((2, 2), dtype=np.float32)
    a[0, 0] = np.inf
    return a
