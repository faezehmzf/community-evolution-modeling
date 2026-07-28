"""Logical artifact schemas for Phase 1 (storage layout remains separate)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tdmec import constants as C
from tdmec.hashing import hash_canonical, hash_edge_order, hash_feature_order, hash_node_order


@dataclass(frozen=True)
class NodeMapSchema:
    """Logical node-map contract (D2). Private IDs must not appear in reports."""

    n_nodes: int = C.N_NODES
    index_min: int = C.NODE_INDEX_MIN
    index_max: int = C.NODE_INDEX_MAX
    mapping_version: str = "node-map-v1"
    external_id_field: str = "external_identifier"  # logical name only
    node_idx_field: str = "node_idx"
    canonical_order_rule: str = "sorted_integer_external_id_ascending"
    # Hash over node order indices 0..N-1 (content), not raw private IDs in reports
    node_order_hash: Optional[str] = None

    def __post_init__(self) -> None:
        if self.n_nodes != C.N_NODES:
            raise ValueError("NodeMapSchema.n_nodes must be 16736")
        if self.index_min != 0 or self.index_max != C.NODE_INDEX_MAX:
            raise ValueError("node index bounds must be 0..16735")

    def compute_order_hash(self, node_indices: Sequence[int]) -> str:
        if list(node_indices) != list(range(self.n_nodes)):
            raise ValueError("canonical node order must be contiguous 0..N-1")
        return hash_node_order(node_indices)

    def report_safe_dict(self) -> Dict[str, Any]:
        """Metadata safe for validation reports (no external identifiers)."""
        return {
            "n_nodes": self.n_nodes,
            "index_min": self.index_min,
            "index_max": self.index_max,
            "mapping_version": self.mapping_version,
            "node_idx_field": self.node_idx_field,
            "canonical_order_rule": self.canonical_order_rule,
            "node_order_hash": self.node_order_hash,
            # external identifier field name only — never values
            "external_id_field_name": self.external_id_field,
        }

    def to_dict(self) -> Dict[str, Any]:
        return self.report_safe_dict()


@dataclass(frozen=True)
class SnapshotRecord:
    snapshot_id: int
    quarter_label: str
    start_boundary_utc: str  # ISO-8601
    end_boundary_utc: str  # exclusive
    boundary_convention: str = C.BOUNDARY_CONVENTION
    is_empty: bool = False
    status: str = "provisional"  # provisional | certified

    def __post_init__(self) -> None:
        if self.snapshot_id < 0:
            raise ValueError("snapshot_id must be non-negative")
        if self.boundary_convention != C.BOUNDARY_CONVENTION:
            raise ValueError("invalid boundary convention")
        if self.status not in ("provisional", "certified"):
            raise ValueError(f"invalid snapshot status: {self.status}")


@dataclass(frozen=True)
class SnapshotRegistrySchema:
    """Quarterly snapshot registry (calendar bounds may be provisional)."""

    snapshots: Tuple[SnapshotRecord, ...]
    calendar_certification_status: str = "PROVISIONAL_DIAGNOSTIC_ONLY"
    frequency: str = C.SNAPSHOT_FREQUENCY

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshots", tuple(self.snapshots))
        if self.frequency != C.SNAPSHOT_FREQUENCY:
            raise ValueError("only quarterly frequency is canonical")
        ids = [s.snapshot_id for s in self.snapshots]
        if ids != sorted(ids):
            raise ValueError("snapshots must be ordered by ascending snapshot_id")
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate snapshot_id")
        if self.calendar_certification_status == C.CERT_CERTIFIED:
            if any(s.status != "certified" for s in self.snapshots):
                raise ValueError(
                    "provisional snapshots cannot claim certified calendar status"
                )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frequency": self.frequency,
            "calendar_certification_status": self.calendar_certification_status,
            "snapshots": [asdict(s) for s in self.snapshots],
        }


@dataclass(frozen=True)
class CanonicalEdgeRecord:
    """Canonical edge fields (Q-WGT). Optional metadata stays separate."""

    snapshot_id: int
    relation_id: int
    source_idx: int
    target_idx: int
    count_raw: int
    weight_log1p: float

    def key(self) -> Tuple[int, int, int, int]:
        return (
            self.snapshot_id,
            self.relation_id,
            self.source_idx,
            self.target_idx,
        )


@dataclass(frozen=True)
class CanonicalEdgeArtifactSchema:
    """Logical schema for directed multiplex edges."""

    edges: Tuple[CanonicalEdgeRecord, ...]
    directed: bool = True
    allow_self_loops: bool = False
    relation_map_hash: Optional[str] = None
    edge_order_hash: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "edges", tuple(self.edges))
        if not self.directed or self.allow_self_loops:
            raise ValueError("edges must be directed without self-loops")

    def compute_edge_order_hash(self) -> str:
        return hash_edge_order(e.key() for e in self.edges)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "directed": self.directed,
            "allow_self_loops": self.allow_self_loops,
            "relation_map_hash": self.relation_map_hash,
            "edge_order_hash": self.edge_order_hash,
            "n_edges": len(self.edges),
            # Do not embed private text; edge records are structural only
            "edges": [asdict(e) for e in self.edges],
        }


@dataclass(frozen=True)
class StructuralArtifactSchema:
    """Logical structural feature contract (physical layout separate)."""

    logical_shape_x: Tuple[int, int, int]  # [T, N, 17]
    logical_shape_mask: Tuple[int, int]  # [T, N]
    feature_names: Tuple[str, ...] = C.STRUCT_FEATURE_NAMES
    dtype: str = C.STRUCT_FEATURE_DTYPE
    schema_version: str = C.STRUCT_FEATURE_SCHEMA_VERSION
    feature_order_hash: Optional[str] = None
    node_order_hash: Optional[str] = None
    physical_layout: str = "unspecified"  # dense|sparse|sharded — not forced here

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(self, "logical_shape_x", tuple(self.logical_shape_x))
        object.__setattr__(self, "logical_shape_mask", tuple(self.logical_shape_mask))
        t, n, f = self.logical_shape_x
        if f != C.F_STRUCT:
            raise ValueError(f"X_struct last dim must be {C.F_STRUCT}, got {f}")
        if t < 0 or n < 1:
            raise ValueError("invalid T/N dimensions for structural artifact")
        # Production contract documents N=16736; fixtures may use smaller N when
        # physical_layout marks a synthetic logical schema explicitly.
        if n != C.N_NODES and self.physical_layout != "synthetic_fixture":
            raise ValueError(
                f"production structural schema N must be {C.N_NODES} "
                f"(got {n}); use physical_layout='synthetic_fixture' for tests"
            )
        if tuple(self.feature_names) != C.STRUCT_FEATURE_NAMES:
            raise ValueError("feature order must match Q-FEAT")
        if self.logical_shape_mask != (t, n):
            raise ValueError("struct_active_mask shape must be [T, N]")
        if self.dtype != C.STRUCT_FEATURE_DTYPE:
            raise ValueError("dtype must be float32")

    def compute_feature_order_hash(self) -> str:
        return hash_feature_order(self.feature_names)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "logical_shape_x": list(self.logical_shape_x),
            "logical_shape_mask": list(self.logical_shape_mask),
            "feature_names": list(self.feature_names),
            "dtype": self.dtype,
            "schema_version": self.schema_version,
            "feature_order_hash": self.feature_order_hash,
            "node_order_hash": self.node_order_hash,
            "physical_layout": self.physical_layout,
        }


@dataclass(frozen=True)
class NodeTextArtifactSchema:
    """Logical node-text embedding artifact (D_text may be fixture-local)."""

    logical_shape_embeddings: Tuple[int, int, int]  # [T, N, D_text]
    logical_shape_mask: Tuple[int, int]
    logical_shape_counts: Tuple[int, int]
    d_text_is_final: bool = False
    dtype: str = "float32"
    unavailable_policy: str = "exact_zero"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "logical_shape_embeddings", tuple(self.logical_shape_embeddings)
        )
        t, n, d = self.logical_shape_embeddings
        if d <= 0:
            raise ValueError("D_text dimension must be positive in a concrete artifact")
        if self.logical_shape_mask != (t, n) or self.logical_shape_counts != (t, n):
            raise ValueError("node text mask/count shapes must be [T, N]")
        if self.unavailable_policy != "exact_zero":
            raise ValueError("unavailable node text must be exact zero")
        # Phase 1 never marks D_text as final production value
        if self.d_text_is_final:
            raise ValueError("D_text must remain non-final until Q-EMB pilot")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "logical_shape_embeddings": list(self.logical_shape_embeddings),
            "logical_shape_mask": list(self.logical_shape_mask),
            "logical_shape_counts": list(self.logical_shape_counts),
            "d_text_is_final": self.d_text_is_final,
            "dtype": self.dtype,
            "unavailable_policy": self.unavailable_policy,
        }


@dataclass(frozen=True)
class EdgeTextArtifactSchema:
    """Edge-text embeddings aligned to canonical edge order."""

    n_edges: int
    d_text: int
    logical_shape_embeddings: Tuple[int, int]  # [E, D_text]
    logical_shape_mask: Tuple[int, ...]  # [E]
    logical_shape_counts: Tuple[int, ...]  # [E]
    edge_order_hash: str
    d_text_is_final: bool = False
    unavailable_policy: str = "exact_zero"
    dtype: str = "float32"

    def __post_init__(self) -> None:
        e, d = self.logical_shape_embeddings
        if e != self.n_edges or d != self.d_text:
            raise ValueError("edge-text embedding shape mismatch")
        if self.logical_shape_mask != (self.n_edges,) or self.logical_shape_counts != (
            self.n_edges,
        ):
            raise ValueError("edge-text mask/count must be shape [E]")
        if self.unavailable_policy != "exact_zero":
            raise ValueError("unavailable edge text must be exact zero")
        if self.d_text_is_final:
            raise ValueError("D_text must remain non-final until Q-EMB pilot")
        if not self.edge_order_hash:
            raise ValueError("edge_order_hash required for edge-text alignment")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_edges": self.n_edges,
            "d_text": self.d_text,
            "logical_shape_embeddings": list(self.logical_shape_embeddings),
            "logical_shape_mask": list(self.logical_shape_mask),
            "logical_shape_counts": list(self.logical_shape_counts),
            "edge_order_hash": self.edge_order_hash,
            "d_text_is_final": self.d_text_is_final,
            "unavailable_policy": self.unavailable_policy,
            "dtype": self.dtype,
        }


@dataclass(frozen=True)
class ModelActiveArtifactSchema:
    """Logical model_active_mask [T, N] (QACT-01)."""

    logical_shape: Tuple[int, int]
    formula: str = "struct_active_mask OR node_text_available_mask"

    def __post_init__(self) -> None:
        object.__setattr__(self, "logical_shape", tuple(self.logical_shape))
        if self.formula != "struct_active_mask OR node_text_available_mask":
            raise ValueError("model_active formula must match QACT-01")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "logical_shape": list(self.logical_shape),
            "formula": self.formula,
        }


@dataclass(frozen=True)
class ShardRef:
    """Physical shard reference using logical relative names only."""

    shard_id: str
    relative_path: str
    checksum_sha256: str
    byte_size: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.shard_id:
            raise ValueError("shard_id required")
        if self.relative_path.startswith("/") or ":\\" in self.relative_path:
            raise ValueError("shard paths must be relative logical names, not absolute")
        if len(self.checksum_sha256) != 64:
            raise ValueError("checksum_sha256 must be 64 hex chars")


@dataclass(frozen=True)
class ManifestSchema:
    """Artifact manifest (QART-01-FRAME)."""

    artifact_type: str
    artifact_version: str
    logical_shapes: Dict[str, List[int]]
    physical_shards: Tuple[ShardRef, ...]
    dtypes: Dict[str, str]
    ordering_rules: Dict[str, str]
    checksums: Dict[str, str]
    hash_algorithm: str = C.HASH_ALGORITHM
    config_hash: str = ""
    source_provenance: Dict[str, Any] = field(default_factory=dict)
    method_contract_version: str = C.METHOD_CONTRACT_VERSION
    certification_status: str = C.CERT_UNVALIDATED
    validation_report_location: str = "validation/report.json"
    unresolved_fields: Tuple[Dict[str, Any], ...] = ()
    calendar_certification_status: str = "PROVISIONAL_DIAGNOSTIC_ONLY"
    dedup_certification_status: str = C.CERT_UNVALIDATED
    embedding_contract_status: str = "PENDING_QEMB_PILOT"
    relation_mapping: Dict[str, int] = field(
        default_factory=lambda: dict(C.RELATION_TO_ID)
    )
    node_order_hash: str = ""
    deterministic_ordering_declared: bool = True
    warnings: Tuple[str, ...] = ()
    hard_failures: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "physical_shards", tuple(self.physical_shards))
        object.__setattr__(self, "unresolved_fields", tuple(self.unresolved_fields))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "hard_failures", tuple(self.hard_failures))
        if self.hash_algorithm != C.HASH_ALGORITHM:
            raise ValueError("unsupported hash algorithm")
        if not self.deterministic_ordering_declared:
            raise ValueError("deterministic ordering must be declared")
        if self.certification_status not in C.CERTIFICATION_STATES:
            raise ValueError(f"invalid certification_status: {self.certification_status}")
        # Existence alone is not certification
        if self.certification_status == C.CERT_CERTIFIED and self.hard_failures:
            raise ValueError("cannot be CERTIFIED with hard_failures present")
        if self.certification_status == C.CERT_CERTIFIED:
            if not self.checksums:
                raise ValueError("CERTIFIED manifest requires checksums")
            if not self.config_hash:
                raise ValueError("CERTIFIED manifest requires config_hash")
            if not self.node_order_hash:
                raise ValueError("CERTIFIED manifest requires node_order_hash")
        shard_ids = [s.shard_id for s in self.physical_shards]
        if len(shard_ids) != len(set(shard_ids)):
            raise ValueError("duplicate shard identifiers")
        if dict(self.relation_mapping) != dict(C.RELATION_TO_ID):
            raise ValueError("manifest relation_mapping must match QREL-01")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "artifact_version": self.artifact_version,
            "logical_shapes": self.logical_shapes,
            "physical_shards": [asdict(s) for s in self.physical_shards],
            "dtypes": self.dtypes,
            "ordering_rules": self.ordering_rules,
            "checksums": self.checksums,
            "hash_algorithm": self.hash_algorithm,
            "config_hash": self.config_hash,
            "source_provenance": self.source_provenance,
            "method_contract_version": self.method_contract_version,
            "certification_status": self.certification_status,
            "validation_report_location": self.validation_report_location,
            "unresolved_fields": list(self.unresolved_fields),
            "calendar_certification_status": self.calendar_certification_status,
            "dedup_certification_status": self.dedup_certification_status,
            "embedding_contract_status": self.embedding_contract_status,
            "relation_mapping": dict(self.relation_mapping),
            "node_order_hash": self.node_order_hash,
            "deterministic_ordering_declared": self.deterministic_ordering_declared,
            "warnings": list(self.warnings),
            "hard_failures": list(self.hard_failures),
        }

    def scientific_hash(self) -> str:
        payload = self.to_dict()
        # Exclude validation narrative lists from reproducibility content hash? Keep them.
        return hash_canonical(payload)
