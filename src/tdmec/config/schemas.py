"""Typed Phase 1 configuration contracts.

These are schemas/validators only — no model forward passes, trainers, or evaluators.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

from tdmec import constants as C
from tdmec.hashing import hash_config, hash_relation_mapping
from tdmec.unresolved import ResolutionGate, UnresolvedValue


class CalendarCertificationStatus(str, Enum):
    UNVALIDATED = "UNVALIDATED"
    PROVISIONAL_DIAGNOSTIC_ONLY = "PROVISIONAL_DIAGNOSTIC_ONLY"
    VALIDATED = "VALIDATED"
    CERTIFIED = "CERTIFIED"


class DedupCertificationStatus(str, Enum):
    UNVALIDATED = "UNVALIDATED"
    VALIDATED = "VALIDATED"
    CERTIFIED = "CERTIFIED"


class EmbeddingContractStatus(str, Enum):
    UNVALIDATED = "UNVALIDATED"
    PENDING_QEMB_PILOT = "PENDING_QEMB_PILOT"
    VALIDATED = "VALIDATED"
    CERTIFIED = "CERTIFIED"


class ArtifactCertificationState(str, Enum):
    UNVALIDATED = C.CERT_UNVALIDATED
    VALIDATED = C.CERT_VALIDATED
    CERTIFIED = C.CERT_CERTIFIED


def _positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return value


def _nonneg_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative int, got {value!r}")
    return value


@dataclass(frozen=True)
class NodeUniverseConfig:
    """D2 frozen node universe."""

    n_nodes: int = C.N_NODES
    index_min: int = C.NODE_INDEX_MIN
    index_max: int = C.NODE_INDEX_MAX
    dataset_b_may_introduce_nodes: bool = False
    order_is_fixed: bool = True

    def __post_init__(self) -> None:
        if self.n_nodes != C.N_NODES:
            raise ValueError(f"n_nodes must be {C.N_NODES}, got {self.n_nodes}")
        if self.index_min != C.NODE_INDEX_MIN or self.index_max != C.NODE_INDEX_MAX:
            raise ValueError("node index range must be 0..16735")
        if self.index_max - self.index_min + 1 != self.n_nodes:
            raise ValueError("n_nodes inconsistent with index range")
        if self.dataset_b_may_introduce_nodes:
            raise ValueError("Dataset B cannot introduce new nodes (D2)")
        if not self.order_is_fixed:
            raise ValueError("canonical node order must be fixed")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RelationConfig:
    """QREL-01 immutable relation mapping."""

    relation_to_id: Mapping[str, int] = field(
        default_factory=lambda: dict(C.RELATION_TO_ID)
    )
    relation_order: Tuple[str, ...] = C.RELATION_ORDER

    def __post_init__(self) -> None:
        # Freeze mapping against accidental mutation via normal config use.
        object.__setattr__(
            self, "relation_to_id", MappingProxyType(dict(self.relation_to_id))
        )
        object.__setattr__(self, "relation_order", tuple(self.relation_order))
        self.validate()

    def validate(self) -> None:
        expected = dict(C.RELATION_TO_ID)
        got = dict(self.relation_to_id)
        if set(got.keys()) != set(expected.keys()):
            missing = set(expected) - set(got)
            unknown = set(got) - set(expected)
            parts = []
            if missing:
                parts.append(f"missing relations: {sorted(missing)}")
            if unknown:
                parts.append(f"unknown relations: {sorted(unknown)}")
            raise ValueError("; ".join(parts))
        if len(got.values()) != len(set(got.values())):
            raise ValueError("duplicate relation IDs are forbidden")
        for name, rid in expected.items():
            if got[name] != rid:
                raise ValueError(
                    f"altered canonical relation ID for {name}: "
                    f"expected {rid}, got {got[name]}"
                )
        if tuple(self.relation_order) != C.RELATION_ORDER:
            raise ValueError(
                "reordered or altered relation_order is forbidden "
                f"(expected {C.RELATION_ORDER}, got {self.relation_order})"
            )
        # Order must match increasing IDs
        ordered_ids = [got[n] for n in self.relation_order]
        if ordered_ids != list(range(C.RELATION_COUNT)):
            raise ValueError("relation_order must match canonical IDs 0..3")

    def mapping_hash(self) -> str:
        return hash_relation_mapping(self.relation_to_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relation_to_id": dict(self.relation_to_id),
            "relation_order": list(self.relation_order),
            "relation_count": C.RELATION_COUNT,
        }


@dataclass(frozen=True)
class CalendarConfig:
    """Q-CAL / QCAL-B01-PROC calendar contract (bounds unresolved)."""

    frequency: str = C.SNAPSHOT_FREQUENCY
    boundary_convention: str = C.BOUNDARY_CONVENTION
    keep_internal_empty_snapshots: bool = True
    provisional_start_label: str = C.PROVISIONAL_CALENDAR_START_LABEL
    provisional_end_label: str = C.PROVISIONAL_CALENDAR_END_LABEL
    provisional_snapshot_count: int = C.PROVISIONAL_SNAPSHOT_COUNT
    certification_status: CalendarCertificationStatus = (
        CalendarCertificationStatus.PROVISIONAL_DIAGNOSTIC_ONLY
    )
    certified_start_label: UnresolvedValue[str] = field(
        default_factory=lambda: UnresolvedValue(
            name="certified_calendar_start",
            gate=ResolutionGate.REVIEW_REQUIRED_AFTER_REAL_DATA_DIAGNOSTICS,
            notes="QCAL-B01: final start requires user approval after coverage report",
        )
    )
    certified_end_label: UnresolvedValue[str] = field(
        default_factory=lambda: UnresolvedValue(
            name="certified_calendar_end",
            gate=ResolutionGate.REVIEW_REQUIRED_AFTER_REAL_DATA_DIAGNOSTICS,
            notes="QCAL-B01: final end requires user approval after coverage report",
        )
    )
    certified_T: UnresolvedValue[int] = field(
        default_factory=lambda: UnresolvedValue(
            name="certified_T",
            gate=ResolutionGate.REVIEW_REQUIRED_AFTER_REAL_DATA_DIAGNOSTICS,
            notes="Exact snapshot count T is POST_DIAGNOSTIC",
        )
    )
    leading_trailing_policy: UnresolvedValue[str] = field(
        default_factory=lambda: UnresolvedValue(
            name="leading_trailing_empty_policy",
            gate=ResolutionGate.REVIEW_REQUIRED_AFTER_REAL_DATA_DIAGNOSTICS,
            notes="QCAL-B01-PROC",
        )
    )

    def __post_init__(self) -> None:
        if self.frequency != C.SNAPSHOT_FREQUENCY:
            raise ValueError("only quarterly snapshots are canonical")
        if self.boundary_convention != C.BOUNDARY_CONVENTION:
            raise ValueError(
                f"boundary_convention must be {C.BOUNDARY_CONVENTION}"
            )
        if not self.keep_internal_empty_snapshots:
            raise ValueError("internal empty snapshots must be kept")
        # Phase 1: certified calendar bounds remain unresolved markers.
        # Claiming CERTIFIED here is always invalid until post-diagnostic approval.
        if self.certification_status == CalendarCertificationStatus.CERTIFIED:
            raise ValueError(
                "CalendarConfig cannot claim CERTIFIED status in Phase 1 "
                "(QCAL-B01-PROC); certified start/end/T remain unresolved"
            )
        if not isinstance(self.certified_start_label, UnresolvedValue):
            raise ValueError("certified_start_label must remain an UnresolvedValue")
        if not isinstance(self.certified_end_label, UnresolvedValue):
            raise ValueError("certified_end_label must remain an UnresolvedValue")
        if not isinstance(self.certified_T, UnresolvedValue):
            raise ValueError("certified_T must remain an UnresolvedValue")
        if self.certified_start_label.resolved() or self.certified_end_label.resolved():
            raise ValueError("certified calendar bounds must not be silently resolved")
        if self.certified_T.resolved():
            raise ValueError("certified_T must not be silently resolved")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frequency": self.frequency,
            "boundary_convention": self.boundary_convention,
            "keep_internal_empty_snapshots": self.keep_internal_empty_snapshots,
            "provisional_start_label": self.provisional_start_label,
            "provisional_end_label": self.provisional_end_label,
            "provisional_snapshot_count": self.provisional_snapshot_count,
            "certification_status": self.certification_status.value,
            "certified_start_label": self.certified_start_label.as_manifest_entry(),
            "certified_end_label": self.certified_end_label.as_manifest_entry(),
            "certified_T": self.certified_T.as_manifest_entry(),
            "leading_trailing_policy": self.leading_trailing_policy.as_manifest_entry(),
        }


@dataclass(frozen=True)
class EdgeArtifactConfig:
    """Q-WGT / QSELF-01 edge artifact contract."""

    directed: bool = True
    allow_self_loops: bool = False
    symmetrize: bool = False
    weight_transform: str = C.WEIGHT_TRANSFORM
    weight_atol: float = C.WEIGHT_LOG1P_ATOL
    weight_rtol: float = C.WEIGHT_LOG1P_RTOL
    required_fields: Tuple[str, ...] = (
        "snapshot_id",
        "relation_id",
        "source_idx",
        "target_idx",
        "count_raw",
        "weight_log1p",
    )

    def __post_init__(self) -> None:
        if not self.directed:
            raise ValueError("canonical edges must be directed")
        if self.allow_self_loops or self.symmetrize:
            raise ValueError("self-loops and symmetrization are forbidden")
        if self.weight_transform != "log1p":
            raise ValueError("canonical weight_transform must be log1p")
        object.__setattr__(self, "required_fields", tuple(self.required_fields))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StructuralFeatureConfig:
    """Q-FEAT structural feature schema."""

    f_struct: int = C.F_STRUCT
    feature_names: Tuple[str, ...] = C.STRUCT_FEATURE_NAMES
    schema_version: str = C.STRUCT_FEATURE_SCHEMA_VERSION
    dtype: str = C.STRUCT_FEATURE_DTYPE
    separate_struct_active_mask: bool = True
    overwrite_raw_with_training_scale: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        if self.f_struct != C.F_STRUCT:
            raise ValueError(f"f_struct must be {C.F_STRUCT}")
        if len(self.feature_names) != C.F_STRUCT:
            raise ValueError(
                f"wrong structural feature count: expected {C.F_STRUCT}, "
                f"got {len(self.feature_names)}"
            )
        if tuple(self.feature_names) != C.STRUCT_FEATURE_NAMES:
            raise ValueError(
                "structural feature names/order must match canonical Q-FEAT schema"
            )
        if self.dtype != C.STRUCT_FEATURE_DTYPE:
            raise ValueError(f"dtype must be {C.STRUCT_FEATURE_DTYPE}")
        if not self.separate_struct_active_mask:
            raise ValueError("struct_active_mask must be a separate artifact")
        if self.overwrite_raw_with_training_scale:
            raise ValueError(
                "raw canonical X_struct must not be overwritten by training scaling"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "f_struct": self.f_struct,
            "feature_names": list(self.feature_names),
            "schema_version": self.schema_version,
            "dtype": self.dtype,
            "separate_struct_active_mask": self.separate_struct_active_mask,
            "overwrite_raw_with_training_scale": self.overwrite_raw_with_training_scale,
        }


@dataclass(frozen=True)
class TextArtifactConfig:
    """Q-TEXT / Q-EMB text artifact dimensions (D_text unresolved)."""

    d_text: UnresolvedValue[int] = field(
        default_factory=lambda: UnresolvedValue(
            name=C.D_TEXT_UNRESOLVED_NAME,
            gate=ResolutionGate.POST_QEMB_PILOT,
            provisional=None,
            notes="D_text finalized only after Q-EMB pilot confirmation",
        )
    )
    node_text_required: bool = True
    edge_text_required: bool = True
    temporal_text_imputation: bool = False
    embedding_contract_status: EmbeddingContractStatus = (
        EmbeddingContractStatus.PENDING_QEMB_PILOT
    )

    def __post_init__(self) -> None:
        if self.temporal_text_imputation:
            raise ValueError("temporal text imputation is forbidden (Q-MISS)")
        if not isinstance(self.d_text, UnresolvedValue):
            raise ValueError("d_text must be an UnresolvedValue until Q-EMB pilot")
        if self.d_text.resolved():
            raise ValueError("Phase 1 must keep D_text unresolved")
        if self.d_text.name != C.D_TEXT_UNRESOLVED_NAME:
            raise ValueError("d_text unresolved marker name must be 'D_text'")
        # Reject accidental hard-coded final D_text via provisional misuse as final
        if self.embedding_contract_status == EmbeddingContractStatus.CERTIFIED:
            raise ValueError(
                "embedding contract cannot be CERTIFIED before Q-EMB pilot"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "d_text": self.d_text.as_manifest_entry(),
            "node_text_required": self.node_text_required,
            "edge_text_required": self.edge_text_required,
            "temporal_text_imputation": self.temporal_text_imputation,
            "embedding_contract_status": self.embedding_contract_status.value,
        }


@dataclass(frozen=True)
class MissingnessConfig:
    """Q-MISS M1 missing-text contract."""

    unavailable_vector_policy: str = "exact_zero"
    use_learned_missing_embedding: bool = False
    drop_nodes_for_missing_text: bool = False
    drop_edges_for_missing_text: bool = False
    carry_forward_text: bool = False
    counts_are_metadata_only: bool = True

    def __post_init__(self) -> None:
        if self.unavailable_vector_policy != "exact_zero":
            raise ValueError("unavailable text must use exact_zero vectors")
        if self.use_learned_missing_embedding:
            raise ValueError("learned missing embedding is forbidden (Q-MISS M1)")
        if self.drop_nodes_for_missing_text or self.drop_edges_for_missing_text:
            raise ValueError("must not drop nodes/edges solely for missing text")
        if self.carry_forward_text:
            raise ValueError("text carry-forward is forbidden")
        if not self.counts_are_metadata_only:
            raise ValueError("valid text counts are metadata, not model features")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActivityMaskConfig:
    """QACT-01 model_active_mask contract."""

    formula: str = "struct_active_mask OR node_text_available_mask"
    include_edge_text_in_model_active: bool = False
    redefine_struct_from_relation_availability: bool = False

    def __post_init__(self) -> None:
        if self.formula != "struct_active_mask OR node_text_available_mask":
            raise ValueError("model_active_mask formula must match QACT-01")
        if self.include_edge_text_in_model_active:
            raise ValueError(
                "edge_text_available_mask must not enter model_active_mask"
            )
        if self.redefine_struct_from_relation_availability:
            raise ValueError(
                "struct_active_mask must not be redefined from relation availability"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ManifestConfig:
    """Manifest metadata requirements (QART-01-FRAME)."""

    hash_algorithm: str = C.HASH_ALGORITHM
    method_contract_version: str = C.METHOD_CONTRACT_VERSION
    require_checksums: bool = True
    require_config_hash: bool = True
    require_provenance: bool = True
    require_deterministic_ordering_declaration: bool = True
    forbid_raw_private_text_in_manifest: bool = True

    def __post_init__(self) -> None:
        if self.hash_algorithm != C.HASH_ALGORITHM:
            raise ValueError(f"hash_algorithm must be {C.HASH_ALGORITHM}")
        if not (
            self.require_checksums
            and self.require_config_hash
            and self.require_provenance
            and self.require_deterministic_ordering_declaration
        ):
            raise ValueError("manifest hard requirements cannot be disabled")
        if not self.forbid_raw_private_text_in_manifest:
            raise ValueError("raw private text must not appear in manifests")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CertificationConfig:
    """Certification state machine + open evidence-dependent fields."""

    state: ArtifactCertificationState = ArtifactCertificationState.UNVALIDATED
    calendar_status: CalendarCertificationStatus = (
        CalendarCertificationStatus.PROVISIONAL_DIAGNOSTIC_ONLY
    )
    dedup_status: DedupCertificationStatus = DedupCertificationStatus.UNVALIDATED
    embedding_status: EmbeddingContractStatus = (
        EmbeddingContractStatus.PENDING_QEMB_PILOT
    )
    coverage_hard_thresholds: UnresolvedValue[Dict[str, float]] = field(
        default_factory=lambda: UnresolvedValue(
            name="numeric_coverage_hard_thresholds",
            gate=ResolutionGate.POST_DIAGNOSTIC,
            provisional=None,
            notes="Do not invent numeric coverage thresholds in Phase 1",
        )
    )
    allow_certified_by_existence_alone: bool = False
    # Declarative gates recorded with the certification record (not auto-filled).
    manifest_checksums_present: bool = False
    config_hash_present: bool = False
    validation_hard_failures_clear: bool = False

    def __post_init__(self) -> None:
        if self.allow_certified_by_existence_alone:
            raise ValueError(
                "a file must never become CERTIFIED merely because it exists"
            )
        if not isinstance(self.coverage_hard_thresholds, UnresolvedValue):
            raise ValueError("coverage_hard_thresholds must remain UnresolvedValue")
        if self.coverage_hard_thresholds.resolved():
            raise ValueError("coverage thresholds must not be silently resolved")
        if self.state == ArtifactCertificationState.CERTIFIED:
            self._assert_certified_dependencies()

    def _assert_certified_dependencies(self) -> None:
        if self.calendar_status != CalendarCertificationStatus.CERTIFIED:
            raise ValueError(
                "invalid certification: artifact CERTIFIED requires "
                "calendar_status=CERTIFIED"
            )
        if self.dedup_status != DedupCertificationStatus.CERTIFIED:
            raise ValueError(
                "invalid certification: artifact CERTIFIED requires "
                "dedup_status=CERTIFIED"
            )
        if self.embedding_status != EmbeddingContractStatus.CERTIFIED:
            raise ValueError(
                "invalid certification: artifact CERTIFIED requires "
                "embedding_status=CERTIFIED"
            )
        if not self.manifest_checksums_present:
            raise ValueError(
                "invalid certification: CERTIFIED requires manifest checksums present"
            )
        if not self.config_hash_present:
            raise ValueError(
                "invalid certification: CERTIFIED requires config_hash present"
            )
        if not self.validation_hard_failures_clear:
            raise ValueError(
                "invalid certification: CERTIFIED requires clear validation hard failures"
            )

    def can_transition_to(self, new_state: ArtifactCertificationState) -> bool:
        allowed = C.ALLOWED_CERTIFICATION_TRANSITIONS[self.state.value]
        return new_state.value in allowed

    def transition(self, new_state: ArtifactCertificationState) -> "CertificationConfig":
        if not self.can_transition_to(new_state):
            raise ValueError(
                f"invalid certification transition: {self.state.value} -> {new_state.value}"
            )
        return CertificationConfig(
            state=new_state,
            calendar_status=self.calendar_status,
            dedup_status=self.dedup_status,
            embedding_status=self.embedding_status,
            coverage_hard_thresholds=self.coverage_hard_thresholds,
            allow_certified_by_existence_alone=False,
            manifest_checksums_present=self.manifest_checksums_present,
            config_hash_present=self.config_hash_present,
            validation_hard_failures_clear=self.validation_hard_failures_clear,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "calendar_status": self.calendar_status.value,
            "dedup_status": self.dedup_status.value,
            "embedding_status": self.embedding_status.value,
            "coverage_hard_thresholds": self.coverage_hard_thresholds.as_manifest_entry(),
            "allow_certified_by_existence_alone": self.allow_certified_by_existence_alone,
            "manifest_checksums_present": self.manifest_checksums_present,
            "config_hash_present": self.config_hash_present,
            "validation_hard_failures_clear": self.validation_hard_failures_clear,
        }


@dataclass(frozen=True)
class ModelDimensionConfig:
    """Configuration-only model dimensions (no forward implementation)."""

    n_nodes: int = C.N_NODES
    n_relations: int = C.RELATION_COUNT
    f_struct: int = C.F_STRUCT
    d_h: int = C.DEFAULT_D_H
    d_rel: int = C.DEFAULT_D_REL
    d_sem: Optional[int] = None  # defaults to d_h when None
    K: int = C.DEFAULT_K
    L: int = C.DEFAULT_L_LAYERS
    fanout: Tuple[int, ...] = C.DEFAULT_FANOUT
    alpha_student_t: float = C.DEFAULT_ALPHA_STUDENT_T
    dropout: float = C.DEFAULT_DROPOUT
    d_text: UnresolvedValue[int] = field(
        default_factory=lambda: UnresolvedValue(
            name=C.D_TEXT_UNRESOLVED_NAME,
            gate=ResolutionGate.POST_QEMB_PILOT,
            notes="Model dims reference unresolved D_text until Q-EMB",
        )
    )
    T: UnresolvedValue[int] = field(
        default_factory=lambda: UnresolvedValue(
            name="T",
            gate=ResolutionGate.POST_DIAGNOSTIC,
            provisional=C.PROVISIONAL_SNAPSHOT_COUNT,
            notes="Provisional T=35 is diagnostic-only",
        )
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "fanout", tuple(self.fanout))
        if self.n_nodes != C.N_NODES:
            raise ValueError("n_nodes must equal frozen N")
        if self.n_relations != C.RELATION_COUNT:
            raise ValueError("n_relations must be 4")
        if self.f_struct != C.F_STRUCT:
            raise ValueError("f_struct must be 17")
        _positive_int("d_h", self.d_h)
        _positive_int("d_rel", self.d_rel)
        _positive_int("K", self.K)
        _positive_int("L", self.L)
        if self.d_sem is not None:
            _positive_int("d_sem", self.d_sem)
            # QPROJ-01 primary contract: d_sem = d_h
            if self.d_sem != self.d_h:
                raise ValueError("primary contract requires d_sem == d_h (QPROJ-01)")
        if any(f <= 0 for f in self.fanout):
            raise ValueError("fanout entries must be positive")
        if self.dropout < 0:
            raise ValueError("dropout must be non-negative")
        if not isinstance(self.d_text, UnresolvedValue):
            raise ValueError("d_text must remain an UnresolvedValue until Q-EMB")
        if self.d_text.resolved():
            raise ValueError("D_text must not be silently resolved in Phase 1")
        if not isinstance(self.T, UnresolvedValue):
            raise ValueError("T must remain an UnresolvedValue until calendar certification")
        if self.T.resolved():
            raise ValueError("T must not be silently resolved in Phase 1")

    def effective_d_sem(self) -> int:
        return self.d_h if self.d_sem is None else self.d_sem

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_nodes": self.n_nodes,
            "n_relations": self.n_relations,
            "f_struct": self.f_struct,
            "d_h": self.d_h,
            "d_rel": self.d_rel,
            "d_sem": self.effective_d_sem(),
            "K": self.K,
            "L": self.L,
            "fanout": list(self.fanout),
            "alpha_student_t": self.alpha_student_t,
            "dropout": self.dropout,
            "d_text": self.d_text.as_manifest_entry(),
            "T": self.T.as_manifest_entry(),
        }


@dataclass(frozen=True)
class TrainingDefaultConfig:
    """Configuration-only training defaults (QTR/QLOSS/QPHASE) — no trainer."""

    optimizer: str = "AdamW"
    lr: float = C.DEFAULT_ADAMW_LR
    weight_decay: float = C.DEFAULT_WEIGHT_DECAY
    grad_clip: float = C.DEFAULT_GRAD_CLIP
    patience: int = C.DEFAULT_PATIENCE
    bptt: int = C.DEFAULT_BPTT
    lambda_struct: float = C.DEFAULT_LAMBDA_STRUCT
    lambda_sem: float = C.DEFAULT_LAMBDA_SEM
    lambda_cluster: float = C.DEFAULT_LAMBDA_CLUSTER
    lambda_reg: float = C.DEFAULT_LAMBDA_REG
    lambda_temp: float = C.DEFAULT_LAMBDA_TEMP
    phase_pretrain_epochs: int = C.DEFAULT_PHASE_PRETRAIN_EPOCHS
    phase_joint_epochs: int = C.DEFAULT_PHASE_JOINT_EPOCHS
    phase_temp_epochs: int = C.DEFAULT_PHASE_TEMP_EPOCHS
    temp_ramp_epochs: int = C.DEFAULT_TEMP_RAMP_EPOCHS
    n_seeds: int = C.DEFAULT_N_SEEDS
    split_train_frac: UnresolvedValue[float] = field(
        default_factory=lambda: UnresolvedValue(
            name="split_train_frac",
            gate=ResolutionGate.POST_CAL,
            provisional=0.70,
            notes="Exact chronological split bounds POST_CAL",
        )
    )
    split_val_frac: UnresolvedValue[float] = field(
        default_factory=lambda: UnresolvedValue(
            name="split_val_frac",
            gate=ResolutionGate.POST_CAL,
            provisional=0.15,
            notes="Exact chronological split bounds POST_CAL",
        )
    )
    split_test_frac: UnresolvedValue[float] = field(
        default_factory=lambda: UnresolvedValue(
            name="split_test_frac",
            gate=ResolutionGate.POST_CAL,
            provisional=0.15,
            notes="Exact chronological split bounds POST_CAL",
        )
    )
    batch_size: UnresolvedValue[int] = field(
        default_factory=lambda: UnresolvedValue(
            name="hardware_batch_size",
            gate=ResolutionGate.POST_DIAGNOSTIC,
            notes="Hardware-specific batch size unresolved",
        )
    )
    use_amp: UnresolvedValue[bool] = field(
        default_factory=lambda: UnresolvedValue(
            name="amp_enabled",
            gate=ResolutionGate.POST_DIAGNOSTIC,
            notes="AMP/OOM behavior is hardware-specific",
        )
    )

    def __post_init__(self) -> None:
        if self.optimizer != "AdamW":
            raise ValueError("primary optimizer contract is AdamW")
        if self.lr <= 0 or self.weight_decay < 0 or self.grad_clip <= 0:
            raise ValueError("invalid optimizer hyperparameters")
        _positive_int("patience", self.patience)
        _positive_int("bptt", self.bptt)
        _positive_int("n_seeds", self.n_seeds)
        for name in (
            "phase_pretrain_epochs",
            "phase_joint_epochs",
            "phase_temp_epochs",
            "temp_ramp_epochs",
        ):
            _positive_int(name, getattr(self, name))

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "optimizer": self.optimizer,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "grad_clip": self.grad_clip,
            "patience": self.patience,
            "bptt": self.bptt,
            "lambda_struct": self.lambda_struct,
            "lambda_sem": self.lambda_sem,
            "lambda_cluster": self.lambda_cluster,
            "lambda_reg": self.lambda_reg,
            "lambda_temp": self.lambda_temp,
            "phase_pretrain_epochs": self.phase_pretrain_epochs,
            "phase_joint_epochs": self.phase_joint_epochs,
            "phase_temp_epochs": self.phase_temp_epochs,
            "temp_ramp_epochs": self.temp_ramp_epochs,
            "n_seeds": self.n_seeds,
            "split_train_frac": self.split_train_frac.as_manifest_entry(),
            "split_val_frac": self.split_val_frac.as_manifest_entry(),
            "split_test_frac": self.split_test_frac.as_manifest_entry(),
            "batch_size": self.batch_size.as_manifest_entry(),
            "use_amp": self.use_amp.as_manifest_entry(),
        }
        return d


@dataclass(frozen=True)
class EvaluationDefaultConfig:
    """Configuration-only evaluation defaults (docs/method/18) — no evaluator."""

    early_stop_on: str = "smoothed_val_loss"
    selection_primary: str = "val_relation_macro_AP"
    use_collapse_guards: bool = True
    use_tie_break: bool = True
    baselines_phase: int = 10
    ablations_phase: int = 11

    def __post_init__(self) -> None:
        if self.baselines_phase < 10 or self.ablations_phase < 11:
            raise ValueError(
                "baselines/ablations remain deferred to Phases 10–11"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetContractConfig:
    """Top-level dataset / method configuration aggregate."""

    node_universe: NodeUniverseConfig = field(default_factory=NodeUniverseConfig)
    relations: RelationConfig = field(default_factory=RelationConfig)
    calendar: CalendarConfig = field(default_factory=CalendarConfig)
    edges: EdgeArtifactConfig = field(default_factory=EdgeArtifactConfig)
    structural_features: StructuralFeatureConfig = field(
        default_factory=StructuralFeatureConfig
    )
    text: TextArtifactConfig = field(default_factory=TextArtifactConfig)
    missingness: MissingnessConfig = field(default_factory=MissingnessConfig)
    activity_mask: ActivityMaskConfig = field(default_factory=ActivityMaskConfig)
    manifest: ManifestConfig = field(default_factory=ManifestConfig)
    certification: CertificationConfig = field(default_factory=CertificationConfig)
    model_dims: ModelDimensionConfig = field(default_factory=ModelDimensionConfig)
    training: TrainingDefaultConfig = field(default_factory=TrainingDefaultConfig)
    evaluation: EvaluationDefaultConfig = field(
        default_factory=EvaluationDefaultConfig
    )

    def __post_init__(self) -> None:
        self.validate_dimension_consistency()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_universe": self.node_universe.to_dict(),
            "relations": self.relations.to_dict(),
            "calendar": self.calendar.to_dict(),
            "edges": self.edges.to_dict(),
            "structural_features": self.structural_features.to_dict(),
            "text": self.text.to_dict(),
            "missingness": self.missingness.to_dict(),
            "activity_mask": self.activity_mask.to_dict(),
            "manifest": self.manifest.to_dict(),
            "certification": self.certification.to_dict(),
            "model_dims": self.model_dims.to_dict(),
            "training": self.training.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "method_contract_version": C.METHOD_CONTRACT_VERSION,
        }

    def config_hash(self) -> str:
        return hash_config(self.to_dict())

    def validate_dimension_consistency(self) -> None:
        if self.model_dims.f_struct != self.structural_features.f_struct:
            raise ValueError("model_dims.f_struct != structural_features.f_struct")
        if self.model_dims.n_nodes != self.node_universe.n_nodes:
            raise ValueError("model_dims.n_nodes != node_universe.n_nodes")
        if self.model_dims.n_relations != len(self.relations.relation_order):
            raise ValueError("model_dims.n_relations inconsistent with RelationConfig")
        if self.model_dims.effective_d_sem() != self.model_dims.d_h:
            raise ValueError("dimension consistency requires d_sem == d_h")
