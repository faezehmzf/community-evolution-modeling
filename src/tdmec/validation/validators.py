"""Reusable Phase 1 validators for TDMEC contracts."""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from tdmec import constants as C
from tdmec.config.schemas import (
    ArtifactCertificationState,
    CalendarCertificationStatus,
    CertificationConfig,
    RelationConfig,
)
from tdmec.hashing import hash_edge_order
from tdmec.schemas.artifacts import (
    CanonicalEdgeRecord,
    ManifestSchema,
    ShardRef,
)
from tdmec.validation.findings import (
    Severity,
    ValidationFinding,
    ValidationReport,
)


def _hard(
    code: str, invariant: str, message: str, **context: Any
) -> ValidationFinding:
    return ValidationFinding(code, invariant, message, Severity.HARD_FAILURE, context)


def _warn(
    code: str, invariant: str, message: str, **context: Any
) -> ValidationFinding:
    return ValidationFinding(code, invariant, message, Severity.WARNING, context)


# ----- node indices / order -------------------------------------------------

def validate_node_index(idx: int, *, n_nodes: int = C.N_NODES) -> ValidationReport:
    report = ValidationReport()
    if not isinstance(idx, (int, np.integer)) or isinstance(idx, bool):
        report.add(_hard("NODE_INDEX_TYPE", "D2", f"node index must be int, got {type(idx)}"))
        return report
    idx = int(idx)
    if idx < 0 or idx >= n_nodes:
        report.add(
            _hard(
                "NODE_INDEX_BOUNDS",
                "D2",
                f"node index {idx} out of range [0, {n_nodes - 1}]",
                node_idx=idx,
                n_nodes=n_nodes,
            )
        )
    return report


def validate_node_order(
    node_indices: Sequence[int], *, n_nodes: int = C.N_NODES
) -> ValidationReport:
    report = ValidationReport()
    expected = list(range(n_nodes))
    got = list(node_indices)
    if got != expected:
        report.add(
            _hard(
                "NODE_ORDER",
                "D2",
                "canonical node order must be contiguous 0..N-1",
                n_nodes=n_nodes,
                got_len=len(got),
            )
        )
    return report


def validate_node_order_hash(
    node_indices: Sequence[int],
    expected_hash: str,
    *,
    n_nodes: int = C.N_NODES,
) -> ValidationReport:
    """Validate both order content and recorded node-order hash."""
    from tdmec.hashing import hash_node_order

    report = validate_node_order(node_indices, n_nodes=n_nodes)
    if report.hard_failures:
        return report
    actual = hash_node_order(node_indices)
    if actual != expected_hash:
        report.add(
            _hard(
                "NODE_ORDER_HASH",
                "D2",
                "node_order_hash mismatch for canonical node order",
                hash_prefix_expected=expected_hash[:12],
                hash_prefix_actual=actual[:12],
            )
        )
    return report


# ----- relations ------------------------------------------------------------

def validate_relation_map(relation_to_id: Mapping[str, int]) -> ValidationReport:
    report = ValidationReport()
    try:
        RelationConfig(relation_to_id=dict(relation_to_id))
    except ValueError as exc:
        report.add(_hard("RELATION_MAP", "QREL-01", str(exc)))
    return report


def validate_relation_id(relation_id: int) -> ValidationReport:
    report = ValidationReport()
    if relation_id < C.RELATION_ID_MIN or relation_id > C.RELATION_ID_MAX:
        report.add(
            _hard(
                "RELATION_ID_BOUNDS",
                "QREL-01",
                f"relation_id {relation_id} not in 0..3",
                relation_id=int(relation_id),
            )
        )
    return report


# ----- snapshots ------------------------------------------------------------

def validate_snapshot_ordering(snapshot_ids: Sequence[int]) -> ValidationReport:
    report = ValidationReport()
    ids = list(snapshot_ids)
    if ids != sorted(ids):
        report.add(
            _hard(
                "SNAPSHOT_ORDER",
                "Q-CAL",
                "snapshot_ids must be strictly ascending sorted order",
            )
        )
    if len(ids) != len(set(ids)):
        report.add(_hard("SNAPSHOT_DUP", "Q-CAL", "duplicate snapshot_id"))
    return report


# ----- edges ----------------------------------------------------------------

def validate_no_self_loops(edges: Sequence[CanonicalEdgeRecord]) -> ValidationReport:
    report = ValidationReport()
    n_self = sum(1 for e in edges if e.source_idx == e.target_idx)
    if n_self != 0:
        report.add(
            _hard(
                "SELF_LOOP",
                "QSELF-01",
                f"canonical self-loop count must be 0, found {n_self}",
                self_loop_count=n_self,
            )
        )
    return report


def validate_unique_edge_keys(
    edges: Sequence[CanonicalEdgeRecord],
) -> ValidationReport:
    report = ValidationReport()
    seen: Set[Tuple[int, int, int, int]] = set()
    for e in edges:
        k = e.key()
        if k in seen:
            report.add(
                _hard(
                    "DUP_EDGE_KEY",
                    "Q-WGT",
                    "duplicate canonical edge key "
                    "(snapshot_id, relation_id, source_idx, target_idx)",
                    snapshot_id=k[0],
                    relation_id=k[1],
                    source_idx=k[2],
                    target_idx=k[3],
                )
            )
            return report
        seen.add(k)
    return report


def validate_count_raw_and_weight(
    edges: Sequence[CanonicalEdgeRecord],
    *,
    atol: float = C.WEIGHT_LOG1P_ATOL,
    rtol: float = C.WEIGHT_LOG1P_RTOL,
) -> ValidationReport:
    report = ValidationReport()
    for e in edges:
        if not isinstance(e.count_raw, (int, np.integer)) or int(e.count_raw) <= 0:
            report.add(
                _hard(
                    "COUNT_RAW",
                    "Q-WGT",
                    "count_raw must be a positive integer for every existing edge",
                    snapshot_id=e.snapshot_id,
                    relation_id=e.relation_id,
                    source_idx=e.source_idx,
                    target_idx=e.target_idx,
                )
            )
            continue
        if not math.isfinite(float(e.weight_log1p)):
            report.add(
                _hard(
                    "WEIGHT_NONFINITE",
                    "Q-WGT",
                    "weight_log1p must be finite",
                    snapshot_id=e.snapshot_id,
                    relation_id=e.relation_id,
                )
            )
            continue
        expected = math.log1p(int(e.count_raw))
        if not math.isclose(float(e.weight_log1p), expected, rel_tol=rtol, abs_tol=atol):
            report.add(
                _hard(
                    "WEIGHT_LOG1P",
                    "Q-WGT",
                    "weight_log1p must equal ln(1+count_raw) within tolerance",
                    expected=expected,
                    got=float(e.weight_log1p),
                    atol=atol,
                    rtol=rtol,
                    snapshot_id=e.snapshot_id,
                    relation_id=e.relation_id,
                    source_idx=e.source_idx,
                    target_idx=e.target_idx,
                )
            )
        report.extend(validate_node_index(e.source_idx).findings)
        report.extend(validate_node_index(e.target_idx).findings)
        report.extend(validate_relation_id(e.relation_id).findings)
    return report


# ----- arrays / dtypes / nan ------------------------------------------------

def validate_finite_array(arr: np.ndarray, *, name: str) -> ValidationReport:
    report = ValidationReport()
    if not np.isfinite(arr).all():
        report.add(
            _hard(
                "NAN_INF",
                "QART-01-FRAME",
                f"{name} contains NaN or Inf",
                array=name,
            )
        )
    return report


def validate_shape(
    arr: np.ndarray, expected: Sequence[int], *, name: str
) -> ValidationReport:
    report = ValidationReport()
    if tuple(arr.shape) != tuple(expected):
        report.add(
            _hard(
                "SHAPE",
                "tensor-schema",
                f"{name} shape {arr.shape} != expected {tuple(expected)}",
                array=name,
                expected=list(expected),
                got=list(arr.shape),
            )
        )
    return report


def validate_dtype(arr: np.ndarray, expected: str, *, name: str) -> ValidationReport:
    report = ValidationReport()
    got = str(arr.dtype)
    # accept float32 / bool / int64 aliases
    ok = got == expected or (
        expected == "float32" and got == "float32"
    ) or (
        expected == "bool" and got in ("bool", "bool_")
    ) or (
        expected.startswith("int") and got.startswith("int")
    )
    if not ok:
        report.add(
            _hard(
                "DTYPE",
                "tensor-schema",
                f"{name} dtype {got} != expected {expected}",
                array=name,
                expected=expected,
                got=got,
            )
        )
    return report


def validate_structural_inactive_exact_zero(
    x_struct: np.ndarray,
    struct_active_mask: np.ndarray,
) -> ValidationReport:
    """Inactive rows must be exact zeros across all 17 features."""
    report = ValidationReport()
    report.extend(
        validate_shape(
            x_struct,
            (struct_active_mask.shape[0], struct_active_mask.shape[1], C.F_STRUCT),
            name="X_struct",
        ).findings
    )
    if report.hard_failures:
        return report
    inactive = ~struct_active_mask.astype(bool)
    if inactive.any():
        rows = x_struct[inactive]
        if not np.all(rows == 0):
            report.add(
                _hard(
                    "STRUCT_INACTIVE_ZERO",
                    "Q-FEAT",
                    "inactive structural rows must be exact zeros",
                )
            )
    report.extend(validate_finite_array(x_struct, name="X_struct").findings)
    return report


def validate_feature_count(feature_names: Sequence[str]) -> ValidationReport:
    report = ValidationReport()
    if len(feature_names) != C.F_STRUCT:
        report.add(
            _hard(
                "FEATURE_COUNT",
                "Q-FEAT",
                f"expected {C.F_STRUCT} features, got {len(feature_names)}",
            )
        )
    elif tuple(feature_names) != C.STRUCT_FEATURE_NAMES:
        report.add(
            _hard(
                "FEATURE_ORDER",
                "Q-FEAT",
                "structural feature names/order mismatch",
            )
        )
    return report


# ----- text missingness -----------------------------------------------------

def validate_exact_zero_when_unavailable(
    embeddings: np.ndarray,
    available_mask: np.ndarray,
    *,
    name: str,
) -> ValidationReport:
    report = ValidationReport()
    mask = available_mask.astype(bool)
    if embeddings.shape[:-1] != mask.shape:
        report.add(
            _hard(
                "MASK_ALIGN",
                "Q-MISS",
                f"{name} embedding/mask leading dims misaligned",
                emb_ndim=int(embeddings.ndim),
                mask_ndim=int(mask.ndim),
            )
        )
        return report
    # Zero-sized eligible structures are representable and valid.
    if embeddings.size == 0 and mask.size == 0:
        return report
    unavailable = ~mask
    if unavailable.any():
        vecs = embeddings[unavailable]
        # Exact equality required (not approximate): Q-MISS exact-zero contract.
        if not np.all(vecs == 0):
            report.add(
                _hard(
                    "EXACT_ZERO_UNAVAILABLE",
                    "Q-MISS",
                    f"{name}: unavailable embeddings must be exact zero",
                    n_unavailable=int(np.count_nonzero(unavailable)),
                )
            )
    report.extend(validate_finite_array(embeddings, name=name).findings)
    return report


def validate_edge_text_alignment(
    n_edges: int,
    embeddings: np.ndarray,
    mask: np.ndarray,
    counts: np.ndarray,
    *,
    expected_edge_order_hash: str,
    actual_edge_order_hash: str,
) -> ValidationReport:
    report = ValidationReport()
    if embeddings.shape[0] != n_edges or mask.shape[0] != n_edges or counts.shape[0] != n_edges:
        report.add(
            _hard(
                "EDGE_TEXT_ALIGN",
                "Q-TEXT/Q-MISS",
                "edge-text tensors must align to canonical edge count E",
                n_edges=n_edges,
                emb_e=int(embeddings.shape[0]),
                mask_e=int(mask.shape[0]),
                count_e=int(counts.shape[0]),
            )
        )
    if expected_edge_order_hash != actual_edge_order_hash:
        report.add(
            _hard(
                "EDGE_ORDER_HASH",
                "Q-TEXT",
                "edge-text edge_order_hash mismatch vs canonical edges",
            )
        )
    report.extend(
        validate_exact_zero_when_unavailable(
            embeddings, mask, name="edge_text"
        ).findings
    )
    return report


# ----- model active ---------------------------------------------------------

def validate_model_active_mask(
    model_active: np.ndarray,
    struct_active: np.ndarray,
    node_text_available: np.ndarray,
    edge_text_available: Optional[np.ndarray] = None,
) -> ValidationReport:
    report = ValidationReport()
    expected = np.logical_or(struct_active.astype(bool), node_text_available.astype(bool))
    if model_active.shape != expected.shape:
        report.add(
            _hard(
                "MODEL_ACTIVE_SHAPE",
                "QACT-01",
                "model_active_mask shape mismatch",
            )
        )
        return report
    if not np.array_equal(model_active.astype(bool), expected):
        report.add(
            _hard(
                "MODEL_ACTIVE",
                "QACT-01",
                "model_active_mask must equal struct_active OR node_text_available",
            )
        )
    # Edge text alone must not activate — informational check when edge mask given
    if edge_text_available is not None:
        # If somehow someone OR'd edge text into model_active, the equality check above fails.
        # Additionally warn if edge_text is True where both struct and node_text are False
        # and model_active is True (already covered). Keep an explicit invariant code:
        edge_only = (
            edge_text_available.astype(bool)
            & ~struct_active.astype(bool)
            & ~node_text_available.astype(bool)
        )
        if edge_only.any() and model_active.astype(bool)[edge_only].any():
            report.add(
                _hard(
                    "MODEL_ACTIVE_EDGE_TEXT",
                    "QACT-01",
                    "edge text alone must not make a node model-active",
                )
            )
    return report


# ----- manifests / checksums / shards ---------------------------------------

def validate_manifest_checksums(
    manifest: ManifestSchema,
    *,
    file_checksums: Optional[Mapping[str, str]] = None,
) -> ValidationReport:
    report = ValidationReport()
    if not manifest.checksums:
        report.add(_hard("MANIFEST_CHECKSUMS", "QART-01-FRAME", "checksums required"))
    for shard in manifest.physical_shards:
        recorded = manifest.checksums.get(shard.shard_id)
        if recorded is None:
            report.add(
                _hard(
                    "MISSING_SHARD_CHECKSUM",
                    "QART-01-FRAME",
                    f"missing checksum for shard {shard.shard_id}",
                    shard_id=shard.shard_id,
                )
            )
            continue
        if recorded != shard.checksum_sha256:
            report.add(
                _hard(
                    "CHECKSUM_MISMATCH",
                    "QART-01-FRAME",
                    f"manifest checksum mismatch for shard {shard.shard_id}",
                    shard_id=shard.shard_id,
                )
            )
        if file_checksums is not None and shard.shard_id in file_checksums:
            if file_checksums[shard.shard_id] != shard.checksum_sha256:
                report.add(
                    _hard(
                        "ARTIFACT_CHECKSUM",
                        "QART-01-FRAME",
                        f"modified artifact fails checksum for shard {shard.shard_id}",
                        shard_id=shard.shard_id,
                    )
                )
    return report


def validate_shard_set(
    expected_shard_ids: Sequence[str],
    present_shards: Sequence[ShardRef],
) -> ValidationReport:
    report = ValidationReport()
    present_ids = [s.shard_id for s in present_shards]
    if len(present_ids) != len(set(present_ids)):
        report.add(_hard("DUP_SHARD", "QART-01-FRAME", "duplicate shard identifiers"))
    missing = [s for s in expected_shard_ids if s not in present_ids]
    if missing:
        report.add(
            _hard(
                "MISSING_SHARD",
                "QART-01-FRAME",
                f"missing shards: {missing}",
                missing_shards=list(missing),
            )
        )
    return report


def validate_schema_version(
    version: str, supported: Sequence[str]
) -> ValidationReport:
    report = ValidationReport()
    if version not in supported:
        report.add(
            _hard(
                "UNSUPPORTED_SCHEMA",
                "QART-01-FRAME",
                f"unsupported schema version {version}",
                version=version,
                supported=list(supported),
            )
        )
    return report


def validate_certification_transition(
    current: ArtifactCertificationState,
    new: ArtifactCertificationState,
) -> ValidationReport:
    report = ValidationReport()
    cfg = CertificationConfig(state=current)
    if not cfg.can_transition_to(new):
        report.add(
            _hard(
                "CERT_TRANSITION",
                "QART-01-FRAME",
                f"invalid certification transition {current.value} -> {new.value}",
                current=current.value,
                new=new.value,
            )
        )
    return report


def validate_certification_ready_for_certified(
    cfg: CertificationConfig,
) -> ValidationReport:
    """Check dependency gates required before an artifact may be CERTIFIED."""
    report = ValidationReport()
    try:
        # Re-run constructor gates by attempting a CERTIFIED view
        CertificationConfig(
            state=ArtifactCertificationState.CERTIFIED,
            calendar_status=cfg.calendar_status,
            dedup_status=cfg.dedup_status,
            embedding_status=cfg.embedding_status,
            coverage_hard_thresholds=cfg.coverage_hard_thresholds,
            allow_certified_by_existence_alone=False,
            manifest_checksums_present=cfg.manifest_checksums_present,
            config_hash_present=cfg.config_hash_present,
            validation_hard_failures_clear=cfg.validation_hard_failures_clear,
        )
    except ValueError as exc:
        report.add(
            _hard(
                "CERT_DEPENDENCY",
                "QART-01-FRAME",
                str(exc),
            )
        )
    return report


def validate_provisional_calendar_not_certified(
    calendar_status: CalendarCertificationStatus | str,
) -> ValidationReport:
    report = ValidationReport()
    status = (
        calendar_status.value
        if isinstance(calendar_status, CalendarCertificationStatus)
        else calendar_status
    )
    if status == C.CERT_CERTIFIED or status == CalendarCertificationStatus.CERTIFIED.value:
        # Creating CalendarConfig with CERTIFIED already fails; this catches registry misuse
        report.add(
            _hard(
                "CAL_PROVISIONAL_CERT",
                "QCAL-B01-PROC",
                "provisional calendar cannot claim certified calendar status",
                status=status,
            )
        )
    return report


def validate_report_privacy(report: ValidationReport) -> ValidationReport:
    """Meta-check: ensure findings do not expose raw private text keys."""
    from tdmec.validation.findings import FORBIDDEN_CONTEXT_KEYS

    out = ValidationReport()
    for f in report.findings:
        if FORBIDDEN_CONTEXT_KEYS.intersection(f.context.keys()):
            out.add(
                _hard(
                    "PRIVACY_LEAK",
                    "Q-DEDUP/QART",
                    "validation report exposes private text fields",
                    finding_code=f.code,
                )
            )
    return out


def combine(*reports: ValidationReport) -> ValidationReport:
    out = ValidationReport()
    for r in reports:
        out.extend(r.findings)
    return out
