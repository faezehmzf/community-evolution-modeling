"""Phase 1 invariant unit tests for TDMEC contracts/schemas.

Synthetic fixtures only — no Dataset A/B, Drive, embeddings, or model code.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tdmec import constants as C
from tdmec.config.schemas import (
    ArtifactCertificationState,
    CalendarCertificationStatus,
    CalendarConfig,
    CertificationConfig,
    DatasetContractConfig,
    RelationConfig,
    StructuralFeatureConfig,
)
from tdmec.fixtures import synthetic as fx
from tdmec.hashing import canonicalize, hash_canonical, hash_config, sha256_bytes
from tdmec.schemas.artifacts import (
    ManifestSchema,
    ShardRef,
    StructuralArtifactSchema,
)
from tdmec.validation.findings import Severity, ValidationFinding, ValidationReport
from tdmec.validation import validators as V


# 1
def test_canonical_relation_map_passes():
    report = V.validate_relation_map(fx.valid_relation_map())
    assert report.ok
    cfg = RelationConfig()
    assert cfg.mapping_hash() == RelationConfig(relation_to_id=dict(C.RELATION_TO_ID)).mapping_hash()


# 2
def test_reordered_relation_map_fails():
    report = V.validate_relation_map(fx.invalid_reordered_relation_map())
    assert not report.ok
    assert any(f.code == "RELATION_MAP" for f in report.hard_failures)
    with pytest.raises(ValueError):
        RelationConfig(relation_to_id=fx.invalid_reordered_relation_map())


# 3
def test_altered_relation_id_fails():
    report = V.validate_relation_map(fx.invalid_altered_relation_ids())
    assert not report.ok
    with pytest.raises(ValueError):
        RelationConfig(relation_to_id=fx.invalid_altered_relation_ids())


# 4
def test_node_index_0_passes():
    assert V.validate_node_index(0).ok


# 5
def test_node_index_16735_passes():
    assert V.validate_node_index(16_735).ok


# 6
def test_node_index_16736_fails():
    report = V.validate_node_index(16_736)
    assert not report.ok
    assert report.hard_failures[0].code == "NODE_INDEX_BOUNDS"


# 7
def test_self_loop_fails():
    report = V.validate_no_self_loops([fx.invalid_self_loop_edge()])
    assert not report.ok
    assert report.hard_failures[0].code == "SELF_LOOP"


# 8
def test_duplicate_edge_key_fails():
    report = V.validate_unique_edge_keys(fx.invalid_duplicate_edges())
    assert not report.ok
    assert report.hard_failures[0].code == "DUP_EDGE_KEY"


# 9
def test_valid_count_raw_log1p_passes():
    report = V.validate_count_raw_and_weight(fx.valid_edges())
    assert report.ok


# 10
def test_invalid_log1p_fails():
    report = V.validate_count_raw_and_weight([fx.invalid_log1p_edge()])
    assert not report.ok
    assert any(f.code == "WEIGHT_LOG1P" for f in report.hard_failures)


# 11
def test_17_feature_schema_passes():
    report = V.validate_feature_count(C.STRUCT_FEATURE_NAMES)
    assert report.ok
    StructuralFeatureConfig()  # must construct
    schema = StructuralArtifactSchema(
        logical_shape_x=(2, C.N_NODES, 17),
        logical_shape_mask=(2, C.N_NODES),
    )
    assert schema.compute_feature_order_hash()


# 12
def test_wrong_feature_count_fails():
    report = V.validate_feature_count(list(C.STRUCT_FEATURE_NAMES)[:-1])
    assert not report.ok
    assert report.hard_failures[0].code == "FEATURE_COUNT"
    with pytest.raises(ValueError):
        StructuralFeatureConfig(f_struct=16, feature_names=C.STRUCT_FEATURE_NAMES[:16])


# 13
def test_unavailable_node_text_exact_zero():
    bundle = fx.valid_activity_bundle()
    report = V.validate_exact_zero_when_unavailable(
        bundle.node_text, bundle.node_text_available, name="node_text"
    )
    assert report.ok


# 14
def test_nonzero_unavailable_node_text_fails():
    emb, mask = fx.invalid_nonzero_unavailable_node_text()
    report = V.validate_exact_zero_when_unavailable(emb, mask, name="node_text")
    assert not report.ok
    assert any(f.code == "EXACT_ZERO_UNAVAILABLE" for f in report.hard_failures)


# 15
def test_unavailable_edge_text_exact_zero():
    edges = fx.valid_edges()
    emb, mask, counts, order_hash = fx.valid_edge_text(edges)
    report = V.validate_edge_text_alignment(
        len(edges), emb, mask, counts,
        expected_edge_order_hash=order_hash,
        actual_edge_order_hash=order_hash,
    )
    assert report.ok
    assert np.all(emb[~mask] == 0)


# 16
def test_edge_text_order_mismatch_fails():
    edges = fx.valid_edges()
    emb, mask, counts, order_hash = fx.valid_edge_text(edges)
    bad = fx.invalid_edge_text_order_hash(edges)
    report = V.validate_edge_text_alignment(
        len(edges), emb, mask, counts,
        expected_edge_order_hash=order_hash,
        actual_edge_order_hash=bad,
    )
    assert not report.ok
    assert any(f.code == "EDGE_ORDER_HASH" for f in report.hard_failures)


# 17
def test_valid_model_active_mask_passes():
    b = fx.valid_activity_bundle()
    report = V.validate_model_active_mask(
        b.model_active, b.struct_active, b.node_text_available
    )
    assert report.ok


# 18
def test_edge_text_alone_does_not_activate_model_active():
    b = fx.valid_activity_bundle()
    # Incorrect mask that activates on edge text alone at node 2
    wrong = b.model_active.copy()
    wrong[0, 2] = True
    report = V.validate_model_active_mask(
        wrong,
        b.struct_active,
        b.node_text_available,
        edge_text_available=b.edge_text_available_nodes,
    )
    assert not report.ok
    codes = {f.code for f in report.hard_failures}
    assert "MODEL_ACTIVE" in codes or "MODEL_ACTIVE_EDGE_TEXT" in codes
    # Correct mask: node 2 stays inactive despite edge-only flag
    assert not b.model_active[0, 2]


# 19
def test_node_text_alone_activates_model_active():
    b = fx.valid_activity_bundle()
    assert not b.struct_active[0, 1]
    assert b.node_text_available[0, 1]
    assert b.model_active[0, 1]


# 20
def test_internal_empty_snapshot_representable():
    reg = fx.valid_snapshot_registry()
    empty = [s for s in reg.snapshots if s.is_empty]
    assert len(empty) == 1
    assert empty[0].snapshot_id == 1
    # no edges in snapshot 1
    assert all(e.snapshot_id != 1 for e in fx.valid_edges())


# 21
def test_provisional_calendar_cannot_claim_certified():
    with pytest.raises(ValueError, match="CERTIFIED"):
        CalendarConfig(certification_status=CalendarCertificationStatus.CERTIFIED)
    report = V.validate_provisional_calendar_not_certified(
        CalendarCertificationStatus.CERTIFIED
    )
    assert not report.ok


# 22
def test_manifest_checksum_passes():
    payload = b"phase1-synthetic-artifact"
    manifest = fx.valid_manifest(artifact_bytes=payload)
    file_checksums = {"edges-000": sha256_bytes(payload)}
    report = V.validate_manifest_checksums(manifest, file_checksums=file_checksums)
    assert report.ok


# 23
def test_modified_artifact_fails_checksum():
    payload = b"phase1-synthetic-artifact"
    manifest = fx.valid_manifest(artifact_bytes=payload)
    modified = sha256_bytes(payload + b"-tampered")
    report = V.validate_manifest_checksums(
        manifest, file_checksums={"edges-000": modified}
    )
    assert not report.ok
    assert any(f.code == "ARTIFACT_CHECKSUM" for f in report.hard_failures)


# 24
def test_configuration_hashing_deterministic():
    a = DatasetContractConfig().config_hash()
    b = DatasetContractConfig().config_hash()
    assert a == b
    assert len(a) == 64


# 25
def test_dictionary_ordering_does_not_alter_hash():
    d1 = {"b": 2, "a": {"y": 1, "x": 0}}
    d2 = {"a": {"x": 0, "y": 1}, "b": 2}
    assert hash_config(d1) == hash_config(d2)
    assert canonicalize(d1) == canonicalize(d2)


# 26
def test_hard_failures_and_warnings_distinct():
    report = ValidationReport()
    report.add(
        ValidationFinding("H1", "inv", "hard", Severity.HARD_FAILURE, {"x": 1})
    )
    report.add(
        ValidationFinding("W1", "inv", "warn", Severity.WARNING, {"x": 2})
    )
    assert len(report.hard_failures) == 1
    assert len(report.warnings) == 1
    assert not report.ok
    assert report.hard_failures[0].severity != report.warnings[0].severity


# 27
def test_validation_reports_do_not_expose_raw_private_text():
    with pytest.raises(ValueError, match="private"):
        ValidationFinding(
            "X", "privacy", "bad", Severity.WARNING, {"raw_text": "SECRET"}
        )
    report = ValidationReport()
    report.add(
        ValidationFinding("OK", "privacy", "safe", Severity.INFO, {"node_idx": 1})
    )
    assert V.validate_report_privacy(report).ok


# 28
def test_invalid_certification_transition_fails():
    report = V.validate_certification_transition(
        ArtifactCertificationState.UNVALIDATED,
        ArtifactCertificationState.CERTIFIED,
    )
    assert not report.ok
    assert report.hard_failures[0].code == "CERT_TRANSITION"
    # Skipping VALIDATED is forbidden; UNVALIDATED -> VALIDATED ok
    ok = V.validate_certification_transition(
        ArtifactCertificationState.UNVALIDATED,
        ArtifactCertificationState.VALIDATED,
    )
    assert ok.ok
    # Direct CERTIFIED construction with unresolved calendar/dedup fails
    with pytest.raises(ValueError):
        CertificationConfig(state=ArtifactCertificationState.CERTIFIED)


# 29
def test_missing_shard_fails():
    shards = fx.valid_manifest().physical_shards
    report = V.validate_shard_set(["edges-000", "edges-001"], shards)
    assert not report.ok
    assert any(f.code == "MISSING_SHARD" for f in report.hard_failures)


# 30
def test_duplicate_shard_fails():
    m = fx.valid_manifest()
    dup = m.physical_shards + m.physical_shards
    report = V.validate_shard_set(["edges-000"], dup)
    assert not report.ok
    assert any(f.code == "DUP_SHARD" for f in report.hard_failures)
    with pytest.raises(ValueError, match="duplicate shard"):
        ManifestSchema(
            artifact_type="x",
            artifact_version="v1",
            logical_shapes={},
            physical_shards=dup,
            dtypes={},
            ordering_rules={"nodes": "asc"},
            checksums={"edges-000": m.physical_shards[0].checksum_sha256},
            config_hash="abc",
            source_provenance={},
        )


# Extra focused tests -------------------------------------------------------

def test_nonpositive_count_raw_fails():
    report = V.validate_count_raw_and_weight([fx.invalid_nonpositive_count()])
    assert not report.ok
    assert any(f.code == "COUNT_RAW" for f in report.hard_failures)


def test_nan_inf_fail():
    assert not V.validate_finite_array(fx.nan_array(), name="x").ok
    assert not V.validate_finite_array(fx.inf_array(), name="x").ok


def test_wrong_node_order_fails():
    order = list(range(10))
    order[0], order[1] = order[1], order[0]
    report = V.validate_node_order(order, n_nodes=10)
    assert not report.ok


def test_wrong_snapshot_order_fails():
    report = V.validate_snapshot_ordering([2, 1, 0])
    assert not report.ok


def test_struct_inactive_exact_zero():
    b = fx.valid_activity_bundle()
    report = V.validate_structural_inactive_exact_zero(b.x_struct, b.struct_active)
    assert report.ok
    bad = b.x_struct.copy()
    bad[0, 2, 0] = 3.0  # inactive node nonzero
    assert not V.validate_structural_inactive_exact_zero(bad, b.struct_active).ok


def test_relation_map_immutable_proxy():
    cfg = RelationConfig()
    with pytest.raises(TypeError):
        cfg.relation_to_id["mention"] = 9  # type: ignore[index]


def test_dataset_contract_hash_stable_under_rebuild():
    h1 = DatasetContractConfig().config_hash()
    h2 = DatasetContractConfig().to_dict()
    # shuffle top-level insertion via rebuild
    rebuilt = {k: h2[k] for k in sorted(h2.keys(), reverse=True)}
    assert hash_config(rebuilt) == h1


def test_weight_log1p_math():
    e = fx.valid_edges()[0]
    assert math.isclose(e.weight_log1p, math.log1p(e.count_raw), abs_tol=1e-12)


def test_unsupported_schema_version():
    report = V.validate_schema_version("nope", ["v1", "qfeat-17-v1"])
    assert not report.ok


# --- Audit strengthening tests ---------------------------------------------

def test_feature_rename_fails():
    names = list(C.STRUCT_FEATURE_NAMES)
    names[0] = "mention_outdegree_renamed"
    report = V.validate_feature_count(names)
    assert not report.ok
    assert report.hard_failures[0].code == "FEATURE_ORDER"
    with pytest.raises(ValueError):
        StructuralFeatureConfig(feature_names=tuple(names))


def test_feature_reorder_fails():
    names = list(C.STRUCT_FEATURE_NAMES)
    names[0], names[1] = names[1], names[0]
    report = V.validate_feature_count(names)
    assert not report.ok
    assert report.hard_failures[0].code == "FEATURE_ORDER"


def test_feature_order_hash_deterministic():
    from tdmec.hashing import hash_feature_order

    a = hash_feature_order(C.STRUCT_FEATURE_NAMES)
    b = hash_feature_order(list(C.STRUCT_FEATURE_NAMES))
    assert a == b
    assert a != hash_feature_order(list(C.STRUCT_FEATURE_NAMES)[::-1])


def test_nested_determinism_enum_set_dataclass_numpy():
    from enum import Enum
    from tdmec.hashing import canonicalize, hash_canonical
    from tdmec.unresolved import ResolutionGate, UnresolvedValue

    class Color(str, Enum):
        RED = "red"

    payload_a = {
        "gate": ResolutionGate.POST_DIAGNOSTIC,
        "tags": {"b", "a"},
        "u": UnresolvedValue(name="T", gate=ResolutionGate.POST_DIAGNOSTIC, provisional=35),
        "color": Color.RED,
        "np_int": np.int64(3),
        "np_float": np.float64(1.5),
        "seq": (1, 2),
    }
    payload_b = {
        "np_float": np.float64(1.5),
        "seq": [1, 2],
        "color": Color.RED,
        "np_int": np.int64(3),
        "u": UnresolvedValue(name="T", gate=ResolutionGate.POST_DIAGNOSTIC, provisional=35),
        "tags": {"a", "b"},
        "gate": ResolutionGate.POST_DIAGNOSTIC,
    }
    assert hash_canonical(payload_a) == hash_canonical(payload_b)
    with pytest.raises(TypeError):
        canonicalize(np.zeros(2))


def test_path_rejected_from_scientific_hash():
    from pathlib import Path
    from tdmec.hashing import canonicalize, assert_no_absolute_paths

    with pytest.raises(ValueError):
        canonicalize(Path("/tmp/x"))
    with pytest.raises(ValueError):
        assert_no_absolute_paths({"p": "/home/ubuntu/secret.parquet"})


def test_privacy_rejects_external_id_and_email_and_path():
    with pytest.raises(ValueError):
        ValidationFinding(
            "X", "privacy", "bad", Severity.WARNING, {"author_account_id": "123"}
        )
    with pytest.raises(ValueError):
        ValidationFinding(
            "X", "privacy", "user@example.com leaked", Severity.WARNING, {"node_idx": 1}
        )
    with pytest.raises(ValueError):
        ValidationFinding(
            "X",
            "privacy",
            "safe message",
            Severity.WARNING,
            {"note": "/workspace/private/data.xlsx"},
        )


def test_negative_infinity_fails():
    a = np.zeros((2,), dtype=np.float32)
    a[0] = -np.inf
    assert not V.validate_finite_array(a, name="x").ok


def test_zero_sized_edge_structures_ok():
    report = V.validate_unique_edge_keys([])
    assert report.ok
    report = V.validate_no_self_loops([])
    assert report.ok
    emb = np.zeros((0, 4), dtype=np.float32)
    mask = np.zeros((0,), dtype=bool)
    assert V.validate_exact_zero_when_unavailable(emb, mask, name="edge").ok


def test_node_order_hash_mismatch_fails():
    from tdmec.hashing import hash_node_order

    order = list(range(5))
    good = hash_node_order(order)
    assert V.validate_node_order_hash(order, good, n_nodes=5).ok
    bad = hash_node_order(list(range(4)) + [0])
    report = V.validate_node_order_hash(order, bad, n_nodes=5)
    assert not report.ok
    assert any(f.code == "NODE_ORDER_HASH" for f in report.hard_failures)


def test_unresolved_values_not_silently_finalized():
    from tdmec.unresolved import UnresolvedValue

    cfg = DatasetContractConfig()
    assert isinstance(cfg.text.d_text, UnresolvedValue)
    assert not cfg.text.d_text.resolved()
    with pytest.raises(ValueError):
        cfg.text.d_text.require()
    assert cfg.model_dims.effective_d_sem() == 64
    assert cfg.model_dims.d_h == 64
    assert cfg.model_dims.d_rel == 16


def test_certification_gated_on_dependencies():
    from tdmec.config.schemas import (
        CalendarCertificationStatus,
        DedupCertificationStatus,
        EmbeddingContractStatus,
    )

    base = CertificationConfig(state=ArtifactCertificationState.VALIDATED)
    report = V.validate_certification_ready_for_certified(base)
    assert not report.ok
    assert any(f.code == "CERT_DEPENDENCY" for f in report.hard_failures)

    # Still blocked without checksum/config/validation gates even if statuses look set
    almost = CertificationConfig(
        state=ArtifactCertificationState.VALIDATED,
        calendar_status=CalendarCertificationStatus.CERTIFIED,
        dedup_status=DedupCertificationStatus.CERTIFIED,
        embedding_status=EmbeddingContractStatus.CERTIFIED,
        manifest_checksums_present=False,
        config_hash_present=True,
        validation_hard_failures_clear=True,
    )
    assert not V.validate_certification_ready_for_certified(almost).ok


def test_extra_and_missing_relations_fail():
    with pytest.raises(ValueError, match="unknown"):
        RelationConfig(
            relation_to_id={
                "mention": 0,
                "retweet": 1,
                "reply": 2,
                "quote": 3,
                "like": 4,
            }
        )
    with pytest.raises(ValueError, match="missing"):
        RelationConfig(
            relation_to_id={"mention": 0, "retweet": 1, "reply": 2}
        )


def test_d_sem_must_equal_d_h():
    from tdmec.config.schemas import ModelDimensionConfig

    with pytest.raises(ValueError, match="d_sem"):
        ModelDimensionConfig(d_sem=32)


def test_package_import_smoke():
    import tdmec

    assert tdmec.__name__ == "tdmec"
    assert hasattr(tdmec, "constants")


def test_canonical_17_feature_list_exact():
    expected = (
        "mention_out_degree",
        "mention_in_degree",
        "mention_out_strength_log1p",
        "mention_in_strength_log1p",
        "retweet_out_degree",
        "retweet_in_degree",
        "retweet_out_strength_log1p",
        "retweet_in_strength_log1p",
        "reply_out_degree",
        "reply_in_degree",
        "reply_out_strength_log1p",
        "reply_in_strength_log1p",
        "quote_out_degree",
        "quote_in_degree",
        "quote_out_strength_log1p",
        "quote_in_strength_log1p",
        "tweet_count_log1p",
    )
    assert C.STRUCT_FEATURE_NAMES == expected
    assert len(C.STRUCT_FEATURE_NAMES) == 17
