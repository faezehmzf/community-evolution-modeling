"""Tests for TDMEC pilot sampling coverage, alignment helpers, and export layout."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tdmec.hashing import hash_canonical
from tdmec_embeddings.config import (
    EmbeddingConfigError,
    EncoderConfig,
    ExportConfig,
    SamplingConfig,
    ValidationConfig,
)
from tdmec_embeddings.eligibility import EligibleTextUnit, cleaned_text_content_hash
from tdmec_embeddings.export import export_tdmec_input_package
from tdmec_embeddings.sampling import DeterministicStratifiedSampler, SamplingError, activity_counts
from tdmec_embeddings.validation import validate_pooled_embeddings, validate_tdmec_input_package


def _event_unit(i: int, *, snapshot: int, relation: int, text: str) -> EligibleTextUnit:
    return EligibleTextUnit(
        modality="event_text",
        source_run_id="a",
        unit_id=f"e{i}",
        unit_hash=hash_canonical({"e": i}),
        content_hash=cleaned_text_content_hash(text),
        preprocessing_hash=hash_canonical({"p": 1}),
        cleaned_text=text,
        snapshot_id=snapshot,
        node_index=None,
        relation_id=relation,
        source_idx=i % 7,
        target_idx=(i + 1) % 7,
        source_file="private.xlsx",
        source_row_number=i,
    )


def test_force_relation_coverage_includes_all_population_relations() -> None:
    units = []
    for i in range(40):
        units.append(
            _event_unit(
                i,
                snapshot=i % 3,
                relation=i % 4,
                text=("x" * (10 + (i % 40))),
            )
        )
    activity = activity_counts(units, "event_text")
    cfg = SamplingConfig(
        strategy="deterministic_stratified_hash",
        seed=11,
        force_relation_coverage=True,
    )
    sampler = DeterministicStratifiedSampler(
        modality="event_text", limit=10, config=cfg, activity=activity
    )
    sampler.extend(units)
    result = sampler.finish()
    assert set(result.selected_by_relation) == {0, 1, 2, 3}
    report = result.report()
    assert report["relation_coverage"]["missing_from_sample"] == []
    assert "mention" in report["relation_coverage"]["relation_names_covered"]
    assert "quote" in report["relation_coverage"]["relation_names_covered"]


def test_force_relation_coverage_fails_when_limit_too_small() -> None:
    units = [
        _event_unit(i, snapshot=0, relation=i % 4, text="abcd")
        for i in range(8)
    ]
    activity = activity_counts(units, "event_text")
    cfg = SamplingConfig(
        strategy="deterministic_stratified_hash",
        seed=3,
        force_relation_coverage=True,
    )
    sampler = DeterministicStratifiedSampler(
        modality="event_text", limit=2, config=cfg, activity=activity
    )
    sampler.extend(units)
    with pytest.raises(SamplingError):
        sampler.finish()


def test_encoder_rejects_normalize_false() -> None:
    with pytest.raises(EmbeddingConfigError):
        EncoderConfig(
            backend="qwen3",
            model_name="Qwen/Qwen3-Embedding-4B",
            model_revision="deadbeefcafebabe000000000000000000000000",
            tokenizer_revision="deadbeefcafebabe000000000000000000000000",
            instruction="x",
            output_dimension=8,
            max_length=32,
            precision="fp32",
            device="cpu",
            batch_size=2,
            max_oom_retries=1,
            normalize=False,
        ).validate()


def test_validate_pooled_embeddings_mask_zero_contract(tmp_path: Path) -> None:
    t, n, d, e = 2, 3, 4, 5
    node = np.zeros((t, n, d), dtype=np.float32)
    node[0, 1] = np.asarray([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    node_mask = np.zeros((t, n), dtype=np.bool_)
    node_mask[0, 1] = True
    node_count = np.zeros((t, n), dtype=np.int64)
    node_count[0, 1] = 1
    edge = np.zeros((e, d), dtype=np.float32)
    edge_mask = np.zeros((e,), dtype=np.bool_)
    edge_count = np.zeros((e,), dtype=np.int64)
    root = tmp_path / "pooled"
    root.mkdir()
    np.save(root / "node_snapshot_embeddings.npy", node)
    np.save(root / "node_text_available_mask.npy", node_mask)
    np.save(root / "node_valid_text_count.npy", node_count)
    np.save(root / "canonical_edge_embeddings.npy", edge)
    np.save(root / "edge_text_available_mask.npy", edge_mask)
    np.save(root / "edge_valid_event_count.npy", edge_count)
    report = validate_pooled_embeddings(root, expected_dimension=d, expected_t=t, expected_n=n, expected_e=e)
    assert report["passed"] is True


def test_export_and_package_validation_roundtrip(tmp_path: Path) -> None:
    # Minimal fake graph + embedding run roots
    graph = tmp_path / "graph"
    run = tmp_path / "emb" / "run1"
    pooled = run / "pooled"
    reports = run / "reports"
    for path in (graph / "edges" / "snapshot=0" / "relation=0", pooled, reports):
        path.mkdir(parents=True)
    t, n, f, d, e = 2, 3, 17, 4, 2
    np.save(graph / "X_struct.npy", np.zeros((t, n, f), dtype=np.float32))
    np.save(graph / "struct_active_mask.npy", np.zeros((t, n), dtype=np.bool_))
    (graph / "struct_feature_names.json").write_text("[]", encoding="utf-8")
    (graph / "snapshot_calendar.json").write_text("{}", encoding="utf-8")
    (graph / "manifest.json").write_text(
        json.dumps({"run_id": "smoke_a_pg_001"}), encoding="utf-8"
    )
    (graph / "validation_report.json").write_text("{}", encoding="utf-8")
    (graph / "checksums.json").write_text("{}", encoding="utf-8")
    # tiny edge parquet not required if include_graph_edges copies tree (empty ok? copytree needs dir)
    (graph / "edges" / "snapshot=0" / "relation=0" / "part-0.parquet").write_bytes(b"PAR1")

    node = np.zeros((t, n, d), dtype=np.float32)
    node_mask = np.zeros((t, n), dtype=np.bool_)
    node_count = np.zeros((t, n), dtype=np.int64)
    edge = np.zeros((e, d), dtype=np.float32)
    edge_mask = np.zeros((e,), dtype=np.bool_)
    edge_count = np.zeros((e,), dtype=np.int64)
    np.save(pooled / "node_snapshot_embeddings.npy", node)
    np.save(pooled / "node_text_available_mask.npy", node_mask)
    np.save(pooled / "node_valid_text_count.npy", node_count)
    np.save(pooled / "canonical_edge_embeddings.npy", edge)
    np.save(pooled / "edge_text_available_mask.npy", edge_mask)
    np.save(pooled / "edge_valid_event_count.npy", edge_count)
    (run / "embedding_manifest.json").write_text(
        json.dumps({"status": "COMPLETED", "embedding_run_id": "run1"}),
        encoding="utf-8",
    )

    from tdmec_embeddings.config import (
        EmbeddingRunConfig,
        EncoderConfig,
        PoolingConfig,
        SamplingConfig,
        SourceConfig,
    )

    config = EmbeddingRunConfig(
        schema_version="tdmec-embedding-run-config-v1",
        execution_mode="mock",
        embedding_run_id="run1",
        output_root=str(tmp_path / "emb"),
        node_source=SourceConfig(artifact_root=str(tmp_path), run_id="smoke_b_pg_001"),
        event_source=SourceConfig(artifact_root=str(graph), run_id="smoke_a_pg_001"),
        input_batch_size=8,
        output_shard_size=8,
        max_node_rows=8,
        max_event_rows=8,
        resume=False,
        dry_run=False,
        encoder=EncoderConfig(
            backend="mock",
            model_name="mock",
            model_revision="n/a",
            tokenizer_revision="n/a",
            instruction="x",
            output_dimension=d,
            max_length=16,
            precision="fp32",
            device="cpu",
            batch_size=2,
            max_oom_retries=0,
        ),
        sampling=SamplingConfig(strategy="deterministic_prefix", seed=1),
        pooling=PoolingConfig(),
        export=ExportConfig(package_name="TDMEC_INPUT", include_graph_edges=True),
    )
    package = tmp_path / "TDMEC_INPUT_run1"
    result = export_tdmec_input_package(
        embedding_run_root=run,
        graph_artifact_root=graph,
        package_root=package,
        config=config,
        alignment_report={"passed": True, "compatibility_hash": "abc"},
    )
    assert Path(result["package_root"]).is_dir()
    report = validate_tdmec_input_package(package, expected_dimension=d)
    assert report["passed"] is True
