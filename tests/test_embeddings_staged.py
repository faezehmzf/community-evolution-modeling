from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tdmec.hashing import hash_canonical, sha256_file
from tdmec_embeddings.config import (
    EmbeddingRunConfig,
    EncoderConfig,
    SamplingConfig,
    SourceConfig,
)
from tdmec_embeddings.eligibility import (
    EligibilityBatch,
    EligibleTextUnit,
    cleaned_text_content_hash,
)
from tdmec_embeddings.file_sources import FileSourceIdentity
from tdmec_embeddings.file_writer import FileEmbeddingRunSpec, FileEmbeddingWriter
from tdmec_embeddings.mock_encoder import DeterministicMockEncoder
from tdmec_embeddings.model_preflight import (
    REQUIRED_HF_VERSIONS,
    preflight_compatibility_key,
    verify_preflight_report,
)
from tdmec_embeddings.pipeline import EmbeddingPipelineError, run_embedding_pipeline
from tdmec_embeddings.staged import assigned_batch_bounds


def _source(tmp_path: Path) -> FileSourceIdentity:
    return FileSourceIdentity(
        source_kind="dataset_b",
        run_id="b-source",
        artifact_root=(tmp_path / "source").resolve(),
        manifest_sha256="1" * 64,
        checksums_sha256="2" * 64,
        config_hash="cfg",
        git_commit="source-git",
        artifact_status="PROVISIONAL",
        certification_status="PROVISIONAL",
        dedup_status="PROVISIONAL",
        calendar_status="PROVISIONAL",
        n_nodes=4,
        n_snapshots=2,
        relation_order=("mention", "retweet", "reply", "quote"),
    )


def _unit(index: int) -> EligibleTextUnit:
    text = f"private-{index}"
    return EligibleTextUnit(
        modality="node_text",
        source_run_id="b-source",
        unit_id=f"u-{index}",
        unit_hash=hash_canonical({"unit": index}),
        content_hash=cleaned_text_content_hash(text),
        preprocessing_hash=hash_canonical({"p": 1}),
        cleaned_text=text,
        snapshot_id=0,
        node_index=index,
    )


def _batch(index: int) -> EligibilityBatch:
    return EligibilityBatch(
        modality="node_text",
        source_batch_index=index,
        source_global_row_offset=index,
        units=(_unit(index),),
    )


def _qwen_config(tmp_path: Path) -> EmbeddingRunConfig:
    return EmbeddingRunConfig(
        schema_version="tdmec-embedding-run-config-v1",
        execution_mode="qwen_preflight",
        embedding_run_id="preflight-test",
        output_root=str(tmp_path / "out"),
        node_source=SourceConfig(str(tmp_path / "b"), "b-run"),
        event_source=SourceConfig(str(tmp_path / "a"), "a-run"),
        input_batch_size=8,
        output_shard_size=8,
        max_node_rows=8,
        max_event_rows=8,
        resume=False,
        dry_run=False,
        encoder=EncoderConfig(
            backend="qwen3",
            model_name="Qwen/Qwen3-Embedding-4B",
            model_revision="5cf2132abc99cad020ac570b19d031efec650f2b",
            tokenizer_revision="5cf2132abc99cad020ac570b19d031efec650f2b",
            instruction="x",
            output_dimension=512,
            max_length=512,
            precision="fp16",
            device="cuda:0",
            batch_size=4,
            max_oom_retries=2,
            attn_implementation="sdpa",
            enable_provisional_mrl_truncation=True,
        ),
        sampling=SamplingConfig(strategy="deterministic_stratified_hash"),
    )


def test_assigned_batch_ranges_are_contiguous_complete_and_disjoint() -> None:
    covered = []
    previous_stop = 0
    for shard_id in range(6):
        start, stop = assigned_batch_bounds(20, shard_id, 6)
        assert start == previous_stop
        covered.extend(range(start, stop))
        previous_stop = stop
    assert covered == list(range(20))


def test_repair_quarantines_only_corrupt_incomplete_shard(tmp_path: Path) -> None:
    encoder = DeterministicMockEncoder(dimension=8)
    spec = FileEmbeddingRunSpec(
        embedding_run_id="repair-test",
        modality="node_text",
        source=_source(tmp_path),
        preprocessing_hash=hash_canonical({"p": 1}),
        encoder=encoder.metadata,
    )
    writer = FileEmbeddingWriter(tmp_path / "out", spec)
    first, second = _batch(0), _batch(1)
    first_commit = writer.write_batch(first, encoder.encode(first.units))
    second_commit = writer.write_batch(second, encoder.encode(second.units))
    first_path = writer.run_root / str(first_commit.shard_relative_path)
    second_path = writer.run_root / str(second_commit.shard_relative_path)
    first_digest = first_path.read_bytes()
    second_path.write_bytes(second_path.read_bytes() + b"corrupt")

    repaired = FileEmbeddingWriter(
        tmp_path / "out", spec, repair_corrupt_shards=True
    )
    assert repaired.is_batch_committed(first) is True
    assert repaired.is_batch_committed(second) is False
    assert first_path.read_bytes() == first_digest
    assert not second_path.exists()
    assert list((repaired.run_root / "quarantine").rglob("*.parquet"))


def test_preflight_report_gate_matches_encoder_stack_and_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _qwen_config(tmp_path)
    monkeypatch.setattr(
        "tdmec_embeddings.model_preflight.git_commit_sha", lambda: "abc123"
    )
    checks = {
        name: True
        for name in (
            "text_count_between_2_and_16",
            "shape_ok",
            "finite",
            "unit_norm",
            "evaluation_mode",
            "inference_mode_used",
            "fp16",
            "single_explicit_gpu",
            "no_meta_parameters",
            "no_cpu_or_disk_offload",
        )
    }
    report = {
        "status": "PASSED",
        "git_commit": "abc123",
        "preflight_compatibility_key": preflight_compatibility_key(config),
        "dependency_versions": dict(REQUIRED_HF_VERSIONS),
        "checks": checks,
    }
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    checksum_path = Path(str(path) + ".sha256.json")
    checksum_path.write_text(json.dumps({"sha256": sha256_file(path)}), encoding="utf-8")
    assert verify_preflight_report(path, config)["status"] == "PASSED"
    report["dependency_versions"]["transformers"] = "5.0.0"
    path.write_text(json.dumps(report), encoding="utf-8")
    checksum_path.write_text(json.dumps({"sha256": sha256_file(path)}), encoding="utf-8")
    with pytest.raises(Exception, match="untested"):
        verify_preflight_report(path, config)


def test_real_pipeline_refuses_to_start_without_passed_model_preflight(
    tmp_path: Path,
) -> None:
    config = _qwen_config(tmp_path)
    with pytest.raises(EmbeddingPipelineError, match="preflight report"):
        run_embedding_pipeline(config, authorize_real_model=True)
