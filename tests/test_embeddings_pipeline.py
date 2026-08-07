"""Focused end-to-end mock pipeline gate and CLI tests.

Status: NOT_EXECUTED in the authoring Studio.
Labels: IMPLEMENTED_NOT_EXECUTED / TRANSFER_PENDING_VALIDATION.

Full mock pipeline execution against smoke artifacts is deferred to the target Studio.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tdmec_embeddings.config import EmbeddingConfigError, load_embedding_config
from tdmec_embeddings.file_cli import build_parser, main
from tdmec_embeddings.pipeline import EmbeddingPipelineError, run_embedding_pipeline
from tdmec_embeddings.config import (
    EmbeddingRunConfig,
    EncoderConfig,
    SamplingConfig,
    SourceConfig,
)


def test_mock_template_yaml_is_present() -> None:
    path = Path("configs/embeddings/mock_end_to_end.yaml")
    assert path.is_file()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["execution_mode"] == "mock"
    assert raw["force"] is False
    assert "IMPLEMENTED_NOT_EXECUTED" in path.read_text(encoding="utf-8")


def test_cli_parser_exposes_required_controls() -> None:
    parser = build_parser()
    action_dests = {action.dest for action in parser._actions}
    for name in (
        "config",
        "embedding_run_id",
        "output_root",
        "max_node_rows",
        "max_event_rows",
        "input_batch_size",
        "output_shard_size",
        "output_dimension",
        "dry_run",
        "resume",
        "force",
        "authorize_real_model",
        "authorize_bounded_pilot",
    ):
        assert name in action_dests


def test_pipeline_refuses_qwen_without_authorization(tmp_path: Path) -> None:
    config = EmbeddingRunConfig(
        schema_version="tdmec-embedding-run-config-v1",
        execution_mode="qwen_preflight",
        embedding_run_id="e1",
        output_root=str(tmp_path / "out"),
        node_source=SourceConfig(str(tmp_path / "b"), "b-run"),
        event_source=SourceConfig(str(tmp_path / "a"), "a-run"),
        input_batch_size=8,
        output_shard_size=8,
        max_node_rows=64,
        max_event_rows=64,
        resume=False,
        dry_run=False,
        encoder=EncoderConfig(
            backend="qwen3",
            model_name="Qwen/Qwen3-Embedding-4B",
            model_revision="abc123def4567890abc123def4567890abc123de",
            tokenizer_revision="abc123def4567890abc123def4567890abc123de",
            instruction="x",
            output_dimension=8,
            max_length=32,
            precision="auto",
            device="cuda:0",
            batch_size=2,
            max_oom_retries=1,
        ),
        sampling=SamplingConfig(strategy="deterministic_stratified_hash"),
    )
    with pytest.raises(EmbeddingPipelineError, match="authorization"):
        run_embedding_pipeline(config, authorize_real_model=False)


def test_cli_force_is_rejected(tmp_path: Path) -> None:
    cfg = {
        "schema_version": "tdmec-embedding-run-config-v1",
        "execution_mode": "mock",
        "embedding_run_id": "mock1",
        "output_root": str(tmp_path / "out"),
        "force": False,
        "resume": False,
        "dry_run": True,
        "input_batch_size": 8,
        "output_shard_size": 8,
        "max_node_rows": 8,
        "max_event_rows": 8,
        "node_source": {"artifact_root": str(tmp_path / "b"), "run_id": "b-run"},
        "event_source": {"artifact_root": str(tmp_path / "a"), "run_id": "a-run"},
        "encoder": {
            "backend": "mock",
            "model_name": "tdmec-deterministic-mock",
            "model_revision": "sha256-counter-v1",
            "tokenizer_revision": "not-applicable",
            "instruction": "",
            "output_dimension": 8,
            "max_length": 32,
            "precision": "fp32",
            "device": "cpu",
            "batch_size": 4,
            "max_oom_retries": 0,
            "local_files_only": True,
            "allow_cpu": True,
        },
        "sampling": {"strategy": "deterministic_prefix", "seed": 1},
        "pooling": {"final_normalization": "none"},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert main(["--config", str(path), "--force"]) == 2


STATUS = "NOT_EXECUTED"
