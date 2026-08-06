"""Focused stratified sampling and config-gate tests.

Status: NOT_EXECUTED in the authoring Studio.
Labels: IMPLEMENTED_NOT_EXECUTED / TRANSFER_PENDING_VALIDATION.
"""
from __future__ import annotations

import pytest

from tdmec.hashing import hash_canonical
from tdmec_embeddings.config import (
    EmbeddingConfigError,
    EmbeddingRunConfig,
    EncoderConfig,
    SamplingConfig,
    SourceConfig,
)
from tdmec_embeddings.eligibility import EligibleTextUnit, cleaned_text_content_hash
from tdmec_embeddings.sampling import DeterministicStratifiedSampler, activity_counts


def _unit(i: int, *, snapshot: int, node: int, text: str) -> EligibleTextUnit:
    return EligibleTextUnit(
        modality="node_text",
        source_run_id="b",
        unit_id=f"u{i}",
        unit_hash=hash_canonical({"u": i}),
        content_hash=cleaned_text_content_hash(text),
        preprocessing_hash=hash_canonical({"p": 1}),
        cleaned_text=text,
        snapshot_id=snapshot,
        node_index=node,
        relation_id=None,
        source_idx=None,
        target_idx=None,
        source_file="private.xlsx",
        source_row_number=i,
    )


def test_stratified_sampler_is_deterministic_and_bounded() -> None:
    units = [
        _unit(i, snapshot=i % 3, node=i % 5, text=("x" * (10 + (i % 40))))
        for i in range(40)
    ]
    activity = activity_counts(units, "node_text")
    cfg = SamplingConfig(strategy="deterministic_stratified_hash", seed=7)
    first = DeterministicStratifiedSampler(
        modality="node_text", limit=10, config=cfg, activity=activity
    )
    second = DeterministicStratifiedSampler(
        modality="node_text", limit=10, config=cfg, activity=activity
    )
    first.extend(units)
    second.extend(units)
    left = first.finish()
    right = second.finish()
    assert len(left.selected_units) == 10
    assert [u.unit_hash for u in left.selected_units] == [
        u.unit_hash for u in right.selected_units
    ]
    report = left.report()
    assert "cleaned_text" not in str(report)
    assert report["seed"] == 7
    assert report["eligible_population"] == 40


def test_preflight_cap_and_force_policy() -> None:
    encoder = EncoderConfig(
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
    )
    with pytest.raises(EmbeddingConfigError):
        EmbeddingRunConfig(
            schema_version="tdmec-embedding-run-config-v1",
            execution_mode="qwen_preflight",
            embedding_run_id="e1",
            output_root="/tmp/out",
            node_source=SourceConfig("/tmp/b", "b"),
            event_source=SourceConfig("/tmp/a", "a"),
            input_batch_size=8,
            output_shard_size=8,
            max_node_rows=65,
            max_event_rows=64,
            resume=False,
            dry_run=False,
            encoder=encoder,
            sampling=SamplingConfig(strategy="deterministic_stratified_hash"),
            force=False,
        ).validate()
    with pytest.raises(EmbeddingConfigError, match="force"):
        EmbeddingRunConfig(
            schema_version="tdmec-embedding-run-config-v1",
            execution_mode="qwen_preflight",
            embedding_run_id="e1",
            output_root="/tmp/out",
            node_source=SourceConfig("/tmp/b", "b"),
            event_source=SourceConfig("/tmp/a", "a"),
            input_batch_size=8,
            output_shard_size=8,
            max_node_rows=64,
            max_event_rows=64,
            resume=False,
            dry_run=False,
            encoder=encoder,
            sampling=SamplingConfig(strategy="deterministic_stratified_hash"),
            force=True,
        ).validate()


STATUS = "NOT_EXECUTED"
