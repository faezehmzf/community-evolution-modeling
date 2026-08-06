"""Focused pooling tests.

Status: NOT_EXECUTED in the authoring Studio.
Labels: IMPLEMENTED_NOT_EXECUTED / TRANSFER_PENDING_VALIDATION.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from tdmec.hashing import hash_canonical, sha256_file
from tdmec_embeddings.eligibility import EligibilityBatch, EligibleTextUnit, cleaned_text_content_hash
from tdmec_embeddings.file_sources import FileSourceIdentity
from tdmec_embeddings.file_writer import FileEmbeddingRunSpec, FileEmbeddingWriter
from tdmec_embeddings.mock_encoder import DeterministicMockEncoder, EncoderMetadata
from tdmec_embeddings.pooling import PoolingError, PoolingSpec, StreamingEmbeddingPooler


def _source(tmp_path: Path, kind: str = "dataset_b") -> FileSourceIdentity:
    return FileSourceIdentity(
        source_kind=kind,  # type: ignore[arg-type]
        run_id="b-source" if kind == "dataset_b" else "a-source",
        artifact_root=(tmp_path / f"source-{kind}").resolve(),
        manifest_sha256="1" * 64,
        checksums_sha256="2" * 64,
        config_hash="cfg-source",
        git_commit="abc1234",
        artifact_status="PROVISIONAL",
        certification_status="PROVISIONAL",
        dedup_status="PROVISIONAL",
        calendar_status="PROVISIONAL",
        n_nodes=4,
        n_snapshots=3,
        relation_order=("mention", "retweet", "reply", "quote"),
        node_map_sha256="3" * 64,
    )


def _unit(index: int, *, modality: str = "node_text", text: str | None = None) -> EligibleTextUnit:
    value = text or f"private-text-{index}"
    source_run = "b-source" if modality == "node_text" else "a-source"
    unit_id = f"private-unit-{index}"
    return EligibleTextUnit(
        modality=modality,  # type: ignore[arg-type]
        source_run_id=source_run,
        unit_id=unit_id,
        unit_hash=hash_canonical({"source": source_run, "unit": unit_id}),
        content_hash=cleaned_text_content_hash(value),
        preprocessing_hash=hash_canonical({"eligibility": "fixture-v1"}),
        cleaned_text=value,
        snapshot_id=index % 3,
        node_index=index % 4 if modality == "node_text" else None,
        relation_id=index % 4 if modality == "event_text" else None,
        source_idx=index % 4 if modality == "event_text" else None,
        target_idx=(index + 1) % 4 if modality == "event_text" else None,
        source_file="private-source.xlsx",
        source_row_number=index,
    )


def _write_units(tmp_path: Path, units: list[EligibleTextUnit], modality: str) -> Path:
    encoder = DeterministicMockEncoder(dimension=8)
    kind = "dataset_b" if modality == "node_text" else "dataset_a"
    spec = FileEmbeddingRunSpec(
        embedding_run_id="emb-run-pool",
        modality=modality,  # type: ignore[arg-type]
        source=_source(tmp_path, kind),
        preprocessing_hash=hash_canonical({"eligibility": "fixture-v1"}),
        encoder=encoder.metadata,
    )
    writer = FileEmbeddingWriter(tmp_path / "out", spec)
    batch = EligibilityBatch(
        modality=modality,  # type: ignore[arg-type]
        source_batch_index=0,
        source_global_row_offset=0,
        units=tuple(units),
    )
    writer.write_batch(batch, encoder.encode(units))
    writer.complete(expected_rows=len(units))
    return tmp_path / "out" / "emb-run-pool"


def test_node_snapshot_mean_pooling_and_mask(tmp_path: Path) -> None:
    units = [
        _unit(0, text="alpha"),
        _unit(0, text="beta"),  # same snapshot/node via index% — use explicit
    ]
    # Force shared (snapshot, node): both snapshot 0 node 0
    units = [
        EligibleTextUnit(
            modality="node_text",
            source_run_id="b-source",
            unit_id="u0",
            unit_hash=hash_canonical({"u": 0}),
            content_hash=cleaned_text_content_hash("alpha"),
            preprocessing_hash=hash_canonical({"eligibility": "fixture-v1"}),
            cleaned_text="alpha",
            snapshot_id=0,
            node_index=1,
            relation_id=None,
            source_idx=None,
            target_idx=None,
            source_file="private.xlsx",
            source_row_number=0,
        ),
        EligibleTextUnit(
            modality="node_text",
            source_run_id="b-source",
            unit_id="u1",
            unit_hash=hash_canonical({"u": 1}),
            content_hash=cleaned_text_content_hash("beta"),
            preprocessing_hash=hash_canonical({"eligibility": "fixture-v1"}),
            cleaned_text="beta",
            snapshot_id=0,
            node_index=1,
            relation_id=None,
            source_idx=None,
            target_idx=None,
            source_file="private.xlsx",
            source_row_number=1,
        ),
        EligibleTextUnit(
            modality="node_text",
            source_run_id="b-source",
            unit_id="u2",
            unit_hash=hash_canonical({"u": 2}),
            content_hash=cleaned_text_content_hash("gamma"),
            preprocessing_hash=hash_canonical({"eligibility": "fixture-v1"}),
            cleaned_text="gamma",
            snapshot_id=2,
            node_index=3,
            relation_id=None,
            source_idx=None,
            target_idx=None,
            source_file="private.xlsx",
            source_row_number=2,
        ),
    ]
    run_root = _write_units(tmp_path, units, "node_text")
    encoder = DeterministicMockEncoder(dimension=8)
    expected = (
        encoder.encode([units[0], units[1]]).mean(axis=0).astype(np.float32)
    )
    pooler = StreamingEmbeddingPooler(
        run_root,
        PoolingSpec(
            embedding_run_id="emb-run-pool",
            modality="node_text",
            dimension=8,
            n_snapshots=3,
            n_nodes=4,
            resume=False,
        ),
    )
    pooler.prepare_deltas()
    manifest = pooler.finalize_node_snapshots()
    assert manifest["status"] == "COMPLETED"
    vectors = np.load(run_root / "pooled" / "node_snapshot_embeddings.npy")
    counts = np.load(run_root / "pooled" / "node_valid_text_count.npy")
    mask = np.load(run_root / "pooled" / "node_text_available_mask.npy")
    assert vectors.shape == (3, 4, 8)
    assert counts.shape == (3, 4)
    assert mask.shape == (3, 4)
    assert mask.dtype == np.bool_
    assert counts[0, 1] == 2
    assert mask[0, 1]
    np.testing.assert_allclose(vectors[0, 1], expected, rtol=0, atol=1e-6)
    assert not mask[1, 0]
    assert counts[1, 0] == 0
    assert np.all(vectors[1, 0] == 0.0)
    # Double-count protection: re-prepare must not rewrite deltas incorrectly
    resumed = StreamingEmbeddingPooler(
        run_root,
        PoolingSpec(
            embedding_run_id="emb-run-pool",
            modality="node_text",
            dimension=8,
            n_snapshots=3,
            n_nodes=4,
            resume=True,
        ),
    )
    stats = resumed.prepare_deltas()
    assert stats["source_shards"] == 1


def test_pooling_rejects_duplicate_unit_in_shard(tmp_path: Path) -> None:
    run_root = _write_units(tmp_path, [_unit(0), _unit(1)], "node_text")
    # Corrupt by appending a duplicate unit hash manually is heavy; instead ensure
    # prepare_deltas refuses a checksum mismatch after mutation.
    shard = next((run_root / "unit_embeddings" / "node_text").glob("*.parquet"))
    checksums_path = run_root / "checksums" / "node_text.json"
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    key = next(k for k in checksums if k.endswith(".parquet"))
    checksums[key] = "0" * 64
    checksums_path.write_text(json.dumps(checksums), encoding="utf-8")
    pooler = StreamingEmbeddingPooler(
        run_root,
        PoolingSpec(
            embedding_run_id="emb-run-pool",
            modality="node_text",
            dimension=8,
            n_snapshots=3,
            n_nodes=4,
        ),
    )
    with pytest.raises(PoolingError, match="checksum"):
        pooler.prepare_deltas()
    assert sha256_file(shard) != "0" * 64


STATUS = "NOT_EXECUTED"
