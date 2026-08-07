"""Focused tests for the deterministic mock encoder and file embedding writer."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from tdmec.hashing import hash_canonical
from tdmec_embeddings.eligibility import (
    EligibilityBatch,
    EligibleTextUnit,
    cleaned_text_content_hash,
)
from tdmec_embeddings.file_sources import FileSourceIdentity
from tdmec_embeddings.file_writer import (
    FileEmbeddingRunSpec,
    FileEmbeddingWriter,
    FileEmbeddingWriterError,
    VectorValidationError,
    WriterCompatibilityError,
)
from tdmec_embeddings.mock_encoder import (
    DeterministicMockEncoder,
    EncoderError,
)


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


def _preprocessing_hash() -> str:
    return hash_canonical({"eligibility": "fixture-v1"})


def _unit(
    index: int,
    *,
    modality: str = "node_text",
    text: str | None = None,
    duplicate_unit_hash: str | None = None,
) -> EligibleTextUnit:
    value = text or f"private-text-{index}"
    source_run = "b-source" if modality == "node_text" else "a-source"
    unit_id = f"private-unit-{index}"
    return EligibleTextUnit(
        modality=modality,  # type: ignore[arg-type]
        source_run_id=source_run,
        unit_id=unit_id,
        unit_hash=duplicate_unit_hash
        or hash_canonical({"source": source_run, "unit": unit_id}),
        content_hash=cleaned_text_content_hash(value),
        preprocessing_hash=_preprocessing_hash(),
        cleaned_text=value,
        snapshot_id=index % 3,
        node_index=index % 4 if modality == "node_text" else None,
        relation_id=index % 4 if modality == "event_text" else None,
        source_idx=index % 4 if modality == "event_text" else None,
        target_idx=(index + 1) % 4 if modality == "event_text" else None,
        source_file="private-source.xlsx",
        source_sheet="Sheet1" if modality == "node_text" else None,
        source_row_number=index + 2,
    )


def _batch(
    units: tuple[EligibleTextUnit, ...],
    *,
    modality: str = "node_text",
    index: int = 0,
    offset: int = 0,
) -> EligibilityBatch:
    return EligibilityBatch(
        modality=modality,  # type: ignore[arg-type]
        source_batch_index=index,
        source_global_row_offset=offset,
        units=units,
    )


def _spec(
    tmp_path: Path,
    encoder: DeterministicMockEncoder,
    *,
    modality: str = "node_text",
    run_id: str = "qemb-dev-test",
) -> FileEmbeddingRunSpec:
    kind = "dataset_b" if modality == "node_text" else "dataset_a"
    return FileEmbeddingRunSpec(
        embedding_run_id=run_id,
        modality=modality,  # type: ignore[arg-type]
        source=_source(tmp_path, kind),
        preprocessing_hash=_preprocessing_hash(),
        encoder=encoder.metadata,
    )


def test_mock_encoder_is_deterministic_normalized_and_batch_independent():
    units = (_unit(0), _unit(1), _unit(2, text="private-text-0"))
    encoder = DeterministicMockEncoder(dimension=13)
    together = encoder.encode(units)
    split = np.vstack((encoder.encode(units[:1]), encoder.encode(units[1:])))

    assert together.shape == (3, 13)
    assert together.dtype == np.float32
    assert np.array_equal(together, split)
    assert np.array_equal(together[0], together[2])
    assert not np.array_equal(together[0], together[1])
    assert np.allclose(np.linalg.norm(together, axis=1), 1.0, atol=1e-6)
    changed_instruction = DeterministicMockEncoder(
        dimension=13, instruction_hash="4" * 64
    )
    assert not np.array_equal(together, changed_instruction.encode(units))


def test_mock_encoder_rejects_content_hash_drift():
    unit = replace(_unit(0), content_hash="0" * 64)
    with pytest.raises(EncoderError, match="content hash drifted"):
        DeterministicMockEncoder().encode((unit,))


def test_writer_commits_atomic_private_unit_shard_and_metadata(tmp_path: Path):
    encoder = DeterministicMockEncoder(dimension=8)
    spec = _spec(tmp_path, encoder)
    batch = _batch((_unit(0), _unit(1)))
    writer = FileEmbeddingWriter(tmp_path / "embeddings", spec)
    commit = writer.write_batch(batch, encoder.encode(batch.units))

    assert commit.status == "COMMITTED"
    assert commit.row_count == 2
    assert commit.total_committed_rows == 2
    assert commit.shard_relative_path is not None
    shard = writer.run_root / commit.shard_relative_path
    table = pq.read_table(shard)
    assert table.num_rows == 2
    assert "cleaned_text" not in table.column_names
    assert table.schema.field("embedding").type.list_size == 8
    assert table["unit_id"].to_pylist() == [
        "private-unit-0",
        "private-unit-1",
    ]
    stored = np.asarray(table["embedding"].to_pylist(), dtype=np.float32)
    assert np.array_equal(stored, encoder.encode(batch.units))

    manifest = json.loads(writer.manifest_path.read_text(encoding="utf-8"))
    checksums = json.loads(writer.checksums_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "IN_PROGRESS"
    assert manifest["accounting"] == {
        "committed_batches": 1,
        "committed_rows": 2,
        "parquet_shards": 1,
    }
    assert "PROVISIONAL_SMOKE_ONLY" in manifest["status_labels"]
    serialized = json.dumps(manifest, sort_keys=True)
    assert "private-text" not in serialized
    assert "private-unit" not in serialized
    assert "private-source.xlsx" not in serialized
    assert commit.shard_relative_path in checksums
    assert not list(writer.run_root.rglob("*.tmp"))


def test_writer_resume_is_idempotent_and_rejects_changed_replay(tmp_path: Path):
    encoder = DeterministicMockEncoder(dimension=8)
    spec = _spec(tmp_path, encoder)
    batch = _batch((_unit(0), _unit(1)))
    vectors = encoder.encode(batch.units)
    first = FileEmbeddingWriter(tmp_path / "embeddings", spec)
    first.write_batch(batch, vectors)

    resumed = FileEmbeddingWriter(tmp_path / "embeddings", spec)
    replay = resumed.write_batch(batch, vectors.copy())
    assert replay.status == "SKIPPED_ALREADY_COMMITTED"
    assert resumed.committed_rows == 2
    assert len(list(resumed.shard_dir.glob("*.parquet"))) == 1

    changed = vectors.copy()
    changed[0] = changed[1]
    with pytest.raises(WriterCompatibilityError, match="different content or vectors"):
        resumed.write_batch(batch, changed)


def test_writer_rejects_bad_vectors_duplicate_units_and_config_drift(tmp_path: Path):
    encoder = DeterministicMockEncoder(dimension=8)
    spec = _spec(tmp_path, encoder)
    writer = FileEmbeddingWriter(tmp_path / "embeddings", spec)
    batch = _batch((_unit(0), _unit(1)))
    vectors = encoder.encode(batch.units)

    with pytest.raises(VectorValidationError, match="shape mismatch"):
        writer.write_batch(batch, vectors[:, :7])
    with pytest.raises(VectorValidationError, match="dtype"):
        writer.write_batch(batch, vectors.astype(np.float64))
    invalid = vectors.copy()
    invalid[0, 0] = np.nan
    with pytest.raises(VectorValidationError, match="NaN or Inf"):
        writer.write_batch(batch, invalid)
    duplicate = _unit(2, duplicate_unit_hash=batch.units[0].unit_hash)
    duplicate_batch = _batch((batch.units[0], duplicate))
    with pytest.raises(FileEmbeddingWriterError, match="repeats a source unit"):
        writer.write_batch(duplicate_batch, encoder.encode(duplicate_batch.units))

    changed_encoder = DeterministicMockEncoder(dimension=9)
    changed_spec = _spec(tmp_path, changed_encoder)
    with pytest.raises(WriterCompatibilityError, match="manifest is incompatible"):
        FileEmbeddingWriter(tmp_path / "embeddings", changed_spec)


def test_empty_batch_checkpoint_and_completion_are_safe(tmp_path: Path):
    encoder = DeterministicMockEncoder(dimension=8)
    writer = FileEmbeddingWriter(
        tmp_path / "embeddings", _spec(tmp_path, encoder)
    )
    empty = _batch(())
    commit = writer.write_batch(empty, encoder.encode(empty.units))
    assert commit.status == "COMMITTED"
    assert commit.row_count == 0
    assert commit.shard_relative_path is None
    assert writer.committed_rows == 0
    assert not list(writer.shard_dir.glob("*.parquet"))
    manifest = writer.complete(expected_rows=0)
    assert manifest["status"] == "COMPLETED"
    replay = writer.write_batch(empty, encoder.encode(empty.units))
    assert replay.status == "SKIPPED_ALREADY_COMMITTED"
    with pytest.raises(FileEmbeddingWriterError, match="completed writer"):
        writer.write_batch(_batch((_unit(3),), index=1), encoder.encode((_unit(3),)))


def test_writer_recovers_published_orphan_shard_without_double_count(tmp_path: Path):
    encoder = DeterministicMockEncoder(dimension=8)
    spec = _spec(tmp_path, encoder)
    batch = _batch((_unit(0), _unit(1)))
    writer = FileEmbeddingWriter(tmp_path / "embeddings", spec)
    writer.write_batch(batch, encoder.encode(batch.units))

    checkpoint = json.loads(writer.checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["batches"] = {}
    checkpoint["committed_rows"] = 0
    writer.checkpoint_path.write_text(
        json.dumps(checkpoint, sort_keys=True), encoding="utf-8"
    )

    recovered = FileEmbeddingWriter(tmp_path / "embeddings", spec)
    assert recovered.committed_rows == 2
    replay = recovered.write_batch(batch, encoder.encode(batch.units))
    assert replay.status == "SKIPPED_ALREADY_COMMITTED"
    assert len(list(recovered.shard_dir.glob("*.parquet"))) == 1


def test_writer_detects_shard_corruption_and_supports_event_identity(tmp_path: Path):
    encoder = DeterministicMockEncoder(dimension=8)
    spec = _spec(tmp_path, encoder, modality="event_text")
    event = _unit(0, modality="event_text")
    batch = _batch((event,), modality="event_text")
    writer = FileEmbeddingWriter(tmp_path / "embeddings", spec)
    commit = writer.write_batch(batch, encoder.encode(batch.units))
    shard = writer.run_root / str(commit.shard_relative_path)
    table = pq.read_table(shard)
    assert table["relation_id"].to_pylist() == [0]
    assert table["source_idx"].to_pylist() == [0]
    assert table["target_idx"].to_pylist() == [1]

    shard.write_bytes(shard.read_bytes() + b"corruption")
    with pytest.raises(FileEmbeddingWriterError, match="checksum mismatch"):
        FileEmbeddingWriter(tmp_path / "embeddings", spec)


def test_embedding_run_identity_cannot_reuse_source_run(tmp_path: Path):
    encoder = DeterministicMockEncoder(dimension=8)
    with pytest.raises(ValueError, match="must differ"):
        FileEmbeddingRunSpec(
            embedding_run_id="b-source",
            modality="node_text",
            source=_source(tmp_path),
            preprocessing_hash=_preprocessing_hash(),
            encoder=encoder.metadata,
        )
