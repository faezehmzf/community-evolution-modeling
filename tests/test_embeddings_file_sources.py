"""Focused tests for bounded, deterministic file-backed embedding sources."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tdmec import constants as C
from tdmec_embeddings.file_sources import (
    CanonicalEdgeFileReader,
    EventTextFileReader,
    FileSourceError,
    NodeTextFileReader,
    load_file_source_identity,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _metadata(root: Path, run_id: str, kind: str) -> None:
    canonical = {
        "frozen_node_count": 4,
        "snapshot_count": 3,
        "certification_status": "UNVALIDATED",
        "dedup_status": "PROVISIONAL",
        "calendar_status": "PROVISIONAL",
    }
    if kind == "dataset_a":
        canonical["relation_order"] = list(C.RELATION_ORDER)
    manifest = {
        "run_id": run_id,
        "config_hash": "cfg-test",
        "git_commit": "abc1234",
        "artifact_status": "PROVISIONAL",
        "canonical": canonical,
    }
    _write_json(root / "manifest.json", manifest)
    _write_json(
        root / "validation_report.json", {"run_id": run_id, "all_passed": True}
    )
    checksums = {
        p.relative_to(root).as_posix(): _sha(p)
        for p in sorted(root.glob("**/*.parquet"))
    }
    _write_json(root / "checksums.json", checksums)


def _node_table(tweet_ids: list[str], rows: list[int]) -> pa.Table:
    n = len(tweet_ids)
    return pa.table(
        {
            "tweet_id": pa.array(tweet_ids, pa.string()),
            "node_index": pa.array([value % 4 for value in rows], pa.int32()),
            "snapshot_id": pa.array([value % 3 for value in rows], pa.int32()),
            "cleaned_text": pa.array(
                [f"private-{value}" for value in rows], pa.string()
            ),
            "text_quality": pa.array(["valid"] * n, pa.string()),
            "source_file": pa.array(["source.xlsx"] * n, pa.string()),
            "source_sheet": pa.array(["Sheet1"] * n, pa.string()),
            "source_row_number": pa.array(rows, pa.int64()),
            "is_duplicate": pa.array([None] * n, pa.bool_()),
            "is_canonical_duplicate": pa.array([None] * n, pa.bool_()),
        }
    )


def _event_table() -> pa.Table:
    return pa.table(
        {
            "signature": pa.array(["sig-a", "sig-b", "sig-c"], pa.string()),
            "snapshot_id": pa.array([0, 0, 1], pa.int32()),
            "relation_id": pa.array([0, 2, 3], pa.int8()),
            "source_idx": pa.array([0, 1, 2], pa.int32()),
            "target_idx": pa.array([1, 2, 3], pa.int32()),
            "cleaned_text": pa.array(["one", "two", "three"], pa.string()),
            "text_hash": pa.array(["h1", "h2", "h3"], pa.string()),
            "text_quality": pa.array(["valid"] * 3, pa.string()),
            "source_file": pa.array(["a.xlsx"] * 3, pa.string()),
            "source_row_number": pa.array([2, 3, 4], pa.int64()),
        }
    )


def _edge_table(snapshot: int, relation: int, source: int) -> pa.Table:
    count = 2
    return pa.table(
        {
            "snapshot_id": pa.array([snapshot], pa.int64()),
            "relation_id": pa.array([relation], pa.int64()),
            "src_index": pa.array([source], pa.int64()),
            "dst_index": pa.array([(source + 1) % 4], pa.int64()),
            "count_raw": pa.array([count], pa.int64()),
            "weight_log1p": pa.array([math.log1p(count)], pa.float64()),
        }
    )


def test_node_reader_is_bounded_naturally_ordered_and_independently_identified(
    tmp_path: Path,
):
    root = tmp_path / "b"
    p10 = root / "normalized_records/source_file=statuses-10.xlsx/part-00000.parquet"
    p2 = root / "normalized_records/source_file=statuses-2.xlsx/part-00000.parquet"
    p10.parent.mkdir(parents=True)
    p2.parent.mkdir(parents=True)
    pq.write_table(_node_table(["ten"], [10]), p10)
    pq.write_table(_node_table(["two-a", "two-b"], [2, 3]), p2)
    _metadata(root, "b-run", "dataset_b")

    reader = NodeTextFileReader(
        root,
        expected_run_id="b-run",
        batch_size=1,
        max_rows=2,
    )
    batches = list(reader.iter_batches())
    assert reader.total_rows == 3
    assert reader.shard_relative_paths[0].startswith(
        "normalized_records/source_file=statuses-2.xlsx/"
    )
    assert [b.num_rows for b in batches] == [1, 1]
    assert [b.global_row_offset for b in batches] == [0, 1]
    assert batches[0].records["source_row_number"].to_pylist() == [2]
    assert batches[0].source.run_id == "b-run"
    assert batches[0].source.source_kind == "dataset_b"
    assert "artifact_root" not in batches[0].source.provenance()


def test_event_reader_preserves_canonical_fields_and_bounds_batches(tmp_path: Path):
    root = tmp_path / "a"
    path = root / "events/canonical_events.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(_event_table(), path)
    _metadata(root, "a-run", "dataset_a")

    identity = load_file_source_identity(
        root,
        source_kind="dataset_a",
        expected_run_id="a-run",
    )
    reader = EventTextFileReader(
        root,
        expected_run_id="a-run",
        identity=identity,
        batch_size=2,
    )
    batches = list(reader.iter_batches())
    assert [b.num_rows for b in batches] == [2, 1]
    assert batches[0].records.schema.names == list(EventTextFileReader.columns)
    assert batches[1].shard_row_offset == 2
    assert batches[0].source is identity


def test_edge_reader_uses_numeric_partition_order_and_canonical_endpoint_names(
    tmp_path: Path,
):
    root = tmp_path / "a"
    p10 = root / "edges/snapshot=10/relation=0/part-00000.parquet"
    p2 = root / "edges/snapshot=2/relation=3/part-00000.parquet"
    p10.parent.mkdir(parents=True)
    p2.parent.mkdir(parents=True)
    pq.write_table(_edge_table(10, 0, 0), p10)
    pq.write_table(_edge_table(2, 3, 1), p2)
    _metadata(root, "a-run", "dataset_a")
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["canonical"]["snapshot_count"] = 12
    _write_json(root / "manifest.json", manifest)

    reader = CanonicalEdgeFileReader(root, expected_run_id="a-run", batch_size=4)
    batches = list(reader.iter_batches())
    assert [b.records["snapshot_id"][0].as_py() for b in batches] == [2, 10]
    assert batches[0].records.schema.names == [
        "snapshot_id",
        "relation_id",
        "source_idx",
        "target_idx",
        "count_raw",
        "weight_log1p",
    ]


def test_source_identity_rejects_run_mismatch_and_checksum_damage(tmp_path: Path):
    root = tmp_path / "b"
    path = root / "normalized_records/source_file=statuses-0.xlsx/part-00000.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(_node_table(["one"], [1]), path)
    _metadata(root, "b-run", "dataset_b")

    with pytest.raises(FileSourceError, match="source run mismatch"):
        load_file_source_identity(
            root,
            source_kind="dataset_b",
            expected_run_id="wrong-run",
        )
    path.write_bytes(path.read_bytes() + b"damage")
    with pytest.raises(FileSourceError, match="checksum mismatch"):
        load_file_source_identity(
            root,
            source_kind="dataset_b",
            expected_run_id="b-run",
        )


def test_reader_rejects_schema_and_bounds_violations(tmp_path: Path):
    missing_root = tmp_path / "missing"
    missing_path = missing_root / "normalized_records/source_file=x/part-00000.parquet"
    missing_path.parent.mkdir(parents=True)
    pq.write_table(pa.table({"tweet_id": ["x"]}), missing_path)
    _metadata(missing_root, "missing-run", "dataset_b")
    with pytest.raises(FileSourceError, match="missing required fields"):
        NodeTextFileReader(missing_root, expected_run_id="missing-run")

    bounds_root = tmp_path / "bounds"
    bounds_path = bounds_root / "normalized_records/source_file=x/part-00000.parquet"
    bounds_path.parent.mkdir(parents=True)
    table = _node_table(["x"], [1]).set_column(
        1, "node_index", pa.array([99], pa.int32())
    )
    pq.write_table(table, bounds_path)
    _metadata(bounds_root, "bounds-run", "dataset_b")
    reader = NodeTextFileReader(bounds_root, expected_run_id="bounds-run")
    with pytest.raises(FileSourceError, match="outside"):
        list(reader.iter_batches())


def test_edge_reader_rejects_partition_identity_mismatch(tmp_path: Path):
    root = tmp_path / "a"
    path = root / "edges/snapshot=1/relation=0/part-00000.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(_edge_table(2, 0, 0), path)
    _metadata(root, "a-run", "dataset_a")
    reader = CanonicalEdgeFileReader(root, expected_run_id="a-run")
    with pytest.raises(FileSourceError, match="partition identity"):
        list(reader.iter_batches())
