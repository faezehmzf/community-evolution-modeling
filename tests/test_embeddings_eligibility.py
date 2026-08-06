"""Focused tests for derived embedding eligibility, deduplication, and hashes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tdmec import constants as C
from tdmec_embeddings.eligibility import (
    EligibilityError,
    EligibilityPolicy,
    EventTextEligibilityProcessor,
    NodeTextEligibilityProcessor,
    cleaned_text_content_hash,
    load_duplicate_canonical_index,
)
from tdmec_embeddings.file_sources import EventTextFileReader, NodeTextFileReader


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
    _write_json(
        root / "manifest.json",
        {
            "run_id": run_id,
            "config_hash": "cfg-test",
            "git_commit": "abc1234",
            "artifact_status": "PROVISIONAL",
            "canonical": canonical,
        },
    )
    _write_json(
        root / "validation_report.json", {"run_id": run_id, "all_passed": True}
    )
    checksums = {
        p.relative_to(root).as_posix(): _sha(p)
        for p in sorted(root.glob("**/*.parquet"))
    }
    _write_json(root / "checksums.json", checksums)


def _node_table() -> pa.Table:
    return pa.table(
        {
            "tweet_id": pa.array(
                ["unique", "null", "empty", "space", "dup", "dup"], pa.string()
            ),
            "node_index": pa.array([0, 0, 1, 1, 2, 2], pa.int32()),
            "snapshot_id": pa.array([0, 0, 1, 1, 2, 2], pa.int32()),
            "cleaned_text": pa.array(
                ["private-good", None, "", "\u00a0\t", "private-drop", "private-keep"],
                pa.string(),
            ),
            "text_quality": pa.array(["valid"] * 6, pa.string()),
            "source_file": pa.array(
                ["z.xlsx", "z.xlsx", "z.xlsx", "z.xlsx", "b.xlsx", "a.xlsx"],
                pa.string(),
            ),
            "source_sheet": pa.array(["Sheet1"] * 6, pa.string()),
            "source_row_number": pa.array([1, 2, 3, 4, 2, 9], pa.int64()),
            "is_duplicate": pa.array([None] * 6, pa.bool_()),
            "is_canonical_duplicate": pa.array([None] * 6, pa.bool_()),
        }
    )


def _duplicate_table(*, canonical_file: str = "a.xlsx") -> pa.Table:
    return pa.table(
        {
            "tweet_id": pa.array(["dup"], pa.string()),
            "duplicate_type": pa.array(["conflicting_id"], pa.string()),
            "occurrence_count": pa.array([2], pa.int64()),
            "canonical_source_file": pa.array([canonical_file], pa.string()),
            "canonical_source_row_number": pa.array([9], pa.int64()),
            "source_locations": pa.array(["a.xlsx:9;b.xlsx:2"], pa.string()),
        }
    )


def _make_node_source(root: Path) -> None:
    normalized = root / "normalized_records/source_file=fixture/part-00000.parquet"
    normalized.parent.mkdir(parents=True)
    pq.write_table(_node_table(), normalized)
    pq.write_table(_duplicate_table(), root / "duplicate_records.parquet")
    _metadata(root, "b-run", "dataset_b")


def _event_table(*, bad_hash: bool = False) -> pa.Table:
    texts = ["event-private", None, "", " \t"]
    good_hash = cleaned_text_content_hash(texts[0])[:16]
    return pa.table(
        {
            "signature": pa.array(["sig-a", "sig-b", "sig-c", "sig-d"], pa.string()),
            "snapshot_id": pa.array([0, 0, 1, 2], pa.int32()),
            "relation_id": pa.array([0, 1, 2, 3], pa.int8()),
            "source_idx": pa.array([0, 0, 1, 2], pa.int32()),
            "target_idx": pa.array([1, 2, 2, 3], pa.int32()),
            "cleaned_text": pa.array(texts, pa.string()),
            "text_hash": pa.array(
                ["incorrect" if bad_hash else good_hash, "", "", ""], pa.string()
            ),
            "text_quality": pa.array(["valid", "missing", "missing", "missing"]),
            "source_file": pa.array(["events.xlsx"] * 4, pa.string()),
            "source_row_number": pa.array([1, 2, 3, 4], pa.int64()),
        }
    )


def _make_event_source(root: Path, *, bad_hash: bool = False) -> None:
    path = root / "events/canonical_events.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(_event_table(bad_hash=bad_hash), path)
    _metadata(root, "a-run", "dataset_a")


def test_node_eligibility_excludes_missing_and_selects_report_canonical(tmp_path: Path):
    root = tmp_path / "b"
    _make_node_source(root)
    reader = NodeTextFileReader(root, expected_run_id="b-run", batch_size=2)
    duplicates = load_duplicate_canonical_index(reader.identity, batch_size=1)
    processor = NodeTextEligibilityProcessor(reader, duplicates)

    batches = list(processor.iter_batches())
    units = [unit for batch in batches for unit in batch.units]
    assert [batch.num_rows for batch in batches] == [1, 0, 1]
    assert len(units) == 2
    assert units[1].source_file == "a.xlsx"
    assert units[1].source_row_number == 9
    assert units[1].cleaned_text == "private-keep"
    assert "private-keep" not in repr(units[1])
    assert "dup" not in repr(units[1])

    report = processor.report
    assert report.input_rows_seen == 6
    assert report.eligible_rows == 2
    assert report.excluded_by_reason == {
        "null_text": 1,
        "empty_text": 1,
        "whitespace_only_text": 1,
        "noncanonical_duplicate": 1,
    }
    assert report.duplicate_groups_declared == 1
    assert report.duplicate_occurrences_seen == 2
    assert report.duplicate_canonical_rows_seen == 1
    assert report.completed_full_source


def test_eligibility_hashes_are_stable_path_free_and_source_sensitive(tmp_path: Path):
    left = tmp_path / "one" / "b"
    right = tmp_path / "two" / "b"
    _make_node_source(left)
    _make_node_source(right)

    def first_unit(root: Path):
        reader = NodeTextFileReader(root, expected_run_id="b-run", batch_size=6)
        index = load_duplicate_canonical_index(reader.identity)
        processor = NodeTextEligibilityProcessor(reader, index)
        units = [unit for batch in processor.iter_batches() for unit in batch.units]
        return reader, units[0]

    left_reader, left_unit = first_unit(left)
    _, right_unit = first_unit(right)
    assert left_unit.content_hash == right_unit.content_hash
    assert left_unit.unit_hash == right_unit.unit_hash
    assert left_unit.preprocessing_hash == right_unit.preprocessing_hash
    assert str(left.resolve()) not in left_unit.preprocessing_hash

    policy = EligibilityPolicy(version="different-policy")
    assert (
        policy.preprocessing_hash(left_reader.identity, "node_text")
        != left_unit.preprocessing_hash
    )


def test_duplicate_index_rejects_nondeterministic_canonical_selection(tmp_path: Path):
    root = tmp_path / "b"
    normalized = root / "normalized_records/source_file=fixture/part-00000.parquet"
    normalized.parent.mkdir(parents=True)
    pq.write_table(_node_table(), normalized)
    pq.write_table(
        _duplicate_table(canonical_file="b.xlsx"), root / "duplicate_records.parquet"
    )
    _metadata(root, "b-run", "dataset_b")
    reader = NodeTextFileReader(root, expected_run_id="b-run")
    with pytest.raises(EligibilityError, match="not deterministic"):
        load_duplicate_canonical_index(reader.identity)


def test_event_eligibility_filters_missing_and_verifies_source_hash(tmp_path: Path):
    root = tmp_path / "a"
    _make_event_source(root)
    reader = EventTextFileReader(root, expected_run_id="a-run", batch_size=2)
    processor = EventTextEligibilityProcessor(reader)

    batches = list(processor.iter_batches())
    units = [unit for batch in batches for unit in batch.units]
    assert len(units) == 1
    assert units[0].relation_id == 0
    assert units[0].content_hash == cleaned_text_content_hash("event-private")
    assert processor.report.eligible_rows == 1
    assert processor.report.source_text_hashes_verified == 1
    assert processor.report.excluded_by_reason == {
        "null_text": 1,
        "empty_text": 1,
        "whitespace_only_text": 1,
    }


def test_event_eligibility_rejects_inconsistent_source_text_hash(tmp_path: Path):
    root = tmp_path / "a"
    _make_event_source(root, bad_hash=True)
    reader = EventTextFileReader(root, expected_run_id="a-run")
    processor = EventTextEligibilityProcessor(reader)
    with pytest.raises(EligibilityError, match="source text hash is inconsistent"):
        list(processor.iter_batches())


def test_bounded_eligibility_is_partial_and_does_not_require_duplicate_canonical(
    tmp_path: Path,
):
    root = tmp_path / "b"
    _make_node_source(root)
    reader = NodeTextFileReader(
        root, expected_run_id="b-run", batch_size=2, max_rows=2
    )
    index = load_duplicate_canonical_index(reader.identity)
    processor = NodeTextEligibilityProcessor(reader, index)
    list(processor.iter_batches())
    assert processor.report.input_rows_seen == 2
    assert not processor.report.completed_full_source
    assert processor.report.duplicate_canonical_rows_seen == 0


def test_report_is_aggregate_only_and_processor_is_single_use(tmp_path: Path):
    root = tmp_path / "b"
    _make_node_source(root)
    reader = NodeTextFileReader(root, expected_run_id="b-run")
    processor = NodeTextEligibilityProcessor(
        reader, load_duplicate_canonical_index(reader.identity)
    )
    with pytest.raises(EligibilityError, match="only after complete iteration"):
        _ = processor.report
    list(processor.iter_batches())
    serialized = json.dumps(processor.report.to_dict(), sort_keys=True)
    for private_value in (
        "private-good",
        "private-drop",
        "private-keep",
        '"unique"',
        '"dup"',
        "a.xlsx",
        "z.xlsx",
    ):
        assert private_value not in serialized
    assert "PROVISIONAL_SMOKE_ONLY" in serialized
    with pytest.raises(EligibilityError, match="single-use"):
        list(processor.iter_batches())
