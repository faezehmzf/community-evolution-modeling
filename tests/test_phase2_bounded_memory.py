"""Regression tests for bounded-memory, transactional Phase 2 diagnostics."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import pytest

from tdmec.hashing import (
    hash_canonical,
    hash_canonical_json_native,
    sha256_file,
)
from tdmec_diagnostics import constants as DC
from tdmec_diagnostics.checkpoint import IncompleteRunError
from tdmec_diagnostics.config import DiagnosticsConfig
from tdmec_diagnostics.dedup_diag import DedupAccumulator, _sha16
from tdmec_diagnostics.pipeline import (
    DiagnosticsPipeline,
    FileSpec,
    run_diagnostics,
)
from tdmec_diagnostics.privacy import (
    assert_privacy_safe_mapping,
    hash_identifier,
    privacy_safe_file_ref,
)
from tdmec_diagnostics.records import DiagnosticEventRecord
from tdmec_diagnostics.reports import scientific_content_hash
from tdmec_diagnostics.transaction_state import TransactionalRunState


def _record(
    row: int,
    *,
    tweet_id: str,
    text: str,
    source_file: str = "high-cardinality.xlsx",
    user_id: str = "1001",
) -> DiagnosticEventRecord:
    return DiagnosticEventRecord(
        dataset="B",
        source_file=source_file,
        source_row_number=row,
        timestamp_raw=1514764800 + row,
        external_user_id=user_id,
        tweet_id=tweet_id,
        text=text,
        node_idx=0,
        struct_active=False,
        node_text_available=True,
        extra={"quarter_label": "2018-Q1"},
    )


def _config(*, chunk_size: int = 3, resume_mode: str = "resume"):
    return DiagnosticsConfig(
        provisional_start_label="2018-Q1",
        provisional_end_label="2018-Q4",
        node_universe_size=2,
        chunk_size=chunk_size,
        enable_checkpoint=True,
        resume_mode=resume_mode,
        dataset_a_source_scheme="synthetic",
        dataset_b_source_scheme="synthetic",
        source_format="synthetic",
    )


def _scientific_report_hashes(result):
    return {
        name: scientific_content_hash(report)
        for name, report in result["reports"].items()
    }


def _artifact_hashes(root: Path):
    return {
        str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
        for path in sorted(path for path in root.rglob("*") if path.is_file())
    }


def _legacy_group_stats(records, *, group_kind: str):
    """Small-fixture reference matching the retired in-memory implementation."""
    groups = defaultdict(list)
    for record in records:
        occurrence = {
            "source_file": privacy_safe_file_ref(record.source_file),
            "source_row_number": record.source_row_number,
            "content_hash": _sha16(record.content_fingerprint_fields()),
        }
        if group_kind == "tweet":
            key = hash_identifier(record.tweet_id, prefix="tid")
        elif group_kind == "full_row":
            key = _sha16(record.full_row_fingerprint_fields())
        else:
            raise AssertionError(group_kind)
        groups[key].append(occurrence)

    multiplicity = Counter(len(rows) for rows in groups.values())
    evidence = []
    concordant = discordant = cross_file = within_file = 0
    for key, rows in groups.items():
        if len(rows) < 2:
            continue
        content_hashes = {row["content_hash"] for row in rows}
        files = {row["source_file"] for row in rows}
        if len(content_hashes) == 1:
            concordant += 1
        else:
            discordant += 1
        if len(files) > 1:
            cross_file += 1
        else:
            within_file += 1
        canonical = min(
            rows,
            key=lambda row: (
                row["source_file"],
                row["source_row_number"],
            ),
        )
        evidence.append(
            (
                key,
                len(rows),
                len(content_hashes),
                len(files),
                canonical["source_file"],
                canonical["source_row_number"],
            )
        )
    return {
        "groups_total": len(groups),
        "duplicate_groups": concordant + discordant,
        "concordant_groups": concordant,
        "discordant_groups": discordant,
        "cross_file_duplicate_groups": cross_file,
        "within_file_duplicate_groups": within_file,
        "extra_duplicate_rows": sum(len(rows) - 1 for rows in groups.values()),
        "rows_before_candidate_exact_collapse": sum(map(len, groups.values())),
        "rows_after_candidate_exact_collapse": len(groups),
        "multiplicity_distribution": {
            str(key): multiplicity[key] for key in sorted(multiplicity)
        },
        "evidence": sorted(evidence),
    }


def test_high_cardinality_stream_retains_no_python_occurrences():
    accumulator = DedupAccumulator()
    small_state_size = None
    for row in range(1, 5_001):
        accumulator.observe(
            _record(
                row,
                tweet_id=str(8_000_000_000_000_000_000 + row),
                text=f"synthetic-{row}",
                user_id=str(10_000 + row),
            )
        )
        if row == 100:
            small_state_size = len(json.dumps(accumulator.to_state()))

    state = accumulator.to_state()
    assert accumulator.retained_occurrences_in_memory == 0
    assert accumulator.disk_occurrence_rows == 10_000
    assert "by_tweet_id" not in state
    assert "by_full_row" not in state
    assert len(json.dumps(state)) <= small_state_size + 100
    assert_privacy_safe_mapping(state)
    accumulator.close()


def test_streaming_scientific_hash_matches_canonical_hash():
    payload = {
        "z": [{"b": 2, "a": 1}, None, True],
        "a": {"float": 1.25, "text": "safe"},
    }
    assert hash_canonical_json_native(payload) == hash_canonical(payload)


def test_disk_backed_dedup_matches_retired_small_fixture_semantics():
    records = [
        _record(1, tweet_id="9001", text="same", source_file="a.xlsx"),
        _record(2, tweet_id="9001", text="same", source_file="b.xlsx"),
        _record(3, tweet_id="9002", text="left", source_file="b.xlsx"),
        _record(4, tweet_id="9002", text="right", source_file="b.xlsx"),
        _record(5, tweet_id="9003", text="unique", source_file="c.xlsx"),
    ]
    accumulator = DedupAccumulator()
    accumulator.observe_many(records)
    report = accumulator.build_report(config_hash="cfg")

    for report_key, group_kind in (
        ("dataset_b_same_id", "tweet"),
        ("exact_full_row", "full_row"),
    ):
        observed = report[report_key]
        reference = _legacy_group_stats(records, group_kind=group_kind)
        for key in (
            "groups_total",
            "duplicate_groups",
            "concordant_groups",
            "discordant_groups",
            "cross_file_duplicate_groups",
            "within_file_duplicate_groups",
            "extra_duplicate_rows",
            "rows_before_candidate_exact_collapse",
            "rows_after_candidate_exact_collapse",
            "multiplicity_distribution",
        ):
            assert observed[key] == reference[key]
        observed_evidence = sorted(
            (
                row["group_key_hash"],
                row["occurrence_count"],
                row["distinct_content_hashes"],
                row["distinct_files"],
                row["canonical_source_file_ref"],
                row["canonical_source_row_number"],
            )
            for row in observed["evidence_table"]
        )
        assert observed_evidence == reference["evidence"]
    accumulator.close()


def test_exact_duplicate_and_conflict_counts_are_preserved():
    accumulator = DedupAccumulator()
    accumulator.observe_many(
        [
            _record(1, tweet_id="1", text="same"),
            _record(2, tweet_id="1", text="same"),
            _record(3, tweet_id="2", text="left"),
            _record(4, tweet_id="2", text="right"),
        ]
    )
    stats = accumulator.build_report(config_hash="cfg")["dataset_b_same_id"]
    assert stats["duplicate_groups"] == 2
    assert stats["concordant_groups"] == 1
    assert stats["discordant_groups"] == 1
    assert stats["conflicting_metadata_groups"] == 1
    assert stats["extra_duplicate_rows"] == 2
    accumulator.close()


def test_mid_workbook_interruption_resumes_without_loss_or_double_count(
    tmp_path: Path,
):
    records = [
        _record(row, tweet_id=str(row // 2), text=f"text-{row // 2}")
        for row in range(1, 21)
    ]
    specs = [
        FileSpec(
            file_ref="one.xlsx",
            dataset="B",
            records=records,
            checksum="stable",
        )
    ]
    config = _config(chunk_size=4)
    interrupted = DiagnosticsPipeline(config, tmp_path, run_id="resume")
    with pytest.raises(IncompleteRunError):
        interrupted.run_on_files(
            specs,
            node_universe_size=2,
            frozen_nodes={0, 1},
            interrupt_after_chunks=2,
        )

    checkpoint_path = (
        tmp_path
        / "diagnostics"
        / "resume"
        / "checkpoints"
        / "diagnostics_checkpoint.json"
    )
    partial_checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert partial_checkpoint["files"]["one.xlsx"]["last_source_row_number"] == 8
    assert partial_checkpoint["files"]["one.xlsx"]["complete"] is False

    resumed = DiagnosticsPipeline(config, tmp_path, run_id="resume").run_on_files(
        specs,
        node_universe_size=2,
        frozen_nodes={0, 1},
    )
    reference = DiagnosticsPipeline(
        replace(config, enable_checkpoint=False),
        tmp_path,
        run_id="reference",
    ).run_on_files(
        specs,
        node_universe_size=2,
        frozen_nodes={0, 1},
    )
    assert resumed["reports"][DC.REPORT_DEDUP]["rows_inspected"] == 20
    assert (
        resumed["reports"][DC.REPORT_DEDUP]
        == reference["reports"][DC.REPORT_DEDUP]
    )


def test_transaction_authority_recovers_from_split_json_publication(
    tmp_path: Path,
):
    records = [
        _record(row, tweet_id=str(row), text=f"text-{row}")
        for row in range(1, 9)
    ]
    specs = [
        FileSpec(
            file_ref="one.xlsx",
            dataset="B",
            records=records,
            checksum="stable",
        )
    ]
    config = _config(chunk_size=2)
    with pytest.raises(IncompleteRunError):
        DiagnosticsPipeline(config, tmp_path, run_id="split").run_on_files(
            specs,
            node_universe_size=2,
            frozen_nodes={0, 1},
            interrupt_after_chunks=1,
        )

    checkpoint_root = (
        tmp_path / "diagnostics" / "split" / "checkpoints"
    )
    (checkpoint_root / "diagnostics_checkpoint.json").write_text(
        '{"broken": true}',
        encoding="utf-8",
    )
    (checkpoint_root / "accumulator_state.json").write_text(
        '{"different": true}',
        encoding="utf-8",
    )

    result = DiagnosticsPipeline(config, tmp_path, run_id="split").run_on_files(
        specs,
        node_universe_size=2,
        frozen_nodes={0, 1},
    )
    assert result["complete"] is True
    assert result["reports"][DC.REPORT_DEDUP]["rows_inspected"] == 8
    checkpoint = json.loads(
        (checkpoint_root / "diagnostics_checkpoint.json").read_text(
            encoding="utf-8"
        )
    )
    accumulators = json.loads(
        (checkpoint_root / "accumulator_state.json").read_text(
            encoding="utf-8"
        )
    )
    assert checkpoint["transaction_generation"] == (
        accumulators["transaction_generation"]
    )
    assert checkpoint["files"]["one.xlsx"]["complete"] is True


def test_uncommitted_chunk_is_rolled_back(tmp_path: Path):
    state = TransactionalRunState(tmp_path, "cfg")
    accumulator = DedupAccumulator(connection=state.connection)
    state.begin()
    accumulator.observe(_record(1, tweet_id="1", text="temporary"))
    state.rollback()
    assert accumulator.disk_occurrence_rows == 0
    state.remove()


def test_sealed_resume_is_idempotent_and_artifacts_do_not_drift(tmp_path: Path):
    records = [
        _record(row, tweet_id=str(row), text=f"text-{row}")
        for row in range(1, 6)
    ]
    specs = [
        FileSpec(
            file_ref="one.xlsx",
            dataset="B",
            records=records,
            checksum="stable",
        )
    ]
    config = _config(chunk_size=2)
    first = DiagnosticsPipeline(config, tmp_path, run_id="sealed").run_on_files(
        specs,
        node_universe_size=2,
        frozen_nodes={0, 1},
    )
    run_root = Path(first["layout"])
    before = _artifact_hashes(run_root)
    second = DiagnosticsPipeline(config, tmp_path, run_id="sealed").run_on_files(
        specs,
        node_universe_size=2,
        frozen_nodes={0, 1},
    )
    after = _artifact_hashes(run_root)
    assert second["resumed_from_sealed"] is True
    assert before == after
    assert not (
        run_root
        / "checkpoints"
        / TransactionalRunState.FILENAME
    ).exists()


def test_explicit_checkpoint_root_is_preserved(tmp_path: Path):
    external_checkpoint_root = tmp_path / "persistent-checkpoints"
    records = [_record(1, tweet_id="1", text="one")]
    specs = [
        FileSpec(
            file_ref="one.xlsx",
            dataset="B",
            records=records,
            checksum="stable",
        )
    ]
    config = _config(chunk_size=1)
    result = DiagnosticsPipeline(
        config,
        tmp_path / "outputs",
        run_id="explicit",
        checkpoint_root=external_checkpoint_root,
    ).run_on_files(
        specs,
        node_universe_size=2,
        frozen_nodes={0, 1},
    )
    assert result["complete"] is True
    assert (external_checkpoint_root / "diagnostics_checkpoint.json").is_file()
    assert (external_checkpoint_root / "accumulator_state.json").is_file()
    assert not (
        external_checkpoint_root / TransactionalRunState.FILENAME
    ).exists()
    assert not (
        Path(result["layout"])
        / "checkpoints"
        / "diagnostics_checkpoint.json"
    ).exists()


def test_public_entry_point_preserves_explicit_checkpoint_root(tmp_path: Path):
    checkpoint_root = tmp_path / "entry-checkpoints"
    result = run_diagnostics(
        output_root=tmp_path / "outputs",
        mode="synthetic",
        run_id="entry-explicit",
        checkpoint_root=checkpoint_root,
    )
    assert result["complete"] is True
    assert (checkpoint_root / "diagnostics_checkpoint.json").is_file()
    assert (checkpoint_root / "accumulator_state.json").is_file()


def test_resumed_and_uninterrupted_reports_and_hashes_are_deterministic(
    tmp_path: Path,
):
    records = [
        _record(row, tweet_id=str(row // 3), text=f"text-{row // 3}")
        for row in range(1, 16)
    ]
    specs = [
        FileSpec(
            file_ref="one.xlsx",
            dataset="B",
            records=records,
            checksum="stable",
        )
    ]
    config = _config(chunk_size=3)
    with pytest.raises(IncompleteRunError):
        DiagnosticsPipeline(config, tmp_path, run_id="resumed").run_on_files(
            specs,
            node_universe_size=2,
            frozen_nodes={0, 1},
            interrupt_after_chunks=2,
        )
    resumed = DiagnosticsPipeline(
        config,
        tmp_path,
        run_id="resumed",
    ).run_on_files(
        specs,
        node_universe_size=2,
        frozen_nodes={0, 1},
    )
    uninterrupted = DiagnosticsPipeline(
        config,
        tmp_path,
        run_id="uninterrupted",
    ).run_on_files(
        specs,
        node_universe_size=2,
        frozen_nodes={0, 1},
    )
    assert _scientific_report_hashes(resumed) == _scientific_report_hashes(
        uninterrupted
    )
    for report in resumed["reports"].values():
        assert_privacy_safe_mapping(report)
