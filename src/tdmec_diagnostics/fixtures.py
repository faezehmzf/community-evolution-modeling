"""Synthetic fixtures for Phase 2 diagnostics (no copied real data)."""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from tdmec_diagnostics.records import DiagnosticEventRecord

# Tiny synthetic node universe for fixtures (not production N)
SYN_NODE_UNIVERSE: Tuple[int, ...] = (0, 1, 2, 3, 4)
SYN_FROZEN_NODES = set(SYN_NODE_UNIVERSE)

# Provisional fixture calendar: 2018-Q1 .. 2018-Q4 (includes internal empty Q3)
SYN_START = "2018-Q1"
SYN_END = "2018-Q4"


def _epoch(year: int, month: int, day: int = 15) -> int:
    return int(dt.datetime(year, month, day, tzinfo=dt.timezone.utc).timestamp())


def build_synthetic_records() -> List[DiagnosticEventRecord]:
    """Return a multi-file synthetic corpus covering Phase 2 edge cases."""
    records: List[DiagnosticEventRecord] = []

    # --- File A1: Dataset A structure + edge text across quarters ---
    # 2018-Q1
    records.append(
        DiagnosticEventRecord(
            dataset="A",
            source_file="syn_a_part1.xlsx",
            source_row_number=1,
            timestamp_raw=_epoch(2018, 1),
            external_user_id="1001",
            tweet_id="900001",
            text="alpha edge event text one",
            relation="mention",
            target_external_user_id="1002",
            node_idx=0,
            target_node_idx=1,
            struct_active=True,
            node_text_available=False,
            edge_text_available=True,
            extra={"quarter_label": "2018-Q1"},
        )
    )
    # Concordant duplicate of above (same composite key fields)
    records.append(
        DiagnosticEventRecord(
            dataset="A",
            source_file="syn_a_part1.xlsx",
            source_row_number=2,
            timestamp_raw=_epoch(2018, 1),
            external_user_id="1001",
            tweet_id="900001b",
            text="alpha edge event text one",
            relation="mention",
            target_external_user_id="1002",
            node_idx=0,
            target_node_idx=1,
            struct_active=True,
            node_text_available=False,
            edge_text_available=True,
            extra={"quarter_label": "2018-Q1"},
        )
    )
    # Discordant: same user+timestamp, different text/target
    records.append(
        DiagnosticEventRecord(
            dataset="A",
            source_file="syn_a_part1.xlsx",
            source_row_number=3,
            timestamp_raw=_epoch(2018, 1),
            external_user_id="1001",
            tweet_id="900002",
            text="discordant different content",
            relation="reply",
            target_external_user_id="1003",
            node_idx=0,
            target_node_idx=2,
            struct_active=True,
            node_text_available=False,
            edge_text_available=True,
            extra={"quarter_label": "2018-Q1"},
        )
    )
    # 2018-Q2 structure-only (edge text unavailable)
    records.append(
        DiagnosticEventRecord(
            dataset="A",
            source_file="syn_a_part1.xlsx",
            source_row_number=4,
            timestamp_raw=_epoch(2018, 4),
            external_user_id="1002",
            tweet_id="900003",
            text=None,
            relation="retweet",
            target_external_user_id="1001",
            node_idx=1,
            target_node_idx=0,
            struct_active=True,
            node_text_available=False,
            edge_text_available=False,
            extra={"quarter_label": "2018-Q2"},
        )
    )
    # Self-loop candidate
    records.append(
        DiagnosticEventRecord(
            dataset="A",
            source_file="syn_a_part1.xlsx",
            source_row_number=5,
            timestamp_raw=_epoch(2018, 4),
            external_user_id="1003",
            tweet_id="900004",
            text="self loop text",
            relation="quote",
            target_external_user_id="1003",
            node_idx=2,
            target_node_idx=2,
            struct_active=True,
            node_text_available=False,
            edge_text_available=True,
            extra={"quarter_label": "2018-Q2"},
        )
    )
    # Invalid relation
    records.append(
        DiagnosticEventRecord(
            dataset="A",
            source_file="syn_a_part1.xlsx",
            source_row_number=6,
            timestamp_raw=_epoch(2018, 4),
            external_user_id="1002",
            tweet_id="900005",
            text="bad relation",
            relation="follow",
            target_external_user_id="1001",
            node_idx=1,
            target_node_idx=0,
            struct_active=True,
            node_text_available=False,
            edge_text_available=True,
            extra={"quarter_label": "2018-Q2"},
        )
    )
    # 2018-Q4 (after internal empty Q3) structure+text via later B join conceptually
    records.append(
        DiagnosticEventRecord(
            dataset="A",
            source_file="syn_a_part1.xlsx",
            source_row_number=7,
            timestamp_raw=_epoch(2018, 10),
            external_user_id="1004",
            tweet_id="900006",
            text="late quarter edge",
            relation="mention",
            target_external_user_id="1001",
            node_idx=3,
            target_node_idx=0,
            struct_active=True,
            node_text_available=False,
            edge_text_available=True,
            extra={"quarter_label": "2018-Q4"},
        )
    )

    # Concordant-only same user+timestamp pair (distinct from discordant group above)
    records.append(
        DiagnosticEventRecord(
            dataset="A",
            source_file="syn_a_part1.xlsx",
            source_row_number=8,
            timestamp_raw=_epoch(2018, 2, 20),
            external_user_id="1002",
            tweet_id="900013",
            text="pure concordant pair text",
            relation="mention",
            target_external_user_id="1001",
            node_idx=1,
            target_node_idx=0,
            struct_active=True,
            node_text_available=False,
            edge_text_available=True,
            extra={"quarter_label": "2018-Q1"},
        )
    )
    records.append(
        DiagnosticEventRecord(
            dataset="A",
            source_file="syn_a_part1.xlsx",
            source_row_number=9,
            timestamp_raw=_epoch(2018, 2, 20),
            external_user_id="1002",
            tweet_id="900014",
            text="pure concordant pair text",
            relation="mention",
            target_external_user_id="1001",
            node_idx=1,
            target_node_idx=0,
            struct_active=True,
            node_text_available=False,
            edge_text_available=True,
            extra={"quarter_label": "2018-Q1"},
        )
    )

    # --- File A2: cross-file duplicate of A1 row1 composite ---
    records.append(
        DiagnosticEventRecord(
            dataset="A",
            source_file="syn_a_part2.xlsx",
            source_row_number=1,
            timestamp_raw=_epoch(2018, 1),
            external_user_id="1001",
            tweet_id="900007",
            text="alpha edge event text one",
            relation="mention",
            target_external_user_id="1002",
            node_idx=0,
            target_node_idx=1,
            struct_active=True,
            node_text_available=False,
            edge_text_available=True,
            extra={"quarter_label": "2018-Q1"},
        )
    )
    # Missing candidate-key fields
    records.append(
        DiagnosticEventRecord(
            dataset="A",
            source_file="syn_a_part2.xlsx",
            source_row_number=2,
            timestamp_raw=None,
            external_user_id="1002",
            tweet_id="900008",
            text="missing timestamp",
            relation="mention",
            target_external_user_id="1001",
            node_idx=1,
            target_node_idx=0,
            struct_active=True,
            node_text_available=False,
            edge_text_available=True,
        )
    )
    # Invalid / unparsable timestamps
    records.append(
        DiagnosticEventRecord(
            dataset="A",
            source_file="syn_a_part2.xlsx",
            source_row_number=3,
            timestamp_raw="not-a-time",
            external_user_id="1002",
            tweet_id="900009",
            text="unparsable ts",
            relation="mention",
            target_external_user_id="1001",
            node_idx=1,
            target_node_idx=0,
            struct_active=True,
            node_text_available=False,
            edge_text_available=False,
        )
    )
    # Epoch outlier (year 1990)
    records.append(
        DiagnosticEventRecord(
            dataset="A",
            source_file="syn_a_part2.xlsx",
            source_row_number=4,
            timestamp_raw=_epoch(1990, 1),
            external_user_id="1002",
            tweet_id="900010",
            text="epoch outlier",
            relation="mention",
            target_external_user_id="1001",
            node_idx=1,
            target_node_idx=0,
            struct_active=True,
            node_text_available=False,
            edge_text_available=False,
        )
    )
    # Leading-empty candidate: record before provisional start (2017)
    records.append(
        DiagnosticEventRecord(
            dataset="A",
            source_file="syn_a_part2.xlsx",
            source_row_number=5,
            timestamp_raw=_epoch(2017, 6),
            external_user_id="1001",
            tweet_id="900011",
            text="before range",
            relation="mention",
            target_external_user_id="1002",
            node_idx=0,
            target_node_idx=1,
            struct_active=True,
            node_text_available=False,
            edge_text_available=True,
        )
    )
    # Trailing-empty candidate: record after provisional end (2019)
    records.append(
        DiagnosticEventRecord(
            dataset="A",
            source_file="syn_a_part2.xlsx",
            source_row_number=6,
            timestamp_raw=_epoch(2019, 6),
            external_user_id="1001",
            tweet_id="900012",
            text="after range",
            relation="mention",
            target_external_user_id="1002",
            node_idx=0,
            target_node_idx=1,
            struct_active=True,
            node_text_available=False,
            edge_text_available=True,
        )
    )

    # --- File B1: Dataset B node text ---
    # Node-text-only in 2018-Q2 for node 4
    records.append(
        DiagnosticEventRecord(
            dataset="B",
            source_file="syn_b_statuses_0.xlsx",
            source_row_number=1,
            timestamp_raw=_epoch(2018, 5),
            external_user_id="1005",
            tweet_id="800001",
            text="node text only content for coverage",
            relation=None,
            node_idx=4,
            struct_active=False,
            node_text_available=True,
            edge_text_available=False,
            extra={"quarter_label": "2018-Q2"},
        )
    )
    # Structure+text conceptually: node 0 also has B text in Q1
    records.append(
        DiagnosticEventRecord(
            dataset="B",
            source_file="syn_b_statuses_0.xlsx",
            source_row_number=2,
            timestamp_raw=_epoch(2018, 2),
            external_user_id="1001",
            tweet_id="800002",
            text="authored text joining structure node",
            relation=None,
            node_idx=0,
            struct_active=False,
            node_text_available=True,
            edge_text_available=False,
            extra={"quarter_label": "2018-Q1"},
        )
    )
    # Exact concordant duplicate same tweet id
    records.append(
        DiagnosticEventRecord(
            dataset="B",
            source_file="syn_b_statuses_0.xlsx",
            source_row_number=3,
            timestamp_raw=_epoch(2018, 2),
            external_user_id="1001",
            tweet_id="800002",
            text="authored text joining structure node",
            relation=None,
            node_idx=0,
            struct_active=False,
            node_text_available=True,
            edge_text_available=False,
            extra={"quarter_label": "2018-Q1"},
        )
    )
    # Discordant same tweet id different text
    records.append(
        DiagnosticEventRecord(
            dataset="B",
            source_file="syn_b_statuses_0.xlsx",
            source_row_number=4,
            timestamp_raw=_epoch(2018, 2),
            external_user_id="1001",
            tweet_id="800003",
            text="first version of conflicting id",
            relation=None,
            node_idx=0,
            struct_active=False,
            node_text_available=True,
            edge_text_available=False,
            extra={"quarter_label": "2018-Q1"},
        )
    )

    # --- File B2: cross-file duplicate + external outside universe + null/empty/long text ---
    records.append(
        DiagnosticEventRecord(
            dataset="B",
            source_file="syn_b_statuses_1.xlsx",
            source_row_number=1,
            timestamp_raw=_epoch(2018, 2),
            external_user_id="1001",
            tweet_id="800003",
            text="second version of conflicting id",
            relation=None,
            node_idx=0,
            struct_active=False,
            node_text_available=True,
            edge_text_available=False,
            extra={"quarter_label": "2018-Q1"},
        )
    )
    # External user outside frozen universe (node_idx None)
    records.append(
        DiagnosticEventRecord(
            dataset="B",
            source_file="syn_b_statuses_1.xlsx",
            source_row_number=2,
            timestamp_raw=_epoch(2018, 5),
            external_user_id="999999",
            tweet_id="800004",
            text="outside universe user text",
            relation=None,
            node_idx=None,
            struct_active=False,
            node_text_available=True,
            edge_text_available=False,
            extra={"quarter_label": "2018-Q2"},
        )
    )
    # Null text
    records.append(
        DiagnosticEventRecord(
            dataset="B",
            source_file="syn_b_statuses_1.xlsx",
            source_row_number=3,
            timestamp_raw=_epoch(2018, 5),
            external_user_id="1002",
            tweet_id="800005",
            text=None,
            relation=None,
            node_idx=1,
            struct_active=False,
            node_text_available=False,
            edge_text_available=False,
            extra={"quarter_label": "2018-Q2"},
        )
    )
    # Empty text
    records.append(
        DiagnosticEventRecord(
            dataset="B",
            source_file="syn_b_statuses_1.xlsx",
            source_row_number=4,
            timestamp_raw=_epoch(2018, 5),
            external_user_id="1002",
            tweet_id="800006",
            text="",
            relation=None,
            node_idx=1,
            struct_active=False,
            node_text_available=False,
            edge_text_available=False,
            extra={"quarter_label": "2018-Q2"},
        )
    )
    # Long text
    long_text = "word " * 500 + "end"
    records.append(
        DiagnosticEventRecord(
            dataset="B",
            source_file="syn_b_statuses_1.xlsx",
            source_row_number=5,
            timestamp_raw=_epoch(2018, 11),
            external_user_id="1004",
            tweet_id="800007",
            text=long_text,
            relation=None,
            node_idx=3,
            struct_active=False,
            node_text_available=True,
            edge_text_available=False,
            extra={"quarter_label": "2018-Q4"},
        )
    )
    # Inactive node-snapshot contributor: mapped user with no struct and no usable text
    # (whitespace only => node_text_available False)
    records.append(
        DiagnosticEventRecord(
            dataset="B",
            source_file="syn_b_statuses_1.xlsx",
            source_row_number=6,
            timestamp_raw=_epoch(2018, 11),
            external_user_id="1003",
            tweet_id="800008",
            text="   ",
            relation=None,
            node_idx=2,
            struct_active=False,
            node_text_available=False,
            edge_text_available=False,
            extra={"quarter_label": "2018-Q4"},
        )
    )
    # Malformed tweet id (float-like)
    records.append(
        DiagnosticEventRecord(
            dataset="B",
            source_file="syn_b_statuses_1.xlsx",
            source_row_number=7,
            timestamp_raw=_epoch(2018, 11),
            external_user_id="1004",
            tweet_id="1.2345e+17",
            text="malformed id text",
            relation=None,
            node_idx=3,
            struct_active=False,
            node_text_available=True,
            edge_text_available=False,
            extra={"quarter_label": "2018-Q4"},
        )
    )

    return records


def records_by_file(
    records: Optional[Sequence[DiagnosticEventRecord]] = None,
) -> Dict[str, List[DiagnosticEventRecord]]:
    recs = list(records) if records is not None else build_synthetic_records()
    out: Dict[str, List[DiagnosticEventRecord]] = {}
    for r in recs:
        out.setdefault(r.source_file, []).append(r)
    return out


def write_fixture_manifest(path: Path) -> Path:
    """Write a privacy-safe inventory of synthetic fixtures (no raw text)."""
    import json

    recs = build_synthetic_records()
    by_file = records_by_file(recs)
    payload = {
        "schema_version": "tdmec-phase2-synthetic-fixtures-v1",
        "note": "Synthetic only; contains no copied real data",
        "provisional_calendar": {"start": SYN_START, "end": SYN_END},
        "node_universe_size": len(SYN_NODE_UNIVERSE),
        "files": {
            f: {
                "n_records": len(rows),
                "datasets": sorted({r.dataset for r in rows}),
            }
            for f, rows in sorted(by_file.items())
        },
        "n_records_total": len(recs),
        "coverage_scenarios": [
            "internal_empty_quarter_2018_Q3",
            "leading_trailing_out_of_range",
            "structure_only",
            "node_text_only",
            "structure_and_node_text",
            "inactive_node_snapshot",
            "edge_text_available_unavailable",
            "external_outside_universe",
            "self_loop_candidate",
            "invalid_relation",
            "null_empty_long_text",
            "concordant_discordant_duplicates",
            "cross_file_duplicates",
            "resumable_multi_file",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
