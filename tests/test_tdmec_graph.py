"""Unit tests for Dataset A provisional graph builders."""
from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import pytest

from tdmec import constants as C
from tdmec_diagnostics.adapters import build_node_universe_lookup_from_ids
from tdmec_graph.aggregate import accumulate_edges, edges_to_records, empty_edge_counts
from tdmec_graph.config import load_graph_config
from tdmec_graph.dedup_a import composite_signature, text_content_hash
from tdmec_graph.events import GraphEvent, RowAccounting, iter_graph_events_from_rows
from tdmec_graph.features import build_structural_tensors
from tdmec_graph.pipeline import GraphPipeline
from tdmec_graph.config import GraphConfig


def _epoch(y, m, d=1):
    return int(dt.datetime(y, m, d, tzinfo=dt.timezone.utc).timestamp())


def test_composite_signature_stable_and_no_float_id():
    a = composite_signature(
        author_account_id="1",
        created_at_utc_iso="2020-01-01T00:00:00+00:00",
        text_hash="abcd",
        relation_id=0,
        target_account_id="2",
    )
    b = composite_signature(
        author_account_id="1",
        created_at_utc_iso="2020-01-01T00:00:00+00:00",
        text_hash="abcd",
        relation_id=0,
        target_account_id="2",
    )
    assert a == b
    assert len(a) == 64


def test_weight_log1p_and_no_self_loops():
    counts = empty_edge_counts()
    counts[(10, 0, 1, 2)] = 3
    counts[(10, 0, 5, 5)] = 9  # self-loop discarded in edges_to_records
    rows = edges_to_records(counts)
    assert len(rows) == 1
    assert rows[0]["count_raw"] == 3
    assert math.isclose(rows[0]["weight_log1p"], math.log1p(3))


def test_structural_features_mask_and_schema():
    edge_counts = {
        (0, 0, 0, 1): 2,  # mention 0->1
    }
    tweet_counts = {(0, 0): 1, (0, 2): 3}  # node 2 tweets only
    x, mask = build_structural_tensors(edge_counts, tweet_counts)
    assert x.shape == (35, 16736, 17)
    assert mask.shape == (35, 16736)
    assert mask[0, 0] and mask[0, 1] and mask[0, 2]
    assert not mask[0, 3]
    assert x.dtype == np.float32
    assert np.all(x[~mask] == 0)
    # mention out-degree of node 0 = 1
    assert x[0, 0, 0] == 1.0
    assert math.isclose(float(x[0, 0, 2]), math.log1p(2), rel_tol=1e-5)
    assert math.isclose(float(x[0, 2, 16]), math.log1p(3), rel_tol=1e-5)
    assert list(C.STRUCT_FEATURE_NAMES)[16] == "tweet_count_log1p"


def _write_a_workbook(path: Path, rows):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "tweets"
    header = [
        "id", "created_at", "user", "text", "user_mentions",
        "retweeted_status", "reply_status", "quoted_status",
    ]
    # Pad to required columns from schema contract — use discovery required set
    from tdmec_diagnostics.schema_contracts import DATASET_A_REQUIRED_COLUMNS
    header = list(DATASET_A_REQUIRED_COLUMNS)
    ws.append(header)
    for r in rows:
        ws.append([r.get(c) for c in header])
    wb.save(path)


def test_event_extraction_synthetic(tmp_path: Path):
    from tdmec_diagnostics.schema_contracts import DATASET_A_REQUIRED_COLUMNS

    ids = ["100", "200", "300"]
    lookup = build_node_universe_lookup_from_ids(ids)
    # Expand lookup to pretend N is small — features require 16736, events don't
    header = list(DATASET_A_REQUIRED_COLUMNS)
    user100 = "{'id': 100, 'username': 'a'}"
    user200 = "{'id': 200, 'username': 'b'}"
    row = {c: None for c in header}
    row.update(
        {
            "id": 1.0,  # untrusted float
            "created_at": str(_epoch(2020, 1, 15)),
            "user": user100,
            "text": "hello @b",
            "user_mentions": "[{'id': 200}]",
            "retweeted_status": None,
            "reply_status": None,
            "quoted_status": None,
        }
    )
    # self-loop mention should be excluded
    row2 = dict(row)
    row2["user_mentions"] = "[{'id': 100}]"
    row2["text"] = "self"
    rows = [[row.get(c) for c in header], [row2.get(c) for c in header]]
    acc = RowAccounting()
    out = list(
        iter_graph_events_from_rows(
            rows,
            header=header,
            source_file="part.xlsx",
            node_lookup=lookup,
            start_row_number=2,
            accounting=acc,
        )
    )
    assert acc.authored_retained == 2
    assert acc.self_loops == 1
    authored0, events0 = out[0]
    assert authored0 is not None
    assert len(events0) == 1
    assert events0[0].relation_id == 0
    assert events0[0].source_idx == lookup.get("100")
    assert events0[0].target_idx == lookup.get("200")
    assert events0[0].cleaned_text == "hello @b"


@pytest.mark.skipif(
    not __import__("os").environ.get("TDMEC_DATABASE_URL"),
    reason="TDMEC_DATABASE_URL not set (Postgres+AGE required)",
)
def test_graph_pipeline_smoke(tmp_path: Path):
    from tdmec_diagnostics.schema_contracts import DATASET_A_REQUIRED_COLUMNS
    from tdmec_pilot.node_map import build_node_map_from_ids, save_node_map

    # Full-size synthetic node map (includes accounts 100/200/300).
    nm = build_node_map_from_ids([str(i) for i in range(16736)])
    map_path = tmp_path / "node_index_map.parquet"
    save_node_map(nm, map_path)

    a_dir = tmp_path / "dataset_a"
    a_dir.mkdir()
    header = list(DATASET_A_REQUIRED_COLUMNS)
    wb_path = a_dir / "core_army_pro_fans_tweets_part_001.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "tweets"
    ws.append(header)
    row = {c: None for c in header}
    row.update(
        {
            "id": 42.0,
            "created_at": str(_epoch(2020, 6, 1)),
            "user": "{'id': 100, 'username': 'a'}",
            "text": "edge text",
            "user_mentions": "[{'id': 200}]",
        }
    )
    ws.append([row.get(c) for c in header])
    # second relation
    row2 = dict(row)
    row2["user_mentions"] = None
    row2["retweeted_status"] = "{'user': {'id': 300}}"
    row2["text"] = "rt text"
    ws.append([row2.get(c) for c in header])
    wb.save(wb_path)

    cfg_path = Path("configs/dataset_a_graph.yaml")
    cfg = load_graph_config(cfg_path)
    # smaller chunks for test
    cfg.raw.setdefault("runtime", {})["chunk_size"] = 10
    out_root = tmp_path / "out"
    pipe = GraphPipeline(
        cfg,
        dataset_a_source=f"local:{a_dir}",
        output_root=str(out_root),
        node_index_map_path=str(map_path),
        cache_root=str(tmp_path / "cache"),
        verbose=False,
    )
    report = pipe.run(keep_work_db=False)
    assert report["all_passed"], report["gates"]
    run_dir = out_root / "graph" / report["run_id"]
    assert (run_dir / "X_struct.npy").is_file()
    assert (run_dir / "struct_active_mask.npy").is_file()
    assert (run_dir / "events" / "canonical_events.parquet").is_file()
    assert (run_dir / "manifest.json").is_file()
    x = np.load(run_dir / "X_struct.npy")
    assert x.shape == (35, 16736, 17)
    ev = pd.read_parquet(run_dir / "events" / "canonical_events.parquet")
    assert len(ev) >= 2
    assert "cleaned_text" in ev.columns
