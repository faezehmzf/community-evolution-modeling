"""Unit + integration tests for the Dataset B controlled pilot.

All tests use synthetic local workbooks + a synthetic node map; no network and no
real datasets are needed.
"""
from __future__ import annotations

import copy
import datetime as dt
from pathlib import Path

import openpyxl
import pandas as pd
import pytest

from tdmec_pilot.config import PilotConfig
from tdmec_pilot.dedup import DuplicateTracker, content_hash
from tdmec_pilot.identifiers import normalize_account_id, normalize_tweet_id
from tdmec_pilot.node_map import build_node_map_from_ids, load_node_map, save_node_map
from tdmec_pilot.pipeline import ConfigIncompatibleError, PilotPipeline
from tdmec_pilot.snapshots import assign_snapshot, boundary_table
from tdmec_pilot.timestamps import parse_created_at
from tdmec_pilot.user_blob import parse_user_blob

DATASET_B_COLUMNS = ["id", "created_at", "user", "text", "likes", "retweets",
                     "reply_count", "quoted_count", "bookmarks", "views"]

# epoch helpers
def _epoch(y, m, d=1):
    return int(dt.datetime(y, m, d, tzinfo=dt.timezone.utc).timestamp())


# --------------------------------------------------------------------- user blob
def test_user_blob_python_literal():
    v = "{'id': 1662724992054829056, 'followers': 235, 'username': 'derJudasBaum', 'political_category': None}"
    u = parse_user_blob(v)
    assert u.ok and u.account_id == "1662724992054829056" and u.username == "derJudasBaum"


def test_user_blob_json_variant():
    v = '{"id": 123456789012345678, "username": "abc", "political_category": null}'
    u = parse_user_blob(v)
    assert u.ok and u.account_id == "123456789012345678" and u.username == "abc"


def test_user_blob_title_with_quotes():
    v = "{'id': 12345678901234567, 'username': 'x', 'title': \"I'm good\"}"
    u = parse_user_blob(v)
    assert u.ok and u.account_id == "12345678901234567"


def test_user_blob_missing_and_malformed():
    assert parse_user_blob(None).error == "missing_user"
    assert parse_user_blob("").error == "missing_user"
    assert parse_user_blob("not a dict at all").ok is False
    assert parse_user_blob("{'followers': 5}").error == "missing_user_id"


def test_user_blob_no_eval_side_effect():
    # A string that would execute code under eval must NOT run; it is malformed.
    u = parse_user_blob("__import__('os').system('echo pwned')")
    assert u.ok is False


# --------------------------------------------------------------------- identifiers
def test_tweet_id_exact_large_integer_string():
    assert normalize_tweet_id("1666797773449207814").value == "1666797773449207814"
    assert normalize_tweet_id(1666797773449207814).value == "1666797773449207814"


def test_tweet_id_float_rejected():
    r = normalize_tweet_id(1.548256971999941e18)
    assert r.ok is False and r.error == "invalid_tweet_id_float"


def test_tweet_id_garbage_rejected():
    assert normalize_tweet_id("abc").ok is False
    assert normalize_tweet_id(None).ok is False


def test_account_id_canonical():
    assert normalize_account_id(123) == "123"
    assert normalize_account_id("123") == "123"
    assert normalize_account_id(1.5) is None


# --------------------------------------------------------------------- timestamps
def test_timestamp_valid_epoch():
    r = parse_created_at(str(_epoch(2020, 5, 1)))
    assert r.ok and r.utc.year == 2020


def test_timestamp_invalid():
    assert parse_created_at("not-a-number").ok is False
    assert parse_created_at("").ok is False
    assert parse_created_at(str(_epoch(1990, 1, 1))).ok is False  # before Twitter


# --------------------------------------------------------------------- snapshots
def test_snapshot_boundaries():
    assert len(boundary_table()) == 35
    assert boundary_table()[0].label == "2017-Q4"
    assert boundary_table()[34].label == "2026-Q2"
    # start of range
    assert assign_snapshot(dt.datetime(2017, 10, 1, tzinfo=dt.timezone.utc)) == 0
    assert assign_snapshot(dt.datetime(2017, 12, 31, tzinfo=dt.timezone.utc)) == 0
    # end of range
    assert assign_snapshot(dt.datetime(2026, 6, 30, tzinfo=dt.timezone.utc)) == 34


def test_snapshot_outside_range():
    assert assign_snapshot(dt.datetime(2017, 9, 30, tzinfo=dt.timezone.utc)) is None  # before
    assert assign_snapshot(dt.datetime(2026, 7, 1, tzinfo=dt.timezone.utc)) is None   # after
    assert assign_snapshot(None) is None


# --------------------------------------------------------------------- node map
def test_node_map_build_and_load(tmp_path):
    nm = build_node_map_from_ids(["30", "10", "20"])
    assert nm.mapping == {"10": 0, "20": 1, "30": 2}
    p = tmp_path / "nm.parquet"
    save_node_map(nm, p)
    loaded = load_node_map(p, expected_count=3, index_min=0, index_max=2)
    assert loaded.get("20") == 1 and loaded.get("999") is None


# --------------------------------------------------------------------- dedup
def test_dedup_exact_vs_conflicting():
    t = DuplicateTracker()
    h1 = content_hash(("a", "1", "same"))
    t.add("111", "f1.xlsx", 2, h1)
    t.add("111", "f2.xlsx", 5, h1)          # exact duplicate
    t.add("222", "f1.xlsx", 3, content_hash(("a", "1", "x")))
    t.add("222", "f1.xlsx", 9, content_hash(("a", "1", "y")))  # conflicting
    rep = {r.tweet_id: r for r in t.duplicate_report()}
    assert rep["111"].duplicate_type == "exact_duplicate"
    assert rep["222"].duplicate_type == "conflicting_id"
    assert rep["111"].canonical_source_row_number == 2  # f1 < f2


# ------------------------------------------------------------------- integration
def _make_b_workbook(path: Path, rows):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(DATASET_B_COLUMNS)
    for r in rows:
        ws.append(r)
    wb.save(path)


def _user(uid):
    return "{'id': %s, 'followers': 5, 'username': 'u%s', 'political_category': None}" % (uid, uid)


@pytest.fixture()
def pilot_env(tmp_path):
    src = tmp_path / "dataset_b"
    src.mkdir()
    # frozen authors: 100,101,102 -> node indices 0,1,2
    node_ids = ["100", "101", "102"]
    nm = build_node_map_from_ids(node_ids)
    nm_path = tmp_path / "node_index_map.parquet"
    save_node_map(nm, nm_path)

    good_ts = str(_epoch(2020, 5, 1))     # snapshot in range
    old_ts = str(_epoch(2015, 1, 1))      # before range -> excluded (but frozen author)
    # file 1
    _make_b_workbook(src / "t1.xlsx", [
        # retained
        ["1000000000000000001", good_ts, _user(100), "hello", 1, 0, 0, 0, 0, 0],
        ["1000000000000000002", good_ts, _user(101), "world", 2, 1, 0, 0, 0, 0],
        # excluded: author not in frozen universe
        ["1000000000000000003", good_ts, _user(999), "outsider", 0, 0, 0, 0, 0, 0],
        # excluded: frozen author but out of snapshot range
        ["1000000000000000004", old_ts, _user(102), "too old", 0, 0, 0, 0, 0, 0],
        # rejected: float tweet id
        [1.0000000000000002e18, good_ts, _user(100), "floatid", 0, 0, 0, 0, 0, 0],
        # rejected: malformed user
        ["1000000000000000006", good_ts, "not-a-user", "baduser", 0, 0, 0, 0, 0, 0],
        # rejected: invalid timestamp
        ["1000000000000000007", "bad-ts", _user(100), "badts", 0, 0, 0, 0, 0, 0],
    ])
    # file 2 (includes duplicates of file1 ids)
    _make_b_workbook(src / "t2.xlsx", [
        # exact duplicate of 1000000000000000001 (same content)
        ["1000000000000000001", good_ts, _user(100), "hello", 1, 0, 0, 0, 0, 0],
        # conflicting duplicate of 1000000000000000002 (different text)
        ["1000000000000000002", good_ts, _user(101), "DIFFERENT", 2, 1, 0, 0, 0, 0],
        # unique retained
        ["1000000000000000010", good_ts, _user(102), "unique", 0, 0, 0, 0, 0, 0],
    ])

    canonical = {
        "frozen_node_count": 3, "valid_node_index_min": 0, "valid_node_index_max": 2,
        "account_join_key": "user.id", "tweet_id_type": "string",
        "snapshot_frequency": "quarterly", "snapshot_start": "2017-Q4",
        "snapshot_end": "2026-Q2", "snapshot_count": 35,
        "node_universe_expansion": "forbidden", "raw_source_modification": "forbidden",
        "text_normalization": {"unicode_form": "NFC", "normalize_newlines": True, "strip_bom": True},
        "duplicate_key": "tweet_id", "duplicate_action": "annotate",
    }
    raw = {"pilot_name": "test", "config_version": 1, "canonical": canonical,
           "input_files": ["t1.xlsx", "t2.xlsx"], "expected_sheet": "Sheet1",
           "expected_columns": DATASET_B_COLUMNS,
           "runtime": {"chunk_size": 3}}
    cfg = PilotConfig(raw=raw)
    return {"cfg": cfg, "src": str(src), "nm_path": str(nm_path),
            "out": str(tmp_path / "out"), "cache": str(tmp_path / "cache")}


def _run(env, run_id=None, fail_after_chunk=None):
    pipe = PilotPipeline(env["cfg"], dataset_b_source=f"local:{env['src']}",
                         output_root=env["out"], node_index_map_path=env["nm_path"],
                         cache_root=env["cache"], run_id=run_id,
                         fail_after_chunk=fail_after_chunk)
    return pipe


def test_integration_full_run_and_accounting(pilot_env):
    pipe = _run(pilot_env)
    report = pipe.run()
    assert report["all_passed"], report["gates"]
    acc = report["accounting"]
    # 10 input rows total (7 + 3)
    assert acc["rows_in"] == 10
    assert acc["rows_in"] == acc["retained"] + acc["excluded"] + acc["rejected"]
    # retained: t1 has 2, t2 has 3 (dup exact, dup conflicting, unique) -> 5
    assert acc["retained"] == 5
    # excluded: outsider + too-old = 2
    assert acc["excluded"] == 2
    # rejected: floatid + baduser + badts = 3
    assert acc["rejected"] == 3


def test_integration_node_index_and_snapshot_gates(pilot_env):
    pipe = _run(pilot_env)
    report = pipe.run()
    g = report["gates"]
    assert g["retained_all_have_node_index"]
    assert g["node_index_in_range"]
    assert g["retained_snapshot_in_range"]
    assert g["tweet_ids_exact_strings"]
    assert g["raw_source_unchanged"]


def test_integration_matched_unmatched_accounts(pilot_env):
    pipe = _run(pilot_env)
    report = pipe.run()
    fin = report["finalize"]
    # authors seen: 100,101,102 (frozen) + 999 (unmatched)
    assert fin["unmatched_authors"] == 1
    assert fin["matched_authors"] == 3
    unmatched = pd.read_parquet(Path(pipe.layout.root) / "unmatched_accounts.parquet")
    assert set(unmatched["author_account_id"]) == {"999"}


def test_integration_duplicate_report(pilot_env):
    pipe = _run(pilot_env)
    pipe.run()
    dup = pd.read_parquet(Path(pipe.layout.root) / "duplicate_records.parquet")
    by_id = {str(r.tweet_id): r.duplicate_type for r in dup.itertuples()}
    assert by_id["1000000000000000001"] == "exact_duplicate"
    assert by_id["1000000000000000002"] == "conflicting_id"


def test_integration_excluded_records_preserved(pilot_env):
    pipe = _run(pilot_env)
    pipe.run()
    xparts = list((Path(pipe.layout.root) / "excluded_records").glob("*.parquet"))
    xdf = pd.concat([pd.read_parquet(p) for p in xparts], ignore_index=True)
    reasons = set(xdf["exclusion_reason"].dropna())
    assert "author_not_in_frozen_universe" in reasons
    assert "outside_canonical_snapshot_range" in reasons
    assert "invalid_tweet_id_float" in reasons
    assert "malformed_user_blob" in reasons
    assert "invalid_timestamp" in reasons


def test_resume_after_interruption(pilot_env):
    # t1 has 7 rows, chunk_size=3 -> 3 chunks; interrupt after first new chunk.
    pipe1 = _run(pilot_env, fail_after_chunk=1)
    run_id = pipe1.run_id
    with pytest.raises(KeyboardInterrupt):
        pipe1.run()
    # resume with same run id -> completes
    pipe2 = _run(pilot_env, run_id=run_id)
    report = pipe2.run()
    assert report["all_passed"], report["gates"]
    # no duplicated normalized records after resume
    nparts = list((Path(pipe2.layout.root) / "normalized_records").glob("**/*.parquet"))
    ndf = pd.concat([pd.read_parquet(p) for p in nparts], ignore_index=True)
    # 5 retained rows total, uniquely by (source_file, source_row_number)
    assert len(ndf) == 5
    assert ndf.duplicated(subset=["source_file", "source_row_number"]).sum() == 0
    assert report["accounting"]["rows_in"] == 10


def test_config_incompatibility_blocks_resume(pilot_env):
    pipe1 = _run(pilot_env)
    run_id = pipe1.run_id
    pipe1.run()
    # change canonical config -> different hash -> must refuse to resume same run dir
    env2 = dict(pilot_env)
    cfg2_raw = copy.deepcopy(pilot_env["cfg"].raw)
    cfg2_raw["canonical"]["snapshot_start"] = "2018-Q1"  # changes hash
    env2["cfg"] = PilotConfig(raw=cfg2_raw)
    pipe2 = PilotPipeline(env2["cfg"], dataset_b_source=f"local:{env2['src']}",
                          output_root=env2["out"], node_index_map_path=env2["nm_path"],
                          cache_root=env2["cache"], run_id=run_id)
    with pytest.raises(ConfigIncompatibleError):
        pipe2.run()
