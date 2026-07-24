"""Unit tests for the discovery tooling using local synthetic workbooks.

These tests require no network and no Google Drive credentials: they exercise
the local-filesystem source adapter, checksum, cache, workbook inspection,
field heuristics, and the Phase-0 verifier end to end.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import openpyxl
import pytest

from tdmec_discovery.cache import DownloadCache
from tdmec_discovery.config import DiscoveryConfig
from tdmec_discovery.fields import classify_columns, schema_signature
from tdmec_discovery.hashing import sha256_file
from tdmec_discovery.sources import LocalSource, build_source, extract_drive_folder_id
from tdmec_discovery.verify import verify_access
from tdmec_discovery.xlsx_inspect import inspect_sheet, workbook_sheet_names


def _make_workbook(path: Path, rows):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(["id", "user_id", "username", "created_at", "text", "lang"])
    for r in rows:
        ws.append(r)
    wb.save(path)


@pytest.fixture()
def dataset_dir(tmp_path: Path) -> Path:
    d = tmp_path / "dataset"
    d.mkdir()
    _make_workbook(
        d / "part_001.xlsx",
        [
            [1001, 5, "alice", "2024-01-01 10:00:00", "hello world", "en"],
            [1002, 5, "alice", "2024-01-02 11:00:00", "", "en"],
            [1003, 6, "bob", "2024-01-03 12:00:00", "another tweet", "ko"],
            [1003, 6, "bob", "2024-01-03 12:00:00", "dup id row", "ko"],  # dup id
        ],
    )
    return d


def test_local_source_lists_and_downloads(dataset_dir: Path, tmp_path: Path):
    src = LocalSource(str(dataset_dir))
    files = src.list_files()
    assert len(files) == 1
    rf = files[0]
    assert rf.ext == ".xlsx"
    assert rf.size and rf.size > 0
    dest = tmp_path / "copy.xlsx"
    src.download(rf, dest)
    assert dest.exists()
    # source unchanged (read-only guarantee)
    assert (dataset_dir / "part_001.xlsx").exists()
    assert sha256_file(dest) == sha256_file(dataset_dir / "part_001.xlsx")


def test_build_source_autodetect(dataset_dir: Path):
    assert isinstance(build_source(str(dataset_dir)), LocalSource)
    assert isinstance(build_source(f"local:{dataset_dir}"), LocalSource)


def test_extract_drive_folder_id():
    fid = "0EXAMPLEfolderID000000000000abcdEFG"  # synthetic, not a real folder id
    assert extract_drive_folder_id(fid) == fid
    url = f"https://drive.google.com/drive/folders/{fid}?usp=sharing"
    assert extract_drive_folder_id(url) == fid


def test_workbook_sheet_names(dataset_dir: Path):
    assert workbook_sheet_names(dataset_dir / "part_001.xlsx") == ["Sheet1"]


def test_inspect_sheet_counts_and_keys(dataset_dir: Path):
    insp = inspect_sheet(
        dataset_dir / "part_001.xlsx",
        key_columns=["id"],
        collect_distinct=["user_id"],
        timestamp_columns=["created_at"],
    )
    assert insp.n_data_rows == 4
    assert insp.columns == ["id", "user_id", "username", "created_at", "text", "lang"]
    assert insp.key_stats["id"]["distinct"] == 3
    assert insp.key_stats["id"]["duplicate_rows"] == 1
    assert insp.key_stats["user_id"]["approx_distinct"] == 2
    # one empty text row counted as null/empty
    assert insp.column_stats["text"].null_count == 1
    assert insp.key_stats["created_at"]["ts_invalid"] == 0


def test_field_classification():
    cols = ["id", "user_id", "username", "created_at", "text", "lang",
            "reply_status", "retweeted_status", "quoted_status", "user_mentions"]
    roles = classify_columns(cols)
    assert roles["tweet_id"] == ["id"]
    assert roles["author_account_id"] == ["user_id"]
    assert roles["timestamp"] == ["created_at"]
    assert roles["reply"] == ["reply_status"]
    assert roles["retweet"] == ["retweeted_status"]
    assert roles["quote"] == ["quoted_status"]
    assert roles["mention"] == ["user_mentions"]


def test_schema_signature_order_insensitive():
    assert schema_signature(["a", "b", "c"]) == schema_signature(["c", "a", "b"])


def test_cache_reuse(dataset_dir: Path, tmp_path: Path):
    src = LocalSource(str(dataset_dir))
    rf = src.list_files()[0]
    cache = DownloadCache(tmp_path / "cache")
    rec1 = cache.get(src, rf)
    assert rec1["reused"] is False
    rec2 = cache.get(src, rf)
    assert rec2["reused"] is True
    assert rec1["sha256"] == rec2["sha256"]


def test_verify_access_local(dataset_dir: Path, tmp_path: Path):
    cfg = DiscoveryConfig(
        dataset_a_source=f"local:{dataset_dir}",
        dataset_b_source=f"local:{dataset_dir}",
        output_root=tmp_path / "out",
        cache_root=tmp_path / "cache",
        drive_output_folder_id=None,
        google_credentials=None,
    )
    summary = verify_access(cfg, out_dir=tmp_path / "out")
    assert summary["datasets"]["A"]["access_confirmed"] is True
    assert summary["datasets"]["A"]["parsed_ok"] is True
    assert (tmp_path / "out" / "access_verification.json").exists()
