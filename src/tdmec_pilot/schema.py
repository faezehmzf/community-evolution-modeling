"""Canonical normalized record schema + Parquet schema + validation."""
from __future__ import annotations

from typing import Dict, List

# Ordered canonical columns for a normalized Dataset B record.
NORMALIZED_COLUMNS: List[str] = [
    "tweet_id",             # exact string
    "author_account_id",    # canonical string of integer id
    "node_index",           # nullable int (frozen universe only)
    "created_at_original",  # raw epoch string (provenance)
    "created_at_utc",       # tz-aware UTC timestamp (nullable)
    "snapshot_id",          # nullable int 0..34
    "text_raw",             # verbatim
    "text_normalized",      # NFC + newline canonicalized
    "text_quality",         # ok | empty | whitespace_only | non_string
    "text_raw_len",
    "text_normalized_len",
    "likes",
    "retweets",
    "reply_count",
    "quoted_count",
    "bookmarks",
    "views",
    "source_file",
    "source_sheet",
    "source_row_number",    # 1-based row number within the sheet
    "record_status",        # retained | excluded | rejected
    "exclusion_reason",     # nullable
    "is_duplicate",         # bool: tweet_id appears more than once in corpus
    "is_canonical_duplicate",  # bool: canonical row for its duplicate group
]

RECORD_STATUSES = {"retained", "excluded", "rejected"}

EXCLUSION_REASONS = {
    None,
    "author_not_in_frozen_universe",
    "outside_canonical_snapshot_range",
    "malformed_user_blob",
    "missing_user_id",
    "missing_user",
    "unexpected_user_type",
    "user_not_object",
    "invalid_tweet_id",
    "invalid_tweet_id_float",
    "invalid_timestamp",
}

_INT_NULLABLE = ["node_index", "snapshot_id", "likes", "retweets", "reply_count",
                 "quoted_count", "bookmarks", "views", "text_raw_len",
                 "text_normalized_len", "source_row_number"]


def records_to_frame(records: List[dict]):
    """Coerce record dicts into a typed pandas DataFrame in canonical column order."""
    import pandas as pd

    df = pd.DataFrame(records, columns=NORMALIZED_COLUMNS)
    for c in _INT_NULLABLE:
        df[c] = pd.array(df[c], dtype="Int64")
    for c in ["tweet_id", "author_account_id", "created_at_original", "text_raw",
              "text_normalized", "text_quality", "source_file", "source_sheet",
              "record_status", "exclusion_reason"]:
        df[c] = df[c].astype("string")
    for c in ["is_duplicate", "is_canonical_duplicate"]:
        df[c] = df[c].astype("boolean")
    df["created_at_utc"] = pd.to_datetime(df["created_at_utc"], utc=True)
    return df


def parquet_schema():
    import pyarrow as pa

    return pa.schema([
        ("tweet_id", pa.string()),
        ("author_account_id", pa.string()),
        ("node_index", pa.int32()),
        ("created_at_original", pa.string()),
        ("created_at_utc", pa.timestamp("us", tz="UTC")),
        ("snapshot_id", pa.int32()),
        ("text_raw", pa.string()),
        ("text_normalized", pa.string()),
        ("text_quality", pa.string()),
        ("text_raw_len", pa.int64()),
        ("text_normalized_len", pa.int64()),
        ("likes", pa.int64()),
        ("retweets", pa.int64()),
        ("reply_count", pa.int64()),
        ("quoted_count", pa.int64()),
        ("bookmarks", pa.int64()),
        ("views", pa.int64()),
        ("source_file", pa.string()),
        ("source_sheet", pa.string()),
        ("source_row_number", pa.int64()),
        ("record_status", pa.string()),
        ("exclusion_reason", pa.string()),
        ("is_duplicate", pa.bool_()),
        ("is_canonical_duplicate", pa.bool_()),
    ])


def validate_columns(columns: List[str], expected: List[str]) -> Dict[str, object]:
    cols = list(columns)
    return {
        "matches": cols == list(expected),
        "actual": cols,
        "missing": [c for c in expected if c not in cols],
        "unexpected": [c for c in cols if c not in expected],
        "order_ok": cols == list(expected),
    }
