"""Documented Dataset A / B schema contracts for Phase 2 adapters.

Field mappings are taken only from active docs/data contracts.
Unsupported or ambiguous schemas are rejected (never guessed).
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Tuple

# ---------------------------------------------------------------------------
# Dataset A — docs/data/02, docs/data/03
# Sheet: tweets; 31 columns (identical across parts)
# ---------------------------------------------------------------------------
DATASET_A_ADAPTER_ID: str = "dataset_a_xlsx_v1"
DATASET_A_SHEET_NAME: str = "tweets"

DATASET_A_REQUIRED_COLUMNS: Tuple[str, ...] = (
    "id",
    "created_at",
    "user",
    "text",
    "user_mentions",
    "retweeted_status",
    "reply_status",
    "quoted_status",
)

# Full documented column set (order not required for equality checks)
DATASET_A_DOCUMENTED_COLUMNS: Tuple[str, ...] = (
    "timestamp",
    "is_removed",
    "id",
    "created_at",
    "is_quote_status",
    "user",
    "text",
    "ocr_text",
    "text_lang",
    "lang",
    "text_tags",
    "text_hashtags",
    "text_emojis",
    "user_mentions",
    "media",
    "likes",
    "retweets",
    "reply_count",
    "quoted_count",
    "reply_status",
    "quoted_status",
    "retweeted_status",
    "location_tags",
    "bookmarks",
    "views",
    "place",
    "impression",
    "engagement",
    "sentiment",
    "topic",
    "copy_count",
)

# Relation extraction paths (authoritative from docs/data/02 / 03)
DATASET_A_RELATION_EXTRACTORS: Mapping[str, str] = MappingProxyType(
    {
        "mention": "user_mentions[].id",
        "retweet": "retweeted_status.user.id",
        "reply": "reply_status.user.id",
        "quote": "quoted_status.user.id",
    }
)

# Dataset A tweet id is float-lossy — forbidden as exact dedup key
DATASET_A_TWEET_ID_TRUSTED: bool = False

# ---------------------------------------------------------------------------
# Dataset B — docs/data/04, docs/data/05
# Sheet: Sheet1; 10 columns fixed order
# ---------------------------------------------------------------------------
DATASET_B_ADAPTER_ID: str = "dataset_b_xlsx_v1"
DATASET_B_SHEET_NAME: str = "Sheet1"

DATASET_B_REQUIRED_COLUMNS: Tuple[str, ...] = (
    "id",
    "created_at",
    "user",
    "text",
    "likes",
    "retweets",
    "reply_count",
    "quoted_count",
    "bookmarks",
    "views",
)

DATASET_B_COLUMN_ORDER: Tuple[str, ...] = DATASET_B_REQUIRED_COLUMNS

DATASET_B_TWEET_ID_TRUSTED: bool = True

# Text field labeling: Dataset B stores uncleaned raw text (contract).
# Phase 2 applies only NFC/newline/BOM normalization for length diagnostics;
# that normalized form is labeled "normalized_raw", not production-cleaned.
DATASET_B_TEXT_FIELD_LABEL: str = "raw_text"
DATASET_A_TEXT_FIELD_LABEL: str = "edge_event_text"
