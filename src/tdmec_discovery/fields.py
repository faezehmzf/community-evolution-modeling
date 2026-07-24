"""Heuristics for classifying tweet/status columns into candidate roles.

Purely name-based (and optionally type-aware) heuristics used to propose which
column plays each role. These are *candidates* to be verified, never assumed.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

ROLE_PATTERNS: Dict[str, List[str]] = {
    "tweet_id": [r"^id$", r"tweet.?id", r"status.?id", r"post.?id"],
    "author_account_id": [r"user.?id", r"author.?id", r"account.?id", r"^uid$"],
    "author_user_blob": [r"^user$", r"^author$"],
    "username": [r"user.?name", r"screen.?name", r"handle"],
    "timestamp": [r"created.?at", r"^timestamp$", r"^time$", r"^date$"],
    "text": [r"^text$", r"full.?text", r"content", r"body", r"^ocr_text$"],
    "reply": [r"reply", r"in.?reply.?to"],
    "retweet": [r"retweet", r"^rt$", r"retweeted.?status"],
    "quote": [r"quote", r"quoted.?status", r"is.?quote"],
    "mention": [r"mention", r"user.?mentions"],
    "language": [r"^lang$", r"language", r"text.?lang"],
}


def classify_columns(columns: Sequence[str]) -> Dict[str, List[str]]:
    """Return role -> list of matching column names (in column order)."""
    result: Dict[str, List[str]] = {role: [] for role in ROLE_PATTERNS}
    for col in columns:
        c = col.strip().lower()
        for role, pats in ROLE_PATTERNS.items():
            if any(re.search(p, c) for p in pats):
                result[role].append(col)
    return result


def primary_candidate(role_map: Dict[str, List[str]], role: str) -> Optional[str]:
    vals = role_map.get(role) or []
    return vals[0] if vals else None


def schema_signature(columns: Sequence[str]) -> str:
    """Order-insensitive signature of a column set (for schema-family grouping)."""
    import hashlib

    norm = sorted(str(c).strip().lower() for c in columns)
    return hashlib.sha256("|".join(norm).encode("utf-8")).hexdigest()[:12]


def ordered_signature(columns: Sequence[str]) -> str:
    """Order-sensitive signature (for detecting column-order variants)."""
    import hashlib

    norm = [str(c).strip().lower() for c in columns]
    return hashlib.sha256("|".join(norm).encode("utf-8")).hexdigest()[:12]
