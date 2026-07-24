"""Tweet-id and account-id normalization (exact, string-safe)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

_SNOWFLAKE_RE = re.compile(r"\d{15,20}")


@dataclass
class NormalizedId:
    value: Optional[str]
    ok: bool
    error: Optional[str] = None


def normalize_tweet_id(value: Any) -> NormalizedId:
    """Return an exact string tweet id, or reject.

    Floats are rejected because 64-bit snowflake ids lose precision as float64
    (Dataset A's failure mode); Dataset B stores ids as strings, so a float here
    signals corruption.
    """
    if value is None:
        return NormalizedId(None, False, "invalid_tweet_id")
    if isinstance(value, bool):
        return NormalizedId(None, False, "invalid_tweet_id")
    if isinstance(value, float):
        return NormalizedId(None, False, "invalid_tweet_id_float")
    if isinstance(value, int):
        s = str(value)
    elif isinstance(value, str):
        s = value.strip()
    else:
        return NormalizedId(None, False, "invalid_tweet_id")
    if not _SNOWFLAKE_RE.fullmatch(s):
        return NormalizedId(None, False, "invalid_tweet_id")
    return NormalizedId(s, True, None)


def normalize_account_id(value: Any) -> Optional[str]:
    """Canonical string form of an integer-compatible account id (or None)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        s = value.strip()
        return s if s.isdigit() else None
    return None
