"""Strict, safe parser for the Dataset B ``user`` blob.

Observed form (Python-literal, single quotes, ``None``):
    {'id': 1662724992054829056, 'followers': 235, 'username': 'x',
     'title': "…", 'political_category': None}

JSON-like variants (double quotes, ``null``) are also handled. Parsing is done
with ``json.loads`` then ``ast.literal_eval`` (a safe literal evaluator) — never
``eval`` and never any code execution. Malformed / missing values are recorded,
not raised.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any, Optional

# Fast fallback extractor used only to cross-check / recover the account id when
# structured parsing fails; it still never executes anything.
_ID_RE = re.compile(r"['\"]id['\"]\s*:\s*(\d+)")
_USERNAME_RE = re.compile(r"['\"](?:username|screen_name)['\"]\s*:\s*['\"]([^'\"]*)['\"]")


@dataclass
class ParsedUser:
    account_id: Optional[str]  # canonical string form of the integer id
    username: Optional[str]
    ok: bool
    error: Optional[str] = None


def _coerce_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        # floats are precision-lossy for 64-bit ids -> reject
        return None
    if isinstance(value, str):
        s = value.strip()
        if re.fullmatch(r"\d+", s):
            return s
        return None
    return None


def parse_user_blob(value: Any) -> ParsedUser:
    """Parse a ``user`` cell into an exact account id + optional username."""
    if value is None:
        return ParsedUser(None, None, False, "missing_user")
    if isinstance(value, dict):
        obj = value
    elif isinstance(value, str):
        s = value.strip()
        if s == "" or s.lower() in {"none", "null", "nan"}:
            return ParsedUser(None, None, False, "missing_user")
        obj = _safe_load(s)
        if obj is None:
            # structured parse failed: attempt regex recovery of the id only
            m = _ID_RE.search(s)
            if m:
                mu = _USERNAME_RE.search(s)
                return ParsedUser(m.group(1), mu.group(1) if mu else None, True,
                                  "recovered_by_regex")
            return ParsedUser(None, None, False, "malformed_user_blob")
    else:
        return ParsedUser(None, None, False, "unexpected_user_type")

    if not isinstance(obj, dict):
        return ParsedUser(None, None, False, "user_not_object")

    account_id = _coerce_id(obj.get("id"))
    username = obj.get("username")
    if username is None:
        username = obj.get("screen_name")
    if username is not None and not isinstance(username, str):
        username = str(username)
    if account_id is None:
        return ParsedUser(None, username, False, "missing_user_id")
    return ParsedUser(account_id, username, True, None)


def _safe_load(s: str) -> Optional[dict]:
    # Try JSON first (double-quoted / null), then Python literal (single-quoted / None).
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return ast.literal_eval(s)
    except Exception:
        return None
