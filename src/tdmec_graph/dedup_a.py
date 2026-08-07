"""Provisional Dataset A Layer-1 composite dedup (not dedup-certified)."""
from __future__ import annotations

import hashlib
from typing import Optional


def text_content_hash(cleaned_text: Optional[str]) -> str:
    payload = "" if cleaned_text is None else cleaned_text
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def composite_signature(
    *,
    author_account_id: str,
    created_at_utc_iso: str,
    text_hash: str,
    relation_id: int,
    target_account_id: str,
) -> str:
    """Deterministic composite key. Float tweet id is forbidden."""
    parts = (
        str(author_account_id),
        str(created_at_utc_iso),
        str(text_hash),
        str(int(relation_id)),
        str(target_account_id),
    )
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()
