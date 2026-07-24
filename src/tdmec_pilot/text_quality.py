"""Conservative text normalization + quality classification.

Non-destructive: ``text_raw`` is preserved verbatim; ``text_normalized`` only
applies Unicode NFC + newline canonicalization + BOM strip. No characters,
mentions, hashtags, urls, emojis, punctuation, or stopwords are removed; no
stemming, lemmatization, or translation.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class TextResult:
    text_raw: Optional[str]
    text_normalized: Optional[str]
    quality: str  # "ok" | "empty" | "whitespace_only" | "non_string"
    raw_len: int
    normalized_len: int


def normalize_text(value: Any, unicode_form: str = "NFC",
                   normalize_newlines: bool = True, strip_bom: bool = True) -> TextResult:
    if value is None:
        return TextResult(None, None, "empty", 0, 0)
    if not isinstance(value, str):
        value = str(value)
    raw = value
    if raw == "":
        return TextResult(raw, raw, "empty", 0, 0)

    norm = raw
    if strip_bom:
        norm = norm.replace("\ufeff", "")
    if normalize_newlines:
        norm = norm.replace("\r\n", "\n").replace("\r", "\n")
    if unicode_form:
        norm = unicodedata.normalize(unicode_form, norm)

    if norm.strip() == "":
        quality = "whitespace_only"
    else:
        quality = "ok"
    return TextResult(raw, norm, quality, len(raw), len(norm))
