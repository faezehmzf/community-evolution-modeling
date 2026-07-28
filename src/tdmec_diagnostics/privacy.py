"""Privacy guards for Phase 2 diagnostic reports.

Reports must never contain raw text, external user identifiers, secrets,
tokens, or private absolute paths.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Set

from tdmec.validation.findings import FORBIDDEN_CONTEXT_KEYS, _ABS_PATH_RE, _EMAIL_RE

# Extra keys forbidden in Phase 2 diagnostic outputs
PHASE2_FORBIDDEN_KEYS: frozenset[str] = frozenset(
    set(FORBIDDEN_CONTEXT_KEYS)
    | {
        "tweet_id",
        "author_id",
        "account_id",
        "username",
        "screen_name",
        "source_user_id",
        "target_user_id",
        "external_user_id",
        "text_raw",
        "text_normalized",
        "cleaned_text",
        "ocr_text",
        "user_blob",
        "full_text",
        "content",
        "password",
        "secret",
        "credentials",
        "api_key",
        "access_token",
        "refresh_token",
        "service_account",
        "private_key",
    }
)

_LONG_TEXT_THRESHOLD = 200


def privacy_safe_file_ref(name: str) -> str:
    """Return a privacy-safe file reference (basename only, no absolute path)."""
    s = str(name).replace("\\", "/")
    if "/" in s:
        s = s.rsplit("/", 1)[-1]
    if _ABS_PATH_RE.search(str(name)):
        # Hash absolute path; keep only extension hint
        digest = hashlib.sha256(str(name).encode("utf-8")).hexdigest()[:12]
        ext = ""
        if "." in s:
            ext = "." + s.rsplit(".", 1)[-1]
        return f"file-{digest}{ext}"
    return s


def hash_identifier(value: Any, *, prefix: str = "id") -> str:
    """Deterministic short hash of an identifier (never emit raw ID)."""
    raw = "" if value is None else str(value)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def hash_text_span(value: Any) -> str:
    """Privacy-safe content fingerprint for reports (not reversible)."""
    if value is None:
        return "text:null"
    raw = str(value)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"text:{digest}"


def assert_privacy_safe_mapping(
    obj: Mapping[str, Any],
    *,
    path: str = "root",
    extra_forbidden: Optional[Iterable[str]] = None,
) -> None:
    """Raise ValueError if *obj* (recursively) contains private content."""
    forbidden: Set[str] = set(PHASE2_FORBIDDEN_KEYS)
    if extra_forbidden:
        forbidden.update(str(x).lower() for x in extra_forbidden)

    def _walk(node: Any, cur: str) -> None:
        if isinstance(node, Mapping):
            for k, v in node.items():
                key = str(k)
                if key.lower() in forbidden or key in PHASE2_FORBIDDEN_KEYS:
                    raise ValueError(
                        f"privacy violation at {cur}.{key}: forbidden key"
                    )
                if isinstance(v, str):
                    _check_string(v, f"{cur}.{key}")
                else:
                    _walk(v, f"{cur}.{key}")
        elif isinstance(node, (list, tuple)):
            for i, item in enumerate(node):
                _walk(item, f"{cur}[{i}]")
        elif isinstance(node, str):
            _check_string(node, cur)

    def _check_string(value: str, cur: str) -> None:
        if _EMAIL_RE.search(value):
            raise ValueError(f"privacy violation at {cur}: email-like content")
        if _ABS_PATH_RE.search(value):
            raise ValueError(f"privacy violation at {cur}: absolute path")
        # Heuristic: very long strings look like leaked raw text
        if len(value) > _LONG_TEXT_THRESHOLD and " " in value and not value.startswith(
            ("sha256:", "id:", "text:", "file-", "cfg:", "run:")
        ):
            raise ValueError(f"privacy violation at {cur}: possible raw text leak")

    _walk(obj, path)


def redact_absolute_paths(message: str) -> str:
    """Replace absolute path substrings with a placeholder."""
    return _ABS_PATH_RE.sub("<redacted-path>", message)


def ensure_no_raw_identifiers(
    report: MutableMapping[str, Any],
    *,
    known_raw_ids: Iterable[str],
) -> None:
    """Fail if any known raw external ID appears as a substring in JSON-ish values."""
    raw_ids = [str(x) for x in known_raw_ids if x is not None and str(x) != ""]
    if not raw_ids:
        return

    def _scan(node: Any, cur: str) -> None:
        if isinstance(node, Mapping):
            for k, v in node.items():
                _scan(v, f"{cur}.{k}")
        elif isinstance(node, (list, tuple)):
            for i, item in enumerate(node):
                _scan(item, f"{cur}[{i}]")
        elif isinstance(node, str):
            for rid in raw_ids:
                # Skip very short tokens to avoid false positives on digits like "1"
                if len(rid) < 6:
                    continue
                if rid in node:
                    raise ValueError(
                        f"privacy violation at {cur}: raw identifier leak"
                    )

    _scan(report, "report")


def sanitize_warning_message(message: str) -> str:
    msg = redact_absolute_paths(str(message))
    if _EMAIL_RE.search(msg):
        msg = _EMAIL_RE.sub("<redacted-email>", msg)
    return msg
