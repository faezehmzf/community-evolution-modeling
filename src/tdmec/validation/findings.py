"""Machine-readable validation findings and reports (privacy-safe)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional


class Severity(str, Enum):
    HARD_FAILURE = "HARD_FAILURE"
    WARNING = "WARNING"
    INFO = "INFO"


# Keys that must never appear in validation finding context.
FORBIDDEN_CONTEXT_KEYS = frozenset(
    {
        "raw_text",
        "private_text",
        "cleaned_text",
        "text",
        "tweet_text",
        "user_blob",
        "author_account_id",
        "external_identifier",
        "external_id",
        "user_id",
        "email",
        "email_address",
        "token",
        "access_token",
        "api_key",
        "password",
        "secret",
        "credentials",
        "private_path",
        "absolute_path",
        "local_path",
    }
)

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_ABS_PATH_RE = re.compile(r"(?:/home/|/Users/|/workspace/|/content/|[A-Za-z]:\\)")


def _looks_private_string(value: str) -> bool:
    if len(value) > 500:
        return True
    if _EMAIL_RE.search(value):
        return True
    if _ABS_PATH_RE.search(value):
        return True
    return False


def sanitize_context(context: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a copy of context with forbidden keys rejected (raises on violation)."""
    bad = FORBIDDEN_CONTEXT_KEYS.intersection(context.keys())
    if bad:
        raise ValueError(
            f"validation context must not expose private fields: {sorted(bad)}"
        )
    out: Dict[str, Any] = {}
    for k, v in context.items():
        key = str(k)
        if key.lower() in FORBIDDEN_CONTEXT_KEYS:
            raise ValueError(
                f"validation context must not expose private fields: [{key}]"
            )
        if isinstance(v, str) and _looks_private_string(v):
            raise ValueError(
                f"context field {key!r} contains disallowed private content; redacted"
            )
        out[key] = v
    return out


@dataclass(frozen=True)
class ValidationFinding:
    code: str
    invariant: str
    message: str
    severity: Severity
    context: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if _EMAIL_RE.search(self.message) or _ABS_PATH_RE.search(self.message):
            raise ValueError(
                "validation message must not expose emails or absolute local paths"
            )
        object.__setattr__(self, "context", sanitize_context(dict(self.context)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "invariant": self.invariant,
            "message": self.message,
            "severity": self.severity.value,
            "context": dict(self.context),
        }


@dataclass
class ValidationReport:
    findings: List[ValidationFinding] = field(default_factory=list)
    schema_version: str = "phase1-validation-report-v1"

    def add(self, finding: ValidationFinding) -> None:
        self.findings.append(finding)

    def extend(self, findings: List[ValidationFinding]) -> None:
        self.findings.extend(findings)

    @property
    def hard_failures(self) -> List[ValidationFinding]:
        return [f for f in self.findings if f.severity == Severity.HARD_FAILURE]

    @property
    def warnings(self) -> List[ValidationFinding]:
        return [f for f in self.findings if f.severity == Severity.WARNING]

    @property
    def ok(self) -> bool:
        return not self.hard_failures

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "ok": self.ok,
            "n_hard_failures": len(self.hard_failures),
            "n_warnings": len(self.warnings),
            "findings": [f.to_dict() for f in self.findings],
        }

    def raise_if_failed(self) -> None:
        if self.hard_failures:
            codes = ", ".join(f.code for f in self.hard_failures)
            raise ValidationError(
                f"validation hard failures: {codes}", report=self
            )


class ValidationError(Exception):
    def __init__(self, message: str, report: Optional[ValidationReport] = None):
        super().__init__(message)
        self.report = report
