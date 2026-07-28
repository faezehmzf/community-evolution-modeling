"""Diagnostic status transitions (Phase 2 never emits CERTIFIED)."""
from __future__ import annotations

from typing import Optional

from tdmec_diagnostics import constants as DC


class DiagnosticStatusError(ValueError):
    pass


def assert_not_certified(status: str) -> None:
    if status in DC.FORBIDDEN_PHASE2_STATUSES or "CERTIFIED" in str(status).upper():
        raise DiagnosticStatusError(
            f"Phase 2 diagnostic artifacts must not use status {status!r}"
        )


def can_transition(from_status: str, to_status: str) -> bool:
    assert_not_certified(from_status)
    assert_not_certified(to_status)
    if from_status not in DC.DIAGNOSTIC_STATUSES:
        return False
    if to_status not in DC.DIAGNOSTIC_STATUSES:
        return False
    return to_status in DC.ALLOWED_DIAGNOSTIC_TRANSITIONS[from_status]


def transition(from_status: str, to_status: str) -> str:
    if not can_transition(from_status, to_status):
        raise DiagnosticStatusError(
            f"invalid diagnostic status transition: {from_status} -> {to_status}"
        )
    return to_status


def finalize_run_status(
    *,
    complete: bool,
    has_hard_failures: bool,
    has_review_flags: bool,
) -> str:
    """Map run outcome to a Phase 2 status (never CERTIFIED)."""
    if has_hard_failures or not complete:
        return DC.UNVALIDATED
    if has_review_flags:
        return transition(DC.UNVALIDATED, DC.REVIEW_REQUIRED)
    return transition(DC.UNVALIDATED, DC.DIAGNOSTIC_COMPLETE)


def require_status(status: Optional[str]) -> str:
    if status is None:
        return DC.UNVALIDATED
    assert_not_certified(status)
    if status not in DC.DIAGNOSTIC_STATUSES:
        raise DiagnosticStatusError(f"unknown diagnostic status: {status!r}")
    return status
