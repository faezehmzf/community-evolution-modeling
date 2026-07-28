"""Explicit markers for evidence-dependent / unresolved configuration values.

Phase 1 must never silently replace these with guessed defaults.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, Optional, TypeVar

T = TypeVar("T")


class ResolutionGate(str, Enum):
    """When an unresolved scientific/engineering value may be finalized."""

    POST_DIAGNOSTIC = "POST_DIAGNOSTIC"
    POST_QEMB_PILOT = "POST_QEMB_PILOT"
    POST_CAL = "POST_CAL"
    REVIEW_REQUIRED_AFTER_REAL_DATA_DIAGNOSTICS = (
        "REVIEW_REQUIRED_AFTER_REAL_DATA_DIAGNOSTICS"
    )
    PENDING_PILOT_AND_USER_CONFIRMATION = "PENDING_PILOT_AND_USER_CONFIRMATION"
    USER_APPROVAL_REQUIRED = "USER_APPROVAL_REQUIRED"


@dataclass(frozen=True)
class UnresolvedValue(Generic[T]):
    """Typed placeholder for a value that must not be invented in Phase 1."""

    name: str
    gate: ResolutionGate
    provisional: Optional[T] = None
    notes: str = ""

    def resolved(self) -> bool:
        return False

    def require(self) -> T:
        raise ValueError(
            f"Unresolved value '{self.name}' (gate={self.gate.value}) "
            f"must not be treated as finalized. {self.notes}".strip()
        )

    def as_manifest_entry(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "gate": self.gate.value,
            "provisional": self.provisional,
            "notes": self.notes,
            "resolved": False,
        }
