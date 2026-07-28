"""Privacy-oriented diagnostic record types (in-memory only).

Raw text and external IDs may exist transiently while computing aggregates.
They must never be written into diagnostic reports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class DiagnosticEventRecord:
    """One event/row presented to diagnostic engines.

    ``external_user_id`` / ``tweet_id`` / ``text`` are processing inputs only.
    """

    dataset: str  # "A" | "B"
    source_file: str
    source_row_number: int
    timestamp_raw: Any = None
    external_user_id: Optional[str] = None
    tweet_id: Optional[str] = None
    text: Optional[str] = None
    relation: Optional[str] = None
    target_external_user_id: Optional[str] = None
    referenced_status_id: Optional[str] = None
    node_idx: Optional[int] = None  # mapped into frozen universe when known
    target_node_idx: Optional[int] = None
    struct_active: bool = False
    node_text_available: bool = False
    edge_text_available: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def content_fingerprint_fields(self) -> Tuple[Any, ...]:
        """Fields used for concordant/discordant content comparison (hashed later)."""
        return (
            self.text if self.text is not None else "",
            self.relation or "",
            self.target_external_user_id or "",
            self.referenced_status_id or "",
        )

    def composite_key_fields(self) -> Tuple[Any, ...]:
        """Candidate Dataset A composite signature fields (QDEDUP-B01 candidate)."""
        return (
            self.external_user_id or "",
            "" if self.timestamp_raw is None else str(self.timestamp_raw),
            self.text if self.text is not None else "",
            self.relation or "",
            self.target_external_user_id or "",
            self.referenced_status_id or "",
        )

    def full_row_fingerprint_fields(self) -> Tuple[Any, ...]:
        return (
            self.dataset,
            self.tweet_id or "",
            self.external_user_id or "",
            "" if self.timestamp_raw is None else str(self.timestamp_raw),
            self.text if self.text is not None else "",
            self.relation or "",
            self.target_external_user_id or "",
            self.referenced_status_id or "",
        )
