"""Phase 2 diagnostic vocabulary and schema versions.

Statuses intentionally exclude CERTIFIED. Phase 2 emits evidence only.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Tuple

from tdmec import constants as C

DIAGNOSTIC_SCHEMA_VERSION: str = "tdmec-phase2-diagnostics-v1"
METHOD_CONTRACT_VERSION: str = C.METHOD_CONTRACT_VERSION

# ---------------------------------------------------------------------------
# Diagnostic run / artifact status (never CERTIFIED in Phase 2)
# ---------------------------------------------------------------------------
UNVALIDATED: str = "UNVALIDATED"
DIAGNOSTIC_COMPLETE: str = "DIAGNOSTIC_COMPLETE"
REVIEW_REQUIRED: str = "REVIEW_REQUIRED"

DIAGNOSTIC_STATUSES: Tuple[str, ...] = (
    UNVALIDATED,
    DIAGNOSTIC_COMPLETE,
    REVIEW_REQUIRED,
)

ALLOWED_DIAGNOSTIC_TRANSITIONS: Mapping[str, Tuple[str, ...]] = MappingProxyType(
    {
        UNVALIDATED: (DIAGNOSTIC_COMPLETE, REVIEW_REQUIRED),
        DIAGNOSTIC_COMPLETE: (REVIEW_REQUIRED,),
        REVIEW_REQUIRED: (),  # terminal for Phase 2 tooling
    }
)

# Explicit rejection of certification vocabulary in Phase 2 outputs
FORBIDDEN_PHASE2_STATUSES: Tuple[str, ...] = ("CERTIFIED", "calendar-certified", "dedup-certified")

# ---------------------------------------------------------------------------
# Timestamp / quarter reason codes (align + extend Phase 1)
# ---------------------------------------------------------------------------
REASON_IN_RANGE: str = "IN_RANGE"
REASON_OUT_BEFORE: str = "OUT_OF_RANGE_BEFORE"
REASON_OUT_AFTER: str = "OUT_OF_RANGE_AFTER"
REASON_INVALID: str = "INVALID_TIMESTAMP"
REASON_UNPARSABLE: str = "UNPARSABLE_TIMESTAMP"
REASON_MISSING: str = "MISSING_TIMESTAMP"
REASON_CORRUPT: str = "CORRUPT_TIMESTAMP"
REASON_EPOCH_OUTLIER: str = "EPOCH_OUTLIER"

TIMESTAMP_REASON_CODES: Tuple[str, ...] = (
    REASON_IN_RANGE,
    REASON_OUT_BEFORE,
    REASON_OUT_AFTER,
    REASON_INVALID,
    REASON_UNPARSABLE,
    REASON_MISSING,
    REASON_CORRUPT,
    REASON_EPOCH_OUTLIER,
)

TIMEZONE_ASSUMPTION: str = "UTC"
BOUNDARY_CONVENTION: str = C.BOUNDARY_CONVENTION  # start_inclusive_end_exclusive
SNAPSHOT_FREQUENCY: str = C.SNAPSHOT_FREQUENCY

# ---------------------------------------------------------------------------
# Dedup classification codes
# ---------------------------------------------------------------------------
DEDUP_EXACT_FULL_ROW: str = "exact_full_row"
DEDUP_COMPOSITE_CONCORDANT: str = "composite_concordant"
DEDUP_COMPOSITE_DISCORDANT: str = "composite_discordant"
DEDUP_SAME_ID_CONCORDANT: str = "same_id_concordant"
DEDUP_SAME_ID_DISCORDANT: str = "same_id_discordant"
DEDUP_CROSS_FILE: str = "cross_file_repeat"
DEDUP_WITHIN_FILE: str = "within_file_repeat"
DEDUP_NULL_KEY: str = "null_or_malformed_key"
DEDUP_CONFLICTING_METADATA: str = "conflicting_metadata"

# ---------------------------------------------------------------------------
# Coverage category codes (QACT-01 / Q-MISS)
# ---------------------------------------------------------------------------
COV_STRUCTURE_ONLY: str = "structure_only"
COV_NODE_TEXT_ONLY: str = "node_text_only"
COV_STRUCTURE_AND_NODE_TEXT: str = "structure_and_node_text"
COV_INACTIVE: str = "fully_inactive"
COV_EDGE_TEXT_AVAILABLE: str = "edge_text_available"
COV_EDGE_TEXT_UNAVAILABLE: str = "edge_text_unavailable"

# ---------------------------------------------------------------------------
# Text quality codes
# ---------------------------------------------------------------------------
TEXT_OK: str = "ok"
TEXT_EMPTY: str = "empty"
TEXT_NULL: str = "null"
TEXT_WHITESPACE_ONLY: str = "whitespace_only"
TEXT_NON_STRING: str = "non_string"

# ---------------------------------------------------------------------------
# Report artifact names
# ---------------------------------------------------------------------------
REPORT_CALENDAR: str = "calendar_report"
REPORT_DEDUP: str = "dedup_report"
REPORT_TEXT_LENGTH: str = "text_length_report"
REPORT_COVERAGE: str = "coverage_report"
REPORT_MANIFEST: str = "execution_manifest"
REPORT_SUMMARY: str = "run_summary"
REPORT_WARNINGS: str = "warnings_and_failures"
REPORT_UNRESOLVED: str = "unresolved_decision_evidence"

DEFAULT_CHUNK_SIZE: int = 10_000
DEFAULT_QUANTILES: Tuple[float, ...] = (0.5, 0.9, 0.95, 0.99, 0.999)
CANDIDATE_MAX_LENGTHS: Tuple[int, ...] = (128, 256, 512, 1024, 2048)

# Decision IDs this phase produces evidence for (not finalizes)
EVIDENCE_DECISION_IDS: Tuple[str, ...] = (
    "QCAL-B01",
    "QCAL-B01-PROC",
    "QDEDUP-B01",
    "QDEDUP-B01-PROC",
    "Q-MISS",
    "QACT-01",
    "QART-01-FRAME",
    "QEMB-LENGTH-DIAG",  # length evidence only; QEMB-X01..X07 unresolved
)
