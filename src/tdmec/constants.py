"""Canonical TDMEC Phase 1 constants (method-locked; not tunable).

Authority: docs/method/03, 12, 14, 17, 21; docs/handoff/07.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Tuple

# ---------------------------------------------------------------------------
# Node universe (D2)
# ---------------------------------------------------------------------------
N_NODES: int = 16_736
NODE_INDEX_MIN: int = 0
NODE_INDEX_MAX: int = 16_735  # inclusive

# ---------------------------------------------------------------------------
# Relations (QREL-01) — immutable canonical IDs
# ---------------------------------------------------------------------------
RELATION_MENTION: str = "mention"
RELATION_RETWEET: str = "retweet"
RELATION_REPLY: str = "reply"
RELATION_QUOTE: str = "quote"

RELATION_ORDER: Tuple[str, ...] = (
    RELATION_MENTION,
    RELATION_RETWEET,
    RELATION_REPLY,
    RELATION_QUOTE,
)

RELATION_TO_ID: Mapping[str, int] = MappingProxyType(
    {
        RELATION_MENTION: 0,
        RELATION_RETWEET: 1,
        RELATION_REPLY: 2,
        RELATION_QUOTE: 3,
    }
)

ID_TO_RELATION: Mapping[int, str] = MappingProxyType(
    {v: k for k, v in RELATION_TO_ID.items()}
)

RELATION_COUNT: int = 4
RELATION_ID_MIN: int = 0
RELATION_ID_MAX: int = 3

# ---------------------------------------------------------------------------
# Edge / self-loop / weight (QSELF-01, Q-WGT)
# ---------------------------------------------------------------------------
ALLOW_SELF_LOOPS: bool = False
EDGE_DIRECTION: str = "directed"  # no automatic symmetrization
WEIGHT_TRANSFORM: str = "log1p"  # natural log: ln(1 + count_raw)
# Absolute tolerance for weight_log1p vs math.log1p(count_raw) in float32/float64
WEIGHT_LOG1P_ATOL: float = 1e-6
WEIGHT_LOG1P_RTOL: float = 1e-6

# ---------------------------------------------------------------------------
# Structural features (Q-FEAT) — exact ordered schema, F_struct = 17
# ---------------------------------------------------------------------------
F_STRUCT: int = 17
STRUCT_FEATURE_SCHEMA_VERSION: str = "qfeat-17-v1"
STRUCT_FEATURE_DTYPE: str = "float32"

# Exact names and order from docs/method/03 / 12 (Q-FEAT). Do not reorder.
STRUCT_FEATURE_NAMES: Tuple[str, ...] = (
    "mention_out_degree",
    "mention_in_degree",
    "mention_out_strength_log1p",
    "mention_in_strength_log1p",
    "retweet_out_degree",
    "retweet_in_degree",
    "retweet_out_strength_log1p",
    "retweet_in_strength_log1p",
    "reply_out_degree",
    "reply_in_degree",
    "reply_out_strength_log1p",
    "reply_in_strength_log1p",
    "quote_out_degree",
    "quote_in_degree",
    "quote_out_strength_log1p",
    "quote_in_strength_log1p",
    "tweet_count_log1p",
)

assert len(STRUCT_FEATURE_NAMES) == F_STRUCT

# ---------------------------------------------------------------------------
# Calendar (Q-CAL / QCAL-B01-PROC) — quarterly; bounds evidence-dependent
# ---------------------------------------------------------------------------
SNAPSHOT_FREQUENCY: str = "quarterly"
BOUNDARY_CONVENTION: str = "start_inclusive_end_exclusive"
# Provisional diagnostic range only — not calendar-certified (QCAL-B01-PROC).
PROVISIONAL_CALENDAR_START_LABEL: str = "2017-Q4"
PROVISIONAL_CALENDAR_END_LABEL: str = "2026-Q2"
PROVISIONAL_SNAPSHOT_COUNT: int = 35  # diagnostic-only

# Timestamp reason codes (config-driven calendar; not finalized bounds)
TIMESTAMP_REASON_CODES: Tuple[str, ...] = (
    "IN_RANGE",
    "OUT_OF_RANGE_BEFORE",
    "OUT_OF_RANGE_AFTER",
    "INVALID_TIMESTAMP",
    "CORRUPT_TIMESTAMP",
    "EPOCH_OUTLIER",
)

# ---------------------------------------------------------------------------
# Certification vocabulary (QART-01-FRAME / docs/method/09)
# ---------------------------------------------------------------------------
CERT_UNVALIDATED: str = "UNVALIDATED"
CERT_VALIDATED: str = "VALIDATED"
CERT_CERTIFIED: str = "CERTIFIED"
CERTIFICATION_STATES: Tuple[str, ...] = (
    CERT_UNVALIDATED,
    CERT_VALIDATED,
    CERT_CERTIFIED,
)

# Allowed transitions (existence alone never grants CERTIFIED)
ALLOWED_CERTIFICATION_TRANSITIONS: Mapping[str, Tuple[str, ...]] = MappingProxyType(
    {
        CERT_UNVALIDATED: (CERT_VALIDATED,),
        CERT_VALIDATED: (CERT_CERTIFIED,),
        CERT_CERTIFIED: (),  # terminal for Phase 1 contract
    }
)

# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------
HASH_ALGORITHM: str = "sha256"
METHOD_CONTRACT_VERSION: str = "tdmec-preimpl-2026-07-28"

# ---------------------------------------------------------------------------
# Model / training / evaluation defaults (configuration contracts only)
# ---------------------------------------------------------------------------
# Confirmed primary experimental defaults (docs/method/05, 06, 12) — config only.
DEFAULT_D_H: int = 64
DEFAULT_K: int = 10
DEFAULT_D_REL: int = 16
DEFAULT_L_LAYERS: int = 1
DEFAULT_FANOUT: Tuple[int, ...] = (15,)
DEFAULT_ALPHA_STUDENT_T: float = 1.0
DEFAULT_DROPOUT: float = 0.0
DEFAULT_BPTT: int = 3
DEFAULT_ADAMW_LR: float = 5e-4
DEFAULT_WEIGHT_DECAY: float = 1e-4
DEFAULT_GRAD_CLIP: float = 1.0
DEFAULT_PATIENCE: int = 20
DEFAULT_LAMBDA_STRUCT: float = 1.0
DEFAULT_LAMBDA_SEM: float = 1.0
DEFAULT_LAMBDA_CLUSTER: float = 1.0
DEFAULT_LAMBDA_REG: float = 0.1
DEFAULT_LAMBDA_TEMP: float = 0.1
DEFAULT_PHASE_PRETRAIN_EPOCHS: int = 100
DEFAULT_PHASE_JOINT_EPOCHS: int = 80
DEFAULT_PHASE_TEMP_EPOCHS: int = 120
DEFAULT_TEMP_RAMP_EPOCHS: int = 24
DEFAULT_N_SEEDS: int = 5

# D_text remains unresolved until Q-EMB pilot — never hard-coded as final.
D_TEXT_UNRESOLVED_NAME: str = "D_text"
