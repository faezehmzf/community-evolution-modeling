"""TDMEC Phase 2: privacy-safe, resumable data diagnostics.

Produces diagnostic evidence for later user decisions. Does not finalize
QCAL-B01, QDEDUP-B01, numeric coverage thresholds, or QEMB-X01..X07.
Does not download embedding models or generate embeddings (Phase 3+).
"""
from __future__ import annotations

from tdmec_diagnostics.constants import (
    DIAGNOSTIC_COMPLETE,
    DIAGNOSTIC_SCHEMA_VERSION,
    REVIEW_REQUIRED,
    UNVALIDATED,
)
from tdmec_diagnostics.config import DiagnosticsConfig, load_diagnostics_config
from tdmec_diagnostics.pipeline import DiagnosticsPipeline, run_diagnostics
from tdmec_diagnostics.schema_contracts import (
    DATASET_A_ADAPTER_ID,
    DATASET_B_ADAPTER_ID,
)

__all__ = [
    "DIAGNOSTIC_COMPLETE",
    "DIAGNOSTIC_SCHEMA_VERSION",
    "REVIEW_REQUIRED",
    "UNVALIDATED",
    "DATASET_A_ADAPTER_ID",
    "DATASET_B_ADAPTER_ID",
    "DiagnosticsConfig",
    "DiagnosticsPipeline",
    "load_diagnostics_config",
    "run_diagnostics",
]

__version__ = "0.2.0-phase2"
