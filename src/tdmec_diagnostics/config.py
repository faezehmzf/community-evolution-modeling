"""Phase 2 diagnostics configuration (runtime-configurable calendar bounds)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from tdmec.hashing import hash_config
from tdmec.unresolved import ResolutionGate, UnresolvedValue
from tdmec_diagnostics import constants as DC
from tdmec_diagnostics.schema_contracts import (
    DATASET_A_ADAPTER_ID,
    DATASET_B_ADAPTER_ID,
)


@dataclass(frozen=True)
class DiagnosticsConfig:
    """Scientific + engineering configuration for Phase 2 diagnostics.

    Calendar bounds here are *provisional diagnostic* inputs only.
    They must not be treated as QCAL-B01 certified values.

    Paths and credentials are never part of the scientific hash.
    """

    schema_version: str = DC.DIAGNOSTIC_SCHEMA_VERSION
    method_contract_version: str = DC.METHOD_CONTRACT_VERSION

    # Provisional calendar (diagnostic-only; QCAL-B01-PROC)
    provisional_start_label: str = "2017-Q4"
    provisional_end_label: str = "2026-Q2"
    keep_internal_empty_quarters: bool = True
    timezone_assumption: str = DC.TIMEZONE_ASSUMPTION
    boundary_convention: str = DC.BOUNDARY_CONVENTION
    snapshot_frequency: str = DC.SNAPSHOT_FREQUENCY

    # Streaming / resume (engineering — excluded from scientific hash)
    chunk_size: int = DC.DEFAULT_CHUNK_SIZE
    enable_checkpoint: bool = True
    resume_mode: str = "resume"  # resume | restart

    # Text-length diagnostics
    quantiles: Tuple[float, ...] = DC.DEFAULT_QUANTILES
    candidate_max_lengths: Tuple[int, ...] = DC.CANDIDATE_MAX_LENGTHS
    enable_tokenizer_diagnostics: bool = False

    # Coverage
    relations: Tuple[str, ...] = ("mention", "retweet", "reply", "quote")
    # None => production N=16736; synthetic fixtures may override with tiny N
    node_universe_size: Optional[int] = None

    # Source scheme labels only (never embed secrets or absolute private paths)
    dataset_a_source_scheme: str = "unset"
    dataset_b_source_scheme: str = "unset"
    dataset_a_adapter_id: str = DATASET_A_ADAPTER_ID
    dataset_b_adapter_id: str = DATASET_B_ADAPTER_ID
    source_format: str = "xlsx"

    # Explicit unresolved markers (must remain unresolved in Phase 2)
    certified_calendar_start: UnresolvedValue[str] = field(
        default_factory=lambda: UnresolvedValue(
            name="certified_calendar_start",
            gate=ResolutionGate.REVIEW_REQUIRED_AFTER_REAL_DATA_DIAGNOSTICS,
            notes="QCAL-B01 requires real-data calendar report + user approval",
        )
    )
    certified_calendar_end: UnresolvedValue[str] = field(
        default_factory=lambda: UnresolvedValue(
            name="certified_calendar_end",
            gate=ResolutionGate.REVIEW_REQUIRED_AFTER_REAL_DATA_DIAGNOSTICS,
            notes="QCAL-B01 requires real-data calendar report + user approval",
        )
    )
    certified_T: UnresolvedValue[int] = field(
        default_factory=lambda: UnresolvedValue(
            name="certified_T",
            gate=ResolutionGate.REVIEW_REQUIRED_AFTER_REAL_DATA_DIAGNOSTICS,
            notes="Exact snapshot count T is POST_DIAGNOSTIC",
        )
    )
    leading_trailing_policy: UnresolvedValue[str] = field(
        default_factory=lambda: UnresolvedValue(
            name="leading_trailing_empty_policy",
            gate=ResolutionGate.REVIEW_REQUIRED_AFTER_REAL_DATA_DIAGNOSTICS,
            notes="QCAL-B01-PROC",
        )
    )
    dedup_signature_a: UnresolvedValue[str] = field(
        default_factory=lambda: UnresolvedValue(
            name="qdedup_b01_dataset_a_signature",
            gate=ResolutionGate.REVIEW_REQUIRED_AFTER_REAL_DATA_DIAGNOSTICS,
            notes="QDEDUP-B01 exact composite signature pending evidence review",
        )
    )
    dedup_l2_thresholds: UnresolvedValue[str] = field(
        default_factory=lambda: UnresolvedValue(
            name="qdedup_b01_l2_thresholds",
            gate=ResolutionGate.REVIEW_REQUIRED_AFTER_REAL_DATA_DIAGNOSTICS,
            notes="QDEDUP-B01 Layer-2 thresholds pending evidence review",
        )
    )
    coverage_hard_thresholds: UnresolvedValue[str] = field(
        default_factory=lambda: UnresolvedValue(
            name="coverage_hard_thresholds",
            gate=ResolutionGate.POST_DIAGNOSTIC,
            notes="Numeric certification thresholds remain unresolved",
        )
    )
    d_text: UnresolvedValue[int] = field(
        default_factory=lambda: UnresolvedValue(
            name="D_text",
            gate=ResolutionGate.POST_QEMB_PILOT,
            notes="Production D_text deferred to Phase 3+ Q-EMB pilot",
        )
    )
    # referenced_status extraction for Dataset A remains unresolved (not guessed)
    dataset_a_referenced_status_extraction: UnresolvedValue[str] = field(
        default_factory=lambda: UnresolvedValue(
            name="dataset_a_referenced_status_extraction",
            gate=ResolutionGate.REVIEW_REQUIRED_AFTER_REAL_DATA_DIAGNOSTICS,
            notes=(
                "Referenced-status id extraction is not silently guessed; "
                "left unresolved pending documented safe extraction rules"
            ),
        )
    )

    def __post_init__(self) -> None:
        if self.snapshot_frequency != DC.SNAPSHOT_FREQUENCY:
            raise ValueError("only quarterly frequency is canonical")
        if self.boundary_convention != DC.BOUNDARY_CONVENTION:
            raise ValueError(
                f"boundary_convention must be {DC.BOUNDARY_CONVENTION}"
            )
        if not self.keep_internal_empty_quarters:
            raise ValueError("internal empty quarters must be preserved")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.timezone_assumption != DC.TIMEZONE_ASSUMPTION:
            raise ValueError("Phase 2 diagnostics assume UTC timestamps")
        if self.resume_mode not in ("resume", "restart"):
            raise ValueError("resume_mode must be 'resume' or 'restart'")
        if self.source_format not in ("xlsx", "synthetic"):
            raise ValueError("source_format must be 'xlsx' or 'synthetic'")
        object.__setattr__(self, "quantiles", tuple(self.quantiles))
        object.__setattr__(
            self, "candidate_max_lengths", tuple(self.candidate_max_lengths)
        )
        object.__setattr__(self, "relations", tuple(self.relations))
        for name in (
            "certified_calendar_start",
            "certified_calendar_end",
            "certified_T",
            "leading_trailing_policy",
            "dedup_signature_a",
            "dedup_l2_thresholds",
            "coverage_hard_thresholds",
            "d_text",
            "dataset_a_referenced_status_extraction",
        ):
            val = getattr(self, name)
            if not isinstance(val, UnresolvedValue):
                raise ValueError(f"{name} must remain an UnresolvedValue in Phase 2")

    def scientific_dict(self) -> Dict[str, Any]:
        """Content that enters the scientific configuration hash.

        Excludes engineering knobs (chunk_size, checkpoint/resume) and all paths.
        """
        return {
            "schema_version": self.schema_version,
            "method_contract_version": self.method_contract_version,
            "provisional_start_label": self.provisional_start_label,
            "provisional_end_label": self.provisional_end_label,
            "keep_internal_empty_quarters": self.keep_internal_empty_quarters,
            "timezone_assumption": self.timezone_assumption,
            "boundary_convention": self.boundary_convention,
            "snapshot_frequency": self.snapshot_frequency,
            "quantiles": list(self.quantiles),
            "candidate_max_lengths": list(self.candidate_max_lengths),
            "enable_tokenizer_diagnostics": self.enable_tokenizer_diagnostics,
            "relations": list(self.relations),
            "node_universe_size": self.node_universe_size,
            "dataset_a_source_scheme": self.dataset_a_source_scheme,
            "dataset_b_source_scheme": self.dataset_b_source_scheme,
            "dataset_a_adapter_id": self.dataset_a_adapter_id,
            "dataset_b_adapter_id": self.dataset_b_adapter_id,
            "source_format": self.source_format,
            "unresolved": {
                "certified_calendar_start": self.certified_calendar_start.as_manifest_entry(),
                "certified_calendar_end": self.certified_calendar_end.as_manifest_entry(),
                "certified_T": self.certified_T.as_manifest_entry(),
                "leading_trailing_policy": self.leading_trailing_policy.as_manifest_entry(),
                "dedup_signature_a": self.dedup_signature_a.as_manifest_entry(),
                "dedup_l2_thresholds": self.dedup_l2_thresholds.as_manifest_entry(),
                "coverage_hard_thresholds": self.coverage_hard_thresholds.as_manifest_entry(),
                "d_text": self.d_text.as_manifest_entry(),
                "dataset_a_referenced_status_extraction": (
                    self.dataset_a_referenced_status_extraction.as_manifest_entry()
                ),
            },
        }

    def engineering_dict(self) -> Dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "enable_checkpoint": self.enable_checkpoint,
            "resume_mode": self.resume_mode,
        }

    def config_hash(self) -> str:
        return hash_config(self.scientific_dict())

    def to_dict(self) -> Dict[str, Any]:
        d = self.scientific_dict()
        d["engineering"] = self.engineering_dict()
        d["config_hash"] = self.config_hash()
        return d


def load_diagnostics_config(path: str | Path | None = None) -> DiagnosticsConfig:
    """Load diagnostics config from YAML, or return defaults."""
    if path is None:
        return DiagnosticsConfig()
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("diagnostics config must be a mapping")
    known = {
        "schema_version",
        "method_contract_version",
        "provisional_start_label",
        "provisional_end_label",
        "keep_internal_empty_quarters",
        "timezone_assumption",
        "boundary_convention",
        "snapshot_frequency",
        "chunk_size",
        "enable_checkpoint",
        "resume_mode",
        "quantiles",
        "candidate_max_lengths",
        "enable_tokenizer_diagnostics",
        "relations",
        "node_universe_size",
        "dataset_a_source_scheme",
        "dataset_b_source_scheme",
        "dataset_a_adapter_id",
        "dataset_b_adapter_id",
        "source_format",
    }
    kwargs: Dict[str, Any] = {}
    for k in known:
        if k in raw:
            kwargs[k] = raw[k]
    if "quantiles" in kwargs:
        kwargs["quantiles"] = tuple(kwargs["quantiles"])
    if "candidate_max_lengths" in kwargs:
        kwargs["candidate_max_lengths"] = tuple(kwargs["candidate_max_lengths"])
    if "relations" in kwargs:
        kwargs["relations"] = tuple(kwargs["relations"])
    return DiagnosticsConfig(**kwargs)
