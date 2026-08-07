"""Calendar diagnostics (QCAL-B01 evidence; does not finalize bounds)."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from tdmec_diagnostics import constants as DC
from tdmec_diagnostics.privacy import privacy_safe_file_ref
from tdmec_diagnostics.quarters import (
    QuarterBoundary,
    build_quarter_range,
    candidate_T,
    classify_timestamp,
    datetime_to_quarter_label,
)
from tdmec_diagnostics.records import DiagnosticEventRecord
from tdmec_diagnostics.status import assert_not_certified


@dataclass
class CalendarAccumulator:
    """Streaming accumulator for calendar diagnostics."""

    boundaries: List[QuarterBoundary]
    timezone_assumption: str = DC.TIMEZONE_ASSUMPTION
    boundary_convention: str = DC.BOUNDARY_CONVENTION

    rows_inspected: int = 0
    rows_accepted: int = 0
    rows_rejected: int = 0

    reason_counts: Counter = field(default_factory=Counter)
    per_quarter_counts_a: Counter = field(default_factory=Counter)
    per_quarter_counts_b: Counter = field(default_factory=Counter)
    per_quarter_nodes_a: Dict[int, Set[int]] = field(
        default_factory=lambda: defaultdict(set)
    )
    per_quarter_nodes_b: Dict[int, Set[int]] = field(
        default_factory=lambda: defaultdict(set)
    )
    # Bounded memory: track only min/max valid epochs (not all timestamps)
    min_valid_epoch: Optional[int] = None
    max_valid_epoch: Optional[int] = None
    files_seen: Set[str] = field(default_factory=set)

    def observe(self, record: DiagnosticEventRecord) -> str:
        self.rows_inspected += 1
        self.files_seen.add(privacy_safe_file_ref(record.source_file))
        utc, sid, reason = classify_timestamp(
            record.timestamp_raw, boundaries=self.boundaries
        )
        self.reason_counts[reason] += 1

        if reason in (
            DC.REASON_MISSING,
            DC.REASON_UNPARSABLE,
            DC.REASON_INVALID,
            DC.REASON_CORRUPT,
            DC.REASON_EPOCH_OUTLIER,
        ):
            self.rows_rejected += 1
            return reason

        # Valid parse (may be out of provisional range)
        if utc is not None:
            epoch = int(utc.timestamp())
            if self.min_valid_epoch is None or epoch < self.min_valid_epoch:
                self.min_valid_epoch = epoch
            if self.max_valid_epoch is None or epoch > self.max_valid_epoch:
                self.max_valid_epoch = epoch

        if reason == DC.REASON_IN_RANGE and sid is not None:
            self.rows_accepted += 1
            if record.dataset.upper() == "A":
                self.per_quarter_counts_a[sid] += 1
                if record.node_idx is not None:
                    self.per_quarter_nodes_a[sid].add(record.node_idx)
            elif record.dataset.upper() == "B":
                self.per_quarter_counts_b[sid] += 1
                if record.node_idx is not None:
                    self.per_quarter_nodes_b[sid].add(record.node_idx)
            else:
                self.rows_rejected += 1
                return "UNKNOWN_DATASET"
        else:
            # Out of provisional range — counted but not accepted into registry bins
            self.rows_rejected += 1
        return reason

    def observe_many(self, records: Iterable[DiagnosticEventRecord]) -> None:
        for r in records:
            self.observe(r)

    def to_state(self) -> Dict[str, Any]:
        """Privacy-safe serializable state (aggregates/counts/file refs only)."""
        return {
            "timezone_assumption": self.timezone_assumption,
            "boundary_convention": self.boundary_convention,
            "provisional_start": (
                self.boundaries[0].label if self.boundaries else None
            ),
            "provisional_end": (
                self.boundaries[-1].label if self.boundaries else None
            ),
            "rows_inspected": self.rows_inspected,
            "rows_accepted": self.rows_accepted,
            "rows_rejected": self.rows_rejected,
            "reason_counts": {
                k: int(self.reason_counts[k]) for k in sorted(self.reason_counts)
            },
            "per_quarter_counts_a": {
                str(k): int(v) for k, v in sorted(self.per_quarter_counts_a.items())
            },
            "per_quarter_counts_b": {
                str(k): int(v) for k, v in sorted(self.per_quarter_counts_b.items())
            },
            "per_quarter_nodes_a": {
                str(sid): sorted(nodes)
                for sid, nodes in sorted(self.per_quarter_nodes_a.items())
            },
            "per_quarter_nodes_b": {
                str(sid): sorted(nodes)
                for sid, nodes in sorted(self.per_quarter_nodes_b.items())
            },
            "min_valid_epoch": self.min_valid_epoch,
            "max_valid_epoch": self.max_valid_epoch,
            "files_seen": sorted(self.files_seen),
        }

    @classmethod
    def from_state(
        cls,
        state: Dict[str, Any],
        *,
        boundaries: List[QuarterBoundary],
    ) -> "CalendarAccumulator":
        """Reconstruct an equivalent accumulator for continued observation."""
        acc = cls(
            boundaries=boundaries,
            timezone_assumption=str(
                state.get("timezone_assumption", DC.TIMEZONE_ASSUMPTION)
            ),
            boundary_convention=str(
                state.get("boundary_convention", DC.BOUNDARY_CONVENTION)
            ),
        )
        acc.rows_inspected = int(state.get("rows_inspected", 0))
        acc.rows_accepted = int(state.get("rows_accepted", 0))
        acc.rows_rejected = int(state.get("rows_rejected", 0))
        for k, v in (state.get("reason_counts") or {}).items():
            acc.reason_counts[k] = int(v)
        for k, v in (state.get("per_quarter_counts_a") or {}).items():
            acc.per_quarter_counts_a[int(k)] = int(v)
        for k, v in (state.get("per_quarter_counts_b") or {}).items():
            acc.per_quarter_counts_b[int(k)] = int(v)
        for sid, nodes in (state.get("per_quarter_nodes_a") or {}).items():
            acc.per_quarter_nodes_a[int(sid)] = set(int(n) for n in nodes)
        for sid, nodes in (state.get("per_quarter_nodes_b") or {}).items():
            acc.per_quarter_nodes_b[int(sid)] = set(int(n) for n in nodes)
        min_ep = state.get("min_valid_epoch")
        max_ep = state.get("max_valid_epoch")
        acc.min_valid_epoch = int(min_ep) if min_ep is not None else None
        acc.max_valid_epoch = int(max_ep) if max_ep is not None else None
        acc.files_seen = set(state.get("files_seen") or [])
        return acc

    def _empty_quarters(self) -> Dict[str, List[str]]:
        labels = [b.label for b in self.boundaries]
        counts = []
        for b in self.boundaries:
            c = self.per_quarter_counts_a[b.snapshot_id] + self.per_quarter_counts_b[
                b.snapshot_id
            ]
            counts.append(c)

        internal_empty: List[str] = []
        leading_empty: List[str] = []
        trailing_empty: List[str] = []

        # Leading empties
        i = 0
        while i < len(counts) and counts[i] == 0:
            leading_empty.append(labels[i])
            i += 1
        # Trailing empties
        j = len(counts) - 1
        while j >= i and counts[j] == 0:
            trailing_empty.append(labels[j])
            j -= 1
        trailing_empty.reverse()
        # Internal empties between first and last non-empty
        for k in range(i, j + 1):
            if counts[k] == 0:
                internal_empty.append(labels[k])

        return {
            "leading_empty_quarters": leading_empty,
            "trailing_empty_quarters": trailing_empty,
            "internal_empty_quarters": internal_empty,
        }

    def build_report(
        self,
        *,
        config_hash: str,
        status: str = DC.DIAGNOSTIC_COMPLETE,
    ) -> Dict[str, Any]:
        assert_not_certified(status)
        empties = self._empty_quarters()

        min_ts = self.min_valid_epoch
        max_ts = self.max_valid_epoch
        min_q = None
        max_q = None
        if min_ts is not None:
            import datetime as dt

            min_q = datetime_to_quarter_label(
                dt.datetime.fromtimestamp(min_ts, tz=dt.timezone.utc)
            )
            max_q = datetime_to_quarter_label(
                dt.datetime.fromtimestamp(max_ts, tz=dt.timezone.utc)
            )

        # Candidate certified bounds: first/last quarters with any records,
        # preserving internal empties inside that span. Leading/trailing policy
        # remains REVIEW_REQUIRED (not finalized).
        first_nonempty = None
        last_nonempty = None
        for b in self.boundaries:
            c = (
                self.per_quarter_counts_a[b.snapshot_id]
                + self.per_quarter_counts_b[b.snapshot_id]
            )
            if c > 0:
                if first_nonempty is None:
                    first_nonempty = b.label
                last_nonempty = b.label

        cand_start = first_nonempty
        cand_end = last_nonempty
        cand_t = None
        if cand_start and cand_end:
            cand_t = candidate_T(build_quarter_range(cand_start, cand_end))

        per_quarter = []
        for b in self.boundaries:
            per_quarter.append(
                {
                    "snapshot_id": b.snapshot_id,
                    "label": b.label,
                    "dataset_a_record_count": int(
                        self.per_quarter_counts_a[b.snapshot_id]
                    ),
                    "dataset_b_record_count": int(
                        self.per_quarter_counts_b[b.snapshot_id]
                    ),
                    "dataset_a_unique_node_idx_count": len(
                        self.per_quarter_nodes_a[b.snapshot_id]
                    ),
                    "dataset_b_unique_node_idx_count": len(
                        self.per_quarter_nodes_b[b.snapshot_id]
                    ),
                    "cross_dataset_covered": (
                        self.per_quarter_counts_a[b.snapshot_id] > 0
                        and self.per_quarter_counts_b[b.snapshot_id] > 0
                    ),
                }
            )

        recommendation = {
            "candidate_certified_start_quarter": cand_start,
            "candidate_certified_end_quarter": cand_end,
            "candidate_T": cand_t,
            "preserve_internal_empty_quarters": True,
            "leading_trailing_policy": "REVIEW_REQUIRED",
            "notes": (
                "Candidates are evidence-derived only. QCAL-B01 remains unresolved "
                "until user review of real-data reports. Do not treat as CERTIFIED."
            ),
        }

        report = {
            "schema_version": DC.DIAGNOSTIC_SCHEMA_VERSION,
            "report_type": DC.REPORT_CALENDAR,
            "status": status,
            "run_configuration_hash": config_hash,
            "timezone_assumption": self.timezone_assumption,
            "boundary_convention": self.boundary_convention,
            "snapshot_frequency": DC.SNAPSHOT_FREQUENCY,
            "provisional_quarter_range": {
                "start": self.boundaries[0].label if self.boundaries else None,
                "end": self.boundaries[-1].label if self.boundaries else None,
                "count": len(self.boundaries),
            },
            "rows_inspected": self.rows_inspected,
            "rows_accepted_for_diagnostics": self.rows_accepted,
            "rows_rejected": self.rows_rejected,
            "reason_counts": {
                k: int(self.reason_counts[k]) for k in sorted(self.reason_counts)
            },
            "minimum_valid_timestamp_epoch": min_ts,
            "maximum_valid_timestamp_epoch": max_ts,
            "minimum_valid_quarter": min_q,
            "maximum_valid_quarter": max_q,
            "observed_vs_provisional": {
                "observed_min_quarter": min_q,
                "observed_max_quarter": max_q,
                "provisional_start": (
                    self.boundaries[0].label if self.boundaries else None
                ),
                "provisional_end": (
                    self.boundaries[-1].label if self.boundaries else None
                ),
            },
            "internal_empty_quarters": empties["internal_empty_quarters"],
            "leading_empty_quarters": empties["leading_empty_quarters"],
            "trailing_empty_quarters": empties["trailing_empty_quarters"],
            "per_quarter_record_counts": per_quarter,
            "source_file_refs": sorted(self.files_seen),
            "recommendation": recommendation,
            "decision_ids": ["QCAL-B01", "QCAL-B01-PROC"],
            "certification_claim": None,
            "unresolved": [
                "QCAL-B01 calendar bounds and T",
                "leading and trailing snapshot policy",
            ],
        }
        return report


def human_calendar_summary(report: Dict[str, Any]) -> str:
    rec = report.get("recommendation", {})
    lines = [
        "# Calendar diagnostics summary",
        "",
        f"- Status: `{report.get('status')}` (not CERTIFIED)",
        f"- Rows inspected: {report.get('rows_inspected')}",
        f"- Rows accepted: {report.get('rows_accepted_for_diagnostics')}",
        f"- Rows rejected: {report.get('rows_rejected')}",
        f"- Valid timestamp range (epoch): "
        f"{report.get('minimum_valid_timestamp_epoch')} .. "
        f"{report.get('maximum_valid_timestamp_epoch')}",
        f"- Valid quarter range: {report.get('minimum_valid_quarter')} .. "
        f"{report.get('maximum_valid_quarter')}",
        f"- Internal empty quarters preserved: "
        f"{report.get('internal_empty_quarters')}",
        f"- Leading empty candidates: {report.get('leading_empty_quarters')}",
        f"- Trailing empty candidates: {report.get('trailing_empty_quarters')}",
        f"- Candidate start: {rec.get('candidate_certified_start_quarter')}",
        f"- Candidate end: {rec.get('candidate_certified_end_quarter')}",
        f"- Candidate T: {rec.get('candidate_T')}",
        "",
        "QCAL-B01 remains REVIEW_REQUIRED. No calendar-certified claim is made.",
    ]
    return "\n".join(lines) + "\n"
