"""Privacy-safe deduplication diagnostics (QDEDUP-B01 evidence only).

Does not mutate source data. Does not finalize the Dataset A signature or L2
thresholds. Does not implement aggressive fuzzy deduplication.
"""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from tdmec_diagnostics import constants as DC
from tdmec_diagnostics.privacy import hash_identifier, privacy_safe_file_ref
from tdmec_diagnostics.records import DiagnosticEventRecord
from tdmec_diagnostics.status import assert_not_certified


def _sha16(fields: Tuple[Any, ...]) -> str:
    h = hashlib.sha256()
    for f in fields:
        h.update(repr(f).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()[:16]


@dataclass
class _Occurrence:
    source_file: str
    source_row_number: int
    content_hash: str
    dataset: str


@dataclass
class DedupAccumulator:
    """Streaming accumulator for exact-duplicate candidate diagnostics."""

    rows_inspected: int = 0
    rows_accepted: int = 0
    rows_rejected: int = 0

    # Grouping maps: key -> occurrences
    by_tweet_id: Dict[str, List[_Occurrence]] = field(default_factory=dict)
    by_composite: Dict[str, List[_Occurrence]] = field(default_factory=dict)
    by_full_row: Dict[str, List[_Occurrence]] = field(default_factory=dict)
    # Dataset A discordant/concordant analysis key: user + timestamp only
    by_user_timestamp: Dict[str, List[_Occurrence]] = field(default_factory=dict)

    null_key_counts: Counter = field(default_factory=Counter)
    multiplicity_tweet: Counter = field(default_factory=Counter)
    multiplicity_composite: Counter = field(default_factory=Counter)

    effects_by_file: Dict[str, Counter] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    effects_by_snapshot: Dict[str, Counter] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    # Privacy-safe aggregate: hashed user -> occurrence count only
    user_occurrence_counts: Counter = field(default_factory=Counter)

    files_seen: Set[str] = field(default_factory=set)
    snapshot_labels_seen: Set[str] = field(default_factory=set)

    # Optional snapshot label provided by caller via record.extra["quarter_label"]
    def observe(
        self,
        record: DiagnosticEventRecord,
        *,
        quarter_label: Optional[str] = None,
    ) -> None:
        self.rows_inspected += 1
        file_ref = privacy_safe_file_ref(record.source_file)
        self.files_seen.add(file_ref)
        qlabel = quarter_label or str(record.extra.get("quarter_label") or "unknown")
        self.snapshot_labels_seen.add(qlabel)

        # User aggregate (hashed only)
        if record.external_user_id:
            self.user_occurrence_counts[
                hash_identifier(record.external_user_id, prefix="user")
            ] += 1

        # Null / malformed candidate keys
        if record.dataset.upper() == "B":
            if not record.tweet_id:
                self.null_key_counts[DC.DEDUP_NULL_KEY] += 1
                self.rows_rejected += 1
                self.effects_by_file[file_ref]["null_key"] += 1
                return
            # Reject float-like scientific notation tweet ids as malformed
            tid = str(record.tweet_id)
            if "e+" in tid.lower() or "." in tid:
                self.null_key_counts[DC.DEDUP_NULL_KEY] += 1
                self.rows_rejected += 1
                self.effects_by_file[file_ref]["malformed_tweet_id"] += 1
                return
        else:
            # Dataset A: composite fields; float tweet id must not be the exact key
            if not record.external_user_id or record.timestamp_raw in (None, ""):
                self.null_key_counts[DC.DEDUP_NULL_KEY] += 1
                self.rows_rejected += 1
                self.effects_by_file[file_ref]["null_key"] += 1
                return

        self.rows_accepted += 1
        content_h = _sha16(record.content_fingerprint_fields())
        occ = _Occurrence(
            source_file=file_ref,
            source_row_number=record.source_row_number,
            content_hash=content_h,
            dataset=record.dataset.upper(),
        )

        # Full-row exact duplicates
        fr = _sha16(record.full_row_fingerprint_fields())
        self.by_full_row.setdefault(fr, []).append(occ)

        if record.dataset.upper() == "B" and record.tweet_id:
            # Store under hashed tweet id only in the accumulator key space
            key = hash_identifier(record.tweet_id, prefix="tid")
            self.by_tweet_id.setdefault(key, []).append(occ)
        else:
            key = _sha16(record.composite_key_fields())
            self.by_composite.setdefault(key, []).append(occ)
            # Concordant vs discordant for A: group by user + timestamp only
            ut_key = _sha16(
                (
                    record.external_user_id or "",
                    "" if record.timestamp_raw is None else str(record.timestamp_raw),
                )
            )
            self.by_user_timestamp.setdefault(ut_key, []).append(occ)

        self.effects_by_file[file_ref]["accepted"] += 1
        self.effects_by_snapshot[qlabel]["accepted"] += 1

    def observe_many(
        self,
        records: Iterable[DiagnosticEventRecord],
        *,
        quarter_label: Optional[str] = None,
    ) -> None:
        for r in records:
            self.observe(r, quarter_label=quarter_label)

    @staticmethod
    def _occ_to_dict(occ: _Occurrence) -> Dict[str, Any]:
        return {
            "source_file": occ.source_file,
            "source_row_number": int(occ.source_row_number),
            "content_hash": occ.content_hash,
            "dataset": occ.dataset,
        }

    @staticmethod
    def _occ_from_dict(d: Dict[str, Any]) -> _Occurrence:
        return _Occurrence(
            source_file=str(d["source_file"]),
            source_row_number=int(d["source_row_number"]),
            content_hash=str(d["content_hash"]),
            dataset=str(d["dataset"]),
        )

    @staticmethod
    def _groups_to_state(
        groups: Dict[str, List[_Occurrence]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        return {
            key: [DedupAccumulator._occ_to_dict(o) for o in occs]
            for key, occs in sorted(groups.items())
        }

    @staticmethod
    def _groups_from_state(
        data: Optional[Dict[str, List[Dict[str, Any]]]],
    ) -> Dict[str, List[_Occurrence]]:
        out: Dict[str, List[_Occurrence]] = {}
        for key, occs in (data or {}).items():
            out[str(key)] = [DedupAccumulator._occ_from_dict(o) for o in occs]
        return out

    def to_state(self) -> Dict[str, Any]:
        """Privacy-safe serializable state (hashed keys, counts, file refs only)."""
        return {
            "rows_inspected": self.rows_inspected,
            "rows_accepted": self.rows_accepted,
            "rows_rejected": self.rows_rejected,
            "by_tweet_id": self._groups_to_state(self.by_tweet_id),
            "by_composite": self._groups_to_state(self.by_composite),
            "by_full_row": self._groups_to_state(self.by_full_row),
            "by_user_timestamp": self._groups_to_state(self.by_user_timestamp),
            "null_key_counts": {
                k: int(self.null_key_counts[k]) for k in sorted(self.null_key_counts)
            },
            "multiplicity_tweet": {
                str(k): int(v) for k, v in sorted(self.multiplicity_tweet.items())
            },
            "multiplicity_composite": {
                str(k): int(v)
                for k, v in sorted(self.multiplicity_composite.items())
            },
            "effects_by_file": {
                f: dict(sorted(ctr.items()))
                for f, ctr in sorted(self.effects_by_file.items())
            },
            "effects_by_snapshot": {
                s: dict(sorted(ctr.items()))
                for s, ctr in sorted(self.effects_by_snapshot.items())
            },
            "user_occurrence_counts": {
                k: int(self.user_occurrence_counts[k])
                for k in sorted(self.user_occurrence_counts)
            },
            "files_seen": sorted(self.files_seen),
            "snapshot_labels_seen": sorted(self.snapshot_labels_seen),
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "DedupAccumulator":
        """Reconstruct an equivalent accumulator for continued observation."""
        acc = cls()
        acc.rows_inspected = int(state.get("rows_inspected", 0))
        acc.rows_accepted = int(state.get("rows_accepted", 0))
        acc.rows_rejected = int(state.get("rows_rejected", 0))
        acc.by_tweet_id = cls._groups_from_state(state.get("by_tweet_id"))
        acc.by_composite = cls._groups_from_state(state.get("by_composite"))
        acc.by_full_row = cls._groups_from_state(state.get("by_full_row"))
        acc.by_user_timestamp = cls._groups_from_state(state.get("by_user_timestamp"))
        for k, v in (state.get("null_key_counts") or {}).items():
            acc.null_key_counts[k] = int(v)
        for k, v in (state.get("multiplicity_tweet") or {}).items():
            acc.multiplicity_tweet[int(k)] = int(v)
        for k, v in (state.get("multiplicity_composite") or {}).items():
            acc.multiplicity_composite[int(k)] = int(v)
        for f, ctr in (state.get("effects_by_file") or {}).items():
            acc.effects_by_file[f].update({kk: int(vv) for kk, vv in ctr.items()})
        for s, ctr in (state.get("effects_by_snapshot") or {}).items():
            acc.effects_by_snapshot[s].update({kk: int(vv) for kk, vv in ctr.items()})
        for k, v in (state.get("user_occurrence_counts") or {}).items():
            acc.user_occurrence_counts[k] = int(v)
        acc.files_seen = set(state.get("files_seen") or [])
        acc.snapshot_labels_seen = set(state.get("snapshot_labels_seen") or [])
        return acc

    def _classify_groups(
        self, groups: Dict[str, List[_Occurrence]]
    ) -> Dict[str, Any]:
        exact_concordant = 0
        discordant = 0
        cross_file = 0
        within_file = 0
        conflicting_metadata = 0
        extra_rows = 0
        multiplicity: Counter = Counter()

        evidence_rows: List[Dict[str, Any]] = []
        for key, occs in groups.items():
            multiplicity[len(occs)] += 1
            if len(occs) < 2:
                continue
            extra_rows += len(occs) - 1
            hashes = {o.content_hash for o in occs}
            files = {o.source_file for o in occs}
            concordant = len(hashes) == 1
            if concordant:
                exact_concordant += 1
                dtype = DC.DEDUP_SAME_ID_CONCORDANT
            else:
                discordant += 1
                conflicting_metadata += 1
                dtype = DC.DEDUP_SAME_ID_DISCORDANT
            if len(files) > 1:
                cross_file += 1
                scope = DC.DEDUP_CROSS_FILE
            else:
                within_file += 1
                scope = DC.DEDUP_WITHIN_FILE

            canon = min(occs, key=lambda o: (o.source_file, o.source_row_number))
            evidence_rows.append(
                {
                    "group_key_hash": key,
                    "duplicate_type": dtype,
                    "scope": scope,
                    "occurrence_count": len(occs),
                    "distinct_content_hashes": len(hashes),
                    "distinct_files": len(files),
                    "canonical_source_file_ref": canon.source_file,
                    "canonical_source_row_number": canon.source_row_number,
                    # Privacy-safe location refs only (file refs already basenames)
                    "source_location_count": len(occs),
                }
            )

        evidence_rows.sort(
            key=lambda r: (
                r["group_key_hash"],
                r["canonical_source_file_ref"],
                r["canonical_source_row_number"],
            )
        )
        before = sum(len(v) for v in groups.values())
        after_exact_collapse = sum(1 for v in groups.values() if v)  # one per key
        return {
            "groups_total": len(groups),
            "duplicate_groups": exact_concordant + discordant,
            "concordant_groups": exact_concordant,
            "discordant_groups": discordant,
            "cross_file_duplicate_groups": cross_file,
            "within_file_duplicate_groups": within_file,
            "conflicting_metadata_groups": conflicting_metadata,
            "extra_duplicate_rows": extra_rows,
            "rows_before_candidate_exact_collapse": before,
            "rows_after_candidate_exact_collapse": after_exact_collapse,
            "multiplicity_distribution": {
                str(k): int(multiplicity[k]) for k in sorted(multiplicity)
            },
            "evidence_table": evidence_rows,
        }

    def build_report(
        self,
        *,
        config_hash: str,
        status: str = DC.DIAGNOSTIC_COMPLETE,
    ) -> Dict[str, Any]:
        assert_not_certified(status)
        tweet_stats = self._classify_groups(self.by_tweet_id)
        composite_stats = self._classify_groups(self.by_composite)
        user_ts_stats = self._classify_groups(self.by_user_timestamp)
        full_row_stats = self._classify_groups(self.by_full_row)

        # Privacy-safe user multiplicity (counts only; no raw IDs)
        user_mult = Counter(self.user_occurrence_counts.values())

        report = {
            "schema_version": DC.DIAGNOSTIC_SCHEMA_VERSION,
            "report_type": DC.REPORT_DEDUP,
            "status": status,
            "run_configuration_hash": config_hash,
            "rows_inspected": self.rows_inspected,
            "rows_accepted_for_diagnostics": self.rows_accepted,
            "rows_rejected": self.rows_rejected,
            "null_or_malformed_key_counts": {
                k: int(self.null_key_counts[k]) for k in sorted(self.null_key_counts)
            },
            "dataset_b_same_id": tweet_stats,
            "dataset_a_candidate_composite": composite_stats,
            "dataset_a_same_user_timestamp": user_ts_stats,
            "exact_full_row": full_row_stats,
            "effects_by_file": {
                f: dict(sorted(self.effects_by_file[f].items()))
                for f in sorted(self.effects_by_file)
            },
            "effects_by_snapshot": {
                s: dict(sorted(self.effects_by_snapshot[s].items()))
                for s in sorted(self.effects_by_snapshot)
            },
            "user_occurrence_multiplicity_distribution": {
                str(k): int(user_mult[k]) for k in sorted(user_mult)
            },
            "candidate_signatures": {
                "dataset_b": "exact_string_tweet_id + content_hash",
                "dataset_a": (
                    "external_user_id + timestamp + text_hash + relation + "
                    "target_id + referenced_status + provenance"
                ),
                "status": "CANDIDATE_ONLY",
                "finalization": "REVIEW_REQUIRED (QDEDUP-B01 unresolved)",
            },
            "provenance_retention_requirements": [
                "Retain source_file and source_row_number for every occurrence",
                "Retain content_hash for concordant/discordant classification",
                "Do not silently drop discordant records",
                "Raw source files must remain unchanged",
            ],
            "source_file_refs": sorted(self.files_seen),
            "decision_ids": ["QDEDUP-B01", "QDEDUP-B01-PROC"],
            "certification_claim": None,
            "unresolved": [
                "QDEDUP-B01 exact signature and thresholds",
                "Layer-2 repeated-span cleaning thresholds",
            ],
            "notes": (
                "Aggressive fuzzy deduplication is out of scope. Discordant "
                "records are reported, not removed. No CERTIFIED claim."
            ),
        }
        return report


def human_dedup_summary(report: Dict[str, Any]) -> str:
    b = report.get("dataset_b_same_id", {})
    a = report.get("dataset_a_candidate_composite", {})
    ut = report.get("dataset_a_same_user_timestamp", {})
    lines = [
        "# Deduplication diagnostics summary",
        "",
        f"- Status: `{report.get('status')}` (not CERTIFIED)",
        f"- Rows inspected: {report.get('rows_inspected')}",
        f"- Dataset B duplicate groups: {b.get('duplicate_groups')}",
        f"- Dataset B concordant / discordant: "
        f"{b.get('concordant_groups')} / {b.get('discordant_groups')}",
        f"- Dataset A candidate composite duplicate groups: "
        f"{a.get('duplicate_groups')}",
        f"- Dataset A same-user+timestamp concordant / discordant: "
        f"{ut.get('concordant_groups')} / {ut.get('discordant_groups')}",
        "",
        "QDEDUP-B01 signature and L2 thresholds remain REVIEW_REQUIRED.",
        "Raw sources were not modified.",
    ]
    return "\n".join(lines) + "\n"
