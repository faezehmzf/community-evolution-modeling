"""Privacy-safe, bounded-memory deduplication diagnostics.

Exact candidate grouping is retained on disk in SQLite.  Only aggregate
counters whose cardinality is bounded by files, snapshots, or the frozen node
universe remain in Python memory.  Source data is never mutated and raw
identifiers or text are never written to the grouping store.
"""
from __future__ import annotations

import hashlib
import sqlite3
import tempfile
from collections import Counter, defaultdict
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from tdmec_diagnostics import constants as DC
from tdmec_diagnostics.privacy import hash_identifier, privacy_safe_file_ref
from tdmec_diagnostics.records import DiagnosticEventRecord
from tdmec_diagnostics.status import assert_not_certified


_GROUP_TWEET_ID = "tweet_id"
_GROUP_COMPOSITE = "composite"
_GROUP_FULL_ROW = "full_row"
_GROUP_USER_TIMESTAMP = "user_timestamp"


def _sha16(fields: Tuple[Any, ...]) -> str:
    h = hashlib.sha256()
    for value in fields:
        h.update(repr(value).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class _Occurrence:
    source_file: str
    source_row_number: int
    content_hash: str
    dataset: str


@dataclass
class DedupAccumulator:
    """Exact deduplication accumulator with disk-backed occurrence grouping."""

    connection: InitVar[Optional[sqlite3.Connection]] = None
    database_path: Optional[Path] = field(default=None, repr=False)

    rows_inspected: int = 0
    rows_accepted: int = 0
    rows_rejected: int = 0

    null_key_counts: Counter = field(default_factory=Counter)
    multiplicity_tweet: Counter = field(default_factory=Counter)
    multiplicity_composite: Counter = field(default_factory=Counter)
    effects_by_file: Dict[str, Counter] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    effects_by_snapshot: Dict[str, Counter] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    files_seen: Set[str] = field(default_factory=set)
    snapshot_labels_seen: Set[str] = field(default_factory=set)

    _connection: sqlite3.Connection = field(init=False, repr=False)
    _temporary_directory: Optional[tempfile.TemporaryDirectory] = field(
        default=None, init=False, repr=False
    )
    _owns_connection: bool = field(default=False, init=False, repr=False)

    def __post_init__(self, connection: Optional[sqlite3.Connection]) -> None:
        if connection is None:
            if self.database_path is None:
                self._temporary_directory = tempfile.TemporaryDirectory(
                    prefix="tdmec-dedup-"
                )
                self.database_path = (
                    Path(self._temporary_directory.name) / "dedup.sqlite"
                )
            self.database_path = Path(self.database_path)
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(str(self.database_path))
            self._owns_connection = True
        self._connection = connection
        self._connection.execute("PRAGMA temp_store = FILE")
        self._connection.execute("PRAGMA cache_size = -8192")
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS dedup_occurrences (
                occurrence_id INTEGER PRIMARY KEY,
                group_kind TEXT NOT NULL,
                group_key TEXT NOT NULL,
                source_file TEXT NOT NULL,
                source_row_number INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                dataset TEXT NOT NULL
            )
            """
        )
        self._connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_dedup_group
            ON dedup_occurrences(group_kind, group_key)
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS dedup_user_counts (
                user_hash TEXT PRIMARY KEY,
                occurrence_count INTEGER NOT NULL
            )
            """
        )

    @property
    def retained_occurrences_in_memory(self) -> int:
        """Number of per-occurrence objects retained by Python."""
        return 0

    @property
    def disk_occurrence_rows(self) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) FROM dedup_occurrences"
        ).fetchone()
        return int(row[0])

    def close(self) -> None:
        if self._owns_connection:
            self._connection.commit()
            self._connection.close()
            self._owns_connection = False
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
            self._temporary_directory = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _store_occurrence(
        self,
        group_kind: str,
        group_key: str,
        occurrence: _Occurrence,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO dedup_occurrences(
                group_kind,
                group_key,
                source_file,
                source_row_number,
                content_hash,
                dataset
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                group_kind,
                group_key,
                occurrence.source_file,
                int(occurrence.source_row_number),
                occurrence.content_hash,
                occurrence.dataset,
            ),
        )

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

        if record.external_user_id:
            user_hash = hash_identifier(record.external_user_id, prefix="user")
            self._connection.execute(
                """
                INSERT INTO dedup_user_counts(user_hash, occurrence_count)
                VALUES (?, 1)
                ON CONFLICT(user_hash) DO UPDATE
                SET occurrence_count = occurrence_count + 1
                """,
                (user_hash,),
            )

        if record.dataset.upper() == "B":
            if not record.tweet_id:
                self.null_key_counts[DC.DEDUP_NULL_KEY] += 1
                self.rows_rejected += 1
                self.effects_by_file[file_ref]["null_key"] += 1
                return
            tweet_id = str(record.tweet_id)
            if "e+" in tweet_id.lower() or "." in tweet_id:
                self.null_key_counts[DC.DEDUP_NULL_KEY] += 1
                self.rows_rejected += 1
                self.effects_by_file[file_ref]["malformed_tweet_id"] += 1
                return
        elif (
            not record.external_user_id
            or record.timestamp_raw in (None, "")
        ):
            self.null_key_counts[DC.DEDUP_NULL_KEY] += 1
            self.rows_rejected += 1
            self.effects_by_file[file_ref]["null_key"] += 1
            return

        self.rows_accepted += 1
        occurrence = _Occurrence(
            source_file=file_ref,
            source_row_number=record.source_row_number,
            content_hash=_sha16(record.content_fingerprint_fields()),
            dataset=record.dataset.upper(),
        )

        self._store_occurrence(
            _GROUP_FULL_ROW,
            _sha16(record.full_row_fingerprint_fields()),
            occurrence,
        )
        if record.dataset.upper() == "B" and record.tweet_id:
            self._store_occurrence(
                _GROUP_TWEET_ID,
                hash_identifier(record.tweet_id, prefix="tid"),
                occurrence,
            )
        else:
            self._store_occurrence(
                _GROUP_COMPOSITE,
                _sha16(record.composite_key_fields()),
                occurrence,
            )
            self._store_occurrence(
                _GROUP_USER_TIMESTAMP,
                _sha16(
                    (
                        record.external_user_id or "",
                        (
                            ""
                            if record.timestamp_raw is None
                            else str(record.timestamp_raw)
                        ),
                    )
                ),
                occurrence,
            )

        self.effects_by_file[file_ref]["accepted"] += 1
        self.effects_by_snapshot[qlabel]["accepted"] += 1

    def observe_many(
        self,
        records: Iterable[DiagnosticEventRecord],
        *,
        quarter_label: Optional[str] = None,
    ) -> None:
        for record in records:
            self.observe(record, quarter_label=quarter_label)

    def to_state(self) -> Dict[str, Any]:
        """Return bounded, privacy-safe aggregate state.

        Per-occurrence grouping rows remain in the transactional SQLite store
        and are deliberately absent from this JSON representation.
        """
        return {
            "state_version": 2,
            "group_storage": "transactional_sqlite",
            "rows_inspected": self.rows_inspected,
            "rows_accepted": self.rows_accepted,
            "rows_rejected": self.rows_rejected,
            "null_key_counts": {
                key: int(self.null_key_counts[key])
                for key in sorted(self.null_key_counts)
            },
            "multiplicity_tweet": {
                str(key): int(value)
                for key, value in sorted(self.multiplicity_tweet.items())
            },
            "multiplicity_composite": {
                str(key): int(value)
                for key, value in sorted(self.multiplicity_composite.items())
            },
            "effects_by_file": {
                file_ref: dict(sorted(counter.items()))
                for file_ref, counter in sorted(self.effects_by_file.items())
            },
            "effects_by_snapshot": {
                snapshot: dict(sorted(counter.items()))
                for snapshot, counter in sorted(self.effects_by_snapshot.items())
            },
            "files_seen": sorted(self.files_seen),
            "snapshot_labels_seen": sorted(self.snapshot_labels_seen),
        }

    @classmethod
    def from_state(
        cls,
        state: Dict[str, Any],
        *,
        connection: Optional[sqlite3.Connection] = None,
    ) -> "DedupAccumulator":
        acc = cls(connection=connection)
        acc.rows_inspected = int(state.get("rows_inspected", 0))
        acc.rows_accepted = int(state.get("rows_accepted", 0))
        acc.rows_rejected = int(state.get("rows_rejected", 0))
        for key, value in (state.get("null_key_counts") or {}).items():
            acc.null_key_counts[key] = int(value)
        for key, value in (state.get("multiplicity_tweet") or {}).items():
            acc.multiplicity_tweet[int(key)] = int(value)
        for key, value in (state.get("multiplicity_composite") or {}).items():
            acc.multiplicity_composite[int(key)] = int(value)
        for file_ref, counter in (state.get("effects_by_file") or {}).items():
            acc.effects_by_file[file_ref].update(
                {key: int(value) for key, value in counter.items()}
            )
        for snapshot, counter in (
            state.get("effects_by_snapshot") or {}
        ).items():
            acc.effects_by_snapshot[snapshot].update(
                {key: int(value) for key, value in counter.items()}
            )
        acc.files_seen = set(state.get("files_seen") or [])
        acc.snapshot_labels_seen = set(state.get("snapshot_labels_seen") or [])
        return acc

    def _classify_groups(self, group_kind: str) -> Dict[str, Any]:
        exact_concordant = 0
        discordant = 0
        cross_file = 0
        within_file = 0
        conflicting_metadata = 0
        extra_rows = 0
        rows_before = 0
        groups_total = 0
        multiplicity: Counter = Counter()
        evidence_rows: List[Dict[str, Any]] = []

        query = """
            WITH group_stats AS (
                SELECT
                    group_key,
                    COUNT(*) AS occurrence_count,
                    COUNT(DISTINCT content_hash) AS distinct_content_hashes,
                    COUNT(DISTINCT source_file) AS distinct_files
                FROM dedup_occurrences
                WHERE group_kind = ?
                GROUP BY group_key
            ),
            canonical AS (
                SELECT
                    group_key,
                    source_file,
                    source_row_number,
                    ROW_NUMBER() OVER (
                        PARTITION BY group_key
                        ORDER BY source_file, source_row_number
                    ) AS row_rank
                FROM dedup_occurrences
                WHERE group_kind = ?
            )
            SELECT
                s.group_key,
                s.occurrence_count,
                s.distinct_content_hashes,
                s.distinct_files,
                c.source_file,
                c.source_row_number
            FROM group_stats AS s
            JOIN canonical AS c
              ON c.group_key = s.group_key AND c.row_rank = 1
            ORDER BY s.group_key
        """
        for row in self._connection.execute(query, (group_kind, group_kind)):
            (
                group_key,
                occurrence_count,
                distinct_content_hashes,
                distinct_files,
                canonical_file,
                canonical_row,
            ) = row
            occurrence_count = int(occurrence_count)
            distinct_content_hashes = int(distinct_content_hashes)
            distinct_files = int(distinct_files)
            groups_total += 1
            rows_before += occurrence_count
            multiplicity[occurrence_count] += 1
            if occurrence_count < 2:
                continue

            extra_rows += occurrence_count - 1
            if distinct_content_hashes == 1:
                exact_concordant += 1
                duplicate_type = DC.DEDUP_SAME_ID_CONCORDANT
            else:
                discordant += 1
                conflicting_metadata += 1
                duplicate_type = DC.DEDUP_SAME_ID_DISCORDANT
            if distinct_files > 1:
                cross_file += 1
                scope = DC.DEDUP_CROSS_FILE
            else:
                within_file += 1
                scope = DC.DEDUP_WITHIN_FILE

            evidence_rows.append(
                {
                    "group_key_hash": str(group_key),
                    "duplicate_type": duplicate_type,
                    "scope": scope,
                    "occurrence_count": occurrence_count,
                    "distinct_content_hashes": distinct_content_hashes,
                    "distinct_files": distinct_files,
                    "canonical_source_file_ref": str(canonical_file),
                    "canonical_source_row_number": int(canonical_row),
                    "source_location_count": occurrence_count,
                }
            )

        return {
            "groups_total": groups_total,
            "duplicate_groups": exact_concordant + discordant,
            "concordant_groups": exact_concordant,
            "discordant_groups": discordant,
            "cross_file_duplicate_groups": cross_file,
            "within_file_duplicate_groups": within_file,
            "conflicting_metadata_groups": conflicting_metadata,
            "extra_duplicate_rows": extra_rows,
            "rows_before_candidate_exact_collapse": rows_before,
            "rows_after_candidate_exact_collapse": groups_total,
            "multiplicity_distribution": {
                str(key): int(multiplicity[key]) for key in sorted(multiplicity)
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
        tweet_stats = self._classify_groups(_GROUP_TWEET_ID)
        composite_stats = self._classify_groups(_GROUP_COMPOSITE)
        user_ts_stats = self._classify_groups(_GROUP_USER_TIMESTAMP)
        full_row_stats = self._classify_groups(_GROUP_FULL_ROW)

        user_multiplicity = Counter(
            int(row[0])
            for row in self._connection.execute(
                "SELECT occurrence_count FROM dedup_user_counts"
            )
        )

        return {
            "schema_version": DC.DIAGNOSTIC_SCHEMA_VERSION,
            "report_type": DC.REPORT_DEDUP,
            "status": status,
            "run_configuration_hash": config_hash,
            "rows_inspected": self.rows_inspected,
            "rows_accepted_for_diagnostics": self.rows_accepted,
            "rows_rejected": self.rows_rejected,
            "null_or_malformed_key_counts": {
                key: int(self.null_key_counts[key])
                for key in sorted(self.null_key_counts)
            },
            "dataset_b_same_id": tweet_stats,
            "dataset_a_candidate_composite": composite_stats,
            "dataset_a_same_user_timestamp": user_ts_stats,
            "exact_full_row": full_row_stats,
            "effects_by_file": {
                file_ref: dict(sorted(self.effects_by_file[file_ref].items()))
                for file_ref in sorted(self.effects_by_file)
            },
            "effects_by_snapshot": {
                snapshot: dict(
                    sorted(self.effects_by_snapshot[snapshot].items())
                )
                for snapshot in sorted(self.effects_by_snapshot)
            },
            "user_occurrence_multiplicity_distribution": {
                str(key): int(user_multiplicity[key])
                for key in sorted(user_multiplicity)
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


def human_dedup_summary(report: Dict[str, Any]) -> str:
    dataset_b = report.get("dataset_b_same_id", {})
    dataset_a = report.get("dataset_a_candidate_composite", {})
    user_timestamp = report.get("dataset_a_same_user_timestamp", {})
    lines = [
        "# Deduplication diagnostics summary",
        "",
        f"- Status: `{report.get('status')}` (not CERTIFIED)",
        f"- Rows inspected: {report.get('rows_inspected')}",
        f"- Dataset B duplicate groups: {dataset_b.get('duplicate_groups')}",
        f"- Dataset B concordant / discordant: "
        f"{dataset_b.get('concordant_groups')} / "
        f"{dataset_b.get('discordant_groups')}",
        f"- Dataset A candidate composite duplicate groups: "
        f"{dataset_a.get('duplicate_groups')}",
        f"- Dataset A same-user+timestamp concordant / discordant: "
        f"{user_timestamp.get('concordant_groups')} / "
        f"{user_timestamp.get('discordant_groups')}",
        "",
        "QDEDUP-B01 signature and L2 thresholds remain REVIEW_REQUIRED.",
        "Raw sources were not modified.",
    ]
    return "\n".join(lines) + "\n"
