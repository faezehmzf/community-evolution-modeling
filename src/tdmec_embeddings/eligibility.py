"""Derived, privacy-safe embedding eligibility over immutable text sources.

This module implements Q-TEXT/Q-MISS decisions that precede model inference:

* embedding input is ``cleaned_text`` only;
* null, empty, and whitespace-only values are excluded;
* Dataset B duplicate tweet ids use the preprocessing duplicate report's
  deterministic canonical ``(source_file, source_row_number)`` occurrence;
* Dataset A consumes canonical published events and verifies their source text
  hashes;
* stable content, unit, and preprocessing hashes are generated without local
  absolute paths.

No source artifact is changed.  No text is encoded or written by this module.
The returned records contain private fields needed by a future encoder, but
their repr and aggregate reports deliberately omit text and source identities.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Literal, Mapping, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from tdmec.hashing import hash_canonical, sha256_file

from .file_sources import (
    EventTextFileReader,
    FileRecordBatch,
    FileSourceError,
    FileSourceIdentity,
    NodeTextFileReader,
)


Modality = Literal["node_text", "event_text"]

STATUS_LABELS = (
    "PROVISIONAL_SMOKE_ONLY",
    "FILE_ARTIFACT_SOURCE",
    "ENGINEERING_VALIDATION",
    "NOT_FOR_FINAL_THESIS_CONCLUSIONS",
)


class EligibilityError(RuntimeError):
    """Raised when a derived eligibility contract cannot be reconciled."""


@dataclass(frozen=True)
class EligibilityPolicy:
    """Path-independent policy whose hash gates later resume compatibility."""

    version: str = "tdmec_embedding_eligibility_v1"
    input_field: str = "cleaned_text"
    text_hash: str = "sha256_utf8_exact_cleaned_text"
    whitespace_rule: str = "python_unicode_strip_v1"
    node_duplicate_rule: str = "preprocessing_report_canonical_file_row_v1"
    event_duplicate_rule: str = "canonical_published_event_signature_v1"

    def preprocessing_hash(
        self, source: FileSourceIdentity, modality: Modality
    ) -> str:
        """Hash eligibility behavior plus the source preprocessing identity."""

        return hash_canonical(
            {
                "policy": {
                    "version": self.version,
                    "input_field": self.input_field,
                    "text_hash": self.text_hash,
                    "whitespace_rule": self.whitespace_rule,
                    "node_duplicate_rule": self.node_duplicate_rule,
                    "event_duplicate_rule": self.event_duplicate_rule,
                },
                "modality": modality,
                "source_kind": source.source_kind,
                "source_config_hash": source.config_hash,
                "source_manifest_sha256": source.manifest_sha256,
            }
        )


@dataclass(frozen=True)
class EligibleTextUnit:
    """One eligible atomic text unit.

    Identity-bearing and textual values are intentionally hidden from repr.
    They must never be copied into aggregate reports or ordinary logs.
    """

    modality: Modality
    source_run_id: str = field(repr=False)
    unit_id: str = field(repr=False)
    unit_hash: str
    content_hash: str
    preprocessing_hash: str
    cleaned_text: str = field(repr=False)
    snapshot_id: int
    node_index: Optional[int] = field(default=None, repr=False)
    relation_id: Optional[int] = None
    source_idx: Optional[int] = field(default=None, repr=False)
    target_idx: Optional[int] = field(default=None, repr=False)
    source_file: str = field(default="", repr=False)
    source_sheet: Optional[str] = field(default=None, repr=False)
    source_row_number: int = field(default=0, repr=False)


@dataclass(frozen=True)
class EligibilityBatch:
    """Eligible output corresponding to one bounded source batch."""

    modality: Modality
    source_batch_index: int
    source_global_row_offset: int
    units: tuple[EligibleTextUnit, ...]

    @property
    def num_rows(self) -> int:
        return len(self.units)


@dataclass(frozen=True)
class DuplicateCanonicalIndex:
    """Bounded lookup of only Dataset B ids declared duplicated."""

    source_run_id: str
    report_sha256: str
    canonical_by_id: Mapping[str, tuple[str, int]] = field(repr=False)
    occurrence_count_by_id: Mapping[str, int] = field(repr=False)
    duplicate_type_counts: Mapping[str, int]

    @property
    def group_count(self) -> int:
        return len(self.canonical_by_id)

    @property
    def declared_extra_rows(self) -> int:
        return sum(int(value) - 1 for value in self.occurrence_count_by_id.values())


@dataclass(frozen=True)
class EligibilityReport:
    """Aggregate-only eligibility accounting safe for JSON reports."""

    modality: Modality
    source_run_id: str
    source_total_rows: int
    input_rows_seen: int
    eligible_rows: int
    excluded_by_reason: Mapping[str, int]
    preprocessing_hash: str
    completed_full_source: bool
    duplicate_groups_declared: int = 0
    duplicate_occurrences_seen: int = 0
    duplicate_canonical_rows_seen: int = 0
    source_text_hashes_verified: int = 0
    status_labels: tuple[str, ...] = STATUS_LABELS

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe report containing no text or private identities."""

        return {
            "modality": self.modality,
            "source_run_id": self.source_run_id,
            "source_total_rows": self.source_total_rows,
            "input_rows_seen": self.input_rows_seen,
            "eligible_rows": self.eligible_rows,
            "excluded_rows": self.input_rows_seen - self.eligible_rows,
            "excluded_by_reason": dict(sorted(self.excluded_by_reason.items())),
            "preprocessing_hash": self.preprocessing_hash,
            "completed_full_source": self.completed_full_source,
            "duplicate_groups_declared": self.duplicate_groups_declared,
            "duplicate_occurrences_seen": self.duplicate_occurrences_seen,
            "duplicate_canonical_rows_seen": self.duplicate_canonical_rows_seen,
            "source_text_hashes_verified": self.source_text_hashes_verified,
            "status_labels": list(self.status_labels),
        }


def cleaned_text_content_hash(cleaned_text: str) -> str:
    """Full SHA-256 over the exact UTF-8 cleaned-text value."""

    return hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest()


def _text_exclusion_reason(value: Any) -> Optional[str]:
    if value is None:
        return "null_text"
    if not isinstance(value, str):
        raise EligibilityError("cleaned_text contains a non-string value")
    if value == "":
        return "empty_text"
    if value.strip() == "":
        return "whitespace_only_text"
    return None


def _parse_source_locations(value: Any, *, group_index: int) -> list[tuple[str, int]]:
    if not isinstance(value, str) or not value:
        raise EligibilityError(
            f"duplicate report group {group_index} has invalid source locations"
        )
    locations: list[tuple[str, int]] = []
    for token in value.split(";"):
        try:
            source_file, row_text = token.rsplit(":", 1)
            row_number = int(row_text)
        except (ValueError, TypeError) as exc:
            raise EligibilityError(
                f"duplicate report group {group_index} has invalid source locations"
            ) from exc
        if not source_file or row_number < 0:
            raise EligibilityError(
                f"duplicate report group {group_index} has invalid source locations"
            )
        locations.append((source_file, row_number))
    return locations


def load_duplicate_canonical_index(
    source: FileSourceIdentity,
    *,
    max_duplicate_groups: int = 1_000_000,
    batch_size: int = 4096,
) -> DuplicateCanonicalIndex:
    """Load and independently validate Dataset B's small duplicate report.

    Memory is bounded by ``max_duplicate_groups`` and never by corpus rows.
    Canonical selection is verified as the lexicographic minimum source-file /
    row pair, matching :class:`tdmec_pilot.dedup.DuplicateTracker`.
    """

    if source.source_kind != "dataset_b":
        raise EligibilityError("duplicate canonical index requires Dataset B")
    if max_duplicate_groups <= 0 or batch_size <= 0:
        raise ValueError("duplicate index limits must be positive")
    path = source.artifact_root / "duplicate_records.parquet"
    if not path.is_file():
        raise EligibilityError("Dataset B duplicate report is missing")
    parquet = pq.ParquetFile(path)
    required = {
        "tweet_id",
        "duplicate_type",
        "occurrence_count",
        "canonical_source_file",
        "canonical_source_row_number",
        "source_locations",
    }
    if not required.issubset(parquet.schema_arrow.names):
        raise EligibilityError("Dataset B duplicate report schema is incomplete")

    canonical: Dict[str, tuple[str, int]] = {}
    occurrences: Dict[str, int] = {}
    type_counts: Counter[str] = Counter()
    group_index = 0
    for batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=sorted(required),
        use_threads=False,
    ):
        for row in batch.to_pylist():
            group_index += 1
            if group_index > max_duplicate_groups:
                raise EligibilityError("duplicate report exceeds configured group limit")
            tweet_id = row["tweet_id"]
            duplicate_type = row["duplicate_type"]
            count = row["occurrence_count"]
            source_file = row["canonical_source_file"]
            source_row = row["canonical_source_row_number"]
            if not isinstance(tweet_id, str) or not tweet_id:
                raise EligibilityError(
                    f"duplicate report group {group_index} has invalid unit identity"
                )
            if tweet_id in canonical:
                raise EligibilityError("duplicate report repeats a unit identity")
            if duplicate_type not in {"exact_duplicate", "conflicting_id"}:
                raise EligibilityError(
                    f"duplicate report group {group_index} has invalid duplicate type"
                )
            if not isinstance(count, int) or isinstance(count, bool) or count < 2:
                raise EligibilityError(
                    f"duplicate report group {group_index} has invalid occurrence count"
                )
            if (
                not isinstance(source_file, str)
                or not source_file
                or not isinstance(source_row, int)
                or isinstance(source_row, bool)
                or source_row < 0
            ):
                raise EligibilityError(
                    f"duplicate report group {group_index} has invalid canonical provenance"
                )
            locations = _parse_source_locations(
                row["source_locations"], group_index=group_index
            )
            if len(locations) != count:
                raise EligibilityError(
                    f"duplicate report group {group_index} occurrence reconciliation failed"
                )
            if (source_file, source_row) != min(locations):
                raise EligibilityError(
                    f"duplicate report group {group_index} canonical selection is not deterministic"
                )
            canonical[tweet_id] = (source_file, source_row)
            occurrences[tweet_id] = count
            type_counts[duplicate_type] += 1

    return DuplicateCanonicalIndex(
        source_run_id=source.run_id,
        report_sha256=sha256_file(path),
        canonical_by_id=canonical,
        occurrence_count_by_id=occurrences,
        duplicate_type_counts=dict(sorted(type_counts.items())),
    )


def _expected_seen(reader: Any) -> int:
    maximum = getattr(reader, "max_rows", None)
    return reader.total_rows if maximum is None else min(reader.total_rows, maximum)


class _EligibilityProcessor:
    modality: Modality

    def __init__(self, reader: Any, policy: Optional[EligibilityPolicy]) -> None:
        self.reader = reader
        self.policy = policy or EligibilityPolicy()
        self.preprocessing_hash = self.policy.preprocessing_hash(
            reader.identity, self.modality
        )
        self._report: Optional[EligibilityReport] = None
        self._started = False

    @property
    def report(self) -> EligibilityReport:
        if self._report is None:
            raise EligibilityError(
                "eligibility report is available only after complete iteration"
            )
        return self._report

    def _begin(self) -> None:
        if self._started:
            raise EligibilityError("eligibility processor is single-use")
        self._started = True


class NodeTextEligibilityProcessor(_EligibilityProcessor):
    """Filter and hash Dataset B node-text units in bounded batches."""

    modality: Modality = "node_text"

    def __init__(
        self,
        reader: NodeTextFileReader,
        duplicate_index: DuplicateCanonicalIndex,
        *,
        policy: Optional[EligibilityPolicy] = None,
    ) -> None:
        super().__init__(reader, policy)
        if reader.identity.source_kind != "dataset_b":
            raise EligibilityError("node-text eligibility requires Dataset B")
        if duplicate_index.source_run_id != reader.identity.run_id:
            raise EligibilityError("duplicate report belongs to another source run")
        self.duplicate_index = duplicate_index

    def iter_batches(self) -> Iterator[EligibilityBatch]:
        self._begin()
        excluded: Counter[str] = Counter()
        total_seen = 0
        eligible = 0
        duplicate_seen = 0
        canonical_seen = 0

        for source_batch in self.reader.iter_batches():
            units: list[EligibleTextUnit] = []
            for row in source_batch.records.to_pylist():
                total_seen += 1
                reason = _text_exclusion_reason(row["cleaned_text"])
                if reason is not None:
                    excluded[reason] += 1
                    continue
                tweet_id = row["tweet_id"]
                canonical = self.duplicate_index.canonical_by_id.get(tweet_id)
                if canonical is not None:
                    duplicate_seen += 1
                    location = (row["source_file"], row["source_row_number"])
                    if location != canonical:
                        excluded["noncanonical_duplicate"] += 1
                        continue
                    canonical_seen += 1
                text = row["cleaned_text"]
                content_hash = cleaned_text_content_hash(text)
                unit_hash = hash_canonical(
                    {
                        "domain": "tdmec_node_text_unit_v1",
                        "source_run_id": source_batch.source.run_id,
                        "tweet_id": tweet_id,
                        "source_file": row["source_file"],
                        "source_row_number": int(row["source_row_number"]),
                    }
                )
                units.append(
                    EligibleTextUnit(
                        modality=self.modality,
                        source_run_id=source_batch.source.run_id,
                        unit_id=tweet_id,
                        unit_hash=unit_hash,
                        content_hash=content_hash,
                        preprocessing_hash=self.preprocessing_hash,
                        cleaned_text=text,
                        snapshot_id=int(row["snapshot_id"]),
                        node_index=int(row["node_index"]),
                        source_file=row["source_file"],
                        source_sheet=row["source_sheet"],
                        source_row_number=int(row["source_row_number"]),
                    )
                )
                eligible += 1
            yield EligibilityBatch(
                modality=self.modality,
                source_batch_index=source_batch.batch_index,
                source_global_row_offset=source_batch.global_row_offset,
                units=tuple(units),
            )

        expected = _expected_seen(self.reader)
        if total_seen != expected:
            raise EligibilityError("node-text source/output accounting failed")
        completed_full = total_seen == self.reader.total_rows
        if completed_full and canonical_seen != self.duplicate_index.group_count:
            raise EligibilityError(
                "not every declared duplicate group has one canonical normalized row"
            )
        self._report = EligibilityReport(
            modality=self.modality,
            source_run_id=self.reader.identity.run_id,
            source_total_rows=self.reader.total_rows,
            input_rows_seen=total_seen,
            eligible_rows=eligible,
            excluded_by_reason=dict(excluded),
            preprocessing_hash=self.preprocessing_hash,
            completed_full_source=completed_full,
            duplicate_groups_declared=self.duplicate_index.group_count,
            duplicate_occurrences_seen=duplicate_seen,
            duplicate_canonical_rows_seen=canonical_seen,
        )


class EventTextEligibilityProcessor(_EligibilityProcessor):
    """Filter and hash canonical Dataset A event-text units."""

    modality: Modality = "event_text"

    def __init__(
        self,
        reader: EventTextFileReader,
        *,
        policy: Optional[EligibilityPolicy] = None,
    ) -> None:
        super().__init__(reader, policy)
        if reader.identity.source_kind != "dataset_a":
            raise EligibilityError("event-text eligibility requires Dataset A")

    def iter_batches(self) -> Iterator[EligibilityBatch]:
        self._begin()
        excluded: Counter[str] = Counter()
        total_seen = 0
        eligible = 0
        verified_hashes = 0

        for source_batch in self.reader.iter_batches():
            units: list[EligibleTextUnit] = []
            for row in source_batch.records.to_pylist():
                total_seen += 1
                reason = _text_exclusion_reason(row["cleaned_text"])
                if reason is not None:
                    excluded[reason] += 1
                    continue
                text = row["cleaned_text"]
                content_hash = cleaned_text_content_hash(text)
                source_text_hash = row["text_hash"]
                if not isinstance(source_text_hash, str) or not source_text_hash:
                    raise EligibilityError("eligible event has no source text hash")
                if source_text_hash != content_hash[: len(source_text_hash)]:
                    raise EligibilityError("event source text hash is inconsistent")
                verified_hashes += 1
                signature = row["signature"]
                unit_hash = hash_canonical(
                    {
                        "domain": "tdmec_event_text_unit_v1",
                        "source_run_id": source_batch.source.run_id,
                        "signature": signature,
                    }
                )
                units.append(
                    EligibleTextUnit(
                        modality=self.modality,
                        source_run_id=source_batch.source.run_id,
                        unit_id=signature,
                        unit_hash=unit_hash,
                        content_hash=content_hash,
                        preprocessing_hash=self.preprocessing_hash,
                        cleaned_text=text,
                        snapshot_id=int(row["snapshot_id"]),
                        relation_id=int(row["relation_id"]),
                        source_idx=int(row["source_idx"]),
                        target_idx=int(row["target_idx"]),
                        source_file=row["source_file"],
                        source_row_number=int(row["source_row_number"]),
                    )
                )
                eligible += 1
            yield EligibilityBatch(
                modality=self.modality,
                source_batch_index=source_batch.batch_index,
                source_global_row_offset=source_batch.global_row_offset,
                units=tuple(units),
            )

        expected = _expected_seen(self.reader)
        if total_seen != expected:
            raise EligibilityError("event-text source/output accounting failed")
        self._report = EligibilityReport(
            modality=self.modality,
            source_run_id=self.reader.identity.run_id,
            source_total_rows=self.reader.total_rows,
            input_rows_seen=total_seen,
            eligible_rows=eligible,
            excluded_by_reason=dict(excluded),
            preprocessing_hash=self.preprocessing_hash,
            completed_full_source=total_seen == self.reader.total_rows,
            source_text_hashes_verified=verified_hashes,
        )


__all__ = [
    "DuplicateCanonicalIndex",
    "EligibilityBatch",
    "EligibilityError",
    "EligibilityPolicy",
    "EligibilityReport",
    "EligibleTextUnit",
    "EventTextEligibilityProcessor",
    "NodeTextEligibilityProcessor",
    "STATUS_LABELS",
    "cleaned_text_content_hash",
    "load_duplicate_canonical_index",
]
