"""Duplicate detection over exact string tweet ids.

Duplicates are never silently dropped. The corpus is scanned to produce a
duplicate report that distinguishes:

* ``exact_duplicate``   - same tweet_id AND identical content hash,
* ``conflicting_id``    - same tweet_id but differing content.

For each duplicate group a deterministic canonical row is selected (lowest
(source_file, source_row_number)). The pilot annotates rather than collapses, so
normalized outputs keep every row; collapsing can be applied later using the
report.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Occurrence:
    source_file: str
    source_row_number: int
    content_hash: str
    record_status: str


@dataclass
class DuplicateReportRow:
    tweet_id: str
    duplicate_type: str          # exact_duplicate | conflicting_id
    occurrence_count: int
    canonical_source_file: str
    canonical_source_row_number: int
    source_locations: str        # "file:row;file:row;…"


def content_hash(fields: Tuple) -> str:
    h = hashlib.sha256()
    for f in fields:
        h.update(repr(f).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()[:16]


class DuplicateTracker:
    """Accumulates tweet-id occurrences across chunks/files."""

    def __init__(self):
        self._by_id: Dict[str, List[Occurrence]] = {}

    def add(self, tweet_id: str, source_file: str, source_row_number: int,
            content_hash_value: str, record_status: str = "retained") -> None:
        self._by_id.setdefault(tweet_id, []).append(
            Occurrence(source_file, source_row_number, content_hash_value, record_status)
        )

    def duplicate_report(self) -> List[DuplicateReportRow]:
        rows: List[DuplicateReportRow] = []
        for tid, occs in self._by_id.items():
            if len(occs) < 2:
                continue
            hashes = {o.content_hash for o in occs}
            dtype = "exact_duplicate" if len(hashes) == 1 else "conflicting_id"
            canon = min(occs, key=lambda o: (o.source_file, o.source_row_number))
            locs = ";".join(f"{o.source_file}:{o.source_row_number}"
                            for o in sorted(occs, key=lambda o: (o.source_file, o.source_row_number)))
            rows.append(DuplicateReportRow(
                tweet_id=tid, duplicate_type=dtype, occurrence_count=len(occs),
                canonical_source_file=canon.source_file,
                canonical_source_row_number=canon.source_row_number,
                source_locations=locs,
            ))
        return rows

    def canonical_keys(self) -> Dict[str, Tuple[str, int]]:
        """tweet_id -> (canonical source_file, source_row_number) for dup groups."""
        out: Dict[str, Tuple[str, int]] = {}
        for tid, occs in self._by_id.items():
            if len(occs) < 2:
                continue
            canon = min(occs, key=lambda o: (o.source_file, o.source_row_number))
            out[tid] = (canon.source_file, canon.source_row_number)
        return out

    def stats(self) -> dict:
        total_ids = len(self._by_id)
        dup_groups = [o for o in self._by_id.values() if len(o) > 1]
        exact = sum(1 for occs in dup_groups if len({o.content_hash for o in occs}) == 1)
        conflicting = len(dup_groups) - exact
        return {
            "unique_tweet_ids": total_ids,
            "duplicate_id_groups": len(dup_groups),
            "exact_duplicate_groups": exact,
            "conflicting_id_groups": conflicting,
            "extra_duplicate_rows": sum(len(o) - 1 for o in dup_groups),
        }
