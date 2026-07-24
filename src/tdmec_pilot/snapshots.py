"""Quarterly snapshot calendar: snapshot 0 = 2017-Q4 … snapshot 34 = 2026-Q2."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import List, Optional, Tuple

SNAPSHOT_START = (2017, 4)  # (year, quarter)
SNAPSHOT_COUNT = 35         # ids 0..34


def _quarter_of_month(month: int) -> int:
    return (month - 1) // 3 + 1


def _quarter_index(year: int, quarter: int) -> int:
    """Absolute quarter index (Q1 2000 = 0-ish reference); monotonic."""
    return year * 4 + (quarter - 1)


_START_IDX = _quarter_index(*SNAPSHOT_START)


@dataclass(frozen=True)
class SnapshotBoundary:
    snapshot_id: int
    label: str
    start_utc: dt.datetime
    end_utc_exclusive: dt.datetime


def _quarter_start(year: int, quarter: int) -> dt.datetime:
    month = (quarter - 1) * 3 + 1
    return dt.datetime(year, month, 1, tzinfo=dt.timezone.utc)


def _add_quarter(year: int, quarter: int) -> Tuple[int, int]:
    q = quarter + 1
    if q > 4:
        return year + 1, 1
    return year, q


def boundary_table() -> List[SnapshotBoundary]:
    out: List[SnapshotBoundary] = []
    y, q = SNAPSHOT_START
    for sid in range(SNAPSHOT_COUNT):
        start = _quarter_start(y, q)
        ny, nq = _add_quarter(y, q)
        end = _quarter_start(ny, nq)
        out.append(SnapshotBoundary(sid, f"{y}-Q{q}", start, end))
        y, q = ny, nq
    return out


def assign_snapshot(utc: Optional[dt.datetime]) -> Optional[int]:
    """Return snapshot id 0..34, or None if outside the canonical range."""
    if utc is None:
        return None
    if utc.tzinfo is None:
        utc = utc.replace(tzinfo=dt.timezone.utc)
    year = utc.year
    quarter = _quarter_of_month(utc.month)
    sid = _quarter_index(year, quarter) - _START_IDX
    if 0 <= sid < SNAPSHOT_COUNT:
        return sid
    return None
