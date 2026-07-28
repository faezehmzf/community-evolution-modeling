"""Configurable quarterly calendar utilities for Phase 2 diagnostics.

Bounds are runtime-configurable and provisional. They must not be treated as
QCAL-B01 certified calendar values.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from tdmec_diagnostics import constants as DC

_LABEL_RE = re.compile(r"^(\d{4})-Q([1-4])$")


@dataclass(frozen=True)
class QuarterLabel:
    year: int
    quarter: int

    @property
    def label(self) -> str:
        return f"{self.year}-Q{self.quarter}"

    @property
    def index(self) -> int:
        return self.year * 4 + (self.quarter - 1)

    def next(self) -> "QuarterLabel":
        if self.quarter == 4:
            return QuarterLabel(self.year + 1, 1)
        return QuarterLabel(self.year, self.quarter + 1)


def parse_quarter_label(label: str) -> QuarterLabel:
    m = _LABEL_RE.match(str(label).strip())
    if not m:
        raise ValueError(f"invalid quarter label: {label!r}")
    return QuarterLabel(int(m.group(1)), int(m.group(2)))


def quarter_of_month(month: int) -> int:
    if month < 1 or month > 12:
        raise ValueError(f"month out of range: {month}")
    return (month - 1) // 3 + 1


def quarter_start_utc(year: int, quarter: int) -> dt.datetime:
    month = (quarter - 1) * 3 + 1
    return dt.datetime(year, month, 1, tzinfo=dt.timezone.utc)


@dataclass(frozen=True)
class QuarterBoundary:
    snapshot_id: int
    label: str
    start_utc: dt.datetime
    end_utc_exclusive: dt.datetime


def build_quarter_range(
    start_label: str,
    end_label: str,
) -> List[QuarterBoundary]:
    """Inclusive start and inclusive end labels → ordered quarter boundaries.

    Boundary convention: start inclusive, end exclusive for the *next* quarter
    start instant (matches Phase 1 BOUNDARY_CONVENTION).
    """
    start = parse_quarter_label(start_label)
    end = parse_quarter_label(end_label)
    if end.index < start.index:
        raise ValueError(
            f"end label {end_label!r} precedes start label {start_label!r}"
        )
    out: List[QuarterBoundary] = []
    cur = start
    sid = 0
    while cur.index <= end.index:
        nxt = cur.next()
        out.append(
            QuarterBoundary(
                snapshot_id=sid,
                label=cur.label,
                start_utc=quarter_start_utc(cur.year, cur.quarter),
                end_utc_exclusive=quarter_start_utc(nxt.year, nxt.quarter),
            )
        )
        cur = nxt
        sid += 1
    return out


def datetime_to_quarter_label(utc: dt.datetime) -> str:
    if utc.tzinfo is None:
        utc = utc.replace(tzinfo=dt.timezone.utc)
    else:
        utc = utc.astimezone(dt.timezone.utc)
    return f"{utc.year}-Q{quarter_of_month(utc.month)}"


def assign_snapshot_id(
    utc: Optional[dt.datetime],
    boundaries: List[QuarterBoundary],
) -> Optional[int]:
    """Assign provisional snapshot id, or None if outside configured range."""
    if utc is None:
        return None
    if utc.tzinfo is None:
        utc = utc.replace(tzinfo=dt.timezone.utc)
    else:
        utc = utc.astimezone(dt.timezone.utc)
    for b in boundaries:
        if b.start_utc <= utc < b.end_utc_exclusive:
            return b.snapshot_id
    return None


def candidate_T(boundaries: List[QuarterBoundary]) -> int:
    """Candidate T = number of provisional quarters including internal empties."""
    return len(boundaries)


# Twitter launch lower bound for epoch outlier checks
_MIN_REASONABLE_EPOCH = int(
    dt.datetime(2006, 3, 21, tzinfo=dt.timezone.utc).timestamp()
)


def classify_timestamp(
    value: object,
    *,
    boundaries: List[QuarterBoundary],
    now: Optional[dt.datetime] = None,
) -> Tuple[Optional[dt.datetime], Optional[int], str]:
    """Parse and classify a timestamp for calendar diagnostics.

    Returns ``(utc_or_None, snapshot_id_or_None, reason_code)``.
    """
    now = now or dt.datetime.now(tz=dt.timezone.utc)
    max_epoch = int(now.timestamp()) + 86400

    if value is None or value == "":
        return None, None, DC.REASON_MISSING

    if isinstance(value, dt.datetime):
        utc = value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)
        utc = utc.astimezone(dt.timezone.utc)
        sid = assign_snapshot_id(utc, boundaries)
        if sid is None:
            if boundaries and utc < boundaries[0].start_utc:
                return utc, None, DC.REASON_OUT_BEFORE
            if boundaries and utc >= boundaries[-1].end_utc_exclusive:
                return utc, None, DC.REASON_OUT_AFTER
            return utc, None, DC.REASON_OUT_BEFORE
        return utc, sid, DC.REASON_IN_RANGE

    try:
        x = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None, None, DC.REASON_UNPARSABLE

    if not (x == x) or x in (float("inf"), float("-inf")):  # NaN/Inf
        return None, None, DC.REASON_CORRUPT

    # Milliseconds heuristic
    if x > 1e12:
        x = x / 1000.0

    try:
        epoch = int(x)
    except (OverflowError, ValueError):
        return None, None, DC.REASON_CORRUPT

    if epoch < _MIN_REASONABLE_EPOCH or epoch > max_epoch:
        return None, None, DC.REASON_EPOCH_OUTLIER

    try:
        utc = dt.datetime.fromtimestamp(epoch, tz=dt.timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None, None, DC.REASON_INVALID

    sid = assign_snapshot_id(utc, boundaries)
    if sid is None:
        if boundaries and utc < boundaries[0].start_utc:
            return utc, None, DC.REASON_OUT_BEFORE
        if boundaries and utc >= boundaries[-1].end_utc_exclusive:
            return utc, None, DC.REASON_OUT_AFTER
        return utc, None, DC.REASON_INVALID
    return utc, sid, DC.REASON_IN_RANGE
