"""Timestamp normalization for Dataset B ``created_at`` (epoch seconds)."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Optional

# Twitter launched 2006-03-21; reject anything before that or absurdly future.
_MIN_EPOCH = int(dt.datetime(2006, 3, 21, tzinfo=dt.timezone.utc).timestamp())


@dataclass
class NormalizedTime:
    utc: Optional[dt.datetime]
    ok: bool
    error: Optional[str] = None


def parse_created_at(value: Any, now: Optional[dt.datetime] = None) -> NormalizedTime:
    now = now or dt.datetime.now(tz=dt.timezone.utc)
    max_epoch = int(now.timestamp()) + 86400  # +1 day tolerance
    if value is None or value == "":
        return NormalizedTime(None, False, "invalid_timestamp")
    if isinstance(value, dt.datetime):
        d = value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)
        return NormalizedTime(d.astimezone(dt.timezone.utc), True, None)
    try:
        x = float(value)
    except (TypeError, ValueError):
        return NormalizedTime(None, False, "invalid_timestamp")
    if x > 1e12:  # milliseconds -> seconds
        x /= 1000.0
    epoch = int(x)
    if epoch < _MIN_EPOCH or epoch > max_epoch:
        return NormalizedTime(None, False, "invalid_timestamp")
    try:
        d = dt.datetime.fromtimestamp(epoch, tz=dt.timezone.utc)
    except (OverflowError, OSError, ValueError):
        return NormalizedTime(None, False, "invalid_timestamp")
    return NormalizedTime(d, True, None)
