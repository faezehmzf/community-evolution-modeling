"""Bounded-memory exact length distributions via frequency counters.

Distinct character-length cardinalities for tweet-scale text are small, so a
``Counter[int]`` yields *exact* quantiles without retaining per-row samples.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def counter_quantile(counts: Counter, q: float) -> Optional[float]:
    """Exact quantile from a non-negative integer frequency counter."""
    total = sum(counts.values())
    if total <= 0:
        return None
    if q <= 0:
        return float(min(counts))
    if q >= 1:
        return float(max(counts))
    # Position in 0..total-1 space (same convention as prior list quantile)
    pos = (total - 1) * q
    lo_rank = int(math.floor(pos))
    hi_rank = int(math.ceil(pos))
    w = pos - lo_rank
    lo_val = hi_val = None
    seen = 0
    for length in sorted(counts):
        c = counts[length]
        if lo_val is None and seen + c > lo_rank:
            lo_val = float(length)
        if hi_val is None and seen + c > hi_rank:
            hi_val = float(length)
            break
        seen += c
    if lo_val is None:
        lo_val = float(max(counts))
    if hi_val is None:
        hi_val = lo_val
    if w == 0 or lo_val == hi_val:
        return lo_val
    return lo_val * (1 - w) + hi_val * w


def counter_summary(
    counts: Counter,
    quantiles: Sequence[float],
    candidate_max_lengths: Sequence[int],
) -> Dict:
    total = int(sum(counts.values()))
    if total == 0:
        return {
            "algorithm": "exact_frequency_counter",
            "exact": True,
            "count": 0,
            "quantiles": {f"q{q}": None for q in quantiles},
            "summary": {
                "median": None,
                "p90": None,
                "p95": None,
                "p99": None,
                "p99_9": None,
                "max": None,
                "count": 0,
            },
            "truncation_rate_estimates": {str(m): None for m in candidate_max_lengths},
            "notes": (
                "Exact quantiles from length-frequency counters; memory bounded by "
                "distinct length cardinality, not row count."
            ),
        }
    qmap = {f"q{q}": counter_quantile(counts, float(q)) for q in quantiles}
    named = {
        "median": counter_quantile(counts, 0.5),
        "p90": counter_quantile(counts, 0.9),
        "p95": counter_quantile(counts, 0.95),
        "p99": counter_quantile(counts, 0.99),
        "p99_9": counter_quantile(counts, 0.999),
        "max": float(max(counts)),
        "count": total,
    }
    trunc = {}
    for m in candidate_max_lengths:
        above = sum(c for length, c in counts.items() if length > m)
        trunc[str(m)] = {"n_above": int(above), "rate": above / total}
    return {
        "algorithm": "exact_frequency_counter",
        "exact": True,
        "quantiles": qmap,
        "summary": named,
        "truncation_rate_estimates": trunc,
        "notes": (
            "Exact quantiles from length-frequency counters; memory bounded by "
            "distinct length cardinality, not row count."
        ),
    }


def merge_counters(a: Counter, b: Counter) -> Counter:
    out = Counter(a)
    out.update(b)
    return out
