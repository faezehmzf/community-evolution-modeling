"""Deterministic bounded sampling for Qwen preflight and TDMEC pilot runs.

IMPLEMENTED_NOT_EXECUTED / TRANSFER_PENDING_VALIDATION.
No sampled text or identity is emitted by the reporting interface.
"""
from __future__ import annotations

import hashlib
import heapq
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from tdmec.constants import RELATION_ORDER

from .config import SamplingConfig
from .eligibility import EligibleTextUnit, Modality
from .implementation_status import IMPLEMENTATION_STATUS_LABELS


class SamplingError(RuntimeError):
    pass


def activity_thresholds(activity: Mapping[int, int]) -> tuple[int, int]:
    """Deterministic integer tercile boundaries without NumPy."""

    values = sorted(int(v) for v in activity.values() if int(v) > 0)
    if not values:
        return (0, 0)
    return (values[(len(values) - 1) // 3], values[(2 * (len(values) - 1)) // 3])


def _activity_level(value: int, thresholds: tuple[int, int]) -> str:
    if value <= thresholds[0]:
        return "low"
    if value <= thresholds[1]:
        return "medium"
    return "high"


def _length_level(length: int, config: SamplingConfig) -> str:
    if length <= config.short_text_max_chars:
        return "short"
    if length <= config.medium_text_max_chars:
        return "medium"
    return "long"


def _priority(seed: int, unit_hash: str) -> int:
    payload = f"tdmec-stratified-sample-v1\x00{seed}\x00{unit_hash}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


@dataclass(order=True)
class _HeapItem:
    negative_priority: int
    unit_hash: str
    unit: EligibleTextUnit = field(compare=False, repr=False)
    stratum: tuple[Any, ...] = field(compare=False)


@dataclass(frozen=True)
class SamplingResult:
    modality: Modality
    eligible_population: int
    selected_units: tuple[EligibleTextUnit, ...] = field(repr=False)
    seed: int = 0
    policy: str = "deterministic_stratified_hash_v1"
    observed_strata: int = 0
    represented_strata: int = 0
    population_by_snapshot: Mapping[int, int] = field(default_factory=dict)
    population_by_relation: Mapping[int, int] = field(default_factory=dict)
    selected_by_snapshot: Mapping[int, int] = field(default_factory=dict)
    selected_by_relation: Mapping[int, int] = field(default_factory=dict)
    force_relation_coverage: bool = False
    relation_coverage: Mapping[str, Any] = field(default_factory=dict)

    def report(self) -> Dict[str, Any]:
        return {
            "modality": self.modality,
            "eligible_population": self.eligible_population,
            "selected_population": len(self.selected_units),
            "seed": self.seed,
            "policy": self.policy,
            "observed_strata": self.observed_strata,
            "represented_strata": self.represented_strata,
            "population_by_snapshot": dict(sorted(self.population_by_snapshot.items())),
            "population_by_relation": dict(sorted(self.population_by_relation.items())),
            "selected_by_snapshot": dict(sorted(self.selected_by_snapshot.items())),
            "selected_by_relation": dict(sorted(self.selected_by_relation.items())),
            "force_relation_coverage": self.force_relation_coverage,
            "relation_coverage": dict(self.relation_coverage),
            "relation_names": list(RELATION_ORDER),
            "status_labels": list(IMPLEMENTATION_STATUS_LABELS),
        }


class DeterministicStratifiedSampler:
    """Keep global bottom-k plus one deterministic representative per stratum.

    When ``force_relation_coverage`` is enabled for ``event_text``, one
    deterministic representative for every relation present in the eligible
    population is reserved before the remaining stratified slots are filled.
    """

    def __init__(
        self,
        *,
        modality: Modality,
        limit: int,
        config: SamplingConfig,
        activity: Mapping[int, int],
    ) -> None:
        if limit <= 0:
            raise ValueError("sample limit must be positive")
        self.modality = modality
        self.limit = limit
        self.config = config
        self.activity = activity
        self.thresholds = activity_thresholds(activity)
        self._global: list[_HeapItem] = []
        self._stratum_best: Dict[tuple[Any, ...], _HeapItem] = {}
        self._relation_best: Dict[int, _HeapItem] = {}
        self._eligible = 0
        self._by_snapshot: Counter[int] = Counter()
        self._by_relation: Counter[int] = Counter()

    def _stratum(self, unit: EligibleTextUnit) -> tuple[Any, ...]:
        node = unit.node_index if self.modality == "node_text" else unit.source_idx
        if node is None:
            raise ValueError("sampling unit lacks its modality node identity")
        relation = -1 if unit.relation_id is None else int(unit.relation_id)
        return (
            int(unit.snapshot_id),
            relation,
            _length_level(len(unit.cleaned_text), self.config),
            _activity_level(int(self.activity.get(int(node), 0)), self.thresholds),
            int(node) % self.config.node_hash_buckets,
        )

    def add(self, unit: EligibleTextUnit) -> None:
        if unit.modality != self.modality:
            raise ValueError("sampler modality mismatch")
        self._eligible += 1
        self._by_snapshot[int(unit.snapshot_id)] += 1
        if unit.relation_id is not None:
            self._by_relation[int(unit.relation_id)] += 1
        stratum = self._stratum(unit)
        priority = _priority(self.config.seed, unit.unit_hash)
        item = _HeapItem(-priority, unit.unit_hash, unit, stratum)
        if len(self._global) < self.limit:
            heapq.heappush(self._global, item)
        elif item > self._global[0]:
            heapq.heapreplace(self._global, item)
        current = self._stratum_best.get(stratum)
        if current is None or item > current:
            self._stratum_best[stratum] = item
        if self.modality == "event_text" and unit.relation_id is not None:
            rid = int(unit.relation_id)
            best = self._relation_best.get(rid)
            if best is None or item > best:
                self._relation_best[rid] = item

    def extend(self, units: Iterable[EligibleTextUnit]) -> None:
        for unit in units:
            self.add(unit)

    def finish(self) -> SamplingResult:
        stratum_items = sorted(
            self._stratum_best.values(), key=lambda item: (-item.negative_priority, item.unit_hash)
        )
        global_items = sorted(
            self._global, key=lambda item: (-item.negative_priority, item.unit_hash)
        )
        selected: list[_HeapItem] = []
        seen: set[str] = set()
        force = bool(self.config.force_relation_coverage and self.modality == "event_text")
        relation_coverage: Dict[str, Any] = {
            "enabled": force,
            "population_relations": sorted(self._by_relation),
            "reserved_relations": [],
            "selected_relations": [],
            "missing_from_population": [],
            "missing_from_sample": [],
        }

        def _take(item: _HeapItem) -> bool:
            if item.unit_hash in seen:
                return False
            if len(selected) >= min(self.limit, self._eligible):
                return False
            selected.append(item)
            seen.add(item.unit_hash)
            return True

        if force:
            if self._eligible > 0 and self.limit < len(self._relation_best):
                raise SamplingError(
                    "event sample limit is smaller than the number of available relations; "
                    "increase max_event_rows or disable force_relation_coverage"
                )
            reserved: list[int] = []
            for rid in sorted(self._relation_best):
                if _take(self._relation_best[rid]):
                    reserved.append(rid)
            relation_coverage["reserved_relations"] = reserved

        for item in stratum_items + global_items:
            _take(item)

        selected.sort(key=lambda item: (-item.negative_priority, item.unit_hash))
        selected_by_snapshot: Counter[int] = Counter()
        selected_by_relation: Counter[int] = Counter()
        for item in selected:
            selected_by_snapshot[int(item.unit.snapshot_id)] += 1
            if item.unit.relation_id is not None:
                selected_by_relation[int(item.unit.relation_id)] += 1

        selected_relations = sorted(selected_by_relation)
        population_relations = sorted(self._by_relation)
        missing_sample = sorted(set(population_relations) - set(selected_relations))
        expected = list(range(len(RELATION_ORDER)))
        missing_population = sorted(set(expected) - set(population_relations))
        relation_coverage.update(
            {
                "selected_relations": selected_relations,
                "missing_from_population": missing_population,
                "missing_from_sample": missing_sample,
                "relation_names_covered": [
                    RELATION_ORDER[rid] for rid in selected_relations if 0 <= rid < len(RELATION_ORDER)
                ],
            }
        )
        if force and missing_sample:
            raise SamplingError(
                "force_relation_coverage failed; sample missed relations present in population: "
                f"{missing_sample}"
            )

        represented = len({item.stratum for item in selected})
        return SamplingResult(
            modality=self.modality,
            eligible_population=self._eligible,
            selected_units=tuple(item.unit for item in selected),
            seed=self.config.seed,
            observed_strata=len(self._stratum_best),
            represented_strata=represented,
            population_by_snapshot=dict(self._by_snapshot),
            population_by_relation=dict(self._by_relation),
            selected_by_snapshot=dict(selected_by_snapshot),
            selected_by_relation=dict(selected_by_relation),
            force_relation_coverage=force,
            relation_coverage=relation_coverage,
        )


def activity_counts(units: Iterable[EligibleTextUnit], modality: Modality) -> Dict[int, int]:
    counts: Counter[int] = Counter()
    for unit in units:
        node = unit.node_index if modality == "node_text" else unit.source_idx
        if node is None:
            raise ValueError("unit lacks activity identity")
        counts[int(node)] += 1
    return dict(counts)


def build_combined_sampling_report(
    *,
    node: Mapping[str, Any],
    event: Mapping[str, Any],
    seed: int,
) -> Dict[str, Any]:
    """Privacy-safe combined sampling report written as ``sampling_report.json``."""

    return {
        "schema_version": "tdmec-sampling-report-v1",
        "seed": seed,
        "node": dict(node),
        "event": dict(event),
        "sampled_counts": {
            "node": int(node.get("selected_population", 0)),
            "event": int(event.get("selected_population", 0)),
        },
        "relation_distribution": {
            "population": dict(event.get("population_by_relation", {})),
            "selected": dict(event.get("selected_by_relation", {})),
            "coverage": dict(event.get("relation_coverage", {})),
        },
        "snapshot_distribution": {
            "node_selected": dict(node.get("selected_by_snapshot", {})),
            "event_selected": dict(event.get("selected_by_snapshot", {})),
        },
        "status_labels": list(IMPLEMENTATION_STATUS_LABELS),
    }


__all__ = [
    "DeterministicStratifiedSampler",
    "SamplingError",
    "SamplingResult",
    "activity_counts",
    "activity_thresholds",
    "build_combined_sampling_report",
]
