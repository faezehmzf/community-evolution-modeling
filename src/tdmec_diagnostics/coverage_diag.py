"""Structural and textual coverage diagnostics (Q-MISS / QACT-01 evidence)."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from tdmec import constants as C
from tdmec_diagnostics import constants as DC
from tdmec_diagnostics.privacy import privacy_safe_file_ref
from tdmec_diagnostics.records import DiagnosticEventRecord
from tdmec_diagnostics.status import assert_not_certified


def model_active_mask(struct_active: bool, node_text_available: bool) -> bool:
    """QACT-01: model_active = struct OR node_text. Edge text must not activate."""
    return bool(struct_active) or bool(node_text_available)


def coverage_category(
    struct_active: bool, node_text_available: bool
) -> str:
    if struct_active and node_text_available:
        return DC.COV_STRUCTURE_AND_NODE_TEXT
    if struct_active and not node_text_available:
        return DC.COV_STRUCTURE_ONLY
    if (not struct_active) and node_text_available:
        return DC.COV_NODE_TEXT_ONLY
    return DC.COV_INACTIVE


@dataclass
class CoverageAccumulator:
    relations: Tuple[str, ...] = C.RELATION_ORDER
    node_universe_size: int = C.N_NODES
    frozen_node_indices: Optional[Set[int]] = None

    rows_inspected: int = 0
    rows_accepted: int = 0
    rows_rejected: int = 0

    # Per snapshot node categories: snapshot -> node_idx -> flags
    node_struct: Dict[str, Set[int]] = field(default_factory=lambda: defaultdict(set))
    node_text: Dict[str, Set[int]] = field(default_factory=lambda: defaultdict(set))
    active_nodes: Dict[str, Set[int]] = field(default_factory=lambda: defaultdict(set))

    edge_counts: Dict[str, Counter] = field(
        default_factory=lambda: defaultdict(Counter)
    )  # snapshot -> relation -> count
    event_counts: Dict[str, Counter] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    edge_text_available: Counter = field(default_factory=Counter)
    edge_text_unavailable: Counter = field(default_factory=Counter)

    valid_text_count_node: Dict[str, Counter] = field(
        default_factory=lambda: defaultdict(Counter)
    )  # snapshot -> node_idx -> count
    valid_text_count_edge: Dict[str, Counter] = field(
        default_factory=lambda: defaultdict(Counter)
    )

    external_outside_universe: int = 0
    frozen_node_rule_exclusions: int = 0
    self_loop_candidates: int = 0
    invalid_relation: int = 0
    missing_required_fields: int = 0

    files_seen: Set[str] = field(default_factory=set)
    snapshots_seen: Set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.relations = tuple(self.relations)
        if self.frozen_node_indices is not None:
            self.frozen_node_indices = set(self.frozen_node_indices)

    def observe(
        self,
        record: DiagnosticEventRecord,
        *,
        quarter_label: Optional[str] = None,
    ) -> None:
        self.rows_inspected += 1
        self.files_seen.add(privacy_safe_file_ref(record.source_file))
        qlabel = quarter_label or str(record.extra.get("quarter_label") or "unknown")
        self.snapshots_seen.add(qlabel)

        # Required fields
        if record.dataset.upper() == "A":
            if record.external_user_id is None or record.timestamp_raw in (None, ""):
                self.missing_required_fields += 1
                self.rows_rejected += 1
                return
            if record.relation is None:
                self.missing_required_fields += 1
                self.rows_rejected += 1
                return
            if record.relation not in self.relations:
                self.invalid_relation += 1
                self.rows_rejected += 1
                return
        elif record.dataset.upper() == "B":
            if record.external_user_id is None or record.timestamp_raw in (None, ""):
                self.missing_required_fields += 1
                self.rows_rejected += 1
                return
        else:
            self.rows_rejected += 1
            return

        # Frozen-node universe rule
        if record.node_idx is None:
            # External id present but not in frozen map
            self.external_outside_universe += 1
            self.frozen_node_rule_exclusions += 1
            self.rows_rejected += 1
            return

        if self.frozen_node_indices is not None:
            if record.node_idx not in self.frozen_node_indices:
                self.external_outside_universe += 1
                self.frozen_node_rule_exclusions += 1
                self.rows_rejected += 1
                return
        elif not (0 <= record.node_idx < self.node_universe_size):
            self.external_outside_universe += 1
            self.frozen_node_rule_exclusions += 1
            self.rows_rejected += 1
            return

        # Self-loop candidates (before exclusion)
        if (
            record.dataset.upper() == "A"
            and record.target_node_idx is not None
            and record.node_idx == record.target_node_idx
        ):
            self.self_loop_candidates += 1
            # Counted as observed candidate; still rejected from canonical edges
            self.rows_rejected += 1
            return

        if (
            record.dataset.upper() == "A"
            and record.target_external_user_id is not None
            and record.external_user_id == record.target_external_user_id
            and record.target_node_idx is None
        ):
            # Self-loop detectable via external ids even if target not mapped
            self.self_loop_candidates += 1
            self.rows_rejected += 1
            return

        self.rows_accepted += 1

        struct = bool(record.struct_active)
        node_text = bool(record.node_text_available)
        edge_text = bool(record.edge_text_available)

        if struct:
            self.node_struct[qlabel].add(record.node_idx)
        if node_text:
            self.node_text[qlabel].add(record.node_idx)
            self.valid_text_count_node[qlabel][record.node_idx] += 1

        if model_active_mask(struct, node_text):
            self.active_nodes[qlabel].add(record.node_idx)

        if record.dataset.upper() == "A" and record.relation:
            self.edge_counts[qlabel][record.relation] += 1
            self.event_counts[qlabel][record.relation] += 1
            if edge_text:
                self.edge_text_available[qlabel] += 1
                self.valid_text_count_edge[qlabel][record.relation] += 1
            else:
                self.edge_text_unavailable[qlabel] += 1
        # Dataset B contributes node-text availability via node_text_available above.
        # Edge text must never activate a node (QACT-01).

    def observe_many(
        self,
        records: Iterable[DiagnosticEventRecord],
        *,
        quarter_label: Optional[str] = None,
    ) -> None:
        for r in records:
            self.observe(r, quarter_label=quarter_label)

    @staticmethod
    def _sets_to_state(mapping: Dict[str, Set[int]]) -> Dict[str, List[int]]:
        return {
            snap: sorted(nodes) for snap, nodes in sorted(mapping.items())
        }

    @staticmethod
    def _counter_map_to_state(
        mapping: Dict[str, Counter],
    ) -> Dict[str, Dict[str, int]]:
        return {
            snap: {str(k): int(v) for k, v in sorted(ctr.items())}
            for snap, ctr in sorted(mapping.items())
        }

    def to_state(self) -> Dict[str, Any]:
        """Privacy-safe serializable state (aggregates/node indices/file refs only)."""
        return {
            "relations": list(self.relations),
            "node_universe_size": self.node_universe_size,
            "frozen_node_indices": (
                sorted(self.frozen_node_indices)
                if self.frozen_node_indices is not None
                else None
            ),
            "rows_inspected": self.rows_inspected,
            "rows_accepted": self.rows_accepted,
            "rows_rejected": self.rows_rejected,
            "node_struct": self._sets_to_state(self.node_struct),
            "node_text": self._sets_to_state(self.node_text),
            "active_nodes": self._sets_to_state(self.active_nodes),
            "edge_counts": self._counter_map_to_state(self.edge_counts),
            "event_counts": self._counter_map_to_state(self.event_counts),
            "edge_text_available": {
                k: int(self.edge_text_available[k])
                for k in sorted(self.edge_text_available)
            },
            "edge_text_unavailable": {
                k: int(self.edge_text_unavailable[k])
                for k in sorted(self.edge_text_unavailable)
            },
            "valid_text_count_node": self._counter_map_to_state(
                self.valid_text_count_node
            ),
            "valid_text_count_edge": self._counter_map_to_state(
                self.valid_text_count_edge
            ),
            "external_outside_universe": self.external_outside_universe,
            "frozen_node_rule_exclusions": self.frozen_node_rule_exclusions,
            "self_loop_candidates": self.self_loop_candidates,
            "invalid_relation": self.invalid_relation,
            "missing_required_fields": self.missing_required_fields,
            "files_seen": sorted(self.files_seen),
            "snapshots_seen": sorted(self.snapshots_seen),
        }

    @classmethod
    def from_state(
        cls,
        state: Dict[str, Any],
        *,
        relations: Optional[Tuple[str, ...]] = None,
        node_universe_size: Optional[int] = None,
        frozen_node_indices: Optional[Set[int]] = None,
    ) -> "CoverageAccumulator":
        """Reconstruct an equivalent accumulator for continued observation."""
        frozen = frozen_node_indices
        if frozen is None and state.get("frozen_node_indices") is not None:
            frozen = set(int(n) for n in state["frozen_node_indices"])
        acc = cls(
            relations=tuple(relations)
            if relations is not None
            else tuple(state.get("relations") or C.RELATION_ORDER),
            node_universe_size=(
                int(node_universe_size)
                if node_universe_size is not None
                else int(state.get("node_universe_size", C.N_NODES))
            ),
            frozen_node_indices=frozen,
        )
        acc.rows_inspected = int(state.get("rows_inspected", 0))
        acc.rows_accepted = int(state.get("rows_accepted", 0))
        acc.rows_rejected = int(state.get("rows_rejected", 0))
        for snap, nodes in (state.get("node_struct") or {}).items():
            acc.node_struct[snap] = set(int(n) for n in nodes)
        for snap, nodes in (state.get("node_text") or {}).items():
            acc.node_text[snap] = set(int(n) for n in nodes)
        for snap, nodes in (state.get("active_nodes") or {}).items():
            acc.active_nodes[snap] = set(int(n) for n in nodes)
        for snap, ctr in (state.get("edge_counts") or {}).items():
            acc.edge_counts[snap].update({k: int(v) for k, v in ctr.items()})
        for snap, ctr in (state.get("event_counts") or {}).items():
            acc.event_counts[snap].update({k: int(v) for k, v in ctr.items()})
        for k, v in (state.get("edge_text_available") or {}).items():
            acc.edge_text_available[k] = int(v)
        for k, v in (state.get("edge_text_unavailable") or {}).items():
            acc.edge_text_unavailable[k] = int(v)
        for snap, ctr in (state.get("valid_text_count_node") or {}).items():
            acc.valid_text_count_node[snap].update(
                {int(k): int(v) for k, v in ctr.items()}
            )
        for snap, ctr in (state.get("valid_text_count_edge") or {}).items():
            acc.valid_text_count_edge[snap].update(
                {k: int(v) for k, v in ctr.items()}
            )
        acc.external_outside_universe = int(state.get("external_outside_universe", 0))
        acc.frozen_node_rule_exclusions = int(
            state.get("frozen_node_rule_exclusions", 0)
        )
        acc.self_loop_candidates = int(state.get("self_loop_candidates", 0))
        acc.invalid_relation = int(state.get("invalid_relation", 0))
        acc.missing_required_fields = int(state.get("missing_required_fields", 0))
        acc.files_seen = set(state.get("files_seen") or [])
        acc.snapshots_seen = set(state.get("snapshots_seen") or [])
        return acc

    def build_report(
        self,
        *,
        config_hash: str,
        status: str = DC.DIAGNOSTIC_COMPLETE,
        expected_snapshots: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        assert_not_certified(status)
        snapshots = sorted(expected_snapshots or self.snapshots_seen)

        per_snapshot: List[Dict[str, Any]] = []
        empty_snapshots: List[str] = []
        relation_empty: Dict[str, List[str]] = {r: [] for r in self.relations}

        category_totals = Counter()

        for snap in snapshots:
            struct_nodes = self.node_struct.get(snap, set())
            text_nodes = self.node_text.get(snap, set())
            active = self.active_nodes.get(snap, set())

            structure_only = struct_nodes - text_nodes
            text_only = text_nodes - struct_nodes
            both = struct_nodes & text_nodes
            # Inactive among observed universe of this snapshot:
            observed = struct_nodes | text_nodes
            # Fully inactive relative to node universe is large; report observed inactive = 0
            # plus universe coverage separately.
            inactive_observed = 0  # nodes never observed are universe-level

            cat_counts = {
                DC.COV_STRUCTURE_ONLY: len(structure_only),
                DC.COV_NODE_TEXT_ONLY: len(text_only),
                DC.COV_STRUCTURE_AND_NODE_TEXT: len(both),
                DC.COV_INACTIVE: inactive_observed,
            }
            for k, v in cat_counts.items():
                category_totals[k] += v

            rel_counts = {
                r: int(self.edge_counts.get(snap, Counter()).get(r, 0))
                for r in self.relations
            }
            for r, c in rel_counts.items():
                if c == 0:
                    relation_empty[r].append(snap)

            n_edges = sum(rel_counts.values())
            if n_edges == 0 and len(observed) == 0:
                empty_snapshots.append(snap)

            # Valid-text count distributions (aggregate, not per-node IDs)
            node_text_counts = list(self.valid_text_count_node.get(snap, Counter()).values())
            edge_text_counts = list(self.valid_text_count_edge.get(snap, Counter()).values())

            per_snapshot.append(
                {
                    "quarter_label": snap,
                    "per_relation_edge_counts": rel_counts,
                    "per_relation_event_counts": {
                        r: int(self.event_counts.get(snap, Counter()).get(r, 0))
                        for r in self.relations
                    },
                    "active_node_count": len(active),
                    "structure_only_node_snapshots": len(structure_only),
                    "node_text_only_node_snapshots": len(text_only),
                    "structure_and_node_text_node_snapshots": len(both),
                    "fully_inactive_node_snapshots_observed": inactive_observed,
                    "edge_text_available_edges": int(self.edge_text_available.get(snap, 0)),
                    "edge_text_unavailable_edges": int(
                        self.edge_text_unavailable.get(snap, 0)
                    ),
                    "valid_text_count_node_distribution": {
                        "n_nodes_with_text": len(node_text_counts),
                        "sum_valid_text": int(sum(node_text_counts)),
                        "max_valid_text": int(max(node_text_counts) if node_text_counts else 0),
                    },
                    "valid_text_count_edge_distribution": {
                        "n_relations_with_text": len(edge_text_counts),
                        "sum_valid_text": int(sum(edge_text_counts)),
                    },
                    "node_universe_coverage_rate": (
                        len(observed) / self.node_universe_size
                        if self.node_universe_size
                        else None
                    ),
                }
            )

        # Observed rates (no hard certification thresholds)
        total_cat = sum(category_totals.values()) or 1
        observed_rates = {
            k: category_totals[k] / total_cat for k in sorted(category_totals)
        }

        warnings: List[Dict[str, Any]] = []
        if empty_snapshots:
            warnings.append(
                {
                    "code": "EMPTY_SNAPSHOTS",
                    "severity": "WARNING",
                    "message": "One or more snapshots have no observed structure or node text",
                    "count": len(empty_snapshots),
                }
            )
        if self.external_outside_universe:
            warnings.append(
                {
                    "code": "EXTERNAL_OUTSIDE_UNIVERSE",
                    "severity": "WARNING",
                    "message": "Records referenced identifiers outside the frozen node universe",
                    "count": self.external_outside_universe,
                }
            )
        if self.self_loop_candidates:
            warnings.append(
                {
                    "code": "SELF_LOOP_CANDIDATES",
                    "severity": "WARNING",
                    "message": "Self-loop candidates observed before exclusion",
                    "count": self.self_loop_candidates,
                }
            )

        return {
            "schema_version": DC.DIAGNOSTIC_SCHEMA_VERSION,
            "report_type": DC.REPORT_COVERAGE,
            "status": status,
            "run_configuration_hash": config_hash,
            "rows_inspected": self.rows_inspected,
            "rows_accepted_for_diagnostics": self.rows_accepted,
            "rows_rejected": self.rows_rejected,
            "model_active_definition": "struct_active_mask OR node_text_available_mask",
            "edge_text_activates_node": False,
            "node_universe_size": self.node_universe_size,
            "category_totals": {k: int(category_totals[k]) for k in sorted(category_totals)},
            "observed_category_rates": observed_rates,
            "per_snapshot": per_snapshot,
            "empty_snapshots": empty_snapshots,
            "relation_empty_snapshots": {
                r: relation_empty[r] for r in self.relations
            },
            "dataset_b_identifiers_outside_frozen_universe": self.external_outside_universe,
            "records_excluded_by_frozen_node_rule": self.frozen_node_rule_exclusions,
            "self_loop_candidates_before_exclusion": self.self_loop_candidates,
            "invalid_relation_values": self.invalid_relation,
            "records_missing_required_fields": self.missing_required_fields,
            "candidate_warnings": warnings,
            "numeric_certification_thresholds": {
                "status": "UNRESOLVED",
                "gate": "POST_DIAGNOSTIC",
                "notes": "Hard thresholds must not be invented in Phase 2",
            },
            "source_file_refs": sorted(self.files_seen),
            "decision_ids": ["Q-MISS", "QACT-01", "QART-01-FRAME"],
            "certification_claim": None,
            "unresolved": [
                "numeric coverage certification thresholds",
            ],
        }


def human_coverage_summary(report: Dict[str, Any]) -> str:
    lines = [
        "# Coverage diagnostics summary",
        "",
        f"- Status: `{report.get('status')}` (not CERTIFIED)",
        f"- Rows inspected: {report.get('rows_inspected')}",
        f"- Model-active definition: {report.get('model_active_definition')}",
        f"- Edge text activates node: {report.get('edge_text_activates_node')}",
        f"- Category totals: {report.get('category_totals')}",
        f"- Empty snapshots: {report.get('empty_snapshots')}",
        f"- Outside frozen universe: "
        f"{report.get('dataset_b_identifiers_outside_frozen_universe')}",
        f"- Self-loop candidates: "
        f"{report.get('self_loop_candidates_before_exclusion')}",
        "",
        "Numeric coverage certification thresholds remain unresolved.",
    ]
    return "\n".join(lines) + "\n"
