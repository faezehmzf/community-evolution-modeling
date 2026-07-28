"""Text-length diagnostics for cleaned/normalized text (no embedding generation).

Quantiles are *exact* via length-frequency counters (bounded by distinct length
cardinality, not row count). They are never approximate streaming sketches.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from tdmec_diagnostics import constants as DC
from tdmec_diagnostics.length_stats import counter_summary
from tdmec_diagnostics.privacy import hash_text_span, privacy_safe_file_ref
from tdmec_diagnostics.records import DiagnosticEventRecord
from tdmec_diagnostics.status import assert_not_certified
from tdmec_diagnostics.tokenizer import TokenizerLengthProbe, NullTokenizerProbe


def _classify_text(value: Any) -> tuple[str, int, int]:
    """Return (quality, char_len, whitespace_token_approx)."""
    if value is None:
        return DC.TEXT_NULL, 0, 0
    if not isinstance(value, str):
        s = str(value)
        tokens = [t for t in s.split() if t]
        return DC.TEXT_NON_STRING, len(s), len(tokens)
    if value == "":
        return DC.TEXT_EMPTY, 0, 0
    if value.strip() == "":
        return DC.TEXT_WHITESPACE_ONLY, len(value), 0
    tokens = [t for t in value.split() if t]
    return DC.TEXT_OK, len(value), len(tokens)


@dataclass
class TextLengthAccumulator:
    quantiles: Sequence[float] = DC.DEFAULT_QUANTILES
    candidate_max_lengths: Sequence[int] = DC.CANDIDATE_MAX_LENGTHS
    tokenizer: TokenizerLengthProbe = field(default_factory=NullTokenizerProbe)

    rows_inspected: int = 0
    rows_accepted: int = 0
    rows_rejected: int = 0

    quality_counts: Dict[str, Counter] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    char_counts: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    token_counts: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    tokenizer_counts: Dict[str, Counter] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    per_snapshot_char: Dict[str, Dict[str, Counter]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(Counter))
    )
    per_relation_char: Dict[str, Counter] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    example_hashes: Dict[str, List[Dict[str, Any]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    files_seen: Set[str] = field(default_factory=set)
    stratified_sample_meta: List[Dict[str, Any]] = field(default_factory=list)

    def observe(
        self,
        record: DiagnosticEventRecord,
        *,
        quarter_label: Optional[str] = None,
    ) -> None:
        self.rows_inspected += 1
        self.files_seen.add(privacy_safe_file_ref(record.source_file))
        ds = record.dataset.upper()
        quality, char_len, tok_approx = _classify_text(record.text)
        self.quality_counts[ds][quality] += 1

        if quality in (DC.TEXT_NULL, DC.TEXT_EMPTY):
            self.rows_rejected += 1
            return

        self.rows_accepted += 1
        self.char_counts[ds][char_len] += 1
        self.token_counts[ds][tok_approx] += 1

        qlabel = quarter_label or str(record.extra.get("quarter_label") or "unknown")
        self.per_snapshot_char[ds][qlabel][char_len] += 1

        if record.relation and ds == "A":
            self.per_relation_char[record.relation][char_len] += 1

        if quality == DC.TEXT_OK and len(self.example_hashes[ds]) < 32:
            self.example_hashes[ds].append(
                {
                    "text_hash": hash_text_span(record.text),
                    "char_length": char_len,
                    "whitespace_token_approx": tok_approx,
                    "quarter_label": qlabel,
                    "relation": record.relation,
                }
            )
            self.stratified_sample_meta.append(
                {
                    "dataset": ds,
                    "quarter_label": qlabel,
                    "char_length": char_len,
                    "text_hash": hash_text_span(record.text),
                }
            )

        if self.tokenizer.is_available:
            tl = self.tokenizer.token_length(record.text or "")
            if tl is not None:
                self.tokenizer_counts[ds][int(tl)] += 1

    def observe_many(
        self,
        records: Iterable[DiagnosticEventRecord],
        *,
        quarter_label: Optional[str] = None,
    ) -> None:
        for r in records:
            self.observe(r, quarter_label=quarter_label)

    def _dist_summary(self, counts: Counter) -> Dict[str, Any]:
        return counter_summary(counts, self.quantiles, self.candidate_max_lengths)

    def to_state(self) -> Dict[str, Any]:
        """Privacy-safe serializable state (counts/hashes only)."""
        return {
            "rows_inspected": self.rows_inspected,
            "rows_accepted": self.rows_accepted,
            "rows_rejected": self.rows_rejected,
            "quality_counts": {ds: dict(c) for ds, c in sorted(self.quality_counts.items())},
            "char_counts": {ds: {str(k): v for k, v in sorted(c.items())} for ds, c in sorted(self.char_counts.items())},
            "token_counts": {ds: {str(k): v for k, v in sorted(c.items())} for ds, c in sorted(self.token_counts.items())},
            "tokenizer_counts": {
                ds: {str(k): v for k, v in sorted(c.items())}
                for ds, c in sorted(self.tokenizer_counts.items())
            },
            "per_snapshot_char": {
                ds: {
                    q: {str(k): v for k, v in sorted(ctr.items())}
                    for q, ctr in sorted(snaps.items())
                }
                for ds, snaps in sorted(self.per_snapshot_char.items())
            },
            "per_relation_char": {
                rel: {str(k): v for k, v in sorted(ctr.items())}
                for rel, ctr in sorted(self.per_relation_char.items())
            },
            "example_hashes": {ds: list(v) for ds, v in sorted(self.example_hashes.items())},
            "files_seen": sorted(self.files_seen),
            "stratified_sample_meta": list(self.stratified_sample_meta),
        }

    @classmethod
    def from_state(
        cls,
        state: Dict[str, Any],
        *,
        quantiles: Sequence[float],
        candidate_max_lengths: Sequence[int],
        tokenizer: Optional[TokenizerLengthProbe] = None,
    ) -> "TextLengthAccumulator":
        acc = cls(
            quantiles=quantiles,
            candidate_max_lengths=candidate_max_lengths,
            tokenizer=tokenizer or NullTokenizerProbe(),
        )
        acc.rows_inspected = int(state.get("rows_inspected", 0))
        acc.rows_accepted = int(state.get("rows_accepted", 0))
        acc.rows_rejected = int(state.get("rows_rejected", 0))
        for ds, c in (state.get("quality_counts") or {}).items():
            acc.quality_counts[ds].update({k: int(v) for k, v in c.items()})
        for ds, c in (state.get("char_counts") or {}).items():
            acc.char_counts[ds].update({int(k): int(v) for k, v in c.items()})
        for ds, c in (state.get("token_counts") or {}).items():
            acc.token_counts[ds].update({int(k): int(v) for k, v in c.items()})
        for ds, c in (state.get("tokenizer_counts") or {}).items():
            acc.tokenizer_counts[ds].update({int(k): int(v) for k, v in c.items()})
        for ds, snaps in (state.get("per_snapshot_char") or {}).items():
            for q, c in snaps.items():
                acc.per_snapshot_char[ds][q].update({int(k): int(v) for k, v in c.items()})
        for rel, c in (state.get("per_relation_char") or {}).items():
            acc.per_relation_char[rel].update({int(k): int(v) for k, v in c.items()})
        for ds, examples in (state.get("example_hashes") or {}).items():
            acc.example_hashes[ds] = list(examples)
        acc.files_seen = set(state.get("files_seen") or [])
        acc.stratified_sample_meta = list(state.get("stratified_sample_meta") or [])
        return acc

    def build_report(
        self,
        *,
        config_hash: str,
        status: str = DC.DIAGNOSTIC_COMPLETE,
    ) -> Dict[str, Any]:
        assert_not_certified(status)
        per_dataset = {}
        for ds in sorted(set(list(self.char_counts) + list(self.quality_counts))):
            total = sum(self.quality_counts[ds].values()) or 1
            per_dataset[ds] = {
                "quality_counts": {
                    k: int(self.quality_counts[ds][k])
                    for k in sorted(self.quality_counts[ds])
                },
                "empty_text_rate": self.quality_counts[ds][DC.TEXT_EMPTY] / total,
                "null_text_rate": self.quality_counts[ds][DC.TEXT_NULL] / total,
                "character_length": self._dist_summary(self.char_counts.get(ds, Counter())),
                "whitespace_token_approx": self._dist_summary(
                    self.token_counts.get(ds, Counter())
                ),
                "tokenizer_length": (
                    self._dist_summary(self.tokenizer_counts.get(ds, Counter()))
                    if self.tokenizer.is_available
                    else {
                        "status": "DEFERRED",
                        "reason": self.tokenizer.unavailable_reason,
                    }
                ),
                "privacy_safe_examples": sorted(
                    self.example_hashes.get(ds, []),
                    key=lambda x: (x["char_length"], x["text_hash"]),
                ),
            }

        per_snapshot = {}
        for ds in sorted(self.per_snapshot_char):
            per_snapshot[ds] = {
                q: self._dist_summary(self.per_snapshot_char[ds][q])
                for q in sorted(self.per_snapshot_char[ds])
            }

        per_relation = {
            rel: self._dist_summary(self.per_relation_char[rel])
            for rel in sorted(self.per_relation_char)
        }

        return {
            "schema_version": DC.DIAGNOSTIC_SCHEMA_VERSION,
            "report_type": DC.REPORT_TEXT_LENGTH,
            "status": status,
            "run_configuration_hash": config_hash,
            "rows_inspected": self.rows_inspected,
            "rows_accepted_for_diagnostics": self.rows_accepted,
            "rows_rejected": self.rows_rejected,
            "quantile_method": {
                "algorithm": "exact_frequency_counter",
                "exact": True,
                "notes": (
                    "Exact quantiles; memory bounded by distinct length cardinality."
                ),
            },
            "per_dataset": per_dataset,
            "per_snapshot": per_snapshot,
            "per_relation_edge_event_text": per_relation,
            "stratified_sampling_metadata": sorted(
                self.stratified_sample_meta,
                key=lambda x: (x["dataset"], x["quarter_label"], x["text_hash"]),
            ),
            "tokenizer_diagnostics": {
                "enabled": bool(self.tokenizer.is_available),
                "status": (
                    "AVAILABLE" if self.tokenizer.is_available else "DEFERRED_TO_PHASE_3"
                ),
                "reason": self.tokenizer.unavailable_reason,
            },
            "source_file_refs": sorted(self.files_seen),
            "decision_ids": ["QEMB-LENGTH-DIAG"],
            "certification_claim": None,
            "unresolved": [
                "embedding instruction",
                "normalization",
                "D_text",
                "final max_length",
                "reduced dimension",
                "QEMB-X01 through QEMB-X07",
            ],
            "notes": (
                "No embedding model download or embedding generation was performed. "
                "Text lengths use NFC-normalized raw text labeled by adapters; "
                "production cleaning/D_text remain unresolved."
            ),
        }


def human_text_length_summary(report: Dict[str, Any]) -> str:
    lines = [
        "# Text-length diagnostics summary",
        "",
        f"- Status: `{report.get('status')}` (not CERTIFIED)",
        f"- Rows inspected: {report.get('rows_inspected')}",
        f"- Quantile method: {report.get('quantile_method')}",
        f"- Tokenizer diagnostics: "
        f"{report.get('tokenizer_diagnostics', {}).get('status')}",
    ]
    for ds, block in sorted((report.get("per_dataset") or {}).items()):
        summ = (block.get("character_length") or {}).get("summary") or {}
        lines.append(
            f"- Dataset {ds}: count={summ.get('count')} median={summ.get('median')} "
            f"p95={summ.get('p95')} max={summ.get('max')} "
            f"null_rate={block.get('null_text_rate')} "
            f"empty_rate={block.get('empty_text_rate')}"
        )
    lines.append("")
    lines.append("QEMB-X01..X07 and D_text remain unresolved. No embeddings generated.")
    return "\n".join(lines) + "\n"
