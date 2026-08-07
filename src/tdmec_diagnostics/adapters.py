"""Concrete Dataset A / Dataset B adapters for Phase 2 diagnostics.

Adapters stream documented workbook schemas into DiagnosticEventRecord rows.
They never mutate source files, never guess column meanings, and never embed
secrets. External IDs remain only in transient processing objects; reports must
hash/redact them.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

from tdmec_diagnostics.records import DiagnosticEventRecord
from tdmec_diagnostics.schema_contracts import (
    DATASET_A_ADAPTER_ID,
    DATASET_A_REQUIRED_COLUMNS,
    DATASET_A_SHEET_NAME,
    DATASET_A_TEXT_FIELD_LABEL,
    DATASET_A_TWEET_ID_TRUSTED,
    DATASET_B_ADAPTER_ID,
    DATASET_B_REQUIRED_COLUMNS,
    DATASET_B_SHEET_NAME,
    DATASET_B_TEXT_FIELD_LABEL,
    DATASET_B_TWEET_ID_TRUSTED,
)
from tdmec_diagnostics.workbook_io import (
    UnsupportedSchemaError,
    iter_xlsx_rows,
    validate_required_columns,
)

# Reuse pilot parsers (already tested) — no reimplementation of blob/id rules.
from tdmec_pilot.identifiers import normalize_account_id, normalize_tweet_id
from tdmec_pilot.text_quality import normalize_text
from tdmec_pilot.user_blob import parse_user_blob

_ID_RE = re.compile(r"['\"]id['\"]\s*:\s*(\d+)")


class AdapterConfigurationError(ValueError):
    """Missing or invalid adapter runtime configuration."""


@dataclass(frozen=True)
class NodeUniverseLookup:
    """Frozen node-universe lookup (external account id → node_idx)."""

    mapping: Mapping[str, int]
    n_nodes: int

    def get(self, account_id: Optional[str]) -> Optional[int]:
        if account_id is None:
            return None
        return self.mapping.get(str(account_id))


def load_node_universe_lookup(
    path: str | Path,
    *,
    expected_count: int = 16736,
) -> NodeUniverseLookup:
    """Load node-index map via the existing pilot loader."""
    from tdmec_pilot.node_map import load_node_map

    nm = load_node_map(
        path,
        expected_count=expected_count,
        index_min=0,
        index_max=expected_count - 1,
    )
    mapping = {str(k): int(v) for k, v in nm.mapping.items()}
    return NodeUniverseLookup(mapping=mapping, n_nodes=expected_count)


def build_node_universe_lookup_from_ids(
    account_ids: Sequence[str],
) -> NodeUniverseLookup:
    """Build a synthetic lookup (tests only). Deterministic sorted assignment."""
    ordered = sorted({str(a) for a in account_ids})
    mapping = {a: i for i, a in enumerate(ordered)}
    return NodeUniverseLookup(mapping=mapping, n_nodes=len(ordered))


def _safe_load_blob(value: Any) -> Any:
    if value is None or value == "":
        return None
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s or s.lower() in {"none", "null", "nan"}:
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def _extract_nested_user_id(blob: Any) -> Optional[str]:
    """Extract user.id from a status blob (retweeted/reply/quoted)."""
    obj = _safe_load_blob(blob) if not isinstance(blob, dict) else blob
    if not isinstance(obj, dict):
        # regex recovery of nested id only
        if isinstance(blob, str):
            # Prefer user.id pattern inside blob
            m = re.search(r"['\"]user['\"]\s*:\s*\{[^}]*['\"]id['\"]\s*:\s*(\d+)", blob)
            if m:
                return m.group(1)
            m2 = _ID_RE.search(blob)
            if m2:
                return m2.group(1)
        return None
    user = obj.get("user")
    if isinstance(user, dict):
        return normalize_account_id(user.get("id"))
    if isinstance(user, str):
        parsed = parse_user_blob(user)
        return parsed.account_id if parsed.ok else None
    # Some blobs may place id at top level
    return normalize_account_id(obj.get("id"))


def _extract_mention_ids(value: Any) -> List[str]:
    obj = _safe_load_blob(value) if not isinstance(value, list) else value
    out: List[str] = []
    if not isinstance(obj, list):
        if isinstance(value, str):
            for m in re.finditer(r"['\"]id['\"]\s*:\s*(\d+)", value):
                out.append(m.group(1))
        return out
    for item in obj:
        if isinstance(item, dict):
            aid = normalize_account_id(item.get("id"))
            if aid:
                out.append(aid)
        elif isinstance(item, (int, str)):
            aid = normalize_account_id(item)
            if aid:
                out.append(aid)
    return out


def _row_dict(header_index: Mapping[str, int], row: Sequence[Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, idx in header_index.items():
        out[name] = row[idx] if idx < len(row) else None
    return out


# ---------------------------------------------------------------------------
# Dataset B adapter
# ---------------------------------------------------------------------------


def iter_dataset_b_records(
    path: str | Path,
    *,
    node_lookup: NodeUniverseLookup,
    source_file_name: Optional[str] = None,
    start_after_source_row: int = 1,
) -> Iterator[DiagnosticEventRecord]:
    """Stream Dataset B rows → DiagnosticEventRecord (node-text diagnostics)."""
    path = Path(path)
    file_name = source_file_name or path.name
    sheet, header, rows = iter_xlsx_rows(
        path,
        expected_sheet=DATASET_B_SHEET_NAME,
        start_after_row_number=start_after_source_row,
    )
    if sheet != DATASET_B_SHEET_NAME:
        raise UnsupportedSchemaError(
            f"adapter={DATASET_B_ADAPTER_ID} expected sheet {DATASET_B_SHEET_NAME!r}, "
            f"got {sheet!r}"
        )
    col = validate_required_columns(
        header, DATASET_B_REQUIRED_COLUMNS, adapter_id=DATASET_B_ADAPTER_ID
    )

    for row_number, row in enumerate(
        rows,
        start=max(2, int(start_after_source_row) + 1),
    ):
        cells = _row_dict(col, row)
        tid = normalize_tweet_id(cells.get("id"))
        user = parse_user_blob(cells.get("user"))
        author_id = user.account_id if user.ok else None
        node_idx = node_lookup.get(author_id) if author_id else None

        raw_text = cells.get("text")
        # Length diagnostics operate on NFC-normalized raw text (not production cleaning)
        norm = normalize_text(raw_text)
        text_for_diag = norm.text_normalized
        node_text_available = norm.quality == "ok"

        yield DiagnosticEventRecord(
            dataset="B",
            source_file=file_name,
            source_row_number=row_number,
            timestamp_raw=cells.get("created_at"),
            external_user_id=author_id,
            tweet_id=tid.value if tid.ok else (
                None if cells.get("id") is None else str(cells.get("id"))
            ),
            text=text_for_diag,
            relation=None,
            target_external_user_id=None,
            node_idx=node_idx,
            target_node_idx=None,
            struct_active=False,
            node_text_available=node_text_available,
            edge_text_available=False,
            extra={
                "adapter_id": DATASET_B_ADAPTER_ID,
                "text_field_label": DATASET_B_TEXT_FIELD_LABEL,
                "tweet_id_trusted": DATASET_B_TWEET_ID_TRUSTED,
                "tweet_id_ok": tid.ok,
                "user_parse_ok": user.ok,
            },
        )


# ---------------------------------------------------------------------------
# Dataset A adapter
# ---------------------------------------------------------------------------


def iter_dataset_a_records(
    path: str | Path,
    *,
    node_lookup: NodeUniverseLookup,
    source_file_name: Optional[str] = None,
    start_after_source_row: int = 1,
) -> Iterator[DiagnosticEventRecord]:
    """Stream Dataset A rows → edge-event DiagnosticEventRecord rows.

    One source tweet may emit multiple records (one per relation target).
    Dataset A ``id`` is marked untrusted (float-lossy) and must not be used as
    an exact dedup key.
    """
    path = Path(path)
    file_name = source_file_name or path.name
    sheet, header, rows = iter_xlsx_rows(
        path,
        expected_sheet=DATASET_A_SHEET_NAME,
        start_after_row_number=start_after_source_row,
    )
    if sheet != DATASET_A_SHEET_NAME:
        raise UnsupportedSchemaError(
            f"adapter={DATASET_A_ADAPTER_ID} expected sheet {DATASET_A_SHEET_NAME!r}, "
            f"got {sheet!r}"
        )
    col = validate_required_columns(
        header,
        DATASET_A_REQUIRED_COLUMNS,
        adapter_id=DATASET_A_ADAPTER_ID,
        allow_extra=True,
    )

    for row_number, row in enumerate(
        rows,
        start=max(2, int(start_after_source_row) + 1),
    ):
        cells = _row_dict(col, row)
        user = parse_user_blob(cells.get("user"))
        author_id = user.account_id if user.ok else None
        node_idx = node_lookup.get(author_id) if author_id else None

        raw_text = cells.get("text")
        norm = normalize_text(raw_text)
        text_for_diag = norm.text_normalized
        edge_text_available = norm.quality == "ok"

        # Untrusted float id — keep string form only for provenance diagnostics
        raw_id = cells.get("id")
        tweet_id_str: Optional[str]
        if isinstance(raw_id, float):
            tweet_id_str = format(raw_id, ".0f")  # lossy; flagged untrusted
        elif raw_id is None or raw_id == "":
            tweet_id_str = None
        else:
            tweet_id_str = str(raw_id).strip()

        targets: List[Tuple[str, str]] = []
        for mid in _extract_mention_ids(cells.get("user_mentions")):
            targets.append(("mention", mid))
        rid = _extract_nested_user_id(cells.get("retweeted_status"))
        if rid:
            targets.append(("retweet", rid))
        pid = _extract_nested_user_id(cells.get("reply_status"))
        if pid:
            targets.append(("reply", pid))
        qid = _extract_nested_user_id(cells.get("quoted_status"))
        if qid:
            targets.append(("quote", qid))

        if not targets:
            # Structure-inactive / relation-missing row still useful for calendar
            # and missing-field diagnostics: emit one record with relation=None.
            yield DiagnosticEventRecord(
                dataset="A",
                source_file=file_name,
                source_row_number=row_number,
                timestamp_raw=cells.get("created_at"),
                external_user_id=author_id,
                tweet_id=tweet_id_str,
                text=text_for_diag,
                relation=None,
                target_external_user_id=None,
                node_idx=node_idx,
                target_node_idx=None,
                struct_active=False,
                node_text_available=False,
                edge_text_available=edge_text_available,
                extra={
                    "adapter_id": DATASET_A_ADAPTER_ID,
                    "text_field_label": DATASET_A_TEXT_FIELD_LABEL,
                    "tweet_id_trusted": DATASET_A_TWEET_ID_TRUSTED,
                    "user_parse_ok": user.ok,
                    "no_relation_targets": True,
                },
            )
            continue

        for relation, target_id in targets:
            tgt_idx = node_lookup.get(target_id)
            yield DiagnosticEventRecord(
                dataset="A",
                source_file=file_name,
                source_row_number=row_number,
                timestamp_raw=cells.get("created_at"),
                external_user_id=author_id,
                tweet_id=tweet_id_str,
                text=text_for_diag,
                relation=relation,
                target_external_user_id=target_id,
                referenced_status_id=None,  # not reliably extractable without guessing
                node_idx=node_idx,
                target_node_idx=tgt_idx,
                struct_active=True,
                node_text_available=False,
                edge_text_available=edge_text_available,
                extra={
                    "adapter_id": DATASET_A_ADAPTER_ID,
                    "text_field_label": DATASET_A_TEXT_FIELD_LABEL,
                    "tweet_id_trusted": DATASET_A_TWEET_ID_TRUSTED,
                    "user_parse_ok": user.ok,
                    # referenced_status_id left unresolved — not silently guessed
                    "referenced_status_id_status": "UNRESOLVED_NOT_EXTRACTED",
                },
            )


ADAPTER_REGISTRY = {
    DATASET_A_ADAPTER_ID: iter_dataset_a_records,
    DATASET_B_ADAPTER_ID: iter_dataset_b_records,
}


def get_adapter(adapter_id: str):
    if adapter_id not in ADAPTER_REGISTRY:
        raise AdapterConfigurationError(
            f"unknown adapter_id={adapter_id!r}; "
            f"supported={sorted(ADAPTER_REGISTRY)}"
        )
    return ADAPTER_REGISTRY[adapter_id]
