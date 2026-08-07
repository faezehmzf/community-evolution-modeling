"""Emit mapped Dataset A relation events from workbook rows."""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterator, List, Optional, Sequence, Tuple

from tdmec import constants as C
from tdmec_diagnostics.adapters import (
    _extract_mention_ids,
    _extract_nested_user_id,
)
from tdmec_diagnostics.schema_contracts import (
    DATASET_A_REQUIRED_COLUMNS,
    DATASET_A_SHEET_NAME,
)
from tdmec_diagnostics.workbook_io import iter_xlsx_rows, validate_required_columns
from tdmec_pilot.snapshots import assign_snapshot
from tdmec_pilot.text_quality import normalize_text
from tdmec_pilot.timestamps import parse_created_at
from tdmec_pilot.user_blob import parse_user_blob

from .dedup_a import composite_signature, text_content_hash


@dataclass(frozen=True)
class AuthoredTweet:
    snapshot_id: int
    source_idx: int
    source_file: str
    source_row_number: int


@dataclass(frozen=True)
class GraphEvent:
    signature: str
    snapshot_id: int
    relation_id: int
    source_idx: int
    target_idx: int
    author_account_id: str
    target_account_id: str
    created_at_utc_iso: str
    cleaned_text: str
    text_hash: str
    source_file: str
    source_row_number: int
    text_quality: str


@dataclass
class RowAccounting:
    rows_inspected: int = 0
    authored_retained: int = 0
    events_emitted: int = 0
    missing_author: int = 0
    author_outside_universe: int = 0
    invalid_timestamp: int = 0
    outside_calendar: int = 0
    no_relation_targets: int = 0
    external_targets: int = 0
    self_loops: int = 0

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def add(self, other: "RowAccounting") -> None:
        for f in fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))


def iter_graph_events_from_rows(
    rows: Sequence[Sequence[Any]],
    *,
    header: Sequence[str],
    source_file: str,
    node_lookup,
    start_row_number: int,
    accounting: Optional[RowAccounting] = None,
) -> Iterator[Tuple[Optional[AuthoredTweet], List[GraphEvent]]]:
    """Yield (authored_tweet_or_None, events) per source row."""
    acc = accounting or RowAccounting()
    col = {c: i for i, c in enumerate(header)}
    missing = [c for c in DATASET_A_REQUIRED_COLUMNS if c not in col]
    if missing:
        raise ValueError(f"{source_file}: missing required columns {missing}")

    for offset, row in enumerate(rows):
        row_number = start_row_number + offset
        acc.rows_inspected += 1
        cells = {name: (row[idx] if idx < len(row) else None) for name, idx in col.items()}

        user = parse_user_blob(cells.get("user"))
        if not user.ok or not user.account_id:
            acc.missing_author += 1
            yield None, []
            continue
        source_idx = node_lookup.get(user.account_id)
        if source_idx is None:
            acc.author_outside_universe += 1
            yield None, []
            continue

        ts = parse_created_at(cells.get("created_at"))
        if not ts.ok or ts.utc is None:
            acc.invalid_timestamp += 1
            yield None, []
            continue
        snapshot_id = assign_snapshot(ts.utc)
        if snapshot_id is None:
            acc.outside_calendar += 1
            yield None, []
            continue

        txt = normalize_text(cells.get("text"))
        cleaned = txt.text_normalized or ""
        th = text_content_hash(cleaned if txt.quality == "ok" else "")
        created_iso = ts.utc.isoformat()

        authored = AuthoredTweet(
            snapshot_id=snapshot_id,
            source_idx=int(source_idx),
            source_file=source_file,
            source_row_number=row_number,
        )
        acc.authored_retained += 1

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
            acc.no_relation_targets += 1
            yield authored, []
            continue

        events: List[GraphEvent] = []
        for relation, target_id in targets:
            if target_id == user.account_id:
                acc.self_loops += 1
                continue
            target_idx = node_lookup.get(target_id)
            if target_idx is None:
                acc.external_targets += 1
                continue
            relation_id = int(C.RELATION_TO_ID[relation])
            sig = composite_signature(
                author_account_id=user.account_id,
                created_at_utc_iso=created_iso,
                text_hash=th,
                relation_id=relation_id,
                target_account_id=target_id,
            )
            events.append(
                GraphEvent(
                    signature=sig,
                    snapshot_id=snapshot_id,
                    relation_id=relation_id,
                    source_idx=int(source_idx),
                    target_idx=int(target_idx),
                    author_account_id=user.account_id,
                    target_account_id=target_id,
                    created_at_utc_iso=created_iso,
                    cleaned_text=cleaned if txt.quality == "ok" else "",
                    text_hash=th,
                    source_file=source_file,
                    source_row_number=row_number,
                    text_quality=txt.quality,
                )
            )
        acc.events_emitted += len(events)
        yield authored, events


def stream_workbook_events(
    path,
    *,
    node_lookup,
    source_file_name: Optional[str] = None,
    start_after_source_row: int = 1,
    chunk_size: int = 100000,
) -> Iterator[Tuple[str, int, List[Tuple[Optional[AuthoredTweet], List[GraphEvent]]], RowAccounting]]:
    """Stream a workbook in chunks: yield (file, chunk_idx, row_results, accounting)."""
    path = Path(path)
    file_name = source_file_name or path.name
    sheet, header, rows_iter = iter_xlsx_rows(
        path,
        expected_sheet=DATASET_A_SHEET_NAME,
        start_after_row_number=start_after_source_row,
    )
    if sheet != DATASET_A_SHEET_NAME:
        raise ValueError(f"expected sheet {DATASET_A_SHEET_NAME!r}, got {sheet!r}")
    validate_required_columns(
        header,
        DATASET_A_REQUIRED_COLUMNS,
        adapter_id="dataset_a_xlsx_v1",
        allow_extra=True,
    )

    chunk: List[Sequence[Any]] = []
    chunk_idx = 0
    chunk_start_row = max(2, int(start_after_source_row) + 1)
    next_row = chunk_start_row
    for row_number, row in enumerate(rows_iter, start=chunk_start_row):
        if not chunk:
            chunk_start_row = row_number
        chunk.append(row)
        next_row = row_number + 1
        if len(chunk) >= chunk_size:
            acc = RowAccounting()
            results = list(
                iter_graph_events_from_rows(
                    chunk,
                    header=header,
                    source_file=file_name,
                    node_lookup=node_lookup,
                    start_row_number=chunk_start_row,
                    accounting=acc,
                )
            )
            yield file_name, chunk_idx, results, acc
            chunk_idx += 1
            chunk = []
    if chunk:
        acc = RowAccounting()
        results = list(
            iter_graph_events_from_rows(
                chunk,
                header=header,
                source_file=file_name,
                node_lookup=node_lookup,
                start_row_number=chunk_start_row,
                accounting=acc,
            )
        )
        yield file_name, chunk_idx, results, acc
