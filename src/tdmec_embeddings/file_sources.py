"""Bounded, deterministic readers for immutable preprocessing artifacts.

This module is intentionally limited to source access. Eligibility filtering,
duplicate selection, hashing of cleaned text, encoding, pooling, and output
publication belong to later embedding subtasks.

Ordering is deterministic without loading a corpus into memory: shards use a
natural path order (so ``part-2`` precedes ``part-10``), and rows retain their
physical Parquet order. Dataset A publication already writes canonical events
and edges in canonical key order; retaining physical order therefore preserves
their published identity and alignment.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Literal, Mapping, Optional, Protocol, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tdmec import constants as C
from tdmec.hashing import sha256_file


SourceKind = Literal["dataset_a", "dataset_b"]


class FileSourceError(RuntimeError):
    """Raised when an immutable file source violates its declared contract."""


@dataclass(frozen=True)
class FileSourceIdentity:
    """Validated identity of one preprocessing source run.

    ``artifact_root`` is operational state and is deliberately omitted from
    :meth:`provenance`; absolute Studio paths must not enter scientific hashes.
    """

    source_kind: SourceKind
    run_id: str
    artifact_root: Path
    manifest_sha256: str
    checksums_sha256: str
    config_hash: str
    git_commit: str
    artifact_status: str
    certification_status: str
    dedup_status: str
    calendar_status: str
    n_nodes: int
    n_snapshots: int
    relation_order: tuple[str, ...]
    node_map_sha256: Optional[str] = None
    recovery_mode: str = "FILE_ARTIFACT_SOURCE"

    def provenance(self) -> Dict[str, Any]:
        """Return a path-free, privacy-safe identity payload."""

        return {
            "source_kind": self.source_kind,
            "run_id": self.run_id,
            "manifest_sha256": self.manifest_sha256,
            "checksums_sha256": self.checksums_sha256,
            "config_hash": self.config_hash,
            "git_commit": self.git_commit,
            "artifact_status": self.artifact_status,
            "certification_status": self.certification_status,
            "dedup_status": self.dedup_status,
            "calendar_status": self.calendar_status,
            "n_nodes": self.n_nodes,
            "n_snapshots": self.n_snapshots,
            "relation_order": list(self.relation_order),
            "node_map_sha256": self.node_map_sha256,
            "recovery_mode": self.recovery_mode,
        }


@dataclass(frozen=True)
class FileRecordBatch:
    """One bounded batch and its non-sensitive physical provenance."""

    source: FileSourceIdentity
    shard_relative_path: str
    shard_index: int
    batch_index: int
    shard_row_offset: int
    global_row_offset: int
    records: pa.RecordBatch

    @property
    def num_rows(self) -> int:
        return self.records.num_rows


class RecordBatchReader(Protocol):
    """Backend-neutral batch-reader surface for later PostgreSQL adapters."""

    identity: FileSourceIdentity
    total_rows: int

    def iter_batches(self) -> Iterator[FileRecordBatch]: ...


_DIGIT_RE = re.compile(r"(\d+)")
_EDGE_PART_RE = re.compile(r"(?:^|/)snapshot=(\d+)/relation=(\d+)/")


def _natural_key(value: str) -> tuple[tuple[int, object], ...]:
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part.casefold())
        for part in _DIGIT_RE.split(value)
        if part
    )


def _read_json_object(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileSourceError(f"required metadata file is missing: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FileSourceError(f"invalid metadata JSON: {path.name}") from exc
    if not isinstance(value, dict):
        raise FileSourceError(f"metadata must be a JSON object: {path.name}")
    return value


def _safe_artifact_path(root: Path, relative_name: str) -> Path:
    relative = Path(relative_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise FileSourceError("unsafe path in checksum manifest")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise FileSourceError("checksum path escapes source artifact root") from exc
    return candidate


def load_file_source_identity(
    artifact_root: str | Path,
    *,
    source_kind: SourceKind,
    expected_run_id: str,
    verify_checksums: bool = True,
) -> FileSourceIdentity:
    """Validate metadata and return an independent Dataset A/B source identity."""

    root = Path(artifact_root)
    if not root.is_dir():
        raise FileSourceError("source artifact root does not exist")
    manifest_path = root / "manifest.json"
    validation_path = root / "validation_report.json"
    checksums_path = root / "checksums.json"
    manifest = _read_json_object(manifest_path)
    validation = _read_json_object(validation_path)
    checksums = _read_json_object(checksums_path)

    run_id = str(manifest.get("run_id", ""))
    if run_id != expected_run_id:
        raise FileSourceError(
            f"source run mismatch: expected {expected_run_id!r}, got {run_id!r}"
        )
    validation_run = validation.get("run_id")
    if validation_run not in (None, run_id):
        raise FileSourceError("validation report run_id does not match manifest")
    if validation.get("all_passed") is not True:
        raise FileSourceError("source validation report does not have all_passed=true")
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in checksums.items()):
        raise FileSourceError("checksums.json must map relative paths to digests")

    if verify_checksums:
        for relative_name, expected in checksums.items():
            path = _safe_artifact_path(root, relative_name)
            if not path.is_file():
                raise FileSourceError(f"checksummed artifact is missing: {relative_name}")
            if sha256_file(path) != expected:
                raise FileSourceError(f"artifact checksum mismatch: {relative_name}")

    canonical = manifest.get("canonical")
    if not isinstance(canonical, Mapping):
        raise FileSourceError("manifest canonical contract is missing")
    try:
        n_nodes = int(canonical["frozen_node_count"])
        n_snapshots = int(canonical["snapshot_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FileSourceError("manifest node/snapshot identity is invalid") from exc
    if n_nodes <= 0 or n_snapshots <= 0:
        raise FileSourceError("manifest node/snapshot counts must be positive")
    if source_kind == "dataset_a":
        relation_order = tuple(canonical.get("relation_order", ()))
        if relation_order != C.RELATION_ORDER:
            raise FileSourceError("Dataset A relation order violates QREL-01")
    else:
        relation_order = C.RELATION_ORDER

    return FileSourceIdentity(
        source_kind=source_kind,
        run_id=run_id,
        artifact_root=root.resolve(),
        manifest_sha256=sha256_file(manifest_path),
        checksums_sha256=sha256_file(checksums_path),
        config_hash=str(manifest.get("config_hash", "")),
        git_commit=str(manifest.get("git_commit", "")),
        artifact_status=str(manifest.get("artifact_status", "UNVALIDATED")),
        certification_status=str(
            canonical.get(
                "certification_status", manifest.get("certification_status", "UNVALIDATED")
            )
        ),
        dedup_status=str(
            canonical.get("dedup_status", manifest.get("dedup_status", "UNVALIDATED"))
        ),
        calendar_status=str(
            canonical.get("calendar_status", manifest.get("calendar_status", "UNVALIDATED"))
        ),
        n_nodes=n_nodes,
        n_snapshots=n_snapshots,
        relation_order=relation_order,
        node_map_sha256=(
            str(manifest["node_map_sha256"])
            if manifest.get("node_map_sha256")
            else None
        ),
    )


def _is_string(dtype: pa.DataType) -> bool:
    return pa.types.is_string(dtype) or pa.types.is_large_string(dtype)


def _is_integer(dtype: pa.DataType) -> bool:
    return pa.types.is_integer(dtype)


def _is_float(dtype: pa.DataType) -> bool:
    return pa.types.is_floating(dtype)


def _is_boolean(dtype: pa.DataType) -> bool:
    return pa.types.is_boolean(dtype)


def _validate_schema(
    schema: pa.Schema,
    required: Mapping[str, Any],
    *,
    shard_name: str,
) -> None:
    missing = [name for name in required if name not in schema.names]
    if missing:
        raise FileSourceError(
            f"Parquet schema missing required fields in {shard_name}: {missing}"
        )
    for name, predicate in required.items():
        dtype = schema.field(name).type
        if not predicate(dtype):
            raise FileSourceError(
                f"Parquet field {name!r} has incompatible type {dtype} in {shard_name}"
            )


def _require_non_null(batch: pa.RecordBatch, names: Sequence[str]) -> None:
    for name in names:
        if batch.column(batch.schema.get_field_index(name)).null_count:
            raise FileSourceError(f"required identity field contains nulls: {name}")


def _integer_bounds(batch: pa.RecordBatch, name: str, low: int, high: int) -> None:
    values = batch.column(batch.schema.get_field_index(name)).to_numpy(
        zero_copy_only=False
    )
    if values.size and (np.any(values < low) or np.any(values > high)):
        raise FileSourceError(f"{name} contains values outside [{low}, {high}]")


class _ParquetBatchReader:
    """Shared deterministic bounded Parquet iteration."""

    columns: tuple[str, ...] = ()
    output_names: tuple[str, ...] = ()
    required_schema: Mapping[str, Any] = {}

    def __init__(
        self,
        artifact_root: str | Path,
        *,
        expected_run_id: str,
        source_kind: SourceKind,
        batch_size: int = 4096,
        max_rows: Optional[int] = None,
        verify_checksums: bool = True,
        identity: Optional[FileSourceIdentity] = None,
    ) -> None:
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        if max_rows is not None and (
            not isinstance(max_rows, int)
            or isinstance(max_rows, bool)
            or max_rows < 0
        ):
            raise ValueError("max_rows must be a non-negative integer or None")
        root = Path(artifact_root).resolve()
        self.identity = identity or load_file_source_identity(
            root,
            source_kind=source_kind,
            expected_run_id=expected_run_id,
            verify_checksums=verify_checksums,
        )
        if self.identity.artifact_root != root:
            raise FileSourceError("provided source identity belongs to another artifact root")
        if (
            self.identity.source_kind != source_kind
            or self.identity.run_id != expected_run_id
        ):
            raise FileSourceError("provided source identity is incompatible with reader")
        self.batch_size = batch_size
        self.max_rows = max_rows
        self._paths = self._discover_paths()
        if not self._paths:
            raise FileSourceError("no source Parquet shards found")
        self.total_rows = 0
        for path in self._paths:
            relative = path.relative_to(root).as_posix()
            parquet = pq.ParquetFile(path)
            _validate_schema(
                parquet.schema_arrow,
                self.required_schema,
                shard_name=relative,
            )
            self.total_rows += parquet.metadata.num_rows

    @property
    def artifact_root(self) -> Path:
        return self.identity.artifact_root

    @property
    def shard_relative_paths(self) -> tuple[str, ...]:
        return tuple(
            path.relative_to(self.artifact_root).as_posix() for path in self._paths
        )

    def _discover_paths(self) -> list[Path]:
        raise NotImplementedError

    def _validate_batch(self, batch: pa.RecordBatch, relative_path: str) -> None:
        raise NotImplementedError

    def _canonicalize(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        if not self.output_names or self.output_names == self.columns:
            return batch
        return pa.RecordBatch.from_arrays(
            list(batch.columns), names=list(self.output_names)
        )

    def iter_batches(self) -> Iterator[FileRecordBatch]:
        emitted = 0
        global_batch_index = 0
        for shard_index, path in enumerate(self._paths):
            relative = path.relative_to(self.artifact_root).as_posix()
            shard_offset = 0
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(
                batch_size=self.batch_size,
                columns=list(self.columns),
                use_threads=False,
            ):
                if self.max_rows is not None:
                    remaining = self.max_rows - emitted
                    if remaining <= 0:
                        return
                    if batch.num_rows > remaining:
                        batch = batch.slice(0, remaining)
                canonical = self._canonicalize(batch)
                self._validate_batch(canonical, relative)
                yield FileRecordBatch(
                    source=self.identity,
                    shard_relative_path=relative,
                    shard_index=shard_index,
                    batch_index=global_batch_index,
                    shard_row_offset=shard_offset,
                    global_row_offset=emitted,
                    records=canonical,
                )
                shard_offset += canonical.num_rows
                emitted += canonical.num_rows
                global_batch_index += 1


class NodeTextFileReader(_ParquetBatchReader):
    """Dataset B normalized-text reader; no eligibility decisions are applied."""

    columns = (
        "tweet_id",
        "node_index",
        "snapshot_id",
        "cleaned_text",
        "text_quality",
        "source_file",
        "source_sheet",
        "source_row_number",
        "is_duplicate",
        "is_canonical_duplicate",
    )
    output_names = columns
    required_schema = {
        "tweet_id": _is_string,
        "node_index": _is_integer,
        "snapshot_id": _is_integer,
        "cleaned_text": _is_string,
        "text_quality": _is_string,
        "source_file": _is_string,
        "source_sheet": _is_string,
        "source_row_number": _is_integer,
        "is_duplicate": _is_boolean,
        "is_canonical_duplicate": _is_boolean,
    }

    def __init__(self, artifact_root: str | Path, **kwargs: Any) -> None:
        super().__init__(artifact_root, source_kind="dataset_b", **kwargs)

    def _discover_paths(self) -> list[Path]:
        base = self.artifact_root / "normalized_records"
        return sorted(
            base.glob("**/*.parquet"),
            key=lambda p: _natural_key(p.relative_to(base).as_posix()),
        )

    def _validate_batch(self, batch: pa.RecordBatch, relative_path: str) -> None:
        del relative_path
        _require_non_null(
            batch,
            (
                "tweet_id",
                "node_index",
                "snapshot_id",
                "source_file",
                "source_row_number",
            ),
        )
        _integer_bounds(batch, "node_index", 0, self.identity.n_nodes - 1)
        _integer_bounds(batch, "snapshot_id", 0, self.identity.n_snapshots - 1)


class EventTextFileReader(_ParquetBatchReader):
    """Dataset A canonical-event reader; no eligibility filtering is applied."""

    columns = (
        "signature",
        "snapshot_id",
        "relation_id",
        "source_idx",
        "target_idx",
        "cleaned_text",
        "text_hash",
        "text_quality",
        "source_file",
        "source_row_number",
    )
    output_names = columns
    required_schema = {
        "signature": _is_string,
        "snapshot_id": _is_integer,
        "relation_id": _is_integer,
        "source_idx": _is_integer,
        "target_idx": _is_integer,
        "cleaned_text": _is_string,
        "text_hash": _is_string,
        "text_quality": _is_string,
        "source_file": _is_string,
        "source_row_number": _is_integer,
    }

    def __init__(self, artifact_root: str | Path, **kwargs: Any) -> None:
        super().__init__(artifact_root, source_kind="dataset_a", **kwargs)

    def _discover_paths(self) -> list[Path]:
        base = self.artifact_root / "events"
        return sorted(
            base.glob("**/*.parquet"),
            key=lambda p: _natural_key(p.relative_to(base).as_posix()),
        )

    def _validate_batch(self, batch: pa.RecordBatch, relative_path: str) -> None:
        del relative_path
        _require_non_null(
            batch,
            (
                "signature",
                "snapshot_id",
                "relation_id",
                "source_idx",
                "target_idx",
                "source_file",
                "source_row_number",
            ),
        )
        _integer_bounds(batch, "source_idx", 0, self.identity.n_nodes - 1)
        _integer_bounds(batch, "target_idx", 0, self.identity.n_nodes - 1)
        _integer_bounds(batch, "snapshot_id", 0, self.identity.n_snapshots - 1)
        _integer_bounds(
            batch, "relation_id", 0, len(self.identity.relation_order) - 1
        )


class CanonicalEdgeFileReader(_ParquetBatchReader):
    """Dataset A canonical-edge reader with normalized endpoint field names."""

    columns = (
        "snapshot_id",
        "relation_id",
        "src_index",
        "dst_index",
        "count_raw",
        "weight_log1p",
    )
    output_names = (
        "snapshot_id",
        "relation_id",
        "source_idx",
        "target_idx",
        "count_raw",
        "weight_log1p",
    )
    required_schema = {
        "snapshot_id": _is_integer,
        "relation_id": _is_integer,
        "src_index": _is_integer,
        "dst_index": _is_integer,
        "count_raw": _is_integer,
        "weight_log1p": _is_float,
    }

    def __init__(self, artifact_root: str | Path, **kwargs: Any) -> None:
        super().__init__(artifact_root, source_kind="dataset_a", **kwargs)

    def _discover_paths(self) -> list[Path]:
        base = self.artifact_root / "edges"

        def edge_key(path: Path) -> tuple[Any, ...]:
            relative = path.relative_to(base).as_posix()
            match = _EDGE_PART_RE.search("/" + relative)
            if not match:
                return (10**9, 10**9, _natural_key(relative))
            return (
                int(match.group(1)),
                int(match.group(2)),
                _natural_key(relative),
            )

        return sorted(base.glob("**/*.parquet"), key=edge_key)

    def _validate_batch(self, batch: pa.RecordBatch, relative_path: str) -> None:
        _require_non_null(batch, self.output_names)
        _integer_bounds(batch, "source_idx", 0, self.identity.n_nodes - 1)
        _integer_bounds(batch, "target_idx", 0, self.identity.n_nodes - 1)
        _integer_bounds(batch, "snapshot_id", 0, self.identity.n_snapshots - 1)
        _integer_bounds(
            batch, "relation_id", 0, len(self.identity.relation_order) - 1
        )
        source = batch.column(
            batch.schema.get_field_index("source_idx")
        ).to_numpy(zero_copy_only=False)
        target = batch.column(
            batch.schema.get_field_index("target_idx")
        ).to_numpy(zero_copy_only=False)
        if np.any(source == target):
            raise FileSourceError("canonical edge source contains a self-loop")
        counts = batch.column(batch.schema.get_field_index("count_raw")).to_numpy(
            zero_copy_only=False
        )
        weights = batch.column(
            batch.schema.get_field_index("weight_log1p")
        ).to_numpy(zero_copy_only=False)
        if np.any(counts <= 0):
            raise FileSourceError("canonical edge count_raw must be positive")
        if not np.all(np.isfinite(weights)):
            raise FileSourceError("canonical edge weights contain NaN or Inf")
        if not np.allclose(
            weights,
            np.log1p(counts),
            rtol=C.WEIGHT_LOG1P_RTOL,
            atol=C.WEIGHT_LOG1P_ATOL,
        ):
            raise FileSourceError(
                "canonical edge weight_log1p is inconsistent with count_raw"
            )
        match = _EDGE_PART_RE.search("/" + relative_path)
        if match:
            expected_snapshot = int(match.group(1))
            expected_relation = int(match.group(2))
            snapshots = batch.column(
                batch.schema.get_field_index("snapshot_id")
            ).to_numpy(zero_copy_only=False)
            relations = batch.column(
                batch.schema.get_field_index("relation_id")
            ).to_numpy(zero_copy_only=False)
            if np.any(snapshots != expected_snapshot) or np.any(
                relations != expected_relation
            ):
                raise FileSourceError(
                    "edge partition identity does not match row identity"
                )


__all__ = [
    "CanonicalEdgeFileReader",
    "EventTextFileReader",
    "FileRecordBatch",
    "FileSourceError",
    "FileSourceIdentity",
    "NodeTextFileReader",
    "RecordBatchReader",
    "load_file_source_identity",
]
