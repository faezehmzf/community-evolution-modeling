"""Atomic, resumable file backend for unit-level embedding shards.

The backend is intentionally independent of PostgreSQL/pgvector.  It commits
one deterministic eligibility batch per Parquet shard and stores enough
self-describing metadata inside the shard to recover if interruption occurs
after shard publication but before checkpoint publication.

Raw text is never persisted here.  Unit identities and source provenance are
private data-artifact fields; manifests, checkpoints, sidecars, exceptions,
and return values contain hashes and aggregate counts only.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tdmec.hashing import hash_canonical, sha256_bytes, sha256_file

from .eligibility import EligibilityBatch, Modality, STATUS_LABELS
from .file_sources import FileSourceIdentity
from .mock_encoder import EncoderMetadata


CommitStatus = Literal["COMMITTED", "SKIPPED_ALREADY_COMMITTED"]
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_SHARD_METADATA_KEY = b"tdmec_embedding_shard_v1"


class FileEmbeddingWriterError(RuntimeError):
    """Base class for safe file-writer failures."""


class WriterCompatibilityError(FileEmbeddingWriterError):
    """Raised when existing state is incompatible with the requested run."""


class VectorValidationError(FileEmbeddingWriterError):
    """Raised when unit vectors are malformed or numerically invalid."""


@dataclass(frozen=True)
class VectorValidationPolicy:
    min_norm: float = 1e-8
    max_norm: float = 1e6
    # After float32 re-L2, norms are typically within ~1e-7 of 1.0.
    # 1e-6 is a tight scientific gate that still absorbs float32 reduction noise.
    normalized_atol: float = 1e-6

    def __post_init__(self) -> None:
        if not 0.0 < self.min_norm < self.max_norm:
            raise ValueError("vector norm limits are invalid")
        if self.normalized_atol <= 0.0:
            raise ValueError("normalized_atol must be positive")

    def payload(self) -> Dict[str, float]:
        return {
            "min_norm": self.min_norm,
            "max_norm": self.max_norm,
            "normalized_atol": self.normalized_atol,
        }

    @classmethod
    def from_encoder_metadata(cls, encoder: EncoderMetadata) -> "VectorValidationPolicy":
        return cls(normalized_atol=float(encoder.normalized_atol))


@dataclass(frozen=True)
class FileEmbeddingRunSpec:
    """Immutable compatibility identity for one modality of an embedding run."""

    embedding_run_id: str
    modality: Modality
    source: FileSourceIdentity
    preprocessing_hash: str
    encoder: EncoderMetadata
    vector_validation: VectorValidationPolicy = VectorValidationPolicy()
    shard_schema_version: str = "tdmec-unit-embedding-parquet-v1"

    def __post_init__(self) -> None:
        if not _RUN_ID_RE.fullmatch(self.embedding_run_id):
            raise ValueError("embedding_run_id is invalid")
        if self.embedding_run_id == self.source.run_id:
            raise ValueError("embedding_run_id must differ from the source run_id")
        expected_kind = "dataset_b" if self.modality == "node_text" else "dataset_a"
        if self.source.source_kind != expected_kind:
            raise ValueError("embedding modality and source kind are inconsistent")
        if not _HEX64_RE.fullmatch(self.preprocessing_hash):
            raise ValueError("preprocessing_hash must be a lowercase SHA-256 digest")

    def compatibility_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": self.shard_schema_version,
            "embedding_run_id": self.embedding_run_id,
            "modality": self.modality,
            "source": self.source.provenance(),
            "preprocessing_hash": self.preprocessing_hash,
            "encoder": self.encoder.payload(),
            "model_hash": self.encoder.model_hash,
            "vector_validation": self.vector_validation.payload(),
            "backend": "FILE_PARQUET",
            "deduplication_contract": (
                "eligible_unique_units_plus_idempotent_source_batch_commits_v1"
            ),
        }

    @property
    def compatibility_hash(self) -> str:
        return hash_canonical(self.compatibility_payload())


@dataclass(frozen=True)
class ShardCommit:
    """Privacy-safe result of one idempotent batch transaction."""

    status: CommitStatus
    batch_key: str
    row_count: int
    shard_relative_path: Optional[str]
    shard_sha256: Optional[str]
    total_committed_rows: int


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FileEmbeddingWriterError(f"invalid writer state file: {path.name}") from exc
    if not isinstance(value, dict):
        raise FileEmbeddingWriterError(f"writer state must be an object: {path.name}")
    return value


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    tmp = Path(handle.name)
    try:
        with handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def _atomic_write_parquet(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    tmp = Path(tmp_name)
    try:
        pq.write_table(table, tmp, compression="zstd", use_dictionary=True)
        with tmp.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def _vector_sha256(vectors: np.ndarray) -> str:
    canonical = np.ascontiguousarray(vectors, dtype="<f4")
    return sha256_bytes(canonical.tobytes(order="C"))


def _batch_key(batch: EligibilityBatch) -> str:
    if batch.source_batch_index < 0 or batch.source_global_row_offset < 0:
        raise FileEmbeddingWriterError("source batch identity must be non-negative")
    return (
        f"b{batch.source_batch_index:08d}"
        f"-o{batch.source_global_row_offset:012d}"
    )


def _batch_input_hash(batch: EligibilityBatch) -> str:
    return hash_canonical(
        {
            "modality": batch.modality,
            "source_batch_index": batch.source_batch_index,
            "source_global_row_offset": batch.source_global_row_offset,
            "ordered_units": [
                {
                    "unit_hash": unit.unit_hash,
                    "content_hash": unit.content_hash,
                    "preprocessing_hash": unit.preprocessing_hash,
                }
                for unit in batch.units
            ],
        }
    )


class FileEmbeddingWriter:
    """Transactional Parquet writer for one embedding modality."""

    def __init__(
        self,
        output_namespace: str | Path,
        spec: FileEmbeddingRunSpec,
    ) -> None:
        self.spec = spec
        self.run_root = Path(output_namespace).resolve() / spec.embedding_run_id
        if self.run_root == spec.source.artifact_root:
            raise ValueError("embedding output must not overwrite its source artifact root")
        self.manifest_path = self.run_root / "manifests" / f"{spec.modality}.json"
        self.checkpoint_path = (
            self.run_root / "checkpoints" / f"{spec.modality}.json"
        )
        self.checksums_path = self.run_root / "checksums" / f"{spec.modality}.json"
        self.shard_dir = self.run_root / "unit_embeddings" / spec.modality
        self._checkpoint: Dict[str, Any]

        if self.manifest_path.is_file():
            manifest = _read_json(self.manifest_path)
            if manifest.get("compatibility_hash") != spec.compatibility_hash:
                status = manifest.get("status")
                raise WriterCompatibilityError(
                    "existing embedding manifest is incompatible with this run specification "
                    f"(existing_status={status!r}). If the prior run is incomplete, re-run with "
                    "--replace-incomplete; completed outputs cannot be overwritten."
                )
        elif self.checkpoint_path.exists() or self.shard_dir.exists():
            raise WriterCompatibilityError(
                "writer artifacts exist without an authoritative modality manifest "
                "(likely an interrupted run). Re-run with --replace-incomplete or --resume "
                "after restoring a compatible checkpoint."
            )
        else:
            self._ensure_layout()
            self._checkpoint = self._new_checkpoint()
            self._flush_state()

        self._ensure_layout()
        if self.checkpoint_path.is_file():
            self._checkpoint = _read_json(self.checkpoint_path)
            self._validate_checkpoint_identity()
        elif not hasattr(self, "_checkpoint"):
            self._checkpoint = self._new_checkpoint()
        self._recover_orphan_shards()
        self._validate_all_committed()
        self._flush_state()

    def _ensure_layout(self) -> None:
        for path in (
            self.manifest_path.parent,
            self.checkpoint_path.parent,
            self.checksums_path.parent,
            self.shard_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def _new_checkpoint(self) -> Dict[str, Any]:
        return {
            "schema_version": "tdmec-file-embedding-checkpoint-v1",
            "embedding_run_id": self.spec.embedding_run_id,
            "modality": self.spec.modality,
            "compatibility_hash": self.spec.compatibility_hash,
            "status": "IN_PROGRESS",
            "committed_rows": 0,
            "batches": {},
        }

    def _validate_checkpoint_identity(self) -> None:
        checkpoint = self._checkpoint
        if checkpoint.get("schema_version") != "tdmec-file-embedding-checkpoint-v1":
            raise WriterCompatibilityError("checkpoint schema version is incompatible")
        if (
            checkpoint.get("embedding_run_id") != self.spec.embedding_run_id
            or checkpoint.get("modality") != self.spec.modality
            or checkpoint.get("compatibility_hash") != self.spec.compatibility_hash
        ):
            raise WriterCompatibilityError("checkpoint identity is incompatible")
        if checkpoint.get("status") not in {"IN_PROGRESS", "COMPLETED"}:
            raise FileEmbeddingWriterError("checkpoint status is invalid")
        if not isinstance(checkpoint.get("batches"), dict):
            raise FileEmbeddingWriterError("checkpoint batch registry is invalid")

    def _manifest_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": "tdmec-file-embedding-manifest-v1",
            "embedding_run_id": self.spec.embedding_run_id,
            "modality": self.spec.modality,
            "status": self._checkpoint["status"],
            "artifact_status": "PROVISIONAL_SMOKE_ONLY",
            "status_labels": list(STATUS_LABELS),
            "compatibility_hash": self.spec.compatibility_hash,
            "configuration": self.spec.compatibility_payload(),
            "accounting": {
                "committed_batches": len(self._checkpoint["batches"]),
                "committed_rows": int(self._checkpoint["committed_rows"]),
                "parquet_shards": sum(
                    1
                    for entry in self._checkpoint["batches"].values()
                    if entry.get("shard_relative_path")
                ),
            },
        }

    def _checksums_payload(self) -> Dict[str, str]:
        checksums: Dict[str, str] = {}
        for entry in self._checkpoint["batches"].values():
            shard = entry.get("shard_relative_path")
            metadata = entry.get("metadata_relative_path")
            if shard:
                checksums[str(shard)] = str(entry["shard_sha256"])
            if metadata:
                checksums[str(metadata)] = str(entry["metadata_sha256"])
        return dict(sorted(checksums.items()))

    def _flush_state(self) -> None:
        _atomic_write_json(self.checkpoint_path, self._checkpoint)
        _atomic_write_json(self.manifest_path, self._manifest_payload())
        _atomic_write_json(self.checksums_path, self._checksums_payload())

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.run_root).as_posix()

    def _embedded_metadata(self, path: Path) -> Dict[str, Any]:
        schema_metadata = pq.read_schema(path).metadata or {}
        raw = schema_metadata.get(_SHARD_METADATA_KEY)
        if raw is None:
            raise FileEmbeddingWriterError("embedding shard has no recovery metadata")
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FileEmbeddingWriterError("embedding shard recovery metadata is invalid") from exc
        if not isinstance(value, dict):
            raise FileEmbeddingWriterError("embedding shard recovery metadata is invalid")
        return value

    def _sidecar_path(self, shard_path: Path) -> Path:
        return shard_path.with_suffix(".metadata.json")

    def _entry_from_shard(self, shard_path: Path) -> Dict[str, Any]:
        embedded = self._embedded_metadata(shard_path)
        if (
            embedded.get("compatibility_hash") != self.spec.compatibility_hash
            or embedded.get("embedding_run_id") != self.spec.embedding_run_id
            or embedded.get("modality") != self.spec.modality
        ):
            raise WriterCompatibilityError("orphan shard identity is incompatible")
        batch_key = embedded.get("batch_key")
        if not isinstance(batch_key, str) or not batch_key:
            raise FileEmbeddingWriterError("orphan shard has no valid batch key")
        parquet = pq.ParquetFile(shard_path)
        row_count = int(embedded.get("row_count", -1))
        if row_count < 0 or parquet.metadata.num_rows != row_count:
            raise FileEmbeddingWriterError("embedding shard row accounting failed")
        sidecar_path = self._sidecar_path(shard_path)
        sidecar = dict(embedded)
        sidecar["shard_relative_path"] = self._relative(shard_path)
        sidecar["shard_sha256"] = sha256_file(shard_path)
        _atomic_write_json(sidecar_path, sidecar)
        return {
            "batch_key": batch_key,
            "batch_input_hash": str(embedded["batch_input_hash"]),
            "vector_sha256": str(embedded["vector_sha256"]),
            "row_count": row_count,
            "source_batch_index": int(embedded["source_batch_index"]),
            "source_global_row_offset": int(embedded["source_global_row_offset"]),
            "shard_relative_path": self._relative(shard_path),
            "shard_sha256": sidecar["shard_sha256"],
            "metadata_relative_path": self._relative(sidecar_path),
            "metadata_sha256": sha256_file(sidecar_path),
            "norm_min": embedded.get("norm_min"),
            "norm_max": embedded.get("norm_max"),
        }

    def _recover_orphan_shards(self) -> None:
        batches = self._checkpoint["batches"]
        referenced = {
            entry.get("shard_relative_path") for entry in batches.values()
        }
        changed = False
        for shard_path in sorted(self.shard_dir.glob("part-*.parquet")):
            relative = self._relative(shard_path)
            if relative in referenced:
                continue
            entry = self._entry_from_shard(shard_path)
            key = entry["batch_key"]
            if key in batches:
                raise FileEmbeddingWriterError(
                    "multiple embedding shards claim the same source batch"
                )
            batches[key] = entry
            changed = True
        if changed:
            self._recount()

    def _recount(self) -> None:
        self._checkpoint["committed_rows"] = sum(
            int(entry["row_count"])
            for entry in self._checkpoint["batches"].values()
        )

    def _validate_entry(self, key: str, entry: Mapping[str, Any]) -> None:
        if entry.get("batch_key") != key:
            raise FileEmbeddingWriterError("checkpoint batch-key reconciliation failed")
        row_count = entry.get("row_count")
        if not isinstance(row_count, int) or isinstance(row_count, bool) or row_count < 0:
            raise FileEmbeddingWriterError("checkpoint row count is invalid")
        shard_relative = entry.get("shard_relative_path")
        if row_count == 0:
            if shard_relative is not None:
                raise FileEmbeddingWriterError("empty committed batch unexpectedly has a shard")
            return
        if not isinstance(shard_relative, str):
            raise FileEmbeddingWriterError("committed batch has no shard")
        shard_path = self.run_root / shard_relative
        if not shard_path.is_file() or sha256_file(shard_path) != entry.get("shard_sha256"):
            raise FileEmbeddingWriterError("committed embedding shard checksum mismatch")
        embedded = self._embedded_metadata(shard_path)
        if (
            embedded.get("batch_key") != key
            or embedded.get("batch_input_hash") != entry.get("batch_input_hash")
            or embedded.get("vector_sha256") != entry.get("vector_sha256")
            or int(embedded.get("row_count", -1)) != row_count
        ):
            raise FileEmbeddingWriterError("committed shard metadata mismatch")
        metadata_relative = entry.get("metadata_relative_path")
        if not isinstance(metadata_relative, str):
            raise FileEmbeddingWriterError("committed shard sidecar is missing")
        metadata_path = self.run_root / metadata_relative
        if (
            not metadata_path.is_file()
            or sha256_file(metadata_path) != entry.get("metadata_sha256")
        ):
            raise FileEmbeddingWriterError("committed shard sidecar checksum mismatch")

    def _validate_all_committed(self) -> None:
        for key, entry in sorted(self._checkpoint["batches"].items()):
            self._validate_entry(key, entry)
        expected = sum(
            int(entry["row_count"])
            for entry in self._checkpoint["batches"].values()
        )
        if int(self._checkpoint.get("committed_rows", -1)) != expected:
            raise FileEmbeddingWriterError("checkpoint total-row accounting failed")

    def _validate_batch(self, batch: EligibilityBatch, vectors: np.ndarray) -> np.ndarray:
        if batch.modality != self.spec.modality:
            raise WriterCompatibilityError("eligibility batch modality is incompatible")
        expected_shape = (batch.num_rows, self.spec.encoder.dimension)
        array = np.asarray(vectors)
        if array.shape != expected_shape:
            raise VectorValidationError(
                f"vector shape mismatch: expected {expected_shape}, got {array.shape}"
            )
        if array.dtype != np.float32:
            raise VectorValidationError("unit embedding dtype must be float32")
        if not np.all(np.isfinite(array)):
            raise VectorValidationError("unit embeddings contain NaN or Inf")
        unit_hashes: set[str] = set()
        for unit in batch.units:
            if (
                unit.modality != self.spec.modality
                or unit.source_run_id != self.spec.source.run_id
                or unit.preprocessing_hash != self.spec.preprocessing_hash
            ):
                raise WriterCompatibilityError("eligible unit provenance is incompatible")
            if not _HEX64_RE.fullmatch(unit.unit_hash) or not _HEX64_RE.fullmatch(
                unit.content_hash
            ):
                raise WriterCompatibilityError("eligible unit hash format is invalid")
            if unit.unit_hash in unit_hashes:
                raise FileEmbeddingWriterError("eligibility batch repeats a source unit")
            unit_hashes.add(unit.unit_hash)
        if batch.num_rows:
            norms = np.linalg.norm(array.astype(np.float64), axis=1)
            policy = self.spec.vector_validation
            if np.any(norms < policy.min_norm) or np.any(norms > policy.max_norm):
                raise VectorValidationError("unit embedding norm is outside valid bounds")
            if self.spec.encoder.unit_normalized and not np.allclose(
                norms, 1.0, rtol=0.0, atol=policy.normalized_atol
            ):
                raise VectorValidationError(
                    "unit embedding is not unit-normalized "
                    f"(atol={policy.normalized_atol}; "
                    f"min={float(norms.min()):.8f}; max={float(norms.max()):.8f})"
                )
        return np.ascontiguousarray(array)

    def _table(
        self,
        batch: EligibilityBatch,
        vectors: np.ndarray,
        embedded_metadata: Mapping[str, Any],
    ) -> pa.Table:
        units = batch.units
        flat = pa.array(vectors.reshape(-1), type=pa.float32())
        vector_column = pa.FixedSizeListArray.from_arrays(
            flat, self.spec.encoder.dimension
        )
        table = pa.table(
            {
                "unit_id": pa.array([unit.unit_id for unit in units], pa.string()),
                "unit_hash": pa.array([unit.unit_hash for unit in units], pa.string()),
                "content_hash": pa.array(
                    [unit.content_hash for unit in units], pa.string()
                ),
                "preprocessing_hash": pa.array(
                    [unit.preprocessing_hash for unit in units], pa.string()
                ),
                "snapshot_id": pa.array(
                    [unit.snapshot_id for unit in units], pa.int32()
                ),
                "node_index": pa.array(
                    [unit.node_index for unit in units], pa.int32()
                ),
                "relation_id": pa.array(
                    [unit.relation_id for unit in units], pa.int8()
                ),
                "source_idx": pa.array(
                    [unit.source_idx for unit in units], pa.int32()
                ),
                "target_idx": pa.array(
                    [unit.target_idx for unit in units], pa.int32()
                ),
                "source_file": pa.array(
                    [unit.source_file for unit in units], pa.string()
                ),
                "source_sheet": pa.array(
                    [unit.source_sheet for unit in units], pa.string()
                ),
                "source_row_number": pa.array(
                    [unit.source_row_number for unit in units], pa.int64()
                ),
                "embedding": vector_column,
            }
        )
        metadata = dict(table.schema.metadata or {})
        metadata[_SHARD_METADATA_KEY] = json.dumps(
            embedded_metadata,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return table.replace_schema_metadata(metadata)

    def is_batch_committed(self, batch: EligibilityBatch) -> bool:
        """Check an input transaction before expensive model inference."""

        if batch.modality != self.spec.modality:
            raise WriterCompatibilityError("eligibility batch modality is incompatible")
        seen: set[str] = set()
        for unit in batch.units:
            if (
                unit.modality != self.spec.modality
                or unit.source_run_id != self.spec.source.run_id
                or unit.preprocessing_hash != self.spec.preprocessing_hash
            ):
                raise WriterCompatibilityError("eligible unit provenance is incompatible")
            if unit.unit_hash in seen:
                raise FileEmbeddingWriterError("eligibility batch repeats a source unit")
            seen.add(unit.unit_hash)
        key = _batch_key(batch)
        existing = self._checkpoint["batches"].get(key)
        if existing is None:
            return False
        if (
            existing.get("batch_input_hash") != _batch_input_hash(batch)
            or int(existing.get("row_count", -1)) != batch.num_rows
        ):
            raise WriterCompatibilityError(
                "a committed source batch was replayed with different input content"
            )
        self._validate_entry(key, existing)
        return True

    def write_batch(
        self, batch: EligibilityBatch, vectors: np.ndarray
    ) -> ShardCommit:
        """Atomically commit one batch or return an idempotent resume no-op."""

        array = self._validate_batch(batch, vectors)
        key = _batch_key(batch)
        input_hash = _batch_input_hash(batch)
        output_hash = _vector_sha256(array)
        existing = self._checkpoint["batches"].get(key)
        if existing is not None:
            if (
                existing.get("batch_input_hash") != input_hash
                or existing.get("vector_sha256") != output_hash
                or int(existing.get("row_count", -1)) != batch.num_rows
            ):
                raise WriterCompatibilityError(
                    "a committed source batch was replayed with different content or vectors"
                )
            self._validate_entry(key, existing)
            return ShardCommit(
                status="SKIPPED_ALREADY_COMMITTED",
                batch_key=key,
                row_count=batch.num_rows,
                shard_relative_path=existing.get("shard_relative_path"),
                shard_sha256=existing.get("shard_sha256"),
                total_committed_rows=int(self._checkpoint["committed_rows"]),
            )
        if self._checkpoint["status"] == "COMPLETED":
            raise FileEmbeddingWriterError("cannot add a new batch to a completed writer")

        if batch.num_rows == 0:
            entry: Dict[str, Any] = {
                "batch_key": key,
                "batch_input_hash": input_hash,
                "vector_sha256": output_hash,
                "row_count": 0,
                "source_batch_index": batch.source_batch_index,
                "source_global_row_offset": batch.source_global_row_offset,
                "shard_relative_path": None,
                "shard_sha256": None,
                "metadata_relative_path": None,
                "metadata_sha256": None,
                "norm_min": None,
                "norm_max": None,
            }
        else:
            norms = np.linalg.norm(array.astype(np.float64), axis=1)
            embedded = {
                "schema_version": self.spec.shard_schema_version,
                "embedding_run_id": self.spec.embedding_run_id,
                "modality": self.spec.modality,
                "compatibility_hash": self.spec.compatibility_hash,
                "model_hash": self.spec.encoder.model_hash,
                "preprocessing_hash": self.spec.preprocessing_hash,
                "batch_key": key,
                "batch_input_hash": input_hash,
                "vector_sha256": output_hash,
                "source_batch_index": batch.source_batch_index,
                "source_global_row_offset": batch.source_global_row_offset,
                "row_count": batch.num_rows,
                "dimension": self.spec.encoder.dimension,
                "dtype": self.spec.encoder.output_dtype,
                "norm_min": float(norms.min()),
                "norm_max": float(norms.max()),
                "status_labels": list(STATUS_LABELS),
            }
            shard_path = self.shard_dir / f"part-{key}.parquet"
            if shard_path.exists():
                raise FileEmbeddingWriterError(
                    "unregistered deterministic shard path already exists"
                )
            table = self._table(batch, array, embedded)
            _atomic_write_parquet(shard_path, table)
            shard_sha = sha256_file(shard_path)
            sidecar_path = self._sidecar_path(shard_path)
            sidecar = dict(embedded)
            sidecar["shard_relative_path"] = self._relative(shard_path)
            sidecar["shard_sha256"] = shard_sha
            _atomic_write_json(sidecar_path, sidecar)
            entry = {
                "batch_key": key,
                "batch_input_hash": input_hash,
                "vector_sha256": output_hash,
                "row_count": batch.num_rows,
                "source_batch_index": batch.source_batch_index,
                "source_global_row_offset": batch.source_global_row_offset,
                "shard_relative_path": self._relative(shard_path),
                "shard_sha256": shard_sha,
                "metadata_relative_path": self._relative(sidecar_path),
                "metadata_sha256": sha256_file(sidecar_path),
                "norm_min": embedded["norm_min"],
                "norm_max": embedded["norm_max"],
            }

        self._checkpoint["batches"][key] = entry
        self._recount()
        self._flush_state()
        return ShardCommit(
            status="COMMITTED",
            batch_key=key,
            row_count=batch.num_rows,
            shard_relative_path=entry.get("shard_relative_path"),
            shard_sha256=entry.get("shard_sha256"),
            total_committed_rows=int(self._checkpoint["committed_rows"]),
        )

    def complete(self, *, expected_rows: Optional[int] = None) -> Dict[str, Any]:
        """Validate all commits and seal this modality checkpoint."""

        self._validate_all_committed()
        committed_rows = int(self._checkpoint["committed_rows"])
        if expected_rows is not None and committed_rows != expected_rows:
            raise FileEmbeddingWriterError(
                "committed embedding count does not match expected_rows"
            )
        self._checkpoint["status"] = "COMPLETED"
        self._flush_state()
        return self._manifest_payload()

    @property
    def committed_rows(self) -> int:
        return int(self._checkpoint["committed_rows"])

    @property
    def status(self) -> str:
        return str(self._checkpoint["status"])


__all__ = [
    "FileEmbeddingRunSpec",
    "FileEmbeddingWriter",
    "FileEmbeddingWriterError",
    "ShardCommit",
    "VectorValidationError",
    "VectorValidationPolicy",
    "WriterCompatibilityError",
]
