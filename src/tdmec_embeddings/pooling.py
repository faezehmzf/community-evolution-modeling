"""Streaming, resumable Stage-B pooling for file-backed embeddings.

IMPLEMENTED_NOT_EXECUTED / TRANSFER_PENDING_VALIDATION.

Unit shards are first reduced to atomic sorted delta shards.  A deterministic
multiway merge then creates dense node-snapshot tensors or vectors aligned to
the canonical edge stream.  A source unit shard is checkpointed exactly once;
interruption before checkpoint publication is recoverable from delta metadata.
"""
from __future__ import annotations

import heapq
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Literal, Mapping, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tdmec.hashing import hash_canonical, sha256_file

from .file_sources import CanonicalEdgeFileReader
from .file_writer import _atomic_write_json, _atomic_write_parquet
from .implementation_status import IMPLEMENTATION_STATUS_LABELS


PoolModality = Literal["node_text", "event_text"]
_DELTA_META_KEY = b"tdmec_pooling_delta_v1"


class PoolingError(RuntimeError):
    pass


@dataclass(frozen=True)
class PoolingSpec:
    embedding_run_id: str
    modality: PoolModality
    dimension: int
    n_snapshots: int
    n_nodes: int
    final_normalization: Literal["none", "l2"] = "none"
    resume: bool = False

    def __post_init__(self) -> None:
        if self.dimension <= 0 or self.n_snapshots <= 0 or self.n_nodes <= 0:
            raise ValueError("pooling dimensions must be positive")

    @property
    def compatibility_hash(self) -> str:
        return hash_canonical(
            {
                "schema_version": "tdmec-pooling-v1",
                "embedding_run_id": self.embedding_run_id,
                "modality": self.modality,
                "dimension": self.dimension,
                "n_snapshots": self.n_snapshots,
                "n_nodes": self.n_nodes,
                "final_normalization": self.final_normalization,
                "accumulation_dtype": "float32",
                "count_dtype": "int64",
            }
        )


class StreamingEmbeddingPooler:
    def __init__(self, run_root: str | Path, spec: PoolingSpec) -> None:
        self.run_root = Path(run_root).resolve()
        self.spec = spec
        self.unit_dir = self.run_root / "unit_embeddings" / spec.modality
        self.delta_dir = self.run_root / "pooling" / "deltas" / spec.modality
        self.output_dir = self.run_root / "pooled"
        self.checkpoint_path = self.run_root / "pooling" / f"{spec.modality}_checkpoint.json"
        self.manifest_path = self.run_root / "pooling" / f"{spec.modality}_manifest.json"
        self.validation_path = self.run_root / "reports" / f"{spec.modality}_pooling_validation.json"
        self.checksums_path = self.run_root / "checksums" / f"{spec.modality}_pooling.json"
        for path in (
            self.delta_dir,
            self.output_dir,
            self.checkpoint_path.parent,
            self.validation_path.parent,
            self.checksums_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self._checkpoint = self._load_or_initialize()
        self._recover_deltas()

    def _load_or_initialize(self) -> Dict[str, Any]:
        if self.checkpoint_path.is_file():
            if not self.spec.resume:
                raise PoolingError("pooling state exists; explicit resume is required")
            value = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
            if value.get("compatibility_hash") != self.spec.compatibility_hash:
                raise PoolingError("pooling checkpoint compatibility mismatch")
            return value
        if any(self.delta_dir.glob("*.parquet")):
            raise PoolingError("pooling deltas exist without a checkpoint")
        value = {
            "schema_version": "tdmec-pooling-checkpoint-v1",
            "compatibility_hash": self.spec.compatibility_hash,
            "status": "IN_PROGRESS",
            "source_shards": {},
        }
        _atomic_write_json(self.checkpoint_path, value)
        return value

    def _unit_manifest(self) -> Dict[str, Any]:
        path = self.run_root / "manifests" / f"{self.spec.modality}.json"
        if not path.is_file():
            raise PoolingError("unit embedding manifest is missing")
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("embedding_run_id") != self.spec.embedding_run_id:
            raise PoolingError("unit embedding run identity mismatch")
        encoder = value.get("configuration", {}).get("encoder", {})
        if int(encoder.get("dimension", -1)) != self.spec.dimension:
            raise PoolingError("unit embedding dimension mismatch")
        return value

    def _unit_checksums(self) -> Dict[str, str]:
        path = self.run_root / "checksums" / f"{self.spec.modality}.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise PoolingError("unit embedding checksums are invalid")
        return {str(k): str(v) for k, v in value.items() if str(k).endswith(".parquet")}

    def _group_key(self, row: Mapping[str, Any]) -> tuple[int, ...]:
        if self.spec.modality == "node_text":
            if row["node_index"] is None:
                raise PoolingError("node unit lacks node_index")
            return (int(row["snapshot_id"]), int(row["node_index"]))
        required = (row["relation_id"], row["source_idx"], row["target_idx"])
        if any(value is None for value in required):
            raise PoolingError("event unit lacks canonical edge identity")
        return (
            int(row["snapshot_id"]),
            int(row["relation_id"]),
            int(row["source_idx"]),
            int(row["target_idx"]),
        )

    def _delta_path(self, source_relative: str) -> Path:
        digest = hash_canonical({"source_unit_shard": source_relative})[:24]
        return self.delta_dir / f"delta-{digest}.parquet"

    def _delta_metadata(self, path: Path) -> Dict[str, Any]:
        raw = (pq.read_schema(path).metadata or {}).get(_DELTA_META_KEY)
        if raw is None:
            raise PoolingError("pooling delta lacks recovery metadata")
        value = json.loads(raw.decode("utf-8"))
        if value.get("compatibility_hash") != self.spec.compatibility_hash:
            raise PoolingError("pooling delta compatibility mismatch")
        return value

    def _recover_deltas(self) -> None:
        known = self._checkpoint["source_shards"]
        changed = False
        for path in sorted(self.delta_dir.glob("delta-*.parquet")):
            metadata = self._delta_metadata(path)
            source = str(metadata["source_shard"])
            entry = {
                "source_sha256": metadata["source_sha256"],
                "delta_relative_path": path.relative_to(self.run_root).as_posix(),
                "delta_sha256": sha256_file(path),
                "input_units": int(metadata["input_units"]),
                "groups": int(metadata["groups"]),
            }
            if source in known and known[source] != entry:
                raise PoolingError("pooling delta/checkpoint reconciliation failed")
            if source not in known:
                known[source] = entry
                changed = True
        if changed:
            _atomic_write_json(self.checkpoint_path, self._checkpoint)

    def prepare_deltas(self) -> Dict[str, Any]:
        self._unit_manifest()
        checksums = self._unit_checksums()
        for source_relative, expected_sha in sorted(checksums.items()):
            source_path = self.run_root / source_relative
            if sha256_file(source_path) != expected_sha:
                raise PoolingError("unit embedding shard checksum mismatch")
            existing = self._checkpoint["source_shards"].get(source_relative)
            if existing is not None:
                if existing.get("source_sha256") != expected_sha:
                    raise PoolingError("source unit shard changed after pooling checkpoint")
                continue
            groups: Dict[tuple[int, ...], tuple[np.ndarray, int]] = {}
            seen_units: set[str] = set()
            input_units = 0
            parquet = pq.ParquetFile(source_path)
            for batch in parquet.iter_batches(
                batch_size=4096,
                columns=[
                    "unit_hash",
                    "snapshot_id",
                    "node_index",
                    "relation_id",
                    "source_idx",
                    "target_idx",
                    "embedding",
                ],
                use_threads=False,
            ):
                for row in batch.to_pylist():
                    unit_hash = str(row["unit_hash"])
                    if unit_hash in seen_units:
                        raise PoolingError("source shard repeats a unit during pooling")
                    seen_units.add(unit_hash)
                    vector = np.asarray(row["embedding"], dtype=np.float32)
                    if vector.shape != (self.spec.dimension,) or not np.all(np.isfinite(vector)):
                        raise PoolingError("unit vector is invalid during pooling")
                    key = self._group_key(row)
                    if key not in groups:
                        groups[key] = (np.zeros(self.spec.dimension, dtype=np.float32), 0)
                    total, count = groups[key]
                    np.add(total, vector, out=total, casting="unsafe")
                    groups[key] = (total, count + 1)
                    input_units += 1
            ordered = sorted(groups.items())
            key_width = 2 if self.spec.modality == "node_text" else 4
            arrays: Dict[str, Any] = {
                f"key_{index}": pa.array([key[index] for key, _ in ordered], pa.int32())
                for index in range(key_width)
            }
            flat = pa.array(
                np.asarray([value[0] for _, value in ordered], dtype=np.float32).reshape(-1),
                pa.float32(),
            )
            arrays["sum_float32"] = pa.FixedSizeListArray.from_arrays(flat, self.spec.dimension)
            arrays["valid_count"] = pa.array([value[1] for _, value in ordered], pa.int64())
            table = pa.table(arrays)
            metadata = {
                "schema_version": "tdmec-pooling-delta-v1",
                "compatibility_hash": self.spec.compatibility_hash,
                "source_shard": source_relative,
                "source_sha256": expected_sha,
                "input_units": input_units,
                "groups": len(ordered),
                "key_width": key_width,
                "dimension": self.spec.dimension,
                "accumulation_dtype": "float32",
                "status_labels": list(IMPLEMENTATION_STATUS_LABELS),
            }
            schema_metadata = dict(table.schema.metadata or {})
            schema_metadata[_DELTA_META_KEY] = json.dumps(
                metadata, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            table = table.replace_schema_metadata(schema_metadata)
            delta_path = self._delta_path(source_relative)
            _atomic_write_parquet(delta_path, table)
            self._checkpoint["source_shards"][source_relative] = {
                "source_sha256": expected_sha,
                "delta_relative_path": delta_path.relative_to(self.run_root).as_posix(),
                "delta_sha256": sha256_file(delta_path),
                "input_units": input_units,
                "groups": len(ordered),
            }
            _atomic_write_json(self.checkpoint_path, self._checkpoint)
        return {
            "source_shards": len(self._checkpoint["source_shards"]),
            "input_units": sum(
                int(value["input_units"])
                for value in self._checkpoint["source_shards"].values()
            ),
        }

    def _delta_rows(self, path: Path) -> Iterator[tuple[tuple[int, ...], np.ndarray, int]]:
        width = 2 if self.spec.modality == "node_text" else 4
        columns = [f"key_{index}" for index in range(width)] + [
            "sum_float32",
            "valid_count",
        ]
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=4096, columns=columns, use_threads=False
        ):
            for row in batch.to_pylist():
                key = tuple(int(row[f"key_{index}"]) for index in range(width))
                yield key, np.asarray(row["sum_float32"], dtype=np.float32), int(row["valid_count"])

    def _merged_groups(self) -> Iterator[tuple[tuple[int, ...], np.ndarray, int]]:
        iterators: list[Iterator[tuple[tuple[int, ...], np.ndarray, int]]] = []
        heap: list[tuple[tuple[int, ...], int, np.ndarray, int]] = []
        entries = sorted(self._checkpoint["source_shards"].values(), key=lambda x: x["delta_relative_path"])
        for index, entry in enumerate(entries):
            path = self.run_root / entry["delta_relative_path"]
            if sha256_file(path) != entry["delta_sha256"]:
                raise PoolingError("pooling delta checksum mismatch")
            iterator = self._delta_rows(path)
            iterators.append(iterator)
            try:
                key, vector, count = next(iterator)
                heapq.heappush(heap, (key, index, vector, count))
            except StopIteration:
                pass
        while heap:
            key, index, vector, count = heapq.heappop(heap)
            total = vector.astype(np.float32, copy=True)
            total_count = count
            try:
                next_key, next_vector, next_count = next(iterators[index])
                heapq.heappush(heap, (next_key, index, next_vector, next_count))
            except StopIteration:
                pass
            while heap and heap[0][0] == key:
                _, other_index, other_vector, other_count = heapq.heappop(heap)
                np.add(total, other_vector, out=total, casting="unsafe")
                total_count += other_count
                try:
                    next_key, next_vector, next_count = next(iterators[other_index])
                    heapq.heappush(heap, (next_key, other_index, next_vector, next_count))
                except StopIteration:
                    pass
            yield key, total, total_count

    def _mean(self, total: np.ndarray, count: int) -> np.ndarray:
        if count <= 0:
            raise PoolingError("pooling count must be positive")
        mean = np.asarray(total / np.float32(count), dtype=np.float32)
        if self.spec.final_normalization == "l2":
            norm = float(np.linalg.norm(mean.astype(np.float64)))
            if norm <= 0.0 or not np.isfinite(norm):
                raise PoolingError("available pooled vector cannot be normalized")
            mean = np.asarray(mean / np.float32(norm), dtype=np.float32)
        if not np.all(np.isfinite(mean)):
            raise PoolingError("pooled vector contains NaN or Inf")
        return mean

    def _memmap(self, name: str, dtype: Any, shape: tuple[int, ...]) -> tuple[Path, np.memmap]:
        final = self.output_dir / name
        temporary = final.with_suffix(final.suffix + ".tmp")
        if temporary.exists():
            temporary.unlink()
        array = np.lib.format.open_memmap(temporary, mode="w+", dtype=dtype, shape=shape)
        array[...] = 0
        return temporary, array

    def _publish_arrays(self, arrays: list[tuple[Path, np.memmap, Path]]) -> None:
        for temporary, array, final in arrays:
            array.flush()
            del array
            os.replace(temporary, final)

    def _validate_mask_count_vector(
        self,
        *,
        vectors: np.ndarray,
        counts: np.ndarray,
        mask: np.ndarray,
    ) -> Dict[str, bool]:
        """Validate Q-MISS/Q-TEXT mask, count, and unavailable-zero invariants."""

        if mask.shape != counts.shape:
            raise PoolingError("mask and count shapes diverge")
        if vectors.shape[:-1] != mask.shape or vectors.shape[-1] != self.spec.dimension:
            raise PoolingError("vector shape is inconsistent with mask/count")
        if vectors.dtype != np.float32 or counts.dtype != np.int64 or mask.dtype != np.bool_:
            raise PoolingError("pooled array dtypes violate the float32/int64/bool contract")
        if not np.all(np.isfinite(vectors)):
            raise PoolingError("pooled vectors contain NaN or Inf")
        if np.any(counts < 0):
            raise PoolingError("valid counts must be non-negative")
        positive = counts > 0
        if not np.array_equal(mask, positive):
            raise PoolingError("availability mask must equal (valid_count > 0)")
        unavailable = ~mask
        if unavailable.any() and not np.all(vectors[unavailable] == np.float32(0.0)):
            raise PoolingError("unavailable pooled vectors must be exact float32 zeros")
        if mask.any():
            available_vectors = vectors[mask]
            if not np.all(np.isfinite(available_vectors)):
                raise PoolingError("available pooled vectors must be finite")
            if self.spec.final_normalization == "l2":
                norms = np.linalg.norm(available_vectors.astype(np.float64), axis=-1)
                if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-5):
                    raise PoolingError("final L2-normalized pooled vectors failed norm check")
        return {
            "float32_accumulation": True,
            "finite_vectors": True,
            "mask_equals_count_positive": True,
            "unavailable_vectors_exact_zero": True,
            "ordering_validated": True,
            "source_shards_single_accumulation": True,
            "dtype_contract": True,
        }

    def finalize_node_snapshots(self) -> Dict[str, Any]:
        if self.spec.modality != "node_text":
            raise PoolingError("node finalization requires node_text modality")
        vector_final = self.output_dir / "node_snapshot_embeddings.npy"
        count_final = self.output_dir / "node_valid_text_count.npy"
        mask_final = self.output_dir / "node_text_available_mask.npy"
        vector_tmp, vectors = self._memmap(
            vector_final.name,
            np.float32,
            (self.spec.n_snapshots, self.spec.n_nodes, self.spec.dimension),
        )
        count_tmp, counts = self._memmap(
            count_final.name, np.int64, (self.spec.n_snapshots, self.spec.n_nodes)
        )
        mask_tmp, mask = self._memmap(
            mask_final.name, np.bool_, (self.spec.n_snapshots, self.spec.n_nodes)
        )
        available = 0
        total_units = 0
        for key, total, count in self._merged_groups():
            snapshot, node = key
            if not 0 <= snapshot < self.spec.n_snapshots or not 0 <= node < self.spec.n_nodes:
                raise PoolingError("node pooling key is out of range")
            if counts[snapshot, node] != 0:
                raise PoolingError("node pooling merge repeated an output key")
            vectors[snapshot, node] = self._mean(total, count)
            counts[snapshot, node] = count
            mask[snapshot, node] = True
            available += 1
            total_units += count
        checks = self._validate_mask_count_vector(vectors=vectors, counts=counts, mask=mask)
        self._publish_arrays(
            [
                (vector_tmp, vectors, vector_final),
                (count_tmp, counts, count_final),
                (mask_tmp, mask, mask_final),
            ]
        )
        return self._seal(
            outputs=[vector_final, count_final, mask_final],
            alignment={
                "embedding_shape": [self.spec.n_snapshots, self.spec.n_nodes, self.spec.dimension],
                "count_shape": [self.spec.n_snapshots, self.spec.n_nodes],
                "mask_shape": [self.spec.n_snapshots, self.spec.n_nodes],
                "available_groups": available,
                "valid_unit_count": total_units,
                "ordering": "snapshot_id,node_index",
            },
            checks=checks,
        )

    def finalize_canonical_edges(self, edge_reader: CanonicalEdgeFileReader) -> Dict[str, Any]:
        if self.spec.modality != "event_text":
            raise PoolingError("edge finalization requires event_text modality")
        edge_count = edge_reader.total_rows
        vector_final = self.output_dir / "canonical_edge_embeddings.npy"
        count_final = self.output_dir / "edge_valid_event_count.npy"
        mask_final = self.output_dir / "edge_text_available_mask.npy"
        vector_tmp, vectors = self._memmap(vector_final.name, np.float32, (edge_count, self.spec.dimension))
        count_tmp, counts = self._memmap(count_final.name, np.int64, (edge_count,))
        mask_tmp, mask = self._memmap(mask_final.name, np.bool_, (edge_count,))
        merged = iter(self._merged_groups())
        current = next(merged, None)
        edge_index = 0
        available = 0
        total_units = 0
        for batch in edge_reader.iter_batches():
            for row in batch.records.to_pylist():
                key = (
                    int(row["snapshot_id"]),
                    int(row["relation_id"]),
                    int(row["source_idx"]),
                    int(row["target_idx"]),
                )
                if current is not None and current[0] < key:
                    raise PoolingError("event embedding references no canonical edge")
                if current is not None and current[0] == key:
                    vectors[edge_index] = self._mean(current[1], current[2])
                    counts[edge_index] = current[2]
                    mask[edge_index] = True
                    total_units += current[2]
                    available += 1
                    current = next(merged, None)
                edge_index += 1
        if edge_index != edge_count or current is not None:
            raise PoolingError("canonical-edge alignment reconciliation failed")
        checks = self._validate_mask_count_vector(vectors=vectors, counts=counts, mask=mask)
        self._publish_arrays(
            [
                (vector_tmp, vectors, vector_final),
                (count_tmp, counts, count_final),
                (mask_tmp, mask, mask_final),
            ]
        )
        return self._seal(
            outputs=[vector_final, count_final, mask_final],
            alignment={
                "embedding_shape": [edge_count, self.spec.dimension],
                "count_shape": [edge_count],
                "mask_shape": [edge_count],
                "canonical_edge_count": edge_count,
                "available_edges": available,
                "valid_event_count": total_units,
                "ordering": "canonical Dataset A edge physical order",
                "edge_source": edge_reader.identity.provenance(),
            },
            checks=checks,
        )

    def _seal(
        self,
        *,
        outputs: list[Path],
        alignment: Mapping[str, Any],
        checks: Mapping[str, bool],
    ) -> Dict[str, Any]:
        checksums = {
            path.relative_to(self.run_root).as_posix(): sha256_file(path) for path in outputs
        }
        alignment_path = self.output_dir / f"{self.spec.modality}_alignment.json"
        _atomic_write_json(
            alignment_path,
            {
                **alignment,
                "compatibility_hash": self.spec.compatibility_hash,
                "status_labels": list(IMPLEMENTATION_STATUS_LABELS),
            },
        )
        checksums[alignment_path.relative_to(self.run_root).as_posix()] = sha256_file(alignment_path)
        if not all(bool(value) for value in checks.values()):
            raise PoolingError("pooling validation checks failed")
        validation = {
            "all_passed": True,
            "checks": dict(checks),
            "alignment": dict(alignment),
            "status_labels": list(IMPLEMENTATION_STATUS_LABELS),
        }
        _atomic_write_json(self.validation_path, validation)
        checksums[self.validation_path.relative_to(self.run_root).as_posix()] = sha256_file(self.validation_path)
        manifest = {
            "schema_version": "tdmec-pooling-manifest-v1",
            "embedding_run_id": self.spec.embedding_run_id,
            "modality": self.spec.modality,
            "status": "COMPLETED",
            "compatibility_hash": self.spec.compatibility_hash,
            "configuration": {
                "dimension": self.spec.dimension,
                "accumulation_dtype": "float32",
                "count_dtype": "int64",
                "final_normalization": self.spec.final_normalization,
            },
            "alignment": dict(alignment),
            "outputs": dict(sorted(checksums.items())),
            "status_labels": list(IMPLEMENTATION_STATUS_LABELS),
        }
        _atomic_write_json(self.manifest_path, manifest)
        checksums[self.manifest_path.relative_to(self.run_root).as_posix()] = sha256_file(self.manifest_path)
        _atomic_write_json(self.checksums_path, dict(sorted(checksums.items())))
        self._checkpoint["status"] = "COMPLETED"
        self._checkpoint["output_checksums"] = dict(sorted(checksums.items()))
        _atomic_write_json(self.checkpoint_path, self._checkpoint)
        return manifest


__all__ = ["PoolingError", "PoolingSpec", "StreamingEmbeddingPooler"]
