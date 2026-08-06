"""File-backed embedding-stage orchestration for transfer to a target Studio.

IMPLEMENTED_NOT_EXECUTED / TRANSFER_PENDING_VALIDATION.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence

from tdmec.hashing import hash_canonical, sha256_file

from .config import EmbeddingRunConfig
from .eligibility import (
    EligibilityBatch,
    EligibleTextUnit,
    EventTextEligibilityProcessor,
    NodeTextEligibilityProcessor,
    load_duplicate_canonical_index,
)
from .file_sources import CanonicalEdgeFileReader, EventTextFileReader, NodeTextFileReader
from .file_writer import FileEmbeddingRunSpec, FileEmbeddingWriter, _atomic_write_json
from .implementation_status import IMPLEMENTATION_STATUS_LABELS
from .mock_encoder import DeterministicMockEncoder, TextEncoder
from .pooling import PoolingSpec, StreamingEmbeddingPooler
from .qwen_encoder import Qwen3Encoder
from .sampling import (
    DeterministicStratifiedSampler,
    SamplingError,
    SamplingResult,
    activity_counts,
    build_combined_sampling_report,
)


class EmbeddingPipelineError(RuntimeError):
    pass


def _node_processor(config: EmbeddingRunConfig, *, full_scan: bool):
    reader = NodeTextFileReader(
        config.node_source.resolved_root(),
        expected_run_id=config.node_source.run_id,
        batch_size=config.input_batch_size,
        max_rows=None if full_scan else config.max_node_rows,
    )
    duplicate_index = load_duplicate_canonical_index(reader.identity)
    return NodeTextEligibilityProcessor(reader, duplicate_index)


def _event_processor(config: EmbeddingRunConfig, *, full_scan: bool):
    reader = EventTextFileReader(
        config.event_source.resolved_root(),
        expected_run_id=config.event_source.run_id,
        batch_size=config.input_batch_size,
        max_rows=None if full_scan else config.max_event_rows,
    )
    return EventTextEligibilityProcessor(reader)


def _eligible_units(processor: Any) -> Iterator[EligibleTextUnit]:
    for batch in processor.iter_batches():
        yield from batch.units


def _stratified_selection(
    config: EmbeddingRunConfig,
    *,
    modality: str,
) -> tuple[tuple[EligibleTextUnit, ...], Dict[str, Any], str]:
    factory = _node_processor if modality == "node_text" else _event_processor
    first = factory(config, full_scan=True)
    activity = activity_counts(_eligible_units(first), modality)  # type: ignore[arg-type]
    first_report = first.report.to_dict()
    second = factory(config, full_scan=True)
    limit = config.max_node_rows if modality == "node_text" else config.max_event_rows
    sampler = DeterministicStratifiedSampler(
        modality=modality,  # type: ignore[arg-type]
        limit=limit,
        config=config.sampling,
        activity=activity,
    )
    sampler.extend(_eligible_units(second))
    result: SamplingResult = sampler.finish()
    second_report = second.report.to_dict()
    if first_report["eligible_rows"] != second_report["eligible_rows"]:
        raise EmbeddingPipelineError("two-pass eligibility accounting is not deterministic")
    return result.selected_units, result.report(), second.preprocessing_hash


def _prefix_units(
    config: EmbeddingRunConfig, *, modality: str
) -> tuple[tuple[EligibleTextUnit, ...], Dict[str, Any], str]:
    processor = (
        _node_processor(config, full_scan=False)
        if modality == "node_text"
        else _event_processor(config, full_scan=False)
    )
    units = tuple(_eligible_units(processor))
    report = processor.report.to_dict()
    return units, {
        "policy": "deterministic_source_prefix_v1",
        "eligible_population_within_limit": len(units),
        "source_eligibility": report,
        "status_labels": list(IMPLEMENTATION_STATUS_LABELS),
    }, processor.preprocessing_hash


def _chunks(units: Sequence[EligibleTextUnit], size: int, modality: str) -> Iterator[EligibilityBatch]:
    for index, start in enumerate(range(0, len(units), size)):
        yield EligibilityBatch(
            modality=modality,  # type: ignore[arg-type]
            source_batch_index=index,
            source_global_row_offset=start,
            units=tuple(units[start : start + size]),
        )
    if not units:
        yield EligibilityBatch(
            modality=modality,  # type: ignore[arg-type]
            source_batch_index=0,
            source_global_row_offset=0,
            units=(),
        )


def _write_modality(
    config: EmbeddingRunConfig,
    *,
    modality: str,
    units: Sequence[EligibleTextUnit],
    preprocessing_hash: str,
    encoder: TextEncoder,
    source: Any,
) -> Dict[str, Any]:
    spec = FileEmbeddingRunSpec(
        embedding_run_id=config.embedding_run_id,
        modality=modality,  # type: ignore[arg-type]
        source=source,
        preprocessing_hash=preprocessing_hash,
        encoder=encoder.metadata,
    )
    writer = FileEmbeddingWriter(config.resolved_output_root(), spec)
    if writer.status == "COMPLETED":
        if not config.resume:
            raise EmbeddingPipelineError("completed unit outputs exist and resume is disabled")
        return json.loads(writer.manifest_path.read_text(encoding="utf-8"))
    for batch in _chunks(units, config.output_shard_size, modality):
        if writer.is_batch_committed(batch):
            continue
        vectors = encoder.encode(batch.units)
        writer.write_batch(batch, vectors)
    return writer.complete(expected_rows=len(units))


def _source_preflight(config: EmbeddingRunConfig) -> Dict[str, Any]:
    node = _node_processor(config, full_scan=False)
    event = _event_processor(config, full_scan=False)
    edges = CanonicalEdgeFileReader(
        config.event_source.resolved_root(),
        expected_run_id=config.event_source.run_id,
        batch_size=config.input_batch_size,
        max_rows=1,
        identity=event.reader.identity,
    )
    return {
        "dry_run": True,
        "configuration_hash": config.scientific_hash(),
        "node_source": node.reader.identity.provenance(),
        "event_source": event.reader.identity.provenance(),
        "node_source_rows": node.reader.total_rows,
        "event_source_rows": event.reader.total_rows,
        "canonical_edge_rows": edges.total_rows,
        "status_labels": list(IMPLEMENTATION_STATUS_LABELS),
    }


def _scale_estimates(
    config: EmbeddingRunConfig,
    *,
    node_population: int,
    event_population: int,
    edge_count: int,
    n_snapshots: int,
    n_nodes: int,
    runtime_report: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    dim = config.encoder.output_dimension
    vector_bytes = dim * 4
    total_units = node_population + event_population
    units_per_second = (runtime_report or {}).get("units_per_second")
    return {
        "eligible_node_population": node_population,
        "eligible_event_population": event_population,
        "expected_unit_embeddings": total_units,
        "estimated_unit_vector_bytes_float32": total_units * vector_bytes,
        "estimated_dense_node_tensor_bytes_float32": n_snapshots * n_nodes * vector_bytes,
        "estimated_dense_edge_tensor_bytes_float32": edge_count * vector_bytes,
        "estimated_total_inference_seconds": (
            total_units / units_per_second if units_per_second else None
        ),
        "excludes_parquet_identity_and_compression_overhead": True,
        "status_labels": list(IMPLEMENTATION_STATUS_LABELS),
    }


def run_embedding_pipeline(
    config: EmbeddingRunConfig,
    *,
    authorize_real_model: bool = False,
    authorize_bounded_pilot: bool = False,
) -> Dict[str, Any]:
    """Execute only when explicitly invoked in the target Studio."""

    config.validate()
    if config.force:
        raise EmbeddingPipelineError(
            "force overwrite is disabled; refuse to destroy completed embedding outputs"
        )
    if config.dry_run:
        return _source_preflight(config)
    if config.execution_mode == "qwen_preflight" and not authorize_real_model:
        raise EmbeddingPipelineError("real Qwen preflight requires explicit authorization")
    if config.execution_mode == "qwen_bounded_pilot" and not (
        authorize_real_model and authorize_bounded_pilot
    ):
        raise EmbeddingPipelineError("bounded Qwen pilot requires both authorization flags")
    if config.execution_mode == "qwen_bounded_pilot" and (
        config.max_node_rows > 10_000 or config.max_event_rows > 10_000
    ):
        raise EmbeddingPipelineError(
            "bounded pilot refuses limits above 10,000; a larger run needs separate authorization"
        )

    run_root = config.resolved_output_root() / config.embedding_run_id
    final_manifest_path = run_root / "embedding_manifest.json"
    if final_manifest_path.is_file() and not config.resume:
        raise EmbeddingPipelineError("completed embedding output exists; overwrite is prohibited")

    if config.execution_mode == "mock":
        encoder: TextEncoder = DeterministicMockEncoder(
            dimension=config.encoder.output_dimension,
            instruction_hash=hash_canonical({"instruction": config.encoder.instruction}),
        )
        node_units, node_sampling, node_preprocessing = _prefix_units(config, modality="node_text")
        event_units, event_sampling, event_preprocessing = _prefix_units(config, modality="event_text")
    else:
        encoder = Qwen3Encoder(config.encoder)
        node_units, node_sampling, node_preprocessing = _stratified_selection(
            config, modality="node_text"
        )
        event_units, event_sampling, event_preprocessing = _stratified_selection(
            config, modality="event_text"
        )

    node_source = _node_processor(config, full_scan=False).reader.identity
    event_source = _event_processor(config, full_scan=False).reader.identity
    node_manifest = _write_modality(
        config,
        modality="node_text",
        units=node_units,
        preprocessing_hash=node_preprocessing,
        encoder=encoder,
        source=node_source,
    )
    event_manifest = _write_modality(
        config,
        modality="event_text",
        units=event_units,
        preprocessing_hash=event_preprocessing,
        encoder=encoder,
        source=event_source,
    )

    node_pool = StreamingEmbeddingPooler(
        run_root,
        PoolingSpec(
            embedding_run_id=config.embedding_run_id,
            modality="node_text",
            dimension=config.encoder.output_dimension,
            n_snapshots=node_source.n_snapshots,
            n_nodes=node_source.n_nodes,
            final_normalization=config.pooling.final_normalization,
            resume=config.resume,
        ),
    )
    node_pool.prepare_deltas()
    node_pool_manifest = node_pool.finalize_node_snapshots()
    edge_reader = CanonicalEdgeFileReader(
        config.event_source.resolved_root(),
        expected_run_id=config.event_source.run_id,
        batch_size=config.input_batch_size,
        identity=event_source,
    )
    event_pool = StreamingEmbeddingPooler(
        run_root,
        PoolingSpec(
            embedding_run_id=config.embedding_run_id,
            modality="event_text",
            dimension=config.encoder.output_dimension,
            n_snapshots=event_source.n_snapshots,
            n_nodes=event_source.n_nodes,
            final_normalization=config.pooling.final_normalization,
            resume=config.resume,
        ),
    )
    event_pool.prepare_deltas()
    event_pool_manifest = event_pool.finalize_canonical_edges(edge_reader)

    runtime = encoder.runtime_report() if isinstance(encoder, Qwen3Encoder) else {
        "backend": "deterministic_mock",
        "status_labels": list(IMPLEMENTATION_STATUS_LABELS),
    }
    reports_dir = run_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(reports_dir / "runtime_memory.json", runtime)
    _atomic_write_json(reports_dir / "node_sampling.json", node_sampling)
    _atomic_write_json(reports_dir / "event_sampling.json", event_sampling)
    sampling_report = build_combined_sampling_report(
        node=node_sampling,
        event=event_sampling,
        seed=config.sampling.seed,
    )
    _atomic_write_json(reports_dir / "sampling_report.json", sampling_report)
    if (
        config.execution_mode != "mock"
        and config.validation.require_relation_coverage
        and event_sampling.get("force_relation_coverage")
    ):
        missing = list(
            (event_sampling.get("relation_coverage") or {}).get("missing_from_sample") or []
        )
        if missing:
            raise EmbeddingPipelineError(
                f"event sample missing required relations present in population: {missing}"
            )
    estimates = _scale_estimates(
        config,
        node_population=int(node_sampling.get("eligible_population", len(node_units))),
        event_population=int(event_sampling.get("eligible_population", len(event_units))),
        edge_count=edge_reader.total_rows,
        n_snapshots=node_source.n_snapshots,
        n_nodes=node_source.n_nodes,
        runtime_report=runtime,
    )
    _atomic_write_json(reports_dir / "scale_estimates.json", estimates)

    final_manifest = {
        "schema_version": "tdmec-embedding-final-manifest-v1",
        "embedding_run_id": config.embedding_run_id,
        "status": "COMPLETED",
        "configuration_hash": config.scientific_hash(),
        "source_runs": {
            "node_source_run_id": config.node_source.run_id,
            "event_source_run_id": config.event_source.run_id,
        },
        "unit_outputs": {"node": node_manifest, "event": event_manifest},
        "pooled_outputs": {"node": node_pool_manifest, "event": event_pool_manifest},
        "reports": {
            "runtime_memory": "reports/runtime_memory.json",
            "node_sampling": "reports/node_sampling.json",
            "event_sampling": "reports/event_sampling.json",
            "sampling_report": "reports/sampling_report.json",
            "scale_estimates": "reports/scale_estimates.json",
        },
        "status_labels": list(IMPLEMENTATION_STATUS_LABELS),
        "automatic_scale_up": False,
    }
    _atomic_write_json(final_manifest_path, final_manifest)
    checksums = {
        path.relative_to(run_root).as_posix(): sha256_file(path)
        for path in sorted(run_root.rglob("*"))
        if path.is_file() and path.name != "all_checksums.json"
    }
    _atomic_write_json(run_root / "all_checksums.json", checksums)
    return final_manifest


__all__ = ["EmbeddingPipelineError", "SamplingError", "run_embedding_pipeline"]
