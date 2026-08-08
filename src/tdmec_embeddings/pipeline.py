"""File-backed embedding-stage orchestration for transfer to a target Studio.

IMPLEMENTED_NOT_EXECUTED / TRANSFER_PENDING_VALIDATION.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence

from tdmec.hashing import hash_canonical, sha256_file

from .alignment import compute_edge_order_fingerprint_from_root
from .config import EmbeddingRunConfig
from .eligibility import (
    EligibilityBatch,
    EligibleTextUnit,
    EventTextEligibilityProcessor,
    NodeTextEligibilityProcessor,
    load_duplicate_canonical_index,
)
from .file_sources import CanonicalEdgeFileReader, EventTextFileReader, NodeTextFileReader
from .file_writer import (
    FileEmbeddingRunSpec,
    FileEmbeddingWriter,
    VectorValidationPolicy,
    _atomic_write_json,
)
from .implementation_status import IMPLEMENTATION_STATUS_LABELS
from .mock_encoder import DeterministicMockEncoder, TextEncoder
from .model_preflight import ModelPreflightError, verify_preflight_report
from .observability import dependency_versions, git_commit_sha, log_event, resource_snapshot, utc_timestamp
from .pooling import PoolingSpec, StreamingEmbeddingPooler
from .qwen_encoder import Qwen3Encoder
from .run_recovery import RunRecoveryError, prepare_embedding_run_root
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
        vector_validation=VectorValidationPolicy.from_encoder_metadata(encoder.metadata),
    )
    writer = FileEmbeddingWriter(config.resolved_output_root(), spec)
    if writer.status == "COMPLETED":
        if not config.resume:
            raise EmbeddingPipelineError("completed unit outputs exist and resume is disabled")
        return json.loads(writer.manifest_path.read_text(encoding="utf-8"))
    batches = tuple(_chunks(units, config.output_shard_size, modality))
    for ordinal, batch in enumerate(batches, start=1):
        if writer.is_batch_committed(batch):
            log_event(
                "input_shard_skipped_validated",
                modality=modality,
                current_shard=ordinal,
                total_shards=len(batches),
                processed_rows=batch.num_rows,
            )
            continue
        log_event(
            "input_shard_started",
            modality=modality,
            current_shard=ordinal,
            total_shards=len(batches),
            eligible_texts=batch.num_rows,
        )
        vectors = encoder.encode(batch.units)
        commit = writer.write_batch(batch, vectors)
        log_event(
            "checkpoint_write_completed",
            modality=modality,
            batch_key=commit.batch_key,
            processed_rows=commit.row_count,
            checksum=commit.shard_sha256,
            total_committed_rows=commit.total_committed_rows,
        )
    manifest = writer.complete(expected_rows=len(units))
    log_event("modality_embedding_completed", modality=modality, rows=len(units))
    return manifest


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
    preflight_report_path: Optional[str | Path] = None,
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
    if config.execution_mode != "mock":
        if preflight_report_path is None:
            raise EmbeddingPipelineError(
                "a compatible PASSED model-only preflight report is required"
            )
        try:
            verify_preflight_report(preflight_report_path, config)
        except ModelPreflightError as exc:
            raise EmbeddingPipelineError(f"model preflight gate failed: {exc}") from exc
    if config.execution_mode == "qwen_bounded_pilot" and (
        config.max_node_rows > 10_000 or config.max_event_rows > 10_000
    ):
        raise EmbeddingPipelineError(
            "bounded pilot refuses limits above 10,000; a larger run needs separate authorization"
        )

    try:
        recovery = prepare_embedding_run_root(
            output_root=config.resolved_output_root(),
            embedding_run_id=config.embedding_run_id,
            resume=config.resume,
            replace_incomplete=config.replace_incomplete,
        )
    except RunRecoveryError as exc:
        raise EmbeddingPipelineError(str(exc)) from exc

    run_root = config.resolved_output_root() / config.embedding_run_id
    final_manifest_path = run_root / "embedding_manifest.json"
    if final_manifest_path.is_file() and not config.resume:
        # prepare_embedding_run_root should have caught this; keep as belt-and-suspenders
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
    try:
        import torch
    except ImportError:
        torch = None
    execution_provenance = {
        "schema_version": "tdmec-embedding-execution-manifest-v1",
        "created_at": utc_timestamp(),
        "git_commit": git_commit_sha(),
        "configuration_hash": config.scientific_hash(),
        "dependency_versions": dependency_versions(),
        "model_name": config.encoder.model_name,
        "model_revision": config.encoder.model_revision,
        "tokenizer_revision": config.encoder.tokenizer_revision,
        "resources": resource_snapshot(torch_module=torch),
        "input_fingerprints": {
            "node_text": node_source.provenance(),
            "event_text": event_source.provenance(),
        },
        "event_alignment_fingerprint": compute_edge_order_fingerprint_from_root(
            config.event_source.resolved_root(),
            expected_run_id=config.event_source.run_id,
            batch_size=config.input_batch_size,
            verify_checksums=False,
        ),
    }
    _atomic_write_json(run_root / "execution_manifest.json", execution_provenance)
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
    completed_modalities = {}
    for name in ("node_text", "event_text"):
        checkpoint_path = run_root / "checkpoints" / f"{name}.json"
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        completed_modalities[name] = {
            "status": checkpoint.get("status"),
            "committed_rows": int(checkpoint.get("committed_rows", 0)),
            "batches": checkpoint.get("batches") or {},
        }
    _atomic_write_json(
        run_root / "completed_shards.json",
        {
            "schema_version": "tdmec-completed-shards-v1",
            "updated_at": utc_timestamp(),
            "modalities": completed_modalities,
            "checksum_policy": "sha256_parquet_and_metadata_sidecar_verified_on_resume",
            "atomic_publish": "temporary_file_fsync_then_os_replace",
        },
    )

    log_event("pooling_started", modality="node_text")
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
    log_event("pooling_completed", modality="node_text")
    edge_reader = CanonicalEdgeFileReader(
        config.event_source.resolved_root(),
        expected_run_id=config.event_source.run_id,
        batch_size=config.input_batch_size,
        identity=event_source,
    )
    log_event("pooling_started", modality="event_text")
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
    log_event("pooling_completed", modality="event_text")

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
        "run_recovery": recovery,
        "provenance": execution_provenance,
        "normalization": {
            "unit_normalized": True,
            "normalized_atol": float(config.encoder.normalized_atol),
            "float32_renorm_after_cast": config.encoder.backend == "qwen3",
            "stage_b_final_normalization": config.pooling.final_normalization,
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
    log_event(
        "embedding_final_validation_completed",
        status="COMPLETED",
        checksum_file_count=len(checksums),
    )
    return final_manifest


__all__ = ["EmbeddingPipelineError", "SamplingError", "run_embedding_pipeline"]
