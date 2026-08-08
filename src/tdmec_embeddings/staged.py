"""Deterministic, resumable unit-embedding shard execution for bounded jobs."""
from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Literal, Sequence

from .alignment import compute_edge_order_fingerprint_from_root
from .config import EmbeddingRunConfig
from .eligibility import EligibilityBatch, EligibleTextUnit
from .file_writer import (
    FileEmbeddingRunSpec,
    FileEmbeddingWriter,
    VectorValidationPolicy,
    _atomic_write_json,
)
from .model_preflight import verify_preflight_report
from .observability import (
    dependency_versions,
    git_commit_sha,
    log_event,
    resource_snapshot,
    utc_timestamp,
)
from .pipeline import (
    _chunks,
    _event_processor,
    _node_processor,
    _prefix_units,
    _stratified_selection,
)
from .qwen_encoder import Qwen3Encoder
from .run_recovery import prepare_embedding_run_root


ModalityName = Literal["node_text", "event_text"]


class StagedEmbeddingError(RuntimeError):
    pass


def assigned_batch_bounds(total_batches: int, shard_id: int, num_shards: int) -> tuple[int, int]:
    if num_shards <= 0:
        raise StagedEmbeddingError("num_shards must be positive")
    if not 0 <= shard_id < num_shards:
        raise StagedEmbeddingError("shard_id must satisfy 0 <= shard_id < num_shards")
    if total_batches < 0:
        raise StagedEmbeddingError("total_batches must be non-negative")
    start = (total_batches * shard_id) // num_shards
    stop = (total_batches * (shard_id + 1)) // num_shards
    return start, stop


def assigned_batches(
    units: Sequence[EligibleTextUnit],
    *,
    batch_size: int,
    modality: ModalityName,
    shard_id: int,
    num_shards: int,
) -> tuple[EligibilityBatch, ...]:
    batches = tuple(_chunks(units, batch_size, modality))
    start, stop = assigned_batch_bounds(len(batches), shard_id, num_shards)
    return batches[start:stop]


def _selection(
    config: EmbeddingRunConfig, modality: ModalityName
) -> tuple[tuple[EligibleTextUnit, ...], Dict[str, Any], str]:
    if config.execution_mode == "mock":
        return _prefix_units(config, modality=modality)
    return _stratified_selection(config, modality=modality)


def _source(config: EmbeddingRunConfig, modality: ModalityName):
    if modality == "node_text":
        return _node_processor(config, full_scan=False).reader.identity
    return _event_processor(config, full_scan=False).reader.identity


def _writer(
    config: EmbeddingRunConfig,
    *,
    modality: ModalityName,
    preprocessing_hash: str,
    source: Any,
    encoder: Qwen3Encoder,
    repair_corrupt_shards: bool = False,
) -> FileEmbeddingWriter:
    spec = FileEmbeddingRunSpec(
        embedding_run_id=config.embedding_run_id,
        modality=modality,
        source=source,
        preprocessing_hash=preprocessing_hash,
        encoder=encoder.metadata,
        vector_validation=VectorValidationPolicy.from_encoder_metadata(encoder.metadata),
    )
    return FileEmbeddingWriter(
        config.resolved_output_root(),
        spec,
        repair_corrupt_shards=repair_corrupt_shards,
    )


def _checkpoint_payload(run_root: Path, modality: ModalityName) -> Dict[str, Any]:
    path = run_root / "checkpoints" / f"{modality}.json"
    if not path.is_file():
        return {"status": None, "committed_rows": 0, "batches": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    return {
        "status": value.get("status"),
        "committed_rows": int(value.get("committed_rows", 0)),
        "batches": value.get("batches") or {},
    }


def write_completed_shards_manifest(run_root: str | Path) -> Dict[str, Any]:
    root = Path(run_root)
    modalities = {
        name: _checkpoint_payload(root, name)
        for name in ("node_text", "event_text")
    }
    payload = {
        "schema_version": "tdmec-completed-shards-v1",
        "updated_at": utc_timestamp(),
        "modalities": modalities,
        "checksum_policy": "sha256_parquet_and_metadata_sidecar_verified_on_resume",
        "atomic_publish": "temporary_file_fsync_then_os_replace",
    }
    _atomic_write_json(root / "completed_shards.json", payload)
    return payload


def _execution_manifest(
    config: EmbeddingRunConfig, *, node_source: Any, event_source: Any
) -> Dict[str, Any]:
    try:
        import torch
    except ImportError:
        torch = None
    return {
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
    }


def run_embedding_shard(
    config: EmbeddingRunConfig,
    *,
    modality: ModalityName,
    shard_id: int,
    num_shards: int,
    preflight_report_path: str | Path,
    wall_clock_budget_seconds: float,
    graceful_stop_seconds: float = 1200.0,
    max_output_batches: int | None = None,
    repair_corrupt_shards: bool = False,
) -> Dict[str, Any]:
    """Process one deterministic contiguous lane and commit only full batches."""

    config.validate()
    verify_preflight_report(preflight_report_path, config)
    if not config.resume:
        config = replace(config, resume=True, replace_incomplete=False)
    if wall_clock_budget_seconds <= graceful_stop_seconds + 60:
        raise StagedEmbeddingError("wall-clock budget must exceed grace interval by 60 seconds")
    if max_output_batches is not None and max_output_batches < 0:
        raise StagedEmbeddingError("max_output_batches must be non-negative")
    started = time.monotonic()
    stop_at = started + wall_clock_budget_seconds - graceful_stop_seconds
    prepare_embedding_run_root(
        output_root=config.resolved_output_root(),
        embedding_run_id=config.embedding_run_id,
        resume=True,
        replace_incomplete=False,
    )
    run_root = config.resolved_output_root() / config.embedding_run_id
    node_source = _source(config, "node_text")
    event_source = _source(config, "event_text")
    run_root.mkdir(parents=True, exist_ok=True)
    execution_path = run_root / "execution_manifest.json"
    current_execution = _execution_manifest(
        config, node_source=node_source, event_source=event_source
    )
    if execution_path.is_file():
        existing = json.loads(execution_path.read_text(encoding="utf-8"))
        current_execution["event_alignment_fingerprint"] = existing.get(
            "event_alignment_fingerprint"
        )
        identity_fields = (
            "git_commit",
            "configuration_hash",
            "model_name",
            "model_revision",
            "tokenizer_revision",
            "input_fingerprints",
        )
        mismatched = [name for name in identity_fields if existing.get(name) != current_execution.get(name)]
        if mismatched:
            raise StagedEmbeddingError(f"execution manifest is incompatible: {mismatched}")
    else:
        current_execution["event_alignment_fingerprint"] = (
            compute_edge_order_fingerprint_from_root(
                config.event_source.resolved_root(),
                expected_run_id=config.event_source.run_id,
                batch_size=config.input_batch_size,
                verify_checksums=False,
            )
        )
        _atomic_write_json(execution_path, current_execution)

    log_event(
        "input_selection_started",
        modality=modality,
        shard_id=shard_id,
        num_shards=num_shards,
    )
    units, sampling, preprocessing_hash = _selection(config, modality)
    batches = assigned_batches(
        units,
        batch_size=config.output_shard_size,
        modality=modality,
        shard_id=shard_id,
        num_shards=num_shards,
    )
    encoder = Qwen3Encoder(config.encoder)
    writer = _writer(
        config,
        modality=modality,
        preprocessing_hash=preprocessing_hash,
        source=node_source if modality == "node_text" else event_source,
        encoder=encoder,
        repair_corrupt_shards=repair_corrupt_shards,
    )
    pending = [batch for batch in batches if not writer.is_batch_committed(batch)]
    log_event(
        "input_selection_completed",
        modality=modality,
        selected_eligible_texts=len(units),
        assigned_output_batches=len(batches),
        pending_output_batches=len(pending),
        source_eligibility=sampling.get("source_eligibility"),
    )
    committed_now = 0
    graceful_stop = False
    for batch in pending:
        if time.monotonic() >= stop_at:
            graceful_stop = True
            log_event(
                "graceful_stop_requested",
                modality=modality,
                reason="wall_clock_budget",
                committed_batches_this_session=committed_now,
            )
            break
        if max_output_batches is not None and committed_now >= max_output_batches:
            graceful_stop = True
            log_event(
                "graceful_stop_requested",
                modality=modality,
                reason="max_output_batches_test_hook",
                committed_batches_this_session=committed_now,
            )
            break
        log_event(
            "input_shard_started",
            modality=modality,
            source_batch_index=batch.source_batch_index,
            source_global_row_offset=batch.source_global_row_offset,
            eligible_texts=batch.num_rows,
        )
        vectors = encoder.encode(batch.units)
        commit = writer.write_batch(batch, vectors)
        committed_now += 1
        write_completed_shards_manifest(run_root)
        log_event(
            "input_shard_completed",
            modality=modality,
            batch_key=commit.batch_key,
            processed_rows=commit.row_count,
            total_committed_rows=commit.total_committed_rows,
            checksum=commit.shard_sha256,
            elapsed_seconds=time.monotonic() - started,
            resources=resource_snapshot(torch_module=encoder._torch),
        )
    assigned_complete = all(writer.is_batch_committed(batch) for batch in batches)
    status = "GRACEFUL_STOP" if graceful_stop and not assigned_complete else "SHARD_RANGE_COMPLETED"
    runtime = encoder.runtime_report()
    reports = run_root / "reports" / "shard_jobs"
    reports.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": "tdmec-embedding-shard-job-v1",
        "status": status,
        "modality": modality,
        "shard_id": shard_id,
        "num_shards": num_shards,
        "assigned_batch_count": len(batches),
        "assigned_row_count": sum(batch.num_rows for batch in batches),
        "committed_batches_this_session": committed_now,
        "assigned_range_complete": assigned_complete,
        "configuration_hash": config.scientific_hash(),
        "git_commit": git_commit_sha(),
        "model_revision": config.encoder.model_revision,
        "runtime": runtime,
        "session_provenance": current_execution,
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_write_json(
        reports / f"{modality}-shard-{shard_id:04d}-of-{num_shards:04d}.json",
        result,
    )
    write_completed_shards_manifest(run_root)
    log_event("shard_job_finished", **{k: result[k] for k in ("status", "modality", "shard_id", "num_shards", "elapsed_seconds")})
    return result


def verify_all_shards_complete(config: EmbeddingRunConfig) -> Dict[str, Any]:
    """Verify exact canonical batch coverage without loading model weights."""

    config.validate()
    results: Dict[str, Any] = {}
    encoder = Qwen3Encoder(config.encoder)  # metadata only; load() is never called
    for modality in ("node_text", "event_text"):
        units, _sampling, preprocessing_hash = _selection(config, modality)
        writer = _writer(
            config,
            modality=modality,
            preprocessing_hash=preprocessing_hash,
            source=_source(config, modality),
            encoder=encoder,
        )
        missing = [
            batch.source_batch_index
            for batch in _chunks(units, config.output_shard_size, modality)
            if not writer.is_batch_committed(batch)
        ]
        if missing:
            raise StagedEmbeddingError(
                f"{modality} is missing canonical output batches: {missing[:20]}"
            )
        results[modality] = {
            "expected_rows": len(units),
            "committed_rows": writer.committed_rows,
            "exact_batch_coverage": True,
        }
    return results


__all__ = [
    "StagedEmbeddingError",
    "assigned_batch_bounds",
    "assigned_batches",
    "run_embedding_shard",
    "verify_all_shards_complete",
    "write_completed_shards_manifest",
]
