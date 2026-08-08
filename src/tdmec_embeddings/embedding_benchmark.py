"""Bounded representative Qwen benchmark with no scientific output publication."""
from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .config import EmbeddingRunConfig
from .file_writer import _atomic_write_json
from .model_preflight import verify_preflight_report
from .observability import git_commit_sha, log_event, resource_snapshot, utc_timestamp
from .pipeline import _stratified_selection
from .qwen_encoder import Qwen3Encoder


class EmbeddingBenchmarkError(RuntimeError):
    pass


def run_bounded_benchmark(
    config: EmbeddingRunConfig,
    *,
    preflight_report_path: str | Path,
    rows_per_modality: int,
    output_path: str | Path,
) -> Dict[str, Any]:
    if not 1 <= rows_per_modality <= 1024:
        raise EmbeddingBenchmarkError("rows_per_modality must be between 1 and 1024")
    verify_preflight_report(preflight_report_path, config)
    bounded = replace(
        config,
        max_node_rows=rows_per_modality,
        max_event_rows=rows_per_modality,
    )
    bounded.validate()
    log_event("benchmark_selection_started", rows_per_modality=rows_per_modality)
    node_units, node_sampling, _node_preprocessing = _stratified_selection(
        bounded, modality="node_text"
    )
    event_units, event_sampling, _event_preprocessing = _stratified_selection(
        bounded, modality="event_text"
    )
    units = tuple(node_units) + tuple(event_units)
    encoder = Qwen3Encoder(bounded.encoder)
    started = time.perf_counter()
    vectors = encoder.encode(units)
    elapsed = time.perf_counter() - started
    runtime = encoder.runtime_report()
    units_per_second = runtime.get("units_per_second")
    token_count = int(runtime.get("token_count") or 0)
    tokens_per_second = token_count / float(runtime.get("elapsed_seconds") or 1.0)
    total_pilot_rows = config.max_node_rows + config.max_event_rows
    report = {
        "schema_version": "tdmec-bounded-embedding-benchmark-v1",
        "status": "PASSED",
        "created_at": utc_timestamp(),
        "git_commit": git_commit_sha(),
        "configuration_hash": config.scientific_hash(),
        "model_revision": config.encoder.model_revision,
        "sample": {
            "node_rows": len(node_units),
            "event_rows": len(event_units),
            "total_rows": len(units),
            "node_sampling": node_sampling,
            "event_sampling": event_sampling,
        },
        "checks": {
            "shape": list(vectors.shape),
            "expected_dimension": config.encoder.output_dimension,
            "finite": bool(np.all(np.isfinite(vectors))),
        },
        "performance": {
            "elapsed_seconds_including_model_load": elapsed,
            "inference_rows_per_second": units_per_second,
            "inference_tokens_per_second": tokens_per_second,
            "estimated_configured_pilot_inference_seconds": (
                total_pilot_rows / units_per_second if units_per_second else None
            ),
        },
        "resources": resource_snapshot(torch_module=encoder._torch),
        "runtime": runtime,
        "scientific_outputs_written": False,
    }
    if not report["checks"]["finite"] or vectors.shape[1] != config.encoder.output_dimension:
        report["status"] = "FAILED"
    _atomic_write_json(Path(output_path), report)
    log_event(
        "benchmark_completed",
        status=report["status"],
        rows=len(units),
        rows_per_second=units_per_second,
        tokens_per_second=tokens_per_second,
    )
    if report["status"] != "PASSED":
        raise EmbeddingBenchmarkError("benchmark numerical checks failed")
    return report


__all__ = ["EmbeddingBenchmarkError", "run_bounded_benchmark"]
