"""End-to-end TDMEC embedding pilot orchestration (encode → pool → align → export).

IMPLEMENTED_NOT_EXECUTED / TRANSFER_PENDING_VALIDATION.

This module does not download models on import.  Real Qwen execution still
requires explicit CLI authorization flags.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .alignment import AlignmentError, align_graph_and_text
from .config import EmbeddingRunConfig
from .export import ExportError, export_tdmec_input_package
from .file_writer import _atomic_write_json
from .implementation_status import IMPLEMENTATION_STATUS_LABELS
from .pipeline import EmbeddingPipelineError, run_embedding_pipeline
from .sampling import SamplingError
from .validation import validate_pooled_embeddings, validate_tdmec_input_package


class PilotPipelineError(RuntimeError):
    pass


def run_tdmec_embedding_pilot(
    config: EmbeddingRunConfig,
    *,
    authorize_real_model: bool = False,
    authorize_bounded_pilot: bool = False,
    skip_export: bool = False,
    package_root: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Run the bounded pilot stack and optionally build ``TDMEC_INPUT``."""

    try:
        embedding_manifest = run_embedding_pipeline(
            config,
            authorize_real_model=authorize_real_model,
            authorize_bounded_pilot=authorize_bounded_pilot,
        )
    except SamplingError as exc:
        raise PilotPipelineError(str(exc)) from exc
    except EmbeddingPipelineError as exc:
        raise PilotPipelineError(str(exc)) from exc

    if config.dry_run:
        return {
            "status": "DRY_RUN",
            "embedding": embedding_manifest,
            "status_labels": list(IMPLEMENTATION_STATUS_LABELS),
        }

    run_root = config.resolved_output_root() / config.embedding_run_id
    reports_dir = run_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    embedding_validation = validate_pooled_embeddings(
        run_root / "pooled",
        expected_dimension=config.encoder.output_dimension,
        normalized_atol=config.validation.normalized_atol,
        require_unit_norm_on_available=False,  # Stage-B N1: arithmetic mean, not re-L2
    )
    _atomic_write_json(reports_dir / "embedding_validation.json", embedding_validation)
    if not embedding_validation["passed"]:
        raise PilotPipelineError(
            "embedding validation failed: " + "; ".join(embedding_validation["failures"])
        )

    try:
        alignment = align_graph_and_text(
            embedding_run_root=run_root,
            graph_artifact_root=config.event_source.resolved_root(),
            expected_graph_run_id=config.event_source.run_id,
            expected_dimension=config.encoder.output_dimension,
            batch_size=config.input_batch_size,
            write_report=True,
        )
    except AlignmentError as exc:
        raise PilotPipelineError(f"graph-text alignment failed: {exc}") from exc

    export_result: Optional[Dict[str, Any]] = None
    package_validation: Optional[Dict[str, Any]] = None
    if not skip_export:
        pkg = Path(package_root) if package_root else (
            config.resolved_output_root()
            / f"{config.export.package_name}_{config.embedding_run_id}"
        )
        try:
            export_result = export_tdmec_input_package(
                embedding_run_root=run_root,
                graph_artifact_root=config.event_source.resolved_root(),
                package_root=pkg,
                config=config,
                alignment_report=alignment,
            )
        except ExportError as exc:
            raise PilotPipelineError(f"TDMEC_INPUT export failed: {exc}") from exc
        package_validation = validate_tdmec_input_package(
            pkg,
            expected_dimension=config.encoder.output_dimension,
            verify_checksums=True,
        )
        _atomic_write_json(
            Path(pkg) / "validation_reports" / "tdmec_input_validation.json",
            package_validation,
        )
        _atomic_write_json(reports_dir / "tdmec_input_validation.json", package_validation)
        if not package_validation["passed"]:
            raise PilotPipelineError(
                "TDMEC_INPUT validation failed: "
                + "; ".join(package_validation["failures"])
            )

    final = {
        "schema_version": "tdmec-embedding-pilot-result-v1",
        "status": "COMPLETED",
        "embedding_run_id": config.embedding_run_id,
        "embedding_run_root": run_root.as_posix(),
        "embedding_manifest_status": embedding_manifest.get("status"),
        "embedding_validation_passed": embedding_validation["passed"],
        "alignment_passed": alignment.get("passed"),
        "alignment_compatibility_hash": alignment.get("compatibility_hash"),
        "export": export_result,
        "package_validation_passed": None
        if package_validation is None
        else package_validation.get("passed"),
        "status_labels": list(IMPLEMENTATION_STATUS_LABELS),
    }
    _atomic_write_json(reports_dir / "pilot_final_report.json", final)
    return final


__all__ = ["PilotPipelineError", "run_tdmec_embedding_pilot"]
