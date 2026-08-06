"""Export a TDMEC_INPUT package joining graph companions and text embeddings.

IMPLEMENTED_NOT_EXECUTED / TRANSFER_PENDING_VALIDATION.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from tdmec.hashing import hash_canonical, sha256_file

from .config import EmbeddingRunConfig, ExportConfig
from .file_writer import _atomic_write_json
from .implementation_status import IMPLEMENTATION_STATUS_LABELS


class ExportError(RuntimeError):
    pass


_GRAPH_ROOT_FILES = (
    "X_struct.npy",
    "struct_active_mask.npy",
    "struct_feature_names.json",
    "snapshot_calendar.json",
    "manifest.json",
    "validation_report.json",
    "checksums.json",
)

_POOLED_FILES = (
    "node_snapshot_embeddings.npy",
    "node_text_available_mask.npy",
    "node_valid_text_count.npy",
    "canonical_edge_embeddings.npy",
    "edge_text_available_mask.npy",
    "edge_valid_event_count.npy",
)


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _rel_checksums(root: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name in {"package_checksums.json", "all_checksums.json"}:
            continue
        out[path.relative_to(root).as_posix()] = sha256_file(path)
    return out


def export_tdmec_input_package(
    *,
    embedding_run_root: str | Path,
    graph_artifact_root: str | Path,
    package_root: str | Path,
    config: EmbeddingRunConfig,
    alignment_report: Optional[Mapping[str, Any]] = None,
    export_config: Optional[ExportConfig] = None,
) -> Dict[str, Any]:
    """Materialize ``TDMEC_INPUT/`` ready for Lightning migration / training prep."""

    run_root = Path(embedding_run_root).resolve()
    graph_root = Path(graph_artifact_root).resolve()
    export_cfg = export_config or config.export
    package = Path(package_root).resolve()
    if package.exists():
        raise ExportError(
            f"package root already exists and overwrite is prohibited: {package}"
        )
    for directory in (
        "manifests",
        "checksums",
        "graph",
        "text_embeddings",
        "configs",
        "validation_reports",
    ):
        (package / directory).mkdir(parents=True, exist_ok=True)

    # Graph companions
    for name in _GRAPH_ROOT_FILES:
        src = graph_root / name
        if src.is_file():
            _copy_file(src, package / "graph" / name)
    edges_src = graph_root / "edges"
    if export_cfg.include_graph_edges:
        if not edges_src.is_dir():
            raise ExportError("graph edges/ directory is required for TDMEC_INPUT")
        _copy_tree(edges_src, package / "graph" / "edges")

    # Text embeddings (pooled)
    pooled = run_root / "pooled"
    for name in _POOLED_FILES:
        src = pooled / name
        if not src.is_file():
            raise ExportError(f"missing pooled tensor required for export: {name}")
        _copy_file(src, package / "text_embeddings" / name)
    for meta_name in ("node_text_alignment.json", "event_text_alignment.json"):
        meta = pooled / meta_name
        if meta.is_file():
            _copy_file(meta, package / "text_embeddings" / "metadata" / meta_name)

    if export_cfg.include_unit_embeddings:
        units = run_root / "unit_embeddings"
        if units.is_dir():
            _copy_tree(units, package / "text_embeddings" / "unit_embeddings")

    # Configs + reports
    encoder_payload = {
        "encoder": config.encoder.scientific_payload(),
        "pooling": config.pooling.__dict__,
        "sampling": config.sampling.__dict__,
        "validation": {
            "normalized_atol": config.validation.normalized_atol,
            "require_finite": config.validation.require_finite,
            "require_relation_coverage": config.validation.require_relation_coverage,
            "expected_relation_ids": list(config.validation.expected_relation_ids),
        },
        "execution_mode": config.execution_mode,
        "embedding_run_id": config.embedding_run_id,
        "configuration_hash": config.scientific_hash(),
    }
    _atomic_write_json(package / "configs" / "encoder_config.json", encoder_payload)
    _atomic_write_json(
        package / "configs" / "pooling_config.json",
        {
            "final_normalization": config.pooling.final_normalization,
            "accumulation_dtype": "float32",
            "missing_text_policy": "exact_zero_plus_boolean_mask_QMISS_M1",
        },
    )

    reports_src = run_root / "reports"
    if reports_src.is_dir():
        for path in reports_src.glob("*.json"):
            _copy_file(path, package / "validation_reports" / path.name)

    embedding_manifest_src = run_root / "embedding_manifest.json"
    if embedding_manifest_src.is_file():
        _copy_file(embedding_manifest_src, package / "manifests" / "embedding_manifest.json")
    graph_manifest_src = graph_root / "manifest.json"
    if graph_manifest_src.is_file():
        _copy_file(graph_manifest_src, package / "manifests" / "graph_manifest.json")

    package_manifest = {
        "schema_version": "tdmec-input-package-v1",
        "package_name": export_cfg.package_name,
        "embedding_run_id": config.embedding_run_id,
        "graph_run_id": config.event_source.run_id,
        "node_text_source_run_id": config.node_source.run_id,
        "configuration_hash": config.scientific_hash(),
        "alignment_compatibility_hash": (alignment_report or {}).get("compatibility_hash"),
        "alignment_passed": bool((alignment_report or {}).get("passed", False)),
        "D_text": config.encoder.output_dimension,
        "model_name": config.encoder.model_name,
        "model_revision": config.encoder.model_revision,
        "status_labels": list(IMPLEMENTATION_STATUS_LABELS)
        + list(config.provisional_labels),
        "contents": {
            "graph": sorted(p.relative_to(package).as_posix() for p in (package / "graph").rglob("*") if p.is_file()),
            "text_embeddings": sorted(
                p.relative_to(package).as_posix()
                for p in (package / "text_embeddings").rglob("*")
                if p.is_file()
            ),
        },
    }
    _atomic_write_json(package / "manifests" / "package_manifest.json", package_manifest)

    checksums = _rel_checksums(package)
    _atomic_write_json(package / "checksums" / "package_checksums.json", checksums)
    package_manifest["package_checksum_count"] = len(checksums)
    package_manifest["package_manifest_hash"] = hash_canonical(
        {k: v for k, v in package_manifest.items() if k != "package_manifest_hash"}
    )
    _atomic_write_json(package / "manifests" / "package_manifest.json", package_manifest)
    # Refresh checksums after rewriting package_manifest
    checksums = _rel_checksums(package)
    _atomic_write_json(package / "checksums" / "package_checksums.json", checksums)
    return {
        "package_root": package.as_posix(),
        "package_manifest": package_manifest,
        "checksum_count": len(checksums),
        "status_labels": list(IMPLEMENTATION_STATUS_LABELS),
    }


__all__ = ["ExportError", "export_tdmec_input_package"]
