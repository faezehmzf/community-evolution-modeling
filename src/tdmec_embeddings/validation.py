"""Validation helpers for TDMEC embedding runs and TDMEC_INPUT packages.

IMPLEMENTED_NOT_EXECUTED / TRANSFER_PENDING_VALIDATION.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from tdmec.hashing import sha256_file

from .implementation_status import IMPLEMENTATION_STATUS_LABELS


class ValidationReportError(RuntimeError):
    pass


def _check(name: str, ok: bool, detail: str, failures: List[str], checks: Dict[str, Any]) -> None:
    checks[name] = {"ok": bool(ok), "detail": detail}
    if not ok:
        failures.append(f"{name}: {detail}")


def validate_pooled_embeddings(
    text_root: str | Path,
    *,
    expected_dimension: int,
    expected_t: Optional[int] = None,
    expected_n: Optional[int] = None,
    expected_e: Optional[int] = None,
    normalized_atol: float = 1e-6,
    require_unit_norm_on_available: bool = False,
) -> Dict[str, Any]:
    root = Path(text_root)
    failures: List[str] = []
    checks: Dict[str, Any] = {}

    node = np.load(root / "node_snapshot_embeddings.npy", mmap_mode="r")
    node_mask = np.load(root / "node_text_available_mask.npy", mmap_mode="r")
    node_count = np.load(root / "node_valid_text_count.npy", mmap_mode="r")
    edge = np.load(root / "canonical_edge_embeddings.npy", mmap_mode="r")
    edge_mask = np.load(root / "edge_text_available_mask.npy", mmap_mode="r")
    edge_count = np.load(root / "edge_valid_event_count.npy", mmap_mode="r")

    _check(
        "node_rank_and_dim",
        node.ndim == 3 and node.shape[-1] == expected_dimension,
        f"shape={list(node.shape)} expected[..., {expected_dimension}]",
        failures,
        checks,
    )
    _check(
        "edge_rank_and_dim",
        edge.ndim == 2 and edge.shape[-1] == expected_dimension,
        f"shape={list(edge.shape)} expected[E, {expected_dimension}]",
        failures,
        checks,
    )
    if expected_t is not None:
        _check("T", node.shape[0] == expected_t, f"got {node.shape[0]}", failures, checks)
    if expected_n is not None:
        _check("N", node.shape[1] == expected_n, f"got {node.shape[1]}", failures, checks)
    if expected_e is not None:
        _check("E", edge.shape[0] == expected_e, f"got {edge.shape[0]}", failures, checks)

    _check("node_finite", bool(np.all(np.isfinite(node))), "NaN/Inf in node embeddings", failures, checks)
    _check("edge_finite", bool(np.all(np.isfinite(edge))), "NaN/Inf in edge embeddings", failures, checks)
    _check(
        "node_mask_count",
        bool(np.array_equal(node_mask, node_count > 0)),
        "node mask must equal count>0",
        failures,
        checks,
    )
    _check(
        "edge_mask_count",
        bool(np.array_equal(edge_mask, edge_count > 0)),
        "edge mask must equal count>0",
        failures,
        checks,
    )
    unavailable_n = ~np.asarray(node_mask)
    unavailable_e = ~np.asarray(edge_mask)
    _check(
        "node_unavailable_zero",
        (not unavailable_n.any())
        or bool(np.all(np.asarray(node)[unavailable_n] == np.float32(0.0))),
        "unavailable node vectors must be exact zeros",
        failures,
        checks,
    )
    _check(
        "edge_unavailable_zero",
        (not unavailable_e.any())
        or bool(np.all(np.asarray(edge)[unavailable_e] == np.float32(0.0))),
        "unavailable edge vectors must be exact zeros",
        failures,
        checks,
    )
    if require_unit_norm_on_available and np.any(node_mask):
        norms = np.linalg.norm(np.asarray(node)[np.asarray(node_mask)].astype(np.float64), axis=-1)
        _check(
            "node_available_unit_norm",
            bool(np.allclose(norms, 1.0, rtol=0.0, atol=normalized_atol)),
            f"available node pooled norms out of atol={normalized_atol}",
            failures,
            checks,
        )

    return {
        "schema_version": "tdmec-embedding-validation-v1",
        "passed": len(failures) == 0,
        "failures": failures,
        "checks": checks,
        "shapes": {
            "node_snapshot_embeddings": list(node.shape),
            "canonical_edge_embeddings": list(edge.shape),
        },
        "status_labels": list(IMPLEMENTATION_STATUS_LABELS),
    }


def validate_tdmec_input_package(
    package_root: str | Path,
    *,
    expected_dimension: Optional[int] = None,
    verify_checksums: bool = True,
) -> Dict[str, Any]:
    root = Path(package_root).resolve()
    failures: List[str] = []
    checks: Dict[str, Any] = {}

    required = [
        "manifests/package_manifest.json",
        "manifests/embedding_manifest.json",
        "manifests/graph_manifest.json",
        "checksums/package_checksums.json",
        "graph/X_struct.npy",
        "graph/struct_active_mask.npy",
        "text_embeddings/node_snapshot_embeddings.npy",
        "text_embeddings/node_text_available_mask.npy",
        "text_embeddings/node_valid_text_count.npy",
        "text_embeddings/canonical_edge_embeddings.npy",
        "text_embeddings/edge_text_available_mask.npy",
        "text_embeddings/edge_valid_event_count.npy",
    ]
    for rel in required:
        path = root / rel
        _check(f"exists:{rel}", path.is_file(), "missing required package file", failures, checks)

    package_manifest = json.loads((root / "manifests" / "package_manifest.json").read_text(encoding="utf-8"))
    dim = expected_dimension or int(package_manifest.get("D_text") or 0)
    if dim <= 0:
        failures.append("D_text missing from package_manifest")
    else:
        text_report = validate_pooled_embeddings(
            root / "text_embeddings",
            expected_dimension=dim,
        )
        checks["text_embeddings"] = text_report
        if not text_report["passed"]:
            failures.extend(text_report["failures"])

    x_struct = np.load(root / "graph" / "X_struct.npy", mmap_mode="r")
    node = np.load(root / "text_embeddings" / "node_snapshot_embeddings.npy", mmap_mode="r")
    _check(
        "graph_text_TN",
        x_struct.shape[:2] == node.shape[:2],
        f"X_struct {list(x_struct.shape)} vs node text {list(node.shape)}",
        failures,
        checks,
    )

    if verify_checksums and (root / "checksums" / "package_checksums.json").is_file():
        recorded = json.loads((root / "checksums" / "package_checksums.json").read_text(encoding="utf-8"))
        mismatched = []
        for rel, digest in recorded.items():
            path = root / rel
            if not path.is_file():
                mismatched.append(rel)
                continue
            if path.name == "package_checksums.json":
                continue
            if sha256_file(path) != digest:
                mismatched.append(rel)
        _check(
            "package_checksums",
            len(mismatched) == 0,
            f"mismatched={mismatched[:10]}",
            failures,
            checks,
        )

    return {
        "schema_version": "tdmec-input-validation-v1",
        "package_root": root.as_posix(),
        "passed": len(failures) == 0,
        "failures": failures,
        "checks": checks,
        "package_manifest_embedding_run_id": package_manifest.get("embedding_run_id"),
        "status_labels": list(IMPLEMENTATION_STATUS_LABELS),
    }


__all__ = [
    "ValidationReportError",
    "validate_pooled_embeddings",
    "validate_tdmec_input_package",
]
