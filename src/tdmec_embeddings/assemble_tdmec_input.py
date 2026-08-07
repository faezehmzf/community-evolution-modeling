"""Assemble a standardized TDMEC_INPUT package from existing graph + pooled text artifacts.

Used for smoke E2E training packages (mock or real pooled tensors) without re-running
the embedding encoder. Layout matches what the future trainer / 10k pilot will consume.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from tdmec import constants as C
from tdmec.hashing import hash_canonical, sha256_file

from .file_writer import _atomic_write_json
from .implementation_status import IMPLEMENTATION_STATUS_LABELS


class AssembleError(RuntimeError):
    pass


_GRAPH_FILES = (
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

_EXPECTED_T = 35
_EXPECTED_N = int(C.N_NODES)
_EXPECTED_E = 794_637
_EXPECTED_F = 17
_EXPECTED_RELATIONS = (0, 1, 2, 3)


def _copy_file(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise AssembleError(f"required file missing: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.is_dir():
        raise AssembleError(f"required directory missing: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _rel_checksums(root: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "package_checksums.json":
            continue
        out[path.relative_to(root).as_posix()] = sha256_file(path)
    return out


def _load_run_id(manifest_path: Path, fallback: str) -> str:
    if not manifest_path.is_file():
        return fallback
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AssembleError(f"invalid manifest JSON: {manifest_path}") from exc
    return str(payload.get("run_id") or payload.get("embedding_run_id") or fallback)


def validate_smoke_shapes(
    package_root: str | Path,
    *,
    expected_t: int = _EXPECTED_T,
    expected_n: int = _EXPECTED_N,
    expected_e: int = _EXPECTED_E,
    expected_f: int = _EXPECTED_F,
) -> Dict[str, Any]:
    """Validate TDMEC_INPUT_smoke_e2e tensor contracts for the DataLoader."""

    root = Path(package_root).resolve()
    failures: list[str] = []
    checks: Dict[str, Any] = {}

    def _fail(name: str, ok: bool, detail: str) -> None:
        checks[name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            failures.append(f"{name}: {detail}")

    x = np.load(root / "graph" / "X_struct.npy", mmap_mode="r")
    sm = np.load(root / "graph" / "struct_active_mask.npy", mmap_mode="r")
    node = np.load(root / "text_embeddings" / "node_snapshot_embeddings.npy", mmap_mode="r")
    node_mask = np.load(root / "text_embeddings" / "node_text_available_mask.npy", mmap_mode="r")
    node_count = np.load(root / "text_embeddings" / "node_valid_text_count.npy", mmap_mode="r")
    edge = np.load(root / "text_embeddings" / "canonical_edge_embeddings.npy", mmap_mode="r")
    edge_mask = np.load(root / "text_embeddings" / "edge_text_available_mask.npy", mmap_mode="r")
    edge_count = np.load(root / "text_embeddings" / "edge_valid_event_count.npy", mmap_mode="r")

    _fail("X_struct_shape", x.shape == (expected_t, expected_n, expected_f), str(list(x.shape)))
    _fail("struct_mask_shape", sm.shape == (expected_t, expected_n), str(list(sm.shape)))
    _fail("node_emb_TN", node.shape[:2] == (expected_t, expected_n), str(list(node.shape)))
    _fail("edge_E", edge.shape[0] == expected_e, str(list(edge.shape)))
    _fail("node_mask_align", node_mask.shape == node.shape[:2], str(list(node_mask.shape)))
    _fail("edge_mask_align", edge_mask.shape == (edge.shape[0],), str(list(edge_mask.shape)))
    _fail("node_mask_count", bool(np.array_equal(node_mask, node_count > 0)), "mask!=count>0")
    _fail("edge_mask_count", bool(np.array_equal(edge_mask, edge_count > 0)), "mask!=count>0")
    unavailable_n = ~np.asarray(node_mask)
    unavailable_e = ~np.asarray(edge_mask)
    _fail(
        "node_unavailable_zero",
        (not unavailable_n.any())
        or bool(np.all(np.asarray(node)[unavailable_n] == np.float32(0.0))),
        "non-zero unavailable node vectors",
    )
    _fail(
        "edge_unavailable_zero",
        (not unavailable_e.any())
        or bool(np.all(np.asarray(edge)[unavailable_e] == np.float32(0.0))),
        "non-zero unavailable edge vectors",
    )
    _fail("finite_node", bool(np.all(np.isfinite(node))), "NaN/Inf in node embeddings")
    _fail("finite_edge", bool(np.all(np.isfinite(edge))), "NaN/Inf in edge embeddings")
    _fail("finite_X_struct", bool(np.all(np.isfinite(x))), "NaN/Inf in X_struct")

    edges_dir = root / "graph" / "edges"
    _fail("edges_dir", edges_dir.is_dir(), "graph/edges missing")
    found_rels = sorted(
        {
            int(p.name.split("=", 1)[1])
            for p in edges_dir.glob("snapshot=*/relation=*")
            if p.is_dir() and p.name.startswith("relation=")
        }
    )
    expected_rels = list(_EXPECTED_RELATIONS)
    _fail(
        "relation_ids",
        found_rels == expected_rels,
        f"found={found_rels} expected={expected_rels}",
    )

    required = [
        "manifests/package_manifest.json",
        "manifests/graph_manifest.json",
        "manifests/embedding_manifest.json",
        "checksums/package_checksums.json",
        "configs/smoke_e2e_config.json",
    ]
    for rel in required:
        _fail(f"exists:{rel}", (root / rel).is_file(), "missing")

    report = {
        "schema_version": "tdmec-smoke-e2e-validation-v1",
        "package_root": root.as_posix(),
        "passed": len(failures) == 0,
        "failures": failures,
        "checks": checks,
        "shapes": {
            "X_struct": list(x.shape),
            "node_snapshot_embeddings": list(node.shape),
            "canonical_edge_embeddings": list(edge.shape),
            "D_text": int(node.shape[-1]),
        },
        "expected": {
            "T": expected_t,
            "N": expected_n,
            "E": expected_e,
            "F_struct": expected_f,
            "relations": list(_EXPECTED_RELATIONS),
        },
        "status_labels": list(IMPLEMENTATION_STATUS_LABELS),
    }
    return report


def assemble_tdmec_input_smoke_e2e(
    *,
    graph_root: str | Path,
    text_pooled_root: str | Path,
    package_root: str | Path,
    package_id: str = "TDMEC_INPUT_smoke_e2e",
    text_source_label: str = "mock_e2e",
    embedding_run_id: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Materialize a trainer-ready smoke E2E package."""

    graph = Path(graph_root).resolve()
    text = Path(text_pooled_root).resolve()
    # Allow either .../pooled or embedding run root containing pooled/
    if (text / "node_snapshot_embeddings.npy").is_file():
        pooled = text
        emb_run_root = text.parent if text.name == "pooled" else text
    elif (text / "pooled" / "node_snapshot_embeddings.npy").is_file():
        pooled = text / "pooled"
        emb_run_root = text
    else:
        raise AssembleError(
            f"could not find pooled node embeddings under {text}"
        )

    package = Path(package_root).resolve()
    if package.exists():
        if not overwrite:
            raise AssembleError(
                f"package root already exists (pass overwrite=True): {package}"
            )
        shutil.rmtree(package)

    for directory in (
        "manifests",
        "checksums",
        "graph",
        "text_embeddings",
        "configs",
        "validation_reports",
    ):
        (package / directory).mkdir(parents=True, exist_ok=True)

    for name in _GRAPH_FILES:
        src = graph / name
        if src.is_file():
            _copy_file(src, package / "graph" / name)
    _copy_tree(graph / "edges", package / "graph" / "edges")

    for name in _POOLED_FILES:
        _copy_file(pooled / name, package / "text_embeddings" / name)
    for meta in ("node_text_alignment.json", "event_text_alignment.json"):
        src = pooled / meta
        if src.is_file():
            _copy_file(src, package / "text_embeddings" / "metadata" / meta)

    graph_run_id = _load_run_id(graph / "manifest.json", "smoke_a_pg_001")
    emb_manifest_src = emb_run_root / "embedding_manifest.json"
    emb_run_id = embedding_run_id or _load_run_id(
        emb_manifest_src, text_source_label
    )
    if emb_manifest_src.is_file():
        _copy_file(emb_manifest_src, package / "manifests" / "embedding_manifest.json")
    else:
        # Minimal embedding manifest when only pooled/ is provided
        node = np.load(pooled / "node_snapshot_embeddings.npy", mmap_mode="r")
        _atomic_write_json(
            package / "manifests" / "embedding_manifest.json",
            {
                "schema_version": "tdmec-embedding-final-manifest-v1",
                "embedding_run_id": emb_run_id,
                "status": "COMPLETED",
                "source": "assembled_from_pooled_tensors",
                "D_text": int(node.shape[-1]),
                "status_labels": list(IMPLEMENTATION_STATUS_LABELS),
            },
        )
    _copy_file(graph / "manifest.json", package / "manifests" / "graph_manifest.json")

    node = np.load(package / "text_embeddings" / "node_snapshot_embeddings.npy", mmap_mode="r")
    d_text = int(node.shape[-1])
    config_payload = {
        "package_id": package_id,
        "graph_run_id": graph_run_id,
        "embedding_run_id": emb_run_id,
        "text_source_label": text_source_label,
        "D_text": d_text,
        "T": _EXPECTED_T,
        "N": _EXPECTED_N,
        "E": _EXPECTED_E,
        "F_struct": _EXPECTED_F,
        "relation_order": list(C.RELATION_ORDER),
        "relation_ids": list(_EXPECTED_RELATIONS),
        "status_labels": list(
            dict.fromkeys(
                [
                    "PROVISIONAL_SMOKE_ONLY",
                    "ENGINEERING_VALIDATION",
                    "NOT_FOR_FINAL_THESIS_CONCLUSIONS",
                    "FILE_ARTIFACT_SOURCE",
                    *IMPLEMENTATION_STATUS_LABELS,
                ]
            )
        ),
        "swap_notes": (
            "Replace text_embeddings/ with smoke-64 pooled tensors using the same "
            "filenames to upgrade from mock to real Qwen embeddings without changing "
            "graph/ or trainer layout."
        ),
    }
    _atomic_write_json(package / "configs" / "smoke_e2e_config.json", config_payload)

    package_manifest = {
        "schema_version": "tdmec-input-package-v1",
        "package_name": package_id,
        "graph_run_id": graph_run_id,
        "embedding_run_id": emb_run_id,
        "D_text": d_text,
        "status_labels": config_payload["status_labels"],
        "contents": {
            "graph": sorted(
                p.relative_to(package).as_posix()
                for p in (package / "graph").rglob("*")
                if p.is_file()
            ),
            "text_embeddings": sorted(
                p.relative_to(package).as_posix()
                for p in (package / "text_embeddings").rglob("*")
                if p.is_file()
            ),
        },
    }
    package_manifest["package_manifest_hash"] = hash_canonical(
        {k: v for k, v in package_manifest.items() if k != "package_manifest_hash"}
    )
    _atomic_write_json(package / "manifests" / "package_manifest.json", package_manifest)

    checksums = _rel_checksums(package)
    _atomic_write_json(package / "checksums" / "package_checksums.json", checksums)

    validation = validate_smoke_shapes(package)
    _atomic_write_json(
        package / "validation_reports" / "smoke_e2e_validation.json", validation
    )
    if not validation["passed"]:
        raise AssembleError(
            "assembled package failed validation: " + "; ".join(validation["failures"])
        )

    return {
        "package_root": package.as_posix(),
        "package_manifest": package_manifest,
        "validation": validation,
        "checksum_count": len(checksums),
        "D_text": d_text,
        "status_labels": config_payload["status_labels"],
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Assemble TDMEC_INPUT_smoke_e2e from existing smoke graph + pooled text"
    )
    p.add_argument(
        "--graph-root",
        default="/teamspace/studios/this_studio/TDMEC_PROJECT_OUTPUTS/graph/smoke_a_pg_001",
    )
    p.add_argument(
        "--text-root",
        default=(
            "/teamspace/studios/this_studio/TDMEC_PROJECT_OUTPUTS/"
            "embeddings/mock_e2e_studio_sample_001"
        ),
        help="Embedding run root (with pooled/) or pooled/ directory itself",
    )
    p.add_argument(
        "--package-root",
        default=(
            "/teamspace/studios/this_studio/TDMEC_PROJECT_OUTPUTS/"
            "tdmec_input/TDMEC_INPUT_smoke_e2e"
        ),
    )
    p.add_argument("--package-id", default="TDMEC_INPUT_smoke_e2e")
    p.add_argument("--text-source-label", default="mock_e2e")
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = assemble_tdmec_input_smoke_e2e(
            graph_root=args.graph_root,
            text_pooled_root=args.text_root,
            package_root=args.package_root,
            package_id=args.package_id,
            text_source_label=args.text_source_label,
            overwrite=bool(args.overwrite),
        )
    except AssembleError as exc:
        print(f"ASSEMBLE_FAILED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AssembleError",
    "assemble_tdmec_input_smoke_e2e",
    "validate_smoke_shapes",
    "main",
]
