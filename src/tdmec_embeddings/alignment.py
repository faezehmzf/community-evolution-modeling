"""Graph–text alignment checks for TDMEC embedding packages.

IMPLEMENTED_NOT_EXECUTED / TRANSFER_PENDING_VALIDATION.

Verifies that pooled text tensors share the Dataset A temporal / node / edge
index contract used by structural features and canonical edges.  Reports contain
hashes and shapes only (no raw text).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from tdmec.constants import RELATION_ORDER
from tdmec.hashing import hash_canonical, sha256_file

from .file_sources import CanonicalEdgeFileReader, load_file_source_identity
from .file_writer import _atomic_write_json
from .implementation_status import IMPLEMENTATION_STATUS_LABELS


class AlignmentError(RuntimeError):
    pass


def _load_npy(path: Path) -> np.ndarray:
    if not path.is_file():
        raise AlignmentError(f"required array is missing: {path.name}")
    return np.load(path, mmap_mode="r")


def compute_edge_order_fingerprint_from_root(
    artifact_root: str | Path,
    *,
    expected_run_id: str,
    batch_size: int = 4096,
    verify_checksums: bool = False,
) -> Dict[str, Any]:
    identity = load_file_source_identity(
        artifact_root,
        source_kind="dataset_a",
        expected_run_id=expected_run_id,
        verify_checksums=verify_checksums,
    )
    reader = CanonicalEdgeFileReader(
        artifact_root,
        expected_run_id=expected_run_id,
        batch_size=batch_size,
        identity=identity,
    )
    digest = hashlib.sha256()
    count = 0
    relation_counts: Dict[int, int] = {i: 0 for i in range(len(RELATION_ORDER))}
    for record_batch in reader.iter_batches():
        rb = record_batch.records
        snap = rb.column(rb.schema.get_field_index("snapshot_id")).to_numpy(zero_copy_only=False)
        rel = rb.column(rb.schema.get_field_index("relation_id")).to_numpy(zero_copy_only=False)
        src = rb.column(rb.schema.get_field_index("source_idx")).to_numpy(zero_copy_only=False)
        dst = rb.column(rb.schema.get_field_index("target_idx")).to_numpy(zero_copy_only=False)
        for i in range(len(snap)):
            digest.update(
                f"{int(snap[i])},{int(rel[i])},{int(src[i])},{int(dst[i])}\n".encode("ascii")
            )
            relation_counts[int(rel[i])] = relation_counts.get(int(rel[i]), 0) + 1
            count += 1
    if count != reader.total_rows:
        raise AlignmentError("edge fingerprint row count diverges from reader.total_rows")
    return {
        "edge_count_hashed": count,
        "edge_order_sha256": digest.hexdigest(),
        "relation_counts": {str(k): int(v) for k, v in sorted(relation_counts.items())},
        "relation_order": list(RELATION_ORDER),
        "canonical_edge_count": int(reader.total_rows),
    }


def align_graph_and_text(
    *,
    embedding_run_root: str | Path,
    graph_artifact_root: str | Path,
    expected_graph_run_id: str,
    expected_dimension: int,
    batch_size: int = 4096,
    write_report: bool = True,
) -> Dict[str, Any]:
    """Validate T/N/E alignment between graph structural tensors and pooled text."""

    run_root = Path(embedding_run_root).resolve()
    graph_root = Path(graph_artifact_root).resolve()
    pooled = run_root / "pooled"
    checks: Dict[str, Any] = {}
    failures: list[str] = []

    x_struct_path = graph_root / "X_struct.npy"
    struct_mask_path = graph_root / "struct_active_mask.npy"
    node_emb = _load_npy(pooled / "node_snapshot_embeddings.npy")
    node_mask = _load_npy(pooled / "node_text_available_mask.npy")
    node_count = _load_npy(pooled / "node_valid_text_count.npy")
    edge_emb = _load_npy(pooled / "canonical_edge_embeddings.npy")
    edge_mask = _load_npy(pooled / "edge_text_available_mask.npy")
    edge_count = _load_npy(pooled / "edge_valid_event_count.npy")
    x_struct = _load_npy(x_struct_path)
    struct_mask = _load_npy(struct_mask_path)

    t_text, n_text, d_text = (int(x) for x in node_emb.shape)
    if d_text != expected_dimension:
        failures.append(
            f"node embedding D={d_text} does not match expected D_text={expected_dimension}"
        )
    if x_struct.ndim != 3:
        failures.append("X_struct must be rank-3 [T,N,F]")
    else:
        t_struct, n_struct, f_struct = (int(x) for x in x_struct.shape)
        checks["T"] = {"text": t_text, "struct": t_struct, "match": t_text == t_struct}
        checks["N"] = {"text": n_text, "struct": n_struct, "match": n_text == n_struct}
        checks["F_struct"] = f_struct
        if t_text != t_struct:
            failures.append("snapshot count T mismatch between node text and X_struct")
        if n_text != n_struct:
            failures.append("node count N mismatch between node text and X_struct")
    if struct_mask.shape != (t_text, n_text):
        failures.append("struct_active_mask shape mismatches node text [T,N]")
    if node_mask.shape != (t_text, n_text) or node_count.shape != (t_text, n_text):
        failures.append("node text mask/count shapes must be [T,N]")
    if edge_emb.ndim != 2 or edge_emb.shape[1] != expected_dimension:
        failures.append("canonical_edge_embeddings must be [E,D_text]")
    if edge_mask.shape != (edge_emb.shape[0],) or edge_count.shape != (edge_emb.shape[0],):
        failures.append("edge mask/count must align to E")

    identity = load_file_source_identity(
        graph_root,
        source_kind="dataset_a",
        expected_run_id=expected_graph_run_id,
        verify_checksums=False,
    )
    if list(identity.relation_order) != list(RELATION_ORDER):
        failures.append("graph relation_order violates QREL-01")
    if identity.n_snapshots != t_text or identity.n_nodes != n_text:
        failures.append("FileSourceIdentity T/N diverge from pooled node tensors")

    fingerprint = compute_edge_order_fingerprint_from_root(
        graph_root,
        expected_run_id=expected_graph_run_id,
        batch_size=batch_size,
        verify_checksums=False,
    )
    e_graph = int(fingerprint["canonical_edge_count"])
    e_text = int(edge_emb.shape[0])
    checks["E"] = {"text": e_text, "graph": e_graph, "match": e_text == e_graph}
    if e_text != e_graph:
        failures.append("canonical edge count E mismatch between graph and edge text tensors")

    # Q-MISS: unavailable vectors must be exact zeros
    if np.any(node_mask) is False:
        pass
    unavailable_nodes = ~np.asarray(node_mask)
    if unavailable_nodes.any() and not np.all(
        np.asarray(node_emb)[unavailable_nodes] == np.float32(0.0)
    ):
        failures.append("unavailable node text vectors must be exact zeros")
    unavailable_edges = ~np.asarray(edge_mask)
    if unavailable_edges.any() and not np.all(
        np.asarray(edge_emb)[unavailable_edges] == np.float32(0.0)
    ):
        failures.append("unavailable edge text vectors must be exact zeros")

    report = {
        "schema_version": "tdmec-graph-text-alignment-v1",
        "embedding_run_root_name": run_root.name,
        "graph_run_id": expected_graph_run_id,
        "checks": checks,
        "edge_order": fingerprint,
        "tensor_shapes": {
            "node_snapshot_embeddings": list(node_emb.shape),
            "node_text_available_mask": list(node_mask.shape),
            "canonical_edge_embeddings": list(edge_emb.shape),
            "X_struct": list(x_struct.shape),
            "struct_active_mask": list(struct_mask.shape),
        },
        "file_hashes": {
            "node_snapshot_embeddings": sha256_file(pooled / "node_snapshot_embeddings.npy"),
            "canonical_edge_embeddings": sha256_file(pooled / "canonical_edge_embeddings.npy"),
            "X_struct": sha256_file(x_struct_path),
            "struct_active_mask": sha256_file(struct_mask_path),
        },
        "failures": failures,
        "passed": len(failures) == 0,
        "status_labels": list(IMPLEMENTATION_STATUS_LABELS),
        "compatibility_hash": hash_canonical(
            {
                "T": t_text,
                "N": n_text,
                "E": e_text,
                "D_text": expected_dimension,
                "edge_order_sha256": fingerprint["edge_order_sha256"],
                "graph_run_id": expected_graph_run_id,
            }
        ),
    }
    if write_report:
        out = run_root / "reports" / "graph_text_alignment_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(out, report)
        report["report_path"] = out.as_posix()
    if failures:
        raise AlignmentError("; ".join(failures))
    return report


__all__ = [
    "AlignmentError",
    "align_graph_and_text",
    "compute_edge_order_fingerprint_from_root",
]
