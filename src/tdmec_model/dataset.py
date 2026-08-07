"""Load TDMEC_INPUT packages into per-snapshot batches (graph + optional text)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch

from tdmec import constants as C

from .types import PackageMeta, SnapshotBatch

_STATUS = (
    "PROVISIONAL_SMOKE_ONLY",
    "ENGINEERING_VALIDATION",
    "NOT_FOR_FINAL_THESIS_CONCLUSIONS",
)

_EDGE_PART_RE = re.compile(r"snapshot=(\d+)/relation=(\d+)/")


class DatasetError(RuntimeError):
    pass


def _as_bool_mask(arr: np.ndarray) -> torch.Tensor:
    out = np.array(arr, dtype=np.bool_, copy=True)
    return torch.from_numpy(out)


def _as_float32(arr: np.ndarray) -> torch.Tensor:
    out = np.array(arr, dtype=np.float32, copy=True)
    return torch.from_numpy(out)


def _as_int64(arr: np.ndarray) -> torch.Tensor:
    out = np.array(arr, dtype=np.int64, copy=True)
    return torch.from_numpy(out)


def _natural_key(text: str) -> Tuple:
    parts = re.split(r"(\d+)", text)
    key: list = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p)
    return tuple(key)


def _edge_part_sort_key(base: Path, path: Path) -> Tuple:
    relative = path.relative_to(base).as_posix()
    match = _EDGE_PART_RE.search(relative)
    if not match:
        return (10**9, 10**9, _natural_key(relative))
    return (int(match.group(1)), int(match.group(2)), _natural_key(relative))


class TDMECInputDataset:
    """DataLoader over a standardized TDMEC_INPUT package.

    Graph edges follow Dataset A partition layout. Canonical edge-text tensors
    are indexed by the same physical parquet order used by the embedding pooler
    (``CanonicalEdgeFileReader``).
    """

    def __init__(
        self,
        package_root: str | Path,
        *,
        device: Optional[torch.device] = None,
        mmap: bool = True,
        load_text: bool = True,
    ) -> None:
        self.root = Path(package_root).resolve()
        self.device = device
        self._mmap = mmap
        self.load_text = bool(load_text)
        self.meta = self._load_meta()
        self._validate_layout()
        mmap_mode = "r" if mmap else None
        self._x_struct = np.load(self.root / "graph" / "X_struct.npy", mmap_mode=mmap_mode)
        self._struct_mask = np.load(
            self.root / "graph" / "struct_active_mask.npy", mmap_mode=mmap_mode
        )
        expected_f = int(C.F_STRUCT)
        if self._x_struct.ndim != 3 or self._x_struct.shape[2] != expected_f:
            raise DatasetError(
                f"X_struct expected [T,N,{expected_f}], got {list(self._x_struct.shape)}"
            )
        if self._x_struct.shape[0] != self.meta.t_snapshots:
            raise DatasetError(
                f"X_struct T={self._x_struct.shape[0]} != meta T={self.meta.t_snapshots}"
            )
        if self._x_struct.shape[1] != C.N_NODES:
            raise DatasetError(f"N must be {C.N_NODES}, got {self._x_struct.shape[1]}")
        if self._struct_mask.shape != self._x_struct.shape[:2]:
            raise DatasetError(
                f"struct_active_mask shape {self._struct_mask.shape} "
                f"!= X_struct[:2] {self._x_struct.shape[:2]}"
            )
        self._snap_to_t: Dict[int, int] = {
            sid: i for i, sid in enumerate(self.meta.snapshot_ids)
        }
        self._quarter: Dict[int, str] = {
            sid: q for sid, q in zip(self.meta.snapshot_ids, self.meta.quarter_labels)
        }

        self._part_canonical_start: Dict[Path, int] = {}
        self._canonical_edge_count = 0
        self._build_canonical_edge_offsets()

        self._node_text = None
        self._node_text_mask = None
        self._edge_text = None
        self._edge_text_mask = None
        if self.load_text:
            self._load_text_tensors(mmap_mode)

    def _load_meta(self) -> PackageMeta:
        cfg_path = self.root / "configs" / "smoke_e2e_config.json"
        pkg_path = self.root / "manifests" / "package_manifest.json"
        cal_path = self.root / "graph" / "snapshot_calendar.json"
        if not cfg_path.is_file():
            raise DatasetError(f"missing config: {cfg_path}")
        if not cal_path.is_file():
            raise DatasetError(f"missing calendar: {cal_path}")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cal = json.loads(cal_path.read_text(encoding="utf-8"))
        snaps = cal.get("snapshots") or []
        if not snaps:
            raise DatasetError("snapshot_calendar.snapshots is empty")
        snapshot_ids = tuple(int(s["snapshot_id"]) for s in snaps)
        quarter_labels = tuple(str(s.get("quarter_label") or "") for s in snaps)
        pkg_id = "TDMEC_INPUT"
        if pkg_path.is_file():
            pkg = json.loads(pkg_path.read_text(encoding="utf-8"))
            pkg_id = str(pkg.get("package_name") or pkg_id)
        labels = tuple(
            dict.fromkeys(list(cfg.get("status_labels") or []) + list(_STATUS))
        )
        return PackageMeta(
            package_root=self.root.as_posix(),
            package_id=pkg_id,
            graph_run_id=str(cfg.get("graph_run_id") or ""),
            embedding_run_id=str(cfg.get("embedding_run_id") or ""),
            d_text=int(cfg.get("D_text") or 0),
            t_snapshots=int(cfg.get("T") or len(snapshot_ids)),
            n_nodes=int(cfg.get("N") or C.N_NODES),
            e_canonical=int(cfg.get("E") or 0),
            relation_ids=tuple(int(x) for x in (cfg.get("relation_ids") or (0, 1, 2, 3))),
            status_labels=labels,
            snapshot_ids=snapshot_ids,
            quarter_labels=quarter_labels,
        )

    def _validate_layout(self) -> None:
        required = [
            "graph/X_struct.npy",
            "graph/struct_active_mask.npy",
            "graph/edges",
            "manifests/package_manifest.json",
        ]
        missing = [rel for rel in required if not (self.root / rel).exists()]
        if missing:
            raise DatasetError(f"package missing required paths: {missing}")
        if self.meta.t_snapshots != len(self.meta.snapshot_ids):
            raise DatasetError(
                f"T={self.meta.t_snapshots} != calendar len={len(self.meta.snapshot_ids)}"
            )
        if list(self.meta.relation_ids) != [0, 1, 2, 3]:
            raise DatasetError(f"unexpected relation_ids={self.meta.relation_ids}")

    def _build_canonical_edge_offsets(self) -> None:
        base = self.root / "graph" / "edges"
        parts = sorted(base.glob("**/*.parquet"), key=lambda p: _edge_part_sort_key(base, p))
        offset = 0
        for part in parts:
            n = int(pq.ParquetFile(part).metadata.num_rows)
            self._part_canonical_start[part.resolve()] = offset
            offset += n
        self._canonical_edge_count = offset
        if self.meta.e_canonical and offset != self.meta.e_canonical:
            raise DatasetError(
                f"canonical edge count from parquet walk={offset} "
                f"!= package E={self.meta.e_canonical}"
            )

    def _load_text_tensors(self, mmap_mode: Optional[str]) -> None:
        node_path = self.root / "text_embeddings" / "node_snapshot_embeddings.npy"
        node_mask_path = self.root / "text_embeddings" / "node_text_available_mask.npy"
        edge_path = self.root / "text_embeddings" / "canonical_edge_embeddings.npy"
        edge_mask_path = self.root / "text_embeddings" / "edge_text_available_mask.npy"
        for p in (node_path, node_mask_path, edge_path, edge_mask_path):
            if not p.is_file():
                raise DatasetError(f"load_text=True but missing {p}")
        self._node_text = np.load(node_path, mmap_mode=mmap_mode)
        self._node_text_mask = np.load(node_mask_path, mmap_mode=mmap_mode)
        self._edge_text = np.load(edge_path, mmap_mode=mmap_mode)
        self._edge_text_mask = np.load(edge_mask_path, mmap_mode=mmap_mode)
        if self._node_text.shape[:2] != self._x_struct.shape[:2]:
            raise DatasetError(
                f"node text TN {self._node_text.shape[:2]} != X_struct {self._x_struct.shape[:2]}"
            )
        if int(self._node_text.shape[-1]) != self.meta.d_text:
            raise DatasetError(
                f"node D_text={self._node_text.shape[-1]} != meta D_text={self.meta.d_text}"
            )
        if self._edge_text.shape[0] != self._canonical_edge_count:
            raise DatasetError(
                f"edge text E={self._edge_text.shape[0]} != canonical={self._canonical_edge_count}"
            )
        if self._edge_text.shape[-1] != self.meta.d_text:
            raise DatasetError(
                f"edge D_text={self._edge_text.shape[-1]} != meta D_text={self.meta.d_text}"
            )

    def __len__(self) -> int:
        return len(self.meta.snapshot_ids)

    @property
    def snapshot_ids(self) -> Tuple[int, ...]:
        return self.meta.snapshot_ids

    @property
    def d_text(self) -> int:
        return int(self.meta.d_text)

    def time_index_for(self, snapshot_id: int) -> int:
        try:
            return self._snap_to_t[int(snapshot_id)]
        except KeyError as exc:
            raise DatasetError(f"unknown snapshot_id={snapshot_id}") from exc

    def _edge_part_paths(self, snapshot_id: int, relation_id: int) -> List[Path]:
        d = (
            self.root
            / "graph"
            / "edges"
            / f"snapshot={snapshot_id}"
            / f"relation={relation_id}"
        )
        if not d.is_dir():
            return []
        return sorted(d.glob("part-*.parquet"))

    def _load_edges(
        self, snapshot_id: int
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        srcs: List[np.ndarray] = []
        dsts: List[np.ndarray] = []
        rels: List[np.ndarray] = []
        counts: List[np.ndarray] = []
        weights: List[np.ndarray] = []
        canon: List[np.ndarray] = []
        for rid in self.meta.relation_ids:
            for part in self._edge_part_paths(snapshot_id, rid):
                table = pq.read_table(
                    part,
                    columns=[
                        "src_index",
                        "dst_index",
                        "relation_id",
                        "count_raw",
                        "weight_log1p",
                    ],
                )
                if table.num_rows == 0:
                    continue
                src = table.column("src_index").to_numpy()
                dst = table.column("dst_index").to_numpy()
                rel = table.column("relation_id").to_numpy()
                cnt = table.column("count_raw").to_numpy()
                w = table.column("weight_log1p").to_numpy()
                if rel.size and not np.all(rel == rid):
                    raise DatasetError(
                        f"relation mismatch in {part}: expected {rid}, got unique={np.unique(rel)}"
                    )
                if src.min(initial=0) < 0 or src.max(initial=0) >= C.N_NODES:
                    raise DatasetError(f"src_index out of range in {part}")
                if dst.min(initial=0) < 0 or dst.max(initial=0) >= C.N_NODES:
                    raise DatasetError(f"dst_index out of range in {part}")
                start = self._part_canonical_start[part.resolve()]
                idx = np.arange(start, start + src.shape[0], dtype=np.int64)
                srcs.append(src.astype(np.int64, copy=False))
                dsts.append(dst.astype(np.int64, copy=False))
                rels.append(np.full(src.shape[0], rid, dtype=np.int64))
                counts.append(cnt.astype(np.int64, copy=False))
                weights.append(w.astype(np.float32, copy=False))
                canon.append(idx)

        if not srcs:
            empty = torch.zeros((0,), dtype=torch.long)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            return (
                edge_index,
                empty,
                empty,
                torch.zeros((0,), dtype=torch.float32),
                empty,
            )
        src_a = np.concatenate(srcs)
        dst_a = np.concatenate(dsts)
        edge_index = torch.from_numpy(np.stack([src_a, dst_a], axis=0))
        relation_id = torch.from_numpy(np.concatenate(rels))
        count_raw = torch.from_numpy(np.concatenate(counts))
        weight_log1p = torch.from_numpy(np.concatenate(weights))
        edge_canonical_idx = torch.from_numpy(np.concatenate(canon))
        return edge_index, relation_id, count_raw, weight_log1p, edge_canonical_idx

    def get_snapshot(self, snapshot_id: int) -> SnapshotBatch:
        t = self.time_index_for(snapshot_id)
        x = _as_float32(self._x_struct[t])
        mask = _as_bool_mask(self._struct_mask[t])
        if not torch.isfinite(x).all():
            raise DatasetError(f"NaN/Inf in X_struct at snapshot_id={snapshot_id}")
        edge_index, relation_id, count_raw, weight_log1p, edge_cidx = self._load_edges(
            snapshot_id
        )

        node_text = None
        node_text_mask = None
        edge_text = None
        edge_text_mask = None
        if self.load_text:
            assert self._node_text is not None
            assert self._node_text_mask is not None
            assert self._edge_text is not None
            assert self._edge_text_mask is not None
            node_text = _as_float32(self._node_text[t])
            node_text_mask = _as_bool_mask(self._node_text_mask[t])
            if edge_cidx.numel() == 0:
                edge_text = torch.zeros((0, self.meta.d_text), dtype=torch.float32)
                edge_text_mask = torch.zeros((0,), dtype=torch.bool)
            else:
                idx = edge_cidx.numpy()
                edge_text = _as_float32(self._edge_text[idx])
                edge_text_mask = _as_bool_mask(self._edge_text_mask[idx])
                # Q-MISS: unavailable → exact zero
                if (~edge_text_mask).any():
                    edge_text = edge_text.clone()
                    edge_text[~edge_text_mask] = 0.0
                if (~node_text_mask).any():
                    node_text = node_text.clone()
                    node_text[~node_text_mask] = 0.0

        batch = SnapshotBatch(
            snapshot_id=int(snapshot_id),
            time_index=int(t),
            x_struct=x,
            struct_active_mask=mask,
            edge_index=edge_index,
            relation_id=relation_id,
            count_raw=count_raw,
            weight_log1p=weight_log1p,
            quarter_label=self._quarter.get(int(snapshot_id)),
            node_text=node_text,
            node_text_available_mask=node_text_mask,
            edge_text=edge_text,
            edge_text_available_mask=edge_text_mask,
            edge_canonical_idx=edge_cidx,
            status_labels=_STATUS,
        )
        if self.device is not None:
            batch = self._to_device(batch, self.device)
        return batch

    @staticmethod
    def _to_device(batch: SnapshotBatch, device: torch.device) -> SnapshotBatch:
        def _maybe(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return None if t is None else t.to(device)

        return SnapshotBatch(
            snapshot_id=batch.snapshot_id,
            time_index=batch.time_index,
            x_struct=batch.x_struct.to(device),
            struct_active_mask=batch.struct_active_mask.to(device),
            edge_index=batch.edge_index.to(device),
            relation_id=batch.relation_id.to(device),
            count_raw=batch.count_raw.to(device),
            weight_log1p=batch.weight_log1p.to(device),
            quarter_label=batch.quarter_label,
            node_text=_maybe(batch.node_text),
            node_text_available_mask=_maybe(batch.node_text_available_mask),
            edge_text=_maybe(batch.edge_text),
            edge_text_available_mask=_maybe(batch.edge_text_available_mask),
            edge_canonical_idx=_maybe(batch.edge_canonical_idx),
            status_labels=batch.status_labels,
        )

    def iter_snapshots(
        self, snapshot_ids: Optional[Sequence[int]] = None
    ) -> List[SnapshotBatch]:
        ids = list(snapshot_ids) if snapshot_ids is not None else list(self.snapshot_ids)
        return [self.get_snapshot(sid) for sid in ids]


__all__ = ["TDMECInputDataset", "DatasetError"]
