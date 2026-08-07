"""Train-time structural feature scaling (QHP-02).

Primary: robust per-feature median/IQR scaling fitted on training snapshots only,
using structurally active rows. Inactive rows are forced to the exact zero vector
after transform. The on-disk raw ``X_struct`` artifact is never modified.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from tdmec import constants as C

_STATUS = (
    "PROVISIONAL_SMOKE_ONLY",
    "ENGINEERING_VALIDATION",
    "NOT_FOR_FINAL_THESIS_CONCLUSIONS",
)


class ScalingError(RuntimeError):
    pass


@dataclass
class ScalerState:
    median: np.ndarray  # [F]
    iqr: np.ndarray  # [F]
    scale: np.ndarray  # [F]  (== iqr, with zeros replaced by 1)
    f_struct: int
    train_snapshot_ids: Tuple[int, ...]
    n_active_rows: int
    method: str = "robust_median_iqr"
    status_labels: Tuple[str, ...] = _STATUS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "f_struct": int(self.f_struct),
            "train_snapshot_ids": list(self.train_snapshot_ids),
            "n_active_rows": int(self.n_active_rows),
            "median": self.median.astype(np.float64).tolist(),
            "iqr": self.iqr.astype(np.float64).tolist(),
            "scale": self.scale.astype(np.float64).tolist(),
            "status_labels": list(self.status_labels),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ScalerState":
        return cls(
            median=np.asarray(payload["median"], dtype=np.float64),
            iqr=np.asarray(payload["iqr"], dtype=np.float64),
            scale=np.asarray(payload["scale"], dtype=np.float64),
            f_struct=int(payload["f_struct"]),
            train_snapshot_ids=tuple(int(x) for x in payload["train_snapshot_ids"]),
            n_active_rows=int(payload["n_active_rows"]),
            method=str(payload.get("method") or "robust_median_iqr"),
            status_labels=tuple(payload.get("status_labels") or _STATUS),
        )


class RobustFeatureScaler:
    """Median/IQR robust scaler (QHP-02 primary experimental default)."""

    def __init__(self, state: Optional[ScalerState] = None) -> None:
        self.state = state

    @property
    def is_fitted(self) -> bool:
        return self.state is not None

    def fit(
        self,
        x_struct: np.ndarray | torch.Tensor,
        struct_active_mask: np.ndarray | torch.Tensor,
        *,
        train_snapshot_ids: Sequence[int],
        time_index_for: Optional[Mapping[int, int]] = None,
    ) -> "RobustFeatureScaler":
        """Fit on active rows from training snapshots only.

        Args:
            x_struct: ``[T, N, F]`` raw features (mmap/ndarray/tensor).
            struct_active_mask: ``[T, N]`` bool mask aligned to ``x_struct``.
            train_snapshot_ids: snapshot ids used for fit (no val/test).
            time_index_for: optional map snapshot_id → time index; if None,
                assumes ``snapshot_id == time_index``.
        """
        x = _as_numpy(x_struct)
        m = _as_numpy(struct_active_mask).astype(bool, copy=False)
        if x.ndim != 3:
            raise ScalingError(f"x_struct must be [T,N,F], got {x.shape}")
        if m.shape != x.shape[:2]:
            raise ScalingError(f"mask shape {m.shape} != x[:2] {x.shape[:2]}")
        f = int(x.shape[2])
        if f != C.F_STRUCT:
            raise ScalingError(f"F_struct must be {C.F_STRUCT}, got {f}")
        if not train_snapshot_ids:
            raise ScalingError("train_snapshot_ids must be non-empty")

        rows = []
        for sid in train_snapshot_ids:
            t = int(time_index_for[sid]) if time_index_for is not None else int(sid)
            if t < 0 or t >= x.shape[0]:
                raise ScalingError(f"time index {t} out of range for snapshot {sid}")
            active = m[t]
            if not active.any():
                continue
            rows.append(np.asarray(x[t][active], dtype=np.float64))
        if not rows:
            raise ScalingError("no active training rows to fit scaler")
        stacked = np.concatenate(rows, axis=0)
        median = np.median(stacked, axis=0)
        q25 = np.percentile(stacked, 25, axis=0)
        q75 = np.percentile(stacked, 75, axis=0)
        iqr = q75 - q25
        scale = iqr.copy()
        scale[scale < 1e-12] = 1.0
        self.state = ScalerState(
            median=median.astype(np.float64),
            iqr=iqr.astype(np.float64),
            scale=scale.astype(np.float64),
            f_struct=f,
            train_snapshot_ids=tuple(int(s) for s in train_snapshot_ids),
            n_active_rows=int(stacked.shape[0]),
        )
        return self

    def transform(
        self,
        x_struct: np.ndarray | torch.Tensor,
        struct_active_mask: np.ndarray | torch.Tensor,
        *,
        as_torch: bool = True,
    ) -> np.ndarray | torch.Tensor:
        """Apply frozen scaler; force inactive rows to exact zeros."""
        if self.state is None:
            raise ScalingError("scaler is not fitted")
        x = _as_numpy(x_struct).astype(np.float32, copy=True)
        m = _as_numpy(struct_active_mask).astype(bool, copy=False)
        if x.ndim == 2:
            # single snapshot [N, F]
            if m.ndim != 1 or m.shape[0] != x.shape[0]:
                raise ScalingError(f"mask shape {m.shape} incompatible with {x.shape}")
            out = (x.astype(np.float64) - self.state.median) / self.state.scale
            out = out.astype(np.float32, copy=False)
            out[~m] = 0.0
        elif x.ndim == 3:
            if m.shape != x.shape[:2]:
                raise ScalingError(f"mask shape {m.shape} != x[:2] {x.shape[:2]}")
            out = (x.astype(np.float64) - self.state.median) / self.state.scale
            out = out.astype(np.float32, copy=False)
            out[~m] = 0.0
        else:
            raise ScalingError(f"unsupported x_struct ndim={x.ndim}")
        if not np.all(np.isfinite(out)):
            raise ScalingError("non-finite values after scaling")
        if as_torch:
            return torch.from_numpy(np.array(out, copy=True))
        return out

    def fit_transform_snapshot(
        self,
        x_struct_tnf: np.ndarray | torch.Tensor,
        mask_tn: np.ndarray | torch.Tensor,
        *,
        train_snapshot_ids: Sequence[int],
        snapshot_id: int,
        time_index_for: Optional[Mapping[int, int]] = None,
    ) -> torch.Tensor:
        """Convenience: fit (if needed) then transform one snapshot."""
        if not self.is_fitted:
            self.fit(
                x_struct_tnf,
                mask_tn,
                train_snapshot_ids=train_snapshot_ids,
                time_index_for=time_index_for,
            )
        t = int(time_index_for[snapshot_id]) if time_index_for is not None else int(snapshot_id)
        x_t = _as_numpy(x_struct_tnf)[t]
        m_t = _as_numpy(mask_tn)[t]
        return self.transform(x_t, m_t, as_torch=True)  # type: ignore[return-value]

    def save(self, path: str | Path) -> None:
        if self.state is None:
            raise ScalingError("cannot save unfitted scaler")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RobustFeatureScaler":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(ScalerState.from_dict(payload))


def _as_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


__all__ = ["RobustFeatureScaler", "ScalerState", "ScalingError"]
