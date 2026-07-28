"""Deterministic hashing utilities for Phase 1 contracts.

Scientific-content hashes must be path-independent and insertion-order stable.
Absolute local paths and wall-clock timestamps must not enter reproducibility hashes.

Note: ``sha256_file`` / ``sha256_bytes`` intentionally mirror the small helpers in
``tdmec_discovery.hashing`` so the Phase 1 ``tdmec`` package remains independently
importable without coupling contract code to discovery I/O.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tdmec.constants import HASH_ALGORITHM

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a declared dependency
    np = None  # type: ignore[assignment]


def _require_sha256() -> str:
    if HASH_ALGORITHM != "sha256":
        raise ValueError(f"unsupported hash algorithm: {HASH_ALGORITHM}")
    return HASH_ALGORITHM


def sha256_hex(data: bytes) -> str:
    _require_sha256()
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Streaming file checksum (constant memory). Algorithm is SHA-256."""
    _require_sha256()
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return sha256_hex(data)


def _is_numpy_scalar(obj: Any) -> bool:
    if np is None:
        return False
    return isinstance(obj, getattr(np, "generic", ()))


def canonicalize(obj: Any) -> Any:
    """Recursively canonicalize for stable JSON serialization.

    Rules:
    - dict / Mapping keys sorted lexicographically (insertion order ignored)
    - tuples become lists (order preserved; intentional sequence semantics)
    - set / frozenset become sorted lists (order-independent)
    - Enum -> ``.value`` then canonicalize
    - dataclass -> ``asdict`` then canonicalize
    - NumPy scalars -> native Python scalars
    - NumPy arrays are rejected (caller must pass an explicit ordered structure)
    - floats: reject NaN/Inf
    - Path objects rejected (machine-specific paths must not enter hashes)
    """
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, Enum):
        return canonicalize(obj.value)
    if isinstance(obj, Path):
        raise ValueError(
            "Path objects must not enter scientific-content hashes; "
            "use relative logical names only"
        )
    if _is_numpy_scalar(obj):
        return canonicalize(obj.item())
    if np is not None and isinstance(obj, np.ndarray):
        raise TypeError(
            "NumPy arrays must not enter scientific hashes directly; "
            "pass an explicit ordered list/tuple structure"
        )
    # bool is already handled; plain ints (not bool)
    if isinstance(obj, int) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError("NaN/Inf floats are forbidden in scientific hashes")
        return float(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return canonicalize(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): canonicalize(obj[k]) for k in sorted(obj.keys(), key=str)}
    if isinstance(obj, (set, frozenset)):
        # Deterministic: sort by canonical JSON of each element
        items = [canonicalize(x) for x in obj]
        items.sort(key=lambda x: json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False))
        return items
    if isinstance(obj, (list, tuple)):
        return [canonicalize(x) for x in obj]
    raise TypeError(f"unsupported type for canonicalize: {type(obj)!r}")


def canonical_json_bytes(obj: Any) -> bytes:
    """Stable UTF-8 JSON with sorted keys and compact separators."""
    canon = canonicalize(obj)
    return json.dumps(
        canon,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def hash_canonical(obj: Any) -> str:
    """SHA-256 of canonical JSON serialization."""
    return sha256_hex(canonical_json_bytes(obj))


def hash_config(config_dict: Mapping[str, Any]) -> str:
    """Deterministic configuration hash (path-independent scientific content)."""
    return hash_canonical(dict(config_dict))


def hash_relation_mapping(relation_to_id: Mapping[str, int]) -> str:
    return hash_canonical({"relation_to_id": dict(relation_to_id)})


def hash_node_order(node_indices: Sequence[int]) -> str:
    """Hash the canonical node order (index sequence)."""
    return hash_canonical({"node_order": [int(i) for i in node_indices]})


def hash_feature_order(feature_names: Sequence[str]) -> str:
    return hash_canonical({"feature_order": list(feature_names)})


def hash_edge_order(
    edges: Iterable[tuple[int, int, int, int]],
) -> str:
    """Hash canonical edge order: each key (snapshot_id, relation_id, src, tgt).

    Caller must supply edges already in the declared canonical sort order.
    The hash covers the ordered sequence, not a set.
    """
    return hash_canonical(
        {
            "edge_order": [
                {
                    "snapshot_id": int(s),
                    "relation_id": int(r),
                    "source_idx": int(src),
                    "target_idx": int(tgt),
                }
                for s, r, src, tgt in edges
            ]
        }
    )


def hash_manifest_payload(payload: Mapping[str, Any]) -> str:
    """Hash a manifest scientific payload (exclude absolute paths)."""
    assert_no_absolute_paths(payload, context="manifest_payload")
    return hash_canonical(dict(payload))


def looks_like_absolute_local_path(value: str) -> bool:
    """Heuristic privacy/determinism guard for absolute local filesystem paths."""
    if not isinstance(value, str) or not value:
        return False
    lowered = value.lower()
    if value.startswith("/") or (len(value) >= 3 and value[1:3] in (":\\", ":/")):
        markers = (
            "/workspace",
            "/home/",
            "/users/",
            "/content/",
            "/tmp/",
            "/var/",
            "c:\\",
            "d:\\",
            "\\users\\",
        )
        if any(m in lowered for m in markers):
            return True
        # POSIX absolute path with home-like or drive-like segments
        if value.startswith("/"):
            return True
    return False


def assert_no_absolute_paths(obj: Any, *, context: str = "payload") -> None:
    """Hard-fail if absolute filesystem paths appear in a hashable payload."""
    if isinstance(obj, str):
        if looks_like_absolute_local_path(obj):
            raise ValueError(f"absolute local path forbidden in {context}: redacted")
        return
    if isinstance(obj, Path):
        raise ValueError(f"Path object forbidden in {context}")
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            assert_no_absolute_paths(k, context=context)
            assert_no_absolute_paths(v, context=context)
        return
    if isinstance(obj, (list, tuple, set, frozenset)):
        for x in obj:
            assert_no_absolute_paths(x, context=context)
