"""TDMEC Phase 1: typed contracts, schemas, hashing, and validators.

This package intentionally excludes Dataset A/B ingestion, embeddings,
model forward passes, training, evaluation, baselines, and ablations.
"""
from __future__ import annotations

from tdmec import constants
from tdmec.hashing import hash_canonical, hash_config, sha256_bytes, sha256_file
from tdmec.unresolved import ResolutionGate, UnresolvedValue

__all__ = [
    "constants",
    "hash_canonical",
    "hash_config",
    "sha256_bytes",
    "sha256_file",
    "ResolutionGate",
    "UnresolvedValue",
]

__version__ = "0.1.0-phase1"
