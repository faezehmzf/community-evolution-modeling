"""Deterministic mock text encoder for engineering validation only.

The mock never approximates or substitutes for Qwen3.  It provides stable,
finite, unit-normalized vectors so batching, persistence, resume, and pooling
code can be tested before any real-model download or inference is authorized.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Protocol, Sequence

import numpy as np

from tdmec.hashing import hash_canonical

from .eligibility import EligibleTextUnit, cleaned_text_content_hash


class EncoderError(RuntimeError):
    """Raised when encoder input or output violates its contract."""


@dataclass(frozen=True)
class EncoderMetadata:
    """Path-free encoder identity used by writer compatibility checks."""

    model_name: str
    model_revision: str
    tokenizer_revision: str
    dimension: int
    output_dtype: str = "float32"
    unit_normalized: bool = True
    instruction_hash: str = "none"
    backend: str = "deterministic_mock"

    def __post_init__(self) -> None:
        if not self.model_name or not self.model_revision or not self.tokenizer_revision:
            raise ValueError("encoder name and revisions must be non-empty")
        if (
            not isinstance(self.dimension, int)
            or isinstance(self.dimension, bool)
            or not 1 <= self.dimension <= 16384
        ):
            raise ValueError("encoder dimension must be in [1, 16384]")
        if self.output_dtype != "float32":
            raise ValueError("Subtask 4 mock output dtype must be float32")
        if not self.instruction_hash:
            raise ValueError("instruction_hash must be non-empty")

    def payload(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "dimension": self.dimension,
            "output_dtype": self.output_dtype,
            "unit_normalized": self.unit_normalized,
            "instruction_hash": self.instruction_hash,
            "backend": self.backend,
        }

    @property
    def model_hash(self) -> str:
        return hash_canonical({"encoder": self.payload()})


class TextEncoder(Protocol):
    """Minimal backend-neutral encoder interface."""

    metadata: EncoderMetadata

    def encode(self, units: Sequence[EligibleTextUnit]) -> np.ndarray: ...


class DeterministicMockEncoder:
    """Generate reproducible vectors from content hashes via SHA-256 expansion."""

    def __init__(self, dimension: int = 16, *, instruction_hash: str = "none") -> None:
        self.metadata = EncoderMetadata(
            model_name="tdmec-deterministic-mock",
            model_revision="sha256-counter-v1",
            tokenizer_revision="not-applicable",
            dimension=dimension,
            instruction_hash=instruction_hash,
        )

    def _encode_content_hash(self, content_hash: str) -> np.ndarray:
        seed = (
            "tdmec-mock-vector-v1\x00"
            + self.metadata.model_hash
            + "\x00"
            + content_hash
        ).encode("ascii")
        needed = self.metadata.dimension
        values: list[np.ndarray] = []
        produced = 0
        counter = 0
        while produced < needed:
            digest = hashlib.sha256(seed + counter.to_bytes(8, "big")).digest()
            block = np.frombuffer(digest, dtype=">u4").astype(np.float64)
            block = (block / float(2**32 - 1)) * 2.0 - 1.0
            values.append(block)
            produced += block.size
            counter += 1
        vector64 = np.concatenate(values)[:needed]
        norm = float(np.linalg.norm(vector64))
        if not np.isfinite(norm) or norm <= 0.0:
            raise EncoderError("mock encoder produced an invalid pre-normalization norm")
        vector = (vector64 / norm).astype(np.float32)
        if not np.all(np.isfinite(vector)):
            raise EncoderError("mock encoder produced NaN or Inf")
        return vector

    def encode(self, units: Sequence[EligibleTextUnit]) -> np.ndarray:
        """Encode a bounded sequence without logging or persisting input text."""

        if not units:
            return np.empty((0, self.metadata.dimension), dtype=np.float32)
        vectors = np.empty((len(units), self.metadata.dimension), dtype=np.float32)
        cache: Dict[str, np.ndarray] = {}
        for index, unit in enumerate(units):
            if cleaned_text_content_hash(unit.cleaned_text) != unit.content_hash:
                raise EncoderError("eligible unit content hash drifted before encoding")
            vector = cache.get(unit.content_hash)
            if vector is None:
                vector = self._encode_content_hash(unit.content_hash)
                cache[unit.content_hash] = vector
            vectors[index] = vector
        norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
        if not np.all(np.isfinite(vectors)) or not np.allclose(
            norms, 1.0, rtol=0.0, atol=1e-6
        ):
            raise EncoderError("mock encoder output validation failed")
        return vectors


__all__ = [
    "DeterministicMockEncoder",
    "EncoderError",
    "EncoderMetadata",
    "TextEncoder",
]
