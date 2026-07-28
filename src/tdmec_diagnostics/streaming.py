"""Chunked / streaming helpers for bounded-memory diagnostics."""
from __future__ import annotations

from typing import Iterable, Iterator, List, Sequence, TypeVar

T = TypeVar("T")


def iter_chunks(items: Iterable[T], chunk_size: int) -> Iterator[List[T]]:
    """Yield lists of at most *chunk_size* items from *items*."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    buf: List[T] = []
    for item in items:
        buf.append(item)
        if len(buf) >= chunk_size:
            yield buf
            buf = []
    if buf:
        yield buf


def stable_sorted_keys(mapping_keys: Iterable[str]) -> List[str]:
    return sorted(str(k) for k in mapping_keys)


def file_progress_key(source_file: str, chunk_index: int) -> str:
    return f"{source_file}::chunk={chunk_index:06d}"
