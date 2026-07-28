"""Checkpoint / resume utilities for Phase 2 diagnostics."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from tdmec_diagnostics.io_utils import atomic_write_json


@dataclass
class FileCheckpoint:
    source_file: str
    source_checksum: Optional[str]
    chunks_completed: List[int] = field(default_factory=list)
    rows_inspected: int = 0
    rows_accepted: int = 0
    rows_rejected: int = 0
    complete: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_file": self.source_file,
            "source_checksum": self.source_checksum,
            "chunks_completed": list(self.chunks_completed),
            "rows_inspected": self.rows_inspected,
            "rows_accepted": self.rows_accepted,
            "rows_rejected": self.rows_rejected,
            "complete": self.complete,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FileCheckpoint":
        return cls(
            source_file=str(d["source_file"]),
            source_checksum=d.get("source_checksum"),
            chunks_completed=[int(x) for x in d.get("chunks_completed", [])],
            rows_inspected=int(d.get("rows_inspected", 0)),
            rows_accepted=int(d.get("rows_accepted", 0)),
            rows_rejected=int(d.get("rows_rejected", 0)),
            complete=bool(d.get("complete", False)),
        )


@dataclass
class DiagnosticsCheckpointStore:
    """File-level progress records for resumable diagnostics."""

    root: Path
    config_hash: str
    files: Dict[str, FileCheckpoint] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self.root / "diagnostics_checkpoint.json"

    def load(self) -> "DiagnosticsCheckpointStore":
        if self.path.is_file():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            stored_hash = data.get("config_hash")
            if stored_hash != self.config_hash:
                raise ConfigIncompatibleError(
                    f"checkpoint config_hash mismatch: stored={stored_hash} "
                    f"current={self.config_hash}"
                )
            self.files = {
                k: FileCheckpoint.from_dict(v)
                for k, v in data.get("files", {}).items()
            }
        return self

    def save(self) -> None:
        payload = {
            "schema_version": "tdmec-phase2-checkpoint-v1",
            "config_hash": self.config_hash,
            "files": {k: v.to_dict() for k, v in sorted(self.files.items())},
        }
        atomic_write_json(self.path, payload)

    def completed_chunks(self, source_file: str) -> Set[int]:
        cp = self.files.get(source_file)
        if cp is None:
            return set()
        return set(cp.chunks_completed)

    def reset_file(self, source_file: str, source_checksum: Optional[str] = None) -> None:
        """Reset progress for a file before a full recompute pass."""
        self.files[source_file] = FileCheckpoint(
            source_file=source_file, source_checksum=source_checksum
        )

    def mark_chunk(
        self,
        source_file: str,
        chunk_index: int,
        *,
        rows_inspected: int,
        rows_accepted: int,
        rows_rejected: int,
        source_checksum: Optional[str] = None,
    ) -> None:
        cp = self.files.get(source_file)
        if cp is None:
            cp = FileCheckpoint(source_file=source_file, source_checksum=source_checksum)
            self.files[source_file] = cp
        if source_checksum is not None:
            if cp.source_checksum is None:
                cp.source_checksum = source_checksum
            elif cp.source_checksum != source_checksum:
                raise InputChecksumDriftError(
                    f"source checksum drift for {source_file}"
                )
        if chunk_index not in cp.chunks_completed:
            cp.chunks_completed.append(chunk_index)
            cp.chunks_completed.sort()
            cp.rows_inspected += rows_inspected
            cp.rows_accepted += rows_accepted
            cp.rows_rejected += rows_rejected

    def mark_file_complete(self, source_file: str) -> None:
        cp = self.files.get(source_file)
        if cp is None:
            cp = FileCheckpoint(source_file=source_file, source_checksum=None)
            self.files[source_file] = cp
        cp.complete = True

    def is_complete(self, expected_files: List[str]) -> bool:
        for f in expected_files:
            cp = self.files.get(f)
            if cp is None or not cp.complete:
                return False
        return True

    def resume_state(self) -> Dict[str, Any]:
        return {
            "n_files_tracked": len(self.files),
            "files_complete": sorted(
                k for k, v in self.files.items() if v.complete
            ),
            "files_incomplete": sorted(
                k for k, v in self.files.items() if not v.complete
            ),
            "total_rows_inspected": sum(v.rows_inspected for v in self.files.values()),
            "total_rows_accepted": sum(v.rows_accepted for v in self.files.values()),
            "total_rows_rejected": sum(v.rows_rejected for v in self.files.values()),
        }


class ConfigIncompatibleError(RuntimeError):
    pass


class InputChecksumDriftError(RuntimeError):
    pass


class IncompleteRunError(RuntimeError):
    pass
