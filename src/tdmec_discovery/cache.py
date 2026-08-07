"""Download cache with resume + change-detection.

Avoids downloading an unchanged file twice by recording (size, sha256) in a
small JSON index alongside the cache. A file is considered already-present when
a cached copy exists and, if an expected size is known, the sizes match.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .hashing import sha256_file
from .sources import BaseSource, RemoteFile


class DownloadCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "_cache_index.json"
        self.index = self._load_index()

    def _load_index(self) -> dict:
        if self.index_path.is_file():
            try:
                return json.loads(self.index_path.read_text())
            except Exception:
                return {}
        return {}

    def _save_index(self) -> None:
        tmp = self.index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.index, indent=1))
        tmp.replace(self.index_path)

    def local_path(self, rf: RemoteFile) -> Path:
        safe = rf.relative_id.replace("/", "__")
        return self.root / safe

    def is_cached(self, rf: RemoteFile) -> bool:
        p = self.local_path(rf)
        if not p.is_file():
            return False
        if rf.size is not None and p.stat().st_size != rf.size:
            return False
        return True

    def get(
        self,
        source: BaseSource,
        rf: RemoteFile,
        compute_hash: bool = True,
        force: bool = False,
    ) -> dict:
        """Ensure ``rf`` is present locally; return a record with path/size/sha256."""
        p = self.local_path(rf)
        reused = False
        if not force and self.is_cached(rf):
            reused = True
        else:
            source.download(rf, p)
        size = p.stat().st_size
        entry = self.index.get(rf.relative_id, {})
        sha = entry.get("sha256")
        if compute_hash and (not reused or not sha or entry.get("size") != size):
            sha = sha256_file(p)
        self.index[rf.relative_id] = {"size": size, "sha256": sha}
        self._save_index()
        return {"path": p, "size": size, "sha256": sha, "reused": reused}

    def evict(self, rf: RemoteFile) -> None:
        """Delete the local cached copy (source is never touched)."""
        p = self.local_path(rf)
        if p.is_file():
            p.unlink()
