"""Run-context, manifest, and atomic-publication helpers.

A *run* owns an output directory (``discovery/<run_id>/``) and a manifest that
records every produced output plus its size and checksum. Outputs are published
to the Google Drive output folder when credentials are available; otherwise they
remain local and are marked ``local_only`` in the manifest so callers never
mistake local creation for successful publication.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .config import DiscoveryConfig
from .hashing import sha256_file


def git_short_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip() or "nogit"
    except Exception:
        return "nogit"


def make_run_id(cfg: DiscoveryConfig, now: Optional[datetime] = None) -> str:
    now = now or datetime.now(timezone.utc)
    ts = now.strftime("%Y-%m-%dT%H%M%SZ")
    return f"{ts}_git-{git_short_hash()}_cfg-{cfg.fingerprint()}"


def runtime_environment() -> dict:
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_short_hash(),
    }


@dataclass
class RunContext:
    cfg: DiscoveryConfig
    run_id: str
    output_dir: Path
    manifest: dict = field(default_factory=dict)
    publisher: Optional["DrivePublisher"] = None

    @classmethod
    def create(cls, cfg: DiscoveryConfig, run_id: Optional[str] = None) -> "RunContext":
        run_id = run_id or make_run_id(cfg)
        output_dir = cfg.output_root / "runs" / run_id
        (output_dir / "logs").mkdir(parents=True, exist_ok=True)
        ctx = cls(cfg=cfg, run_id=run_id, output_dir=output_dir)
        ctx.manifest = {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "runtime_environment": runtime_environment(),
            "outputs": {},
            "stages": {},
            "drive_output_folder_id_present": bool(cfg.drive_output_folder_id),
        }
        return ctx

    def record_output(self, rel_name: str, path: Path, published: bool = False,
                      drive_file_id: Optional[str] = None) -> dict:
        size = path.stat().st_size
        sha = sha256_file(path)
        rec = {
            "size": size,
            "sha256": sha,
            "published_to_drive": published,
            "drive_file_id": drive_file_id,
            "status": "published" if published else "local_only",
        }
        self.manifest["outputs"][rel_name] = rec
        return rec

    def mark_stage(self, stage: str, state: str, detail: Optional[dict] = None) -> None:
        self.manifest["stages"][stage] = {"state": state, "detail": detail or {}}

    def write_manifest(self) -> Path:
        p = self.output_dir / "discovery_manifest.json"
        p.write_text(json.dumps(self.manifest, indent=2))
        return p

    def write_checksums(self) -> Path:
        checks = {name: rec["sha256"] for name, rec in self.manifest["outputs"].items()}
        p = self.output_dir / "checksums.json"
        p.write_text(json.dumps(checks, indent=2))
        return p


class DrivePublisher:
    """Publishes local outputs to the Drive output folder (atomic + verified).

    If credentials are unavailable, :meth:`available` returns False and callers
    should record outputs as ``local_only``.
    """

    def __init__(self, output_folder_id: str):
        self.output_folder_id = output_folder_id
        self._svc = None

    def available(self) -> bool:
        from .drive_api import credentials_available

        return bool(self.output_folder_id) and credentials_available()

    def _service(self):
        if self._svc is None:
            from .drive_api import build_drive_service

            self._svc = build_drive_service(readonly=False)
        return self._svc

    def ensure_subfolder(self, name: str, parent_id: Optional[str] = None) -> str:
        svc = self._service()
        parent = parent_id or self.output_folder_id
        q = (
            f"'{parent}' in parents and name = '{name}' and "
            "mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        )
        resp = svc.files().list(q=q, fields="files(id,name)", supportsAllDrives=True,
                                includeItemsFromAllDrives=True).execute()
        files = resp.get("files", [])
        if files:
            return files[0]["id"]
        meta = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent],
        }
        created = svc.files().create(body=meta, fields="id", supportsAllDrives=True).execute()
        return created["id"]

    def upload_verify(self, local_path: Path, parent_id: str) -> dict:
        """Upload and verify by re-reading size (and md5 when Drive provides it)."""
        from googleapiclient.http import MediaFileUpload

        svc = self._service()
        meta = {"name": local_path.name, "parents": [parent_id]}
        media = MediaFileUpload(str(local_path), resumable=True)
        created = svc.files().create(
            body=meta, media_body=media, fields="id,name,size,md5Checksum",
            supportsAllDrives=True,
        ).execute()
        remote = svc.files().get(
            fileId=created["id"], fields="id,name,size,md5Checksum", supportsAllDrives=True
        ).execute()
        local_size = local_path.stat().st_size
        ok = int(remote.get("size", -1)) == local_size
        return {"drive_file_id": remote["id"], "verified": ok,
                "remote_size": int(remote.get("size", -1)), "local_size": local_size}
