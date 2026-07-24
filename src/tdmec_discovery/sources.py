"""Source adapters for enumerating and downloading dataset files.

A *source* is described by a string with an optional scheme prefix:

    local:/path/to/dir          local filesystem directory
    gdrive-anon:<id|url>        public Google Drive folder (anonymous, gdown)
    gdrive-api:<id>            Google Drive folder via authenticated API

A bare path or bare Drive id/URL is auto-detected.

All adapters are STRICTLY READ-ONLY: they never create, rename, move, delete or
re-upload anything in the source. Downloads always target a caller-provided
local destination outside the source.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class RemoteFile:
    """A single enumerated source file (metadata only until downloaded)."""

    relative_id: str  # stable relative identifier within the source
    name: str  # sanitized basename
    ext: str  # lowercased extension incl. dot, e.g. ".xlsx"
    size: Optional[int] = None  # reported size in bytes, if known pre-download
    modified: Optional[str] = None  # ISO8601 modification time, if available
    ref: str = ""  # opaque adapter handle (drive id or local path) - not committed
    extra: Dict[str, object] = field(default_factory=dict)


_DRIVE_ID_RE = re.compile(r"[-\w]{25,}")


def _parse_source(source: str) -> tuple[str, str]:
    """Return (scheme, target) for a source string, auto-detecting when needed."""
    if source.startswith(("local:", "gdrive-anon:", "gdrive-api:")):
        scheme, _, target = source.partition(":")
        return scheme, target
    # Auto-detect: an existing path -> local; a drive-looking id/url -> anon.
    if os.path.exists(source) or source.startswith(("/", "./", "../", "~")):
        return "local", source
    if "drive.google.com" in source or _DRIVE_ID_RE.fullmatch(source):
        return "gdrive-anon", source
    # Default to local; caller will get a clear error if it does not exist.
    return "local", source


def extract_drive_folder_id(target: str) -> str:
    """Extract a Drive folder id from a raw id or a share URL."""
    if "drive.google.com" in target:
        m = re.search(r"/folders/([-\w]{25,})", target)
        if m:
            return m.group(1)
        m = re.search(r"[?&]id=([-\w]{25,})", target)
        if m:
            return m.group(1)
    return target


def sanitize_name(name: str) -> str:
    """Return a safe basename with no directory components or private prefixes."""
    return os.path.basename(name.replace("\\", "/"))


class BaseSource:
    scheme = "base"

    def list_files(self) -> List[RemoteFile]:  # pragma: no cover - interface
        raise NotImplementedError

    def download(self, rf: RemoteFile, dest: Path) -> Path:  # pragma: no cover
        raise NotImplementedError


class LocalSource(BaseSource):
    """Adapter for an ordinary local filesystem directory (recursive)."""

    scheme = "local"

    def __init__(self, root: str):
        self.root = Path(root).expanduser()

    def list_files(self) -> List[RemoteFile]:
        if not self.root.is_dir():
            raise FileNotFoundError(f"local source directory not found: {self.root}")
        out: List[RemoteFile] = []
        for p in sorted(self.root.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(self.root).as_posix()
            st = p.stat()
            out.append(
                RemoteFile(
                    relative_id=rel,
                    name=sanitize_name(p.name),
                    ext=p.suffix.lower(),
                    size=st.st_size,
                    modified=_iso(st.st_mtime),
                    ref=str(p),
                )
            )
        return out

    def download(self, rf: RemoteFile, dest: Path) -> Path:
        # Read-only: copy the bytes to dest (never touch the source file).
        import shutil

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(rf.ref, dest)
        return dest


class GDriveAnonSource(BaseSource):
    """Anonymous (public link) Google Drive folder via gdown. Read-only."""

    scheme = "gdrive-anon"

    def __init__(self, target: str):
        self.folder_id = extract_drive_folder_id(target)

    def _folder_url(self) -> str:
        return f"https://drive.google.com/drive/folders/{self.folder_id}"

    def list_files(self) -> List[RemoteFile]:
        import gdown

        entries = gdown.download_folder(
            url=self._folder_url(), skip_download=True, quiet=True, use_cookies=False
        )
        if entries is None:
            raise RuntimeError(
                "gdown could not enumerate the folder (not public, too many files, "
                "or access denied)."
            )
        out: List[RemoteFile] = []
        for e in entries:
            rel = e.path
            name = sanitize_name(rel)
            out.append(
                RemoteFile(
                    relative_id=rel,
                    name=name,
                    ext=Path(name).suffix.lower(),
                    size=None,  # gdown listing does not expose sizes
                    ref=e.id,
                )
            )
        return out

    def download(self, rf: RemoteFile, dest: Path) -> Path:
        import gdown

        dest.parent.mkdir(parents=True, exist_ok=True)
        result = gdown.download(id=rf.ref, output=str(dest), quiet=True)
        if not result or not dest.exists():
            raise RuntimeError(f"gdown failed to download file id (ref hidden)")
        return dest


class GDriveApiSource(BaseSource):
    """Authenticated Google Drive folder via the API. Read-only.

    Requires ``GOOGLE_APPLICATION_CREDENTIALS`` (service account) or ADC.
    """

    scheme = "gdrive-api"

    def __init__(self, target: str):
        self.folder_id = extract_drive_folder_id(target)
        self._service = None

    def _svc(self):
        if self._service is None:
            from .drive_api import build_drive_service

            self._service = build_drive_service(readonly=True)
        return self._service

    def list_files(self) -> List[RemoteFile]:
        svc = self._svc()
        out: List[RemoteFile] = []
        page_token = None
        q = f"'{self.folder_id}' in parents and trashed = false"
        while True:
            resp = (
                svc.files()
                .list(
                    q=q,
                    fields="nextPageToken, files(id,name,size,modifiedTime,mimeType)",
                    pageSize=1000,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                    pageToken=page_token,
                )
                .execute()
            )
            for f in resp.get("files", []):
                name = sanitize_name(f["name"])
                out.append(
                    RemoteFile(
                        relative_id=name,
                        name=name,
                        ext=Path(name).suffix.lower(),
                        size=int(f["size"]) if f.get("size") else None,
                        modified=f.get("modifiedTime"),
                        ref=f["id"],
                        extra={"mimeType": f.get("mimeType", "")},
                    )
                )
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
        return out

    def download(self, rf: RemoteFile, dest: Path) -> Path:
        from googleapiclient.http import MediaIoBaseDownload

        svc = self._svc()
        dest.parent.mkdir(parents=True, exist_ok=True)
        req = svc.files().get_media(fileId=rf.ref, supportsAllDrives=True)
        with open(dest, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, req)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        return dest


def build_source(source: str) -> BaseSource:
    """Factory: build the correct adapter for a source string."""
    scheme, target = _parse_source(source)
    if scheme == "local":
        return LocalSource(target)
    if scheme == "gdrive-anon":
        return GDriveAnonSource(target)
    if scheme == "gdrive-api":
        return GDriveApiSource(target)
    raise ValueError(f"unknown source scheme: {scheme}")


def _iso(ts: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
