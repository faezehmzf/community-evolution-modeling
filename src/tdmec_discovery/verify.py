"""Phase 0 access verification and output-write verification."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .cache import DownloadCache
from .config import DiscoveryConfig
from .hashing import sha256_file
from .sources import build_source

_ZIP_MAGIC = b"PK\x03\x04"
_HTML_MARKERS = (b"<!doctype html", b"<html", b"<!DOCTYPE HTML")


def _looks_like_html(head: bytes) -> bool:
    low = head.lower()
    return any(m.lower() in low for m in _HTML_MARKERS)


def _classify_download(path: Path, expected_size: Optional[int]) -> dict:
    size = path.stat().st_size
    with open(path, "rb") as fh:
        head = fh.read(512)
    is_zip = head.startswith(_ZIP_MAGIC)
    is_html = _looks_like_html(head)
    verdict = "ok"
    if is_html:
        verdict = "html_page"  # login / warning / permission page
    elif size == 0:
        verdict = "empty"
    elif expected_size is not None and size != expected_size:
        verdict = "size_mismatch"
    elif not is_zip and path.suffix.lower() == ".xlsx":
        verdict = "not_xlsx"
    return {
        "downloaded_size": size,
        "expected_size": expected_size,
        "is_zip_xlsx": is_zip,
        "looks_like_html": is_html,
        "verdict": verdict,
    }


def verify_access(cfg: DiscoveryConfig, out_dir: Path,
                  representative: Optional[dict] = None) -> dict:
    """Phase 0: verify byte-level access for both datasets.

    Downloads exactly one representative file per dataset, verifies it parses as
    a real workbook, and writes a full local report + a sanitized summary.
    """
    from .xlsx_inspect import workbook_sheet_names

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = DownloadCache(cfg.cache_root)
    representative = representative or {}

    full = {"verified_at": datetime.now(timezone.utc).isoformat(), "datasets": {}}
    summary = {"verified_at": full["verified_at"], "datasets": {}}

    for ds, src in (("A", cfg.dataset_a_source), ("B", cfg.dataset_b_source)):
        entry = {"configured": bool(src)}
        if not src:
            entry["error"] = "source_not_configured"
            full["datasets"][ds] = entry
            summary["datasets"][ds] = {"configured": False, "access_confirmed": False}
            continue
        source = build_source(src)
        files = source.list_files()
        xlsx = [f for f in files if f.ext == ".xlsx"]
        entry["visible_file_count"] = len(files)
        entry["xlsx_count"] = len(xlsx)
        entry["extensions"] = sorted({f.ext for f in files})
        # choose representative: caller-named, else first xlsx
        chosen_name = representative.get(ds)
        chosen = None
        if chosen_name:
            chosen = next((f for f in xlsx if f.name == chosen_name), None)
        if chosen is None and xlsx:
            chosen = xlsx[0]
        if chosen is None:
            entry["error"] = "no_xlsx_to_test"
            full["datasets"][ds] = entry
            summary["datasets"][ds] = {"configured": True, "access_confirmed": False,
                                       "visible_file_count": len(files)}
            continue
        rec = cache.get(source, chosen)
        cls = _classify_download(rec["path"], chosen.size)
        parsed_ok = False
        sheets = []
        parse_error = None
        if cls["verdict"] == "ok":
            try:
                sheets = workbook_sheet_names(rec["path"])
                parsed_ok = True
            except Exception as e:  # noqa: BLE001
                parse_error = f"{type(e).__name__}: {e}"
        entry["representative_file"] = chosen.name
        entry["download_classification"] = cls
        entry["sha256"] = rec["sha256"]
        entry["parsed_ok"] = parsed_ok
        entry["sheet_names"] = sheets
        entry["parse_error"] = parse_error
        entry["access_confirmed"] = parsed_ok and cls["verdict"] == "ok"
        full["datasets"][ds] = entry
        summary["datasets"][ds] = {
            "configured": True,
            "visible_file_count": len(files),
            "xlsx_count": len(xlsx),
            "representative_file": chosen.name,
            "downloaded_size": cls["downloaded_size"],
            "is_zip_xlsx": cls["is_zip_xlsx"],
            "parsed_ok": parsed_ok,
            "sheet_names": sheets,
            "access_confirmed": entry["access_confirmed"],
        }

    (out_dir / "access_verification_full.json").write_text(json.dumps(full, indent=2, default=str))
    (out_dir / "access_verification.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def verify_drive_write(cfg: DiscoveryConfig) -> dict:
    """Create, re-read, and delete a small test file in the output Drive folder."""
    result = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "output_folder_id_configured": bool(cfg.drive_output_folder_id),
        "credentials_available": False,
        "write_access_confirmed": False,
        "steps": {},
    }
    if not cfg.drive_output_folder_id:
        result["reason"] = "TDMEC_OUTPUT_DRIVE_FOLDER_ID not set"
        return result
    try:
        from .drive_api import credentials_available
    except Exception as e:  # noqa: BLE001
        result["reason"] = f"drive libs unavailable: {e}"
        return result
    if not credentials_available():
        result["reason"] = "no Google Drive credentials in environment"
        return result
    result["credentials_available"] = True

    try:
        from .drive_api import build_drive_service
        from googleapiclient.http import MediaInMemoryUpload

        svc = build_drive_service(readonly=False)
        payload = json.dumps(
            {"purpose": "Verify Cursor Cloud Agent write access", "status": "success"}
        ).encode("utf-8")
        media = MediaInMemoryUpload(payload, mimetype="application/json")
        meta = {"name": "_cursor_write_access_test.json",
                "parents": [cfg.drive_output_folder_id]}
        created = svc.files().create(body=meta, media_body=media, fields="id",
                                     supportsAllDrives=True).execute()
        fid = created["id"]
        result["steps"]["create"] = "ok"

        # re-read
        buf = svc.files().get_media(fileId=fid, supportsAllDrives=True).execute()
        readback = json.loads(buf.decode("utf-8") if isinstance(buf, bytes) else buf)
        result["steps"]["read"] = "ok" if readback.get("status") == "success" else "mismatch"

        svc.files().delete(fileId=fid, supportsAllDrives=True).execute()
        result["steps"]["delete"] = "ok"
        result["write_access_confirmed"] = result["steps"]["read"] == "ok"
    except Exception as e:  # noqa: BLE001
        result["reason"] = f"{type(e).__name__}: {e}"
    return result
