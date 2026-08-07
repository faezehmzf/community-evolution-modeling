"""Authenticated Google Drive API helpers (read and output-write).

Credentials are resolved, in order, from:

1. ``GOOGLE_APPLICATION_CREDENTIALS`` pointing at a service-account JSON key.
2. Application Default Credentials (ADC).

This module NEVER prints, serializes, or commits credential material.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

READONLY_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
READWRITE_SCOPES = ["https://www.googleapis.com/auth/drive"]


def _credentials(scopes):
    from google.auth import default as adc_default
    from google.oauth2 import service_account

    key_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if key_path and Path(key_path).is_file():
        return service_account.Credentials.from_service_account_file(
            key_path, scopes=scopes
        )
    creds, _ = adc_default(scopes=scopes)
    return creds


def build_drive_service(readonly: bool = True):
    """Build a Drive v3 service client. Raises if no credentials are available."""
    from googleapiclient.discovery import build

    scopes = READONLY_SCOPES if readonly else READWRITE_SCOPES
    creds = _credentials(scopes)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def credentials_available() -> bool:
    """Return True iff Drive credentials appear to be configured (no I/O secrets leaked)."""
    key_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if key_path and Path(key_path).is_file():
        return True
    try:
        from google.auth import default as adc_default

        adc_default(scopes=READONLY_SCOPES)
        return True
    except Exception:
        return False
