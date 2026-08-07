"""Configuration loading for the discovery tooling.

All dataset locations and output roots are supplied through environment
variables (optionally seeded from a git-ignored ``.env`` file). No Google Drive
URLs, folder ids, or credentials are ever hardcoded in source.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def load_env_file(path: str | os.PathLike) -> None:
    """Load ``KEY=VALUE`` pairs from a dotenv-style file into ``os.environ``.

    Existing environment variables are NOT overwritten (env wins over file).
    Missing files are ignored so this is safe to call unconditionally.
    """
    p = Path(path)
    if not p.is_file():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class DiscoveryConfig:
    """Resolved discovery configuration."""

    dataset_a_source: Optional[str]
    dataset_b_source: Optional[str]
    output_root: Path
    cache_root: Path
    drive_output_folder_id: Optional[str]
    google_credentials: Optional[str]

    def fingerprint(self) -> str:
        """A stable, non-sensitive hash of the *shape* of the configuration.

        Uses the source *scheme* (not the private folder id/URL) plus output and
        cache roots so the run-id config hash never leaks a private id.
        """
        import hashlib

        def scheme(v: Optional[str]) -> str:
            if not v:
                return "none"
            return v.split(":", 1)[0] if ":" in v else "path"

        material = "|".join(
            [
                scheme(self.dataset_a_source),
                scheme(self.dataset_b_source),
                str(self.output_root),
                str(self.cache_root),
                "drive_out" if self.drive_output_folder_id else "no_drive_out",
                "creds" if self.google_credentials else "no_creds",
            ]
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:6]


def load_config(env_file: Optional[str] = None) -> DiscoveryConfig:
    """Load configuration from the environment (and optional dotenv file)."""
    if env_file:
        load_env_file(env_file)
    else:
        # Best-effort local override, never committed.
        load_env_file(Path("config") / "discovery.local.env")

    return DiscoveryConfig(
        dataset_a_source=os.environ.get("DATASET_A_SOURCE")
        or _drive_id_source(os.environ.get("DATASET_A_DRIVE_FOLDER_ID")),
        dataset_b_source=os.environ.get("DATASET_B_SOURCE")
        or _drive_id_source(os.environ.get("DATASET_B_DRIVE_FOLDER_ID")),
        output_root=Path(os.environ.get("DISCOVERY_OUTPUT_ROOT", "./artifacts/discovery")),
        cache_root=Path(os.environ.get("DISCOVERY_CACHE_ROOT", "/tmp/tdmec_cache")),
        drive_output_folder_id=os.environ.get("TDMEC_OUTPUT_DRIVE_FOLDER_ID"),
        google_credentials=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
    )


def _drive_id_source(folder_id: Optional[str]) -> Optional[str]:
    if not folder_id:
        return None
    return f"gdrive-anon:{folder_id}"
