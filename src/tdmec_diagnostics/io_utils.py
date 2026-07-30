"""Atomic I/O helpers for diagnostic artifacts."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def git_short_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip() or "nogit"
    except Exception:
        return "nogit"


def make_run_id(config_hash: str, now: Optional[datetime] = None) -> str:
    now = now or datetime.now(timezone.utc)
    ts = now.strftime("%Y-%m-%dT%H%M%SZ")
    return f"{ts}_git-{git_short_hash()}_cfg-{config_hash[:8]}"


def runtime_environment() -> Dict[str, Any]:
    # cwd omitted from scientific hashes; included only in operational manifest.
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_short_hash(),
    }


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


class DiagnosticsRunLayout:
    """Filesystem layout for a Phase 2 diagnostics run."""

    def __init__(self, output_root: str | Path, run_id: str):
        self.root = Path(output_root) / "diagnostics" / run_id
        self.run_id = run_id

    def ensure(self) -> "DiagnosticsRunLayout":
        for sub in ["reports", "checkpoints", "logs", "human"]:
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        return self

    @property
    def manifest_path(self) -> Path:
        return self.root / "execution_manifest.json"

    @property
    def checkpoint_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def human_dir(self) -> Path:
        return self.root / "human"

    def report_path(self, name: str) -> Path:
        return self.reports_dir / f"{name}.json"

    def human_path(self, name: str) -> Path:
        return self.human_dir / f"{name}.md"
