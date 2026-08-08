"""Privacy-safe line logging and runtime provenance for embedding jobs."""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


DEPENDENCY_NAMES = (
    "torch",
    "transformers",
    "accelerate",
    "tokenizers",
    "safetensors",
    "huggingface_hub",
    "numpy",
    "pandas",
    "pyarrow",
    "PyYAML",
    "psutil",
)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def log_event(event: str, **fields: Any) -> None:
    """Emit one flushed JSON line; never pass raw text or secrets in fields."""

    payload = {"timestamp": utc_timestamp(), "event": event, **fields}
    print(json.dumps(payload, sort_keys=True, allow_nan=False), flush=True)


def dependency_versions(names: Iterable[str] = DEPENDENCY_NAMES) -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def git_commit_sha(repo_root: str | Path = ".") -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(repo_root),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


def resource_snapshot(*, torch_module: Any = None, disk_path: str | Path = ".") -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import psutil

        vm = psutil.virtual_memory()
        proc = psutil.Process()
        report.update(
            {
                "cpu_ram_total_bytes": int(vm.total),
                "cpu_ram_available_bytes": int(vm.available),
                "process_rss_bytes": int(proc.memory_info().rss),
            }
        )
    except ImportError:
        report.update(
            {
                "cpu_ram_total_bytes": None,
                "cpu_ram_available_bytes": None,
                "process_rss_bytes": None,
            }
        )
    usage = shutil.disk_usage(Path(disk_path).resolve())
    report.update(
        {
            "disk_total_bytes": int(usage.total),
            "disk_used_bytes": int(usage.used),
            "disk_free_bytes": int(usage.free),
        }
    )
    torch = torch_module
    if torch is not None:
        report["torch_version"] = getattr(torch, "__version__", None)
        report["cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
        report["cuda_available"] = bool(torch.cuda.is_available())
        if report["cuda_available"]:
            device = torch.device("cuda:0")
            free, total = torch.cuda.mem_get_info(device)
            report.update(
                {
                    "gpu_name": torch.cuda.get_device_name(device),
                    "gpu_capability": list(torch.cuda.get_device_capability(device)),
                    "gpu_vram_total_bytes": int(total),
                    "gpu_vram_free_bytes": int(free),
                    "gpu_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                    "gpu_reserved_bytes": int(torch.cuda.memory_reserved(device)),
                }
            )
    return report


def model_memory_report(model: Any) -> Dict[str, Any]:
    devices: Dict[str, int] = {}
    dtypes: Dict[str, int] = {}
    parameter_count = 0
    parameter_bytes = 0
    meta_parameters = 0
    for parameter in model.parameters():
        count = int(parameter.numel())
        size = count * int(parameter.element_size())
        parameter_count += count
        parameter_bytes += size
        device = str(parameter.device)
        dtype = str(parameter.dtype)
        devices[device] = devices.get(device, 0) + size
        dtypes[dtype] = dtypes.get(dtype, 0) + size
        if getattr(parameter, "is_meta", False):
            meta_parameters += 1
    return {
        "parameter_count": parameter_count,
        "parameter_bytes": parameter_bytes,
        "parameter_bytes_by_device": dict(sorted(devices.items())),
        "parameter_bytes_by_dtype": dict(sorted(dtypes.items())),
        "meta_parameter_tensors": meta_parameters,
        "hf_device_map": getattr(model, "hf_device_map", None),
    }


class Heartbeat:
    """Periodic resource lines for otherwise silent download/load/inference stages."""

    def __init__(
        self,
        stage: str,
        *,
        interval_seconds: float = 60.0,
        torch_module: Any = None,
        disk_path: str | Path = ".",
    ) -> None:
        self.stage = stage
        self.interval_seconds = max(1.0, float(interval_seconds))
        self.torch_module = torch_module
        self.disk_path = disk_path
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "Heartbeat":
        def run() -> None:
            while not self._stop.wait(self.interval_seconds):
                log_event(
                    "heartbeat",
                    stage=self.stage,
                    resources=resource_snapshot(
                        torch_module=self.torch_module, disk_path=self.disk_path
                    ),
                )

        self._thread = threading.Thread(target=run, daemon=True, name="tdmec-heartbeat")
        self._thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


__all__ = [
    "DEPENDENCY_NAMES",
    "Heartbeat",
    "dependency_versions",
    "git_commit_sha",
    "log_event",
    "model_memory_report",
    "resource_snapshot",
    "utc_timestamp",
]
