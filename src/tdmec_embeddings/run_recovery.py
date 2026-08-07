"""Run-directory assessment and incomplete-run recovery for embedding outputs.

COMPLETED runs are never deleted. Interrupted / incompatible incomplete runs may
be removed only when ``replace_incomplete`` is explicitly authorized.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional


class RunRecoveryError(RuntimeError):
    pass


def assess_embedding_run(run_root: str | Path) -> Dict[str, Any]:
    """Classify an embedding run directory without mutating it."""

    root = Path(run_root)
    report: Dict[str, Any] = {
        "run_root": root.as_posix(),
        "exists": root.is_dir(),
        "final_manifest_present": False,
        "final_status": None,
        "modality_statuses": {},
        "has_unit_shards": False,
        "has_checkpoints": False,
        "classification": "absent",
        "completed": False,
        "incomplete": False,
        "safe_to_replace": False,
    }
    if not root.is_dir():
        return report

    final_path = root / "embedding_manifest.json"
    if final_path.is_file():
        report["final_manifest_present"] = True
        try:
            final = json.loads(final_path.read_text(encoding="utf-8"))
            report["final_status"] = final.get("status")
        except json.JSONDecodeError:
            report["final_status"] = "CORRUPT"

    for modality in ("node_text", "event_text"):
        manifest = root / "manifests" / f"{modality}.json"
        checkpoint = root / "checkpoints" / f"{modality}.json"
        shards = root / "unit_embeddings" / modality
        status = None
        if manifest.is_file():
            try:
                status = json.loads(manifest.read_text(encoding="utf-8")).get("status")
            except json.JSONDecodeError:
                status = "CORRUPT"
        elif checkpoint.is_file():
            try:
                status = json.loads(checkpoint.read_text(encoding="utf-8")).get("status")
            except json.JSONDecodeError:
                status = "CORRUPT"
        report["modality_statuses"][modality] = status
        if shards.is_dir() and any(shards.glob("*.parquet")):
            report["has_unit_shards"] = True
        if checkpoint.is_file():
            report["has_checkpoints"] = True

    statuses = [report["final_status"], *report["modality_statuses"].values()]
    statuses = [s for s in statuses if s is not None]
    completed = report["final_status"] == "COMPLETED" and all(
        report["modality_statuses"].get(m) == "COMPLETED" for m in ("node_text", "event_text")
    )
    report["completed"] = completed
    if completed:
        report["classification"] = "completed"
        report["incomplete"] = False
        report["safe_to_replace"] = False
        return report

    has_artifacts = (
        report["final_manifest_present"]
        or report["has_unit_shards"]
        or report["has_checkpoints"]
        or any(report["modality_statuses"].values())
        or (root / "pooled").is_dir()
        or (root / "reports").is_dir()
    )
    if not has_artifacts:
        report["classification"] = "empty"
        return report

    report["classification"] = "incomplete"
    report["incomplete"] = True
    report["safe_to_replace"] = True
    return report


def replace_incomplete_run(run_root: str | Path) -> Dict[str, Any]:
    """Delete an incomplete run directory. Refuses COMPLETED outputs."""

    root = Path(run_root)
    assessment = assess_embedding_run(root)
    if assessment["classification"] == "absent":
        return {"action": "noop", "assessment": assessment}
    if assessment["completed"]:
        raise RunRecoveryError(
            "refusing to delete a COMPLETED embedding run; overwrite is prohibited"
        )
    if not assessment["safe_to_replace"]:
        raise RunRecoveryError(
            f"run directory is not classified as replaceable incomplete "
            f"(classification={assessment['classification']!r})"
        )
    shutil.rmtree(root)
    return {"action": "deleted", "assessment": assessment}


def prepare_embedding_run_root(
    *,
    output_root: str | Path,
    embedding_run_id: str,
    resume: bool,
    replace_incomplete: bool,
) -> Dict[str, Any]:
    """Apply resume / replace-incomplete policy before pipeline writers start."""

    run_root = Path(output_root).resolve() / embedding_run_id
    assessment = assess_embedding_run(run_root)
    actions: List[str] = []

    if assessment["classification"] in {"absent", "empty"}:
        return {
            "run_root": run_root.as_posix(),
            "assessment": assessment,
            "actions": actions,
        }

    if assessment["completed"]:
        if resume:
            actions.append("resume_completed_allowed")
            return {
                "run_root": run_root.as_posix(),
                "assessment": assessment,
                "actions": actions,
            }
        raise RunRecoveryError(
            f"completed embedding output exists at {run_root}; "
            "pass --resume to reuse it, overwrite is prohibited"
        )

    # Incomplete / interrupted / incompatible leftovers
    if replace_incomplete:
        replace_incomplete_run(run_root)
        actions.append("replaced_incomplete")
        return {
            "run_root": run_root.as_posix(),
            "assessment": assess_embedding_run(run_root),
            "actions": actions,
        }
    if resume:
        actions.append("resume_incomplete")
        return {
            "run_root": run_root.as_posix(),
            "assessment": assessment,
            "actions": actions,
        }
    raise RunRecoveryError(
        f"incomplete embedding artifacts exist at {run_root} "
        f"(classification={assessment['classification']!r}, "
        f"modality_statuses={assessment['modality_statuses']}). "
        "Pass --resume to continue a compatible checkpoint, or "
        "--replace-incomplete to delete the failed run and start clean."
    )


__all__ = [
    "RunRecoveryError",
    "assess_embedding_run",
    "prepare_embedding_run_root",
    "replace_incomplete_run",
]
