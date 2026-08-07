"""Machine-readable and human-readable diagnostic report assembly."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from tdmec.hashing import hash_canonical_json_native
from tdmec_diagnostics import constants as DC
from tdmec_diagnostics.privacy import assert_privacy_safe_mapping, sanitize_warning_message
from tdmec_diagnostics.status import assert_not_certified, finalize_run_status


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=str)


def scientific_content_hash(report: Dict[str, Any]) -> str:
    """Hash scientific content excluding wall-clock timestamps."""
    payload = {
        k: v
        for k, v in report.items()
        if k
        not in {
            "generated_at",
            "wall_clock_timestamp",
            "runtime_environment",
            "captured_at",
        }
    }
    return hash_canonical_json_native(payload)


def build_warnings_report(
    *,
    config_hash: str,
    warnings: List[Dict[str, Any]],
    hard_failures: List[Dict[str, Any]],
    status: str,
) -> Dict[str, Any]:
    assert_not_certified(status)
    cleaned_w = []
    for w in warnings:
        item = dict(w)
        if "message" in item:
            item["message"] = sanitize_warning_message(str(item["message"]))
        cleaned_w.append(item)
    cleaned_h = []
    for h in hard_failures:
        item = dict(h)
        if "message" in item:
            item["message"] = sanitize_warning_message(str(item["message"]))
        cleaned_h.append(item)
    cleaned_w.sort(key=lambda x: (x.get("code", ""), x.get("message", "")))
    cleaned_h.sort(key=lambda x: (x.get("code", ""), x.get("message", "")))
    return {
        "schema_version": DC.DIAGNOSTIC_SCHEMA_VERSION,
        "report_type": DC.REPORT_WARNINGS,
        "status": status,
        "run_configuration_hash": config_hash,
        "warning_counts": len(cleaned_w),
        "hard_failure_counts": len(cleaned_h),
        "warnings": cleaned_w,
        "hard_failures": cleaned_h,
    }


def build_unresolved_evidence(
    *,
    config_hash: str,
    calendar_report: Dict[str, Any],
    dedup_report: Dict[str, Any],
    text_report: Dict[str, Any],
    coverage_report: Dict[str, Any],
    status: str = DC.REVIEW_REQUIRED,
) -> Dict[str, Any]:
    assert_not_certified(status)
    return {
        "schema_version": DC.DIAGNOSTIC_SCHEMA_VERSION,
        "report_type": DC.REPORT_UNRESOLVED,
        "status": status,
        "run_configuration_hash": config_hash,
        "items": [
            {
                "decision_id": "QCAL-B01",
                "evidence_refs": ["calendar_report"],
                "candidate": calendar_report.get("recommendation"),
                "finalization": "REVIEW_REQUIRED",
            },
            {
                "decision_id": "QDEDUP-B01",
                "evidence_refs": ["dedup_report"],
                "candidate": dedup_report.get("candidate_signatures"),
                "finalization": "REVIEW_REQUIRED",
            },
            {
                "decision_id": "coverage_hard_thresholds",
                "evidence_refs": ["coverage_report"],
                "candidate": coverage_report.get("observed_category_rates"),
                "finalization": "UNRESOLVED",
            },
            {
                "decision_id": "QEMB-X01..X07 / D_text",
                "evidence_refs": ["text_length_report"],
                "candidate": {
                    "length_summaries_present": bool(text_report.get("per_dataset")),
                    "tokenizer_status": text_report.get("tokenizer_diagnostics"),
                },
                "finalization": "DEFERRED_TO_PHASE_3",
            },
        ],
        "notes": (
            "Phase 2 produces evidence only. No item below is CERTIFIED."
        ),
    }


def build_execution_manifest(
    *,
    run_id: str,
    config_hash: str,
    config_dict: Dict[str, Any],
    source_files: List[Dict[str, Any]],
    processing_status: str,
    rows_inspected: int,
    rows_accepted: int,
    rows_rejected: int,
    warning_counts: int,
    hard_failure_counts: int,
    resume_state: Dict[str, Any],
    report_hashes: Dict[str, str],
    runtime_environment: Dict[str, Any],
    real_data_executed: bool,
) -> Dict[str, Any]:
    assert_not_certified(processing_status)
    manifest = {
        "schema_version": DC.DIAGNOSTIC_SCHEMA_VERSION,
        "report_type": DC.REPORT_MANIFEST,
        "run_id": run_id,
        "run_configuration_hash": config_hash,
        "config": config_dict,
        "source_file_identifiers": sorted(
            source_files, key=lambda x: x.get("file_ref", "")
        ),
        "processing_status": processing_status,
        "rows_inspected": rows_inspected,
        "rows_accepted_for_diagnostics": rows_accepted,
        "rows_rejected": rows_rejected,
        "warning_counts": warning_counts,
        "hard_failure_counts": hard_failure_counts,
        "resume_state": resume_state,
        "report_content_hashes": {
            k: report_hashes[k] for k in sorted(report_hashes)
        },
        "real_data_executed": real_data_executed,
        "certification_claim": None,
        # Wall-clock separated from scientific hashes
        "wall_clock_timestamp": datetime.now(timezone.utc).isoformat(),
        "runtime_environment": runtime_environment,
    }
    return manifest


def build_run_summary(
    *,
    config_hash: str,
    status: str,
    calendar_status: str,
    dedup_status: str,
    text_status: str,
    coverage_status: str,
    real_data_executed: bool,
    real_data_status_message: str,
) -> Dict[str, Any]:
    assert_not_certified(status)
    return {
        "schema_version": DC.DIAGNOSTIC_SCHEMA_VERSION,
        "report_type": DC.REPORT_SUMMARY,
        "status": status,
        "run_configuration_hash": config_hash,
        "component_status": {
            "calendar": calendar_status,
            "dedup": dedup_status,
            "text_length": text_status,
            "coverage": coverage_status,
        },
        "real_data_executed": real_data_executed,
        "real_data_status_message": real_data_status_message,
        "phase3_plus_implemented": False,
        "certification_claim": None,
        "notes": (
            "All Phase 2 artifacts use UNVALIDATED / DIAGNOSTIC_COMPLETE / "
            "REVIEW_REQUIRED only."
        ),
    }


def seal_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """Validate privacy and attach scientific content hash (not wall-clock)."""
    status = report.get("status", DC.UNVALIDATED)
    assert_not_certified(str(status))
    out = {k: v for k, v in report.items() if k != "scientific_content_hash"}
    assert_privacy_safe_mapping(out)
    out["scientific_content_hash"] = scientific_content_hash(out)
    # Re-check after adding hash
    assert_privacy_safe_mapping(out)
    return out


def human_run_summary(summary: Dict[str, Any]) -> str:
    lines = [
        "# Phase 2 diagnostics run summary",
        "",
        f"- Overall status: `{summary.get('status')}`",
        f"- Config hash: `{summary.get('run_configuration_hash')}`",
        f"- Component status: {summary.get('component_status')}",
        f"- Real data executed: {summary.get('real_data_executed')}",
        f"- Real data status: {summary.get('real_data_status_message')}",
        f"- Phase 3+ implemented: {summary.get('phase3_plus_implemented')}",
        "",
        "No CERTIFIED claims are made in Phase 2.",
    ]
    return "\n".join(lines) + "\n"
