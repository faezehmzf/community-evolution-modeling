"""Isolated, supervised Qwen model-only preflight (no Dataset A/B processing)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from tdmec.hashing import hash_canonical, sha256_file

from .config import EmbeddingRunConfig, load_embedding_config
from .eligibility import EligibleTextUnit, cleaned_text_content_hash
from .file_writer import _atomic_write_json
from .observability import (
    dependency_versions,
    git_commit_sha,
    log_event,
    resource_snapshot,
    utc_timestamp,
)
from .qwen_encoder import Qwen3Encoder


REQUIRED_HF_VERSIONS = {
    # Verify only: never ask pip to install or replace Kaggle's CUDA wheel.
    "torch": "2.10.0+cu128",
    "transformers": "4.55.4",
    "accelerate": "1.10.1",
    "tokenizers": "0.21.4",
    "safetensors": "0.5.3",
    "huggingface_hub": "0.34.4",
}


class ModelPreflightError(RuntimeError):
    pass


def preflight_compatibility_key(config: EmbeddingRunConfig) -> str:
    return hash_canonical(
        {
            "schema_version": "tdmec-model-preflight-compatibility-v1",
            "encoder": config.encoder.scientific_payload(),
            "required_hf_versions": REQUIRED_HF_VERSIONS,
        }
    )


def _preflight_units(config: EmbeddingRunConfig) -> tuple[EligibleTextUnit, ...]:
    texts = (
        "A short English sentence about community change.",
        "این یک متن کوتاه فارسی درباره تغییرات اجتماعی است.",
        "یک پیام چندزبانه درباره شبکه و جامعه.",
        "La comunidad cambia con el tiempo.",
        "La communauté évolue au fil du temps.",
        "تفاعل اجتماعي قصير عبر الزمن.",
        "短い多言語ソーシャルネットワーク文です。",
        "A neutral reply acknowledges a different point of view.",
    )
    preprocessing_hash = hash_canonical({"model_preflight": "multilingual-v1"})
    units = []
    for index, text in enumerate(texts):
        units.append(
            EligibleTextUnit(
                modality="node_text",
                source_run_id="model-only-preflight",
                unit_id=f"preflight-{index}",
                unit_hash=hash_canonical({"preflight_unit": index}),
                content_hash=cleaned_text_content_hash(text),
                preprocessing_hash=preprocessing_hash,
                cleaned_text=text,
                snapshot_id=0,
                node_index=index,
                relation_id=None,
                source_idx=None,
                target_idx=None,
                source_file="model-only-preflight",
                source_row_number=index,
            )
        )
    return tuple(units)


def _assert_dependency_versions(versions: Dict[str, Any]) -> None:
    mismatches = {
        name: {"expected": expected, "actual": versions.get(name)}
        for name, expected in REQUIRED_HF_VERSIONS.items()
        if versions.get(name) != expected
    }
    if mismatches:
        raise ModelPreflightError(f"untested Hugging Face dependency stack: {mismatches}")


def run_worker(config: EmbeddingRunConfig, output_path: str | Path) -> Dict[str, Any]:
    if config.encoder.backend != "qwen3":
        raise ModelPreflightError("model-only preflight requires backend=qwen3")
    versions = dependency_versions()
    _assert_dependency_versions(versions)
    commit = git_commit_sha()
    log_event(
        "model_preflight_started",
        git_commit=commit,
        configuration_hash=config.scientific_hash(),
        model_revision=config.encoder.model_revision,
        dependency_versions=versions,
    )
    started = time.perf_counter()
    encoder = Qwen3Encoder(config.encoder)
    encoder.load()
    before_inference = resource_snapshot(torch_module=encoder._torch)
    units = _preflight_units(config)
    vectors = encoder.encode(units)
    finite = bool(np.all(np.isfinite(vectors)))
    shape_ok = vectors.shape == (len(units), config.encoder.output_dimension)
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
    norm_ok = bool(
        np.allclose(
            norms,
            1.0,
            rtol=0.0,
            atol=float(config.encoder.normalized_atol),
        )
    )
    runtime = encoder.runtime_report()
    placement = runtime.get("model_memory") or {}
    passed = all(
        (
            finite,
            shape_ok,
            norm_ok,
            runtime.get("inference_dtype") == "fp16",
            placement.get("meta_parameter_tensors") == 0,
            set((placement.get("parameter_bytes_by_device") or {}).keys())
            == {config.encoder.device},
            not bool(encoder._model.training),
        )
    )
    report = {
        "schema_version": "tdmec-model-only-preflight-v1",
        "status": "PASSED" if passed else "FAILED",
        "created_at": utc_timestamp(),
        "git_commit": commit,
        "configuration_hash": config.scientific_hash(),
        "preflight_compatibility_key": preflight_compatibility_key(config),
        "model": encoder.metadata.payload(),
        "dependency_versions": versions,
        "checks": {
            "text_count_between_2_and_16": 2 <= len(units) <= 16,
            "shape": list(vectors.shape),
            "expected_shape": [len(units), config.encoder.output_dimension],
            "shape_ok": shape_ok,
            "finite": finite,
            "unit_norm": norm_ok,
            "evaluation_mode": not bool(encoder._model.training),
            "inference_mode_used": True,
            "fp16": runtime.get("inference_dtype") == "fp16",
            "single_explicit_gpu": set(
                (placement.get("parameter_bytes_by_device") or {}).keys()
            )
            == {config.encoder.device},
            "no_meta_parameters": placement.get("meta_parameter_tensors") == 0,
            "no_cpu_or_disk_offload": True,
        },
        "resources_before_inference": before_inference,
        "resources_after_inference": resource_snapshot(torch_module=encoder._torch),
        "runtime": runtime,
        "elapsed_seconds": time.perf_counter() - started,
        "datasets_processed": False,
    }
    report_path = Path(output_path)
    _atomic_write_json(report_path, report)
    _atomic_write_json(
        Path(str(report_path) + ".sha256.json"),
        {"schema_version": "tdmec-model-preflight-checksum-v1", "sha256": sha256_file(report_path)},
    )
    log_event(
        "model_preflight_completed",
        status=report["status"],
        elapsed_seconds=report["elapsed_seconds"],
        output_path=Path(output_path).name,
    )
    if not passed:
        raise ModelPreflightError("model-only preflight checks failed")
    return report


def verify_preflight_report(
    path: str | Path, config: EmbeddingRunConfig, *, require_current_commit: bool = True
) -> Dict[str, Any]:
    report_path = Path(path)
    if not report_path.is_file():
        raise ModelPreflightError(f"model preflight report is missing: {report_path}")
    checksum_path = Path(str(report_path) + ".sha256.json")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        checksum = json.loads(checksum_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelPreflightError("model preflight report/checksum is invalid") from exc
    if checksum.get("sha256") != sha256_file(report_path):
        raise ModelPreflightError("model preflight report checksum mismatch")
    if report.get("status") != "PASSED":
        raise ModelPreflightError("model preflight did not pass")
    if report.get("preflight_compatibility_key") != preflight_compatibility_key(config):
        raise ModelPreflightError("model preflight is incompatible with the encoder config")
    _assert_dependency_versions(report.get("dependency_versions") or {})
    if require_current_commit and report.get("git_commit") != git_commit_sha():
        raise ModelPreflightError("model preflight was produced by a different Git commit")
    checks = report.get("checks") or {}
    required = (
        "text_count_between_2_and_16",
        "shape_ok",
        "finite",
        "unit_norm",
        "evaluation_mode",
        "inference_mode_used",
        "fp16",
        "single_explicit_gpu",
        "no_meta_parameters",
        "no_cpu_or_disk_offload",
    )
    failed = [name for name in required if checks.get(name) is not True]
    if failed:
        raise ModelPreflightError(f"model preflight checks are incomplete: {failed}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Supervised Qwen model-only preflight")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_embedding_config(args.config)
        if args.worker:
            run_worker(config, args.output)
            return 0
        if not 60 <= args.timeout_seconds <= 1800:
            raise ModelPreflightError("timeout must be between 60 and 1800 seconds")
        command = [
            sys.executable,
            "-u",
            "-m",
            "tdmec_embeddings.model_preflight",
            "--config",
            args.config,
            "--output",
            args.output,
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--worker",
        ]
        log_event("model_preflight_supervisor_started", timeout_seconds=args.timeout_seconds)
        process = subprocess.Popen(command)
        try:
            return_code = process.wait(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            log_event("model_preflight_timeout", timeout_seconds=args.timeout_seconds)
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=20)
            return 124
        if return_code != 0:
            return return_code
        verify_preflight_report(args.output, config)
        return 0
    except (OSError, ValueError, ModelPreflightError) as exc:
        log_event("model_preflight_failed", error_type=type(exc).__name__, message=str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ModelPreflightError",
    "REQUIRED_HF_VERSIONS",
    "preflight_compatibility_key",
    "run_worker",
    "verify_preflight_report",
]
