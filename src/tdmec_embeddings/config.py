"""Configuration contract for transferable file-backed embedding runs.

IMPLEMENTED_NOT_EXECUTED / TRANSFER_PENDING_VALIDATION.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional, Tuple

import yaml

from tdmec.hashing import hash_canonical


ExecutionMode = Literal["mock", "qwen_preflight", "qwen_bounded_pilot"]


class EmbeddingConfigError(ValueError):
    pass


def _expand_path_text(value: str) -> str:
    return os.path.expandvars(str(value)).strip()


@dataclass(frozen=True)
class SourceConfig:
    artifact_root: str
    run_id: str

    def resolved_root(self) -> Path:
        return Path(_expand_path_text(self.artifact_root)).expanduser().resolve()


@dataclass(frozen=True)
class EncoderConfig:
    backend: Literal["mock", "qwen3"]
    model_name: str
    model_revision: str
    tokenizer_revision: str
    instruction: str
    output_dimension: int
    max_length: int
    precision: Literal["auto", "bf16", "fp16", "fp32"]
    device: str
    batch_size: int
    max_oom_retries: int
    local_files_only: bool = False
    allow_cpu: bool = False
    attn_implementation: Optional[str] = None
    enable_provisional_mrl_truncation: bool = False
    normalize: bool = True
    normalized_atol: float = 1e-5

    def validate(self) -> None:
        if self.backend == "qwen3":
            if not self.model_name.startswith("Qwen/Qwen3-Embedding-"):
                raise EmbeddingConfigError("real encoder must use the Qwen3 embedding family")
            for label, value in (
                ("model_revision", self.model_revision),
                ("tokenizer_revision", self.tokenizer_revision),
            ):
                expanded = _expand_path_text(value)
                if (
                    not expanded
                    or expanded in {"main", "latest"}
                    or "${" in expanded
                    or expanded.startswith("$")
                    or "REQUIRED" in expanded
                    or "UNRESOLVED" in expanded
                    or "REPLACE_WITH" in expanded
                    or "PROVISIONAL" in expanded
                ):
                    raise EmbeddingConfigError(
                        f"{label} must be an immutable pinned revision resolved in the target Studio "
                        "(export QWEN3_MODEL_REVISION / QWEN3_TOKENIZER_REVISION before loading)"
                    )
            if not self.normalize:
                raise EmbeddingConfigError(
                    "Qwen3 unit L2 normalization is mandatory; normalize=false is forbidden"
                )
        if self.output_dimension <= 0 or self.max_length <= 0 or self.batch_size <= 0:
            raise EmbeddingConfigError("encoder dimensions, length, and batch size must be positive")
        if self.max_oom_retries < 0:
            raise EmbeddingConfigError("max_oom_retries must be non-negative")
        if not self.device:
            raise EmbeddingConfigError("device selection must be explicit")
        if self.normalized_atol <= 0.0:
            raise EmbeddingConfigError("normalized_atol must be positive")

    def scientific_payload(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "instruction_hash": hash_canonical({"instruction": self.instruction}),
            "output_dimension": self.output_dimension,
            "max_length": self.max_length,
            "precision": self.precision,
            "attn_implementation": self.attn_implementation,
            "enable_provisional_mrl_truncation": self.enable_provisional_mrl_truncation,
            "normalize": self.normalize,
            "normalized_atol": self.normalized_atol,
        }


@dataclass(frozen=True)
class SamplingConfig:
    strategy: Literal["deterministic_prefix", "deterministic_stratified_hash"]
    seed: int = 20260804
    node_hash_buckets: int = 64
    short_text_max_chars: int = 64
    medium_text_max_chars: int = 256
    force_relation_coverage: bool = True

    def validate(self) -> None:
        if self.node_hash_buckets <= 0:
            raise EmbeddingConfigError("node_hash_buckets must be positive")
        if not 0 < self.short_text_max_chars < self.medium_text_max_chars:
            raise EmbeddingConfigError("text-length bucket boundaries are invalid")


@dataclass(frozen=True)
class PoolingConfig:
    final_normalization: Literal["none", "l2"] = "none"
    delta_batch_rows: int = 4096

    def validate(self) -> None:
        if self.delta_batch_rows <= 0:
            raise EmbeddingConfigError("pooling delta_batch_rows must be positive")


@dataclass(frozen=True)
class ValidationConfig:
    normalized_atol: float = 1e-5
    require_finite: bool = True
    require_relation_coverage: bool = True
    expected_relation_ids: Tuple[int, ...] = (0, 1, 2, 3)

    def validate(self) -> None:
        if self.normalized_atol <= 0.0:
            raise EmbeddingConfigError("validation.normalized_atol must be positive")
        if not self.expected_relation_ids:
            raise EmbeddingConfigError("expected_relation_ids must be non-empty")


@dataclass(frozen=True)
class ExportConfig:
    package_name: str = "TDMEC_INPUT"
    include_unit_embeddings: bool = False
    include_graph_edges: bool = True

    def validate(self) -> None:
        if not self.package_name:
            raise EmbeddingConfigError("export.package_name must be non-empty")


@dataclass(frozen=True)
class EmbeddingRunConfig:
    schema_version: str
    execution_mode: ExecutionMode
    embedding_run_id: str
    output_root: str
    node_source: SourceConfig
    event_source: SourceConfig
    input_batch_size: int
    output_shard_size: int
    max_node_rows: int
    max_event_rows: int
    resume: bool
    dry_run: bool
    encoder: EncoderConfig
    sampling: SamplingConfig
    pooling: PoolingConfig = field(default_factory=PoolingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    force: bool = False
    provisional_labels: tuple[str, ...] = (
        "IMPLEMENTED_NOT_EXECUTED",
        "TRANSFER_PENDING_VALIDATION",
        "PROVISIONAL_SMOKE_ONLY",
        "NOT_FOR_FINAL_THESIS_CONCLUSIONS",
    )

    def validate(self) -> None:
        if self.schema_version != "tdmec-embedding-run-config-v1":
            raise EmbeddingConfigError("unsupported embedding configuration schema")
        if not self.embedding_run_id or self.embedding_run_id in {
            self.node_source.run_id,
            self.event_source.run_id,
        }:
            raise EmbeddingConfigError("embedding run identity must be independent")
        for value in (
            self.input_batch_size,
            self.output_shard_size,
            self.max_node_rows,
            self.max_event_rows,
        ):
            if value <= 0:
                raise EmbeddingConfigError("batch and row limits must be positive")
        if self.force:
            raise EmbeddingConfigError(
                "force overwrite is disabled by policy; completed outputs must not be overwritten"
            )
        if self.execution_mode == "mock" and self.encoder.backend != "mock":
            raise EmbeddingConfigError("mock mode requires the mock encoder")
        if self.execution_mode != "mock" and self.encoder.backend != "qwen3":
            raise EmbeddingConfigError("real modes require the Qwen3 encoder")
        if self.execution_mode == "mock":
            if self.sampling.strategy != "deterministic_prefix":
                raise EmbeddingConfigError("mock mode uses deterministic_prefix sampling")
        else:
            if self.sampling.strategy != "deterministic_stratified_hash":
                raise EmbeddingConfigError("real modes require stratified sampling")
        cap = 64 if self.execution_mode == "qwen_preflight" else 10_000
        if self.execution_mode != "mock" and (
            self.max_node_rows > cap or self.max_event_rows > cap
        ):
            raise EmbeddingConfigError(
                f"{self.execution_mode} row limits may not exceed {cap} per modality"
            )
        if self.execution_mode == "qwen_preflight" and (
            self.max_node_rows > 64 or self.max_event_rows > 64
        ):
            raise EmbeddingConfigError(
                "preflight refuses limits above 64; select qwen_bounded_pilot for larger caps"
            )
        self.encoder.validate()
        self.sampling.validate()
        self.pooling.validate()
        self.validation.validate()
        self.export.validate()

    def resolved_output_root(self) -> Path:
        return Path(_expand_path_text(self.output_root)).expanduser().resolve()

    def scientific_hash(self) -> str:
        return hash_canonical(
            {
                "schema_version": self.schema_version,
                "execution_mode": self.execution_mode,
                "embedding_run_id": self.embedding_run_id,
                "node_source_run_id": self.node_source.run_id,
                "event_source_run_id": self.event_source.run_id,
                "limits": {
                    "max_node_rows": self.max_node_rows,
                    "max_event_rows": self.max_event_rows,
                },
                "encoder": self.encoder.scientific_payload(),
                "sampling": self.sampling.__dict__,
                "pooling": self.pooling.__dict__,
                "validation": {
                    "normalized_atol": self.validation.normalized_atol,
                    "require_finite": self.validation.require_finite,
                    "require_relation_coverage": self.validation.require_relation_coverage,
                    "expected_relation_ids": list(self.validation.expected_relation_ids),
                },
                "export": self.export.__dict__,
            }
        )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EmbeddingConfigError(f"{label} must be a mapping")
    return value


def _optional_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    return _mapping(value, label)


def load_embedding_config(path: str | Path) -> EmbeddingRunConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    root = _mapping(raw, "configuration")
    node = _mapping(root.get("node_source"), "node_source")
    event = _mapping(root.get("event_source"), "event_source")
    encoder_raw = dict(_mapping(root.get("encoder"), "encoder"))
    sampling_raw = dict(_mapping(root.get("sampling"), "sampling"))
    pooling = _optional_mapping(root.get("pooling"), "pooling")
    validation_raw = dict(_optional_mapping(root.get("validation"), "validation"))
    export_raw = dict(_optional_mapping(root.get("export"), "export"))
    for key in ("model_revision", "tokenizer_revision", "model_name", "instruction", "device"):
        if key in encoder_raw and isinstance(encoder_raw[key], str):
            encoder_raw[key] = _expand_path_text(encoder_raw[key])
    if "expected_relation_ids" in validation_raw:
        validation_raw["expected_relation_ids"] = tuple(
            int(v) for v in validation_raw["expected_relation_ids"]
        )
    config = EmbeddingRunConfig(
        schema_version=str(root.get("schema_version", "")),
        execution_mode=str(root.get("execution_mode", "")),  # type: ignore[arg-type]
        embedding_run_id=str(root.get("embedding_run_id", "")),
        output_root=str(root.get("output_root", "")),
        node_source=SourceConfig(**node),
        event_source=SourceConfig(**event),
        input_batch_size=int(root.get("input_batch_size", 0)),
        output_shard_size=int(root.get("output_shard_size", 0)),
        max_node_rows=int(root.get("max_node_rows", 0)),
        max_event_rows=int(root.get("max_event_rows", 0)),
        resume=bool(root.get("resume", False)),
        dry_run=bool(root.get("dry_run", False)),
        force=bool(root.get("force", False)),
        encoder=EncoderConfig(**encoder_raw),  # type: ignore[arg-type]
        sampling=SamplingConfig(**sampling_raw),
        pooling=PoolingConfig(**pooling),
        validation=ValidationConfig(**validation_raw),
        export=ExportConfig(**export_raw),
    )
    config.validate()
    return config


__all__ = [
    "EmbeddingConfigError",
    "EmbeddingRunConfig",
    "EncoderConfig",
    "ExportConfig",
    "PoolingConfig",
    "SamplingConfig",
    "SourceConfig",
    "ValidationConfig",
    "load_embedding_config",
]
