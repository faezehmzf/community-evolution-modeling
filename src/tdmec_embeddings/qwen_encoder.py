"""Lazy Qwen3 embedding backend for authorized target-Studio / Kaggle execution.

IMPLEMENTED_NOT_EXECUTED / TRANSFER_PENDING_VALIDATION.

Canonical unit-vector pipeline (no bypass allowed):

    text → tokenize → Qwen3 forward → last-token pool
      → MRL truncate to ``output_dimension``
      → L2 normalize
      → finite + unit-norm validation
      → float32 numpy

Imports and model loading are deliberately lazy; importing this module cannot
download a model.  Norm validation failures raise ``EncoderError``.
"""
from __future__ import annotations

import platform
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np

from tdmec.hashing import hash_canonical

from .config import EncoderConfig
from .eligibility import EligibleTextUnit, cleaned_text_content_hash
from .implementation_status import IMPLEMENTATION_STATUS_LABELS
from .mock_encoder import EncoderMetadata, EncoderError


class QwenEnvironmentError(EncoderError):
    pass


class QwenOutOfMemoryError(EncoderError):
    pass


@dataclass
class QwenRuntimeStats:
    encoded_units: int = 0
    token_count: int = 0
    truncated_units: int = 0
    batches: int = 0
    oom_retries: int = 0
    elapsed_seconds: float = 0.0
    observed_batch_sizes: list[int] = field(default_factory=list)
    token_lengths: list[int] = field(default_factory=list, repr=False)
    peak_gpu_allocated_bytes: int = 0
    peak_gpu_reserved_bytes: int = 0
    peak_system_rss_bytes: int = 0

    def report(self) -> Dict[str, Any]:
        lengths = sorted(self.token_lengths)

        def percentile(p: float) -> Optional[int]:
            if not lengths:
                return None
            return lengths[round((len(lengths) - 1) * p)]

        return {
            "encoded_units": self.encoded_units,
            "token_count": self.token_count,
            "truncated_units": self.truncated_units,
            "truncation_rate": (
                self.truncated_units / self.encoded_units if self.encoded_units else 0.0
            ),
            "batches": self.batches,
            "oom_retries": self.oom_retries,
            "elapsed_seconds": self.elapsed_seconds,
            "units_per_second": (
                self.encoded_units / self.elapsed_seconds if self.elapsed_seconds else None
            ),
            "observed_batch_sizes": list(self.observed_batch_sizes),
            "token_length": {
                "min": lengths[0] if lengths else None,
                "p50": percentile(0.50),
                "p90": percentile(0.90),
                "p95": percentile(0.95),
                "p99": percentile(0.99),
                "p999": percentile(0.999),
                "max": lengths[-1] if lengths else None,
            },
            "peak_gpu_allocated_bytes": self.peak_gpu_allocated_bytes,
            "peak_gpu_reserved_bytes": self.peak_gpu_reserved_bytes,
            "peak_system_rss_bytes": self.peak_system_rss_bytes,
            "status_labels": list(IMPLEMENTATION_STATUS_LABELS),
        }


class Qwen3Encoder:
    """Bounded Qwen3 encoder with explicit OOM backoff and no substitution."""

    def __init__(self, config: EncoderConfig) -> None:
        config.validate()
        if config.backend != "qwen3":
            raise ValueError("Qwen3Encoder requires backend=qwen3")
        if not config.normalize:
            raise ValueError(
                "Qwen3Encoder requires normalize=true; unit-norm bypass is forbidden"
            )
        self.config = config
        self.metadata = EncoderMetadata(
            model_name=config.model_name,
            model_revision=config.model_revision,
            tokenizer_revision=config.tokenizer_revision,
            dimension=config.output_dimension,
            output_dtype="float32",
            unit_normalized=True,
            instruction_hash=hash_canonical({"instruction": config.instruction}),
            backend="qwen3_transformers_last_token",
        )
        self.stats = QwenRuntimeStats()
        self._torch: Any = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._device: Any = None
        self._inference_dtype_name: Optional[str] = None

    def _resolve_device_and_dtype(self, torch: Any) -> tuple[Any, Any, str]:
        requested = self.config.device
        if requested.startswith("cuda"):
            if not torch.cuda.is_available():
                raise QwenEnvironmentError("CUDA was requested but is unavailable")
            device = torch.device(requested)
            major, _minor = torch.cuda.get_device_capability(device)
            bf16_supported = bool(torch.cuda.is_bf16_supported()) and major >= 8
            precision = self.config.precision
            if precision == "auto":
                precision = "bf16" if bf16_supported else "fp16"
            if precision == "bf16" and not bf16_supported:
                raise QwenEnvironmentError("bf16 requested on unsupported CUDA hardware")
            dtype = {
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
                "fp32": torch.float32,
            }[precision]
            return device, dtype, precision
        if requested == "cpu":
            if not self.config.allow_cpu:
                raise QwenEnvironmentError(
                    "CPU Qwen execution is disabled; set allow_cpu only for an explicit small check"
                )
            if self.config.precision not in {"auto", "fp32"}:
                raise QwenEnvironmentError("CPU execution requires fp32 precision")
            return torch.device("cpu"), torch.float32, "fp32"
        raise QwenEnvironmentError("device must be cpu or an explicit cuda device")

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            import transformers
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise QwenEnvironmentError(
                "target Studio lacks torch/transformers embedding dependencies"
            ) from exc
        device, dtype, dtype_name = self._resolve_device_and_dtype(torch)
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            revision=self.config.tokenizer_revision,
            padding_side="left",
            local_files_only=self.config.local_files_only,
            trust_remote_code=False,
        )
        model_kwargs: Dict[str, Any] = {
            "revision": self.config.model_revision,
            "torch_dtype": dtype,
            "local_files_only": self.config.local_files_only,
            "trust_remote_code": False,
        }
        if self.config.attn_implementation:
            model_kwargs["attn_implementation"] = self.config.attn_implementation
        model = AutoModel.from_pretrained(self.config.model_name, **model_kwargs)
        model.eval()
        model.to(device)
        native_dimension = int(model.config.hidden_size)
        if self.config.output_dimension > native_dimension:
            raise QwenEnvironmentError("requested dimension exceeds model hidden size")
        if (
            self.config.output_dimension < native_dimension
            and not self.config.enable_provisional_mrl_truncation
        ):
            raise QwenEnvironmentError(
                "reduced dimension requires explicit provisional MRL truncation authorization"
            )
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        self._inference_dtype_name = dtype_name
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        self._transformers_version = transformers.__version__

    def _format(self, text: str) -> str:
        if not self.config.instruction:
            return text
        return f"Instruct: {self.config.instruction}\nQuery:{text}"

    @staticmethod
    def _last_token_pool(last_hidden_state: Any, attention_mask: Any, torch: Any) -> Any:
        left_padded = bool(attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padded:
            return last_hidden_state[:, -1]
        lengths = attention_mask.sum(dim=1) - 1
        rows = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
        return last_hidden_state[rows, lengths]

    def _update_memory(self) -> None:
        try:
            import psutil

            self.stats.peak_system_rss_bytes = max(
                self.stats.peak_system_rss_bytes,
                int(psutil.Process().memory_info().rss),
            )
        except ImportError:
            pass
        if self._device.type == "cuda":
            self.stats.peak_gpu_allocated_bytes = max(
                self.stats.peak_gpu_allocated_bytes,
                int(self._torch.cuda.max_memory_allocated(self._device)),
            )
            self.stats.peak_gpu_reserved_bytes = max(
                self.stats.peak_gpu_reserved_bytes,
                int(self._torch.cuda.max_memory_reserved(self._device)),
            )

    def _encode_batch(self, units: Sequence[EligibleTextUnit]) -> np.ndarray:
        texts: list[str] = []
        for unit in units:
            if cleaned_text_content_hash(unit.cleaned_text) != unit.content_hash:
                raise EncoderError("eligible unit content hash drifted before Qwen encoding")
            texts.append(self._format(unit.cleaned_text))
        lengths_payload = self._tokenizer(
            texts,
            padding=False,
            truncation=False,
            add_special_tokens=True,
            return_length=True,
        )
        lengths = [int(value) for value in lengths_payload["length"]]
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with self._torch.inference_mode():
            output = self._model(**encoded)
            # 1) last-token pool  2) MRL truncate  3) L2 normalize (after truncate)
            vectors = self._last_token_pool(
                output.last_hidden_state, encoded["attention_mask"], self._torch
            )
            vectors = vectors[:, : self.config.output_dimension]
            vectors = self._torch.nn.functional.normalize(vectors, p=2, dim=1)
        array = self._validate_unit_vectors(
            vectors.detach().to(self._torch.float32).cpu().numpy(),
            expected_rows=len(units),
        )
        self.stats.encoded_units += len(units)
        self.stats.token_count += sum(min(value, self.config.max_length) for value in lengths)
        self.stats.truncated_units += sum(value > self.config.max_length for value in lengths)
        self.stats.token_lengths.extend(lengths)
        self.stats.batches += 1
        self.stats.observed_batch_sizes.append(len(units))
        self._update_memory()
        return array

    def _validate_unit_vectors(self, array: np.ndarray, *, expected_rows: int) -> np.ndarray:
        """Hard-fail if dimension, finiteness, or unit-norm contracts are violated."""

        if array.dtype != np.float32:
            array = np.asarray(array, dtype=np.float32)
        if array.shape != (expected_rows, self.config.output_dimension):
            raise EncoderError("Qwen output dimension validation failed")
        if not np.all(np.isfinite(array)):
            raise EncoderError("Qwen output contains NaN or Inf")
        norms = np.linalg.norm(array.astype(np.float64), axis=1)
        atol = float(self.config.normalized_atol)
        if not np.allclose(norms, 1.0, rtol=0.0, atol=atol):
            raise EncoderError(
                "Qwen normalized-vector norm validation failed "
                f"(atol={atol}; min={float(norms.min()):.8f}; max={float(norms.max()):.8f})"
            )
        return array

    def _is_cuda_oom(self, exc: BaseException) -> bool:
        """Accept real torch.cuda.OutOfMemoryError and compatible test doubles."""

        candidates: list[type[BaseException]] = []
        for owner_name in ("cuda",):
            owner = getattr(self._torch, owner_name, None)
            oom_type = getattr(owner, "OutOfMemoryError", None)
            if isinstance(oom_type, type) and issubclass(oom_type, BaseException):
                candidates.append(oom_type)
        oom_type = getattr(self._torch, "OutOfMemoryError", None)
        if isinstance(oom_type, type) and issubclass(oom_type, BaseException):
            candidates.append(oom_type)
        if candidates and isinstance(exc, tuple(candidates)):
            return True
        message = str(exc).lower()
        return "out of memory" in message or message.strip() == "oom"

    def encode(self, units: Sequence[EligibleTextUnit]) -> np.ndarray:
        self.load()
        if not units:
            return np.empty((0, self.config.output_dimension), dtype=np.float32)
        outputs: list[np.ndarray] = []
        cursor = 0
        batch_size = min(self.config.batch_size, len(units))
        retries = 0
        started = time.perf_counter()
        try:
            while cursor < len(units):
                current = units[cursor : cursor + batch_size]
                try:
                    outputs.append(self._encode_batch(current))
                    cursor += len(current)
                    retries = 0
                except Exception as exc:
                    if not self._is_cuda_oom(exc):
                        raise
                    self.stats.oom_retries += 1
                    retries += 1
                    if self._device.type == "cuda":
                        self._torch.cuda.empty_cache()
                    if batch_size == 1 or retries > self.config.max_oom_retries:
                        raise QwenOutOfMemoryError(
                            "Qwen inference exhausted bounded OOM retries"
                        ) from exc
                    batch_size = max(1, batch_size // 2)
        finally:
            self.stats.elapsed_seconds += time.perf_counter() - started
        return np.concatenate(outputs, axis=0)

    def runtime_report(self) -> Dict[str, Any]:
        report = self.stats.report()
        report.update(
            {
                "model": self.metadata.payload(),
                "model_hash": self.metadata.model_hash,
                "inference_dtype": self._inference_dtype_name,
                "device": str(self._device) if self._device is not None else self.config.device,
                "python_version": platform.python_version(),
                "torch_version": getattr(self._torch, "__version__", None),
                "transformers_version": getattr(self, "_transformers_version", None),
                "max_length": self.config.max_length,
            }
        )
        return report


__all__ = [
    "Qwen3Encoder",
    "QwenEnvironmentError",
    "QwenOutOfMemoryError",
    "QwenRuntimeStats",
]
