"""Lazy Qwen3 embedding backend for authorized target-Studio / Kaggle execution.

IMPLEMENTED_NOT_EXECUTED / TRANSFER_PENDING_VALIDATION.

Canonical unit-vector pipeline (no bypass allowed):

    text → tokenize → Qwen3 forward → last-token pool
      → MRL truncate to ``output_dimension``
      → L2 normalize
      → finite + unit-norm validation
      → float32 numpy

Model / tokenizer loading is authenticated via ``HF_TOKEN`` (or
``HUGGING_FACE_HUB_TOKEN``). Anonymous Hub access is not assumed.

Imports and model loading are deliberately lazy; importing this module cannot
download a model.  Norm validation failures raise ``EncoderError``.
"""
from __future__ import annotations

import inspect
import os
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


_HF_TOKEN_ENV_KEYS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def resolve_hf_token(*, require: bool = False) -> Optional[str]:
    """Return the first non-empty Hugging Face Hub token from the environment.

    Tokens are never logged or returned in runtime reports; callers only receive
    the secret string for ``from_pretrained(..., token=...)``.
    """

    for key in _HF_TOKEN_ENV_KEYS:
        value = os.environ.get(key, "").strip()
        if value:
            return value
    if require:
        raise QwenEnvironmentError(
            "HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) is required for authenticated "
            "Hugging Face Hub model loading; anonymous download is not supported"
        )
    return None


def model_dtype_load_kwargs(dtype: Any) -> Dict[str, Any]:
    """Choose ``dtype`` vs deprecated ``torch_dtype`` for the installed Transformers.

    Newer Transformers (≈4.56+) prefer ``dtype`` and warn on ``torch_dtype``.
    Older releases only accept ``torch_dtype``. Signature inspection keeps the
    encoder compatible across the pinned ``transformers>=4.51,<5`` range.
    """

    try:
        from transformers.modeling_utils import PreTrainedModel

        parameters = inspect.signature(PreTrainedModel.from_pretrained).parameters
        if "dtype" in parameters:
            return {"dtype": dtype}
    except Exception:
        pass
    return {"torch_dtype": dtype}


def l2_normalize_rows_float32(array: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize in float32 for published unit embeddings.

    GPU fp16/bf16 ``F.normalize`` can leave residuals of order 1e-4–1e-3 after
    cast to float32. Re-normalizing in float32 makes ``||v||_2 = 1`` in the
    storage dtype before validation and Parquet write.
    """

    vectors = np.asarray(array, dtype=np.float32)
    if vectors.ndim != 2:
        raise EncoderError("L2 normalize expects a rank-2 embedding matrix")
    if vectors.shape[0] == 0:
        return vectors
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise EncoderError("cannot L2-normalize non-finite or zero embedding rows")
    return np.asarray(vectors / norms.astype(np.float32), dtype=np.float32)


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
            normalized_atol=float(config.normalized_atol),
            instruction_hash=hash_canonical({"instruction": config.instruction}),
            backend="qwen3_transformers_last_token",
        )
        self.stats = QwenRuntimeStats()
        self._torch: Any = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._device: Any = None
        self._inference_dtype_name: Optional[str] = None
        self._hf_token_provided: bool = False
        self._model_dtype_kwarg: Optional[str] = None

    def _hub_auth_kwargs(self) -> Dict[str, Any]:
        """Authenticated Hub kwargs for tokenizer/model ``from_pretrained`` calls."""

        # Offline/cache-only runs may omit the token; Hub downloads must authenticate.
        token = resolve_hf_token(require=not self.config.local_files_only)
        self._hf_token_provided = token is not None
        kwargs: Dict[str, Any] = {}
        if token is not None:
            kwargs["token"] = token
        return kwargs

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
        auth_kwargs = self._hub_auth_kwargs()
        dtype_kwargs = model_dtype_load_kwargs(dtype)
        self._model_dtype_kwarg = next(iter(dtype_kwargs.keys()))
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            revision=self.config.tokenizer_revision,
            padding_side="left",
            local_files_only=self.config.local_files_only,
            trust_remote_code=False,
            **auth_kwargs,
        )
        model_kwargs: Dict[str, Any] = {
            "revision": self.config.model_revision,
            "local_files_only": self.config.local_files_only,
            "trust_remote_code": False,
            **dtype_kwargs,
            **auth_kwargs,
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
            # 1) last-token pool  2) MRL truncate  3) L2 in inference dtype
            vectors = self._last_token_pool(
                output.last_hidden_state, encoded["attention_mask"], self._torch
            )
            vectors = vectors[:, : self.config.output_dimension]
            vectors = self._torch.nn.functional.normalize(vectors, p=2, dim=1)
        # Cast to float32 then re-L2 so published vectors are unit-norm in the
        # storage dtype.  fp16/bf16 GPU normalize alone can leave ||v||≈1±3e-4.
        array = vectors.detach().to(self._torch.float32).cpu().numpy()
        array = l2_normalize_rows_float32(array)
        array = self._validate_unit_vectors(array, expected_rows=len(units))
        self.stats.encoded_units += len(units)
        self.stats.token_count += sum(min(value, self.config.max_length) for value in lengths)
        self.stats.truncated_units += sum(value > self.config.max_length for value in lengths)
        self.stats.token_lengths.extend(lengths)
        self.stats.batches += 1
        self.stats.observed_batch_sizes.append(len(units))
        self._update_memory()
        return array

    def _validate_unit_vectors(self, array: np.ndarray, *, expected_rows: int) -> np.ndarray:
        """Hard-fail if dimension, finiteness, or unit-norm contracts are violated.

        Validation always runs **after** float32 re-L2. Default atol=1e-6 is
        appropriate for float32 unit vectors (far tighter than raw fp16 residuals
        of ~1e-3–1e-4, which are eliminated by the re-normalization step).
        """

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
                "model_dtype_kwarg": self._model_dtype_kwarg,
                "device": str(self._device) if self._device is not None else self.config.device,
                "python_version": platform.python_version(),
                "torch_version": getattr(self._torch, "__version__", None),
                "transformers_version": getattr(self, "_transformers_version", None),
                "max_length": self.config.max_length,
                "hf_token_provided": bool(self._hf_token_provided),
                "local_files_only": bool(self.config.local_files_only),
                "unit_normalized": True,
                "normalized_atol": float(self.config.normalized_atol),
                "float32_renorm_after_cast": True,
                "hf_home_set": bool(os.environ.get("HF_HOME", "").strip()),
                "huggingface_hub_cache_set": bool(
                    os.environ.get("HUGGINGFACE_HUB_CACHE", "").strip()
                ),
            }
        )
        return report


__all__ = [
    "Qwen3Encoder",
    "QwenEnvironmentError",
    "QwenOutOfMemoryError",
    "QwenRuntimeStats",
    "l2_normalize_rows_float32",
    "model_dtype_load_kwargs",
    "resolve_hf_token",
]
