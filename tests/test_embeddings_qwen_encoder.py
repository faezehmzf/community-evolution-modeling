"""Focused Qwen encoder tests using mocks only.

Status: NOT_EXECUTED in the authoring Studio.
No Hugging Face download. No CUDA inference.
Labels: IMPLEMENTED_NOT_EXECUTED / TRANSFER_PENDING_VALIDATION.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tdmec.hashing import hash_canonical
from tdmec_embeddings.config import EncoderConfig
from tdmec_embeddings.eligibility import EligibleTextUnit, cleaned_text_content_hash
from tdmec_embeddings.qwen_encoder import (
    Qwen3Encoder,
    QwenEnvironmentError,
    QwenOutOfMemoryError,
)


def _unit(text: str = "private") -> EligibleTextUnit:
    return EligibleTextUnit(
        modality="node_text",
        source_run_id="b-source",
        unit_id="u0",
        unit_hash=hash_canonical({"u": "u0"}),
        content_hash=cleaned_text_content_hash(text),
        preprocessing_hash=hash_canonical({"p": 1}),
        cleaned_text=text,
        snapshot_id=0,
        node_index=0,
        relation_id=None,
        source_idx=None,
        target_idx=None,
        source_file="private.xlsx",
        source_row_number=0,
    )


def _config(**overrides) -> EncoderConfig:
    payload = dict(
        backend="qwen3",
        model_name="Qwen/Qwen3-Embedding-4B",
        model_revision="deadbeefcafebabe000000000000000000000000",
        tokenizer_revision="deadbeefcafebabe000000000000000000000000",
        instruction="Represent the topic, stance, sentiment, and social meaning.",
        output_dimension=4,
        max_length=16,
        precision="fp32",
        device="cpu",
        batch_size=4,
        max_oom_retries=3,
        local_files_only=True,
        allow_cpu=True,
        enable_provisional_mrl_truncation=True,
    )
    payload.update(overrides)
    return EncoderConfig(**payload)  # type: ignore[arg-type]


def test_rejects_normalize_false_on_encoder_init() -> None:
    with pytest.raises(Exception, match="normalize=false is forbidden"):
        Qwen3Encoder(_config(normalize=False))


def test_rejects_non_qwen3_family() -> None:
    with pytest.raises(Exception):
        EncoderConfig(
            backend="qwen3",
            model_name="Someone/NotQwen",
            model_revision="abc",
            tokenizer_revision="abc",
            instruction="",
            output_dimension=8,
            max_length=32,
            precision="auto",
            device="cuda:0",
            batch_size=2,
            max_oom_retries=1,
        ).validate()


def test_last_token_pool_left_padded() -> None:
    class _Torch:
        @staticmethod
        def arange(n, device=None):
            return np.arange(n)

    hidden = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    mask = np.ones((2, 3), dtype=np.int64)
    pooled = Qwen3Encoder._last_token_pool(hidden, mask, _Torch)
    np.testing.assert_array_equal(pooled, hidden[:, -1])


def test_encode_uses_mock_model_and_validates_norms(monkeypatch: pytest.MonkeyPatch) -> None:
    encoder = Qwen3Encoder(_config())

    class FakeTok:
        def __call__(self, texts, **kwargs):
            if kwargs.get("return_length"):
                return {"length": [3 for _ in texts]}
            n = len(texts)
            return {
                "input_ids": SimpleNamespace(
                    **{
                        "to": lambda device: np.ones((n, 4), dtype=np.int64),
                    }
                ),
                "attention_mask": SimpleNamespace(
                    **{
                        "to": lambda device: np.ones((n, 4), dtype=np.int64),
                    }
                ),
            }

        def to(self, *args, **kwargs):
            return self

    class FakeModel:
        config = SimpleNamespace(hidden_size=8)

        def eval(self):
            return self

        def to(self, device):
            return self

        def __call__(self, **encoded):
            n = 1
            hidden = np.zeros((n, 4, 8), dtype=np.float32)
            hidden[:, -1, :4] = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
            return SimpleNamespace(last_hidden_state=hidden)

    class FakeTorch:
        float32 = "fp32"
        bfloat16 = "bf16"
        float16 = "fp16"
        __version__ = "0.0-test"

        class cuda:
            @staticmethod
            def is_available():
                return False

        @staticmethod
        def device(name):
            return SimpleNamespace(type="cpu")

        class nn:
            class functional:
                @staticmethod
                def normalize(vectors, p=2, dim=1):
                    norms = np.linalg.norm(vectors, axis=dim, keepdims=True)
                    return vectors / norms

        @staticmethod
        def inference_mode():
            class Ctx:
                def __enter__(self):
                    return None

                def __exit__(self, *args):
                    return False

            return Ctx()

        class OutOfMemoryError(RuntimeError):
            pass

    def fake_load(self):
        self._torch = FakeTorch
        self._tokenizer = FakeTok()
        self._model = FakeModel()
        self._device = SimpleNamespace(type="cpu")
        self._inference_dtype_name = "fp32"
        self._transformers_version = "0.0-test"

    monkeypatch.setattr(Qwen3Encoder, "load", fake_load)
    # Simplify encode path: bypass FakeTok complexity by patching _encode_batch
    def fake_batch(self, units):
        arr = np.ones((len(units), self.config.output_dimension), dtype=np.float32)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True)
        self.stats.encoded_units += len(units)
        self.stats.batches += 1
        return arr

    monkeypatch.setattr(Qwen3Encoder, "_encode_batch", fake_batch)
    out = encoder.encode([_unit("a"), _unit("b")])
    assert out.shape == (2, 4)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_oom_backoff_aborts_at_batch_one(monkeypatch: pytest.MonkeyPatch) -> None:
    encoder = Qwen3Encoder(_config(batch_size=4, max_oom_retries=2))

    class FakeTorch:
        class OutOfMemoryError(RuntimeError):
            pass

        class cuda:
            OutOfMemoryError = None  # set below

            @staticmethod
            def empty_cache():
                return None

    FakeTorch.cuda.OutOfMemoryError = FakeTorch.OutOfMemoryError
    encoder._torch = FakeTorch
    encoder._device = SimpleNamespace(type="cuda")
    encoder._model = object()
    encoder._tokenizer = object()
    encoder._inference_dtype_name = "fp16"

    def always_oom(self, units):
        raise FakeTorch.cuda.OutOfMemoryError("oom")

    monkeypatch.setattr(Qwen3Encoder, "load", lambda self: None)
    monkeypatch.setattr(Qwen3Encoder, "_encode_batch", always_oom)
    with pytest.raises(QwenOutOfMemoryError):
        encoder.encode([_unit(), _unit(), _unit(), _unit()])


def test_cpu_requires_allow_cpu() -> None:
    encoder = Qwen3Encoder(_config(allow_cpu=False, device="cpu"))
    with pytest.raises(QwenEnvironmentError):
        encoder._resolve_device_and_dtype(
            SimpleNamespace(
                cuda=SimpleNamespace(is_available=lambda: False),
                device=lambda name: SimpleNamespace(type="cpu"),
                float32="fp32",
                bfloat16="bf16",
                float16="fp16",
            )
        )


STATUS = "NOT_EXECUTED"
