"""Focused Qwen encoder tests using mocks only.

Status: NOT_EXECUTED in the authoring Studio.
No Hugging Face download. No CUDA inference.
Labels: IMPLEMENTED_NOT_EXECUTED / TRANSFER_PENDING_VALIDATION.
"""
from __future__ import annotations

import json
from pathlib import Path
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


def test_resolve_hf_token_requires_when_downloading(monkeypatch: pytest.MonkeyPatch) -> None:
    from tdmec_embeddings.qwen_encoder import resolve_hf_token

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    assert resolve_hf_token(require=False) is None
    with pytest.raises(QwenEnvironmentError, match="HF_TOKEN"):
        resolve_hf_token(require=True)
    monkeypatch.setenv("HF_TOKEN", "hf_test_token_not_real")
    assert resolve_hf_token(require=True) == "hf_test_token_not_real"


def test_hub_auth_kwargs_pass_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test_token_not_real")
    encoder = Qwen3Encoder(_config(local_files_only=False))
    kwargs = encoder._hub_auth_kwargs()
    assert kwargs == {"token": "hf_test_token_not_real"}
    assert encoder._hf_token_provided is True


def test_hub_auth_kwargs_refuse_anonymous_download(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    encoder = Qwen3Encoder(_config(local_files_only=False))
    with pytest.raises(QwenEnvironmentError, match="HF_TOKEN"):
        encoder._hub_auth_kwargs()


def test_hub_auth_allows_offline_without_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    encoder = Qwen3Encoder(_config(local_files_only=True))
    assert encoder._hub_auth_kwargs() == {}
    assert encoder._hf_token_provided is False


def test_model_dtype_kwargs_fallback_without_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    from tdmec_embeddings.qwen_encoder import model_dtype_load_kwargs
    import sys

    # Ensure missing transformers.modeling_utils triggers torch_dtype fallback
    monkeypatch.setitem(sys.modules, "transformers", None)
    monkeypatch.setitem(sys.modules, "transformers.modeling_utils", None)
    out = model_dtype_load_kwargs("fp16")
    assert out == {"torch_dtype": "fp16"}


def test_model_dtype_kwargs_prefer_dtype_when_signature_has_it(monkeypatch: pytest.MonkeyPatch) -> None:
    from tdmec_embeddings.qwen_encoder import model_dtype_load_kwargs
    import sys
    import types

    class FakePretrained:
        @staticmethod
        def from_pretrained(*args, dtype=None, torch_dtype=None, **kwargs):
            return None

    modeling = types.ModuleType("transformers.modeling_utils")
    modeling.PreTrainedModel = FakePretrained
    transformers_mod = types.ModuleType("transformers")
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)
    monkeypatch.setitem(sys.modules, "transformers.modeling_utils", modeling)
    out = model_dtype_load_kwargs("fp16")
    assert out == {"dtype": "fp16"}


def test_load_passes_token_revision_and_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test_token_not_real")
    encoder = Qwen3Encoder(_config(local_files_only=False, allow_cpu=True, device="cpu"))

    captured: dict = {}

    class FakeTok:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            captured["tokenizer"] = kwargs
            return object()

    class FakeModel:
        config = SimpleNamespace(hidden_size=8)

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            captured["model"] = kwargs
            inst = cls()
            return inst

        def eval(self):
            return self

        def to(self, device):
            return self

    fake_transformers = SimpleNamespace(__version__="4.56.0-test")
    fake_torch = SimpleNamespace(
        __version__="2.0-test",
        float32="fp32",
        bfloat16="bf16",
        float16="fp16",
        device=lambda name: SimpleNamespace(type="cpu"),
        cuda=SimpleNamespace(is_available=lambda: False),
    )

    def patched_load(self):
        device, dtype, dtype_name = self._resolve_device_and_dtype(fake_torch)
        auth_kwargs = self._hub_auth_kwargs()
        dtype_kwargs = {"dtype": dtype}
        self._model_dtype_kwarg = "dtype"
        FakeTok.from_pretrained(
            self.config.model_name,
            revision=self.config.tokenizer_revision,
            padding_side="left",
            local_files_only=self.config.local_files_only,
            trust_remote_code=False,
            **auth_kwargs,
        )
        FakeModel.from_pretrained(
            self.config.model_name,
            revision=self.config.model_revision,
            local_files_only=self.config.local_files_only,
            trust_remote_code=False,
            **dtype_kwargs,
            **auth_kwargs,
        )
        self._torch = fake_torch
        self._tokenizer = object()
        self._model = FakeModel()
        self._device = SimpleNamespace(type="cpu")
        self._inference_dtype_name = dtype_name
        self._transformers_version = fake_transformers.__version__

    monkeypatch.setattr(Qwen3Encoder, "load", patched_load)
    encoder.load()
    assert captured["tokenizer"]["token"] == "hf_test_token_not_real"
    assert captured["tokenizer"]["revision"] == encoder.config.tokenizer_revision
    assert captured["model"]["token"] == "hf_test_token_not_real"
    assert captured["model"]["revision"] == encoder.config.model_revision
    assert "dtype" in captured["model"]
    assert "torch_dtype" not in captured["model"]
    report = encoder.runtime_report()
    assert report["hf_token_provided"] is True
    assert report["model_dtype_kwarg"] == "dtype"
    assert "hf_test_token_not_real" not in str(report)


def test_l2_normalize_rows_float32_restores_unit_norm() -> None:
    from tdmec_embeddings.qwen_encoder import l2_normalize_rows_float32

    # Simulate fp16-like residuals after cast
    raw = np.asarray(
        [
            [0.5, 0.5, 0.5, 0.5],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    raw[0] *= np.float32(1.0003)
    out = l2_normalize_rows_float32(raw)
    norms = np.linalg.norm(out.astype(np.float64), axis=1)
    assert np.allclose(norms, 1.0, rtol=0.0, atol=1e-6)


def test_replace_incomplete_run_policy(tmp_path: Path) -> None:
    from tdmec_embeddings.run_recovery import (
        RunRecoveryError,
        assess_embedding_run,
        prepare_embedding_run_root,
        replace_incomplete_run,
    )

    run = tmp_path / "out" / "run1"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "node_text.json").write_text(
        json.dumps({"status": "IN_PROGRESS"}), encoding="utf-8"
    )
    assessment = assess_embedding_run(run)
    assert assessment["incomplete"] is True
    assert assessment["safe_to_replace"] is True

    with pytest.raises(RunRecoveryError, match="replace-incomplete"):
        prepare_embedding_run_root(
            output_root=tmp_path / "out",
            embedding_run_id="run1",
            resume=False,
            replace_incomplete=False,
        )

    result = prepare_embedding_run_root(
        output_root=tmp_path / "out",
        embedding_run_id="run1",
        resume=False,
        replace_incomplete=True,
    )
    assert result["actions"] == ["replaced_incomplete"]
    assert not run.exists()

    # Completed runs cannot be replaced
    done = tmp_path / "out" / "run2"
    done.mkdir(parents=True)
    (done / "embedding_manifest.json").write_text(
        json.dumps({"status": "COMPLETED"}), encoding="utf-8"
    )
    (done / "manifests").mkdir()
    for modality in ("node_text", "event_text"):
        (done / "manifests" / f"{modality}.json").write_text(
            json.dumps({"status": "COMPLETED"}), encoding="utf-8"
        )
    with pytest.raises(RunRecoveryError, match="COMPLETED"):
        replace_incomplete_run(done)


STATUS = "NOT_EXECUTED"
