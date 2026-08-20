# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization.exl3 import (
    Exl3Config,
    Exl3Int8EmbeddingMethod,
    _encode_int8_embedding,
)


def _dequantize(q_weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return q_weight.to(torch.float32) * scales.to(torch.float32).unsqueeze(1)


def test_int8_integer_round_trip_is_exact() -> None:
    expected = torch.arange(-127, 128, dtype=torch.int16).to(torch.int8).unsqueeze(0)
    weight = expected.to(torch.float32) * 0.25

    q_weight, scales = _encode_int8_embedding(weight, chunk_rows=1)

    assert q_weight.dtype == torch.int8
    assert scales.dtype == torch.float16
    assert torch.equal(q_weight, expected)
    assert torch.equal(_dequantize(q_weight, scales), weight)


def test_int8_float_round_trip_respects_rowwise_error_bound() -> None:
    generator = torch.Generator().manual_seed(1234)
    weight = torch.randn((7, 19), generator=generator, dtype=torch.float32) * 3.0

    q_weight, scales = _encode_int8_embedding(weight, chunk_rows=3)
    reconstructed = _dequantize(q_weight, scales)
    exact_scales = weight.abs().amax(dim=1) / 127.0
    scale_rounding = (scales.float() - exact_scales).abs() * 127.0
    bound = exact_scales / 2.0 + scale_rounding + 1.0e-6

    assert torch.all((reconstructed - weight).abs() <= bound.unsqueeze(1))


def test_int8_encoder_handles_zero_rows_and_empty_tables() -> None:
    weight = torch.tensor([[0.0, 0.0, 0.0], [1.0, -1.0, 0.5]])

    q_weight, scales = _encode_int8_embedding(weight, chunk_rows=1)
    empty_q, empty_scales = _encode_int8_embedding(torch.empty((0, 3)))

    assert torch.equal(q_weight[0], torch.zeros(3, dtype=torch.int8))
    assert scales[0].isfinite()
    assert torch.equal(_dequantize(q_weight, scales)[0], weight[0])
    assert empty_q.shape == (0, 3)
    assert empty_scales.shape == (0,)


def test_int8_chunking_is_bit_identical() -> None:
    weight = torch.linspace(-5.0, 7.0, 11 * 13).reshape(11, 13)

    q_one, scale_one = _encode_int8_embedding(weight, chunk_rows=1)
    q_four, scale_four = _encode_int8_embedding(weight, chunk_rows=4)
    q_full, scale_full = _encode_int8_embedding(weight, chunk_rows=64)

    assert torch.equal(q_one, q_four)
    assert torch.equal(q_one, q_full)
    assert torch.equal(scale_one, scale_four)
    assert torch.equal(scale_one, scale_full)


def test_quant_method_targets_exact_embedding_type_and_excludes_lm_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedding_type = type("VocabParallelEmbedding", (torch.nn.Module,), {})
    lm_head_type = type("ParallelLMHead", (embedding_type,), {})
    embedding_subclass = type("EmbeddingSubclass", (embedding_type,), {})
    config = Exl3Config()
    monkeypatch.setenv("VLLM_EXL3_EMBED_ONLINE_BITS", "8")

    assert isinstance(
        config.get_quant_method(embedding_type(), "model.embed_tokens"),
        Exl3Int8EmbeddingMethod,
    )
    assert not isinstance(
        config.get_quant_method(lm_head_type(), "lm_head"),
        Exl3Int8EmbeddingMethod,
    )
    assert config.get_quant_method(embedding_subclass(), "model.embed_tokens") is None


def test_quant_method_is_inert_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    embedding_type = type("VocabParallelEmbedding", (torch.nn.Module,), {})
    monkeypatch.delenv("VLLM_EXL3_EMBED_ONLINE_BITS", raising=False)

    assert Exl3Config().get_quant_method(embedding_type(), "model.embed_tokens") is None


def _embedding_call_keywords(path: Path, class_name: str) -> set[str]:
    tree = ast.parse(path.read_text())
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    init = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    call = next(
        node
        for node in ast.walk(init)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "VocabParallelEmbedding"
    )
    return {keyword.arg for keyword in call.keywords if keyword.arg is not None}


def test_qwen_target_and_mtp_wire_quant_config_and_prefix() -> None:
    models = Path(__file__).parents[2] / "vllm" / "model_executor" / "models"

    assert _embedding_call_keywords(models / "qwen3_5.py", "Qwen3_5Model") >= {
        "quant_config",
        "prefix",
    }
    assert _embedding_call_keywords(
        models / "qwen3_5_mtp.py", "Qwen3_5MultiTokenPredictor"
    ) >= {"quant_config", "prefix"}


def test_post_load_surface_preserves_native_mtp_module_sharing() -> None:
    method = Exl3Int8EmbeddingMethod()
    embedding = torch.nn.Module()
    embedding.register_parameter(
        "weight",
        torch.nn.Parameter(torch.arange(24, dtype=torch.bfloat16).reshape(6, 4), False),
    )

    method.process_weights_after_loading(embedding)
    target = SimpleNamespace(embed_tokens=embedding)
    draft = SimpleNamespace(embed_tokens=target.embed_tokens)

    assert embedding.weight.shape == (0, 4)
    assert draft.embed_tokens is target.embed_tokens
    assert draft.embed_tokens.q_weight is target.embed_tokens.q_weight
    assert draft.embed_tokens.embed_scale is target.embed_tokens.embed_scale


def test_tied_embeddings_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_EXL3_EMBED_ONLINE_BITS", "8")
    hf_config = SimpleNamespace(tie_word_embeddings=True)

    with pytest.raises(ValueError, match="tied word embeddings"):
        Exl3Config._require_untied_int8_embedding(hf_config)
    with pytest.raises(ValueError, match="tied word embeddings"):
        Exl3Int8EmbeddingMethod().tie_weights(torch.nn.Module(), torch.nn.Module())
