# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.vocab_parallel_embedding as embedding_module
import vllm.v1.spec_decode.llm_base_proposer as proposer_module
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

from vllm.model_executor.layers.quantization.exl3 import (
    Exl3Config,
    Exl3Int8EmbeddingMethod,
    _encode_int8_embedding,
)
from vllm.v1.worker.gpu.spec_decode.eagle.utils import _should_share


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


def _int8_embedding(
    q_weight: torch.Tensor,
    embed_scale: torch.Tensor,
) -> torch.nn.Module:
    embedding = torch.nn.Module()
    embedding.register_parameter(
        "weight",
        torch.nn.Parameter(
            torch.empty((0, q_weight.shape[-1]), dtype=torch.bfloat16), False
        ),
    )
    embedding.register_buffer("q_weight", q_weight.clone())
    embedding.register_buffer("embed_scale", embed_scale.clone())
    return embedding


def _share_native_embedding(
    monkeypatch: pytest.MonkeyPatch,
    draft: torch.nn.Module,
    target: torch.nn.Module,
    has_own_embed_tokens: bool | None,
) -> torch.nn.Module:
    monkeypatch.setattr(
        proposer_module,
        "get_pp_group",
        lambda: SimpleNamespace(world_size=1),
    )
    draft_model = SimpleNamespace(model=SimpleNamespace(embed_tokens=draft))
    if has_own_embed_tokens is not None:
        draft_model.has_own_embed_tokens = has_own_embed_tokens
    proposer = SimpleNamespace(model=draft_model)
    target_language_model = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=target)
    )

    proposer_module.SpecDecodeBaseProposer._maybe_share_embeddings(
        proposer, target_language_model
    )
    return draft_model.model.embed_tokens


def test_quantized_embedding_comparison_uses_rows_and_scales() -> None:
    q_weight = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int8)
    scales = torch.tensor([0.25, 0.5], dtype=torch.float16)
    target = _int8_embedding(q_weight, scales)
    equal = _int8_embedding(q_weight, scales)
    distinct_rows = _int8_embedding(q_weight.clone(), scales)
    distinct_rows.q_weight[1, 2] += 1
    distinct_scales = _int8_embedding(q_weight, scales.clone())
    distinct_scales.embed_scale[0] *= 2
    mtp = SimpleNamespace(has_own_embed_tokens=False)
    wrong_width = _int8_embedding(torch.ones((2, 4), dtype=torch.int8), scales)
    owner = SimpleNamespace(has_own_embed_tokens=True)

    assert _should_share(owner, "has_own_embed_tokens", equal, target)
    assert not _should_share(owner, "has_own_embed_tokens", distinct_rows, target)
    assert not _should_share(owner, "has_own_embed_tokens", distinct_scales, target)
    assert not _should_share(owner, "has_own_embed_tokens", None, target)
    assert _should_share(
        mtp,
        "has_own_embed_tokens",
        equal,
        target,
        validate_embedding_compatibility=True,
    )
    assert not _should_share(
        mtp,
        "has_own_embed_tokens",
        wrong_width,
        target,
        validate_embedding_compatibility=True,
    )
    assert torch.equal(equal.weight, distinct_rows.weight)


def test_native_eagle_shares_only_equal_quantized_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q_weight = torch.tensor([[1, 2], [3, 4]], dtype=torch.int8)
    scales = torch.tensor([0.5, 0.25], dtype=torch.float16)
    target = _int8_embedding(q_weight, scales)
    equal = _int8_embedding(q_weight, scales)
    distinct = _int8_embedding(q_weight, scales)
    distinct.q_weight[0, 0] += 1

    shared = _share_native_embedding(monkeypatch, equal, target, True)
    kept = _share_native_embedding(monkeypatch, distinct, target, True)

    assert shared is target
    assert kept is distinct


def test_native_mtp_sharing_validates_width_and_quantization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q_weight = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int8)
    scales = torch.tensor([0.5, 0.25], dtype=torch.float16)
    target = _int8_embedding(q_weight, scales)
    compatible = _int8_embedding(q_weight + 1, scales)
    wrong_width = _int8_embedding(torch.ones((2, 4), dtype=torch.int8), scales)
    dense = torch.nn.Embedding(2, 3)

    shared = _share_native_embedding(monkeypatch, compatible, target, None)
    kept_width = _share_native_embedding(monkeypatch, wrong_width, target, None)
    kept_quantization = _share_native_embedding(monkeypatch, dense, target, None)

    assert shared is target
    assert kept_width is wrong_width
    assert kept_quantization is dense


def test_embedding_dequantizes_multidimensional_ids_and_zero_rows() -> None:
    method = Exl3Int8EmbeddingMethod()
    embedding = torch.nn.Module()
    weight = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, -1.0, 0.5], [2.0, 4.0, -2.0]],
        dtype=torch.float32,
    )
    embedding.register_parameter(
        "weight", torch.nn.Parameter(weight.clone(), requires_grad=False)
    )
    method.process_weights_after_loading(embedding)
    input_ids = torch.tensor([[0, 2], [1, 0]])

    output = method.embedding(embedding, input_ids)
    expected = _dequantize(embedding.q_weight, embedding.embed_scale)[input_ids]

    assert output.shape == (2, 2, 3)
    assert torch.equal(output, expected)
    assert torch.count_nonzero(output[input_ids == 0]) == 0


def test_vocab_parallel_int8_embedding_masks_and_all_reduces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_EXL3_EMBED_ONLINE_BITS", "8")
    monkeypatch.setattr(
        embedding_module, "get_tensor_model_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(embedding_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        embedding_module, "get_virtual_tp_vocab_padding_size", lambda size: size
    )
    def eager_mask(
        input_: torch.Tensor,
        org_start: int,
        org_end: int,
        org_padding: int,
        added_start: int,
        added_end: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        org_mask = (input_ >= org_start) & (input_ < org_end)
        added_mask = (input_ >= added_start) & (input_ < added_end)
        added_offset = added_start - (org_end - org_start) - org_padding
        valid_offset = (org_start * org_mask) + (added_offset * added_mask)
        valid_mask = org_mask | added_mask
        return valid_mask * (input_ - valid_offset), ~valid_mask

    monkeypatch.setattr(embedding_module, "get_masked_input_and_mask", eager_mask)
    local_outputs: list[torch.Tensor] = []
    peer_output = torch.tensor(
        [[[0.0, 0.0], [5.0, 6.0]], [[0.0, 0.0], [7.0, 8.0]]]
    )

    def fake_all_reduce(output: torch.Tensor) -> torch.Tensor:
        local_outputs.append(output.clone())
        return output + peer_output

    monkeypatch.setattr(
        embedding_module, "tensor_model_parallel_all_reduce", fake_all_reduce
    )
    embedding = VocabParallelEmbedding(
        4,
        2,
        params_dtype=torch.float32,
        padding_size=1,
        quant_config=Exl3Config(),
        prefix="model.embed_tokens",
    )
    assert isinstance(embedding.quant_method, Exl3Int8EmbeddingMethod)
    embedding.weight.data.zero_()
    embedding.quant_method.process_weights_after_loading(embedding)
    embedding.q_weight.copy_(
        torch.tensor([[2, 4], [6, 8]], dtype=torch.int8)
    )
    embedding.embed_scale.fill_(0.5)

    output = embedding(torch.tensor([[0, 2], [1, 3]]))

    assert torch.equal(
        local_outputs[0],
        torch.tensor([[[1.0, 2.0], [0.0, 0.0]], [[3.0, 4.0], [0.0, 0.0]]]),
    )
    assert torch.equal(
        output,
        torch.tensor([[[1.0, 2.0], [5.0, 6.0]], [[3.0, 4.0], [7.0, 8.0]]]),
    )


def test_tied_embeddings_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_EXL3_EMBED_ONLINE_BITS", "8")
    hf_config = SimpleNamespace(tie_word_embeddings=True)

    with pytest.raises(ValueError, match="tied word embeddings"):
        Exl3Config._require_untied_int8_embedding(hf_config)
    with pytest.raises(ValueError, match="tied word embeddings"):
        Exl3Int8EmbeddingMethod().tie_weights(torch.nn.Module(), torch.nn.Module())
