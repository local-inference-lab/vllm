# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for GLM-5.3 tensor-parallel padding (TP3).

The config hook pads MLA heads, KDA heads and the shared-expert width; the
model loads the logical checkpoint into the padded layouts with zero tails.
These tests emulate each TP rank on CPU by overriding the rank.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import vllm.model_executor.layers.linear as linear_module
import vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn as kda_module
import vllm.model_executor.parameter as parameter_module
from vllm.config.model import ModelConfig
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.models.config import (
    MODELS_CONFIG_MAP,
    Glm5NextForCausalLMConfig,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    PackedvLLMParameter,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.glm5_next import Glm5NextConfig

TP = 3

# ---------------------------------------------------------------------------
# Config hook
# ---------------------------------------------------------------------------


def _glm53_model_config(
    architecture: str = "Glm5NextForConditionalGeneration",
    multimodal_config=None,
):
    hf_config = Glm5NextConfig(
        text_config={
            "num_attention_heads": 64,
            "num_key_value_heads": 64,
            "moe_intermediate_size": 2048,
            "n_shared_experts": 1,
            "n_routed_experts": 288,
            "intermediate_size": 12288,
            "first_k_dense_replace": 3,
            "vocab_size": 154880,
            "linear_attn_config": {
                "head_dim": 128,
                "num_heads": 64,
                "short_conv_kernel_size": 4,
                "gate_lower_bound": -5.0,
            },
        },
        vision_config={"num_heads": 16},
    )
    model_config = SimpleNamespace(
        architecture=architecture,
        hf_config=hf_config,
        hf_text_config=hf_config.text_config,
        multimodal_config=multimodal_config,
        model_arch_config=None,
    )
    model_config.get_model_arch_config = lambda: SimpleNamespace(
        total_num_attention_heads=hf_config.text_config.num_attention_heads
    )
    return model_config


def _parallel_config(tp_size: int, enable_expert_parallel: bool = True):
    return SimpleNamespace(
        tensor_parallel_size=tp_size, enable_expert_parallel=enable_expert_parallel
    )


@pytest.fixture
def cuda_platform(monkeypatch):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)


def test_glm53_architectures_use_the_padding_hook() -> None:
    for architecture in (
        "Glm5NextForCausalLM",
        "Glm5NextForConditionalGeneration",
        "Glm5NextMTPModel",
    ):
        assert MODELS_CONFIG_MAP[architecture] is Glm5NextForCausalLMConfig


@pytest.mark.parametrize("architecture", ["Glm5NextForConditionalGeneration"])
def test_tp3_pads_glm53_geometry(cuda_platform, architecture) -> None:
    model_config = _glm53_model_config(architecture)
    parallel_config = _parallel_config(TP)

    # Idempotent: the hook runs from both verify paths.
    ModelConfig._update_model_config_for_parallelism(model_config, parallel_config)
    ModelConfig._update_model_config_for_parallelism(model_config, parallel_config)

    text = model_config.hf_text_config
    assert (text.num_attention_heads, text.original_num_attention_heads) == (72, 64)
    assert (text.num_key_value_heads, text.original_num_key_value_heads) == (72, 64)
    assert (text.linear_num_heads, text.original_linear_num_heads) == (66, 64)
    assert text.linear_attn_config["num_heads"] == 66
    assert text.linear_attn_config["original_num_heads"] == 64
    assert text.shared_expert_intermediate_size == 2112
    assert text.original_shared_expert_intermediate_size == 2048
    # Routed experts and the dense MLP are not padded.
    assert text.moe_intermediate_size == 2048
    assert text.intermediate_size == 12288
    assert text.vocab_size == 154880
    # The multimodal wrapper forwards to the padded text config.
    assert model_config.hf_config.num_attention_heads == 72
    assert model_config.hf_config.linear_num_heads == 66
    assert model_config.model_arch_config.total_num_attention_heads == 72
    # Per-rank sizes.
    assert 72 // TP == 24 and 24 % 8 == 0
    assert 66 // TP == 22
    assert 2112 // TP == 704 and 704 % 64 == 0


def test_mtp_draft_config_is_padded(cuda_platform) -> None:
    model_config = _glm53_model_config("Glm5NextMTPModel")
    ModelConfig._update_model_config_for_parallelism(model_config, _parallel_config(3))
    assert model_config.hf_text_config.num_attention_heads == 72
    assert model_config.hf_text_config.shared_expert_intermediate_size == 2112


def test_verify_and_update_config_pads_before_hybrid_sizing(cuda_platform) -> None:
    model_config = _glm53_model_config()
    vllm_config = SimpleNamespace(
        model_config=model_config, parallel_config=_parallel_config(TP)
    )
    Glm5NextForCausalLMConfig.verify_and_update_config(vllm_config)
    assert model_config.hf_text_config.linear_num_heads == 66


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
def test_divisible_tp_sizes_are_noops(cuda_platform, tp_size) -> None:
    model_config = _glm53_model_config()
    before = model_config.hf_text_config.to_dict()
    ModelConfig._update_model_config_for_parallelism(
        model_config, _parallel_config(tp_size, enable_expert_parallel=False)
    )
    text = model_config.hf_text_config
    assert text.to_dict() == before
    assert not hasattr(text, "original_num_attention_heads")
    assert not hasattr(text, "shared_expert_intermediate_size")
    assert model_config.model_arch_config is None


def test_tp3_without_expert_parallel_raises(cuda_platform) -> None:
    model_config = _glm53_model_config()
    with pytest.raises(ValueError, match="--enable-expert-parallel"):
        ModelConfig._update_model_config_for_parallelism(
            model_config, _parallel_config(TP, enable_expert_parallel=False)
        )


@pytest.mark.parametrize("language_model_only", [False, True])
def test_tp3_switches_unsplittable_vision_tower_to_data_parallel(
    cuda_platform, language_model_only
) -> None:
    # The tower is built even with --language-model-only, so switch regardless.
    multimodal_config = SimpleNamespace(
        mm_encoder_tp_mode="weights", language_model_only=language_model_only
    )
    model_config = _glm53_model_config(multimodal_config=multimodal_config)
    ModelConfig._update_model_config_for_parallelism(model_config, _parallel_config(TP))
    assert multimodal_config.mm_encoder_tp_mode == "data"
    assert model_config.hf_text_config.num_attention_heads == 72


@pytest.mark.parametrize("tp_size", [2, 4])
def test_splittable_vision_tower_keeps_weights_mode(cuda_platform, tp_size) -> None:
    multimodal_config = SimpleNamespace(mm_encoder_tp_mode="weights")
    ModelConfig._update_model_config_for_parallelism(
        _glm53_model_config(multimodal_config=multimodal_config),
        _parallel_config(tp_size),
    )
    assert multimodal_config.mm_encoder_tp_mode == "weights"


def test_non_cuda_platforms_are_untouched(monkeypatch) -> None:
    monkeypatch.setattr(current_platform, "is_cuda", lambda: False)
    model_config = _glm53_model_config()
    ModelConfig._update_model_config_for_parallelism(model_config, _parallel_config(3))
    assert model_config.hf_text_config.num_attention_heads == 64


def test_tp3_vocab_storage_is_padded_with_zero_rows() -> None:
    vocab, dim = 154880, 4
    checkpoint = torch.randn(vocab, dim)
    shards = []
    for rank in range(TP):
        group = SimpleNamespace(rank_in_group=rank, world_size=TP)
        embedding = VocabParallelEmbedding(vocab, dim, parallel_group=group)
        assert embedding.num_embeddings_padded == 154944
        embedding.weight_loader(embedding.weight, checkpoint)
        shards.append(embedding.weight.data)
    full = torch.cat(shards)
    torch.testing.assert_close(full[:vocab], checkpoint, rtol=0, atol=0)
    assert torch.count_nonzero(full[vocab:]) == 0


# ---------------------------------------------------------------------------
# Padded loading
# ---------------------------------------------------------------------------


@pytest.fixture
def tp_rank(monkeypatch):
    """Emulate a TP rank for layers and parameters built on CPU."""
    state = {"rank": 0}

    def set_rank(rank: int) -> None:
        state["rank"] = rank

    for module in (linear_module, parameter_module, kda_module):
        if hasattr(module, "get_tensor_model_parallel_rank"):
            monkeypatch.setattr(
                module, "get_tensor_model_parallel_rank", lambda: state["rank"]
            )
        if hasattr(module, "get_tensor_model_parallel_world_size"):
            monkeypatch.setattr(
                module, "get_tensor_model_parallel_world_size", lambda: TP
            )
    return set_rank


def _allow_padding(layer: torch.nn.Module) -> torch.nn.Module:
    for param in layer.parameters():
        set_weight_attrs(param, {"allow_tp_padding": True})
    return layer


def _toy_mla_reference(x, wq, wkv, wo, heads: int, head_dim: int):
    """MLA-like attention: head-major Q, interleaved per-head [K, V] rows."""
    tokens = x.shape[0]
    q = (x @ wq.T).view(tokens, heads, head_dim).transpose(0, 1)
    kv = (x @ wkv.T).view(tokens, heads, 2 * head_dim).transpose(0, 1)
    k, v = kv.split(head_dim, dim=-1)
    scores = q @ k.transpose(-1, -2) / head_dim**0.5
    out = torch.softmax(scores, dim=-1) @ v
    return out.transpose(0, 1).reshape(tokens, heads * head_dim) @ wo.T


def test_padded_mla_heads_match_unpadded_attention(tp_rank) -> None:
    torch.manual_seed(0)
    hidden, head_dim, logical_heads, padded_heads = 16, 4, 4, 6
    wq = torch.randn(logical_heads * head_dim, hidden)
    wkv = torch.randn(logical_heads * 2 * head_dim, hidden)
    wo = torch.randn(hidden, logical_heads * head_dim)
    x = torch.randn(5, hidden)
    expected = _toy_mla_reference(x, wq, wkv, wo, logical_heads, head_dim)

    local_heads = padded_heads // TP
    partial_sums = []
    padded_wo_columns = []
    for rank in range(TP):
        tp_rank(rank)
        q_proj = _allow_padding(
            ColumnParallelLinear(hidden, padded_heads * head_dim, bias=False)
        )
        kv_proj = _allow_padding(
            ColumnParallelLinear(hidden, padded_heads * 2 * head_dim, bias=False)
        )
        o_proj = _allow_padding(
            RowParallelLinear(padded_heads * head_dim, hidden, bias=False)
        )
        for layer in (q_proj, kv_proj, o_proj):
            layer.weight.data.fill_(float("nan"))
        q_proj.weight_loader(q_proj.weight, wq)
        kv_proj.weight_loader(kv_proj.weight, wkv)
        o_proj.weight_loader(o_proj.weight, wo)
        padded_wo_columns.append(o_proj.weight.data)

        partial_sums.append(
            _toy_mla_reference(
                x,
                q_proj.weight.data,
                kv_proj.weight.data,
                o_proj.weight.data,
                local_heads,
                head_dim,
            )
        )
        # Rank-local shard boundaries follow the unpadded checkpoint.
        rows = local_heads * head_dim
        available = max(0, min(rows, logical_heads * head_dim - rank * rows))
        torch.testing.assert_close(
            q_proj.weight.data[:available], wq[rank * rows : rank * rows + available]
        )
        assert torch.count_nonzero(q_proj.weight.data[available:]) == 0
        assert torch.count_nonzero(o_proj.weight.data[:, available:]) == 0

    # The all-reduce of the per-rank partial sums equals the unpadded model.
    torch.testing.assert_close(sum(partial_sums), expected)
    full_wo = torch.cat(padded_wo_columns, dim=1)
    torch.testing.assert_close(full_wo[:, : wo.shape[1]], wo, rtol=0, atol=0)
    assert torch.count_nonzero(full_wo[:, wo.shape[1] :]) == 0


def test_padded_shared_expert_matches_unpadded_mlp(tp_rank) -> None:
    torch.manual_seed(0)
    hidden, logical, padded = 8, 20, 24
    gate = torch.randn(logical, hidden)
    up = torch.randn(logical, hidden)
    down = torch.randn(hidden, logical)
    x = torch.randn(3, hidden)
    expected = (F.silu(x @ gate.T) * (x @ up.T)) @ down.T

    outputs = []
    for rank in range(TP):
        tp_rank(rank)
        gate_up = _allow_padding(
            MergedColumnParallelLinear(hidden, [padded] * 2, bias=False)
        )
        down_proj = _allow_padding(RowParallelLinear(padded, hidden, bias=False))
        gate_up.weight.data.fill_(float("nan"))
        down_proj.weight.data.fill_(float("nan"))
        gate_up.weight_loader(gate_up.weight, gate, 0)
        gate_up.weight_loader(gate_up.weight, up, 1)
        down_proj.weight_loader(down_proj.weight, down)
        local_gate, local_up = gate_up.weight.data.chunk(2)
        h = F.silu(x @ local_gate.T) * (x @ local_up.T)
        outputs.append(h @ down_proj.weight.data.T)
        if rank == TP - 1:
            tail = padded // TP - (logical - rank * (padded // TP))
            assert torch.count_nonzero(local_gate[-tail:]) == 0
            assert torch.count_nonzero(local_up[-tail:]) == 0
            assert torch.count_nonzero(down_proj.weight.data[:, -tail:]) == 0

    torch.testing.assert_close(sum(outputs), expected)


def test_nvfp4_packed_row_parallel_weight_and_scales_pad_together(tp_rank) -> None:
    """NVFP4 packs two values per byte and has one scale per 16 inputs."""
    out_features, logical, padded, group = 4, 2048, 2112, 16
    packed = torch.randint(1, 256, (out_features, logical // 2), dtype=torch.uint8)
    scales = torch.randint(1, 256, (out_features, logical // group), dtype=torch.uint8)
    local = padded // TP

    for rank in range(TP):
        tp_rank(rank)
        weight = PackedvLLMParameter(
            data=torch.full((out_features, local // 2), 0xFF, dtype=torch.uint8),
            input_dim=1,
            output_dim=0,
            packed_dim=1,
            packed_factor=2,
            weight_loader=lambda *_: None,
        )
        scale = GroupQuantScaleParameter(
            data=torch.full((out_features, local // group), 0xFF, dtype=torch.uint8),
            input_dim=1,
            output_dim=0,
            weight_loader=lambda *_: None,
        )
        for param in (weight, scale):
            param.allow_tp_padding = True
        weight.load_row_parallel_weight(packed)
        scale.load_row_parallel_weight(scales)

        available = max(0, min(local, logical - rank * local))
        start = rank * local
        torch.testing.assert_close(
            weight.data[:, : available // 2],
            packed[:, start // 2 : (start + available) // 2],
        )
        torch.testing.assert_close(
            scale.data[:, : available // group],
            scales[:, start // group : (start + available) // group],
        )
        # A zero byte is two FP4 zeros; a zero scale also zeroes the group.
        assert torch.count_nonzero(weight.data[:, available // 2 :]) == 0
        assert torch.count_nonzero(scale.data[:, available // group :]) == 0


def test_kda_per_head_loaders_zero_fill_padded_heads(tp_rank) -> None:
    head_dim, logical_heads, padded_heads = 2, 4, 6
    local_heads = padded_heads // TP
    projection = padded_heads * head_dim
    dt_bias = torch.randn(logical_heads * head_dim)
    a_log = torch.randn(logical_heads)
    old_a_log = a_log.view(1, 1, logical_heads, 1)
    conv = torch.randn(logical_heads * head_dim, 1, 4)

    for rank in range(TP):
        tp_rank(rank)
        heads_available = max(0, min(local_heads, logical_heads - rank * local_heads))
        rows_available = heads_available * head_dim

        dt_param = torch.full((local_heads * head_dim,), float("nan"))
        kda_module._tp_padded_weight_loader(0)(dt_param, dt_bias)
        start = rank * local_heads * head_dim
        torch.testing.assert_close(
            dt_param[:rows_available], dt_bias[start : start + rows_available]
        )
        assert torch.count_nonzero(dt_param[rows_available:]) == 0

        for source in (a_log, old_a_log):
            a_param = torch.full((local_heads,), float("nan"))
            kda_module.a_log_weight_loader(0, True)(a_param, source)
            head_start = rank * local_heads
            torch.testing.assert_close(
                a_param[:heads_available],
                a_log[head_start : head_start + heads_available],
            )
            assert torch.count_nonzero(a_param[heads_available:]) == 0

        conv_param = torch.full((3 * local_heads * head_dim, 1, 4), float("nan"))
        loader = kda_module._make_fused_conv1d_weight_loader(
            [projection] * 3, TP, rank, allow_tp_padding=True
        )
        for shard_id in range(3):
            loader(conv_param, conv + shard_id, shard_id)
        for shard_id, shard in enumerate(conv_param.chunk(3)):
            torch.testing.assert_close(
                shard[:rows_available],
                (conv + shard_id)[start : start + rows_available],
            )
            assert torch.count_nonzero(shard[rows_available:]) == 0


def test_kda_input_projection_pads_heads_and_replicates_f_a(tp_rank) -> None:
    hidden, head_dim, logical_heads, padded_heads = 8, 2, 4, 6
    local_heads = padded_heads // TP
    local_proj = local_heads * head_dim
    checkpoint = {
        "q": torch.randn(logical_heads * head_dim, hidden),
        "k": torch.randn(logical_heads * head_dim, hidden),
        "v": torch.randn(logical_heads * head_dim, hidden),
        "b": torch.randn(logical_heads, hidden),
        "f_a": torch.randn(head_dim, hidden),
    }

    for rank in range(TP):
        tp_rank(rank)
        layer = kda_module._KimiGDNMergedColumnParallelLinear(
            hidden,
            [padded_heads * head_dim] * 3 + [padded_heads, head_dim],
            replicated_shard_id=4,
            tp_size=TP,
            bias=False,
        )
        _allow_padding(layer)
        layer.weight.data.fill_(float("nan"))
        for shard_id, name in enumerate(checkpoint):
            layer.weight_loader(layer.weight, checkpoint[name], shard_id)
        q, k, v, b, f_a = layer.weight.data.split(
            [local_proj] * 3 + [local_heads, head_dim]
        )
        heads_available = max(0, min(local_heads, logical_heads - rank * local_heads))
        rows = heads_available * head_dim
        for local, name in ((q, "q"), (k, "k"), (v, "v")):
            torch.testing.assert_close(
                local[:rows],
                checkpoint[name][rank * local_proj : rank * local_proj + rows],
            )
            assert torch.count_nonzero(local[rows:]) == 0
        torch.testing.assert_close(
            b[:heads_available],
            checkpoint["b"][rank * local_heads : rank * local_heads + heads_available],
        )
        assert torch.count_nonzero(b[heads_available:]) == 0
        torch.testing.assert_close(f_a, checkpoint["f_a"])
