# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm.models.deepseek_v4.nvidia.mtp as mtp_module
from vllm.models.deepseek_v4.nvidia.mtp import DeepSeekV4MultiTokenPredictorLayer
from vllm.models.deepseek_v4.nvidia.vl_model import (
    _make_deepseek_v4_vl_weights_mapper,
)


@pytest.mark.parametrize("expert_dtype", ["fp4", "fp8"])
def test_vl_weights_mapper_reroots_text_weights(expert_dtype):
    mapper = _make_deepseek_v4_vl_weights_mapper(expert_dtype, image_enabled=True)
    apply = mapper._map_name

    assert apply("layers.0.ffn.gate.bias") == (
        "language_model.model.layers.0.ffn.gate.e_score_correction_bias"
    )
    assert apply("layers.7.ffn.gate.bias_vl") == (
        "language_model.model.layers.7.ffn.gate.bias_vl"
    )
    assert apply("head.weight") == "language_model.lm_head.weight"
    assert apply("embed.weight") == "language_model.model.embed_tokens.weight"
    assert apply("norm.weight") == "language_model.model.norm.weight"
    assert apply("hc_head_fn") == "language_model.model.hc_head_fn"
    assert apply("layers.0.ffn.shared_experts.w2.weight") == (
        "language_model.model.layers.0.ffn.shared_experts.down_proj.weight"
    )

    expert_scale = apply("layers.0.ffn.experts.3.w1.scale")
    if expert_dtype == "fp4":
        assert expert_scale == (
            "language_model.model.layers.0.ffn.experts.3.w1.weight_scale"
        )
    else:
        assert expert_scale == (
            "language_model.model.layers.0.ffn.experts.3.w1.weight_scale_inv"
        )


def test_vl_weights_mapper_vision_weights_passthrough():
    mapper = _make_deepseek_v4_vl_weights_mapper("fp4", image_enabled=True)
    apply = mapper._map_name

    assert apply("vision.blocks.0.attn.wqkv.weight") == (
        "vision.blocks.0.attn.wqkv.weight"
    )
    assert apply("vision.patch_embed.proj.bias") == "vision.patch_embed.proj.bias"
    assert apply("vision.norm.weight") == "vision.norm.weight"
    assert apply("aligner.w1.weight") == "aligner.w1.weight"
    assert apply("image_start") == "image_start"
    assert apply("image_pad") == "image_pad"


def test_vl_weights_mapper_drops_mtp_weights():
    mapper = _make_deepseek_v4_vl_weights_mapper("fp4", image_enabled=True)

    assert mapper._map_name("mtp.0.ffn.gate.bias_vl") is None
    assert mapper._map_name("mtp.0.hc_attn_base") is None


def test_vl_weights_mapper_drops_tower_when_image_disabled():
    mapper = _make_deepseek_v4_vl_weights_mapper("fp4", image_enabled=False)

    assert mapper._map_name("vision.blocks.0.attn.wqkv.weight") is None
    assert mapper._map_name("aligner.w1.weight") is None
    assert mapper._map_name("image_start") is None
    # bias_vl still loads: the MoE gate keeps it whenever vision_n_layers > 0
    assert mapper._map_name("layers.7.ffn.gate.bias_vl") is not None


class _MTPBlockCapture(nn.Module):
    use_sequence_parallel = True

    def __init__(self) -> None:
        super().__init__()
        self.input_ids: torch.Tensor | None = None

    def forward(self, *, positions, x, input_ids):
        self.input_ids = input_ids
        return x, x, x, x

    def _should_run_b12x_mhc(self, _: int) -> bool:
        return False


class _NormStub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.variance_epsilon = 1e-6


def test_mtp_sequence_parallel_shards_input_ids(monkeypatch):
    """MTP MoE routing ids must stay aligned with sharded hidden-state rows."""
    layer = DeepSeekV4MultiTokenPredictorLayer.__new__(
        DeepSeekV4MultiTokenPredictorLayer
    )
    nn.Module.__init__(layer)
    layer.hc_mult = 1
    layer.config = SimpleNamespace(hidden_size=2)
    layer.enorm = _NormStub()
    layer.hnorm = _NormStub()
    layer.h_proj = nn.Identity()
    layer.e_proj = nn.Identity()
    layer.mtp_block = _MTPBlockCapture()

    sharded_inputs: list[torch.Tensor] = []

    def shard(tensor: torch.Tensor) -> torch.Tensor:
        sharded_inputs.append(tensor)
        return tensor[::2]

    def identity_rmsnorm(inputs_embeds, _, previous_hidden_states, *__):
        return inputs_embeds, previous_hidden_states

    monkeypatch.setattr(mtp_module, "fused_mtp_input_rmsnorm", identity_rmsnorm)
    monkeypatch.setattr(mtp_module, "sp_shard", shard)
    monkeypatch.setattr(mtp_module, "sp_all_gather", lambda tensor: tensor)
    monkeypatch.setattr(
        mtp_module,
        "mhc_post_tilelang",
        lambda hidden_states, *_: hidden_states,
    )

    input_ids = torch.tensor([7, 11, 13, 17])
    layer(
        input_ids=input_ids,
        positions=torch.arange(4),
        previous_hidden_states=torch.arange(8, dtype=torch.float32).view(4, 2),
        inputs_embeds=torch.ones(4, 2),
    )

    assert len(sharded_inputs) == 3
    assert layer.mtp_block.input_ids is not None
    torch.testing.assert_close(layer.mtp_block.input_ids, torch.tensor([7, 13]))
