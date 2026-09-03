# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor import parameter
from vllm.model_executor.layers import linear
from vllm.models.glm5next.nvidia import multimodal as glm5next_multimodal


@pytest.fixture
def tp3_linear_state(monkeypatch):
    monkeypatch.setattr(
        glm5next_multimodal, "get_tensor_model_parallel_world_size", lambda: 3
    )
    monkeypatch.setattr(
        glm5next_multimodal.parallel_state,
        "get_tensor_model_parallel_rank",
        lambda: 2,
    )
    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: 3)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: 2)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 3)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 2)


def test_glm5next_vision_tp3_attention_shards_and_zeros_local_tail(
    monkeypatch, tp3_linear_state, default_vllm_config
) -> None:
    class FakeEncoderAttention(torch.nn.Module):
        def __init__(self, **kwargs) -> None:
            super().__init__()
            self.kwargs = kwargs

    monkeypatch.setattr(glm5next_multimodal, "is_vit_use_data_parallel", lambda: False)
    monkeypatch.setattr(glm5next_multimodal, "MMEncoderAttention", FakeEncoderAttention)

    attention = glm5next_multimodal.Glm5NextVisionAttention(
        embed_dim=8,
        num_heads=18,
        projection_size=1152,
        loaded_num_heads=16,
        loaded_projection_size=1024,
    )

    assert attention.head_dim == 64
    assert attention.num_attention_heads_per_partition == 6
    assert attention.q_norm.weight.shape == (64,)
    assert attention.qkv.weight.shape == (1152, 8)
    assert attention.proj.weight.shape == (8, 384)

    q = torch.arange(1024 * 8, dtype=torch.float32).view(1024, 8)
    k = q + 10000
    v = q + 20000
    attention.qkv.weight.weight_loader(attention.qkv.weight, q, "q")
    attention.qkv.weight.weight_loader(attention.qkv.weight, k, "k")
    attention.qkv.weight.weight_loader(attention.qkv.weight, v, "v")

    for offset, checkpoint in zip((0, 384, 768), (q, k, v)):
        torch.testing.assert_close(
            attention.qkv.weight[offset : offset + 256], checkpoint[768:1024]
        )
        torch.testing.assert_close(
            attention.qkv.weight[offset + 256 : offset + 384],
            torch.zeros(128, 8),
        )

    proj = torch.arange(8 * 1024, dtype=torch.float32).view(8, 1024)
    attention.proj.weight.weight_loader(attention.proj.weight, proj)
    torch.testing.assert_close(attention.proj.weight[:, :256], proj[:, 768:1024])
    torch.testing.assert_close(attention.proj.weight[:, 256:], torch.zeros(8, 128))


def test_glm5next_vision_tp3_mlp_shards_and_zeros_local_tail(
    monkeypatch, tp3_linear_state, default_vllm_config
) -> None:
    monkeypatch.setattr(glm5next_multimodal, "is_vit_use_data_parallel", lambda: False)
    mlp = glm5next_multimodal.Glm5NextVisionMLP(
        in_features=8,
        hidden_features=4098,
        loaded_hidden_features=4096,
        swiglu_limit=10.0,
    )

    assert mlp.gate_up_proj.weight.shape == (2732, 8)
    assert mlp.down_proj.weight.shape == (8, 1366)

    gate_up = torch.arange(8192 * 8, dtype=torch.float32).view(8192, 8)
    mlp.gate_up_proj.weight.weight_loader(mlp.gate_up_proj.weight, gate_up)
    for local_offset, checkpoint_offset in ((0, 0), (1366, 4096)):
        torch.testing.assert_close(
            mlp.gate_up_proj.weight[local_offset : local_offset + 1364],
            gate_up[checkpoint_offset + 2732 : checkpoint_offset + 4096],
        )
        torch.testing.assert_close(
            mlp.gate_up_proj.weight[local_offset + 1364 : local_offset + 1366],
            torch.zeros(2, 8),
        )

    down = torch.arange(8 * 4096, dtype=torch.float32).view(8, 4096)
    mlp.down_proj.weight.weight_loader(mlp.down_proj.weight, down)
    torch.testing.assert_close(mlp.down_proj.weight[:, :1364], down[:, 2732:4096])
    torch.testing.assert_close(mlp.down_proj.weight[:, 1364:], torch.zeros(8, 2))


def test_glm5next_vision_tp3_merger_shards_only_divisible_weights(
    monkeypatch, tp3_linear_state, default_vllm_config
) -> None:
    class FakeProjection(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            self.disable_tp = kwargs["disable_tp"]

    monkeypatch.setattr(glm5next_multimodal, "is_vit_use_data_parallel", lambda: False)
    monkeypatch.setattr(glm5next_multimodal, "ColumnParallelLinear", FakeProjection)

    merger = glm5next_multimodal.Glm5NextPatchMerger(
        d_model=4,
        context_dim=10242,
        loaded_context_dim=10240,
        swiglu_limit=10.0,
    )

    assert merger.proj.disable_tp
    assert merger.gate_up_proj.weight.shape == (6828, 4)
    assert merger.down_proj.weight.shape == (4, 3414)

    gate = torch.arange(10240 * 4, dtype=torch.float32).view(10240, 4)
    merger.gate_up_proj.weight.weight_loader(
        merger.gate_up_proj.weight, gate, loaded_shard_id=0
    )
    torch.testing.assert_close(merger.gate_up_proj.weight[:3412], gate[6828:10240])
    torch.testing.assert_close(merger.gate_up_proj.weight[3412:3414], torch.zeros(2, 4))

    up = gate + 100000
    merger.gate_up_proj.weight.weight_loader(
        merger.gate_up_proj.weight, up, loaded_shard_id=1
    )
    torch.testing.assert_close(merger.gate_up_proj.weight[3414:6826], up[6828:10240])
    torch.testing.assert_close(merger.gate_up_proj.weight[6826:6828], torch.zeros(2, 4))

    down = torch.arange(4 * 10240, dtype=torch.float32).view(4, 10240)
    merger.down_proj.weight.weight_loader(merger.down_proj.weight, down)
    torch.testing.assert_close(merger.down_proj.weight[:, :3412], down[:, 6828:10240])
    torch.testing.assert_close(merger.down_proj.weight[:, 3412:], torch.zeros(4, 2))


def _record_vision_geometry(monkeypatch, vision_config, *, data_parallel: bool, tp: int):
    recorded = SimpleNamespace(block=None, merger=None, rope=None)

    class FakeModule(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            self.proj = SimpleNamespace(weight=torch.empty(0))

    class FakeBlock(torch.nn.Module):
        def __init__(self, **kwargs) -> None:
            super().__init__()
            recorded.block = kwargs

    class FakeMerger(torch.nn.Module):
        def __init__(self, **kwargs) -> None:
            super().__init__()
            recorded.merger = kwargs

    def fake_rope(**kwargs):
        recorded.rope = kwargs
        return object()

    monkeypatch.setattr(
        glm5next_multimodal, "is_vit_use_data_parallel", lambda: data_parallel
    )
    monkeypatch.setattr(
        glm5next_multimodal, "get_tensor_model_parallel_world_size", lambda: tp
    )
    monkeypatch.setattr(glm5next_multimodal, "Glm5NextVisionPatchEmbed", FakeModule)
    monkeypatch.setattr(glm5next_multimodal, "Glm5NextVisionBlock", FakeBlock)
    monkeypatch.setattr(glm5next_multimodal, "Glm5NextPatchMerger", FakeMerger)
    monkeypatch.setattr(glm5next_multimodal, "Conv2dLayer", FakeModule)
    monkeypatch.setattr(glm5next_multimodal, "RMSNorm", FakeModule)
    monkeypatch.setattr(glm5next_multimodal, "get_rope", fake_rope)
    monkeypatch.setattr(
        glm5next_multimodal, "get_vit_attn_backend", lambda **kwargs: object()
    )

    transformer = glm5next_multimodal.Glm5NextVisionTransformer(
        SimpleNamespace(swiglu_limit=10.0), vision_config
    )
    return transformer, recorded


def test_glm5next_vision_tp3_consumes_direct_physical_geometry(monkeypatch) -> None:
    vision_config = SimpleNamespace(
        patch_size=14,
        temporal_patch_size=2,
        in_channels=3,
        depth=1,
        hidden_size=1024,
        num_heads=18,
        original_num_heads=16,
        intermediate_size=4098,
        original_intermediate_size=4096,
        spatial_merge_size=2,
        out_hidden_size=4096,
        projection_intermediate_size=10242,
        original_projection_intermediate_size=10240,
        glm53_tp3_attention_projection_size=1152,
        glm53_tp3_padding=True,
        rms_norm_eps=1e-6,
        swiglu_limit=10.0,
    )
    before = vars(vision_config).copy()

    transformer, recorded = _record_vision_geometry(
        monkeypatch, vision_config, data_parallel=False, tp=3
    )

    assert vars(vision_config) == before
    assert transformer.tp_size == 3
    assert transformer.num_heads == 18
    assert transformer.attention_projection_size == 1152
    assert recorded.rope["head_size"] == 64
    assert recorded.block["num_heads"] == 18
    assert recorded.block["loaded_num_heads"] == 16
    assert recorded.block["projection_size"] == 1152
    assert recorded.block["loaded_projection_size"] == 1024
    assert recorded.block["mlp_hidden_dim"] == 4098
    assert recorded.block["loaded_mlp_hidden_dim"] == 4096
    assert recorded.merger["context_dim"] == 10242
    assert recorded.merger["loaded_context_dim"] == 10240


@pytest.mark.parametrize(
    ("data_parallel", "tp", "expected_tp"),
    [(True, 3, 1), (False, 4, 4)],
)
def test_glm5next_vision_unpadded_modes_are_exact_geometry_noops(
    monkeypatch, data_parallel: bool, tp: int, expected_tp: int
) -> None:
    vision_config = SimpleNamespace(
        patch_size=14,
        temporal_patch_size=2,
        in_channels=3,
        depth=1,
        hidden_size=1024,
        num_heads=16,
        intermediate_size=4096,
        spatial_merge_size=2,
        out_hidden_size=4096,
        projection_intermediate_size=10240,
        rms_norm_eps=1e-6,
        swiglu_limit=10.0,
    )
    before = vars(vision_config).copy()

    transformer, recorded = _record_vision_geometry(
        monkeypatch, vision_config, data_parallel=data_parallel, tp=tp
    )

    assert vars(vision_config) == before
    assert transformer.tp_size == expected_tp
    assert transformer.num_heads == 16
    assert transformer.attention_projection_size == 1024
    assert recorded.rope["head_size"] == 64
    assert recorded.block["num_heads"] == 16
    assert recorded.block["loaded_num_heads"] is None
    assert recorded.block["projection_size"] == 1024
    assert recorded.block["loaded_projection_size"] is None
    assert recorded.block["mlp_hidden_dim"] == 4096
    assert recorded.block["loaded_mlp_hidden_dim"] is None
    assert recorded.merger["context_dim"] == 10240
    assert recorded.merger["loaded_context_dim"] is None
