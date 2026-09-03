# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor import parameter
from vllm.model_executor.layers import linear
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import modelopt
from vllm.model_executor.parameter import (
    BlockQuantScaleParameter,
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PackedvLLMParameter,
    PerTensorScaleParameter,
)
from vllm.models.glm5next.nvidia import attention as glm_attention
from vllm.models.glm5next.nvidia import model as glm_model
from vllm.models.glm5next.nvidia import mtp as glm_mtp
from vllm.models.glm5next.nvidia.mtp import Glm5NextMultiTokenPredictorLayer


def _set_tp3_rank2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: 3)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: 2)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 3)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 2)


def test_explicit_loaded_sizes_zero_rank_local_destination_tails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_tp3_rank2(monkeypatch)

    column = ColumnParallelLinear(1, 6, bias=False, loaded_output_size=5)
    column.weight.weight_loader(
        column.weight, torch.arange(1, 6, dtype=column.weight.dtype).unsqueeze(1)
    )
    torch.testing.assert_close(column.weight[:, 0], column.weight.new_tensor([5, 0]))

    merged = MergedColumnParallelLinear(
        1, [6, 6], bias=False, loaded_output_sizes=[5, 5]
    )
    merged.weight.weight_loader(
        merged.weight,
        torch.arange(1, 6, dtype=merged.weight.dtype).unsqueeze(1),
        0,
    )
    torch.testing.assert_close(merged.weight[:2, 0], merged.weight.new_tensor([5, 0]))

    qkv = QKVParallelLinear(
        hidden_size=1,
        head_size=1,
        total_num_heads=6,
        total_num_kv_heads=3,
        loaded_total_num_heads=4,
        loaded_total_num_kv_heads=2,
        bias=False,
    )
    for shard_id, size in (("q", 4), ("k", 2), ("v", 2)):
        qkv.weight.weight_loader(
            qkv.weight,
            torch.ones((size, 1), dtype=qkv.weight.dtype),
            shard_id,
        )
    torch.testing.assert_close(qkv.weight[:, 0], qkv.weight.new_zeros(4))

    row = RowParallelLinear(6, 1, bias=False, loaded_input_size=5)
    row.weight.weight_loader(
        row.weight, torch.arange(1, 6, dtype=row.weight.dtype).unsqueeze(0)
    )
    torch.testing.assert_close(row.weight[0], row.weight.new_tensor([5, 0]))


def test_loaded_sizes_reject_invalid_physical_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_tp3_rank2(monkeypatch)
    with pytest.raises(ValueError, match="exceeds physical size"):
        ColumnParallelLinear(1, 6, bias=False, loaded_output_size=7)
    with pytest.raises(ValueError, match="same length"):
        MergedColumnParallelLinear(1, [6, 6], bias=False, loaded_output_sizes=[5])
    with pytest.raises(ValueError, match="exceeds physical size"):
        RowParallelLinear(6, 1, bias=False, loaded_input_size=7)


def test_padded_loader_rejects_truncated_checkpoint_before_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_tp3_rank2(monkeypatch)
    column = ColumnParallelLinear(1, 6, bias=False, loaded_output_size=5)
    column.weight.data.fill_(17)

    with pytest.raises(ValueError, match="expected 5, got 4"):
        column.weight.weight_loader(
            column.weight,
            torch.ones((4, 1), dtype=column.weight.dtype),
        )

    torch.testing.assert_close(column.weight, column.weight.new_full((2, 1), 17))

    merged = MergedColumnParallelLinear(
        1, [6, 6], bias=False, loaded_output_sizes=[5, 5]
    )
    merged.weight.data.fill_(17)
    with pytest.raises(ValueError, match="expected 5, got 4"):
        merged.weight.weight_loader(
            merged.weight,
            torch.ones((4, 1), dtype=merged.weight.dtype),
            0,
        )
    torch.testing.assert_close(merged.weight, merged.weight.new_full((4, 1), 17))

    qkv = QKVParallelLinear(
        hidden_size=1,
        head_size=1,
        total_num_heads=6,
        total_num_kv_heads=3,
        loaded_total_num_heads=4,
        loaded_total_num_kv_heads=2,
        bias=False,
    )
    qkv.weight.data.fill_(17)
    with pytest.raises(ValueError, match="expected 4, got 3"):
        qkv.weight.weight_loader(
            qkv.weight,
            torch.ones((3, 1), dtype=qkv.weight.dtype),
            "q",
        )
    torch.testing.assert_close(qkv.weight, qkv.weight.new_full((4, 1), 17))

    row = RowParallelLinear(6, 1, bias=False, loaded_input_size=5)
    row.weight.data.fill_(17)
    with pytest.raises(ValueError, match="expected 5, got 4"):
        row.weight.weight_loader(
            row.weight,
            torch.ones((1, 4), dtype=row.weight.dtype),
        )
    torch.testing.assert_close(row.weight, row.weight.new_full((1, 2), 17))


@pytest.mark.parametrize(
    ("parameter_type", "error"),
    [
        (PackedvLLMParameter, "packed_factor=4"),
        (BlockQuantScaleParameter, "quantization block size 4"),
    ],
)
def test_padded_loader_rejects_unaligned_quantized_boundaries_before_write(
    monkeypatch: pytest.MonkeyPatch,
    parameter_type: type[parameter.BasevLLMParameter],
    error: str,
) -> None:
    _set_tp3_rank2(monkeypatch)
    column = ColumnParallelLinear(1, 6, bias=False, loaded_output_size=4)
    column.weight_block_size = (4, 4)
    kwargs = {
        "data": torch.full((1, 1), 23.0),
        "input_dim": 1,
        "output_dim": 0,
        "weight_loader": lambda *_args, **_kwargs: None,
    }
    if parameter_type is PackedvLLMParameter:
        kwargs.update(packed_factor=4, packed_dim=0)
    quantized_param = parameter_type(**kwargs)

    with pytest.raises(ValueError, match=error):
        column.weight_loader(
            quantized_param,
            torch.ones((1, 1), dtype=quantized_param.dtype),
        )

    torch.testing.assert_close(quantized_param, quantized_param.new_full((1, 1), 23))


def test_padded_v2_loader_preserves_unsharded_per_tensor_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_tp3_rank2(monkeypatch)
    row = RowParallelLinear(6, 1, bias=False, loaded_input_size=4)
    scale = PerTensorScaleParameter(
        data=torch.zeros(1, dtype=torch.float32),
        weight_loader=lambda *_args, **_kwargs: None,
    )

    row.weight_loader_v2(scale, torch.tensor([3.0]))

    torch.testing.assert_close(scale, torch.tensor([3.0]))


@pytest.mark.parametrize(
    ("parameter_type", "storage_factor", "physical_size", "loaded_size", "dtype"),
    [
        (ModelWeightParameter, 2, 6, 4, torch.uint8),
        (GroupQuantScaleParameter, 16, 96, 64, torch.float8_e4m3fn),
    ],
)
def test_padded_nvfp4_row_loader_converts_logical_to_storage_width(
    monkeypatch: pytest.MonkeyPatch,
    parameter_type: type[parameter.BasevLLMParameter],
    storage_factor: int,
    physical_size: int,
    loaded_size: int,
    dtype: torch.dtype,
) -> None:
    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 3)
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 1)
    row = RowParallelLinear(
        physical_size,
        1,
        bias=False,
        loaded_input_size=loaded_size,
    )
    local_storage_size = physical_size // 3 // storage_factor
    loaded_storage_size = loaded_size // storage_factor
    quantized_param = parameter_type(
        data=torch.zeros((1, local_storage_size), dtype=dtype),
        input_dim=1,
        input_dim_storage_factor=storage_factor,
        output_dim=0,
        weight_loader=lambda *_args, **_kwargs: None,
    )
    loaded = torch.arange(
        1,
        loaded_storage_size + 1,
        dtype=torch.uint8,
    ).to(dtype)

    row.weight_loader_v2(quantized_param, loaded.unsqueeze(0))

    torch.testing.assert_close(
        quantized_param,
        loaded[local_storage_size : 2 * local_storage_size].unsqueeze(0),
    )


@pytest.mark.parametrize(
    "method_type",
    (
        modelopt.ModelOptNvFp4LinearMethod,
        modelopt.ModelOptNvFp4W4A16LinearMethod,
    ),
)
def test_modelopt_nvfp4_row_parameters_declare_storage_widths(
    monkeypatch: pytest.MonkeyPatch,
    method_type: type,
) -> None:
    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 3)
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(
        modelopt,
        "init_nvfp4_linear_kernel",
        lambda **_kwargs: SimpleNamespace(input_quant_key=lambda: None),
    )
    config = modelopt.ModelOptNvFp4Config(
        quant_method=(
            "NVFP4"
            if method_type is modelopt.ModelOptNvFp4LinearMethod
            else "W4A16_NVFP4"
        ),
        is_checkpoint_nvfp4_serialized=True,
        group_size=16,
    )
    holder = torch.nn.Module()
    method_type(config).create_weights(
        holder,
        input_size_per_partition=32,
        output_partition_sizes=[1],
        input_size=96,
        output_size=1,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *_args, **_kwargs: None,
    )

    assert holder.weight.input_dim_storage_factor == 2
    assert holder.weight_scale.input_dim_storage_factor == 16

    row = RowParallelLinear(96, 1, bias=False, loaded_input_size=64)
    for param, storage_factor in (
        (holder.weight, 2),
        (holder.weight_scale, 16),
    ):
        loaded = torch.arange(
            1,
            64 // storage_factor + 1,
            dtype=torch.uint8,
        ).to(param.dtype)
        row.weight_loader_v2(param, loaded.unsqueeze(0))
        local_width = 32 // storage_factor
        torch.testing.assert_close(
            param,
            loaded[local_width : 2 * local_width].unsqueeze(0),
        )


def test_modelopt_mxfp8_scale_loads_padded_tp3_storage_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 3)
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(modelopt, "init_mxfp8_linear_kernel", lambda: object())
    config = modelopt.ModelOptMxFp8Config(
        is_checkpoint_mxfp8_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
    )
    holder = torch.nn.Module()
    modelopt.ModelOptMxFp8LinearMethod(config).create_weights(
        holder,
        input_size_per_partition=4096,
        output_partition_sizes=[1],
        input_size=12288,
        output_size=1,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *_args, **_kwargs: None,
    )

    assert holder.weight_scale.input_dim_storage_factor == 32
    loaded = torch.arange(1, 129, dtype=holder.weight_scale.dtype).unsqueeze(0)
    row = RowParallelLinear(12288, 1, bias=False, loaded_input_size=4096)
    row.weight_loader_v2(holder.weight_scale, loaded)
    torch.testing.assert_close(
        holder.weight_scale, torch.zeros_like(holder.weight_scale)
    )


@pytest.mark.parametrize("tp3", [False, True])
def test_mla_projection_loaded_sizes_are_tp3_only(
    monkeypatch: pytest.MonkeyPatch,
    tp3: bool,
) -> None:
    captured: dict[str, dict] = {}

    class FakeLinear(torch.nn.Module):
        def __init__(self, *args, prefix: str, **kwargs) -> None:
            super().__init__()
            captured[prefix] = kwargs

    class FakeModule(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

    monkeypatch.setattr(glm_attention, "ColumnParallelLinear", FakeLinear)
    monkeypatch.setattr(glm_attention, "RowParallelLinear", FakeLinear)
    monkeypatch.setattr(glm_attention, "DeepSeekV2FusedQkvAProjLinear", FakeLinear)
    monkeypatch.setattr(glm_attention, "RMSNorm", FakeModule)
    monkeypatch.setattr(glm_attention, "MultiHeadLatentAttentionWrapper", FakeModule)
    monkeypatch.setattr(
        glm_attention,
        "get_tensor_model_parallel_world_size",
        lambda: 3 if tp3 else 4,
    )

    config = SimpleNamespace(
        rms_norm_eps=1e-5,
        rope_parameters=None,
        index_topk=None,
        glm53_tp3_padding=tp3,
    )
    if tp3:
        config.original_num_attention_heads = 64

    glm_attention.Glm5NextMLAAttention(
        vllm_config=SimpleNamespace(),
        config=config,
        hidden_size=4096,
        num_heads=72 if tp3 else 64,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        q_lora_rank=1536,
        kv_lora_rank=512,
        skip_rope=True,
        prefix="attn",
    )

    loaded_key = "loaded_output_size"
    if tp3:
        assert captured["attn.q_b_proj"][loaded_key] == 64 * 192
        assert captured["attn.kv_b_proj"][loaded_key] == 64 * 256
        assert captured["attn.o_proj"]["loaded_input_size"] == 64 * 128
    else:
        assert loaded_key not in captured["attn.q_b_proj"]
        assert loaded_key not in captured["attn.kv_b_proj"]
        assert "loaded_input_size" not in captured["attn.o_proj"]


def test_shared_expert_uses_physical_tp3_width_and_logical_load_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {}

    class FakeMerged(torch.nn.Module):
        def __init__(self, input_size, output_sizes, **kwargs) -> None:
            super().__init__()
            calls["gate_up"] = (input_size, output_sizes, kwargs)

    class FakeRow(torch.nn.Module):
        def __init__(self, input_size, output_size, **kwargs) -> None:
            super().__init__()
            calls["down"] = (input_size, output_size, kwargs)

    class FakeActivation(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

    monkeypatch.setattr(glm_model, "MergedColumnParallelLinear", FakeMerged)
    monkeypatch.setattr(glm_model, "RowParallelLinear", FakeRow)
    monkeypatch.setattr(glm_model, "SiluAndMul", FakeActivation)

    glm_model.Glm5NextMLP(
        hidden_size=4096,
        intermediate_size=2112,
        loaded_intermediate_size=2048,
        hidden_act="silu",
    )

    assert calls["gate_up"][1] == [2112, 2112]
    assert calls["gate_up"][2]["loaded_output_sizes"] == [2048, 2048]
    assert calls["down"][0] == 2112
    assert calls["down"][2]["loaded_input_size"] == 2048


@pytest.mark.parametrize("tp3", [False, True])
def test_mtp_tp3_objects_are_created_only_for_active_geometry(
    monkeypatch: pytest.MonkeyPatch,
    tp3: bool,
) -> None:
    calls = {}

    class FakeNorm(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

    class FakeColumn(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            calls["projection"] = (args, kwargs)

    class FakeSharedHead(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            calls["standard_shared_head"] = True

    class FakeParallelLMHead(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            calls["padded_head"] = (args, kwargs)

    class FakeDecoder(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

    monkeypatch.setattr(glm_mtp, "RMSNorm", FakeNorm)
    monkeypatch.setattr(glm_mtp, "ColumnParallelLinear", FakeColumn)
    monkeypatch.setattr(glm_mtp, "SharedHead", FakeSharedHead)
    monkeypatch.setattr(glm_mtp, "ParallelLMHead", FakeParallelLMHead)
    monkeypatch.setattr(glm_mtp, "Glm5NextDecoderLayer", FakeDecoder)
    monkeypatch.setattr(glm_mtp, "current_platform", SimpleNamespace(device_type="cpu"))

    config = SimpleNamespace(
        hidden_size=4096,
        rms_norm_eps=1e-5,
        index_topk=4,
        index_kpool=1,
        vocab_size=154880,
        glm53_tp3_padding=tp3,
    )
    if tp3:
        config.glm53_tp3_mtp_projection_size = 4098
        config.glm53_tp3_vocab_padding_size = 192
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=config)
        ),
        quant_config=None,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
    )

    layer = glm_mtp.Glm5NextMultiTokenPredictorLayer(vllm_config, "model.layers.45")

    if tp3:
        assert isinstance(layer.eh_proj, FakeColumn)
        assert calls["projection"][0] == (8192, 4098)
        assert calls["projection"][1]["loaded_output_size"] == 4096
        assert calls["projection"][1]["gather_output"]
        assert calls["padded_head"][1]["padding_size"] == 192
        assert "standard_shared_head" not in calls
    else:
        assert type(layer.eh_proj) is torch.nn.Linear
        assert layer.eh_proj.in_features == 8192
        assert layer.eh_proj.out_features == 4096
        assert calls["standard_shared_head"]
        assert "projection" not in calls
        assert "padded_head" not in calls


@pytest.mark.parametrize("tp3", [False, True])
def test_target_vocab_storage_padding_is_tp3_only(
    monkeypatch: pytest.MonkeyPatch,
    tp3: bool,
) -> None:
    calls = {}

    class FakeModel(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

    class FakeHead(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            calls["head"] = (args, kwargs)

    class FakeLogits:
        def __init__(self, vocab_size, **kwargs) -> None:
            calls["logits_vocab_size"] = vocab_size

    monkeypatch.setattr(glm_model, "Glm5NextModel", FakeModel)
    monkeypatch.setattr(glm_model, "ParallelLMHead", FakeHead)
    monkeypatch.setattr(glm_model, "LogitsProcessor", FakeLogits)
    monkeypatch.setattr(
        glm_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_last_rank=True),
    )

    config = SimpleNamespace(vocab_size=154880, hidden_size=4096)
    if tp3:
        config.glm53_tp3_padding = True
        config.glm53_tp3_vocab_padding_size = 192
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config),
        quant_config=None,
    )

    glm_model.Glm5NextForCausalLM(vllm_config=vllm_config)

    assert calls["head"][0] == (154880, 4096)
    assert calls["logits_vocab_size"] == 154880
    if tp3:
        assert calls["head"][1]["padding_size"] == 192
    else:
        assert "padding_size" not in calls["head"][1]


def test_mtp_projection_narrows_padded_output_contiguously() -> None:
    class PaddedProjection(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            output = value.new_zeros((*value.shape[:-1], 6))
            output[..., :4] = value[..., :4]
            return output

    class RecordingBlock(torch.nn.Module):
        def forward(self, *, hidden_states: torch.Tensor, **kwargs):
            assert hidden_states.shape == (2, 4)
            assert hidden_states.is_contiguous()
            return hidden_states, None, None, None

    class SharedHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = lambda hidden_states, residual: (hidden_states, None)

    layer = object.__new__(Glm5NextMultiTokenPredictorLayer)
    torch.nn.Module.__init__(layer)
    layer.enorm = torch.nn.Identity()
    layer.hnorm = torch.nn.Identity()
    layer.eh_proj = PaddedProjection()
    layer.mtp_block = RecordingBlock()
    layer.shared_head = SharedHead()

    hidden_states, recycled = layer(
        input_ids=torch.zeros(2, dtype=torch.long),
        positions=torch.arange(2),
        previous_hidden_states=torch.ones(2, 4),
        inputs_embeds=torch.ones(2, 4),
    )

    assert hidden_states.is_contiguous()
    torch.testing.assert_close(recycled, hidden_states)
