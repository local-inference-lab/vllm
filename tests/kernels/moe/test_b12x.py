# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the b12x tensor-parallel MoE integration."""

import weakref
from dataclasses import dataclass, replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import vllm.model_executor.layers.fused_moe.b12x as b12x
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
import vllm.model_executor.layers.fused_moe.oracle.mxfp4 as mxfp4_oracle
import vllm.model_executor.layers.fused_moe.oracle.nvfp4 as nvfp4_oracle
from tests.kernels.moe.utils import make_dummy_moe_config
from tests.kernels.quantization.nvfp4_utils import (
    dequantize_nvfp4_to_dtype,
    quant_nvfp4_tensor,
)
from tests.kernels.utils import torch_moe
from tests.quantization.reference_mxfp4 import dq_mxfp4_torch
from vllm import _custom_ops as ops
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.b12x import B12xExperts
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    mxfp4_w4a16_moe_quant_config,
    nvfp4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    select_deepseek_v4_mxfp4_moe_backend,
    select_mxfp4_moe_backend,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    select_nvfp4_moe_backend,
)
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config, Mxfp4MoEMethod
from vllm.model_executor.layers.quantization.utils.b12x_moe import (
    prepare_nvfp4_moe_layer_for_b12x,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import mxfp4_quantize
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp4Static,
    kMxfp8Dynamic,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed


def _quantize_nvfp4_linear(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weights_q = []
    scales = []
    global_scales = []
    for expert_weight in weight:
        weight_q, scale, global_scale = quant_nvfp4_tensor(
            expert_weight,
            is_sf_swizzled_layout=False,
        )
        weights_q.append(weight_q)
        scales.append(scale)
        global_scales.append(global_scale)
    return torch.stack(weights_q), torch.stack(scales), torch.stack(global_scales)


def _dequantize_nvfp4_linear(
    tensor_fp4: torch.Tensor,
    tensor_sf: torch.Tensor,
    global_scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    return dequantize_nvfp4_to_dtype(
        tensor_fp4,
        tensor_sf,
        global_scale,
        dtype=dtype,
        device=tensor_fp4.device,
        is_sf_linear_layout=True,
    )


def _nvfp4_activation_reference(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    a1_scale: torch.Tensor,
    a2_scale: torch.Tensor,
) -> torch.Tensor:
    tokens, hidden_size = hidden_states.shape
    topk = topk_ids.shape[1]
    routed_input = (
        hidden_states[:, None, :]
        .expand(-1, topk, -1)
        .reshape(tokens * topk, hidden_size)
    )
    routed_output = torch.zeros(
        tokens * topk,
        hidden_size,
        dtype=torch.float32,
        device=hidden_states.device,
    )
    flat_ids = topk_ids.reshape(-1)

    for expert in range(w1.shape[0]):
        mask = flat_ids == expert
        if not mask.any():
            continue
        a1_q, a1_block_scale = ops.scaled_fp4_quant(
            routed_input[mask],
            a1_scale[expert],
            is_sf_swizzled_layout=False,
        )
        a1 = _dequantize_nvfp4_linear(
            a1_q,
            a1_block_scale,
            a1_scale[expert],
            torch.float32,
        )
        fc1 = a1 @ w1[expert].float().t()
        gate, up = fc1.chunk(2, dim=-1)
        intermediate = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
        a2_q, a2_block_scale = ops.scaled_fp4_quant(
            intermediate,
            a2_scale[expert],
            is_sf_swizzled_layout=False,
        )
        a2 = _dequantize_nvfp4_linear(
            a2_q,
            a2_block_scale,
            a2_scale[expert],
            torch.float32,
        )
        routed_output[mask] = a2 @ w2[expert].float().t()

    return (
        routed_output.view(tokens, topk, hidden_size)
        .mul(topk_weights[..., None])
        .sum(dim=1)
        .to(hidden_states.dtype)
    )


def _has_b12x_moe() -> bool:
    return (
        torch.cuda.is_available()
        and current_platform.is_device_capability_family(120)
        and B12xExperts._supports_current_device()
    )


def _make_b12x_moe_kernel(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk: int,
    activation: MoEActivation,
    quant_config: FusedMoEQuantConfig,
) -> mk.FusedMoEKernel:
    num_experts = w1.shape[0]
    moe_config = make_dummy_moe_config(
        num_experts=num_experts,
        experts_per_token=topk,
        hidden_dim=hidden_states.shape[1],
        intermediate_size=w2.shape[2] * 2,
        in_dtype=hidden_states.dtype,
        activation=activation,
    )
    experts = B12xExperts(moe_config, quant_config)
    layer = SimpleNamespace(
        activation=activation,
        apply_router_weight_on_input=False,
        w13_weight=w1,
        w2_weight=w2,
        w13_weight_scale=quant_config.w1_scale,
        w2_weight_scale=quant_config.w2_scale,
    )
    if quant_config.weight_quant_dtype == "nvfp4":
        layer.w13_weight_scale_2 = quant_config.g1_alphas
        layer.w2_weight_scale_2 = quant_config.g2_alphas
        if quant_config.quant_dtype is not None:
            assert quant_config.a1_gscale is not None
            assert quant_config.a2_gscale is not None
            layer.w13_input_scale = 1.0 / quant_config.a1_gscale
            layer.w2_input_scale = 1.0 / quant_config.a2_gscale
    experts.process_weights_after_loading(layer)
    return mk.FusedMoEKernel(
        maybe_make_prepare_finalize(
            moe=moe_config,
            quant_config=quant_config,
            allow_new_interface=True,
            use_monolithic=False,
        ),
        experts,
    )


def _run_b12x_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    score: torch.Tensor,
    topk: int,
    activation: MoEActivation,
    quant_config: FusedMoEQuantConfig,
) -> torch.Tensor:
    num_experts = w1.shape[0]
    kernel = _make_b12x_moe_kernel(
        hidden_states,
        w1,
        w2,
        topk,
        activation,
        quant_config,
    )
    topk_weights, topk_ids, _ = fused_topk(
        hidden_states, score, topk, renormalize=False
    )
    return kernel.apply(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=activation,
        global_num_experts=num_experts,
        expert_map=None,
        apply_router_weight_on_input=False,
    )


def _quant_config(
    weight_dtype: str,
    activation_dtype: str | None,
    *,
    numerical_recipe: str = "default",
):
    scale = torch.ones(1, dtype=torch.float32)
    return FusedMoEQuantConfig.make(
        quant_dtype=activation_dtype,
        weight_dtype=weight_dtype,
        w1_scale=scale,
        w2_scale=scale,
        g1_alphas=scale,
        g2_alphas=scale,
        a1_gscale=scale,
        a2_gscale=scale,
        numerical_recipe=numerical_recipe,
    )


def test_b12x_moe_supports_only_tensor_parallel() -> None:
    parallel = FusedMoEParallelConfig.make_no_parallel()

    assert B12xExperts._supports_parallel_config(parallel)
    assert not B12xExperts._supports_parallel_config(
        replace(parallel, use_ep=True, ep_size=2)
    )
    all2all = replace(parallel, use_ep=True, dp_size=2)
    assert all2all.use_all2all_kernels
    assert not B12xExperts._supports_parallel_config(all2all)
    assert not B12xExperts._supports_parallel_config(
        replace(parallel, enable_eplb=True)
    )


_SITU_REASON = "kernel supports only SiTU beta=4 and linear_beta=25"
_UNINTERLEAVED_W4A8_REASON = "kernel does not support swigluoai_uninterleave with W4A8"


@pytest.mark.parametrize(
    "config_kwargs,overrides,weight_key,activation_key,expected_reason",
    [
        pytest.param(
            {"hidden_dim": 128, "activation": MoEActivation.SWIGLUOAI},
            {},
            kMxfp4Static,
            None,
            "kernel does not support MoEActivation.SWIGLUOAI activation",
            id="interleaved-swigluoai",
        ),
        pytest.param(
            {
                "hidden_dim": 128,
                "activation": MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            },
            {},
            kMxfp4Static,
            kMxfp8Dynamic,
            _UNINTERLEAVED_W4A8_REASON,
            id="mxfp4-w4a8-uninterleaved-swigluoai",
        ),
        pytest.param(
            {
                "hidden_dim": 128,
                "activation": MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            },
            {},
            kMxfp4Static,
            None,
            None,
            id="w4a16-uninterleaved-swigluoai",
        ),
        pytest.param(
            {"activation": MoEActivation.RELU2_NO_MUL},
            {},
            kMxfp4Static,
            kMxfp8Dynamic,
            "MXFP4 W4A8 supports only SiLU and SiTU",
            id="mxfp4-w4a8-relu2",
        ),
        pytest.param(
            {"hidden_dim": 128},
            {},
            kMxfp4Static,
            kMxfp8Dynamic,
            (
                "MXFP4 W4A8 requires hidden size divisible by 256 and per-rank "
                "intermediate size divisible by 32"
            ),
            id="mxfp4-w4a8-alignment",
        ),
        pytest.param(
            {"hidden_dim": 128, "intermediate_size": 48},
            {"intermediate_size_per_partition": 64},
            kMxfp4Static,
            None,
            "MXFP4 requires the per-rank intermediate size to be divisible by 32",
            id="mxfp4-tp-scale-groups",
        ),
        pytest.param(
            {"activation": MoEActivation.SITU},
            {"activation_situ_beta": 3.0, "activation_situ_linear_beta": 25.0},
            kMxfp4Static,
            None,
            _SITU_REASON,
            id="situ-beta",
        ),
        pytest.param(
            {"activation": MoEActivation.SITU},
            {"activation_situ_beta": 4.0, "activation_situ_linear_beta": 25.0},
            kMxfp4Static,
            None,
            None,
            id="situ-standard-parameters",
        ),
    ],
)
def test_b12x_moe_config_support(
    monkeypatch: pytest.MonkeyPatch,
    config_kwargs,
    overrides,
    weight_key,
    activation_key,
    expected_reason: str | None,
) -> None:
    monkeypatch.setattr(B12xExperts, "_supports_current_device", lambda: True)
    config = make_dummy_moe_config(
        **{"hidden_dim": 256, "intermediate_size": 64, **config_kwargs}
    )
    for name, value in overrides.items():
        setattr(config, name, value)

    supported, reason = B12xExperts.is_supported_config(
        B12xExperts,
        config,
        weight_key,
        activation_key,
        mk.FusedMoEActivationFormat.Standard,
    )

    assert (supported, reason) == (expected_reason is None, expected_reason)


@pytest.mark.parametrize(
    "activation_key,force_a16,expected_backend",
    [
        (kMxfp8Dynamic, False, Mxfp4MoeBackend.B12X_MXFP4_MXFP8),
        (None, False, Mxfp4MoeBackend.B12X_MXFP4_MXFP8),
    ],
)
def test_explicit_b12x_mxfp4_selection(
    monkeypatch: pytest.MonkeyPatch,
    activation_key,
    force_a16: bool,
    expected_backend: Mxfp4MoeBackend,
) -> None:
    monkeypatch.setattr(B12xExperts, "_supports_current_device", lambda: True)
    monkeypatch.setattr(mxfp4_oracle, "_user_moe_activation_override", lambda: None)
    monkeypatch.setattr(
        mxfp4_oracle.envs,
        "VLLM_B12X_MOE_FP4_FORCE_A16",
        force_a16,
    )
    config = make_dummy_moe_config(hidden_dim=256, intermediate_size=64)
    config.moe_backend = "b12x"

    backend, experts_cls = select_mxfp4_moe_backend(
        config,
        activation_key=activation_key,
    )

    assert backend == expected_backend
    assert experts_cls is B12xExperts


def test_explicit_b12x_mxfp4_force_a16_uses_a16_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(B12xExperts, "_supports_current_device", lambda: True)
    monkeypatch.setattr(mxfp4_oracle, "_user_moe_activation_override", lambda: None)
    monkeypatch.setattr(
        mxfp4_oracle.envs,
        "VLLM_B12X_MOE_FP4_FORCE_A16",
        True,
    )
    config = make_dummy_moe_config(hidden_dim=128, intermediate_size=64)
    config.moe_backend = "b12x"

    backend, experts_cls = select_mxfp4_moe_backend(
        config,
        activation_key=kMxfp8Dynamic,
    )

    assert backend == Mxfp4MoeBackend.B12X_MXFP4_BF16
    assert experts_cls is B12xExperts


@pytest.mark.parametrize(
    "force_a16,expected_backend",
    [
        (False, Mxfp4MoeBackend.B12X_MXFP4_MXFP8),
        (True, Mxfp4MoeBackend.B12X_MXFP4_BF16),
    ],
)
def test_deepseek_v4_b12x_activation_selection(
    monkeypatch: pytest.MonkeyPatch,
    force_a16: bool,
    expected_backend: Mxfp4MoeBackend,
) -> None:
    monkeypatch.setattr(B12xExperts, "_supports_current_device", lambda: True)
    monkeypatch.setattr(
        mxfp4_oracle.envs,
        "VLLM_B12X_MOE_FP4_FORCE_A16",
        force_a16,
    )
    config = make_dummy_moe_config(hidden_dim=256, intermediate_size=64)
    config.moe_backend = "b12x"

    backend, experts_cls = select_deepseek_v4_mxfp4_moe_backend(config)

    assert backend == expected_backend
    assert experts_cls is B12xExperts


def test_compressed_tensors_mxfp4_preserves_checkpoint_packing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
        compressed_tensors_moe_w4a4_mxfp4 as ct_mxfp4,
    )

    monkeypatch.setattr(
        ct_mxfp4.CutlassExpertsMxfp4,
        "_supports_current_device",
        lambda: False,
    )
    monkeypatch.setattr(
        ct_mxfp4,
        "select_mxfp4_moe_backend",
        lambda moe: (Mxfp4MoeBackend.B12X_MXFP4_MXFP8, B12xExperts),
    )
    monkeypatch.setattr(
        ct_mxfp4,
        "prepare_moe_fp4_layer_for_marlin",
        lambda layer: pytest.fail("b12x must not use Marlin packing"),
    )
    moe_config = SimpleNamespace(w13_num_shards=2, moe_backend="b12x")
    method = ct_mxfp4.CompressedTensorsW4A4Mxfp4MoEMethod(moe_config)
    processed_layers: list[torch.nn.Module] = []
    fake_experts = SimpleNamespace(
        process_weights_after_loading=processed_layers.append
    )
    kernel = SimpleNamespace(fused_experts=fake_experts)
    monkeypatch.setattr(method, "get_fused_moe_quant_config", lambda _: object())
    monkeypatch.setattr(ct_mxfp4, "make_mxfp4_moe_kernel", lambda **_: kernel)
    layer = torch.nn.Module()
    layer._expert_routing_tables = lambda: ()
    method.create_weights(
        layer,
        num_experts=2,
        hidden_size=64,
        intermediate_size_per_partition=32,
        params_dtype=torch.bfloat16,
    )
    w13_packed_data = layer.w13_weight_packed.data
    w2_packed_data = layer.w2_weight_packed.data

    method.process_weights_after_loading(layer)

    assert layer.w13_weight.data.data_ptr() == w13_packed_data.data_ptr()
    assert layer.w2_weight.data.data_ptr() == w2_packed_data.data_ptr()
    assert processed_layers == [layer]


def test_b12x_mxfp4_falls_back_to_a16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(B12xExperts, "_supports_current_device", lambda: True)
    monkeypatch.setattr(mxfp4_oracle, "_user_moe_activation_override", lambda: None)
    monkeypatch.setattr(
        mxfp4_oracle.envs,
        "VLLM_B12X_MOE_FP4_FORCE_A16",
        False,
    )
    config = make_dummy_moe_config(hidden_dim=128, intermediate_size=64)
    config.moe_backend = "b12x"

    backend, experts_cls = select_mxfp4_moe_backend(config)

    assert backend == Mxfp4MoeBackend.B12X_MXFP4_BF16
    assert experts_cls is B12xExperts


@pytest.mark.parametrize(
    "activation_key,force_a16,expected_activation_key",
    [
        (kNvfp4Dynamic, False, kNvfp4Dynamic),
        (kMxfp8Dynamic, False, kMxfp8Dynamic),
        (kNvfp4Dynamic, True, None),
    ],
)
def test_explicit_b12x_nvfp4_selection(
    monkeypatch: pytest.MonkeyPatch,
    activation_key,
    force_a16: bool,
    expected_activation_key,
) -> None:
    selected_activation_keys = []

    def is_supported_config(cls, config, weight_key, activation_key, activation_format):
        selected_activation_keys.append(activation_key)
        return True, None

    monkeypatch.setattr(B12xExperts, "is_supported_config", is_supported_config)
    monkeypatch.setattr(
        nvfp4_oracle.envs,
        "VLLM_B12X_MOE_FP4_FORCE_A16",
        force_a16,
    )
    config = make_dummy_moe_config(hidden_dim=128, intermediate_size=64)
    config.moe_backend = "b12x"

    backend, experts_cls = select_nvfp4_moe_backend(
        config,
        weight_key=kNvfp4Static,
        activation_key=activation_key,
    )

    assert backend == NvFp4MoeBackend.B12X
    assert experts_cls is B12xExperts
    assert selected_activation_keys == [expected_activation_key]


@pytest.mark.parametrize(
    "force_a16,expected_quant_dtype", [(False, "nvfp4"), (True, None)]
)
def test_b12x_nvfp4_force_a16_updates_quant_config(
    monkeypatch: pytest.MonkeyPatch,
    force_a16: bool,
    expected_quant_dtype,
) -> None:
    monkeypatch.setattr(
        nvfp4_oracle.envs,
        "VLLM_B12X_MOE_FP4_FORCE_A16",
        force_a16,
    )
    scale = torch.ones(1)

    quant_config = nvfp4_oracle.make_nvfp4_moe_quant_config(
        backend=NvFp4MoeBackend.B12X,
        w13_scale=scale,
        w2_scale=scale,
        w13_scale_2=scale,
        w2_scale_2=scale,
        a13_scale=scale,
        a2_scale=scale,
    )

    assert quant_config.quant_dtype == expected_quant_dtype


def test_b12x_nvfp4_force_a16_leaves_row_order_for_b12x_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        nvfp4_oracle.envs,
        "VLLM_B12X_MOE_FP4_FORCE_A16",
        True,
    )
    reorder_w13 = None

    def prepare_for_b12x(**kwargs):
        nonlocal reorder_w13
        reorder_w13 = kwargs["reorder_w13"]
        return (
            kwargs["w13"],
            kwargs["w13_scale"],
            kwargs["w13_scale_2"],
            kwargs["a13_scale"],
            kwargs["w2"],
            kwargs["w2_scale"],
            kwargs["w2_scale_2"],
            kwargs["a2_scale"],
        )

    monkeypatch.setattr(
        nvfp4_oracle,
        "prepare_nvfp4_moe_layer_for_b12x",
        prepare_for_b12x,
    )
    tensor = torch.ones(1)

    nvfp4_oracle.convert_to_nvfp4_moe_kernel_format(
        nvfp4_backend=NvFp4MoeBackend.B12X,
        layer=SimpleNamespace(),
        w13=tensor,
        w13_scale=tensor,
        w13_scale_2=tensor,
        a13_scale=tensor,
        w2=tensor,
        w2_scale=tensor,
        w2_scale_2=tensor,
        a2_scale=tensor,
        is_act_and_mul=True,
    )

    assert reorder_w13 is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_b12x_nvfp4_preparation_pads_each_gated_half() -> None:
    device = torch.device("cuda")
    num_experts, hidden_size, intermediate_size = 2, 64, 48
    w13 = torch.ones(
        num_experts,
        2 * intermediate_size,
        hidden_size // 2,
        dtype=torch.uint8,
        device=device,
    )
    w13_scale = torch.ones(
        num_experts,
        2 * intermediate_size,
        hidden_size // 16,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    w2 = torch.ones(
        num_experts,
        hidden_size,
        intermediate_size // 2,
        dtype=torch.uint8,
        device=device,
    )
    w2_scale = torch.ones(
        num_experts,
        hidden_size,
        intermediate_size // 16,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    global_scale = torch.ones(num_experts, device=device)
    input_scale = torch.tensor([[1.0, 2.0], [3.0, 1.0]], device=device)

    prepared = prepare_nvfp4_moe_layer_for_b12x(
        w13,
        w13_scale,
        global_scale,
        input_scale,
        w2,
        w2_scale,
        global_scale,
        input_scale,
        is_act_and_mul=True,
    )

    prepared_w13, prepared_w13_scale, _, prepared_a13 = prepared[:4]
    prepared_w2, prepared_w2_scale, _, prepared_a2 = prepared[4:]
    assert prepared_w13.shape == (num_experts, 128, hidden_size // 2)
    assert prepared_w13_scale.shape == (num_experts, 128, hidden_size // 16)
    assert prepared_w2.shape == (num_experts, hidden_size, 32)
    assert prepared_w2_scale.shape == (num_experts, 128, 4)
    torch.testing.assert_close(prepared_a13, torch.tensor([2.0, 3.0], device=device))
    torch.testing.assert_close(prepared_a2, torch.tensor([2.0, 3.0], device=device))


def test_b12x_moe_uses_minimax_swiglu_parameters() -> None:
    config = make_dummy_moe_config(
        hidden_dim=128,
        intermediate_size=64,
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
    )
    config.swiglu_limit = 7.0
    config.swiglu_alpha = 1.702
    config.swiglu_beta = 1.0
    experts = B12xExperts(config, _quant_config("mxfp4", None))

    assert experts._swiglu_params(config.activation) == (7.0, 1.702, 1.0)


def test_b12x_moe_warmup_runs_each_planner_regime_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experts = B12xExperts(
        make_dummy_moe_config(
            num_experts=4,
            experts_per_token=2,
            hidden_dim=128,
            intermediate_size=64,
        ),
        _quant_config("mxfp4", None),
    )
    prepared = SimpleNamespace(
        num_experts=4,
        hidden_size=128,
        intermediate_size=64,
        w1_fp4=torch.empty(0),
    )
    layer = SimpleNamespace(
        activation=MoEActivation.SILU,
        apply_router_weight_on_input=False,
    )
    planned_tokens = []
    launched_tokens = []
    launched_dtypes = []
    launched_buffers = []
    launched_resources = []
    synchronized = []

    with pytest.raises(RuntimeError, match="process_weights_after_loading"):
        experts.warmup_launches(layer, token_counts=(1,))
    experts._prepared_experts = prepared

    def fake_execution_plan(**kwargs):
        tokens = kwargs["tokens"]
        if tokens <= 2:
            signature = ("micro", "decode")
        elif tokens <= 4:
            signature = ("dynamic", "small")
        else:
            signature = ("dynamic", "large")
        return SimpleNamespace(
            implementation=signature[0],
            execution=signature[1],
        )

    def fake_plan(**kwargs):
        planned_tokens.append(kwargs["tokens"])
        return SimpleNamespace(
            scratch_specs=lambda: [SimpleNamespace(dtype=torch.uint8, shape=(64,))]
        )

    def fake_run(**kwargs):
        launched_tokens.append(kwargs["hidden_states"].shape[0])
        launched_dtypes.append(kwargs["topk_ids"].dtype)
        launched_buffers.append(
            tuple(
                kwargs[name].data_ptr()
                for name in ("hidden_states", "output", "topk_weights", "scratch")
            )
        )
        launched_resources.append(
            tuple(
                weakref.ref(kwargs[name])
                for name in (
                    "hidden_states",
                    "output",
                    "topk_ids",
                    "topk_weights",
                    "scratch",
                )
            )
        )

    def fake_synchronize():
        assert all(
            resource() is not None
            for launch in launched_resources
            for resource in launch
        )
        synchronized.append(True)

    monkeypatch.setattr(b12x, "_b12x_moe_execution_plan", fake_execution_plan)
    monkeypatch.setattr(b12x, "_run_b12x_moe_plan", fake_run)
    monkeypatch.setattr(experts, "_plan", fake_plan)
    monkeypatch.setattr(b12x.torch.accelerator, "synchronize", fake_synchronize)

    warmed = experts.warmup_launches(layer, token_counts=(1, 2, 3, 4, 8))

    assert warmed == 3
    assert planned_tokens == [1, 3, 8]
    assert launched_tokens == [1, 1, 3, 3, 8, 8]
    assert launched_dtypes == [torch.int32, torch.int64] * 3
    # Both route-id dtypes of one regime run on the same input, output,
    # weight and scratch buffers; regimes do not share buffers.
    assert launched_buffers[0] == launched_buffers[1]
    assert launched_buffers[2] == launched_buffers[3]
    assert launched_buffers[4] == launched_buffers[5]
    assert len({launched_buffers[0], launched_buffers[2], launched_buffers[4]}) == 3
    assert synchronized == [True]


class _DeepseekV41Mxfp4Config(Mxfp4Config):
    def _make_moe_method(self, moe):
        return Mxfp4MoEMethod(moe, numerical_recipe="deepseek_v41")


@pytest.mark.skipif(not _has_b12x_moe(), reason="requires b12x MoE on SM120")
@torch.inference_mode()
def test_deepseek_v41_generic_factory_preserves_fp32_oracle_and_replay(
    workspace_init,
) -> None:
    """Exercise the production Factory→RoutedExperts→B12xExperts pipeline."""
    from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution

    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.fused_moe import FusedMoEFactory
    from vllm.model_executor.warmup.b12x_warmup import _collect_warmup_units

    vllm_config = VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    with set_current_vllm_config(vllm_config), torch.device("cuda"):
        moe = FusedMoEFactory(
            num_experts=1,
            top_k=1,
            hidden_size=256,
            intermediate_size=128,
            params_dtype=torch.bfloat16,
            quant_config=_DeepseekV41Mxfp4Config(),
            tp_size=1,
            dp_size=1,
            pcp_size=1,
            scoring_func="sqrtsoftplus",
            e_score_correction_bias=torch.zeros(1, device="cuda"),
            prefix="deepseek-v41-generic-oracle",
        )
        routed = moe.routed_experts
        for shard, code, exponent, shape in (
            ("w1", 0x22, 123, (128, 128)),
            ("w3", 0xCC, 123, (128, 128)),
            ("w2", 0x22, 120, (256, 64)),
        ):
            name = "w2_weight" if shard == "w2" else "w13_weight"
            for suffix, value, payload_shape in (
                ("", code, shape),
                ("_scale", exponent, (shape[0], shape[1] // 16)),
            ):
                routed.weight_loader(
                    getattr(routed, name + suffix),
                    torch.full(payload_shape, value, dtype=torch.uint8, device="cuda"),
                    f"experts.{name}{suffix}",
                    shard_id=shard,
                    expert_id=0,
                )
        routed.quant_method.process_weights_after_loading(routed)
        # Exercise the worker's real model-dtype handoff, not a mocked provider.
        (warmup_unit,) = _collect_warmup_units(routed, (1, 2, 4), torch.bfloat16)
        warmup_unit.compile()

        x = (
            torch.tensor(
                [1 / 16, -1 / 16, 1 / 32],
                device="cuda",
                dtype=torch.bfloat16,
            )[:, None]
            .expand(-1, 256)
            .contiguous()
        )
        logits = torch.zeros((3, 1), dtype=torch.float32, device="cuda")

        def oracle():
            gate = (x.float().sum(-1, keepdim=True) / 16).bfloat16().float()
            mid = (torch.nn.functional.silu(gate) * (-2 * gate)).bfloat16().float()
            scale = torch.exp2(torch.ceil(torch.log2(mid.abs().clamp_min(1e-4) / 448)))
            return (
                (mid / scale)
                .to(torch.float8_e4m3fn)
                .float()
                .mul_(scale)
                .bfloat16()
                .float()
                .expand_as(x)
            )

        with set_forward_context(None, vllm_config, num_tokens=x.shape[0]):
            output = moe(x, logits)
        assert output.dtype == torch.float32
        torch.testing.assert_close(output, oracle(), rtol=0, atol=0)
        freeze_kernel_resolution("V4.1 shared MoE output replay")
        try:
            with set_forward_context(None, vllm_config, num_tokens=2):
                small = moe(x[:2], logits[:2])
            torch.testing.assert_close(small, oracle()[:2], rtol=0, atol=0)
            graph = torch.cuda.CUDAGraph()
            with (
                set_forward_context(None, vllm_config, num_tokens=x.shape[0]),
                torch.cuda.graph(graph),
            ):
                captured_output = moe(x, logits)
            x.neg_()
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(captured_output, oracle(), rtol=0, atol=0)
        finally:
            unfreeze_kernel_resolution()


def test_b12x_moe_live_prefill_uses_registered_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experts = B12xExperts(
        make_dummy_moe_config(
            num_experts=4,
            experts_per_token=2,
            hidden_dim=128,
            intermediate_size=64,
            max_num_tokens=2048,
        ),
        _quant_config("mxfp4", None),
    )
    experts._prepared_experts = SimpleNamespace(
        plan=object(),
        num_experts=4,
        hidden_size=128,
        intermediate_size=64,
        w1_fp4=torch.empty(0),
    )
    layer = SimpleNamespace(
        activation=MoEActivation.SILU,
        apply_router_weight_on_input=False,
    )
    experts.get_b12x_warmup_unit(
        layer,
        (1, 4, 128, 2048),
        torch.bfloat16,
    )

    caps_seen = []

    class FakeCaps:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def fake_plan(caps):
        caps_seen.append(caps)
        return object()

    extension = SimpleNamespace(Caps=FakeCaps, plan=fake_plan)
    monkeypatch.setattr(b12x, "_require_b12x_fused_moe", lambda: extension)

    prefill_plan = experts._plan(
        tokens=571,
        topk=2,
        activation=MoEActivation.SILU,
    )
    assert (
        experts._plan(
            tokens=777,
            topk=2,
            activation=MoEActivation.SILU,
        )
        is prefill_plan
    )
    experts._plan(
        tokens=128,
        topk=2,
        activation=MoEActivation.SILU,
    )

    assert [caps.max_tokens for caps in caps_seen] == [2048, 128]
    assert [caps.core_token_counts for caps in caps_seen] == [(2048,), (128,)]
    with pytest.raises(ValueError, match="exceeds the configured prefill capacity"):
        experts._plan(
            tokens=2049,
            topk=2,
            activation=MoEActivation.SILU,
        )


def test_b12x_moe_reload_reprepares_current_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = SimpleNamespace(
        discards_source_parameters=False,
        quant_modes=("w4a16",),
        io_dtype="bfloat16",
        activation="silu",
    )
    prepared_inputs: list[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ] = []

    def prepare_weights(**kwargs):
        prepared_inputs.append(
            (
                kwargs["w1_fp4"],
                kwargs["w2_fp4"],
                kwargs["w1_blockscale"],
                kwargs["w2_blockscale"],
            )
        )
        return SimpleNamespace(plan=plan)

    extension = SimpleNamespace(
        plan_weights=lambda **_: plan, prepare_weights=prepare_weights
    )
    monkeypatch.setattr(b12x, "_require_b12x_fused_moe", lambda: extension)
    experts = B12xExperts(
        make_dummy_moe_config(num_experts=2, hidden_dim=4, intermediate_size=8),
        _quant_config("mxfp4", None),
    )
    layer = SimpleNamespace(
        activation=MoEActivation.SILU,
        apply_router_weight_on_input=False,
        w13_weight=torch.full((2, 8, 2), 1, dtype=torch.uint8),
        w2_weight=torch.full((2, 4, 4), 1, dtype=torch.uint8),
        w13_weight_scale=torch.full((2, 8, 1), 1, dtype=torch.uint8),
        w2_weight_scale=torch.full((2, 4, 1), 1, dtype=torch.uint8),
    )

    experts.process_weights_after_loading(layer)
    layer.w13_weight = torch.full_like(layer.w13_weight, 2)
    layer.w2_weight = torch.full_like(layer.w2_weight, 2)
    layer.w13_weight_scale = torch.full_like(layer.w13_weight_scale, 3)
    layer.w2_weight_scale = torch.full_like(layer.w2_weight_scale, 3)
    experts.process_weights_after_loading(layer)

    assert len(prepared_inputs) == 2
    assert prepared_inputs[-1][0] is layer.w13_weight
    assert prepared_inputs[-1][1] is layer.w2_weight
    assert prepared_inputs[-1][2] is layer.w13_weight_scale
    assert prepared_inputs[-1][3] is layer.w2_weight_scale


def test_b12x_source_release_preserves_prepared_storage_owner() -> None:
    layer = torch.nn.Module()
    for name, shape in (
        ("w13_weight", (4, 32, 16)),
        ("w2_weight", (4, 64, 8)),
        ("w13_weight_scale", (4, 32, 2)),
        ("w2_weight_scale", (4, 64, 1)),
    ):
        layer.register_parameter(
            name,
            torch.nn.Parameter(
                torch.empty(shape, dtype=torch.uint8),
                requires_grad=False,
            ),
        )
    experts = B12xExperts(
        make_dummy_moe_config(hidden_dim=128, intermediate_size=64),
        _quant_config("mxfp4", None),
    )
    owner = SimpleNamespace(
        w1_fp4=layer.w13_weight,
        w2_fp4=layer.w2_weight,
        w1_blockscale=layer.w13_weight_scale,
        w2_blockscale=layer.w2_weight_scale,
    )
    experts._prepared_experts = owner
    owner_tensors = (
        owner.w1_fp4,
        owner.w2_fp4,
        owner.w1_blockscale,
        owner.w2_blockscale,
    )
    owner_ptrs = tuple(tensor.untyped_storage().data_ptr() for tensor in owner_tensors)

    experts._release_source_parameters(layer)
    experts._release_source_parameters(layer)

    assert layer.w13_weight.numel() == 0
    assert layer.w2_weight.numel() == 0
    assert layer.w13_weight_scale.numel() == 0
    assert layer.w2_weight_scale.numel() == 0
    assert (
        tuple(tensor.untyped_storage().data_ptr() for tensor in owner_tensors)
        == owner_ptrs
    )


def test_b12x_moe_rejects_router_weight_on_input_for_w4a8() -> None:
    experts = B12xExperts(
        make_dummy_moe_config(hidden_dim=256, intermediate_size=64),
        _quant_config("mxfp4", "mxfp8"),
    )
    layer = SimpleNamespace(
        activation=MoEActivation.SILU,
        apply_router_weight_on_input=True,
    )

    with pytest.raises(
        ValueError,
        match="apply_router_weight_on_input only with W4A16",
    ):
        experts.process_weights_after_loading(layer)


def test_b12x_moe_workspace_uses_prepared_router_weight_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experts = B12xExperts(
        make_dummy_moe_config(hidden_dim=128, intermediate_size=64),
        _quant_config("mxfp4", None),
    )
    prepared = SimpleNamespace(
        plan=SimpleNamespace(discards_source_parameters=False),
    )
    layer = SimpleNamespace(
        activation=MoEActivation.SILU,
        apply_router_weight_on_input=True,
        w13_weight=torch.empty(0),
        w2_weight=torch.empty(0),
        w13_weight_scale=torch.empty(0),
        w2_weight_scale=torch.empty(0),
    )
    monkeypatch.setattr(experts, "_prepare_experts", lambda **kwargs: prepared)
    planned = []

    def fake_plan(**kwargs):
        planned.append(kwargs)
        return SimpleNamespace(
            scratch_specs=lambda: [SimpleNamespace(dtype=torch.uint8, shape=(64,))]
        )

    monkeypatch.setattr(experts, "_plan", fake_plan)

    experts.process_weights_after_loading(layer)
    assert layer.b12x_warmup_provider is experts
    experts.workspace_shapes(
        8,
        128,
        128,
        2,
        4,
        4,
        None,
        MoEActivation.SILU,
    )

    assert planned == [
        {
            "tokens": 8,
            "topk": 2,
            "activation": MoEActivation.SILU,
            "apply_router_weight_on_input": True,
        }
    ]


@dataclass
class _B12xMoeCase:
    hidden_states: torch.Tensor
    score: torch.Tensor
    w1: torch.Tensor
    w2: torch.Tensor
    w1_ref: torch.Tensor
    w2_ref: torch.Tensor
    quant_config: FusedMoEQuantConfig
    activation: MoEActivation
    activation_dtype: str | None
    topk: int = 2


def _make_b12x_moe_case(
    weight_dtype: str,
    activation_dtype: str | None,
    *,
    activation: MoEActivation = MoEActivation.SILU,
    tokens: int = 16,
    seed: int = 19,
) -> _B12xMoeCase:
    set_random_seed(seed)
    num_experts, hidden_size, intermediate_size = 4, 512, 128
    dtype = torch.bfloat16
    hidden_states = torch.randn((tokens, hidden_size), device="cuda", dtype=dtype) / 10
    w1_rows = 2 * intermediate_size if activation.is_gated else intermediate_size
    w1 = (
        torch.randn(
            (num_experts, w1_rows, hidden_size),
            device="cuda",
            dtype=dtype,
        )
        / 15
    )
    w2 = (
        torch.randn(
            (num_experts, hidden_size, intermediate_size),
            device="cuda",
            dtype=dtype,
        )
        / 15
    )

    if weight_dtype == "mxfp4":
        w1_q, w1_scale = mxfp4_quantize(w1)
        w2_q, w2_scale = mxfp4_quantize(w2)
        w1_ref = torch.stack(
            [dq_mxfp4_torch(w1_q[e], w1_scale[e], dtype) for e in range(num_experts)]
        )
        w2_ref = torch.stack(
            [dq_mxfp4_torch(w2_q[e], w2_scale[e], dtype) for e in range(num_experts)]
        )
        if activation_dtype is None:
            quant_config = mxfp4_w4a16_moe_quant_config(
                w1_scale=w1_scale,
                w2_scale=w2_scale,
            )
        else:
            quant_config = FusedMoEQuantConfig.make(
                quant_dtype=activation_dtype,
                weight_dtype=weight_dtype,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
            )
    elif weight_dtype == "nvfp4":
        w1_q, w1_scale, w1_global_scale = _quantize_nvfp4_linear(w1)
        w2_q, w2_scale, w2_global_scale = _quantize_nvfp4_linear(w2)
        w1_ref = torch.stack(
            [
                _dequantize_nvfp4_linear(
                    w1_q[e], w1_scale[e], w1_global_scale[e], dtype
                )
                for e in range(num_experts)
            ]
        )
        w2_ref = torch.stack(
            [
                _dequantize_nvfp4_linear(
                    w2_q[e], w2_scale[e], w2_global_scale[e], dtype
                )
                for e in range(num_experts)
            ]
        )
        input_scale = torch.full(
            (num_experts,),
            1.0 if activation_dtype is None else 1.0 / 1024.0,
            device="cuda",
            dtype=torch.float32,
        )
        prepared = prepare_nvfp4_moe_layer_for_b12x(
            w1_q,
            w1_scale,
            1.0 / w1_global_scale,
            input_scale,
            w2_q,
            w2_scale,
            1.0 / w2_global_scale,
            input_scale,
            is_act_and_mul=activation.is_gated,
            reorder_w13=False,
        )
        w1_q, w1_scale, w1_alpha, a1_scale = prepared[:4]
        w2_q, w2_scale, w2_alpha, a2_scale = prepared[4:]
        if activation_dtype is None:
            quant_config = nvfp4_w4a16_moe_quant_config(
                g1_alphas=w1_alpha,
                g2_alphas=w2_alpha,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
            )
        else:
            quant_config = FusedMoEQuantConfig.make(
                quant_dtype=activation_dtype,
                weight_dtype=weight_dtype,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                g1_alphas=w1_alpha,
                g2_alphas=w2_alpha,
                a1_gscale=1.0 / a1_scale,
                a2_gscale=1.0 / a2_scale,
            )
    else:
        raise ValueError(f"unsupported test weight dtype {weight_dtype}")

    return _B12xMoeCase(
        hidden_states=hidden_states,
        score=torch.randn((tokens, num_experts), device="cuda", dtype=dtype),
        w1=w1_q,
        w2=w2_q,
        w1_ref=w1_ref,
        w2_ref=w2_ref,
        quant_config=quant_config,
        activation=activation,
        activation_dtype=activation_dtype,
    )


@pytest.mark.skipif(not _has_b12x_moe(), reason="requires b12x MoE on SM120")
@pytest.mark.parametrize(
    "weight_dtype,activation_dtype,activation",
    [
        pytest.param("mxfp4", "mxfp8", MoEActivation.SILU, id="mxfp4-mxfp8"),
        pytest.param("mxfp4", None, MoEActivation.SILU, id="mxfp4-bf16"),
        pytest.param("nvfp4", "nvfp4", MoEActivation.SILU, id="nvfp4-nvfp4"),
        pytest.param("nvfp4", "mxfp8", MoEActivation.SILU, id="nvfp4-mxfp8"),
        pytest.param("nvfp4", None, MoEActivation.SILU, id="nvfp4-bf16-silu"),
        pytest.param(
            "nvfp4",
            None,
            MoEActivation.RELU2_NO_MUL,
            id="nvfp4-bf16-relu2",
        ),
    ],
)
@torch.inference_mode()
def test_b12x_moe_matches_torch(
    weight_dtype: str,
    activation_dtype: str | None,
    activation: MoEActivation,
    workspace_init,
) -> None:
    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        case = _make_b12x_moe_case(
            weight_dtype,
            activation_dtype,
            activation=activation,
        )
        reference = torch_moe(
            case.hidden_states,
            case.w1_ref,
            case.w2_ref,
            case.score,
            case.topk,
            activation=case.activation,
        )
        if activation_dtype == "nvfp4":
            topk_weights, topk_ids, _ = fused_topk(
                case.hidden_states,
                case.score,
                case.topk,
                renormalize=False,
            )
            reference = _nvfp4_activation_reference(
                case.hidden_states,
                case.w1_ref,
                case.w2_ref,
                topk_weights,
                topk_ids,
                case.quant_config.a1_gscale,
                case.quant_config.a2_gscale,
            )

        output = _run_b12x_moe(
            case.hidden_states,
            case.w1,
            case.w2,
            case.score,
            case.topk,
            case.activation,
            case.quant_config,
        )
    torch.testing.assert_close(output, reference, atol=2e-1, rtol=2e-1)
    cosine = torch.nn.functional.cosine_similarity(
        output.flatten().float(),
        reference.flatten().float(),
        dim=0,
    )
    assert cosine > 0.99


@pytest.mark.skipif(not _has_b12x_moe(), reason="requires b12x MoE on SM120")
@pytest.mark.parametrize(
    "weight_dtype,activation_dtype",
    [
        pytest.param("mxfp4", "mxfp8", id="w4a8"),
        pytest.param("nvfp4", None, id="w4a16"),
    ],
)
@torch.inference_mode()
def test_b12x_moe_cuda_graph_replay(
    weight_dtype: str,
    activation_dtype: str | None,
    workspace_init,
) -> None:
    from vllm.v1.worker.workspace import lock_workspace

    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        case = _make_b12x_moe_case(
            weight_dtype,
            activation_dtype,
            tokens=4,
            seed=23,
        )
        kernel = _make_b12x_moe_kernel(
            case.hidden_states,
            case.w1,
            case.w2,
            case.topk,
            case.activation,
            case.quant_config,
        )
        topk_weights, topk_ids, _ = fused_topk(
            case.hidden_states,
            case.score,
            case.topk,
            renormalize=False,
        )
        assert topk_weights.dtype == torch.float32 and topk_weights.is_contiguous()
        assert topk_ids.dtype == torch.int32 and topk_ids.is_contiguous()

        def apply() -> torch.Tensor:
            return kernel.apply(
                hidden_states=case.hidden_states,
                w1=case.w1,
                w2=case.w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=case.activation,
                global_num_experts=case.w1.shape[0],
                expert_map=None,
                apply_router_weight_on_input=False,
            )

        expected = apply().clone()
        lock_workspace()
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        with torch.cuda.graph(graph, stream=stream):
            actual = apply()
        graph.replay()
        torch.accelerator.synchronize()

    assert torch.isfinite(expected).all()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def _run_deepseek_v41_factory_tp(rank: int, port: int) -> None:
    from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution

    from tests.utils import init_test_distributed_environment
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.fused_moe import FusedMoEFactory
    from vllm.model_executor.layers.fused_moe.runner import (
        shared_experts as shared_experts_module,
    )
    from vllm.models.deepseek_v4_1 import b12x_layers
    from vllm.v1.worker.workspace import (
        collect_cuda_graph_capture_resources,
        current_workspace_manager,
        init_workspace_manager,
        lock_workspace,
    )

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    vllm_config = VllmConfig(
        parallel_config=ParallelConfig(tensor_parallel_size=4, pipeline_parallel_size=1)
    )
    with (
        set_current_vllm_config(vllm_config),
        torch.device(device),
        torch.inference_mode(),
    ):
        try:
            init_test_distributed_environment(4, 1, rank, str(port), local_rank=rank)
            init_workspace_manager(device)
            moe = FusedMoEFactory(
                num_experts=4,
                top_k=1,
                hidden_size=256,
                intermediate_size=2304,
                params_dtype=torch.bfloat16,
                quant_config=_DeepseekV41Mxfp4Config(),
                tp_size=4,
                dp_size=1,
                pcp_size=1,
                scoring_func="sqrtsoftplus",
                e_score_correction_bias=torch.zeros(4, device=device),
                prefix=f"deepseek-v41-generic-tp-{rank}",
            )
            routed = moe.routed_experts
            assert routed.w13_weight.shape == (4, 1152, 128)
            assert routed.w13_weight_scale.shape == (4, 1152, 8)
            assert routed.w2_weight.shape == (4, 256, 288)
            assert routed.w2_weight_scale.shape == (4, 256, 18)
            for expert_id in range(4):
                for shard, name, code, exponent, shape in (
                    ("w1", "w13_weight", 0x22, 123, (2304, 128)),
                    ("w3", "w13_weight", 0xCC, 123, (2304, 128)),
                    ("w2", "w2_weight", 0x22, None, (256, 1152)),
                ):
                    routed.weight_loader(
                        getattr(routed, name),
                        torch.full(shape, code, dtype=torch.uint8, device=device),
                        f"experts.{name}",
                        shard_id=shard,
                        expert_id=expert_id,
                    )
                    if exponent is not None:
                        routed.weight_loader(
                            getattr(routed, f"{name}_scale"),
                            torch.full(
                                (shape[0], shape[1] // 16),
                                exponent,
                                dtype=torch.uint8,
                                device=device,
                            ),
                            f"experts.{name}_scale",
                            shard_id=shard,
                            expert_id=expert_id,
                        )
                w2_scale = torch.full((256, 72), 127, dtype=torch.uint8, device=device)
                for shard_rank in range(4):
                    w2_scale[:, shard_rank * 18 : (shard_rank + 1) * 18] = (
                        120 + shard_rank + expert_id
                    )
                routed.weight_loader(
                    routed.w2_weight_scale,
                    w2_scale,
                    "experts.w2_weight_scale",
                    shard_id="w2",
                    expert_id=expert_id,
                )
            assert torch.all(routed.w13_weight[:, :576] == 0x22)
            assert torch.all(routed.w13_weight[:, 576:] == 0xCC)
            assert torch.all(routed.w13_weight_scale == 123)
            assert torch.all(routed.w2_weight == 0x22)
            for expert_id in range(4):
                assert torch.all(
                    routed.w2_weight_scale[expert_id] == 120 + rank + expert_id
                )
            source_prepared_storage = tuple(
                (
                    tensor.untyped_storage().data_ptr(),
                    tensor.untyped_storage().nbytes(),
                    tensor.numel() * tensor.element_size(),
                )
                for tensor in (
                    routed.w13_weight,
                    routed.w13_weight_scale,
                    routed.w2_weight,
                    routed.w2_weight_scale,
                )
            )
            routed.quant_method.process_weights_after_loading(routed)
            prepared = routed._b12x_prepared_experts.representation.value
            assert prepared.n64_repack
            for packed, (source_ptr, source_storage_bytes, source_bytes) in zip(
                (
                    prepared.w13_rp,
                    prepared.w13_sfb,
                    prepared.w2_rp,
                    prepared.w2_sfb,
                ),
                source_prepared_storage,
                strict=True,
            ):
                assert packed.untyped_storage().data_ptr() == source_ptr
                assert packed.untyped_storage().nbytes() == source_storage_bytes
                assert packed.numel() * packed.element_size() == source_bytes
            x = (
                torch.tensor(
                    [1 / 16, -1 / 16, 1 / 32, -1 / 32],
                    dtype=torch.bfloat16,
                    device=device,
                )[:, None]
                .expand(-1, 256)
                .contiguous()
            )
            logits = torch.eye(4, dtype=torch.float32, device=device) * 20

            def oracle():
                gate = (x.float().sum(-1, keepdim=True) / 16).bfloat16().float()
                mid = (torch.nn.functional.silu(gate) * (-2 * gate)).bfloat16().float()
                scale = torch.exp2(
                    torch.ceil(torch.log2(mid.abs().clamp_min(1e-4) / 448))
                )
                quantized = (mid / scale).to(torch.float8_e4m3fn).float() * scale
                selected = logits.argmax(-1)[:, None]
                expected = torch.zeros_like(x, dtype=torch.float32)
                for shard_rank in range(4):
                    partial = quantized * (
                        576 * torch.exp2((selected + shard_rank - 7).float())
                    )
                    expected += partial.bfloat16().float()
                return expected

            with set_forward_context(None, vllm_config, num_tokens=4):
                output = moe(x, logits)
            assert output.dtype == torch.float32
            torch.testing.assert_close(output, oracle(), rtol=0, atol=0)
            graph = torch.cuda.CUDAGraph()
            freeze_kernel_resolution("V4.1 shared TP MoE replay")
            try:
                with set_forward_context(None, vllm_config, num_tokens=2):
                    small = moe(x[:2], logits[:2])
                torch.testing.assert_close(small, oracle()[:2], rtol=0, atol=0)
                with (
                    collect_cuda_graph_capture_resources() as retained,
                    set_forward_context(None, vllm_config, num_tokens=4),
                    torch.cuda.graph(graph),
                ):
                    replayed = moe(x, logits)
                x.neg_()
                logits.copy_(logits.roll(1, dims=1))
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(replayed, oracle(), rtol=0, atol=0)
                del retained
            finally:
                unfreeze_kernel_resolution()
                # NCCL communicator abort waits for captured graphs to release it.
                graph.reset()

            from vllm.models.deepseek_v4.nvidia.model import DeepseekV4MLP
            from vllm.models.deepseek_v4_1.quant_config import DeepseekV41FP8Config

            shared = DeepseekV4MLP(
                hidden_size=256,
                intermediate_size=128,
                hidden_act="silu",
                swiglu_limit=10.0,
                quant_config=DeepseekV41FP8Config(
                    is_checkpoint_fp8_serialized=True,
                    weight_block_size=[32, 32],
                ),
                reduce_results=False,
                prefix=f"deepseek-v41-composition-{rank}.shared_experts",
            )
            shared.gate_up_proj.weight[:32].fill_(1.0)
            shared.gate_up_proj.weight[32:].fill_(1 / 16)
            shared.down_proj.weight.fill_(0.5)
            for linear in (shared.gate_up_proj, shared.down_proj):
                linear.weight_scale_inv.view(torch.uint8).fill_(127)
                linear.quant_method.process_weights_after_loading(linear)
            composition = FusedMoEFactory(
                num_experts=2,
                top_k=2,
                hidden_size=256,
                intermediate_size=128,
                params_dtype=torch.bfloat16,
                quant_config=_DeepseekV41Mxfp4Config(),
                tp_size=4,
                dp_size=1,
                pcp_size=1,
                scoring_func="sqrtsoftplus",
                e_score_correction_bias=torch.zeros(2, device=device),
                shared_experts=shared,
                prefix=f"deepseek-v41-composition-{rank}",
            )
            composition_weights = composition.routed_experts
            for expert_id in range(2):
                for shard, code, exponent, shape in (
                    ("w1", 0x22, 123, (128, 128)),
                    ("w3", 0xCC, 123, (128, 128)),
                    ("w2", 0x22, 130 if expert_id == 0 else 120, (256, 64)),
                ):
                    name = "w2_weight" if shard == "w2" else "w13_weight"
                    for suffix, value, payload_shape in (
                        ("", code, shape),
                        ("_scale", exponent, (shape[0], shape[1] // 16)),
                    ):
                        composition_weights.weight_loader(
                            getattr(composition_weights, name + suffix),
                            torch.full(
                                payload_shape, value, dtype=torch.uint8, device=device
                            ),
                            f"experts.{name}{suffix}",
                            shard_id=shard,
                            expert_id=expert_id,
                        )
            composition_weights.quant_method.process_weights_after_loading(
                composition_weights
            )
            composition_input = torch.full(
                (1, 256), 1 / 16, dtype=torch.bfloat16, device=device
            )
            composition_logits = torch.full(
                (1, 2), 10.0, dtype=torch.float32, device=device
            )

            def quantized_uniform(value):
                scale = torch.exp2(
                    torch.ceil(torch.log2(value.abs().clamp_min(1e-4) / 448))
                )
                return (value / scale).to(torch.float8_e4m3fn).float() * scale

            def composition_oracle():
                total = composition_input.float().sum(-1, keepdim=True)
                gate = (total / 16).bfloat16().float()
                mid = (
                    (torch.nn.functional.silu(gate) * (-2 * gate) * 0.5)
                    .bfloat16()
                    .float()
                )
                routed_mid = quantized_uniform(mid)
                local_routed = (routed_mid * 32 * 8).bfloat16().float()
                local_routed += (routed_mid * 32 / 128).bfloat16().float()
                shared_gate = total.bfloat16().float().clamp(max=10)
                shared_up = (total / 16).bfloat16().float().clamp(-10, 10)
                shared_mid = (
                    (torch.nn.functional.silu(shared_gate) * shared_up)
                    .bfloat16()
                    .float()
                )
                local_shared = (
                    (quantized_uniform(shared_mid) * 32 * 0.5).bfloat16().float()
                )
                return ((local_routed + local_shared) * 4).expand(1, 256)

            with set_forward_context(None, vllm_config, num_tokens=1):
                warmup = composition(composition_input, composition_logits)
            torch.testing.assert_close(
                warmup, torch.full_like(warmup, -128.75), rtol=0, atol=0
            )

            manager = current_workspace_manager()
            workspace_sizes = [
                workspace.numel() if workspace is not None else 0
                for workspace in manager._current_workspaces
            ]
            lock_workspace()
            allocations: list[list[torch.Tensor]] = []
            preallocated_bindings: list[torch.Tensor] = []
            preallocated_lookups: list[torch.Tensor | None] = []
            allocation_streams: list[int] = []
            shared_streams: list[int] = []
            get_simultaneous = manager.get_simultaneous
            use_preallocated_workspace = (
                shared_experts_module.use_preallocated_workspace
            )
            current_preallocated_workspace = b12x_layers.current_preallocated_workspace

            def observe_allocation(*requests):
                regions = get_simultaneous(*requests)
                allocations.append(regions)
                allocation_streams.append(torch.cuda.current_stream().cuda_stream)
                return regions

            def observe_preallocated_binding(workspace):
                preallocated_bindings.append(workspace)
                return use_preallocated_workspace(workspace)

            def observe_preallocated_lookup():
                workspace = current_preallocated_workspace()
                preallocated_lookups.append(workspace)
                shared_streams.append(torch.cuda.current_stream().cuda_stream)
                return workspace

            for threshold in (0, 4):
                allocations.clear()
                preallocated_bindings.clear()
                preallocated_lookups.clear()
                allocation_streams.clear()
                shared_streams.clear()
                with (
                    patch.object(
                        shared_experts_module.envs,
                        "VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD",
                        threshold,
                    ),
                    patch.object(
                        manager, "get_simultaneous", side_effect=observe_allocation
                    ),
                    patch.object(
                        shared_experts_module,
                        "use_preallocated_workspace",
                        side_effect=observe_preallocated_binding,
                    ),
                    patch.object(
                        b12x_layers,
                        "current_preallocated_workspace",
                        side_effect=observe_preallocated_lookup,
                    ),
                    set_forward_context(None, vllm_config, num_tokens=1),
                ):
                    composed = composition(composition_input, composition_logits)
                assert len(allocations) == 1
                regions = allocations[0]
                assert len(regions) == 4
                storage = regions[0].untyped_storage().data_ptr()
                assert all(
                    region.untyped_storage().data_ptr() == storage for region in regions
                )
                for index, left in enumerate(regions):
                    for right in regions[index + 1 :]:
                        assert (
                            left.data_ptr() + left.numel() * left.element_size()
                            <= right.data_ptr()
                            or right.data_ptr() + right.numel() * right.element_size()
                            <= left.data_ptr()
                        )
                assert len(preallocated_bindings) == 1
                assert preallocated_bindings[0] is regions[-1]
                assert len(preallocated_lookups) == 2
                assert all(
                    workspace is regions[-1] for workspace in preallocated_lookups
                )
                assert all(
                    (stream != allocation_streams[0]) == bool(threshold)
                    for stream in shared_streams
                )
                assert workspace_sizes == [
                    workspace.numel() if workspace is not None else 0
                    for workspace in manager._current_workspaces
                ]
                # Native routed (-192 - 0.1875) and shared (+160), per rank.
                torch.testing.assert_close(
                    composed, torch.full_like(composed, -128.75), rtol=0, atol=0
                )
            graph = torch.cuda.CUDAGraph()
            freeze_kernel_resolution("V4.1 real shared block32 workspace reuse")
            try:
                with (
                    collect_cuda_graph_capture_resources() as retained,
                    set_forward_context(None, vllm_config, num_tokens=1),
                    torch.cuda.graph(graph),
                ):
                    replayed = composition(composition_input, composition_logits)
                for value in (1 / 16, -1 / 16, 1 / 32):
                    composition_input.fill_(value)
                    graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(
                        replayed, composition_oracle(), rtol=0, atol=0
                    )
                assert workspace_sizes == [
                    workspace.numel() if workspace is not None else 0
                    for workspace in manager._current_workspaces
                ]
                del retained
            finally:
                unfreeze_kernel_resolution()
                graph.reset()
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


@pytest.mark.distributed(num_gpus=4)
@pytest.mark.skipif(
    not _has_b12x_moe() or torch.cuda.device_count() < 4,
    reason="requires four SM120 GPUs and b12x MoE",
)
def test_deepseek_v41_generic_factory_true_tp_loading() -> None:
    from vllm.utils.network_utils import get_open_port

    torch.multiprocessing.spawn(
        _run_deepseek_v41_factory_tp,
        args=(get_open_port(),),
        nprocs=4,
        join=True,
    )
