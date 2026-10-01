# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the b12x tensor-parallel MoE integration."""

from __future__ import annotations

import gc
import weakref
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import TYPE_CHECKING

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
from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    select_deepseek_v4_mxfp4_moe_backend,
    select_mxfp4_moe_backend,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    select_nvfp4_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.b12x_moe import (
    prepare_nvfp4_moe_layer_for_b12x,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import mxfp4_quantize
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp4Static,
    kMxfp8Dynamic,
    kMxfp8Static,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform
from vllm.utils.b12x import B12xPreparationUnit, B12xWorkload
from vllm.utils.torch_utils import set_random_seed

if TYPE_CHECKING:
    from b12x.preparation import PreparationSession


def _prepare(
    layer,
    *,
    device,
    counts,
    fixed=(),
    output_dtype=torch.bfloat16,
    max_tokens=None,
    autotune=False,
    stage="weights",
):
    """Collect a layer's preparation units and fill their plans in place."""
    from b12x.preparation import PreparationSession

    max_tokens = max_tokens or max(counts)
    workload = B12xWorkload(
        stage=stage,
        token_counts=tuple(sorted(set(counts))),
        fixed_token_counts=tuple(sorted(set(fixed))),
        output_dtype=output_dtype,
        max_tokens=max_tokens,
        max_seqs=1,
        max_model_len=max_tokens,
    )
    provider = layer.b12x_preparation_provider
    units = list(provider.get_b12x_preparation_units(layer, workload))
    requests = tuple(request for unit in units for request in unit.requests)
    session = PreparationSession(device=device, autotune=autotune)
    session.prepare(requests, autotune=autotune)
    return session, units


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


def _has_b12x_mxfp8_moe() -> bool:
    return _has_b12x_moe() and b12x._b12x_has_mxfp8_moe()


def _make_b12x_moe_kernel(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk: int,
    activation: MoEActivation,
    quant_config: FusedMoEQuantConfig,
    *,
    fixed_token_counts: tuple[int, ...] = (),
) -> tuple[mk.FusedMoEKernel, PreparationSession, list[B12xPreparationUnit]]:
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
    tokens = int(hidden_states.shape[0])
    experts.process_weights_after_loading(layer)
    session, units = _prepare(
        layer,
        device=hidden_states.device,
        counts=(*fixed_token_counts, tokens),
        fixed=fixed_token_counts,
        output_dtype=hidden_states.dtype,
        max_tokens=tokens,
    )
    return (
        mk.FusedMoEKernel(
            maybe_make_prepare_finalize(
                moe=moe_config,
                quant_config=quant_config,
                allow_new_interface=True,
                use_monolithic=False,
            ),
            experts,
        ),
        session,
        units,
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
    kernel, session, _ = _make_b12x_moe_kernel(
        hidden_states,
        w1,
        w2,
        topk,
        activation,
        quant_config,
    )
    try:
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
    finally:
        session.close()


def _quant_config(
    weight_dtype: str,
    activation_dtype: str | None,
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
            {},
            {"activation": MoEActivation.RELU2_NO_MUL},
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
    """An explicit `moe_backend='b12x'` selects the b12x MXFP4 W4A8 experts.

    The activation key -- dynamic MXFP8 or absent -- does not change the
    choice while A16 forcing is off.
    """
    monkeypatch.setattr(B12xExperts, "_supports_current_device", lambda: True)
    monkeypatch.setattr(mxfp4_oracle, "_user_moe_activation_override", lambda: None)
    monkeypatch.setattr(
        mxfp4_oracle.envs,
        "VLLM_B12X_MOE_FP4_FORCE_A16",
        force_a16,
    )
    config = make_dummy_moe_config(
        hidden_dim=2560, intermediate_size=640, experts_per_token=10
    )
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
    """DeepSeek V4 b12x selection follows the forced-A16 knob.

    MXFP4 weights keep the dynamic-MXFP8 activation contract unless A16 is
    forced, which selects the BF16 contract.
    """
    monkeypatch.setattr(B12xExperts, "_supports_current_device", lambda: True)
    monkeypatch.setattr(
        mxfp4_oracle.envs,
        "VLLM_B12X_MOE_FP4_FORCE_A16",
        force_a16,
    )
    config = make_dummy_moe_config(
        hidden_dim=2560, intermediate_size=640, experts_per_token=10
    )
    config.moe_backend = "b12x"

    backend, experts_cls = select_deepseek_v4_mxfp4_moe_backend(config)

    assert backend == expected_backend
    assert experts_cls is B12xExperts


def test_deepseek_v4_flashinfer_cutlass_falls_through_to_w4a8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit flashinfer_cutlass must try the W4A8 variant when the BF16
    variant is unsupported (the BF16 variant is gated to SM90)."""
    from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe import (
        FlashInferExperts,
    )

    monkeypatch.setattr(FlashInferExperts, "_supports_current_device", lambda: True)

    def sm120_quant_gate(weight_key, activation_key):
        return (weight_key, activation_key) == (
            mxfp4_oracle.kMxfp4Static,
            mxfp4_oracle.kMxfp8Dynamic,
        )

    monkeypatch.setattr(
        FlashInferExperts, "_supports_quant_scheme", staticmethod(sm120_quant_gate)
    )
    config = make_dummy_moe_config(hidden_dim=256, intermediate_size=64)
    config.moe_backend = "flashinfer_cutlass"

    backend, experts_cls = select_deepseek_v4_mxfp4_moe_backend(config)

    assert backend == Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_MXFP8
    assert experts_cls is FlashInferExperts


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


@pytest.mark.parametrize(
    ("layer_max_input_scale", "expected_a13_scale", "expected_a2_scale"),
    [
        ("0", torch.tensor([2.0, 3.0]), torch.tensor([7.0, 5.0])),
        ("w13", torch.tensor(3.0), torch.tensor([7.0, 5.0])),
        ("w2", torch.tensor([2.0, 3.0]), torch.tensor(7.0)),
        ("all", torch.tensor(3.0), torch.tensor(7.0)),
        ("1", torch.tensor(3.0), torch.tensor(7.0)),
    ],
)
def test_b12x_nvfp4_layer_max_input_scales_are_independent(
    monkeypatch: pytest.MonkeyPatch,
    layer_max_input_scale: str,
    expected_a13_scale: torch.Tensor,
    expected_a2_scale: torch.Tensor,
) -> None:
    monkeypatch.setattr(
        nvfp4_oracle.envs,
        "VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE",
        layer_max_input_scale,
    )
    a13_scale = torch.tensor([2.0, 3.0])
    a2_scale = torch.tensor([7.0, 5.0])

    def prepare_for_b12x(**kwargs):
        return (
            kwargs["w13"],
            kwargs["w13_scale"],
            kwargs["w13_scale_2"],
            a13_scale,
            kwargs["w2"],
            kwargs["w2_scale"],
            kwargs["w2_scale_2"],
            a2_scale,
        )

    monkeypatch.setattr(
        nvfp4_oracle,
        "prepare_nvfp4_moe_layer_for_b12x",
        prepare_for_b12x,
    )
    tensor = torch.ones(1)

    prepared = nvfp4_oracle.convert_to_nvfp4_moe_kernel_format(
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

    torch.testing.assert_close(prepared[3], expected_a13_scale)
    torch.testing.assert_close(prepared[7], expected_a2_scale)


@pytest.mark.parametrize("uniform", [True, False])
def test_b12x_nvfp4_preparation_preserves_static_scale_contract(
    monkeypatch: pytest.MonkeyPatch, uniform: bool
) -> None:
    """Loaded uniform scales admit shared quantization without losing the vector."""
    fused_moe = pytest.importorskip("b12x.moe.fused_moe")
    from b12x.moe.fused_moe._impl import B12XFP4ExpertWeights

    # Keep the real scale-ownership check; only CUDA weight packing is excluded.
    def prepare_weights(*, plan, weights):
        return B12XFP4ExpertWeights(
            plan=plan._impl,
            a1_gscale=weights.input_scale,
            a2_gscale=weights.intermediate_scale,
            w1_fp4=weights.w13,
            w1_blockscale=weights.w13_block_scales,
            w1_alphas=weights.w13_global_scales,
            w2_fp4=weights.w2,
            w2_blockscale=weights.w2_block_scales,
            w2_alphas=weights.w2_global_scales,
            immutable_input_scales=weights.immutable_input_scales,
        )

    monkeypatch.setattr(fused_moe, "prepare_weights", prepare_weights)
    monkeypatch.setattr(b12x, "_require_b12x_fused_moe", lambda: fused_moe)
    monkeypatch.setattr(b12x, "_is_current_stream_capturing", lambda: False)
    scales = torch.ones(4)
    if not uniform:
        scales[-1] = torch.nextafter(scales[-1], torch.tensor(float("inf")))
    config = make_dummy_moe_config(num_experts=4, hidden_dim=128, intermediate_size=128)
    quant = FusedMoEQuantConfig.make(
        "nvfp4",
        w1_scale=torch.ones(4, 256, 8, dtype=torch.float8_e4m3fn),
        w2_scale=torch.ones(4, 128, 8, dtype=torch.float8_e4m3fn),
        g1_alphas=torch.ones(4),
        g2_alphas=torch.ones(4),
        a1_gscale=scales,
        a2_gscale=torch.ones(4),
    )
    experts = B12xExperts(config, quant)
    prepared = experts._prepare_experts(
        w1=torch.empty(4, 256, 64, dtype=torch.uint8),
        w2=torch.empty(4, 128, 64, dtype=torch.uint8),
        activation=MoEActivation.SILU,
        params_dtype=torch.bfloat16,
    )

    assert prepared.a1_gscale is scales
    assert prepared.can_share_input(input_scales_static=True) is uniform
    assert not prepared.can_share_input(input_scales_static=False)
    scales[-1] = 2.0
    assert not prepared.can_share_input(input_scales_static=True)


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


@pytest.mark.parametrize(
    "tokens,topk,num_experts",
    ((1, 6, 384), (4, 6, 384), (6, 6, 384), (8, 6, 384), (64, 8, 288)),
)
def test_b12x_moe_candidate_calls_share_bounded_trial_storage(
    tokens: int, topk: int, num_experts: int
) -> None:
    """Repeated candidate calls reuse one activation/output tensor set (a
    weakref cache), while each call's scratch is a fresh, correctly shaped
    trial-only allocation rather than a caller-owned workspace region."""

    class FakeState:
        def __init__(self):
            self.scratch = SimpleNamespace(
                scratch_specs=lambda: (
                    SimpleNamespace(
                        shape=(32,), dtype=torch.uint8, device=torch.device("cpu")
                    ),
                )
            )
            self.bound = None

        def bind(self, **kwargs):
            self.bound = kwargs
            return SimpleNamespace(run=lambda: None)

    prepared = SimpleNamespace(
        device=torch.device("cpu"),
        hidden_size=16,
        num_experts=num_experts,
        plan=SimpleNamespace(activation=SimpleNamespace(io_dtype=torch.bfloat16)),
    )
    factory = b12x._prepared_moe_call_factory(
        tokens=tokens,
        topk=topk,
        prepared=prepared,
        output_dtype=torch.bfloat16,
    )
    first_state = FakeState()
    second_state = FakeState()

    first_call = factory(first_state)
    second_call = factory(second_state)
    assert first_state.bound is not None
    assert second_state.bound is not None

    for name in ("a", "output", "topk_ids", "topk_weights"):
        assert first_state.bound[name] is second_state.bound[name]
    for scratch in (first_state.bound["scratch"][0], second_state.bound["scratch"][0]):
        assert scratch.shape == (32,)
        assert scratch.dtype == torch.uint8
        assert scratch.device == torch.device("cpu")
    assert first_state.bound["scratch"][0] is not second_state.bound["scratch"][0]
    assert not first_call.capture_safe
    assert not second_call.capture_safe
    first_call.restore()
    ids = first_state.bound["topk_ids"]
    assert all(row.unique().numel() == topk for row in ids)
    assert torch.isfinite(first_state.bound["a"]).all()
    torch.testing.assert_close(
        first_state.bound["topk_weights"].sum(dim=1), torch.ones(tokens)
    )
    for first_producer, second_producer in zip(
        first_call.benchmark_producers,
        second_call.benchmark_producers,
        strict=True,
    ):
        first_producer()
        first_ids = ids.clone()
        assert all(row.unique().numel() == topk for row in ids)
        second_producer()
        torch.testing.assert_close(ids, first_ids)

    # Published plans retain call.owners after discarding their priming calls.
    published_owners = first_call.owners + second_call.owners
    trial_refs = [
        weakref.ref(tensor)
        for bound in (first_state.bound, second_state.bound)
        for tensor in (
            bound["a"],
            bound["output"],
            bound["topk_ids"],
            bound["topk_weights"],
            *bound["scratch"],
        )
    ]
    first_state.bound = second_state.bound = None
    del first_call, second_call, first_producer, second_producer, ids, scratch
    gc.collect()
    assert all(ref() is None for ref in trial_refs)
    assert published_owners == ()


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
    num_experts: int = 4,
    hidden_size: int = 512,
    intermediate_size: int = 128,
    topk: int = 2,
) -> _B12xMoeCase:
    """Build one deterministic B12X MoE case for a weight/activation pair.

    Quantizes freshly seeded BF16 expert weights, keeps the dequantized
    tensors as the reference, and assembles the quant config matching the
    requested activation dtype.
    """
    if weight_dtype == "mxfp8" and not b12x._b12x_has_mxfp8_moe():
        pytest.skip("the installed b12x has no MXFP8 W8A8 MoE recipe")
    set_random_seed(seed)
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
    elif weight_dtype == "mxfp8":
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
            _mxfp8_e4m3_quantize_torch,
            dequant_mxfp8_to_bf16,
        )

        # Serialized ModelOpt layout: raw E4M3 values plus unswizzled uint8
        # UE8M0 K/32 scale grids, one byte per value (no FP4 packing).
        w1_q, w1_scale = _mxfp8_e4m3_quantize_torch(w1, is_sf_swizzled_layout=False)
        w2_q, w2_scale = _mxfp8_e4m3_quantize_torch(w2, is_sf_swizzled_layout=False)
        w1_ref = dequant_mxfp8_to_bf16(w1_q, w1_scale)
        w2_ref = dequant_mxfp8_to_bf16(w2_q, w2_scale)
        assert w1_scale.dtype is torch.uint8 and w2_scale.dtype is torch.uint8
        quant_config = FusedMoEQuantConfig.make(
            quant_dtype="mxfp8",
            weight_dtype="mxfp8",
            block_shape=[1, 32],
            is_scale_swizzled=False,
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
        topk=topk,
    )


@pytest.mark.skipif(not _has_b12x_moe(), reason="requires b12x MoE on SM120")
@pytest.mark.parametrize(
    "weight_dtype,activation_dtype,activation",
    [
        pytest.param("mxfp4", "mxfp8", MoEActivation.SILU, id="mxfp4-mxfp8"),
        pytest.param("mxfp4", None, MoEActivation.SILU, id="mxfp4-bf16"),
        pytest.param(
            "mxfp8",
            "mxfp8",
            MoEActivation.SILU,
            id="mxfp8-w8a8",
        ),
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
    """Each b12x MoE quantization lane matches the dequantized PyTorch MoE."""
    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        case = _make_b12x_moe_case(
            weight_dtype,
            activation_dtype,
            activation=activation,
            # Reduced step5500 geometry (E16 K2560 I640 topk10, M<=16) keeps
            # the MXFP8 lane inside the small-GPU validation memory budget.
            **(
                dict(
                    num_experts=16,
                    hidden_size=2560,
                    intermediate_size=640,
                    topk=10,
                    tokens=16,
                )
                if weight_dtype == "mxfp8"
                else {}
            ),
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
        pytest.param("mxfp8", "mxfp8", id="w8a8-mx"),
        pytest.param("nvfp4", None, id="w4a16"),
    ],
)
@torch.inference_mode()
def test_b12x_moe_cuda_graph_replay(
    weight_dtype: str,
    activation_dtype: str | None,
    workspace_init,
) -> None:
    """A captured b12x MoE graph replays with the eager call's output."""
    from vllm.v1.worker.workspace import lock_workspace

    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        case = _make_b12x_moe_case(
            weight_dtype,
            activation_dtype,
            tokens=4,
            seed=23,
            # Reduced step5500 geometry for the MXFP8 lane (see matches_torch).
            **(
                dict(
                    num_experts=16,
                    hidden_size=2560,
                    intermediate_size=640,
                    topk=10,
                )
                if weight_dtype == "mxfp8"
                else {}
            ),
        )
        kernel, session, _ = _make_b12x_moe_kernel(
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
        try:
            with session.capture(), torch.cuda.graph(graph, stream=stream):
                actual = apply()
            graph.replay()
            torch.accelerator.synchronize()
        finally:
            graph.reset()
            session.close()

    assert torch.isfinite(expected).all()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


# MXFP8 runs the full step5500-scale geometry, so its rows stop at 16 tokens
# to stay under the 1 GiB test allocator cap; other lanes keep 128.
_TUNING_CASES = [
    (w, a, t)
    for w, a in [("nvfp4", "nvfp4"), ("mxfp4", "mxfp8"), ("nvfp4", None)]
    for t in (4, 128)
] + [("mxfp8", "mxfp8", t) for t in (4, 16)]


@pytest.mark.skipif(not _has_b12x_moe(), reason="requires b12x MoE on SM120")
@pytest.mark.parametrize(
    "weight_dtype,activation_dtype,tokens",
    _TUNING_CASES,
    ids=[f"{t}-{w}-{a}" for w, a, t in _TUNING_CASES],
)
@torch.inference_mode()
def test_b12x_moe_tuning_times_native_candidate_without_capture(
    weight_dtype, activation_dtype, tokens, workspace_init, monkeypatch
) -> None:
    """Tuning uses gated events with live producers and fixed MoE storage."""
    from b12x.preparation._measurement import _prepare_race
    from b12x.preparation.types import require_prepared

    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        case = _make_b12x_moe_case(
            weight_dtype,
            activation_dtype,
            tokens=tokens,
            **(
                dict(
                    num_experts=16,
                    hidden_size=2560,
                    intermediate_size=640,
                    topk=10,
                )
                if weight_dtype == "mxfp8"
                else {}
            ),
        )
        _, session, units = _make_b12x_moe_kernel(
            case.hidden_states,
            case.w1,
            case.w2,
            case.topk,
            case.activation,
            case.quant_config,
        )
        race = None
        try:
            request = units[0].requests[0]
            plan = request.plan
            state = require_prepared(plan, plan.component_id)
            call = request.benchmark_call(state)
            assert not call.capture_safe
            call.restore()
            call.invoke()
            expected = call.output.clone()
            assert torch.isfinite(expected).all() and torch.count_nonzero(expected)
            address = call.output.data_ptr()
            session.freeze()
            # _TimedCall.replay() runs samples * len(producers) kernel passes
            # and cycles benchmark_producers[index % len]; every replay's last
            # pass therefore routes through the final tuning pattern, not the
            # pattern-0 restore/invoke captured above (tokens 4-8 build five
            # route-sharing patterns; other sizes carry one). Anchor the
            # comparison to that exact pattern through the same frozen binding.
            producers = call.benchmark_producers or (call.produce,)
            call.reset()
            producers[-1]()
            call.invoke()
            torch.accelerator.synchronize()
            expected = call.output.clone()

            def forbidden_capture(*args, **kwargs):
                raise AssertionError("autotuning must not capture CUDA graphs")

            run = call.run

            def checked_run():
                allocated = torch.accelerator.memory_stats()["allocation.all.allocated"]
                result = run()
                assert (
                    torch.accelerator.memory_stats()["allocation.all.allocated"]
                    == allocated
                )
                return result

            call.run = checked_run
            monkeypatch.setattr(torch.cuda, "CUDAGraph", forbidden_capture)
            race = _prepare_race(
                [call], device_ordinal=torch.accelerator.current_device_index()
            )
            for _ in range(3):
                call.output.fill_(float("nan"))
                race.timers[0].replay()
                torch.accelerator.synchronize()
                assert call.output.data_ptr() == address
                torch.testing.assert_close(call.output, expected, atol=2e-2, rtol=2e-2)
        finally:
            if race is not None:
                race.close()
            session.close()


@pytest.mark.skipif(not _has_b12x_moe(), reason="requires b12x MoE on SM120")
@pytest.mark.parametrize(
    "weight_dtype,activation_dtype",
    [
        ("nvfp4", "nvfp4"),
        ("mxfp4", "mxfp8"),
        ("mxfp8", "mxfp8"),
        ("nvfp4", None),
    ],
    ids=["nvfp4", "w4a8", "w8a8-mx", "w4a16"],
)
@torch.inference_mode()
@pytest.mark.parametrize("capacity", [4, 128])
def test_b12x_moe_prefill_capacity_and_exact_decode_reuse(
    weight_dtype,
    activation_dtype,
    workspace_init,
    capacity,
) -> None:
    """One prepared kernel serves every row count up to its capacity.

    Once the session is frozen, eager calls at mixed row counts and route-id
    dtypes, plus a capture-scope graph replay, must stay correct without
    reallocating workspace buffers.
    """
    from b12x._lib.runtime_control import kernel_resolution_guard

    from vllm.v1.worker.workspace import current_workspace_manager, lock_workspace

    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        case = _make_b12x_moe_case(
            weight_dtype,
            activation_dtype,
            tokens=capacity,
            **(
                dict(
                    num_experts=16,
                    hidden_size=2560,
                    intermediate_size=640,
                    topk=10,
                )
                if weight_dtype == "mxfp8"
                else {}
            ),
        )
        kernel, session, _ = _make_b12x_moe_kernel(
            case.hidden_states,
            case.w1,
            case.w2,
            case.topk,
            case.activation,
            case.quant_config,
            fixed_token_counts=(4,) if capacity > 4 else (),
        )
        weights, ids, _ = fused_topk(
            case.hidden_states,
            case.score,
            case.topk,
            renormalize=False,
        )
        ids64 = ids.to(torch.int64)
        if activation_dtype == "nvfp4":
            reference = _nvfp4_activation_reference(
                case.hidden_states,
                case.w1_ref,
                case.w2_ref,
                weights,
                ids,
                case.quant_config.a1_gscale,
                case.quant_config.a2_gscale,
            )
        else:
            reference = torch_moe(
                case.hidden_states,
                case.w1_ref,
                case.w2_ref,
                case.score,
                case.topk,
                activation=case.activation,
            )

        def apply(rows, route_ids=ids):
            return kernel.apply(
                hidden_states=case.hidden_states[:rows],
                w1=case.w1,
                w2=case.w2,
                topk_weights=weights[:rows],
                topk_ids=route_ids[:rows],
                activation=case.activation,
                global_num_experts=case.w1.shape[0],
                expert_map=None,
                apply_router_weight_on_input=False,
            )

        def check(output, rows):
            assert torch.isfinite(output).all() and torch.count_nonzero(output)
            torch.testing.assert_close(output, reference[:rows], atol=2e-1, rtol=2e-1)
            cosine = torch.nn.functional.cosine_similarity(
                output.flatten().float(),
                reference[:rows].flatten().float(),
                dim=0,
            )
            assert cosine > 0.99

        # MXFP8 keeps its declared 128-token capacity but applies at most
        # 16 rows per call to stay under the 1 GiB test allocator cap.
        active_rows = min(capacity, 16) if weight_dtype == "mxfp8" else capacity
        try:
            apply(active_rows)
            if capacity == 4 and (weight_dtype, activation_dtype) == ("mxfp4", "mxfp8"):
                # Tiny W4A8 uses static M in mainline; eager first use is legal.
                check(apply(3), 3)
            lock_workspace()
            buffers = tuple(
                (buffer.data_ptr(), buffer.numel())
                for buffer in current_workspace_manager()._current_workspaces
                if buffer is not None
            )
            session.freeze()
            with kernel_resolution_guard("prepared MoE capacity reuse"):
                for rows in (
                    count
                    for count in (
                        (4, 3, 11, 16)
                        if weight_dtype == "mxfp8"
                        else (4, 3, 11, 31, 125, 128)
                    )
                    if count <= capacity
                ):
                    for route_ids in (ids, ids64):
                        check(apply(rows, route_ids), rows)
                graph = torch.cuda.CUDAGraph()
                try:
                    with session.capture(), torch.cuda.graph(graph):
                        actual = apply(4)
                    graph.replay()
                    torch.accelerator.synchronize()
                    check(actual, 4)
                finally:
                    graph.reset()
            assert buffers == tuple(
                (buffer.data_ptr(), buffer.numel())
                for buffer in current_workspace_manager()._current_workspaces
                if buffer is not None
            )
        finally:
            session.close()


_MXFP8_W8A8_ALIGNMENT_REASON = (
    "MXFP8 W8A8 requires hidden size divisible by 128 and "
    "per-rank intermediate size divisible by 32"
)


@pytest.mark.parametrize(
    "config_kwargs,expected_reason",
    [
        pytest.param(
            {"hidden_dim": 2560, "intermediate_size": 640},
            None,
            id="step5500-shape-supported",
        ),
        pytest.param(
            {"hidden_dim": 2560, "intermediate_size": 320},
            None,
            id="step5500-tp2-shard-supported",
        ),
        pytest.param(
            {"hidden_dim": 256, "intermediate_size": 48},
            _MXFP8_W8A8_ALIGNMENT_REASON,
            id="intermediate-48-rejected",
        ),
        pytest.param(
            {"hidden_dim": 100, "intermediate_size": 128},
            _MXFP8_W8A8_ALIGNMENT_REASON,
            id="hidden-misaligned",
        ),
    ],
)
def test_b12x_mxfp8_w8a8_config_support(
    monkeypatch: pytest.MonkeyPatch,
    config_kwargs,
    expected_reason: str | None,
) -> None:
    """MXFP8 W8A8 selects only on b12x-compatible ModelOpt geometries."""
    monkeypatch.setattr(B12xExperts, "_supports_current_device", lambda: True)
    monkeypatch.setattr(b12x, "_b12x_has_mxfp8_moe", lambda: True)
    config = make_dummy_moe_config(
        activation=MoEActivation.SILU,
        **config_kwargs,
    )

    supported, reason = B12xExperts.is_supported_config(
        B12xExperts,
        config,
        kMxfp8Static,
        kMxfp8Dynamic,
        mk.FusedMoEActivationFormat.Standard,
    )

    assert (supported, reason) == (expected_reason is None, expected_reason)


def test_b12x_mxfp8_w8a8_rejects_non_silu_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MXFP8 W8A8 on b12x supports SiLU only."""
    monkeypatch.setattr(B12xExperts, "_supports_current_device", lambda: True)
    monkeypatch.setattr(b12x, "_b12x_has_mxfp8_moe", lambda: True)
    config = make_dummy_moe_config(
        hidden_dim=256,
        intermediate_size=64,
        activation=MoEActivation.GELU_TANH,
    )

    supported, reason = B12xExperts.is_supported_config(
        B12xExperts,
        config,
        kMxfp8Static,
        kMxfp8Dynamic,
        mk.FusedMoEActivationFormat.Standard,
    )

    assert (supported, reason) == (False, "MXFP8 W8A8 supports only SiLU")


def test_b12x_mxfp8_w8a8_requires_the_b12x_recipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An installed b12x without the MXFP8 recipe is rejected at selection,
    not after the model has loaded."""
    monkeypatch.setattr(B12xExperts, "_supports_current_device", lambda: True)
    monkeypatch.setattr(b12x, "_b12x_has_mxfp8_moe", lambda: False)
    config = make_dummy_moe_config(
        hidden_dim=2560,
        intermediate_size=640,
        activation=MoEActivation.SILU,
    )

    supported, reason = B12xExperts.is_supported_config(
        B12xExperts,
        config,
        kMxfp8Static,
        kMxfp8Dynamic,
        mk.FusedMoEActivationFormat.Standard,
    )

    assert (supported, reason) == (
        False,
        "the installed b12x has no MXFP8 W8A8 MoE recipe",
    )


def test_explicit_b12x_mxfp8_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    """moe_backend='b12x' resolves the MXFP8 oracle to the b12x experts."""
    import vllm.model_executor.layers.fused_moe.oracle.mxfp8 as mxfp8_oracle

    monkeypatch.setattr(B12xExperts, "_supports_current_device", lambda: True)
    monkeypatch.setattr(b12x, "_b12x_has_mxfp8_moe", lambda: True)
    config = make_dummy_moe_config(
        hidden_dim=2560, intermediate_size=640, experts_per_token=10
    )
    config.moe_backend = "b12x"

    backend, experts_cls = mxfp8_oracle.select_mxfp8_moe_backend(
        config, prepares_b12x=True
    )

    assert backend is mxfp8_oracle.Fp8MoeBackend.B12X_MXFP8
    assert experts_cls is B12xExperts


def test_explicit_b12x_mxfp8_selection_requires_b12x_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Quantization methods that do not prepare B12X experts (compressed
    tensors, INC, online MXFP8) reject moe_backend='b12x' at init."""
    import vllm.model_executor.layers.fused_moe.oracle.mxfp8 as mxfp8_oracle

    monkeypatch.setattr(B12xExperts, "_supports_current_device", lambda: True)
    monkeypatch.setattr(b12x, "_b12x_has_mxfp8_moe", lambda: True)
    config = make_dummy_moe_config(
        hidden_dim=2560, intermediate_size=640, experts_per_token=10
    )
    config.moe_backend = "b12x"

    with pytest.raises(ValueError, match="does not prepare B12X experts"):
        mxfp8_oracle.select_mxfp8_moe_backend(config)


def test_b12x_mxfp8_auto_selection_keeps_conservative_priority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Automatic selection must not displace an established backend: b12x is
    registered only ahead of the dequantize-to-BF16 emulation."""
    import vllm.model_executor.layers.fused_moe.oracle.mxfp8 as mxfp8_oracle

    backends = mxfp8_oracle._SUPPORTED_BACKENDS
    assert backends[-1] is mxfp8_oracle.Fp8MoeBackend.EMULATION
    assert backends[-2] is mxfp8_oracle.Fp8MoeBackend.B12X_MXFP8

    def always_supported(cls, config, weight_key, activation_key, activation_format):
        """Stand-in: the established backend claims every config."""
        del config, weight_key, activation_key, activation_format
        return True, None

    established = type(
        "EstablishedExperts",
        (mk.FusedMoEExperts,),
        {
            "is_supported_config": staticmethod(always_supported),
        },
    )
    monkeypatch.setattr(
        mxfp8_oracle,
        "_mxfp8_backend_to_kernel_cls",
        lambda backend: (
            [B12xExperts]
            if backend is mxfp8_oracle.Fp8MoeBackend.B12X_MXFP8
            else [established]
        ),
    )
    config = make_dummy_moe_config(hidden_dim=256, intermediate_size=64)

    backend, experts_cls = mxfp8_oracle.select_mxfp8_moe_backend(
        config, prepares_b12x=True
    )

    assert backend is not mxfp8_oracle.Fp8MoeBackend.B12X_MXFP8
    assert experts_cls is established


def test_b12x_mxfp8_auto_selection_when_nothing_else_supports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With every established backend unsupported, b12x wins over emulation,
    but only for quantization methods that prepare B12X experts."""
    import vllm.model_executor.layers.fused_moe.oracle.mxfp8 as mxfp8_oracle

    def supported(cls, config, weight_key, activation_key, activation_format):
        """Stand-in: the b12x experts claim the config."""
        del config, weight_key, activation_key, activation_format
        return True, None

    def unsupported(cls, config, weight_key, activation_key, activation_format):
        """Stand-in: the established backend rejects every config."""
        del config, weight_key, activation_key, activation_format
        return False, "established backend disabled"

    # Disable every established backend explicitly: on SM12x hardware Marlin
    # genuinely supports MXFP8 and outranks b12x.
    established = type(
        "EstablishedExperts",
        (mk.FusedMoEExperts,),
        {"is_supported_config": staticmethod(unsupported)},
    )
    monkeypatch.setattr(B12xExperts, "is_supported_config", staticmethod(supported))
    monkeypatch.setattr(
        mxfp8_oracle,
        "_mxfp8_backend_to_kernel_cls",
        lambda backend: (
            [B12xExperts]
            if backend is mxfp8_oracle.Fp8MoeBackend.B12X_MXFP8
            else [established]
        ),
    )
    config = make_dummy_moe_config(
        hidden_dim=2560, intermediate_size=640, experts_per_token=10
    )

    backend, experts_cls = mxfp8_oracle.select_mxfp8_moe_backend(
        config, prepares_b12x=True
    )

    assert backend is mxfp8_oracle.Fp8MoeBackend.B12X_MXFP8
    assert experts_cls is B12xExperts
    with pytest.raises(ValueError, match="No MXFP8 MoE backends available"):
        mxfp8_oracle.select_mxfp8_moe_backend(config)


def test_b12x_mxfp8_preparation_passes_source_tensors_verbatim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real b12x ``plan_weights``/``prepare_weights`` API receives the
    serialized ModelOpt MXFP8 tensors unmodified through the vLLM adapter:
    the plan declares ``mxfp8_e8m0_k32`` / ``w8a8_mx`` / ``w31``, and the
    prepared owner keeps the E4M3 bytes (w2 verbatim, the gate-first FC1
    with its halves flipped to kernel order) while b12x (not vLLM) performs
    the host scale swizzle."""
    fused_moe = pytest.importorskip("b12x.moe.fused_moe")
    from b12x.moe.fused_moe.source import PackedSourceFormat

    if not hasattr(PackedSourceFormat, "MXFP8_E8M0_K32"):
        pytest.skip("b12x build lacks the mxfp8_e8m0_k32 source format")

    # Only vLLM-side registration surface is stubbed; plan_weights and
    # prepare_weights run for real against the installed b12x.
    monkeypatch.setattr(b12x, "_is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(b12x, "set_b12x_preparation_provider", lambda *_, **__: None)
    monkeypatch.setattr(
        b12x, "_register_b12x_moe_output_collective", lambda *_, **__: None
    )

    num_experts, hidden_size, intermediate_size = 4, 256, 128
    generator = torch.Generator().manual_seed(0)

    def random_e4m3(*shape: int) -> torch.Tensor:
        # Distinct bytes make the FC1 half order observable; 0x7F/0xFF are NaN.
        values = torch.randint(0, 256, shape, generator=generator, dtype=torch.uint8)
        values[(values & 0x7F) == 0x7F] = 0x38
        return values.view(torch.float8_e4m3fn)

    w13 = random_e4m3(num_experts, 2 * intermediate_size, hidden_size)
    w2 = random_e4m3(num_experts, hidden_size, intermediate_size)
    # b12x flips the gate-first FC1 in place, so compare against copies.
    w13_source = w13.view(torch.uint8).clone()
    w2_source = w2.view(torch.uint8).clone()
    w13_scale = torch.full(
        (num_experts, 2 * intermediate_size, hidden_size // 32),
        130,
        dtype=torch.uint8,
    )
    w2_scale = torch.full(
        (num_experts, hidden_size, intermediate_size // 32),
        131,
        dtype=torch.uint8,
    )
    quant_config = FusedMoEQuantConfig.make(
        "mxfp8",
        block_shape=[1, 32],
        is_scale_swizzled=False,
        w1_scale=w13_scale,
        w2_scale=w2_scale,
    )
    experts = B12xExperts(
        make_dummy_moe_config(
            num_experts=num_experts,
            hidden_dim=hidden_size,
            intermediate_size=intermediate_size,
        ),
        quant_config,
    )
    layer = torch.nn.Module()
    layer.register_parameter("w13_weight", torch.nn.Parameter(w13, requires_grad=False))
    layer.register_parameter("w2_weight", torch.nn.Parameter(w2, requires_grad=False))
    layer.register_parameter(
        "w13_weight_scale", torch.nn.Parameter(w13_scale, requires_grad=False)
    )
    layer.register_parameter(
        "w2_weight_scale", torch.nn.Parameter(w2_scale, requires_grad=False)
    )
    layer.activation = MoEActivation.SILU
    layer.apply_router_weight_on_input = False

    try:
        experts.process_weights_after_loading(layer)
    except RuntimeError as exc:  # host packing may require a CUDA device
        if "CUDA" in str(exc):
            pytest.skip(f"b12x mxfp8 host preparation requires CUDA: {exc}")
        raise

    impl = experts._prepared_experts._impl
    source = experts._prepared_experts.plan.source
    assert source.format is PackedSourceFormat.MXFP8_E8M0_K32
    assert source.w13_layout == fused_moe.W13Layout.W31
    assert experts._prepared_experts.plan.activation.mode == "a8"
    assert experts._prepared_experts.plan.activation.nonlinearity == "silu"
    assert experts._prepared_experts.plan.activation.io_dtype is torch.bfloat16
    # MXFP8 stores one byte per value: the declared geometry carries the
    # loaded extents unchanged (no FP4-style doubling of the w2 last dim).
    assert experts._prepared_experts.plan.geometry.intermediate_size == (
        intermediate_size
    )
    assert experts._prepared_experts.plan.geometry.hidden_size == hidden_size

    # Byte-exact E4M3 preservation through the prepared owner (canonical
    # aliases, with the representation's *_values spelling as fallback).
    w13_prepared = getattr(impl, "w13_values", None)
    w13_prepared = w13_prepared if w13_prepared is not None else impl.w1_fp4
    w2_prepared = getattr(impl, "w2_values", None)
    w2_prepared = w2_prepared if w2_prepared is not None else impl.w2_fp4
    up_first = torch.cat(
        [w13_source[:, intermediate_size:], w13_source[:, :intermediate_size]],
        dim=1,
    )
    assert torch.equal(w13_prepared.view(torch.uint8).cpu(), up_first)
    assert torch.equal(w2_prepared.view(torch.uint8).cpu(), w2_source)

    # The b12x plan discarded the source parameters, so the raw tensors are
    # only reachable through the prepared owner.
    assert layer.w13_weight.numel() == 0
    assert layer.w2_weight.numel() == 0


_MXFP8_E = 16
_MXFP8_K = 2560
_MXFP8_TOPK = 10
_MXFP8_TOKENS = 16


def _modelopt_mxfp8_method(intermediate: int):
    """Build the real ModelOptMxFp8FusedMoE with moe_backend='b12x' at
    reduced step5500 geometry, with weights allocated on the layer."""
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptMxFp8Config,
        ModelOptMxFp8FusedMoE,
    )

    quant_config = ModelOptMxFp8Config(
        is_checkpoint_mxfp8_serialized=True,
        kv_cache_quant_algo="NO_QUANT",
        exclude_modules=[],
    )
    moe_config = make_dummy_moe_config(
        num_experts=_MXFP8_E,
        experts_per_token=_MXFP8_TOPK,
        hidden_dim=_MXFP8_K,
        intermediate_size=intermediate,
        in_dtype=torch.bfloat16,
        max_num_tokens=128,
    )
    moe_config.moe_backend = "b12x"
    method = ModelOptMxFp8FusedMoE(quant_config=quant_config, moe_config=moe_config)
    assert method.mxfp8_backend is Fp8MoeBackend.B12X_MXFP8
    assert method.experts_cls is B12xExperts

    layer = torch.nn.Module()
    layer.intermediate_size_per_partition = intermediate
    layer.hidden_size = _MXFP8_K
    layer.moe_config = moe_config
    layer.activation = MoEActivation.SILU
    layer.apply_router_weight_on_input = False
    layer.num_expert_group = None
    layer.topk_group = None
    layer.e_score_correction_bias = None
    layer.routed_scaling_factor = None
    layer.global_num_experts = _MXFP8_E
    layer.expert_map = None
    layer._expert_routing_tables = lambda: None
    # create_weights registers ModelWeightParameter members; torch.device
    # makes torch.empty allocate them on CUDA like a real worker would.
    with torch.device("cuda"):
        method.create_weights(
            layer,
            num_experts=_MXFP8_E,
            hidden_size=_MXFP8_K,
            intermediate_size_per_partition=intermediate,
            params_dtype=torch.bfloat16,
        )
    return method, layer


@pytest.mark.skipif(
    not _has_b12x_mxfp8_moe(), reason="requires b12x MXFP8 MoE on SM120"
)
@torch.inference_mode()
# 640 is the TP=1 expert width; 320 is the TP=2 shard, which B12X prepares
# unpadded (split up/gate FC1 descriptors, TMA zero-filled tail tile).
@pytest.mark.parametrize("intermediate", [640, 320], ids=["tp1-i640", "tp2-i320"])
def test_b12x_modelopt_mxfp8_post_load_prepares_and_matches_torch(
    dist_init,
    workspace_init,
    intermediate: int,
) -> None:
    """Drive the real ModelOptMxFp8FusedMoE lifecycle (create_weights ->
    serialized weights -> process_weights_after_loading -> declare+prepare ->
    apply) with the b12x w8a8_mx backend at reduced step5500 geometry,
    against the dequantized torch reference. This exercises the full
    vLLM -> B12X integration through the quant method, not B12xExperts in
    isolation."""
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        _mxfp8_e4m3_quantize_torch,
        dequant_mxfp8_to_bf16,
    )

    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        monkeypatch = pytest.MonkeyPatch()
        try:
            # The TP collective descriptor needs live parallel state a unit
            # test does not build; plan preparation does not depend on it.
            monkeypatch.setattr(
                "vllm.model_executor.layers.fused_moe.b12x."
                "_register_b12x_moe_output_collective",
                lambda *_, **__: None,
            )
            with torch.device("cuda"):
                method, layer = _modelopt_mxfp8_method(intermediate)
                set_random_seed(29)
                hidden_states = (
                    torch.randn(
                        (_MXFP8_TOKENS, _MXFP8_K), device="cuda", dtype=torch.bfloat16
                    )
                    / 10
                )
                score = torch.randn(
                    (_MXFP8_TOKENS, _MXFP8_E), device="cuda", dtype=torch.bfloat16
                )
                w1_bf16 = (
                    torch.randn(
                        (_MXFP8_E, 2 * intermediate, _MXFP8_K),
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                    / 15
                )
                w2_bf16 = (
                    torch.randn(
                        (_MXFP8_E, _MXFP8_K, intermediate),
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                    / 15
                )
                w13_q, w13_scale = _mxfp8_e4m3_quantize_torch(
                    w1_bf16, is_sf_swizzled_layout=False
                )
                w2_q, w2_scale = _mxfp8_e4m3_quantize_torch(
                    w2_bf16, is_sf_swizzled_layout=False
                )
                layer.w13_weight.data.copy_(w13_q)
                layer.w2_weight.data.copy_(w2_q)
                layer.w13_weight_scale.data.copy_(w13_scale)
                layer.w2_weight_scale.data.copy_(w2_scale)

                method.process_weights_after_loading(layer)

            experts = method.moe_kernel.fused_experts
            assert isinstance(experts, B12xExperts)
            assert experts._prepared_experts is not None
            # Preparation consumed the source parameters.
            assert layer.w13_weight.numel() == 0
            assert layer.w2_weight.numel() == 0
            assert (
                experts._prepared_experts.plan.geometry.intermediate_size
                == intermediate
            )

            # Declare and prepare the serving shapes exactly as the warmup
            # driver does; apply() resolves only session-prepared plans.
            session, _ = _prepare(
                layer,
                device=torch.device("cuda"),
                counts=(_MXFP8_TOKENS,),
                output_dtype=torch.bfloat16,
                max_tokens=128,
            )
            try:
                topk_weights, topk_ids, _ = fused_topk(
                    hidden_states, score, _MXFP8_TOPK, renormalize=False
                )
                output = method.apply(
                    layer,
                    hidden_states,
                    topk_weights,
                    topk_ids,
                    shared_experts=None,
                    shared_experts_input=None,
                )
            finally:
                session.close()
        finally:
            monkeypatch.undo()

    reference = torch_moe(
        hidden_states,
        dequant_mxfp8_to_bf16(w13_q, w13_scale),
        dequant_mxfp8_to_bf16(w2_q, w2_scale),
        score,
        _MXFP8_TOPK,
        activation=MoEActivation.SILU,
    )
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, reference, atol=2e-1, rtol=2e-1)
    cosine = torch.nn.functional.cosine_similarity(
        output.flatten().float(), reference.flatten().float(), dim=0
    )
    assert cosine > 0.99
