# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass

import pytest
import torch

from tests.utils import get_open_port, init_test_distributed_environment
from vllm.model_executor.kernels.linear import (
    _LINEAR_BACKEND_KERNEL_MAP,
    _POSSIBLE_FP8_BLOCK_KERNELS,
    _POSSIBLE_FP8_KERNELS,
    _POSSIBLE_MXFP4_KERNELS,
    _POSSIBLE_MXFP8_KERNELS,
    _POSSIBLE_NVFP4_KERNELS,
    B12xFp8BlockScaledMMKernel,
    B12xMxFp4LinearKernel,
    B12xMxfp8LinearKernel,
    B12xNvFp4LinearKernel,
    B12xTensorFP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
    Mxfp8LinearLayerConfig,
    init_fp8_linear_kernel,
    init_mxfp4_linear_kernel,
    init_mxfp8_linear_kernel,
    init_nvfp4_linear_kernel,
)
from vllm.model_executor.kernels.linear.nvfp4.base import NvFp4LinearLayerConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
    kFp8StaticTensorSym,
    kMxfp4Dynamic,
)
from vllm.platforms import PlatformEnum
from vllm.utils.b12x import B12xWorkload, register_b12x_layer
from vllm.utils.torch_utils import _encode_layer_name


def _prepare(layer, *, device, counts, fixed=(), output_dtype=torch.bfloat16,
             max_tokens=None, autotune=False, stage="weights"):
    """Collect a layer's preparation units and fill their plans in place."""
    from b12x.preparation import PreparationSession

    max_tokens = max_tokens or max(counts)
    workload = B12xWorkload(
        stage=stage, token_counts=tuple(sorted(set(counts))),
        fixed_token_counts=tuple(sorted(set(fixed))), output_dtype=output_dtype,
        max_tokens=max_tokens, max_seqs=1, max_model_len=max_tokens,
    )
    provider = layer.b12x_preparation_provider
    units = list(provider.get_b12x_preparation_units(layer, workload))
    requests = tuple(request for unit in units for request in unit.requests)
    session = PreparationSession(device=device, autotune=autotune)
    session.prepare(requests, autotune=autotune)
    return session, units


class _WeakNamespace(types.SimpleNamespace):
    """A SimpleNamespace stand-in for a layer/owner: register_b12x_layer keys
    its registry by weakref, and plain SimpleNamespace does not support one."""

    __slots__ = ("__weakref__",)


@pytest.mark.parametrize(
    ("kernel_cls", "kernels", "before", "after", "initializer", "kwargs"),
    [
        (
            B12xMxFp4LinearKernel,
            _POSSIBLE_MXFP4_KERNELS[PlatformEnum.CUDA],
            "HummingMxFp4LinearKernel",
            "EmulationMxfp4LinearKernel",
            init_mxfp4_linear_kernel,
            {"activation_quant_key": kMxfp4Dynamic},
        ),
        (
            B12xNvFp4LinearKernel,
            _POSSIBLE_NVFP4_KERNELS[PlatformEnum.CUDA],
            "FbgemmNvFp4LinearKernel",
            "EmulationNvFp4LinearKernel",
            init_nvfp4_linear_kernel,
            {},
        ),
        (
            B12xMxfp8LinearKernel,
            _POSSIBLE_MXFP8_KERNELS[PlatformEnum.CUDA],
            "MarlinMxfp8LinearKernel",
            "EmulationMxfp8LinearKernel",
            init_mxfp8_linear_kernel,
            {},
        ),
        (
            B12xTensorFP8ScaledMMLinearKernel,
            _POSSIBLE_FP8_KERNELS[PlatformEnum.CUDA],
            "CutlassFP8ScaledMMLinearKernel",
            "PerTensorTorchFP8ScaledMMLinearKernel",
            init_fp8_linear_kernel,
            {
                "activation_quant_key": kFp8StaticTensorSym,
                "weight_quant_key": kFp8StaticTensorSym,
                "input_dtype": torch.bfloat16,
                "out_dtype": torch.bfloat16,
                "weight_shape": (2048, 2048),
            },
        ),
        (
            B12xFp8BlockScaledMMKernel,
            _POSSIBLE_FP8_BLOCK_KERNELS[PlatformEnum.CUDA],
            "CutlassFp8BlockScaledMMKernel",
            "MarlinFP8ScaledMMLinearKernel",
            init_fp8_linear_kernel,
            {
                "activation_quant_key": kFp8Dynamic128Sym,
                "weight_quant_key": kFp8Static128BlockSym,
                "input_dtype": torch.bfloat16,
                "out_dtype": torch.bfloat16,
                "weight_shape": (2048, 2048),
            },
        ),
    ],
)
def test_b12x_backend_registration_priority_and_selection(
    monkeypatch,
    default_vllm_config,
    kernel_cls,
    kernels,
    before: str,
    after: str,
    initializer,
    kwargs: dict,
) -> None:
    import vllm.model_executor.kernels.linear as linear_mod

    assert kernel_cls in _LINEAR_BACKEND_KERNEL_MAP["b12x"]
    names = [kernel.__name__ for kernel in kernels]
    assert names.index(before) < names.index(kernel_cls.__name__) < names.index(after)

    monkeypatch.setattr(linear_mod.current_platform, "_enum", PlatformEnum.CUDA)
    monkeypatch.setattr(linear_mod, "_get_linear_backend", lambda: "b12x")
    monkeypatch.setattr(
        kernel_cls,
        "is_supported",
        classmethod(lambda cls, compute_capability=None: (True, None)),
    )
    monkeypatch.setattr(
        kernel_cls,
        "can_implement",
        classmethod(lambda cls, config: (True, None)),
    )

    assert isinstance(initializer(**kwargs), kernel_cls)


def test_b12x_module_lookup_is_dynamo_safe(monkeypatch) -> None:
    import vllm.utils.b12x as b12x_utils

    module = types.ModuleType("b12x.gemm.blockscaled")
    module.run = lambda x: x + 1  # type: ignore[attr-defined]
    monkeypatch.setitem(
        b12x_utils._B12X_SUBMODULES,
        "b12x.gemm.blockscaled",
        module,
    )

    @torch.compile(backend="eager", fullgraph=True)
    def forward(x: torch.Tensor) -> torch.Tensor:
        blockscaled = b12x_utils.get_b12x_blockscaled()
        assert blockscaled is not None
        return blockscaled.run(x)  # type: ignore[attr-defined]

    x = torch.ones(1)
    torch.testing.assert_close(forward(x), x + 1)


def test_b12x_tensor_fp8_can_implement_supported_config() -> None:
    config = FP8ScaledMMLinearLayerConfig(
        activation_quant_key=kFp8StaticTensorSym,
        weight_quant_key=kFp8StaticTensorSym,
        weight_shape=(64, 128),
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
    )

    can_implement, reason = B12xTensorFP8ScaledMMLinearKernel.can_implement(config)

    assert can_implement
    assert reason is None


def test_b12x_block_fp8_checks_runtime_support(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear.scaled_mm.b12x as b12x_mod

    platform = types.SimpleNamespace(
        is_cuda=lambda: True,
        is_device_capability_family=lambda family: family == 120,
    )
    monkeypatch.setattr(b12x_mod, "current_platform", platform)

    monkeypatch.setattr(
        b12x_mod,
        "_import_b12x_blockscaled",
        lambda: types.SimpleNamespace(is_supported=lambda: False),
    )

    supported, reason = B12xFp8BlockScaledMMKernel.is_supported()

    assert not supported
    assert reason == "B12X regular block-FP8 GEMM is not supported"


def test_b12x_block_fp8_requires_matching_supported_dtypes() -> None:
    def config(input_dtype: torch.dtype, out_dtype: torch.dtype):
        return FP8ScaledMMLinearLayerConfig(
            activation_quant_key=kFp8Dynamic128Sym,
            weight_quant_key=kFp8Static128BlockSym,
            weight_shape=(256, 128),
            input_dtype=input_dtype,
            out_dtype=out_dtype,
        )

    can_implement, reason = B12xFp8BlockScaledMMKernel.can_implement(
        config(torch.float32, torch.float32)
    )
    assert not can_implement
    assert reason == "Supports only bf16/fp16 input dtype"

    can_implement, reason = B12xFp8BlockScaledMMKernel.can_implement(
        config(torch.bfloat16, torch.float16)
    )
    assert not can_implement
    assert reason == "Input and output dtype must match"

    can_implement, reason = B12xFp8BlockScaledMMKernel.can_implement(
        config(torch.float16, torch.float16)
    )
    assert can_implement
    assert reason is None


def test_b12x_block_fp8_requires_aligned_features() -> None:
    def can_implement(weight_shape: tuple[int, int]):
        config = FP8ScaledMMLinearLayerConfig(
            activation_quant_key=kFp8Dynamic128Sym,
            weight_quant_key=kFp8Static128BlockSym,
            weight_shape=weight_shape,
            input_dtype=torch.bfloat16,
            out_dtype=torch.bfloat16,
        )
        return B12xFp8BlockScaledMMKernel.can_implement(config)

    assert can_implement((256, 192)) == (
        False,
        "Input features must be a positive multiple of 128",
    )
    assert can_implement((192, 256)) == (
        False,
        "Output features must be a positive multiple of 128",
    )


def test_b12x_tensor_fp8_process_weights_packs_modelopt_layout(
    monkeypatch,
) -> None:
    import vllm.model_executor.kernels.linear.scaled_mm.b12x as b12x_mod

    calls = []
    packed = types.SimpleNamespace(out_features=64)

    def pack(weight: torch.Tensor, output_scale: torch.Tensor):
        calls.append((weight, output_scale))
        return packed

    monkeypatch.setattr(
        b12x_mod,
        "_import_b12x_tensor_fp8",
        lambda: types.SimpleNamespace(pack_weight=pack),
    )
    layer = torch.nn.Module()
    layer.prefix = "model.layers.0.self_attn.qkv_proj"
    original_weight = (
        torch.randn((128, 64), dtype=torch.float32).clamp(-4, 4).to(torch.float8_e4m3fn)
    )
    layer.weight = torch.nn.Parameter(original_weight, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(torch.tensor(0.25), requires_grad=False)
    layer.input_scale = torch.nn.Parameter(torch.tensor(0.5), requires_grad=False)
    weight_loader = object()
    scale_loader = object()
    layer.weight.weight_loader = weight_loader
    layer.weight_scale.weight_loader = scale_loader
    kernel = object.__new__(B12xTensorFP8ScaledMMLinearKernel)
    kernel.config = types.SimpleNamespace(weight_shape=(64, 128))
    kernel.layer_param_names = (
        "weight",
        "weight_scale",
        "input_scale",
        "input_scale_ub",
    )

    kernel.process_weights_after_loading(layer)

    assert layer.b12x_tensor_fp8_packed_weight is packed
    assert len(calls) == 1
    weight, output_scale = calls[0]
    torch.testing.assert_close(weight, original_weight.T.contiguous())
    torch.testing.assert_close(output_scale, torch.tensor([0.125]))
    assert layer.weight.numel() == 0
    assert layer.weight_scale.numel() == 0
    assert layer.weight.weight_loader is weight_loader
    assert layer.weight_scale.weight_loader is scale_loader
    torch.testing.assert_close(layer.input_scale, torch.tensor(0.5))


def test_b12x_tensor_fp8_apply_quantizes_and_uses_prepared_plan(
    monkeypatch,
) -> None:
    import vllm.model_executor.kernels.linear.scaled_mm.b12x as b12x_mod

    # Bypass the CUDA-only op dispatch key (see the mxfp8 apply test above)
    # and run the real op body directly.
    monkeypatch.setattr(
        b12x_mod, "run_b12x_tensor_fp8_linear", b12x_mod._b12x_tensor_fp8_linear
    )

    calls = []

    def mm(
        source: torch.Tensor,
        packed_weight,
        *,
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype,
        plan: object,
    ) -> torch.Tensor:
        calls.append((source, packed_weight, bias, out_dtype, plan))
        return torch.full(
            (source.shape[0], packed_weight.out_features),
            3.0,
            dtype=out_dtype,
        )

    monkeypatch.setattr(
        b12x_mod,
        "_import_b12x_tensor_fp8",
        lambda: types.SimpleNamespace(mm=mm),
    )

    layer = torch.nn.Module()
    packed = types.SimpleNamespace(out_features=48)
    layer.b12x_tensor_fp8_packed_weight = packed
    layer.weight = torch.nn.Parameter(
        torch.empty((128, 48), dtype=torch.float8_e4m3fn),
        requires_grad=False,
    )
    layer.weight_scale = torch.nn.Parameter(torch.tensor(0.25), requires_grad=False)
    layer.input_scale = torch.nn.Parameter(torch.tensor(0.5), requires_grad=False)
    plan = object()
    layer.b12x_tensor_fp8_plans = {6: plan}
    name = "tensor-fp8-apply-probe"
    layer.b12x_layer_name = _encode_layer_name(name)
    register_b12x_layer(name, layer)
    x = torch.empty((2, 3, 128), dtype=torch.bfloat16)
    x_q = torch.empty((6, 128), dtype=torch.float8_e4m3fn)
    bias = torch.empty((48,), dtype=torch.bfloat16)
    kernel = object.__new__(B12xTensorFP8ScaledMMLinearKernel)
    kernel.config = types.SimpleNamespace(out_dtype=torch.bfloat16)
    kernel.layer_param_names = (
        "weight",
        "weight_scale",
        "input_scale",
        "input_scale_ub",
    )
    kernel.quant_fp8 = lambda source, scale, scale_ub: (x_q, scale)

    output = kernel.apply_weights(layer, x, bias)

    assert output.shape == (2, 3, 48)
    assert output.dtype == torch.bfloat16
    assert len(calls) == 1
    source, called_packed, called_bias, out_dtype, called_plan = calls[0]
    assert source.data_ptr() == x_q.data_ptr()
    assert called_packed is packed
    assert called_bias is bias
    assert out_dtype == torch.bfloat16
    assert called_plan is plan


def test_b12x_mxfp8_can_implement_supported_config() -> None:
    can_implement, reason = B12xMxfp8LinearKernel.can_implement(
        Mxfp8LinearLayerConfig()
    )

    assert can_implement
    assert reason is None


def test_b12x_mxfp8_support_check_reports_missing_import(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear.mxfp8.b12x as b12x_mod

    monkeypatch.setattr(b12x_mod.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        b12x_mod.current_platform,
        "is_device_capability_family",
        lambda family: family == 120,
    )
    monkeypatch.setattr(b12x_mod, "_import_b12x_blockscaled", lambda: None)

    is_supported, reason = B12xMxfp8LinearKernel.is_supported()

    assert not is_supported
    assert reason == "Install the B12X backend with `pip install vllm[b12x]`"


def test_b12x_mxfp8_support_respects_runtime_probe(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear.mxfp8.b12x as b12x_mod

    monkeypatch.setattr(b12x_mod.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        b12x_mod.current_platform,
        "is_device_capability_family",
        lambda family: family == 120,
    )
    monkeypatch.setattr(
        b12x_mod,
        "_import_b12x_blockscaled",
        lambda: types.SimpleNamespace(is_supported=lambda: False),
    )

    is_supported, reason = B12xMxfp8LinearKernel.is_supported()

    assert not is_supported
    assert reason == "b12x.gemm.blockscaled is not supported"


def test_b12x_mxfp8_process_weights_packs_modelopt_layout(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear.mxfp8.b12x as b12x_mod

    calls = []
    packed = types.SimpleNamespace(out_features=48)

    def pack(weight: torch.Tensor, weight_scale: torch.Tensor):
        calls.append((weight, weight_scale))
        return packed

    monkeypatch.setattr(
        b12x_mod,
        "_import_b12x_blockscaled",
        lambda: types.SimpleNamespace(pack_weight=pack),
    )

    layer = torch.nn.Module()
    layer.prefix = "model.layers.0.self_attn.qkv_proj"
    layer.weight = torch.nn.Parameter(
        torch.empty((48, 128), dtype=torch.float8_e4m3fn),
        requires_grad=False,
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.empty((64, 8), dtype=torch.uint8),
        requires_grad=False,
    )
    weight_loader = object()
    scale_loader = object()
    layer.weight.weight_loader = weight_loader
    layer.weight_scale.weight_loader = scale_loader
    kernel = object.__new__(B12xMxfp8LinearKernel)

    kernel.process_weights_after_loading(layer)

    assert layer.b12x_mxfp8_packed_weight is packed
    assert len(calls) == 1
    weight, weight_scale = calls[0]
    assert weight.shape == (48, 128)
    assert weight_scale.shape == (48, 4)
    assert weight.dtype == torch.float8_e4m3fn
    assert weight_scale.dtype == torch.uint8
    assert layer.weight.numel() == 0
    assert layer.weight_scale.numel() == 0
    assert layer.weight.weight_loader is weight_loader
    assert layer.weight_scale.weight_loader is scale_loader


def test_b12x_mxfp8_reload_reuses_packed_tensor_addresses(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear.mxfp8.b12x as b12x_mod

    @dataclass(frozen=True)
    class Rows:
        values: torch.Tensor
        scale_mma: torch.Tensor

    @dataclass(frozen=True)
    class PackedWeight:
        weight: Rows
        in_features: int
        padded_in_features: int
        out_features: int

    def pack(weight: torch.Tensor, weight_scale: torch.Tensor) -> PackedWeight:
        return PackedWeight(
            weight=Rows(values=weight.clone(), scale_mma=weight_scale.clone()),
            in_features=int(weight.shape[1]),
            padded_in_features=int(weight.shape[1]),
            out_features=int(weight.shape[0]),
        )

    monkeypatch.setattr(
        b12x_mod,
        "_import_b12x_blockscaled",
        lambda: types.SimpleNamespace(pack_weight=pack),
    )
    layer = torch.nn.Module()
    layer.prefix = "model.layers.0.mlp.down_proj"
    layer.weight = torch.nn.Parameter(
        torch.zeros((48, 128), dtype=torch.float8_e4m3fn),
        requires_grad=False,
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.zeros((48, 4), dtype=torch.uint8),
        requires_grad=False,
    )
    kernel = object.__new__(B12xMxfp8LinearKernel)

    kernel.process_weights_after_loading(layer)
    packed = layer.b12x_mxfp8_packed_weight
    holder = layer.b12x_linear
    values_ptr = packed.weight.values.data_ptr()
    scales_ptr = packed.weight.scale_mma.data_ptr()

    layer.weight = torch.nn.Parameter(
        torch.ones((48, 128), dtype=torch.float8_e4m3fn),
        requires_grad=False,
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.full((48, 4), 3, dtype=torch.uint8),
        requires_grad=False,
    )
    kernel.process_weights_after_loading(layer)

    assert layer.b12x_mxfp8_packed_weight is packed
    # The holder, and with it the declared plan, survives a reload into the
    # same packed storage.
    assert layer.b12x_linear is holder
    assert packed.weight.values.data_ptr() == values_ptr
    assert packed.weight.scale_mma.data_ptr() == scales_ptr
    torch.testing.assert_close(
        packed.weight.values,
        torch.ones((48, 128), dtype=torch.float8_e4m3fn),
    )
    torch.testing.assert_close(
        packed.weight.scale_mma,
        torch.full((48, 4), 3, dtype=torch.uint8),
    )
    assert layer.weight.numel() == 0
    assert layer.weight_scale.numel() == 0


def test_b12x_blockscaled_call_factory_shares_source_storage_across_calls() -> None:
    """The holder's candidate-call factory reuses one source tensor per row
    count while an earlier call keeps it referenced (a weakref cache)."""
    from vllm.model_executor.kernels.linear.b12x_blockscaled import (
        B12xBlockscaledLinear,
    )

    packed = types.SimpleNamespace(
        in_features=8,
        padded_in_features=8,
        out_features=4,
        weight=types.SimpleNamespace(
            values=torch.empty((4, 8)),
            scale_mma=torch.empty((4, 1)),
        ),
    )
    holder = B12xBlockscaledLinear(
        packed, recipe="mxfp8", activation_mode="auto", layer_name="layer"
    )
    factory = holder._call_factory(2)
    sources = []

    def state():
        return types.SimpleNamespace(
            required_workspace=32,
            run=lambda source, *_args, **_kwargs: sources.append(source),
        )

    first = factory(state())
    second = factory(state())
    first.produce()
    second.produce()

    assert first.run() is None
    assert second.run() is None
    assert sources[0] is sources[1]
    assert not first.capture_safe
    assert not second.capture_safe


@pytest.mark.parametrize("workspace_nbytes", [None, 123_456_789])
def test_b12x_nvfp4_declares_caller_workspace_capacity(
    monkeypatch,
    workspace_nbytes,
) -> None:
    import vllm.model_executor.kernels.linear.b12x_blockscaled as blockscaled_mod
    from vllm.model_executor.kernels.linear.b12x_blockscaled import (
        B12xBlockscaledLinear,
    )

    queries = []

    class Declaration:
        token_counts = (65_536,)

        def request(self, **kwargs):
            return types.SimpleNamespace(name=kwargs["name"])

    api = types.SimpleNamespace(
        BlockscaledQuery=lambda **kwargs: kwargs,
        plan_regimes=lambda query, **_kwargs: queries.append(query) or Declaration(),
    )
    monkeypatch.setattr(blockscaled_mod, "get_b12x_blockscaled", lambda: api)
    if workspace_nbytes is None:
        monkeypatch.delenv(
            "VLLM_B12X_BLOCKSCALED_WORKSPACE_MAX_BYTES", raising=False
        )
        expected = 2_000_000_000
    else:
        monkeypatch.setenv(
            "VLLM_B12X_BLOCKSCALED_WORKSPACE_MAX_BYTES", str(workspace_nbytes)
        )
        expected = workspace_nbytes
    packed = types.SimpleNamespace(
        in_features=4_304,
        padded_in_features=4_320,
        out_features=3_456,
        global_scale_kind="multiplier",
        values=torch.empty(0),
        scale_mma=torch.empty(0),
        global_scale=torch.tensor(1.0),
    )
    holder = B12xBlockscaledLinear(
        packed, recipe="nvfp4", activation_mode="a16", layer_name="layer"
    )
    workload = B12xWorkload(
        stage="weights", token_counts=(65_536,), fixed_token_counts=(),
        output_dtype=torch.bfloat16, max_tokens=65_536, max_seqs=1, max_model_len=65_536,
    )

    holder.unit(workload, name="x")

    assert queries == [
        {
            "recipe": "nvfp4",
            "num_tokens": 65_536,
            "in_features": 4_304,
            "padded_in_features": 4_320,
            "out_features": 3_456,
            "activation_mode": "a16",
            "activation_scale_available": False,
            "global_scale_kind": "multiplier",
            "source_contiguous": True,
            "source_aligned": True,
            "workspace_form": "provided",
            "workspace_nbytes": expected,
            "expected_m": None,
        }
    ]


@pytest.fixture
def _mock_b12x_cuda_fp8_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    import vllm.model_executor.layers.quantization.utils.fp8_utils as fp8_utils

    monkeypatch.setattr(
        fp8_utils,
        "current_platform",
        types.SimpleNamespace(
            is_fp8_fnuz=lambda: False,
            is_rocm=lambda: False,
            fp8_dtype=lambda: torch.float8_e4m3fn,
            is_xpu=lambda: False,
            is_cuda_alike=lambda: True,
        ),
    )


@pytest.mark.usefixtures("_mock_b12x_cuda_fp8_platform")
def test_b12x_block_fp8_process_weights_keeps_native_block_layout() -> None:
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.empty((128, 128), dtype=torch.float8_e4m3fn),
        requires_grad=False,
    )
    layer.weight_scale_inv = torch.nn.Parameter(
        torch.empty((1, 1), dtype=torch.float32),
        requires_grad=False,
    )
    layer.weight_block_size = [128, 128]
    weight_loader = object()
    scale_loader = object()
    layer.weight.weight_loader = weight_loader
    layer.weight_scale_inv.weight_loader = scale_loader
    kernel = object.__new__(B12xFp8BlockScaledMMKernel)

    kernel.process_weights_after_loading(layer)

    assert layer.weight.shape == (128, 128)
    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight_scale_inv.shape == (1, 1)
    assert layer.weight_scale_inv.dtype == torch.float32
    assert layer.weight.weight_loader is weight_loader
    assert layer.weight_scale_inv.weight_loader is scale_loader


@pytest.mark.parametrize("scale_dtype", [torch.float8_e8m0fnu, torch.uint8])
@pytest.mark.usefixtures("_mock_b12x_cuda_fp8_platform")
def test_b12x_block_fp8_upcasts_e8m0_weight_scales(scale_dtype) -> None:
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.empty((128, 128), dtype=torch.float8_e4m3fn),
        requires_grad=False,
    )
    scale_bytes = torch.tensor([[125]], dtype=torch.uint8)
    layer.weight_scale_inv = torch.nn.Parameter(
        scale_bytes.view(scale_dtype),
        requires_grad=False,
    )
    layer.weight_block_size = [128, 128]
    kernel = object.__new__(B12xFp8BlockScaledMMKernel)

    kernel.process_weights_after_loading(layer)

    assert layer.weight_scale_inv.dtype == torch.float32
    torch.testing.assert_close(
        layer.weight_scale_inv,
        torch.tensor([[0.25]], dtype=torch.float32),
    )


def test_b12x_mxfp8_apply_delegates_to_layer_held_linear_holder(monkeypatch) -> None:
    """apply_weights reshapes the activation and hands it to the layer-held
    ``B12xBlockscaledLinear.run``, resolved through the layer-name op body."""
    import vllm.utils.b12x as b12x_utils
    import vllm.model_executor.kernels.linear.mxfp8.b12x as b12x_mod

    # The op is registered under this host's platform dispatch key (CUDA, even
    # with no visible device), so run the real op body directly rather than
    # through torch.ops, which cannot dispatch to CPU tensors here.
    monkeypatch.setattr(
        b12x_mod, "run_b12x_blockscaled_linear", b12x_utils._b12x_blockscaled_linear
    )

    calls = []

    def fake_run(source: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        calls.append((source, bias))
        return source.new_full((source.shape[0], 48), 3.0)

    layer = torch.nn.Module()
    layer.b12x_mxfp8_packed_weight = types.SimpleNamespace(out_features=48)
    layer.b12x_linear = types.SimpleNamespace(run=fake_run)
    name = "mxfp8-apply-delegates-probe"
    layer.b12x_layer_name = _encode_layer_name(name)
    register_b12x_layer(name, layer)
    x = torch.empty((2, 3, 128), dtype=torch.bfloat16)
    bias = torch.empty((48,), dtype=torch.bfloat16)
    kernel = object.__new__(B12xMxfp8LinearKernel)

    output = kernel.apply_weights(layer, x, bias)

    assert output.shape == (2, 3, 48)
    assert output.dtype == x.dtype
    assert len(calls) == 1
    source, called_bias = calls[0]
    assert source.shape == (6, 128)
    assert called_bias is bias
    assert source.data_ptr() == x.data_ptr()


def test_b12x_block_fp8_apply_uses_prepared_plan(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear.scaled_mm.b12x as b12x_mod

    # Bypass the CUDA-only op dispatch key (see the mxfp8 apply test above)
    # and run the real op body directly.
    monkeypatch.setattr(
        b12x_mod, "run_b12x_block_fp8_linear", b12x_mod._b12x_block_fp8_linear
    )

    calls = []

    def mm_block_fp8(*args, **kwargs):
        calls.append((args, kwargs))
        return torch.full(
            (args[0].shape[0], args[2].shape[0]),
            13.0,
            dtype=kwargs["out_dtype"],
        )

    monkeypatch.setattr(
        b12x_mod,
        "_import_b12x_blockscaled",
        lambda: types.SimpleNamespace(mm_block_fp8=mm_block_fp8),
    )

    a = torch.empty((6, 128), dtype=torch.float8_e4m3fn)
    weight = torch.empty((256, 128), dtype=torch.float8_e4m3fn)
    a_scale = torch.empty((6, 1), dtype=torch.float32)
    weight_scale = torch.empty((2, 1), dtype=torch.float32)
    plan = object()
    layer = torch.nn.Module()
    layer.b12x_block_fp8_plans = {6: plan}
    name = "block-fp8-apply-probe"
    layer.b12x_layer_name = _encode_layer_name(name)
    register_b12x_layer(name, layer)
    kernel = object.__new__(B12xFp8BlockScaledMMKernel)
    kernel.config = types.SimpleNamespace(out_dtype=torch.bfloat16)
    kernel._b12x_block_fp8_owner = layer

    output = kernel.apply_block_scaled_mm(a, weight, a_scale, weight_scale)

    assert output.shape == (6, 256)
    assert output.dtype == torch.bfloat16
    assert len(calls) == 1
    assert calls[0] == (
        (a, a_scale, weight, weight_scale),
        {"plan": plan, "out_dtype": torch.bfloat16},
    )
    torch.testing.assert_close(output, torch.full_like(output, 13.0))


def test_b12x_mxfp4_requires_dynamic_activations() -> None:
    config = types.SimpleNamespace(activation_quant_key=kMxfp4Dynamic)
    can_implement, reason = B12xMxFp4LinearKernel.can_implement(config)

    assert can_implement
    assert reason is None

    config.activation_quant_key = None
    can_implement, reason = B12xMxFp4LinearKernel.can_implement(config)

    assert not can_implement
    assert reason == "B12X MXFP4 GEMM requires dynamic MXFP4 activations"


@pytest.mark.parametrize(
    ("kernel_cls", "module_name", "scale_dtype"),
    [
        (
            B12xMxFp4LinearKernel,
            "vllm.model_executor.kernels.linear.mxfp4.b12x",
            torch.uint8,
        ),
        (
            B12xNvFp4LinearKernel,
            "vllm.model_executor.kernels.linear.nvfp4.b12x",
            torch.float8_e4m3fn,
        ),
    ],
)
def test_b12x_fp4_processes_scale_and_preserves_loader(
    monkeypatch,
    kernel_cls,
    module_name: str,
    scale_dtype: torch.dtype,
) -> None:
    scale = torch.empty((48, 8), dtype=scale_dtype)
    swizzled_scale = torch.empty((128, 8), dtype=scale_dtype)
    intrinsics = types.SimpleNamespace(swizzle_block_scale=lambda value: swizzled_scale)
    monkeypatch.setattr(
        importlib.import_module(module_name),
        "_import_b12x_intrinsics",
        lambda: intrinsics,
    )
    layer = torch.nn.Module()
    layer.prefix = "model.layers.0.mlp.shared_expert.down_proj"
    layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
    weight_loader = object()
    layer.weight_scale.weight_loader = weight_loader
    if kernel_cls is B12xNvFp4LinearKernel:
        layer.weight = torch.empty((48, 64), dtype=torch.uint8)
        layer.weight_global_scale = torch.tensor(0.5)
        packed = object()

        def pack_weight(weight, scale, *, recipe, global_scale):
            assert weight.data_ptr() == layer.weight.data_ptr()
            assert scale.data_ptr() == swizzled_scale.data_ptr()
            assert global_scale is layer.weight_global_scale
            assert recipe == "nvfp4"
            return packed

        monkeypatch.setattr(
            importlib.import_module(module_name),
            "_import_b12x_blockscaled",
            lambda: types.SimpleNamespace(pack_weight=pack_weight),
        )
    kernel = object.__new__(kernel_cls)

    kernel.process_weights_after_loading(layer)

    assert layer.weight_scale.data_ptr() == swizzled_scale.data_ptr()
    assert layer.weight_scale.weight_loader is weight_loader
    if kernel_cls is B12xNvFp4LinearKernel:
        assert layer.b12x_nvfp4_packed_weight is packed


def test_b12x_mxfp4_apply_calls_native_blockscaled_gemm(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear.mxfp4.b12x as b12x_mod
    import vllm.utils.flashinfer as flashinfer_utils

    # Bypass the CUDA-only op dispatch key (see the mxfp8 apply test above)
    # and run the real op body directly.
    monkeypatch.setattr(b12x_mod, "run_b12x_mxfp4_linear", b12x_mod._b12x_mxfp4_linear)

    calls: list[tuple] = []
    x_packed = torch.empty((6, 64), dtype=torch.uint8)
    x_scale_storage = torch.empty((128, 4), dtype=torch.uint8)

    def mm_mxfp4(*args, **kwargs):
        calls.append((args, kwargs))
        return torch.full((6, 48), 3.0, dtype=torch.bfloat16)

    monkeypatch.setattr(
        flashinfer_utils,
        "flashinfer_mxfp4_quantize",
        lambda *args, **kwargs: (x_packed, x_scale_storage),
    )
    monkeypatch.setattr(
        b12x_mod,
        "_import_b12x_blockscaled",
        lambda: types.SimpleNamespace(mm_mxfp4=mm_mxfp4),
    )

    layer = torch.nn.Module()
    layer.output_size_per_partition = 48
    layer.weight = torch.empty((48, 64), dtype=torch.uint8)
    layer.weight_scale = torch.empty((128, 4), dtype=torch.uint8)
    plan = object()
    layer.b12x_mxfp4_plans = {6: plan}
    name = "mxfp4-apply-probe"
    layer.b12x_layer_name = _encode_layer_name(name)
    register_b12x_layer(name, layer)
    x = torch.empty((2, 3, 128), dtype=torch.bfloat16)
    bias = torch.ones(48, dtype=torch.bfloat16)
    kernel = object.__new__(B12xMxFp4LinearKernel)

    output = kernel.apply_weights(layer, x, bias)

    assert output.shape == (2, 3, 48)
    torch.testing.assert_close(output, torch.full_like(output, 4.0))
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (
        x_packed,
        x_scale_storage,
        layer.weight,
        layer.weight_scale,
    )
    assert kwargs == {"plan": plan, "out_dtype": torch.bfloat16}




def test_b12x_w4a16_modelopt_vision_width_preserves_bf16_activations(monkeypatch):
    """The real 4304-wide vision MLP must not fall back or quantize activations."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((12, 0), (12, 1)):
        pytest.skip("SM120/SM121 required")
    from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4Config, ModelOptNvFp4W4A16LinearMethod,
    )
    import vllm.model_executor.kernels.linear.nvfp4.b12x as native
    import vllm.model_executor.parameter as parameter
    from vllm.v1.worker.workspace import (
        current_workspace_manager,
        init_workspace_manager,
        reset_workspace_manager,
    )

    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)

    monkeypatch.setenv("VLLM_B12X_NVFP4_ACTIVATION_MODE", "auto")
    def reject_activation_quantization(*args, **kwargs):
        raise AssertionError("W4A16 must not quantize BF16 activations")
    monkeypatch.setattr(native, "scaled_fp4_quant", reject_activation_quantization)
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(3816)
    n, k = 1152, 4304
    config = VllmConfig(kernel_config=KernelConfig(linear_backend="b12x"))
    with set_current_vllm_config(config), torch.no_grad():
        method = ModelOptNvFp4W4A16LinearMethod(ModelOptNvFp4Config(
            quant_method="W4A16_NVFP4", is_checkpoint_nvfp4_serialized=True,
        ))
        layer = torch.nn.Module()
        with torch.device(device):
            method.create_weights(
                layer, input_size_per_partition=k, output_partition_sizes=[n],
                input_size=k, output_size=n, params_dtype=torch.bfloat16,
            )
        codes = torch.randint(0, 16, (n, k), dtype=torch.uint8, device=device)
        scales = (torch.rand(n, k // 16, device=device) + 0.125).to(torch.float8_e4m3fn)
        table = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6],
                             device=device)
        decoded = table[codes.long()] * scales.float().repeat_interleave(16, 1) * 0.25
        layer.weight.copy_(codes[:, ::2] | codes[:, 1::2] << 4)
        layer.weight_scale.copy_(scales)
        layer.weight_scale_2.fill_(0.25)
        reset_workspace_manager()
        init_workspace_manager(device)
        current_workspace_manager().reserve_all(((1,), torch.uint8))
        method.process_weights_after_loading(layer)
        counts = (1, 2, 4, 8, 32)
        session, _ = _prepare(
            layer, device=device, counts=counts, fixed=(1, 2, 4, 8), max_tokens=32,
        )
        try:
            for rows in reversed(counts):
                source = torch.randn(rows, k, dtype=torch.bfloat16, device=device) * 0.125
                expected = (source.float() @ decoded.T).bfloat16()
                actual = method.apply(layer, source)
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.125)
                graph = torch.cuda.CUDAGraph()
                with session.capture():
                    try:
                        with torch.cuda.graph(graph):
                            output = method.apply(layer, source)
                        source.mul_(-0.5)
                        allocated = torch.cuda.memory_allocated(device)
                        graph.replay()
                        torch.cuda.synchronize(device)
                        assert torch.cuda.memory_allocated(device) == allocated
                        expected = (source.float() @ decoded.T).bfloat16()
                        torch.testing.assert_close(output, expected, rtol=0.02, atol=0.125)
                        assert (output.float() - expected.float()).norm() / expected.float().norm() < 0.005
                    finally:
                        graph.reset()
        finally:
            session.close()
            reset_workspace_manager()


def test_b12x_nvfp4_fp16_preserves_quantized_path(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear.nvfp4.b12x as b12x_mod

    # Bypass the CUDA-only op dispatch key (see the mxfp8 apply test above)
    # and run the real op body directly.
    monkeypatch.setattr(
        b12x_mod,
        "run_b12x_nvfp4_serialized_linear",
        b12x_mod._b12x_nvfp4_serialized_linear,
    )

    calls: list[tuple] = []
    quant_calls: list[tuple] = []
    x_packed = torch.empty((6, 64), dtype=torch.uint8)
    x_scale_storage = torch.empty((128, 8), dtype=torch.float8_e4m3fn)

    def quant(*args, **kwargs):
        quant_calls.append((args, kwargs))
        return x_packed, x_scale_storage

    def mm_nvfp4(*args, **kwargs):
        calls.append((args, kwargs))
        return torch.full((6, 48), 3.0, dtype=torch.float16)

    monkeypatch.setattr(b12x_mod, "scaled_fp4_quant", quant)
    monkeypatch.setattr(
        b12x_mod,
        "_import_b12x_blockscaled",
        lambda: types.SimpleNamespace(mm_nvfp4=mm_nvfp4),
    )

    layer = torch.nn.Module()
    layer.output_size_per_partition = 48
    layer.weight = torch.empty((48, 64), dtype=torch.uint8)
    layer.weight_scale = torch.empty((128, 8), dtype=torch.float8_e4m3fn)
    layer.input_global_scale_inv = torch.tensor(2.0)
    layer.alpha = torch.tensor(0.25)
    layer.b12x_activation_mode = "auto"
    layer.b12x_bf16_input_supported = True
    plan = object()
    layer.b12x_nvfp4_serialized_plans = {6: plan}
    name = "nvfp4-fp16-apply-probe"
    layer.b12x_layer_name = _encode_layer_name(name)
    register_b12x_layer(name, layer)
    x = torch.empty((2, 3, 256), dtype=torch.float16)[..., ::2]
    bias = torch.ones(48, dtype=torch.float16)
    kernel = object.__new__(B12xNvFp4LinearKernel)

    output = kernel.apply_weights(layer, x, bias)

    assert output.shape == (2, 3, 48)
    torch.testing.assert_close(output, torch.full_like(output, 4.0))
    assert len(quant_calls) == 1
    quant_args, quant_kwargs = quant_calls[0]
    assert quant_args[0].shape == (6, 128)
    assert quant_args[0].data_ptr() == x.data_ptr()
    assert quant_args[1] is layer.input_global_scale_inv
    assert not quant_args[0].is_contiguous()
    assert quant_kwargs == {"is_sf_swizzled_layout": True}
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (
        x_packed,
        x_scale_storage,
        layer.weight,
        layer.weight_scale,
        layer.alpha,
    )
    assert kwargs == {"plan": plan, "out_dtype": torch.float16}


class _SerializedPlanProbe:
    """Stands in for a b12x fixed plan: records its query and its request."""

    def __init__(self, query=None):
        self.query = query
        self.request_kwargs = None

    def request(self, **kwargs):
        self.request_kwargs = kwargs
        return types.SimpleNamespace(name=kwargs["name"])


def _serialized_probe_layer(name: str) -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.weight = torch.empty((48, 128), dtype=torch.uint8)
    layer.weight_scale = torch.empty((128, 16), dtype=torch.float8_e4m3fn)
    layer.input_global_scale_inv = torch.tensor(2.0)
    layer.alpha = torch.tensor(0.25)
    layer.b12x_weight_only = False
    layer.b12x_activation_mode = "quantized"
    layer.b12x_bf16_input_supported = True
    layer.b12x_nvfp4_serialized_activations = True
    layer.b12x_nvfp4_packed_weight = types.SimpleNamespace(
        in_features=256, padded_in_features=256, out_features=48,
        values=torch.empty(0, dtype=torch.uint8),
    )
    layer.b12x_layer_name = _encode_layer_name(name)
    layer.b12x_nvfp4_serialized_plans = {}
    return layer


def _serialized_query(rows: int, output_dtype: str) -> dict:
    return {
        "recipe": "nvfp4", "call_kind": "serialized", "max_rows": rows,
        "in_features": 256, "padded_in_features": 256, "out_features": 48,
        "input_dtype": "uint8", "output_dtype": output_dtype,
        "expected_m": rows, "alpha_mode": "tensor",
    }


def test_b12x_nvfp4_serialized_lookup_declares_an_unplanned_row_count_without_preparing(
    monkeypatch,
) -> None:
    import b12x.preparation as preparation
    import vllm.model_executor.kernels.linear.nvfp4.b12x as b12x_mod

    declared: list[_SerializedPlanProbe] = []
    api = types.SimpleNamespace(
        FixedBlockscaledQuery=lambda **kwargs: kwargs,
        plan=lambda query: declared.append(_SerializedPlanProbe(query)) or declared[-1],
    )
    monkeypatch.setattr(b12x_mod, "_import_b12x_blockscaled", lambda: api)
    prepared = []
    monkeypatch.setattr(
        preparation, "prepare_default", lambda request: prepared.append(request)
    )
    calls = []
    monkeypatch.setattr(
        b12x_mod, "_serialized_call",
        lambda layer, rows, out_dtype: calls.append((layer, rows, out_dtype)) or f"call:{rows}",
    )
    layer = _serialized_probe_layer("nvfp4-serialized-lookup-probe")
    planned = _SerializedPlanProbe()
    layer.b12x_nvfp4_serialized_plans[6] = planned

    assert b12x_mod._serialized_plan_for(layer, 6, torch.bfloat16) is planned
    assert declared == [] and prepared == [] and calls == []

    plan = b12x_mod._serialized_plan_for(layer, 11, torch.bfloat16)

    assert b12x_mod._serialized_plan_for(layer, 11, torch.bfloat16) is plan
    assert declared == [plan]
    assert layer.b12x_nvfp4_serialized_plans == {6: planned, 11: plan}
    assert plan.query == _serialized_query(11, "bfloat16")
    # Serving never prepares: the plan materializes its default on first use.
    assert plan.request_kwargs is None
    assert calls == [] and prepared == []


def test_b12x_nvfp4_serialized_runtime_declaration_matches_the_startup_unit(
    monkeypatch,
) -> None:
    """A row count declared at serving time gets the query the startup
    preparation unit would have declared for it."""
    import b12x.preparation as preparation
    import vllm.model_executor.kernels.linear.nvfp4.b12x as b12x_mod

    api = types.SimpleNamespace(
        FixedBlockscaledQuery=lambda **kwargs: kwargs,
        plan=_SerializedPlanProbe,
    )
    monkeypatch.setattr(b12x_mod, "_import_b12x_blockscaled", lambda: api)
    monkeypatch.setattr(preparation, "prepare_default", lambda request: None)
    monkeypatch.setattr(
        b12x_mod, "_serialized_call",
        lambda layer, rows, out_dtype: (layer.b12x_layer_name, rows, out_dtype),
    )
    kernel = object.__new__(B12xNvFp4LinearKernel)
    workload = B12xWorkload(
        stage="weights", token_counts=(11,), fixed_token_counts=(),
        output_dtype=torch.float16, max_tokens=11, max_seqs=1, max_model_len=11,
    )
    startup_layer = _serialized_probe_layer("nvfp4-serialized-probe")
    (unit,) = kernel.get_b12x_preparation_units(startup_layer, workload)
    startup_plan = startup_layer.b12x_nvfp4_serialized_plans[11]
    runtime_layer = _serialized_probe_layer("nvfp4-serialized-probe")
    runtime_plan = b12x_mod._serialized_plan_for(runtime_layer, 11, torch.float16)

    assert unit.key == ("nvfp4-serialized-probe", "serialized", (11,))
    assert startup_plan.query == runtime_plan.query == _serialized_query(11, "float16")
    assert startup_plan.request_kwargs == {
        "name": "linear.nvfp4.nvfp4-serialized-probe.serialized.m11",
        "prepare_call": (startup_layer.b12x_layer_name, 11, torch.float16),
        "benchmark_call": (startup_layer.b12x_layer_name, 11, torch.float16),
    }
    # Serving never prepares: the runtime plan is declared only.
    assert runtime_plan.request_kwargs is None


def test_b12x_nvfp4_serialized_prepare_call_primes_the_layer_weights(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear.nvfp4.b12x as b12x_mod

    quant_calls = []
    values = torch.empty((3, 128), dtype=torch.uint8)
    source_scales = torch.empty((128, 16), dtype=torch.float8_e4m3fn)

    def quant(source, scale, **kwargs):
        quant_calls.append((source, scale, kwargs))
        return values, source_scales

    monkeypatch.setattr(b12x_mod, "scaled_fp4_quant", quant)
    runs = []
    state = types.SimpleNamespace(
        run_serialized=lambda *args, **kwargs: runs.append((args, kwargs)) or "out",
    )
    layer = _serialized_probe_layer("nvfp4-serialized-call-probe")

    call = b12x_mod._serialized_call(layer, 3, torch.float16)(state)
    call.produce()
    assert call.run() == "out"

    assert call.owners == (
        layer.weight, layer.weight_scale, layer.alpha, layer.input_global_scale_inv,
    )
    ((source, scale, kwargs),) = quant_calls
    assert source.shape == (3, 256) and source.dtype == torch.float16
    assert scale is layer.input_global_scale_inv
    assert kwargs == {"is_sf_swizzled_layout": True}
    assert runs == [(
        (values, source_scales, layer.weight, layer.weight_scale, layer.alpha),
        {
            "ab_dtype": "float4_e2m1fn", "sf_dtype": "float8_e4m3fn",
            "c_dtype": "float16", "sf_vec_size": 16, "block_fp8": False,
            "stream": None,
        },
    )]


@pytest.mark.parametrize("mode", ["auto", "a16", "quantized"])
def test_b12x_nvfp4_bf16_delegates_to_layer_held_linear_holder(monkeypatch, mode) -> None:
    """The packed-BF16 path is chosen regardless of activation mode and
    delegates to the layer-held ``B12xBlockscaledLinear`` holder."""
    import vllm.utils.b12x as b12x_utils
    import vllm.model_executor.kernels.linear.nvfp4.b12x as b12x_mod

    # Bypass the CUDA-only op dispatch key (see the mxfp8 apply test above)
    # and run the real op body directly.
    monkeypatch.setattr(
        b12x_mod, "run_b12x_blockscaled_linear", b12x_utils._b12x_blockscaled_linear
    )

    calls = []

    def fake_run(source: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        calls.append((source, bias))
        return source.new_full((6, 48), 4.0)

    packed = types.SimpleNamespace(out_features=48)
    name = "nvfp4-bf16-apply-probe"
    layer = _WeakNamespace(
        weight=torch.empty(48, 64, dtype=torch.uint8),
        b12x_nvfp4_packed_weight=packed,
        b12x_linear=types.SimpleNamespace(run=fake_run),
        b12x_layer_name=_encode_layer_name(name),
        b12x_activation_mode=mode,
        b12x_bf16_input_supported=True,
        b12x_weight_only=False,
        input_global_scale_inv=torch.tensor(2.0),
    )
    register_b12x_layer(name, layer)
    x = torch.randn(2, 3, 128, dtype=torch.bfloat16)
    bias = torch.ones(48, dtype=torch.bfloat16)

    def reject_quant(*args, **kwargs):
        pytest.fail("BF16 activation quantization must be owned by b12x")

    monkeypatch.setattr(b12x_mod, "scaled_fp4_quant", reject_quant)
    kernel = object.__new__(B12xNvFp4LinearKernel)
    output = kernel.apply_weights(layer, x, bias)
    assert output.shape == (2, 3, 48)
    assert len(calls) == 1
    source, called_bias = calls[0]
    assert source.data_ptr() == x.data_ptr()
    assert called_bias is bias


@pytest.mark.parametrize("recipe", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize(
    "common,override,expected",
    [
        (None, None, "auto"),
        ("a16", None, "a16"),
        ("a16", "auto", "auto"),
        ("quantized", "a16", "a16"),
        ("a16", "quantized", "quantized"),
    ],
)
def test_b12x_dense_precision_override_precedence(
    monkeypatch, recipe, common, override, expected
):
    from vllm.utils.b12x import get_b12x_dense_activation_mode

    for name, value in (
        ("VLLM_B12X_DENSE_ACTIVATION_MODE", common),
        (f"VLLM_B12X_{recipe.upper()}_ACTIVATION_MODE", override),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert get_b12x_dense_activation_mode(recipe) == expected


@pytest.mark.parametrize("recipe", ["nvfp4", "mxfp8"])
def test_b12x_dense_precision_rejects_invalid_override(monkeypatch, recipe):
    from vllm.utils.b12x import get_b12x_dense_activation_mode

    monkeypatch.setenv(f"VLLM_B12X_{recipe.upper()}_ACTIVATION_MODE", "guess")
    with pytest.raises(ValueError, match="Invalid value"):
        get_b12x_dense_activation_mode(recipe)


@pytest.mark.parametrize("recipe", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("mode", ["auto", "a16", "quantized"])
def test_b12x_dense_precision_gpu_graph_replay(monkeypatch, recipe, mode):
    """Prepare real exact-M executions, then verify numerical and graph replay."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (12, 0),
        (12, 1),
    ):
        pytest.skip("SM120/SM121 required")
    blockscaled = pytest.importorskip("b12x.gemm.blockscaled")
    from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager
    from vllm._custom_ops import scaled_fp4_quant
    from tests.kernels.quantization.nvfp4_utils import dequantize_nvfp4_to_dtype
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        _mxfp8_e4m3_quantize_torch,
    )

    monkeypatch.setenv(f"VLLM_B12X_{recipe.upper()}_ACTIVATION_MODE", mode)
    torch.manual_seed(1234)
    n, k = 4096, 5376
    layer = torch.nn.Module()
    if recipe == "nvfp4":
        codes = torch.randint(0, 16, (n, k), device="cuda", dtype=torch.uint8)
        values = codes[:, ::2] | (codes[:, 1::2] << 4)
        scales = (torch.rand(n, k // 16, device="cuda") + 0.125).to(torch.float8_e4m3fn)
        lut = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
            device="cuda",
        )
        decoded = lut[codes.long()] * scales.float().repeat_interleave(16, 1) * 0.25
        layer.weight_global_scale = torch.tensor([0.25], device="cuda")
        layer.input_global_scale_inv = torch.tensor([128.0], device="cuda")
        layer.alpha = layer.weight_global_scale / layer.input_global_scale_inv
        kernel = B12xNvFp4LinearKernel(NvFp4LinearLayerConfig())
        counts = (1, 8, 32)
    else:
        values = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
        exponents = torch.randint(
            124, 130, (n, k // 32), device="cuda", dtype=torch.uint8
        )
        scales = exponents
        decoded = values.float() * torch.exp2(
            exponents.float() - 127
        ).repeat_interleave(32, 1)
        kernel = object.__new__(B12xMxfp8LinearKernel)
        counts = (1, 2, 4, 8, 14, 32)
    layer.weight = torch.nn.Parameter(values, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scales, requires_grad=False)
    reset_workspace_manager()
    init_workspace_manager(torch.device("cuda"))
    fixed = tuple(count for count in counts if count < max(counts))
    try:
        kernel.process_weights_after_loading(layer)
        packed = getattr(layer, f"b12x_{recipe}_packed_weight")
        if recipe == "nvfp4":
            assert packed.values.data_ptr() == layer.weight.data_ptr()
            assert packed.scale_mma.data_ptr() == layer.weight_scale.data_ptr()
            assert packed.global_scale is layer.weight_global_scale
        session, _ = _prepare(
            layer, device=torch.device("cuda"), counts=counts, fixed=fixed,
            max_tokens=max(counts),
        )
        bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)

        def reference(source, a16):
            if a16:
                return (source.float() @ decoded.T).bfloat16() + bias
            if recipe == "nvfp4":
                aq, sf = scaled_fp4_quant(
                    source, layer.input_global_scale_inv, is_sf_swizzled_layout=True
                )
                return (
                    dequantize_nvfp4_to_dtype(
                        aq, sf, layer.input_global_scale_inv, torch.float32, source.device,
                    ) @ decoded.T
                ).bfloat16() + bias
            aq, sf = _mxfp8_e4m3_quantize_torch(source)
            query = blockscaled.query_from_call((aq, sf), packed, out_dtype=source.dtype)
            plan = reference_plans.get(query)
            if plan is None:
                plan = reference_plans[query] = blockscaled.plan(query)
            return blockscaled.mm((aq, sf), packed, bias=bias, out_dtype=source.dtype, plan=plan)

        reference_plans = {}

        for m in reversed(counts):
            source = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            config = layer.b12x_linear.plan.variants[m].selection.config
            expected = reference(source, config.mode == "a16")
            for _ in range(3):
                actual = kernel.apply_weights(layer, source, bias)
            torch.testing.assert_close(actual, expected, atol=0.5, rtol=0.02)
            graph = torch.cuda.CUDAGraph()
            with session.capture():
                with torch.cuda.graph(graph):
                    output = kernel.apply_weights(layer, source, bias)
                pointer = output.data_ptr()
                for _ in range(2):
                    source.normal_()
                    output.fill_(float("nan"))
                    allocated = torch.accelerator.memory_allocated()
                    graph.replay()
                    torch.accelerator.synchronize()
                    assert torch.accelerator.memory_allocated() == allocated
                    assert output.data_ptr() == pointer
                    assert torch.isfinite(output).all() and torch.count_nonzero(output)
            expected = reference(source, config.mode == "a16")
            relative = (output.float() - expected.float()).norm() / expected.float().norm()
            assert relative < 0.005
    finally:
        session.close()
        reset_workspace_manager()


def test_v41_execution_regimes_cover_padded_capture_sizes(monkeypatch):
    from types import SimpleNamespace

    from vllm.models.deepseek_v4_1 import b12x_layers

    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4096, max_num_seqs=3),
        speculative_config=SimpleNamespace(num_speculative_tokens=2),
        compilation_config=SimpleNamespace(
            cudagraph_capture_sizes=[1, 8, 16, 32],
            max_cudagraph_capture_size=32,
        ),
    )
    monkeypatch.setattr(b12x_layers, "get_current_vllm_config", lambda: config)
    capacities = b12x_layers._execution_capacities()
    # Nine live rows can occupy a larger graph; do not route its padding to
    # the scheduler-capacity prefill regime.
    assert min(capacity for capacity in capacities if capacity >= 32) == 32
    assert max(capacities) == 4096


@pytest.mark.parametrize("leading_shape", [(17,), (1, 17)])
def test_v41_block32_adapter_preserves_native_output_view_and_replay(
    monkeypatch,
    leading_shape,
):
    """Real FP8 execution must adapt vLLM output ranks without copying scratch."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("native b12x projection requires SM12x")
    from types import SimpleNamespace

    from b12x._lib.runtime_control import kernel_resolution_guard
    from vllm.models.deepseek_v4_1 import b12x_layers
    from vllm.v1.worker import workspace

    device = torch.device("cuda")
    monkeypatch.setattr(b12x_layers, "_capacity", lambda: 17)
    monkeypatch.setattr(b12x_layers, "_execution_capacities", lambda: (1, 8, 17))
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    monkeypatch.setattr(workspace, "_manager", workspace.WorkspaceManager(device))
    torch.manual_seed(41107)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.randn(96, 128, device=device).to(torch.float8_e4m3fn),
        requires_grad=False,
    )
    layer.weight_scale_inv = torch.nn.Parameter(
        torch.full((3, 4), 0.125, device=device).to(torch.float8_e8m0fnu),
        requires_grad=False,
    )
    method = b12x_layers.B12xFP8LinearMethod(
        SimpleNamespace(weight_block_size=[32, 32])
    )
    method.process_weights_after_loading(layer)
    from b12x._lib import dense_gemm

    dense_gemm._cached_alpha_one(device)
    allocated_before = torch.cuda.memory_allocated(device)
    session, _ = _prepare(layer, device=device, counts=(1, 8, 17), fixed=(1, 8))
    assert torch.cuda.memory_allocated(device) == allocated_before
    session.freeze()
    manager = workspace.current_workspace_manager()
    manager.reserve_all(*(
        (spec.shape, spec.dtype)
        for plan in layer.b12x_plans for spec in plan.scratch_specs()
    ))
    manager.lock()
    source = torch.randn((*leading_shape, 128), device=device, dtype=torch.bfloat16)

    def oracle():
        values = source.float().reshape(-1, 4, 32)
        scales = torch.exp2(
            torch.ceil(
                torch.log2(values.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448.0)
            )
        )
        values = ((values / scales).to(torch.float8_e4m3fn).float() * scales).reshape(
            -1, 128
        )
        weights = layer.weight.float() * 0.125
        return (values @ weights.T).bfloat16().view(*leading_shape, 96)

    apply = torch.compile(lambda x: method.apply(layer, x), fullgraph=True)
    graph = torch.cuda.CUDAGraph()
    try:
        with kernel_resolution_guard("V4.1 vLLM block32 output-view replay"):
            torch.testing.assert_close(apply(source), oracle(), rtol=0.01, atol=0.01)
            for rows in (1, 3, 8, 9, 17):
                live = source.reshape(-1, 128)[:rows]
                torch.testing.assert_close(
                    apply(live), oracle().reshape(-1, 96)[:rows],
                    rtol=0.01, atol=0.01,
                )
            with workspace.collect_cuda_graph_capture_resources() as retained:
                with session.capture(), torch.cuda.graph(graph):
                    captured = apply(source)
            source.mul_(0.5)
            captured.fill_(float("nan"))
            address = captured.data_ptr()
            allocated = torch.cuda.memory_allocated(device)
            graph.replay()
            torch.cuda.synchronize()
            assert captured.data_ptr() == address
            assert torch.cuda.memory_allocated(device) == allocated
            assert torch.isfinite(captured).all() and torch.count_nonzero(captured) > 0
            torch.testing.assert_close(captured, oracle(), rtol=0.01, atol=0.01)
            del retained
    finally:
        graph.reset()
        session.close()


def _check_v41_vocab_embedding_and_tied_head(device):
    from b12x._lib.runtime_control import kernel_resolution_guard
    from vllm.distributed.parallel_state import graph_capture
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )
    from vllm.models.deepseek_v4_1.quant_config import DeepseekV41FP8Config
    from vllm.v1.worker import workspace

    vocab, hidden = 131, 128
    quant_config = DeepseekV41FP8Config(
        is_checkpoint_fp8_serialized=True,
        weight_block_size=[32, 32],
    )
    with torch.device(device):
        target = VocabParallelEmbedding(
            vocab,
            hidden,
            params_dtype=torch.bfloat16,
            quant_config=quant_config,
        )
        head = ParallelLMHead(
            vocab,
            hidden,
            params_dtype=torch.bfloat16,
            quant_config=quant_config,
        ).tie_weights(target)
    checkpoint = (
        torch.arange(vocab, device=device)[:, None] / 16
        + torch.arange(hidden, device=device)[None, :] / 128
    ).bfloat16()
    target.weight_loader(target.weight, checkpoint)
    ids = torch.tensor([0, 1, 63, 64, 95, 96, 127, 130], device=device)
    probe = torch.zeros((1, hidden), dtype=torch.bfloat16, device=device)
    probe[0, 0] = 1
    target.quant_method.process_weights_after_loading(target)
    allocated_before = torch.cuda.memory_allocated(device)
    session, _ = _prepare(target, device=device, counts=(1, 3, ids.numel()))
    assert torch.cuda.memory_allocated(device) == allocated_before
    session.freeze()
    processor = LogitsProcessor(vocab)

    def check_head():
        logits = processor(head, probe)
        if logits is not None:  # Gather returns logits only on its destination.
            torch.testing.assert_close(
                logits,
                checkpoint[:, 0].unsqueeze(0),
                rtol=0,
                atol=0,
                check_dtype=False,
            )

    graph = torch.cuda.CUDAGraph()
    try:
        lookup = torch.compile(
            lambda values: target.quant_method.embedding(target, values), fullgraph=True,
        )
        for dtype in (torch.int32, torch.int64):
            local_ids = torch.arange(ids.numel(), device=device, dtype=dtype)
            for rows in (1, 3, ids.numel()):
                actual = lookup(local_ids[:rows])
                torch.testing.assert_close(
                    actual, target.weight[local_ids[:rows].long()], rtol=0, atol=0,
                )
            with session.capture(), torch.cuda.graph(graph):
                local_out = lookup(local_ids)
            address = local_out.data_ptr()
            allocated = torch.cuda.memory_allocated(device)
            local_ids.add_(7)
            local_out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize(device)
            assert local_out.data_ptr() == address
            assert torch.cuda.memory_allocated(device) == allocated
            torch.testing.assert_close(
                local_out, target.weight[local_ids.long()], rtol=0, atol=0,
            )
            graph.reset()

        torch.testing.assert_close(target(ids), checkpoint[ids], rtol=0, atol=0)
        check_head()
        with (
            session.capture(),
            kernel_resolution_guard("V4.1 sharded vocabulary embedding replay"),
            workspace.collect_cuda_graph_capture_resources() as retained,
            graph_capture(device=device) as capture_context,
            torch.cuda.graph(graph, stream=capture_context.stream),
        ):
            captured = target(ids)
        address = captured.data_ptr()
        for offset in (3, 17):
            ids.add_(offset).remainder_(vocab)
            checkpoint.neg_()
            target.weight_loader(target.weight, checkpoint)
            captured.fill_(float("nan"))
            allocated = torch.cuda.memory_allocated(device)
            graph.replay()
            torch.cuda.synchronize()
            assert captured.data_ptr() == address
            assert torch.cuda.memory_allocated(device) == allocated
            torch.testing.assert_close(captured, checkpoint[ids], rtol=0, atol=0)
            # Reload the target after tying: the head must see current weights,
            # not a copied or prepacked snapshot from embedding finalization.
            check_head()
        del retained
    finally:
        graph.reset()
        session.close()


def test_v41_vocab_embedding_global_ids_and_target_weight_tie(request, monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("native b12x embedding requires SM12x")
    monkeypatch.setenv("VLLM_MXFP8_LM_HEAD", "0")
    request.getfixturevalue("dist_init")
    with torch.no_grad():
        _check_v41_vocab_embedding_and_tied_head(torch.device("cuda", 0))


def _run_v41_sharded_embedding(rank, port):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    with set_current_vllm_config(VllmConfig()), torch.no_grad():
        try:
            init_test_distributed_environment(2, 1, rank, str(port), local_rank=rank)
            _check_v41_vocab_embedding_and_tied_head(device)
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


@pytest.mark.distributed(num_gpus=2)
def test_v41_vocab_embedding_sharded_global_ids_and_target_weight_tie(monkeypatch):
    # This test already spawns fresh workers. Forking an outer test process
    # after a preceding CUDA test would inherit an unusable CUDA context.
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("native b12x embedding requires two GPUs")
    if any(torch.cuda.get_device_capability(i)[0] != 12 for i in range(2)):
        pytest.skip("native b12x embedding requires two SM12x GPUs")
    monkeypatch.setenv("VLLM_MXFP8_LM_HEAD", "0")
    torch.multiprocessing.spawn(
        _run_v41_sharded_embedding,
        args=(get_open_port(),),
        nprocs=2,
        join=True,
    )


@torch.inference_mode()
@pytest.mark.parametrize("broadcast", [False, True])
@pytest.mark.parametrize("tokens", [3, 129])
def test_v41_mhc_shares_scratch_and_preserves_live_outputs(
    monkeypatch, broadcast, tokens
):
    from types import SimpleNamespace

    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.norm import mhc
    from b12x.preparation import FrozenMapping, PreparationSession

    from vllm.models.deepseek_v4_1 import b12x_layers
    from vllm.utils.b12x import B12xWorkload, register_b12x_layer
    from vllm.v1.worker import workspace

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("native b12x mHC requires SM12x")
    device = torch.device("cuda", torch.cuda.current_device())
    capacity, hidden = 4096, 5120
    monkeypatch.setattr(b12x_layers, "_capacity", lambda: capacity)
    monkeypatch.setattr(
        b12x_layers, "_execution_capacities", lambda: (1, 8, 24, capacity)
    )
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    manager = workspace.WorkspaceManager(device)
    monkeypatch.setattr(workspace, "_manager", manager)
    config = SimpleNamespace(
        hidden_size=hidden,
        hc_mult=4,
        rms_norm_eps=1e-20,
        hc_eps=1e-6,
        hc_sinkhorn_iters=20,
    )
    allocated = torch.cuda.memory_allocated(device)
    with torch.device(device):
        first, second = b12x_layers.B12xMHC(config), b12x_layers.B12xMHC(config)
    # Model construction must not reserve capacity-sized activations per layer.
    assert torch.cuda.memory_allocated(device) - allocated < 1024**2
    torch.manual_seed(4124096)
    shape = (tokens, hidden) if broadcast else (tokens, 4, hidden)
    residual = torch.randn(shape, device=device, dtype=torch.bfloat16)
    fn = torch.randn(24, 4 * hidden, device=device) * 0.001
    first_fn = fn.view(24, 4, hidden).sum(1) if broadcast else fn
    scale = torch.full((3,), 0.1, device=device)
    bias = torch.zeros(24, device=device)
    norm = torch.ones(hidden, device=device, dtype=torch.bfloat16)
    identity = torch.zeros(tokens, 4, device=device)
    identity[:, 0] = 1
    first_owner = _WeakNamespace(
        _b12x_mhc=first,
        hc_attn_fn=first_fn,
        hc_attn_fn_broadcast=first_fn,
        hc_ffn_fn=fn,
        hc_attn_scale=scale,
        hc_ffn_scale=scale,
        hc_attn_base=bias,
        hc_ffn_base=bias,
        attn_norm=SimpleNamespace(weight=norm),
        ffn_norm=SimpleNamespace(weight=norm),
    )
    second_owner = _WeakNamespace(
        _b12x_mhc=second,
        hc_attn_fn=fn,
        hc_attn_fn_broadcast=fn,
        hc_ffn_fn=fn,
        hc_attn_scale=scale,
        hc_ffn_scale=scale,
        hc_attn_base=bias,
        hc_ffn_base=bias,
        attn_norm=SimpleNamespace(weight=norm),
        ffn_norm=SimpleNamespace(weight=norm),
    )
    first_name = f"v41-mhc-first-{id(first_owner):x}"
    second_name = f"v41-mhc-second-{id(second_owner):x}"
    register_b12x_layer(first_name, first_owner)
    register_b12x_layer(second_name, second_owner)
    first.bind_layer_name(first_name)
    second.bind_layer_name(second_name)

    workload = B12xWorkload(
        stage="weights", token_counts=(1, 8, 24, capacity),
        fixed_token_counts=(1, 8, 24), output_dtype=torch.bfloat16,
        max_tokens=capacity, max_seqs=1, max_model_len=capacity,
    )
    units = [
        *first.get_b12x_preparation_units(first_owner, workload),
        *second.get_b12x_preparation_units(second_owner, workload),
    ]
    requests = tuple(request for unit in units for request in unit.requests)
    session = PreparationSession(device=device, autotune=False)
    session.prepare(requests, autotune=False)

    def run():
        a = first.pre(residual, first_fn, scale, bias, norm, None)
        b = second.post_pre(a[3] * 0.125, a[0], a[1], a[2], fn, scale, bias, norm, a[4])
        return (*a, b[0], *b)

    reference_plans = [
        mhc.plan(
            mhc.Caps(device=device, max_tokens=tokens, hidden_size=hidden, split_k=hidden // 64),
            invocation=FrozenMapping(invocation),
        )
        for invocation in (
            dict(operation="pre", output_mode="functional", lagged_mix=True,
                 expanded_residual=not broadcast, has_norm_weight=True,
                 rms_eps=1e-20, hc_eps=1e-6, sinkhorn_iters=20, norm_eps=1e-20),
            dict(operation="pre", output_mode="functional", lagged_mix=True,
                 expanded_residual=True, has_norm_weight=True,
                 rms_eps=1e-20, hc_eps=1e-6, sinkhorn_iters=20, norm_eps=1e-20),
            dict(operation="post", output_mode="functional"),
        )
    ]

    def expected():
        incoming = identity
        state = residual
        outputs = []
        for index in range(2):
            predicted = torch.empty_like(incoming)
            result = mhc.run_pre(
                state, first_fn if index == 0 else fn, scale, bias,
                pre_mix=incoming, pre_out=predicted, norm_weight=norm,
                norm_eps=1e-20, rms_eps=1e-20, hc_eps=1e-6, sinkhorn_iters=20,
                plan=reference_plans[index],
            )
            outputs.extend((*result, predicted))
            if index == 0:
                state = mhc.run_post(
                    result[3] * 0.125, result[0], result[1], result[2],
                    plan=reference_plans[2],
                )
                outputs.append(state)
            incoming = predicted
        return outputs

    for operation in ("pre", "post_pre", "post"):
        assert first._plan_for(operation, tokens) is first._plans[(operation, capacity)]
        assert first._plan_for(operation, 8) is first._plans[(operation, 8)]
    actual = run()
    reference = expected()
    for got, want in zip(actual, reference, strict=True):
        torch.testing.assert_close(got, want, rtol=2e-5, atol=0.008)
    retained_outputs = [tensor.clone() for tensor in actual]
    # Poison every scratch buffer the prepared "pre"/"post_pre" plans declare
    # at full capacity: the retained outputs above must be independent copies,
    # not views into shared scratch storage.
    specs = [
        spec
        for plan in (
            first._plans[("pre", capacity)], first._plans[("post_pre", capacity)],
        )
        for spec in plan.memory_requirements().scratch
    ]
    buffers = manager.get_simultaneous(*((spec.shape, spec.dtype) for spec in specs))
    for buffer in buffers:
        buffer.fill_(173)
    for got, want in zip(actual, retained_outputs, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)

    manager.lock()
    graph = torch.cuda.CUDAGraph()
    try:
        with (
            kernel_resolution_guard("V4.1 shared mHC workspace"),
            session.capture(),
            workspace.collect_cuda_graph_capture_resources() as resources,
            torch.cuda.graph(graph),
        ):
            captured = run()
        for factor in (0.5, -1.0):
            residual.mul_(factor)
            graph.replay()
            torch.cuda.synchronize()
            for got, want in zip(captured, expected(), strict=True):
                torch.testing.assert_close(got, want, rtol=2e-5, atol=0.008)
        del resources
    finally:
        graph.reset()
        session.close()


def _holder_with_prepared_state(monkeypatch, *, required_workspace: int):
    """A holder whose plan resolves to a fake prepared state; mm is recorded."""
    import b12x.preparation as preparation
    import vllm.model_executor.kernels.linear.b12x_blockscaled as module
    from vllm.model_executor.kernels.linear.b12x_blockscaled import B12xBlockscaledLinear

    holder = B12xBlockscaledLinear.__new__(B12xBlockscaledLinear)
    holder.layer_name = "layer.linear"
    holder.packed = types.SimpleNamespace(out_features=8, in_features=16)
    holder.activation_scale = None
    holder.plan = types.SimpleNamespace(prepared=object())
    state = types.SimpleNamespace(required_workspace=required_workspace)
    monkeypatch.setattr(preparation, "require_prepared", lambda plan, component, device=None: state)
    calls = []
    monkeypatch.setattr(
        module, "get_b12x_blockscaled",
        lambda: types.SimpleNamespace(mm=lambda *args, **kwargs: calls.append(kwargs) or "out"),
    )
    return holder, calls


def test_b12x_holder_reports_its_prepared_scratch_requirement(monkeypatch) -> None:
    holder, _ = _holder_with_prepared_state(monkeypatch, required_workspace=4096)
    assert holder.get_workspace_size(11) == 4096
    holder.plan = types.SimpleNamespace(
        prepared=None,
        scratch_specs=lambda: (types.SimpleNamespace(nbytes=100), types.SimpleNamespace(nbytes=28)),
    )
    assert holder.get_workspace_size(11) == 128


def test_b12x_holder_runs_inside_the_reserved_scratch_when_one_is_bound(monkeypatch) -> None:
    import vllm.v1.worker.workspace as workspace

    holder, calls = _holder_with_prepared_state(monkeypatch, required_workspace=64)
    monkeypatch.setattr(
        workspace, "current_workspace_manager",
        lambda: (_ for _ in ()).throw(AssertionError("the manager must not be asked")),
    )
    reserved = torch.zeros(256, dtype=torch.uint8)
    source = torch.zeros(3, 16, dtype=torch.bfloat16)
    with workspace.use_preallocated_workspace(reserved):
        assert holder.run(source, None) == "out"
    (kwargs,) = calls
    assert kwargs["workspace"].data_ptr() == reserved.data_ptr()
    assert kwargs["workspace"].numel() == 64
    with workspace.use_preallocated_workspace(torch.zeros(32, dtype=torch.uint8)):
        with pytest.raises(ValueError, match="reserved scratch holds 32 bytes"):
            holder.run(source, None)


def test_b12x_holder_draws_from_the_manager_without_a_reserved_scratch(monkeypatch) -> None:
    import vllm.v1.worker.workspace as workspace

    holder, calls = _holder_with_prepared_state(monkeypatch, required_workspace=64)
    drawn = torch.zeros(64, dtype=torch.uint8)
    requests = []
    monkeypatch.setattr(
        workspace, "current_workspace_manager",
        lambda: types.SimpleNamespace(get_simultaneous=lambda *specs: requests.append(specs) or (drawn,)),
    )
    assert holder.run(torch.zeros(3, 16, dtype=torch.bfloat16), None) == "out"
    assert requests == [(((64,), torch.uint8),)]
    assert calls[0]["workspace"] is drawn


def test_b12x_linear_methods_report_their_kernel_scratch_requirement() -> None:
    from vllm.model_executor.layers.linear import LinearMethodBase

    class _Kernel:
        def get_workspace_size(self, layer, rows):
            return 7 * rows

    class _Bare(LinearMethodBase):
        def create_weights(self, *args, **kwargs):
            raise NotImplementedError

        def apply(self, *args, **kwargs):
            raise NotImplementedError

    class _Method(_Bare):
        def __init__(self):
            self.kernel = _Kernel()

    layer = torch.nn.Module()
    assert _Method().get_workspace_size(layer, 3) == 21
    assert _Bare().get_workspace_size(layer, 3) == 0
    mxfp8 = B12xMxfp8LinearKernel.__new__(B12xMxfp8LinearKernel)
    assert mxfp8.get_workspace_size(layer, 3) == 0
    layer.b12x_linear = types.SimpleNamespace(get_workspace_size=lambda rows: 5 * rows)
    assert mxfp8.get_workspace_size(layer, 3) == 15


@pytest.mark.parametrize("recipe", ["block", "tensor"])
@torch.inference_mode()
def test_b12x_fp8_eager_unplanned_rows_use_default_and_replay(recipe):
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.preparation import PreparationSession
    from vllm.model_executor.kernels.linear.scaled_mm import b12x as module

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("requires SM12x")
    torch.manual_seed(38)
    device = torch.device("cuda", torch.cuda.current_device())
    n, k, capacity = 512, 256, 128
    values = torch.randn(n, k, device=device).to(torch.float8_e4m3fn)
    source = torch.randn(capacity, k, device=device).to(torch.float8_e4m3fn)
    layer = torch.nn.Module()
    name = f"fp8-eager-{recipe}-{id(layer):x}"
    layer.b12x_layer_name = _encode_layer_name(name)
    register_b12x_layer(name, layer)
    if recipe == "block":
        kernel = object.__new__(B12xFp8BlockScaledMMKernel)
        kernel.config = types.SimpleNamespace(out_dtype=torch.bfloat16)
        layer.weight = torch.nn.Parameter(values, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(torch.full((n // 128, k // 128), 0.5, device=device), requires_grad=False)
        layer.b12x_block_fp8_plans = {}
        plans = layer.b12x_block_fp8_plans
        scales = torch.full((capacity, k // 128), 0.25, device=device)
        def run(rows):
            return module.run_b12x_block_fp8_linear(source[:rows], scales[:rows], layer.weight, layer.weight_scale, torch.bfloat16, layer.b12x_layer_name)
    else:
        kernel = object.__new__(B12xTensorFP8ScaledMMLinearKernel)
        kernel.config = types.SimpleNamespace(out_dtype=torch.bfloat16)
        api = module._import_b12x_tensor_fp8()
        layer.b12x_tensor_fp8_packed_weight = api.pack_weight(values, torch.tensor([0.125], device=device))
        layer.b12x_tensor_fp8_plans = {}
        plans = layer.b12x_tensor_fp8_plans
        def run(rows):
            return module.run_b12x_tensor_fp8_linear(source[:rows], None, n, torch.bfloat16, layer.b12x_layer_name)
    workload = B12xWorkload(
        stage="weights", token_counts=(4, capacity), fixed_token_counts=(4,),
        output_dtype=torch.bfloat16, max_tokens=capacity, max_seqs=4, max_model_len=1024,
    )
    units = kernel.get_b12x_preparation_units(layer, workload)
    def expected(rows):
        return (source[:rows].float() @ values.float().T * 0.125).bfloat16()
    with PreparationSession(device=device, autotune=False) as session:
        session.prepare(tuple(request for unit in units for request in unit.requests))
        assert set(plans) == {4, capacity}
        actual = run(11)
        assert plans[11].prepared is not None
        assert plans[11].selection.source == "fixed"
        torch.testing.assert_close(actual, expected(11), rtol=0.02, atol=0.125)
        session.freeze()
        with kernel_resolution_guard("FP8 prepared exact-M execution"):
            for rows in (4, 11, capacity):
                torch.testing.assert_close(run(rows), expected(rows), rtol=0.02, atol=0.125)
            graph = torch.cuda.CUDAGraph()
            try:
                with session.capture(), torch.cuda.graph(graph):
                    captured = run(4)
                pointer = captured.data_ptr()
                source.copy_((-source.float()).to(source.dtype))
                graph.replay()
                torch.cuda.synchronize()
                assert captured.data_ptr() == pointer
                torch.testing.assert_close(captured, expected(4), rtol=0.02, atol=0.125)
            finally:
                graph.reset()


@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
@torch.no_grad()
def test_v41_unquantized_prepares_dtypes_and_exact_rows_before_replay(output_dtype):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("native b12x projection requires SM12x")
    from vllm.models.deepseek_v4_1.b12x_layers import B12xLinearMethod

    device = torch.device("cuda", torch.cuda.current_device())
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.randn(384, 5120, device=device).bfloat16().mul_(0.125),
        requires_grad=False,
    )
    layer.out_dtype = output_dtype
    method = B12xLinearMethod()
    method.process_weights_after_loading(layer)
    allocated_before = torch.cuda.memory_allocated(device)
    session, _ = _prepare(
        layer, device=device, counts=(1, 8, 256), fixed=(1, 8),
        output_dtype=output_dtype,
    )
    assert torch.cuda.memory_allocated(device) == allocated_before
    session.freeze()
    graph = torch.cuda.CUDAGraph()
    tolerance = 0.015 if output_dtype == torch.bfloat16 else 1e-4
    try:
        apply = torch.compile(lambda x: method.apply(layer, x), fullgraph=True)
        for dtype in (torch.bfloat16, torch.float32):
            source = torch.randn(256, 5120, device=device, dtype=dtype).mul_(0.125)
            for rows in (1, 8, 17, 256):
                actual = method.apply(layer, source[:rows])
                expected = (source[:rows].float() @ layer.weight.float().T).to(output_dtype)
                torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
            apply(source)
            with session.capture(), torch.cuda.graph(graph):
                captured = apply(source)
            pointer = captured.data_ptr()
            source.neg_()
            captured.fill_(float("nan"))
            allocated = torch.cuda.memory_allocated(device)
            graph.replay()
            torch.cuda.synchronize(device)
            assert captured.data_ptr() == pointer
            assert torch.cuda.memory_allocated(device) == allocated
            assert torch.isfinite(captured).all() and torch.count_nonzero(captured) > 0
            expected = (source.float() @ layer.weight.float().T).to(output_dtype)
            torch.testing.assert_close(captured, expected, rtol=tolerance, atol=tolerance)
            graph.reset()
    finally:
        graph.reset()
        session.close()


class FakePlan:
    def __init__(self, invocation=None) -> None:
        self.invocation = invocation

    def request(self, *, name, prepare_call, benchmark_call):
        return types.SimpleNamespace(name=name)


def _stub_b12x_preparation(monkeypatch) -> None:
    """Let CPU-only tests build preparation units without the b12x package."""
    try:
        importlib.import_module("b12x.preparation")
    except ImportError:
        package = types.ModuleType("b12x")
        package.__path__ = []
        preparation = types.ModuleType("b12x.preparation")
        preparation.FrozenMapping = dict
        monkeypatch.setitem(sys.modules, "b12x", package)
        monkeypatch.setitem(sys.modules, "b12x.preparation", preparation)


@pytest.mark.parametrize("first_layer", [True, False])
@pytest.mark.parametrize("model", ["deepseek_v4", "glm5next"])
def test_b12x_mhc_declares_the_operations_each_layer_runs(
    monkeypatch, model: str, first_layer: bool
) -> None:
    """Every operation a layer runs is prepared up to the workload capacity.

    Only the first mHC layer holds the broadcast projection and runs ``pre``,
    and GLM keeps no BF16 FFN projection under its own norm names. A missing
    declaration surfaces at the first live call as ``prepared capacity 0``.
    """
    from vllm.models.deepseek_v4.nvidia import b12x as dsv4_b12x

    hidden, mult = 64, 4
    fake_mhc = types.SimpleNamespace(
        MULT=mult,
        DEFAULT_BLOCK_K=64,
        DEFAULT_BLOCK_H=64,
        Caps=lambda **caps: caps,
        plan=lambda caps, *, invocation: FakePlan(dict(invocation)),
        run_pre=None,
        run_post=None,
        run_post_pre=None,
    )
    monkeypatch.setattr(dsv4_b12x, "_require_b12x_mhc", lambda: fake_mhc)
    _stub_b12x_preparation(monkeypatch)

    fn = torch.zeros(24, mult * hidden)
    norm = types.SimpleNamespace(weight=torch.ones(hidden), variance_epsilon=1e-6)
    if model == "deepseek_v4":
        # DeepSeek V4 constructs with the default operands.
        extra = {}
        named = {
            "attn_norm": norm,
            "ffn_norm": norm,
            "hc_ffn_fn_bf16": fn.to(torch.bfloat16),
        }
    else:
        extra = {
            "operands": dsv4_b12x.MHCOperands(
                attn_norm="input_layernorm",
                ffn_norm="post_attention_layernorm",
                ffn_fn_bf16=None,
            )
        }
        named = {"input_layernorm": norm, "post_attention_layernorm": norm}
    layer = types.SimpleNamespace(
        hc_attn_fn=fn,
        hc_ffn_fn=fn,
        hc_attn_scale=torch.ones(3),
        hc_ffn_scale=torch.ones(3),
        hc_attn_base=torch.zeros(24),
        hc_ffn_base=torch.zeros(24),
        hc_attn_fn_broadcast=fn.view(24, mult, hidden).sum(1) if first_layer else None,
        **named,
    )
    mhc = dsv4_b12x.B12xMHCResidual(
        hidden_size=hidden,
        hc_mult=mult,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
        **extra,
    )
    workload = B12xWorkload(
        stage="weights",
        token_counts=(1, 8, 8192),
        fixed_token_counts=(1, 8),
        output_dtype=torch.bfloat16,
        max_tokens=8192,
        max_seqs=1,
        max_model_len=8192,
    )

    (unit,) = mhc.get_b12x_preparation_units(layer, workload)

    expected = (
        (("pre",) if first_layer else ())
        + ("post_pre",)
        + (("post_pre_bf16",) if model == "deepseek_v4" else ())
        + ("post",)
    )
    assert {operation for operation, _ in mhc._plans} == set(expected)
    assert len(unit.requests) == 3 * len(expected)
    for operation in expected:
        assert mhc._plan_for(operation, 5) is mhc._plans[(operation, 8192)]
        invocation = mhc._plans[(operation, 8)].invocation
        assert invocation["has_fn_bf16"] == (operation == "post_pre_bf16")
    if not first_layer:
        with pytest.raises(RuntimeError, match="prepared capacity 0"):
            mhc._plan_for("pre", 8)


def test_b12x_mhc_keeps_plans_across_workloads(monkeypatch) -> None:
    from vllm.models.deepseek_v4.nvidia import b12x as dsv4_b12x

    hidden, mult = 64, 4
    fake_mhc = types.SimpleNamespace(
        MULT=mult,
        DEFAULT_BLOCK_K=64,
        DEFAULT_BLOCK_H=64,
        Caps=lambda **caps: caps,
        plan=lambda caps, *, invocation: FakePlan(dict(invocation)),
        run_pre=None,
        run_post=None,
        run_post_pre=None,
    )
    monkeypatch.setattr(dsv4_b12x, "_require_b12x_mhc", lambda: fake_mhc)
    _stub_b12x_preparation(monkeypatch)

    fn = torch.zeros(24, mult * hidden)
    norm = types.SimpleNamespace(weight=torch.ones(hidden), variance_epsilon=1e-6)
    layer = types.SimpleNamespace(
        hc_attn_fn=fn,
        hc_ffn_fn=fn,
        hc_attn_scale=torch.ones(3),
        hc_ffn_scale=torch.ones(3),
        hc_attn_base=torch.zeros(24),
        hc_ffn_base=torch.zeros(24),
        hc_attn_fn_broadcast=fn.view(24, mult, hidden).sum(1),
        attn_norm=norm,
        ffn_norm=norm,
        hc_ffn_fn_bf16=fn.to(torch.bfloat16),
    )
    mhc = dsv4_b12x.B12xMHCResidual(
        hidden_size=hidden,
        hc_mult=mult,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
    )

    def workload(max_tokens: int, fixed: tuple[int, ...]) -> B12xWorkload:
        return B12xWorkload(
            stage="weights",
            token_counts=fixed + (max_tokens,),
            fixed_token_counts=fixed,
            output_dtype=torch.bfloat16,
            max_tokens=max_tokens,
            max_seqs=1,
            max_model_len=max_tokens,
        )

    mhc.get_b12x_preparation_units(layer, workload(64, (8,)))
    shared = mhc._plans[("post", 8)]
    mhc.get_b12x_preparation_units(layer, workload(128, (16,)))

    assert mhc._plans[("post", 8)] is shared
    assert ("post", 16) in mhc._plans
    assert ("post", 128) in mhc._plans
    assert mhc._plan_for("post", 16) is mhc._plans[("post", 16)]
    assert mhc._plan_for("post", 100) is mhc._plans[("post", 128)]


@pytest.mark.parametrize("kind", ["block", "tensor"])
def test_b12x_fp8_preparation_units_key_on_their_planned_rows(
    monkeypatch, kind: str
) -> None:
    import vllm.model_executor.kernels.linear.scaled_mm.b12x as b12x_mod

    layer = torch.nn.Module()
    name = f"fp8-{kind}-preparation-key-probe"
    layer.b12x_layer_name = _encode_layer_name(name)
    register_b12x_layer(name, layer)
    workload = B12xWorkload(
        stage="weights",
        token_counts=(1, 8, 64),
        fixed_token_counts=(),
        output_dtype=torch.bfloat16,
        max_tokens=64,
        max_seqs=1,
        max_model_len=64,
    )
    if kind == "block":
        layer.weight = torch.nn.Parameter(
            torch.empty((256, 128), dtype=torch.float8_e4m3fn), requires_grad=False
        )
        layer.weight_scale_inv = torch.nn.Parameter(
            torch.empty((2, 1)), requires_grad=False
        )
        layer.b12x_block_fp8_plans = {}
        monkeypatch.setattr(
            b12x_mod,
            "_block_fp8_plan",
            lambda layer, rows, dtype: layer.b12x_block_fp8_plans.setdefault(
                rows, FakePlan()
            ),
        )
        kernel = object.__new__(B12xFp8BlockScaledMMKernel)
    else:
        layer.b12x_tensor_fp8_packed_weight = types.SimpleNamespace(
            values=torch.empty((256, 128), dtype=torch.float8_e4m3fn),
            in_features=128,
        )
        layer.b12x_tensor_fp8_plans = {}
        monkeypatch.setattr(
            b12x_mod,
            "_tensor_fp8_plan",
            lambda layer, rows, dtype: layer.b12x_tensor_fp8_plans.setdefault(
                rows, FakePlan()
            ),
        )
        kernel = object.__new__(B12xTensorFP8ScaledMMLinearKernel)
        kernel.config = types.SimpleNamespace(out_dtype=torch.bfloat16)

    (unit,) = kernel.get_b12x_preparation_units(layer, workload)

    assert unit.name == f"{kind.upper()}_FP8"
    assert unit.key == (name, (1, 8, 64))
    assert tuple(request.name for request in unit.requests) == tuple(
        f"linear.{kind}_fp8.{name}.m{rows}" for rows in (1, 8, 64)
    )
