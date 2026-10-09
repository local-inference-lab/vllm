# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from tests.kernels.moe.utils import make_test_quant_config
from tests.kernels.quantization.nvfp4_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    dequantize_nvfp4_to_dtype,
)
from tests.kernels.utils import torch_moe
from vllm import _custom_ops as ops
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe import (
    FlashInferExperts,
    is_valid_flashinfer_cutlass_fused_moe,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_cutlass_fused_moe
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.torch_utils import set_random_seed

if not has_flashinfer_cutlass_fused_moe() or not current_platform.has_device_capability(
    100
):
    pytest.skip(
        "Requires flashinfer_cutlass_fused_moe and nvfp4 support",
        allow_module_level=True,
    )

MNK_FACTORS = [
    (2, 1024, 1024),
    (2, 3072, 1024),
    (2, 3072, 1536),
    (64, 1024, 1536),
    (64, 3072, 1024),
    (64, 2048, 1536),
    (224, 1024, 1024),
    (224, 1024, 1536),
]


@pytest.mark.parametrize(
    "activation",
    [MoEActivation.SWIGLUOAI, MoEActivation.SWIGLUOAI_UNINTERLEAVE],
)
def test_flashinfer_swigluoai_params_are_forwarded(activation, monkeypatch):
    from flashinfer.fused_moe.core import ActivationType

    moe_config = FusedMoEConfig(
        num_experts=2,
        experts_per_token=1,
        hidden_dim=128,
        intermediate_size=128,
        num_local_experts=2,
        num_logical_experts=2,
        activation=activation,
        device="cuda",
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.bfloat16,
        routing_method=RoutingMethodType.TopK,
    )
    quant_config = FusedMoEQuantConfig.make(
        gemm1_alpha=1.702,
        gemm1_beta=1.0,
        gemm1_clamp_limit=7.0,
    )
    experts = FlashInferExperts(moe_config=moe_config, quant_config=quant_config)

    call_args = {}

    def fake_flashinfer_cutlass_fused_moe(**kwargs):
        call_args.update(kwargs)

    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.experts."
        "flashinfer_cutlass_moe.flashinfer_cutlass_fused_moe",
        fake_flashinfer_cutlass_fused_moe,
    )
    experts.apply(
        output=torch.empty((1, 128), device="cuda", dtype=torch.bfloat16),
        hidden_states=torch.empty((1, 128), device="cuda", dtype=torch.bfloat16),
        w1=torch.empty((2, 256, 128), device="cuda", dtype=torch.bfloat16),
        w2=torch.empty((2, 128, 128), device="cuda", dtype=torch.bfloat16),
        topk_weights=torch.ones((1, 1), device="cuda", dtype=torch.float32),
        topk_ids=torch.zeros((1, 1), device="cuda", dtype=torch.int64),
        activation=activation,
        global_num_experts=2,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=None,
        workspace2=None,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    assert experts._supports_activation(activation)
    assert call_args["activation_type"] == ActivationType.Swiglu
    for name, value in (
        ("swiglu_alpha", 1.702),
        ("swiglu_beta", 1.0),
        ("swiglu_limit", 7.0),
    ):
        torch.testing.assert_close(
            call_args[name],
            torch.full((2,), value, device="cuda", dtype=torch.float32),
        )


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("e", [40, 64, 256])
@pytest.mark.parametrize("topk", [1, 6, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("activation", [MoEActivation.SILU, MoEActivation.RELU2_NO_MUL])
@torch.inference_mode()
def test_flashinfer_fp4_moe_no_graph(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
    activation: MoEActivation,
    workspace_init,
):
    set_random_seed(7)
    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10

        quant_blocksize = 16
        is_gated_act = activation.is_gated

        w1_q, w2_q, quant_config = make_test_quant_config(
            e,
            n,
            k,
            in_dtype=dtype,
            quant_dtype="nvfp4",
            block_shape=None,
            per_act_token_quant=False,
            make_gate=is_gated_act,
        )

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids, _ = fused_topk(a, score, topk, renormalize=False)

        assert is_valid_flashinfer_cutlass_fused_moe(a, w1_q, w2_q)

        moe_config = FusedMoEConfig(
            num_experts=e,
            experts_per_token=topk,
            hidden_dim=k,
            intermediate_size=n,
            num_local_experts=e,
            num_logical_experts=e,
            activation=activation,
            device="cuda",
            moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
            in_dtype=dtype,
            routing_method=RoutingMethodType.TopK,
            max_num_tokens=next_power_of_2(m),
        )

        flashinfer_experts = FusedMoEKernel(
            maybe_make_prepare_finalize(
                moe=moe_config,
                quant_config=quant_config,
                allow_new_interface=True,
                use_monolithic=False,
            ),
            FlashInferExperts(moe_config=moe_config, quant_config=quant_config),
        )

        flashinfer_output = flashinfer_experts.apply(
            hidden_states=a,
            w1=w1_q,
            w2=w2_q,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=e,
            expert_map=None,
            apply_router_weight_on_input=False,
        )

        # Reference check:
        a_global_scale = (
            (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(a.flatten(), dim=-1)
        ).to(torch.float32)
        a_fp4, a_scale_interleaved = ops.scaled_fp4_quant(a, a_global_scale)
        _, m_k = a_fp4.shape
        a_in_dtype = dequantize_nvfp4_to_dtype(
            a_fp4,
            a_scale_interleaved,
            a_global_scale,
            dtype=a.dtype,
            device=a.device,
            block_size=quant_blocksize,
        )

        w1_d = torch.empty(
            (e, (2 if is_gated_act else 1) * n, k), device="cuda", dtype=dtype
        )
        w2_d = torch.empty((e, k, n), device="cuda", dtype=dtype)

        for idx in range(0, e):
            w1_d[idx] = dequantize_nvfp4_to_dtype(
                w1_q[idx],
                quant_config.w1_scale[idx],
                (1 / quant_config.g1_alphas[idx]),
                dtype=dtype,
                device=w1_q.device,
                block_size=quant_blocksize,
            )
            w2_d[idx] = dequantize_nvfp4_to_dtype(
                w2_q[idx],
                quant_config.w2_scale[idx],
                (1 / quant_config.g2_alphas[idx]),
                dtype=dtype,
                device=w2_q.device,
                block_size=quant_blocksize,
            )

        torch_output = torch_moe(
            a_in_dtype, w1_d, w2_d, score, topk, activation=activation
        )

        torch.testing.assert_close(
            torch_output, flashinfer_output, atol=1e-1, rtol=1e-1
        )


@pytest.mark.parametrize(
    "backend,quant_dtype",
    [
        ("cutlass", "nvfp4"),
        ("cutedsl", "nvfp4"),
        ("trtllm", "nvfp4"),
        ("b12x", "nvfp4"),
        ("cutlass_a16", "nvfp4"),
        ("emulation", "nvfp4"),
        ("cutlass", torch.float8_e4m3fn),
        ("cutlass", "mxfp8"),
        ("cutlass", None),
    ],
)
def test_preallocated_moe_output_matches_activation_packing(
    monkeypatch, backend, quant_dtype
):
    import vllm.model_executor.layers.fused_moe.modular_kernel as mk
    from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import (
        CutlassExpertsFp4,
    )
    from vllm.model_executor.layers.fused_moe.experts.flashinfer_b12x_moe import (
        FlashInferB12xExperts,
    )
    from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_moe import (
        FlashInferCuteDSLExperts,
    )
    from vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe import (
        Nvfp4QuantizationEmulationTritonExperts,
    )
    from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
        TrtLlmNvFp4ExpertsModular,
    )
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
        FusedMoEMethodBase,
    )

    classes = {
        "cutlass": FlashInferExperts,
        "cutedsl": FlashInferCuteDSLExperts,
        "trtllm": TrtLlmNvFp4ExpertsModular,
        "b12x": FlashInferB12xExperts,
        "cutlass_a16": CutlassExpertsFp4,
        "emulation": Nvfp4QuantizationEmulationTritonExperts,
    }
    moe = SimpleNamespace(
        in_dtype=torch.bfloat16,
        hidden_dim=128,
        intermediate_size_per_partition=256,
        experts_per_token=2,
        num_experts=4,
        num_local_experts=4,
        activation=MoEActivation.SILU,
    )
    experts = object.__new__(classes[backend])
    experts.moe_config = moe
    experts.quant_config = FusedMoEQuantConfig.make(quant_dtype=quant_dtype)
    experts.hidden_dim = moe.hidden_dim
    experts.per_token_activation = False
    impl = object.__new__(mk.FusedMoEKernelModularImpl)
    impl.fused_experts = experts
    from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
        MoEPrepareAndFinalizeNoDPEPModular,
    )

    impl.prepare_finalize = MoEPrepareAndFinalizeNoDPEPModular()
    method = SimpleNamespace(moe=moe, moe_kernel=SimpleNamespace(impl=impl))
    hidden = torch.empty((7, moe.hidden_dim), dtype=moe.in_dtype)
    monkeypatch.setattr(mk.current_platform, "is_cpu", lambda: True)

    workspace, shared = FusedMoEMethodBase.prepare_workspace(method, hidden, 512)

    assert workspace[2].shape == hidden.shape
    assert shared.numel() == 512
    assert shared.untyped_storage().data_ptr() not in {
        buffer.untyped_storage().data_ptr() for buffer in workspace
    }


@pytest.mark.parametrize("rows", [1, 32, 4096])
@torch.inference_mode()
def test_flashinfer_nvfp4_shared_workspace_graph(rows, workspace_init):
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
        FusedMoEMethodBase,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
        MoEPrepareAndFinalizeNoDPEPModular,
    )

    set_random_seed(13)
    hidden_dim, intermediate, experts_count, topk = 4096, 512, 4, 2
    hidden = torch.randn(rows, hidden_dim, device="cuda", dtype=torch.bfloat16) / 10
    w1, w2, quant = make_test_quant_config(
        experts_count,
        intermediate,
        hidden_dim,
        in_dtype=hidden.dtype,
        quant_dtype="nvfp4",
        block_shape=None,
        per_act_token_quant=False,
    )
    moe = FusedMoEConfig(
        num_experts=experts_count,
        experts_per_token=topk,
        hidden_dim=hidden_dim,
        intermediate_size=intermediate,
        num_local_experts=experts_count,
        num_logical_experts=experts_count,
        activation=MoEActivation.SILU,
        device="cuda",
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=hidden.dtype,
        routing_method=RoutingMethodType.TopK,
        max_num_tokens=rows,
    )
    kernel = FusedMoEKernel(
        MoEPrepareAndFinalizeNoDPEPModular(),
        FlashInferExperts(moe_config=moe, quant_config=quant),
    )
    weights, ids, _ = fused_topk(
        hidden,
        torch.randn(rows, experts_count, device="cuda", dtype=hidden.dtype),
        topk,
        renormalize=False,
    )
    kwargs = dict(
        hidden_states=hidden,
        w1=w1,
        w2=w2,
        topk_weights=weights,
        topk_ids=ids,
        activation=moe.activation,
        global_num_experts=experts_count,
        expert_map=None,
        apply_router_weight_on_input=False,
    )
    expected = kernel.apply(**kwargs).clone()
    method = SimpleNamespace(moe=moe, moe_kernel=kernel)
    workspace, shared = FusedMoEMethodBase.prepare_workspace(method, hidden, 4096)
    shared.fill_(37)
    actual = kernel.apply(**kwargs, workspace=workspace)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = kernel.apply(**kwargs, workspace=workspace)
    torch.accelerator.synchronize()
    allocations = torch.accelerator.memory_stats()["allocation.all.allocated"]
    for _ in range(3):
        workspace[2].fill_(float("nan"))
        graph.replay()
    torch.accelerator.synchronize()
    assert torch.accelerator.memory_stats()["allocation.all.allocated"] == allocations
    assert replayed.data_ptr() == workspace[2].data_ptr()
    assert torch.all(shared == 37)
    torch.testing.assert_close(replayed, expected, atol=0, rtol=0)


if __name__ == "__main__":
    test_flashinfer_fp4_moe_no_graph((2, 1024, 1024), 40, 1, torch.half)
