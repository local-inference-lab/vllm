# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native b12x execution boundaries for the V4.1 checkpoint contract."""

from __future__ import annotations

from bisect import bisect_left
from functools import cache
from weakref import WeakValueDictionary

import torch
from b12x.gemm import bf16_gemv, block_fp8_linear
from b12x.norm import hyperconnection, mhc
from b12x.sequence import embedding
from torch import nn

from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.linear import LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    create_fp8_scale_parameter,
    create_fp8_weight_parameter,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)
from vllm.model_executor.parameter import BlockQuantScaleParameter
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    retain_cuda_graph_capture_resource,
)


def _capacity() -> int:
    return get_current_vllm_config().scheduler_config.max_num_batched_tokens


def _execution_capacities() -> tuple[int, ...]:
    config = get_current_vllm_config()
    capacity = _capacity()
    spec = config.speculative_config
    decode_capacity = min(
        capacity,
        max(
            config.scheduler_config.max_num_seqs
            * (1 + (spec.num_speculative_tokens if spec is not None else 0)),
            config.compilation_config.max_cudagraph_capture_size or 0,
        ),
    )
    graph_sizes = config.compilation_config.cudagraph_capture_sizes or ()
    from .ced import ced_decoder_start

    hf = getattr(getattr(config, "model_config", None), "hf_config", None)
    ced_capacities = ()
    if hf is not None and ced_decoder_start(hf) is not None:
        window = hf.sliding_window
        ced_capacities = range(
            window,
            min(capacity, config.scheduler_config.max_num_seqs * window) + 1,
            window,
        )
    return tuple(
        sorted(
            {
                capacity,
                decode_capacity,
                *(size for size in graph_sizes if 0 < size <= decode_capacity),
                *ced_capacities,
            }
        )
    )


_LINEARS: WeakValueDictionary[int, nn.Module] = WeakValueDictionary()


@cache
def _norm_plan(device, capacity, hidden):
    return hyperconnection.plan(
        hyperconnection.Caps(
            device=device, max_tokens=capacity, hidden_size=hidden, streams=1, lowrank=1
        )
    )


@cache
def _mhc_plan(device, capacity, hidden):
    return mhc.plan(mhc.Caps(device=device, max_tokens=capacity, hidden_size=hidden))


@torch.library.custom_op("vllm::dsv41_block32_linear", mutates_args=("out",))
def _block32_linear(x: torch.Tensor, out: torch.Tensor, key: int) -> None:
    layer = _LINEARS[key]
    rows = x.numel() // layer.weight.shape[1]
    index = bisect_left(layer.b12x_capacities, rows)
    if index == len(layer.b12x_capacities):
        raise ValueError("V4.1 linear rows exceed planned capacity")
    plan = layer.b12x_plans[index]
    scratch = current_workspace_manager().get_simultaneous(*plan.shapes_and_dtypes())
    binding = block_fp8_linear.bind(
        plan,
        scratch=scratch,
        source=x,
        packed_weight=layer.b12x_weight,
        output=out.view(-1, layer.weight.shape[0], 1),
    )
    retain_cuda_graph_capture_resource(binding)
    block_fp8_linear.run(binding=binding)


@_block32_linear.register_fake
def _block32_linear_fake(x, out, key):
    return None


@torch.library.custom_op("vllm::dsv41_embedding_out", mutates_args=("out",))
def _embedding_out(weight: torch.Tensor, ids: torch.Tensor, out: torch.Tensor) -> None:
    embedding.run(weight, ids, out=out)
    retain_cuda_graph_capture_resource(out)


@_embedding_out.register_fake
def _embedding_out_fake(weight, ids, out):
    return None


class B12xEmbeddingMethod(UnquantizedEmbeddingMethod):
    """Retain sharded weight loading/ties; replace only token-row compute."""

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        for id_dtype in (torch.int32, torch.int64):
            embedding.precompile(layer.weight, id_dtype=id_dtype)

    def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:
        # Each invocation owns its result: DSpark retains earlier Markov rows
        # for confidence evaluation. Graph capture owns these fixed allocations.
        out = torch.empty(
            (*input_.shape, layer.weight.shape[1]),
            device=layer.weight.device,
            dtype=layer.weight.dtype,
        )
        _embedding_out(layer.weight, input_, out)
        return out


class B12xLinearMethod(UnquantizedLinearMethod):
    def process_weights_after_loading(self, layer: nn.Module) -> None:
        for input_dtype in (torch.bfloat16, torch.float32):
            bf16_gemv.precompile(
                layer.weight,
                input_dtype=input_dtype,
                output_dtype=getattr(layer, "out_dtype", torch.bfloat16),
            )

    def apply(
        self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        if bias is not None:
            raise ValueError("V4.1 native projections require bias-free weights")
        out = bf16_gemv.mm(
            x, layer.weight, output_dtype=getattr(layer, "out_dtype", torch.bfloat16)
        )
        retain_cuda_graph_capture_resource(out)
        return out


class B12xFP8LinearMethod(LinearMethodBase):
    """Checkpoint block32 FP8, without a foreign kernel selection phase."""

    def __init__(self, quant_config):
        if quant_config.weight_block_size != [32, 32]:
            raise ValueError("V4.1 requires checkpoint block32 FP8 linears")

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = sum(output_partition_sizes)
        layer.orig_dtype = params_dtype
        layer.weight_block_size = [32, 32]
        layer.register_parameter(
            "weight",
            create_fp8_weight_parameter(
                sum(output_partition_sizes), input_size_per_partition, loader
            ),
        )
        layer.register_parameter(
            "weight_scale_inv",
            create_fp8_scale_parameter(
                BlockQuantScaleParameter,
                output_partition_sizes,
                input_size_per_partition,
                [32, 32],
                loader,
                scale_dtype=torch.float8_e8m0fnu,
            ),
        )

    def process_weights_after_loading(self, layer):
        layer.b12x_weight = block_fp8_linear.pack_weight(
            layer.weight, layer.weight_scale_inv, block_size=(32, 32)
        )
        layer.b12x_capacities = _execution_capacities()
        layer.b12x_plans = tuple(
            block_fp8_linear.plan(
                block_fp8_linear.Caps(
                    device=layer.weight.device,
                    max_tokens=capacity,
                    in_features=layer.weight.shape[1],
                    out_features=layer.weight.shape[0],
                    block_size=(32, 32),
                )
            )
            for capacity in layer.b12x_capacities
        )
        for capacity in layer.b12x_capacities:
            block_fp8_linear.prewarm(
                layer.b12x_weight, (1, capacity), expected_m=capacity
            )
        layer.b12x_key = id(layer)
        _LINEARS[layer.b12x_key] = layer

    def apply(self, layer, x, bias=None):
        if bias is not None:
            raise ValueError("V4.1 block32 projections require bias-free weights")
        out = torch.empty(
            (*x.shape[:-1], layer.weight.shape[0]),
            dtype=torch.bfloat16,
            device=x.device,
        )
        _block32_linear(x, out, layer.b12x_key)
        return out


@torch.library.custom_op("vllm::dsv41_rmsnorm", mutates_args=("out",))
def _rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor, eps: float, capacity: int
) -> None:
    p = _norm_plan(x.device, capacity, x.shape[-1])
    # These unused binding slots must remain disjoint from normalized storage.
    binding = hyperconnection.bind(
        p,
        normalized=out,
        bottleneck=x.view(-1)[: x.shape[0]].view(-1, 1),
        block_input=x,
        tokens=x.shape[0],
    )
    hyperconnection.run_grouped_rmsnorm(
        x, weight, eps=eps, binding=binding, zero_centered=False
    )


@_rmsnorm.register_fake
def _rmsnorm_fake(x, weight, out, eps, capacity):
    return None


class B12xRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=torch.bfloat16), requires_grad=False
        )
        self.variance_epsilon = eps
        self.capacity = _capacity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, shape[-1]).contiguous()
        out = torch.empty_like(x)
        _rmsnorm(
            x,
            self.weight,
            out,
            self.variance_epsilon,
            self.capacity * (shape[-2] if len(shape) > 2 else 1),
        )
        return out.view(shape)


@torch.library.custom_op(
    "vllm::dsv41_mhc_pre", mutates_args=("residual_out", "y", "post", "comb", "pre_out")
)
def _mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm: torch.Tensor,
    pre: torch.Tensor,
    residual_out: torch.Tensor,
    y: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    pre_out: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    iterations: int,
    capacity: int,
    previous_output: torch.Tensor | None = None,
    previous_post: torch.Tensor | None = None,
    previous_comb: torch.Tensor | None = None,
) -> None:
    # Use fixed capacity buckets so B12X can plan decode and prefill separately.
    # Live row counts never become compilation or plan-cache keys.
    decode_capacity = min(capacity, 64)
    if residual.shape[0] <= decode_capacity:
        capacity = decode_capacity
    plan = _mhc_plan(residual.device, capacity, y.shape[-1])
    (scratch,) = current_workspace_manager().get_simultaneous(*plan.shapes_and_dtypes())
    binding = mhc.bind(
        plan,
        scratch=scratch,
        tokens=residual.shape[0],
        y=y,
        post=post,
        comb=comb,
        out=residual_out,
        expected_m=capacity,
        pre_out=pre_out,
    )
    # Outputs are caller-owned tensors with ordinary PyTorch lifetimes.
    # Retain only the shared arena, not every layer's transient activations.
    retain_cuda_graph_capture_resource(scratch)
    if previous_output is None:
        operation = mhc.run_pre
        inputs = (residual, fn, scale, base)
    else:
        operation = mhc.run_post_pre
        inputs = (
            previous_output,
            residual,
            previous_post,
            previous_comb,
            fn,
            scale,
            base,
        )
    operation(
        *inputs,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=iterations,
        norm_weight=norm,
        norm_eps=rms_eps,
        pre_mix=pre,
        binding=binding,
    )


@_mhc_pre.register_fake
def _mhc_pre_fake(
    residual,
    fn,
    scale,
    base,
    norm,
    pre,
    residual_out,
    y,
    post,
    comb,
    pre_out,
    rms_eps,
    hc_eps,
    iterations,
    capacity,
    previous_output=None,
    previous_post=None,
    previous_comb=None,
):
    return None


@torch.library.custom_op("vllm::dsv41_mhc_post", mutates_args=("out",))
def _mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    out: torch.Tensor,
) -> None:
    mhc.run_post(x, residual, post, comb, out=out)


@_mhc_post.register_fake
def _mhc_post_fake(x, residual, post, comb, out):
    return None


class B12xMHC(nn.Module):
    """Plan shared scratch; allocate only the outputs live at each invocation."""

    def __init__(self, config):
        super().__init__()
        self.capacity = _capacity()
        self.capacities = _execution_capacities()
        self.hidden_size = config.hidden_size
        self.rms_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.iterations = config.hc_sinkhorn_iters
        if config.hc_mult != 4:
            raise ValueError("V4.1 mHC requires four streams")

    def pre(
        self,
        residual,
        fn,
        scale,
        base,
        norm,
        pre,
        *,
        previous_output=None,
        previous_post=None,
        previous_comb=None,
    ):
        tokens = residual.shape[0]
        index = bisect_left(self.capacities, tokens)
        if index == len(self.capacities):
            raise ValueError("V4.1 mHC rows exceed planned capacity")
        if pre is None:
            # The read-only initial mix must not share storage with native
            # scratch. Only the first sublayer needs this small live-row input.
            pre = torch.zeros((tokens, 4), dtype=torch.float32, device=residual.device)
            pre[:, 0].fill_(1)
        residual_out = torch.empty(
            (tokens, 4, self.hidden_size), dtype=residual.dtype, device=residual.device
        )
        y = torch.empty(
            (tokens, self.hidden_size), dtype=residual.dtype, device=residual.device
        )
        post = torch.empty((tokens, 4), dtype=torch.float32, device=residual.device)
        comb = torch.empty((tokens, 4, 4), dtype=torch.float32, device=residual.device)
        pre_out = torch.empty((tokens, 4), dtype=torch.float32, device=residual.device)
        _mhc_pre(
            residual,
            fn,
            scale,
            base,
            norm,
            pre,
            residual_out,
            y,
            post,
            comb,
            pre_out,
            self.rms_eps,
            self.hc_eps,
            self.iterations,
            self.capacities[index],
            previous_output,
            previous_post,
            previous_comb,
        )
        return residual_out, post, comb, y, pre_out

    def post_pre(self, x, residual, post, comb, fn, scale, base, norm, pre):
        return self.pre(
            residual,
            fn,
            scale,
            base,
            norm,
            pre,
            previous_output=x,
            previous_post=post,
            previous_comb=comb,
        )

    def post(self, x, residual, post, comb):
        out = torch.empty_like(residual)
        _mhc_post(x, residual, post, comb, out)
        return out


def collapse(state: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    out = torch.empty(
        (state.shape[0], state.shape[-1]), dtype=state.dtype, device=state.device
    )
    mhc.run_collapse(state, pre, out=out)
    return out


def stream_mean(state: torch.Tensor) -> torch.Tensor:
    out = torch.empty(
        (state.shape[0], state.shape[-1]), dtype=state.dtype, device=state.device
    )
    mhc.run_collapse(state, None, out=out)
    return out
