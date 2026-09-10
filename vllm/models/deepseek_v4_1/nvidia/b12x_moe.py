# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 expert-parallel MXFP4 with FP32 local accumulation and BF16 transport."""

import torch
from flashinfer.b12x.moe import fused_moe
from flashinfer.b12x.norm import hyperconnection
from torch import nn

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.models.deepseek_v4.nvidia.model import DeepseekV4MegaMoEExperts
from vllm.triton_utils import tl, triton
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    retain_cuda_graph_capture_resource,
)

from ..b12x_layers import B12xLinearMethod


@triton.jit(do_not_specialize=["count"])
def _local_ids(
    source, dest, count, start: tl.constexpr, end: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(source + i, i < count, other=-1)
    tl.store(dest + i, tl.where((v >= start) & (v < end), v - start, -1), i < count)


class B12xV41Experts(DeepseekV4MegaMoEExperts):
    """Reuse the existing expert loader only; no MegaMoE execution is inherited."""

    def __init__(self, vllm_config, *, num_experts, top_k, prefix):
        config = vllm_config.model_config.hf_config
        ranks = get_tensor_model_parallel_world_size()
        if num_experts % ranks:
            raise ValueError("V4.1 experts must partition evenly over TP ranks")
        super().__init__(
            vllm_config,
            num_experts=num_experts,
            num_local_experts=num_experts // ranks,
            experts_start_idx=get_tensor_model_parallel_rank() * (num_experts // ranks),
            top_k=top_k,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            prefix=prefix,
        )
        self.limit = config.swiglu_limit
        self.prepared = None

    def finalize_weights(self, shared_experts=None):
        if self.prepared is not None:
            return
        # The inherited loader places checkpoint w1 (gate) before w3 (up).
        # b12x names that gate/up source order "w31", not "w13".
        wp = fused_moe.plan_weights(
            quant_modes="w4a8_mx",
            source_format="fp4_e8m0_k32",
            activation="silu",
            params_dtype=torch.bfloat16,
            num_experts=self.num_local_experts,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            w13_layout="w31",
            numerical_recipe="deepseek_v41",
        )
        unit = torch.ones(
            self.num_local_experts, dtype=torch.float32, device=self.w13_weight.device
        )
        self.prepared = fused_moe.prepare_weights(
            plan=wp,
            params_dtype=torch.bfloat16,
            w1_fp4=self.w13_weight,
            w2_fp4=self.w2_weight,
            w1_blockscale=self.w13_weight_scale,
            w2_blockscale=self.w2_weight_scale,
            w1_global_scale=unit,
            w2_global_scale=unit,
            a1_gscale=unit,
            a2_gscale=unit,
        )
        self.plan = fused_moe.plan(
            fused_moe.Caps(
                device=self.w13_weight.device,
                max_tokens=self.max_num_tokens,
                num_topk=self.top_k,
                weight_plan=wp,
                quant_mode="w4a8_mx",
                core_token_counts=(self.max_num_tokens,),
                route_num_experts=0,
                swiglu_limit=self.limit,
                frozen=True,
            )
        )
        (spec,) = self.plan.scratch_specs()
        self.local_ids = torch.empty(
            (self.max_num_tokens, self.top_k), dtype=torch.int32, device=spec.device
        )
        if wp.discards_source_parameters:
            # Match the existing b12x loader's ownership transfer: retaining
            # the complete checkpoint and packed experts doubles serving HBM.
            for name in (
                "w13_weight",
                "w2_weight",
                "w13_weight_scale",
                "w2_weight_scale",
            ):
                param = getattr(self, name)
                param.data = torch.empty(0, dtype=param.dtype, device=param.device)

    def run_native(self, x, weights, ids, out):
        if self.prepared is None:
            raise RuntimeError("V4.1 experts must be finalized before execution")
        local = self.local_ids[: x.shape[0]]
        _local_ids[(triton.cdiv(ids.numel(), 256),)](
            ids, local, ids.numel(), self.experts_start_idx, self.experts_end_idx, 256
        )
        scratch = current_workspace_manager().get_simultaneous(
            *((s.shape, s.dtype) for s in self.plan.scratch_specs())
        )
        binding = fused_moe.bind(
            self.plan,
            scratch=scratch,
            a=x,
            experts=self.prepared,
            topk_weights=weights,
            topk_ids=local,
            output=out,
            input_scales_static=True,
            unit_scale_contract=True,
        )
        retain_cuda_graph_capture_resource(binding)
        fused_moe.run(binding=binding)

    def forward(self, x, weights, ids):
        out = torch.empty_like(x)
        _experts(x, weights, ids, out, self.prefix)
        return out


@torch.library.custom_op("vllm::dsv41_experts", mutates_args=("out",))
def _experts(
    x: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    out: torch.Tensor,
    prefix: str,
) -> None:
    get_forward_context().no_compile_layers[prefix].run_native(x, weights, ids, out)


@_experts.register_fake
def _experts_fake(x, weights, ids, out, prefix):
    return None


@torch.library.custom_op(
    "vllm::dsv41_route", mutates_args=("selected", "ids", "weights")
)
def _route(
    logits: torch.Tensor,
    bias: torch.Tensor,
    image_bias: torch.Tensor | None,
    image_mask: torch.Tensor | None,
    selected: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    scaling: float,
) -> None:
    fused_moe.route_topk(
        logits,
        selected,
        ids,
        weights,
        renormalize=True,
        score_func="sqrtsoftplus",
        correction_bias=bias,
        image_correction_bias=image_bias,
        image_mask=image_mask,
        routed_scaling_factor=scaling,
    )


@_route.register_fake
def _route_fake(logits, bias, image_bias, image_mask, selected, ids, weights, scaling):
    return None


class B12xV41SharedExperts(nn.Module):
    def __init__(self, config, quant_config, prefix):
        super().__init__()
        intermediate = config.moe_intermediate_size * config.n_shared_experts
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate, intermediate],
            bias=False,
            disable_tp=True,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate,
            config.hidden_size,
            bias=False,
            disable_tp=True,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.limit = config.swiglu_limit

    def forward(self, x):
        gu, _ = self.gate_up_proj(x)
        act = torch.empty(
            (gu.shape[0], gu.shape[1] // 2), dtype=gu.dtype, device=gu.device
        )
        hyperconnection.run_swiglu(gu, limit=self.limit, out=act)
        return self.down_proj(act)[0]


class DeepseekV4MoE(nn.Module):
    def __init__(self, vllm_config, prefix="", use_sequence_parallel=False):
        super().__init__()
        if use_sequence_parallel:
            raise ValueError("V4.1 native EP consumes replicated token rows")
        config = vllm_config.model_config.hf_config
        if vllm_config.kernel_config.moe_backend not in ("auto", "b12x"):
            raise ValueError("V4.1 requires the native b12x MoE backend")
        if vllm_config.kernel_config.linear_backend not in ("auto", "b12x"):
            raise ValueError("V4.1 requires native b12x linear kernels")
        if config.scoring_func != "sqrtsoftplus" or not config.norm_topk_prob:
            raise ValueError("V4.1 requires normalized sqrtsoftplus routing")
        if getattr(config, "gate_temp", 1.0) != 1.0:
            raise ValueError("V4.1 published checkpoint requires gate_temp=1")
        if getattr(config, "expert_dtype", "fp4") != "fp4":
            raise ValueError("V4.1 native expert recipe requires MXFP4")
        if vllm_config.parallel_config.enable_eplb:
            raise ValueError("V4.1 fixed expert ownership does not support EPLB")
        is_draft = extract_layer_index(prefix) >= config.num_hidden_layers
        self.is_draft = is_draft
        self.n_routed_experts = (
            config.dspark_n_routed_experts if is_draft else config.n_routed_experts
        )
        self.n_activated_experts = (
            config.dspark_num_experts_per_tok
            if is_draft
            else config.num_experts_per_tok
        )
        self.n_logical_experts = self.n_physical_experts = self.n_routed_experts
        self.n_redundant_experts = 0
        self.n_local_physical_experts = (
            self.n_routed_experts // get_tensor_model_parallel_world_size()
        )
        self.n_shared_experts = config.n_shared_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.use_mega_moe = True  # existing checkpoint mapping, not a compute backend
        self.gate = ReplicatedLinear(
            config.hidden_size,
            self.n_routed_experts,
            bias=False,
            params_dtype=torch.bfloat16,
            prefix=f"{prefix}.gate",
        )
        self.gate.quant_method = B12xLinearMethod()
        self.gate.out_dtype = torch.float32
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(self.n_routed_experts, dtype=torch.float32), requires_grad=False
        )
        self.gate.bias_vl = nn.Parameter(
            torch.empty(self.n_routed_experts, dtype=torch.float32), requires_grad=False
        )
        self.experts = B12xV41Experts(
            vllm_config,
            num_experts=self.n_routed_experts,
            top_k=self.n_activated_experts,
            prefix=f"{prefix}.experts",
        )
        self.shared_experts = B12xV41SharedExperts(
            config, vllm_config.quant_config, f"{prefix}.shared_experts"
        )

    def forward(self, x, input_ids=None):
        if input_ids is None:
            raise ValueError("V4.1 modality routing requires raw input IDs")
        logits, _ = self.gate(x)
        shape = (x.shape[0], self.n_activated_experts)
        selected = torch.empty(shape, dtype=torch.float32, device=x.device)
        weights = torch.empty_like(selected)
        ids = torch.empty(shape, dtype=torch.int32, device=x.device)
        _route(
            logits,
            self.gate.e_score_correction_bias,
            None if self.is_draft else self.gate.bias_vl,
            None if self.is_draft else input_ids == 129264,
            selected,
            ids,
            weights,
            self.routed_scaling_factor,
        )
        routed = self.experts(x, weights, ids)
        routed = tensor_model_parallel_all_reduce(routed)
        shared = self.shared_experts(x)
        out = torch.empty_like(shared)
        hyperconnection.run_add(routed, shared, out=out)
        return out

    def finalize_mega_moe_weights(self):
        self.experts.finalize_weights()
