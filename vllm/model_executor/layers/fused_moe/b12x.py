# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""b12x modular tensor-parallel fused MoE backend."""

import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
    kMxfp8Dynamic,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.b12x import (
    set_b12x_preparation_provider,
    B12xPreparationUnit,
    B12xWorkload,
    PreparationResourceUnavailableError,
    get_b12x_fused_moe,
    reuse_packed_weight_storage,
)

logger = init_logger(__name__)

_B12X_MOE_MODES: dict[
    tuple[torch.dtype | str | None, torch.dtype | str | None],
    tuple[str, str, str],
] = {
    ("mxfp4", "mxfp8"): ("w4a8_mx", "fp4_e8m0_k32", "w31"),
    ("mxfp4", None): ("w4a16", "fp4_e8m0_k32", "w31"),
    ("nvfp4", "nvfp4"): ("nvfp4", "modelopt_nvfp4", "w31"),
    ("nvfp4", "mxfp8"): ("w4a8_nvfp4", "modelopt_nvfp4", "w31"),
    ("nvfp4", None): ("w4a16", "modelopt_nvfp4", "w31"),
}


def _require_b12x_fused_moe() -> Any:
    fused_moe = get_b12x_fused_moe()
    assert fused_moe is not None
    return fused_moe


def _b12x_activation_name(activation: MoEActivation) -> str:
    if activation == MoEActivation.SILU:
        return "silu"
    if activation in (MoEActivation.RELU2, MoEActivation.RELU2_NO_MUL):
        return "relu2"
    return activation.value


@dataclass(frozen=True)
class _PreparedMoECall:
    """Priming tensors for one exact prepared MoE variant.

    These are trial-only bindings over the real loaded expert representation;
    serving always binds the layer-held prepared plan to caller tensors.
    """

    state: Any
    tokens: int
    topk: int
    prepared: Any
    output_dtype: torch.dtype

    def make(self, tensors):
        from b12x.preparation import PreparedCall

        scratch = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
            for spec in self.state.scratch.scratch_specs()
        )
        (
            hidden,
            activation_source,
            output,
            route_ids,
            route_weights,
            ids,
            weights,
        ) = tensors

        def reset() -> None:
            output.zero_()
            for buffer in scratch:
                buffer.zero_()

        def produce() -> None:
            hidden.copy_(activation_source)
            ids.copy_(route_ids)
            weights.copy_(route_weights)

        def restore() -> None:
            reset()
            produce()

        binding = self.state.bind(
            scratch=scratch,
            a=hidden,
            experts=self.prepared,
            topk_weights=weights,
            topk_ids=ids,
            output=output,
            input_scales_static=True,
        )
        return PreparedCall(
            run=binding.run,
            output=output,
            produce=produce,
            reset=reset,
            restore=restore,
            capture_safe=False,
        )


def _prepared_moe_call_factory(
    *, tokens: int, topk: int, prepared: Any, output_dtype: torch.dtype
):
    shared = None

    def factory(state: Any):
        nonlocal shared
        tensors = None if shared is None else tuple(ref() for ref in shared)
        if tensors is None or any(tensor is None for tensor in tensors):
            device = prepared.device
            hidden = torch.empty(
                (tokens, int(prepared.hidden_size)),
                dtype=prepared.plan.activation.io_dtype,
                device=device,
            )
            activation_source = torch.empty_like(hidden).normal_(
                mean=0.0, std=0.125
            )
            output = torch.empty(hidden.shape, dtype=output_dtype, device=device)
            route_rows = torch.arange(
                tokens, dtype=torch.int32, device=device
            ).unsqueeze(1)
            route_columns = torch.arange(
                topk, dtype=torch.int32, device=device
            ).unsqueeze(0)
            route_ids = (route_rows + route_columns).remainder_(
                int(prepared.num_experts)
            ).contiguous()
            route_logits = (
                route_rows.to(dtype=torch.float32) * 0.03125
                + route_columns.to(dtype=torch.float32) * 0.125
            )
            route_weights = torch.softmax(route_logits, dim=-1).contiguous()
            ids = torch.empty_like(route_ids)
            weights = torch.empty_like(route_weights)
            tensors = (
                hidden,
                activation_source,
                output,
                route_ids,
                route_weights,
                ids,
                weights,
            )
            shared = tuple(weakref.ref(tensor) for tensor in tensors)
        return _PreparedMoECall(
            state=state,
            tokens=tokens,
            topk=topk,
            prepared=prepared,
            output_dtype=output_dtype,
        ).make(tensors)

    return factory


def _is_current_stream_capturing() -> bool:
    is_capturing = getattr(torch.cuda, "is_current_stream_capturing", None)
    return bool(is_capturing is not None and is_capturing())




def _normalize_topk_weights(topk_weights: torch.Tensor) -> torch.Tensor:
    if topk_weights.dtype == torch.float32 and topk_weights.is_contiguous():
        return topk_weights
    if _is_current_stream_capturing():
        raise RuntimeError(
            "b12x MoE topk_weights normalization would allocate during CUDA capture"
        )
    return topk_weights.to(dtype=torch.float32).contiguous()




def _replace_parameter_with_empty(
    layer: torch.nn.Module,
    name: str,
) -> torch.Tensor | None:
    parameter = getattr(layer, name, None)
    if not isinstance(parameter, torch.Tensor):
        return None
    empty = torch.empty((0,), dtype=parameter.dtype, device=parameter.device)
    replace_parameter(layer, name, empty)
    return getattr(layer, name)


def _normalize_expert_scale(scale: torch.Tensor) -> torch.Tensor:
    if scale.ndim == 2:
        if scale.shape[1] not in (1, 2):
            raise ValueError(
                "expected an expert scale with one or two columns, got "
                f"{tuple(scale.shape)}"
            )
        scale = scale[:, 0]
    return scale.to(dtype=torch.float32).contiguous()


class B12xExperts(mk.FusedMoEExpertsModular):
    """FP4 MoE experts backed by the b12x SM12x planned API."""

    def __init__(
        self,
        moe_config: mk.FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        if quant_config.weight_quant_dtype not in ("mxfp4", "nvfp4"):
            raise ValueError(
                "b12x MoE requires MXFP4 or NVFP4 weights, got "
                f"{quant_config.weight_quant_dtype}"
            )
        scheme = (
            quant_config.weight_quant_dtype,
            quant_config.quant_dtype,
        )
        try:
            self._quant_mode, self._source_format, self._w13_layout = _B12X_MOE_MODES[
                scheme
            ]
        except KeyError as exc:
            raise ValueError(
                f"unsupported b12x MoE quantization scheme {scheme}"
            ) from exc
        self._prepared_experts: Any | None = None
        self._source_parameters_released = False
        self._unit_scales: dict[torch.device, torch.Tensor] = {}
        self._apply_router_weight_on_input = False
        self._plan: Any | None = None
        self._plan_key: tuple | None = None
        self._plan_activation: MoEActivation | None = None
        self._plan_route_on_input: bool | None = None

    def _unit_scale(self, device: torch.device, num_experts: int) -> torch.Tensor:
        scale = self._unit_scales.get(device)
        if scale is None or scale.numel() != num_experts:
            scale = torch.ones(num_experts, dtype=torch.float32, device=device)
            self._unit_scales[device] = scale
        return scale

    def _weight_global_scale(
        self,
        device: torch.device,
        num_experts: int,
        scale: torch.Tensor | None,
        name: str,
    ) -> torch.Tensor:
        if self._source_format != "modelopt_nvfp4":
            return self._unit_scale(device, num_experts)
        if scale is None:
            raise ValueError(f"b12x NVFP4 MoE requires {name}")
        scale = _normalize_expert_scale(scale)
        if scale.numel() != num_experts:
            raise ValueError(
                f"b12x NVFP4 MoE expected {num_experts} {name} values, "
                f"got {scale.numel()}"
            )
        return scale.to(device=device)

    def _swiglu_params(
        self,
        activation: MoEActivation,
    ) -> tuple[float | None, float | None, float | None]:
        if activation in (
            MoEActivation.SITU,
            MoEActivation.RELU2,
            MoEActivation.RELU2_NO_MUL,
        ):
            return None, None, None

        limit = self.quant_config.gemm1_clamp_limit
        if limit is None:
            limit = self.moe_config.swiglu_limit
        if activation != MoEActivation.SWIGLUOAI_UNINTERLEAVE:
            return limit, None, None

        alpha = self.quant_config.gemm1_alpha
        if alpha is None:
            alpha = self.moe_config.swiglu_alpha
        beta = self.quant_config.gemm1_beta
        if beta is None:
            beta = self.moe_config.swiglu_beta
        return limit, alpha, beta

    def _prepare_experts(
        self,
        *,
        w1: torch.Tensor,
        w2: torch.Tensor,
        activation: MoEActivation,
        params_dtype: torch.dtype,
    ) -> Any:
        quant_mode = self._quant_mode
        if _is_current_stream_capturing():
            raise RuntimeError(
                "b12x MoE weights must be prepared before CUDA graph capture"
            )
        if self.w1_scale is None or self.w2_scale is None:
            raise ValueError("b12x MoE requires w1 and w2 block scales")

        fused_moe = _require_b12x_fused_moe()

        num_experts = int(w1.shape[0])
        hidden_size = int(w2.shape[1])
        intermediate_size = int(w2.shape[2]) * 2
        unit_scale = self._unit_scale(w1.device, num_experts)
        w1_global_scale = self._weight_global_scale(
            w1.device, num_experts, self.g1_alphas, "w1 global scales"
        )
        w2_global_scale = self._weight_global_scale(
            w2.device, num_experts, self.g2_alphas, "w2 global scales"
        )

        if quant_mode in ("nvfp4", "w4a8_nvfp4"):
            if self.a1_gscale is None or self.a2_gscale is None:
                raise ValueError("b12x NVFP4 MoE requires activation global scales")
            a1_gscale = _normalize_expert_scale(self.a1_gscale).to(w1.device)
            a2_gscale = _normalize_expert_scale(self.a2_gscale).to(w2.device)
        else:
            a1_gscale = unit_scale
            a2_gscale = unit_scale

        limit, alpha, beta = self._swiglu_params(activation)
        mode = {
            "w4a16": fused_moe.ActivationMode.A16,
            "w4a8_mx": fused_moe.ActivationMode.A8,
            "w4a8_nvfp4": fused_moe.ActivationMode.A8,
            "nvfp4": fused_moe.ActivationMode.A4,
        }[quant_mode]
        weight_plan = fused_moe.plan_weights(
            source=fused_moe.PackedSource(
                format=fused_moe.PackedSourceFormat(self._source_format),
                w13_layout=fused_moe.W13Layout(self._w13_layout),
            ),
            activation=fused_moe.ActivationSpec(
                mode=mode,
                nonlinearity=_b12x_activation_name(activation),
                io_dtype=params_dtype,
                swiglu_limit=limit,
                swiglu_alpha=alpha,
                swiglu_beta=beta,
            ),
            geometry=fused_moe.MoEGeometry(
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
            ),
        )
        return fused_moe.prepare_weights(
            plan=weight_plan,
            weights=fused_moe.PackedWeights(
                w13=w1,
                w2=w2,
                w13_block_scales=self.w1_scale,
                w2_block_scales=self.w2_scale,
                w13_global_scales=w1_global_scale,
                w2_global_scales=w2_global_scale,
                input_scale=a1_gscale,
                intermediate_scale=a2_gscale,
            ),
        )

    def _refresh_quant_config(self, layer: torch.nn.Module) -> None:
        self.quant_config._w1.scale = layer.w13_weight_scale
        self.quant_config._w2.scale = layer.w2_weight_scale
        if self._source_format != "modelopt_nvfp4":
            return

        self.quant_config._w1.alpha_or_gscale = layer.w13_weight_scale_2
        self.quant_config._w2.alpha_or_gscale = layer.w2_weight_scale_2
        if self._quant_mode in ("nvfp4", "w4a8_nvfp4"):
            self.quant_config._a1.alpha_or_gscale = 1.0 / layer.w13_input_scale
            self.quant_config._a2.alpha_or_gscale = 1.0 / layer.w2_input_scale

    def _release_source_parameters(self, layer: torch.nn.Module) -> None:
        if self._source_parameters_released:
            return
        w1_scale = _replace_parameter_with_empty(layer, "w13_weight_scale")
        w2_scale = _replace_parameter_with_empty(layer, "w2_weight_scale")
        if w1_scale is not None:
            self.quant_config._w1.scale = w1_scale
        if w2_scale is not None:
            self.quant_config._w2.scale = w2_scale
        _replace_parameter_with_empty(layer, "w13_weight")
        _replace_parameter_with_empty(layer, "w2_weight")
        self._source_parameters_released = True

    def _reuse_prepared_storage(self, layer: torch.nn.Module, prepared: Any) -> Any:
        previous = getattr(layer, "_b12x_prepared_experts", None)
        prepared = reuse_packed_weight_storage(previous, prepared)
        if prepared is not previous:
            self._plan = None
            self._plan_key = None
        self._prepared_experts = prepared
        layer._b12x_prepared_experts = prepared
        return prepared

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self._apply_router_weight_on_input = layer.apply_router_weight_on_input
        if self._apply_router_weight_on_input and self._quant_mode != "w4a16":
            raise ValueError(
                "b12x MoE supports apply_router_weight_on_input only with W4A16"
            )
        self._source_parameters_released = False
        self._refresh_quant_config(layer)
        prepared = self._prepare_experts(
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            activation=layer.activation,
            params_dtype=self.moe_config.in_dtype,
        )
        prepared = self._reuse_prepared_storage(layer, prepared)
        if prepared.plan._impl.discards_source_parameters:
            self._release_source_parameters(layer)
        if not getattr(layer, "b12x_preparation_suppressed", False):
            set_b12x_preparation_provider(layer, self)
        _register_b12x_moe_output_collective(
            layer, hidden_size=int(prepared.hidden_size)
        )

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: mk.FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        if moe_config.has_bias:
            return False, "kernel does not support expert biases"
        if moe_config.in_dtype not in (torch.float16, torch.bfloat16):
            return (
                False,
                f"kernel does not support {moe_config.in_dtype} input/output dtype",
            )
        if moe_config.activation == MoEActivation.SITU and (
            moe_config.activation_situ_beta != 4.0
            or moe_config.activation_situ_linear_beta != 25.0
        ):
            return False, "kernel supports only SiTU beta=4 and linear_beta=25"
        if (
            activation_key is not None
            and moe_config.activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE
        ):
            return (
                False,
                "kernel does not support swigluoai_uninterleave with W4A8",
            )
        unpadded_intermediate_size = (
            moe_config.intermediate_size_per_partition_unpadded
            or moe_config.intermediate_size_per_partition
        )
        if weight_key == kMxfp4Static and unpadded_intermediate_size % 32 != 0:
            return (
                False,
                "MXFP4 requires the per-rank intermediate size to be divisible by 32",
            )
        if weight_key == kMxfp4Static and activation_key == kMxfp8Dynamic:
            if moe_config.activation not in (
                MoEActivation.SILU,
                MoEActivation.SITU,
            ):
                return False, "MXFP4 W4A8 supports only SiLU and SiTU"
            if (
                moe_config.hidden_dim % 256 != 0
                or moe_config.intermediate_size_per_partition % 32 != 0
            ):
                return (
                    False,
                    (
                        "MXFP4 W4A8 requires hidden size divisible by 256 and "
                        "per-rank intermediate size divisible by 32"
                    ),
                )
        return mk.FusedMoEExperts.is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )

    @staticmethod
    def _supports_current_device() -> bool:
        if not (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(120)
        ):
            return False
        fused_moe = get_b12x_fused_moe()
        if fused_moe is None:
            return False
        return fused_moe.is_supported()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) in (
            (kMxfp4Static, kMxfp8Dynamic),
            (kMxfp4Static, None),
            (kNvfp4Static, kNvfp4Dynamic),
            (kNvfp4Static, kMxfp8Dynamic),
            (kNvfp4Static, None),
        )

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (
            MoEActivation.SILU,
            MoEActivation.SITU,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            MoEActivation.RELU2_NO_MUL,
        )

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return (
            not moe_parallel_config.use_ep
            and moe_parallel_config.ep_size == 1
            and not moe_parallel_config.use_all2all_kernels
            and not moe_parallel_config.enable_eplb
        )

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def _prepared(self) -> Any:
        if self._prepared_experts is None:
            raise RuntimeError(
                "b12x MoE weights must be prepared by process_weights_after_loading"
            )
        return self._prepared_experts

    def _prepared_plan(
        self, *, activation: MoEActivation, apply_router_weight_on_input: bool
    ) -> Any:
        plan = self._plan
        if (
            plan is None
            or activation != self._plan_activation
            or bool(apply_router_weight_on_input) != self._plan_route_on_input
        ):
            raise PreparationResourceUnavailableError(
                "b12x MoE has no prepared plan for this activation/routing"
            )
        return plan

    def _plan_for_tokens(
        self, tokens: int, *, activation: MoEActivation, apply_router_weight_on_input: bool
    ) -> Any:
        """Reuse the declared prefill capacity and exact graph variants."""
        plan = self._prepared_plan(
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
        assert self._plan_key is not None
        capacity = max(self._plan_key[0])
        if int(tokens) > capacity:
            raise ValueError(
                f"live MoE token count {tokens} exceeds the configured prefill "
                f"capacity {capacity}"
            )
        return plan

    def get_b12x_preparation_units(
        self, layer: torch.nn.Module, workload: B12xWorkload
    ) -> Sequence[B12xPreparationUnit]:
        if workload.stage != "weights":
            return ()
        if workload.output_dtype != self.moe_config.in_dtype:
            raise ValueError("b12x MoE output dtype differs from its loaded contract")
        prepared = self._prepared()
        counts = tuple(sorted({workload.max_tokens, *workload.fixed_token_counts}))
        activation = layer.activation
        topk = int(self.moe_config.experts_per_token)
        route_on_input = bool(layer.apply_router_weight_on_input)
        fused_moe = _require_b12x_fused_moe()
        plan_key = (counts, activation, route_on_input, id(prepared))
        plan = self._plan if getattr(self, "_plan_key", None) == plan_key else None
        if plan is None:
            # The layer holds one plan for its serving shapes; a later call with
            # the same workload reuses it so the prepared state stays installed.
            plan = fused_moe.plan_execution(
                experts=prepared,
                capacity=fused_moe.ExecutionCapacity(
                    max_tokens=max(counts),
                    top_k=topk,
                    warmup_token_counts=counts,
                    route_num_experts=0,
                ),
                routing=fused_moe.RoutingSpec(
                    apply_router_weight_on_input=route_on_input,
                ),
            )
            self._plan = plan
            self._plan_key = plan_key
            self._plan_activation = activation
            self._plan_route_on_input = route_on_input
        name = f"fused_moe:{id(layer)}"
        if hasattr(plan, "token_counts"):
            calls = {
                count: _prepared_moe_call_factory(
                    tokens=count,
                    topk=topk,
                    prepared=prepared,
                    output_dtype=self.output_dtype,
                )
                for count in plan.token_counts
            }
            benchmark_calls = {
                count: _prepared_moe_call_factory(
                    tokens=count,
                    topk=topk,
                    prepared=prepared,
                    output_dtype=self.output_dtype,
                )
                for count in plan.token_counts
            }
            request = plan.request(
                name=name,
                prepare_calls=calls,
                benchmark_calls=benchmark_calls,
            )
        else:
            prepare_call = _prepared_moe_call_factory(
                tokens=counts[0],
                topk=topk,
                prepared=prepared,
                output_dtype=self.output_dtype,
            )
            benchmark_call = _prepared_moe_call_factory(
                tokens=counts[0],
                topk=topk,
                prepared=prepared,
                output_dtype=self.output_dtype,
            )
            request = plan.request(
                name=name,
                prepare_call=prepare_call,
                benchmark_call=benchmark_call,
            )
        key = (
            self._quant_mode, self._source_format, self._w13_layout,
            activation, route_on_input, counts, workload.output_dtype,
        )
        return (
            B12xPreparationUnit(
                name=self._quant_mode.upper(),
                key=key,
                requests=(request,),
                stage="weights",
                autotune=not workload.eager_only,
            ),
        )

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        if w1.numel() and w2.numel():
            return super().moe_problem_size(a1, w1, w2, topk_ids)
        prepared = self._prepared()
        tokens = int(a1.shape[0] if a1.ndim == 2 else a1.shape[1])
        return (
            int(prepared.num_experts),
            tokens,
            int(prepared.intermediate_size) * 2,
            int(a1.shape[-1]),
            int(topk_ids.shape[1]),
        )

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        del N, global_num_experts, local_num_experts, expert_tokens_meta
        plan = self._plan_for_tokens(
            int(M),
            activation=activation,
            apply_router_weight_on_input=self._apply_router_weight_on_input,
        )
        required_nbytes = sum(spec.nbytes for spec in plan.scratch_specs())
        itemsize = self.moe_config.in_dtype.itemsize
        return (0,), (max(1, (required_nbytes + itemsize - 1) // itemsize),), (M, K)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ) -> None:
        del w1, w2, global_num_experts
        del a1q_scale, a2_scale, workspace13, expert_tokens_meta
        if expert_map is not None:
            raise ValueError("b12x TP MoE does not support expert maps")
        if bool(apply_router_weight_on_input) != self._apply_router_weight_on_input:
            raise ValueError(
                "apply_router_weight_on_input does not match the prepared b12x MoE plan"
            )
        prepared = self._prepared()
        # Native routes are specialized for both int32 and int64 identifiers
        # during materialization.  Preserve the caller's representation: a
        # conversion here would allocate during capture and would silently
        # discard the prepared int64 path.
        if topk_ids.dtype not in (torch.int32, torch.int64) or not topk_ids.is_contiguous():
            raise TypeError("b12x MoE topk_ids must be contiguous int32 or int64")
        topk_weights = _normalize_topk_weights(topk_weights)
        plan = self._plan_for_tokens(
            int(hidden_states.shape[0]),
            activation=activation,
            apply_router_weight_on_input=bool(apply_router_weight_on_input),
        )
        if workspace2 is None or not workspace2.is_contiguous():
            raise ValueError("b12x MoE requires contiguous caller-owned workspace2")
        scratch = workspace2.view(-1).view(torch.uint8)
        binding = _require_b12x_fused_moe().bind(
            plan,
            scratch=scratch,
            a=hidden_states,
            experts=prepared,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            output=output,
            input_scales_static=True,
        )
        _require_b12x_fused_moe().run(binding=binding)

    def moe_sum(self, input: torch.Tensor, output: torch.Tensor) -> None:
        raise NotImplementedError("LoRA is not supported for B12xExperts")

def _register_b12x_moe_output_collective(
    layer: torch.nn.Module, *, hidden_size: int
) -> None:
    """Describe the rank-local routed-MoE output to the existing TP transport."""
    from vllm.distributed.parallel_state import register_b12x_collective_describer
    from vllm.distributed.device_communicators.b12x_pcie_all_reduce import (
        B12xPcieInvocation,
    )


    def describe(workload):
        prefix = layer.layer_name
        return tuple(
            B12xPcieInvocation(
                name=f"{prefix}.moe_output_all_reduce.m{rows}.lane{workload.lane}",
                operation="all_reduce",
                shape=(rows, hidden_size),
                dtype=workload.output_dtype,
            )
            for rows in workload.token_counts
        )

    register_b12x_collective_describer(layer, describe)
