# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lossless NVFP4 expert scale storage with native ModelOpt arithmetic."""

import itertools
from dataclasses import dataclass, field, replace

import regex as re
import torch

import vllm.envs as envs
from vllm.config import get_current_vllm_config_or_none
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.b12x import (
    B12xExperts,
    _is_current_stream_capturing,
)
from vllm.model_executor.layers.fused_moe.config import (
    nvfp4_moe_quant_config,
    nvfp4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import set_default_torch_num_threads

logger = init_logger(__name__)

_FORWARD_TOKENS = itertools.count()


def _forward_token() -> int | None:
    """An id of the current forward pass, or None outside of one."""
    from vllm.forward_context import get_forward_context, is_forward_context_available

    if not is_forward_context_available():
        return None
    context = get_forward_context()
    token = context.additional_kwargs.get("b12x_csf_token")
    if token is None:
        token = next(_FORWARD_TOKENS)
        context.additional_kwargs["b12x_csf_token"] = token
    return token


def _stage_max_tokens() -> int:
    """Calls up to this size read compressed scales per stage, without expansion."""
    try:
        from b12x.moe.fused_moe._impl import W4A16_CSF_STAGE_MAX_TOKENS
    except ImportError:
        return 1536
    return W4A16_CSF_STAGE_MAX_TOKENS


@dataclass
class Nvfp4CsfState:
    scale_scratch: tuple[torch.Tensor, ...] | None = None
    scale_layers: dict[int, "Nvfp4CsfMoEMethod"] = field(default_factory=dict)
    scale_stream: torch.cuda.Stream | None = None
    scale_prefetch: tuple[int, int, torch.cuda.Event] | None = None


class Nvfp4CsfMoEMethod(FusedMoEMethodBase):
    """Decode selected scale planes into model-owned native E4M3 scratch."""

    def __init__(self, moe, owner, *, use_a16=False):
        super().__init__(moe)
        self.owner = owner
        self.use_a16 = use_a16 or bool(envs.VLLM_B12X_MOE_FP4_FORCE_A16)
        config = get_current_vllm_config_or_none()
        if config is not None and (
            config.parallel_config.pipeline_parallel_size != 1
            or config.parallel_config.use_ubatching
        ):
            raise NotImplementedError(
                "CSF shared scratch requires PP1 without ubatching"
            )
        parallel = moe.moe_parallel_config
        if (
            parallel.use_ep
            or parallel.ep_size != 1
            or parallel.dp_size != 1
            or parallel.use_all2all_kernels
            or parallel.enable_eplb
        ):
            raise NotImplementedError("NVFP4-CSF supports TP without EP/DP")
        if (
            moe.activation != MoEActivation.SILU
            or moe.in_dtype != torch.bfloat16
            or moe.has_bias
        ):
            raise ValueError("NVFP4-CSF requires bias-free BF16 SwiGLU experts")

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        match = re.search(r"(?:^|\.)layers\.(\d+)\.", layer.layer_name)
        if match is None or params_dtype != torch.bfloat16:
            raise ValueError("NVFP4-CSF requires a numbered BF16 expert layer")
        self.layer_index = int(match.group(1))
        self.num_experts, self.hidden_size = num_experts, hidden_size
        self.local_intermediate = intermediate_size_per_partition
        # A TP-padded expert (GLM-5.3 at TP6) keeps its checkpoint width.
        vllm_config = get_current_vllm_config_or_none()
        model_config = getattr(vllm_config, "model_config", None)
        self.checkpoint_width = getattr(
            getattr(model_config, "hf_text_config", None),
            "original_moe_intermediate_size",
            None,
        )
        self._components = {}
        self._scale_parts = {}
        self._seen = set()
        for prefix in ("w13", "w2"):
            for component, dtype in (
                ("weight", torch.uint8),
                ("weight_scale_2", torch.float32),
                ("input_scale", torch.float32),
            ):
                param = torch.nn.Parameter(
                    torch.empty(0, dtype=dtype), requires_grad=False
                )
                param.weight_loader = self.weight_loader
                param.csf_component = component
                layer.register_parameter(f"{prefix}_{component}", param)
            scales = torch.nn.Module()
            for component, dtype in (
                ("fixed", torch.uint8),
                ("exceptions", torch.uint32),
            ):
                param = torch.nn.Parameter(
                    torch.empty(0, dtype=dtype), requires_grad=False
                )
                param.weight_loader = self.weight_loader
                param.csf_component = component
                scales.register_parameter(f"nvfp4_csf_{component}", param)
            layer.add_module(f"{prefix}_weight_scale", scales)

    def weight_loader(
        self,
        param,
        loaded_weight,
        weight_name=None,
        shard_id=None,
        expert_id=None,
        return_success=False,
    ):
        from .utils.nvfp4_csf_utils import _slice_scale_plane

        if (
            shard_id not in ("w1", "w2", "w3")
            or expert_id is None
            or not 0 <= expert_id < self.num_experts
        ):
            raise ValueError("CSF tensor requires a valid expert and projection")
        field_name = param.csf_component
        key = (expert_id, shard_id)
        identity = (*key, field_name)
        if identity in self._seen:
            raise ValueError(f"Duplicate CSF tensor: {identity}")
        self._seen.add(identity)
        width = self.checkpoint_width or self.local_intermediate * self.moe.tp_size
        first = self.moe.tp_rank * self.local_intermediate
        last = min(first + self.local_intermediate, width)
        if first >= last:
            raise ValueError("CSF TP rank has no checkpoint channels")
        h, n = self.hidden_size, self.local_intermediate
        values = self._components.setdefault(key, {})
        if field_name == "weight":
            expected = (h, width // 2) if shard_id == "w2" else (width, h // 2)
            if (
                tuple(loaded_weight.shape) != expected
                or loaded_weight.dtype != torch.uint8
            ):
                raise ValueError(f"Invalid NVFP4 weight shape or dtype: {weight_name}")
            shape = (h, n // 2) if shard_id == "w2" else (n, h // 2)
            local = torch.zeros(shape, dtype=torch.uint8, device="cpu")
            if shard_id == "w2":
                local[:, : (last - first) // 2].copy_(
                    loaded_weight[:, first // 2 : last // 2].cpu()
                )
            else:
                local[: last - first].copy_(loaded_weight[first:last].cpu())
            values[field_name] = local
        elif field_name in ("weight_scale_2", "input_scale"):
            if loaded_weight.numel() != 1 or loaded_weight.dtype != torch.float32:
                raise ValueError(f"CSF {field_name} must contain one FP32 value")
            values[field_name] = loaded_weight.detach().cpu().clone()
        else:
            parts = self._scale_parts.setdefault(key, {})
            parts[field_name] = loaded_weight.detach().cpu()
            if len(parts) == 2:
                if shard_id == "w2":
                    rows, columns = h, width // 16
                    row_slice, column_slice = (0, h), (first // 16, last // 16)
                    out_rows, out_columns = h, n // 16
                else:
                    rows, columns = width, h // 16
                    row_slice, column_slice = (first, last), (0, columns)
                    out_rows, out_columns = n, columns
                fixed, exceptions = _slice_scale_plane(
                    parts["fixed"],
                    parts["exceptions"],
                    rows,
                    columns,
                    row_slice,
                    column_slice,
                    out_rows,
                    out_columns,
                )
                values["fixed"] = torch.from_numpy(fixed)
                values["exceptions"] = torch.from_numpy(exceptions)
                del self._scale_parts[key]
        return True if return_success else None

    def _expert_tensors(self):
        from vllm.model_executor.model_loader.csf_utils import CsfMatrix

        from .utils.nvfp4_csf_utils import TensorView

        for expert in range(self.num_experts):
            projections = []
            for shard in ("w3", "w1", "w2"):
                values = self._components.get((expert, shard), {})
                required = {"weight", "fixed", "exceptions", "weight_scale_2"}
                if not self.use_a16:
                    required.add("input_scale")
                missing = required - values.keys()
                if missing:
                    raise ValueError(
                        f"Missing CSF tensors for expert {expert}/{shard}: "
                        f"{sorted(missing)}"
                    )
                projections.append(
                    CsfMatrix(
                        weight=TensorView(values["weight"]),
                        fixed=values["fixed"],
                        exceptions=values["exceptions"],
                        global_scale=values["weight_scale_2"],
                        input_scale=values.get(
                            "input_scale",
                            torch.ones((), dtype=torch.float32, device="cpu"),
                        ),
                    )
                )
            yield tuple(projections)

    def get_fused_moe_quant_config(self, layer):
        return self.moe_quant_config

    def process_weights_after_loading(self, layer):
        from b12x.moe import fused_moe

        from .utils.nvfp4_csf_utils import prepare_nvfp4_csf_weights

        device = layer.w13_weight.device
        e, h, n = self.num_experts, self.hidden_size, self.local_intermediate
        # Native E4M3 scale storage: the F8_128x4 grid (rows in 128s, k-groups
        # in 4s), which TP-padded shards (GLM-5.3 TP6: 352 channels) fill.
        shapes = (
            (e, round_up(2 * n, 128), h // 16),
            (e, h, round_up(n // 16, 4)),
        )
        if self.owner.scale_scratch is None:
            self.owner.scale_scratch = tuple(
                torch.empty(s, dtype=torch.float8_e4m3fn, device=device) for s in shapes
            )
        scratch = self.owner.scale_scratch
        if any(
            tuple(t.shape) != s or t.device != device for t, s in zip(scratch, shapes)
        ):
            raise ValueError("NVFP4-CSF shared scratch geometry/device mismatch")
        with set_default_torch_num_threads(1):
            weights = prepare_nvfp4_csf_weights(
                self._expert_tensors(),
                num_experts=e,
                hidden_size=h,
                intermediate_size=n,
                tp_rank=0,
                tp_size=1,
                device=device,
                w13_scale_scratch=scratch[0],
                w2_scale_scratch=scratch[1],
                local_size=n,
            )
        self._components.clear()
        self._scale_parts.clear()
        self._seen.clear()
        packed = weights.packed
        layer_max = envs.VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE
        packed = replace(
            packed,
            input_scale=(
                packed.input_scale.amin()
                if layer_max in ("1", "all", "w13")
                else packed.input_scale
            ),
            intermediate_scale=(
                packed.intermediate_scale.amin()
                if layer_max in ("1", "all", "w2")
                else packed.intermediate_scale
            ),
        )
        weights = replace(weights, packed=packed)
        plan = fused_moe.plan_weights(
            source=fused_moe.PackedSource(format="modelopt_nvfp4", w13_layout="w13"),
            activation=fused_moe.ActivationSpec(
                mode="a16" if self.use_a16 else "a4",
                nonlinearity="silu",
                io_dtype=torch.bfloat16,
                swiglu_limit=self.moe.swiglu_limit,
            ),
            geometry=fused_moe.MoEGeometry(
                num_experts=e, hidden_size=h, intermediate_size=n
            ),
        )
        prepared = fused_moe.prepare_weights(plan=plan, weights=weights)
        quant_builder = (
            nvfp4_w4a16_moe_quant_config if self.use_a16 else nvfp4_moe_quant_config
        )
        self.moe_quant_config = quant_builder(
            g1_alphas=packed.w13_global_scales,
            g2_alphas=packed.w2_global_scales,
            **(
                {}
                if self.use_a16
                else {
                    "a1_gscale": packed.input_scale,
                    "a2_gscale": packed.intermediate_scale,
                }
            ),
            w1_scale=scratch[0],
            w2_scale=scratch[1],
            gemm1_clamp_limit=self.moe.swiglu_limit,
        )
        backend = B12xExperts(self.moe, self.moe_quant_config)
        backend.install_prepared_experts(layer, prepared)
        self.moe_kernel = mk.FusedMoEKernel(
            MoEPrepareAndFinalizeNoDPEPModular(), backend
        )
        self.backend, self.prepared = backend, prepared
        self.moe_done = torch.cuda.Event()
        self.scales_ready = torch.cuda.Event()
        if self.use_a16 and envs.VLLM_B12X_CSF_SCALE_PREFETCH:
            self.owner.scale_layers[self.layer_index] = self
            if self.owner.scale_stream is None:
                self.owner.scale_stream = torch.cuda.Stream(device)
        logger.info(
            "NVFP4-CSF layer %d rank %d/%d: native NVFP4 %s, "
            "shared scale scratch %d bytes",
            self.layer_index,
            self.moe.tp_rank,
            self.moe.tp_size,
            "A16" if self.use_a16 else "A4",
            sum(t.numel() for t in scratch),
        )

    def apply(
        self,
        layer,
        x,
        topk_weights,
        topk_ids,
        shared_experts,
        shared_experts_input,
        workspace=None,
    ):
        moe_kernel = self.moe_kernel
        assert moe_kernel is not None

        def run():
            return moe_kernel.apply(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=layer.activation,
                global_num_experts=layer.global_num_experts,
                expert_map=layer.expert_map,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                shared_experts=shared_experts,
                shared_experts_input=shared_experts_input,
                workspace=workspace,
            )

        owner = self.owner
        if self.layer_index not in owner.scale_layers or _is_current_stream_capturing():
            return run()
        token = _forward_token()
        pending, owner.scale_prefetch = owner.scale_prefetch, None
        stream = torch.cuda.current_stream()
        if pending is not None:
            # The shared scratch must never race an expansion still in flight.
            stream.wait_event(pending[2])
        self.backend.scales_expanded = (
            token is not None
            and pending is not None
            and (pending[:2] == (self.layer_index, token))
        )
        try:
            result = run()
        finally:
            self.backend.scales_expanded = False
        following = owner.scale_layers.get(self.layer_index + 1)
        if token is not None and following is not None and self._expands(x):
            # The next layer's MoE would expand its scales into the same scratch:
            # do it now on a side stream, overlapping that layer's attention.
            from b12x.moe import fused_moe

            self.moe_done.record(stream)
            side = owner.scale_stream
            side.wait_event(self.moe_done)
            with torch.cuda.stream(side):
                fused_moe.expand_scales(following.prepared)
                following.scales_ready.record(side)
            owner.scale_prefetch = (
                following.layer_index,
                token,
                following.scales_ready,
            )
        return result

    def _expands(self, x: torch.Tensor) -> bool:
        """Whether this step's call expands scales: smaller A16 calls read
        compressed scales per stage."""
        return int(x.shape[0]) > _stage_max_tokens()
