# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lossless NVFP4 expert scale storage with native ModelOpt arithmetic."""

from dataclasses import replace

import regex as re
import torch

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.b12x import B12xExperts
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
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.utils.torch_utils import set_default_torch_num_threads

from .modelopt import ModelOptMixedPrecisionConfig

logger = init_logger(__name__)


class Nvfp4CsfConfig(ModelOptMixedPrecisionConfig):
    """Compressed main experts; retained ModelOpt formats for all other tensors."""

    checkpoint_root: str
    scale_scratch: tuple[torch.Tensor, ...] | None

    @classmethod
    def get_name(cls):
        return "nvfp4_csf"

    @classmethod
    def get_min_capability(cls):
        return 120

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        if (hf_quant_cfg or {}).get("quant_method") == cls.get_name():
            return cls.get_name()
        return None

    @classmethod
    def from_config(cls, config):
        if config.get("format_version") != 1 or not config.get("checkpoint_root"):
            raise ValueError("NVFP4-CSF requires format_version=1 and checkpoint_root")
        original = config.get("source_quantization_config")
        if not isinstance(original, dict):
            raise ValueError("NVFP4-CSF requires source_quantization_config")
        result = super().from_config(original)
        assert isinstance(result, cls)
        result.checkpoint_root = config["checkpoint_root"]
        result.scale_scratch = None
        return result

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, RoutedExperts):
            match = re.search(r"(?:^|\.)layers\.(\d+)\.", prefix)
            if match is None:
                raise ValueError("NVFP4-CSF routed experts require a numbered layer")
            config = get_current_vllm_config()
            if (
                int(match.group(1))
                < config.model_config.hf_text_config.num_hidden_layers
            ):
                if config.model_config.hf_text_config.model_type not in (
                    "glm5_next_text",
                    "qwen3_8_flash_next_text",
                    "qwen4_exp_text",
                ):
                    raise ValueError(
                        "NVFP4-CSF supports GLM-5.3-Flash and Qwen3.8-Flash-Next"
                    )
                if (
                    config.parallel_config.pipeline_parallel_size != 1
                    or config.parallel_config.use_ubatching
                ):
                    raise NotImplementedError(
                        "NVFP4-CSF shared scratch requires PP1 without ubatching"
                    )
                if config.load_config.load_format != "nvfp4_csf":
                    raise ValueError("NVFP4-CSF requires --load-format nvfp4_csf")
                algo = self._resolve_quant_algo(prefix)
                if algo not in ("NVFP4", "W4A16_NVFP4"):
                    raise ValueError(
                        "NVFP4-CSF requires NVFP4 source expert calibration"
                    )
                return Nvfp4CsfMoEMethod(
                    layer.moe_config, self, use_a16=algo == "W4A16_NVFP4"
                )
        return super().get_quant_method(layer, prefix)


class Nvfp4CsfMoEMethod(FusedMoEMethodBase):
    """Decode selected scale planes into model-owned native E4M3 scratch."""

    def __init__(self, moe, owner, *, use_a16=False):
        super().__init__(moe)
        self.owner = owner
        self.use_a16 = use_a16 or bool(envs.VLLM_B12X_MOE_FP4_FORCE_A16)
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
        for name in ("w13_weight", "w2_weight"):
            layer.register_buffer(
                name, torch.empty(0, dtype=torch.uint8), persistent=False
            )

    def get_fused_moe_quant_config(self, layer):
        return self.moe_quant_config

    def process_weights_after_loading(self, layer):
        from b12x.moe import fused_moe
        from b12x.moe.checkpoints.nvfp4_csf import read_nvfp4_csf_layer

        tp, rank = (
            get_tensor_model_parallel_world_size(),
            get_tensor_model_parallel_rank(),
        )
        device = layer.w13_weight.device
        e, h, n = self.num_experts, self.hidden_size, self.local_intermediate
        shapes = ((e, 2 * n, h // 16), (e, h, n // 16))
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
            weights = read_nvfp4_csf_layer(
                self.owner.checkpoint_root,
                self.layer_index,
                num_experts=e,
                hidden_size=h,
                intermediate_size=n * tp,
                tp_rank=rank,
                tp_size=tp,
                device=device,
                w13_scale_scratch=scratch[0],
                w2_scale_scratch=scratch[1],
            )
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
        logger.info(
            "NVFP4-CSF layer %d rank %d/%d: native NVFP4 %s, "
            "shared scale scratch %d bytes",
            self.layer_index,
            rank,
            tp,
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
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
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
