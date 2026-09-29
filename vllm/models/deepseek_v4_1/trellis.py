# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent two-bit routed experts with native DeepSeek V4.1 tensors."""

import re

import torch

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
    FusedMoEQuantConfig,
    FusedMoEQuantDesc,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

from .quant_config import DeepseekV41FP8Config

logger = init_logger(__name__)


class DeepseekV41TrellisConfig(DeepseekV41FP8Config):
    """Trellis main experts; native FP8/BF16 dense and FP4 draft weights."""

    @classmethod
    def get_name(cls):
        return "trellis_dense"

    @classmethod
    def get_min_capability(cls):
        return 120

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        if (
            user_quant in (None, cls.get_name())
            and hf_quant_cfg is not None
            and hf_quant_cfg.get("quant_method") == cls.get_name()
            and getattr(hf_config, "model_type", None)
            in ("deepseek_v41", "deepseek_v41_text")
        ):
            return cls.get_name()
        return None

    @classmethod
    def from_config(cls, config):
        required = {
            "format_version": 1,
            "manifest": "trellis-manifest.json",
            "bits": 2,
            "codebook": "lut_e4m3",
        }
        if any(config.get(key) != value for key, value in required.items()):
            raise ValueError("DS4.1 requires the independent K2 trellis manifest")
        # This container preserves the source's UE8M0 block32 dense tensors.
        return cls(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[32, 32],
        )

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, RoutedExperts):
            match = re.search(r"(?:^|\.)layers\.(\d+)\.", prefix)
            if match is None:
                raise ValueError("DS4.1 routed experts require a numbered layer")
            config = get_current_vllm_config().model_config.hf_config
            if int(match.group(1)) < config.num_hidden_layers:
                return IndependentTrellisMoEMethod(layer.moe_config)
        return super().get_quant_method(layer, prefix)


class IndependentTrellisMoEMethod(FusedMoEMethodBase):
    """Load each TP extent directly into B12X's compressed MoE executor."""

    def __init__(self, moe):
        super().__init__(moe)
        parallel = moe.moe_parallel_config
        if (
            parallel.use_ep
            or parallel.ep_size != 1
            or parallel.dp_size != 1
            or parallel.use_all2all_kernels
            or parallel.enable_eplb
        ):
            raise NotImplementedError("independent trellis supports TP without EP/DP")
        if (
            moe.activation != MoEActivation.SILU
            or moe.in_dtype != torch.bfloat16
            or moe.has_bias
        ):
            raise ValueError("DS4.1 trellis requires bias-free BF16 SwiGLU experts")

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        if params_dtype != torch.bfloat16:
            raise ValueError("DS4.1 trellis requires BF16 activations")
        match = re.search(r"(?:^|\.)layers\.(\d+)\.", layer.layer_name)
        if match is None:
            raise ValueError("DS4.1 trellis requires a numbered layer")
        self.layer_index = int(match.group(1))
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.local_intermediate = intermediate_size_per_partition
        # The adapter loads expert payloads separately. Empty nonpersistent
        # handles satisfy the modular executor without registering fake weights.
        for name in ("w13_weight", "w2_weight"):
            layer.register_buffer(
                name, torch.empty(0, dtype=torch.uint8), persistent=False
            )

    def get_fused_moe_quant_config(self, layer):
        return FusedMoEQuantConfig(
            _a1=FusedMoEQuantDesc(),
            _a2=FusedMoEQuantDesc(),
            _w1=FusedMoEQuantDesc(dtype="trellis_dense"),
            _w2=FusedMoEQuantDesc(dtype="trellis_dense"),
        )

    def process_weights_after_loading(self, layer):
        from b12x.moe import fused_moe
        from b12x.moe.checkpoints.independent import read_independent_layer

        config = get_current_vllm_config()
        tp = get_tensor_model_parallel_world_size()
        rank = get_tensor_model_parallel_rank()
        source, weights = read_independent_layer(
            config.model_config.model,
            self.layer_index,
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
            intermediate_size=self.local_intermediate * tp,
            tp_rank=rank,
            tp_size=tp,
        )
        plan = fused_moe.plan_weights(
            source=source,
            activation=fused_moe.ActivationSpec(
                mode="a16",
                nonlinearity="silu",
                io_dtype=torch.bfloat16,
                rotation_dtype=torch.float16,
                swiglu_limit=self.moe.swiglu_limit,
            ),
            geometry=fused_moe.MoEGeometry(
                num_experts=self.num_experts,
                hidden_size=self.hidden_size,
                intermediate_size=self.local_intermediate,
            ),
        )
        prepared = fused_moe.prepare_weights(
            plan=plan,
            weights=weights,
            device=layer.w13_weight.device,
        )
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        backend = B12xExperts(self.moe, self.moe_quant_config)
        backend.install_prepared_experts(layer, prepared)
        self.moe_kernel = mk.FusedMoEKernel(
            MoEPrepareAndFinalizeNoDPEPModular(),
            backend,
        )
        logger.info(
            "Independent K2 trellis layer %d rank %d/%d: %d intermediate channels, "
            "BF16 inputs, H128/FP16 transforms, SwiGLU limit %s",
            self.layer_index,
            rank,
            tp,
            self.local_intermediate,
            self.moe.swiglu_limit,
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
