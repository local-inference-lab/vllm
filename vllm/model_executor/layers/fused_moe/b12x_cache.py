# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in CPU expert loading for the prepared b12x SM120 cache."""

import torch

from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4FusedMoE
from vllm.utils.b12x import B12xPreparationUnit, register_b12x_unit_provider


class _CacheProvider:
    def __init__(self, config):
        from b12x.integration.vllm.expert_cache import (
            ExpertCacheModel,
            ExpertCacheServingConfig,
        )
        from b12x.moe.fused_moe.cache_source import checkpoint_fingerprint

        if (
            config.model_config.quantization != "modelopt_fp4"
            or config.model_config.dtype != torch.bfloat16
            or config.kernel_config.moe_backend != "b12x"
        ):
            raise ValueError(
                "expert cache requires ModelOpt NVFP4, BF16 and the b12x MoE backend"
            )
        parallel = config.parallel_config
        if (
            parallel.tensor_parallel_size != 1
            or parallel.data_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
            or parallel.enable_expert_parallel
            or parallel.enable_dbo
            or config.speculative_config is not None
            or config.lora_config is not None
        ):
            raise ValueError(
                "experimental expert-cache loader requires TP/DP/PP=1 "
                "without EP, DBO, speculation or LoRA"
            )
        settings = ExpertCacheServingConfig(
            **config.additional_config["b12x_expert_cache"]
        )
        if torch.cuda.get_device_capability() != (12, 0):
            raise ValueError(
                "experimental canonical expert-cache loader requires SM120"
            )
        self.model = ExpertCacheModel(
            settings,
            checkpoint_fingerprint(config.model_config.model),
            torch.device("cuda", torch.accelerator.current_device_index()),
        )
        self.top_k = None
        register_b12x_unit_provider(self)

    def get_b12x_preparation_units(self, owner, workload):
        from b12x.moe import fused_moe as moe

        if workload.stage != "weights":
            return ()
        if (
            workload.lane
            or workload.eager_only
            or workload.output_dtype != torch.bfloat16
        ):
            raise ValueError("expert cache requires one BF16 target execution lane")
        self.model.declare(
            moe.ExecutionCapacity(max_tokens=workload.max_tokens, top_k=self.top_k)
        )
        return (
            B12xPreparationUnit(
                name="EXPERT_CACHE",
                key=("sm120_canonical_w4a16",),
                requests=self.model.requests(),
                stage="weights",
                autotune=False,
            ),
        )


def cache_provider(config=None, *, create=False):
    config = config or get_current_vllm_config()
    if not isinstance(config.additional_config, dict):
        return None
    settings = config.additional_config.get("b12x_expert_cache")
    if settings is None:
        return None
    owner = getattr(config, "_b12x_expert_cache_provider", None)
    if owner is None and create:
        owner = _CacheProvider(config)
        object.__setattr__(config, "_b12x_expert_cache_provider", owner)
    return owner


class ModelOptNvFp4CacheMoE(ModelOptNvFp4FusedMoE):
    """Use the existing checkpoint loader and its TP row/column semantics on CPU."""

    requires_device_loading = False
    supports_pre_processed_weights = False

    def __init__(self, quant_config, moe_config, prefix):
        FusedMoEMethodBase.__init__(self, moe_config)
        self.quant_config, self.prefix = quant_config, prefix
        self.use_a16 = True
        self.use_global_sf = False
        if (
            quant_config.group_size != 16
            or not moe_config.is_act_and_mul
            or moe_config.has_bias
        ):
            raise ValueError(
                "expert cache requires unbiased gated NVFP4 with K16 scales"
            )
        self.nvfp4_backend = NvFp4MoeBackend.B12X
        self.provider = cache_provider(create=True)
        if self.provider.top_k not in (None, moe_config.experts_per_token):
            raise ValueError(
                "first cache loader requires one declared routing top-k across layers"
            )
        self.provider.top_k = moe_config.experts_per_token

    @property
    def supports_eplb(self):
        return False

    def maybe_roundup_sizes(
        self,
        hidden_size,
        intermediate_size_per_partition,
        act_dtype,
        moe_parallel_config,
    ):
        if hidden_size % 128 or intermediate_size_per_partition % 128:
            raise ValueError(
                "cache source geometry requires H and local I divisible by 128"
            )
        return hidden_size, intermediate_size_per_partition

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **attrs,
    ):
        e, h, i = num_experts, hidden_size, intermediate_size_per_partition
        if params_dtype != torch.bfloat16:
            raise ValueError("expert cache requires BF16 model activations")
        # Packed weights, K16 scale grids, two global and input scale vectors.
        self.provider.model.reserve_source(e * (3 * h * i // 2 + 3 * h * i // 16 + 24))
        with torch.device("cpu"):
            super().create_weights(layer, e, h, i, params_dtype, **attrs)
        if any(p.device.type != "cpu" for p in layer.parameters(recurse=False)):
            raise RuntimeError("routed cache source was allocated on the device")

    def process_weights_after_loading(self, layer):
        from b12x.moe import fused_moe as moe

        if (
            layer.apply_router_weight_on_input
            or layer.activation.value != "silu"
            or any(
                getattr(layer, n, None) is not None
                for n in ("swiglu_limit", "swiglu_alpha", "swiglu_beta")
            )
        ):
            raise ValueError(
                "cache loader supports unmodified SiLU and output route weighting"
            )
        if not torch.equal(
            layer.w13_weight_scale_2[:, 0], layer.w13_weight_scale_2[:, 1]
        ):
            raise ValueError(
                "gate/up global scales differ; the cache loader never requantizes"
            )
        e, rows, packed_h = layer.w13_weight.shape
        plan = moe.plan_weights(
            source=moe.PackedSource(format="modelopt_nvfp4", w13_layout="w31"),
            activation=moe.ActivationSpec(
                mode="a16", nonlinearity="silu", io_dtype=torch.bfloat16
            ),
            geometry=moe.MoEGeometry(
                num_experts=e, hidden_size=packed_h * 2, intermediate_size=rows // 2
            ),
            constraints=moe.WeightPlanConstraints(required_packing="source_native"),
        )
        source = moe.ExpertWeightSource(
            plan=plan,
            weights=moe.PackedWeights(
                w13=layer.w13_weight,
                w2=layer.w2_weight,
                w13_block_scales=layer.w13_weight_scale,
                w2_block_scales=layer.w2_weight_scale,
                w13_global_scales=layer.w13_weight_scale_2[:, 0].contiguous(),
                w2_global_scales=layer.w2_weight_scale_2,
                input_scale=layer.w13_input_scale,
                intermediate_scale=layer.w2_input_scale,
                checkpoint_fingerprint=self.provider.model.checkpoint_id,
                layer_name=self.prefix,
            ),
            owners=tuple(layer.parameters(recurse=False)),
        )
        self.provider.model.add_source(source)

    def get_fused_moe_quant_config(self, layer):
        # This backend binds its numerical contract through b12x preparation.
        return None

    def apply(
        self,
        layer,
        x,
        topk_weights,
        topk_ids,
        shared_experts=None,
        shared_experts_input=None,
        workspace=None,
    ):
        from b12x.moe import fused_moe as moe

        if shared_experts is not None or workspace is not None:
            raise ValueError(
                "experimental cache backend does not own shared-expert scratch"
            )
        model = self.provider.model
        binding = moe.bind(
            model.plans[self.prefix], a=x, topk_ids=topk_ids, topk_weights=topk_weights
        )
        output = moe.run(binding=binding)
        model.observe(self.prefix, topk_ids)
        return output
