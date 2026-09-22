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
            parallel.data_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
            or parallel.enable_expert_parallel
            or parallel.enable_dbo
            or parallel.use_sequence_parallel_moe
            or parallel.decode_context_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or config.speculative_config is not None
            or config.lora_config is not None
        ):
            raise ValueError(
                "expert-cache loader requires one TP group with DP/PP=1 "
                "without EP, sequence/context parallelism, DBO, speculation or LoRA"
            )
        settings = ExpertCacheServingConfig(
            **config.additional_config["b12x_expert_cache"]
        )
        if torch.cuda.get_device_capability() != (12, 0):
            raise ValueError(
                "experimental canonical expert-cache loader requires SM120"
            )
        # Validate every target expert before layer construction can allocate
        # sources. MTP is optional draft storage, not part of this target lane.
        if getattr(config.model_config.hf_config, "model_type", None) == "qwen3_next":
            from b12x.integration.vllm.checkpoint import audit

            self.checkpoint_audit = audit(config.model_config.model, check_values=True)
            host_lower_bound = self.checkpoint_audit["host_expert_lower_bound_bytes"]
            if parallel.tensor_parallel_size > 1:
                geometry = self.checkpoint_audit["geometry"]
                local_i, remainder = divmod(
                    geometry["intermediate"], parallel.tensor_parallel_size
                )
                if remainder or local_i % 128:
                    raise ValueError(
                        "TP shard requires an integral intermediate dimension "
                        "divisible by 128"
                    )
                e, h, layers = (
                    geometry["experts"],
                    geometry["hidden"],
                    geometry["layers"],
                )
                payload = e * (3 * h * local_i // 2 + 3 * h * local_i // 16)
                # Source retains two gate/up global/input scales; canonical
                # execution needs one global scalar for each matrix pair.
                host_lower_bound = layers * (2 * payload + e * (24 + 8))
            if host_lower_bound + settings.host_safety_bytes > settings.host_bytes:
                raise ValueError(
                    "CPU expert sources plus mapped backing exceed the host envelope"
                )
        from vllm.distributed import get_tensor_model_parallel_rank

        self.model = ExpertCacheModel(
            settings,
            checkpoint_fingerprint(config.model_config.model),
            torch.device("cuda", torch.accelerator.current_device_index()),
            tp_rank=get_tensor_model_parallel_rank(),
            tp_size=parallel.tensor_parallel_size,
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
        from b12x.moe.residency import ExpertShard

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
            shard=ExpertShard(
                experts=e,
                hidden=packed_h * 2,
                intermediate=rows // 2,
                global_intermediate=rows // 2 * self.provider.model.tp_size,
                intermediate_start=rows // 2 * self.provider.model.tp_rank,
                rank=self.provider.model.tp_rank,
                world_size=self.provider.model.tp_size,
            ),
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

        # MoERunner owns external shared execution, stream ordering and output
        # combination. This method returns only routed output and never invokes
        # the shared wrapper. Its prepared routed scratch cannot be borrowed.
        if workspace is not None:
            raise ValueError(
                "experimental cache backend does not accept caller-owned scratch"
            )
        model = self.provider.model
        binding = moe.bind(
            model.plans[self.prefix], a=x, topk_ids=topk_ids, topk_weights=topk_weights
        )
        output = moe.run(binding=binding)
        model.observe(self.prefix, topk_ids)
        return output
