# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in native P8 routed experts over a ModelOpt NVFP4 carrier.

The runtime is separately licensed and lazily imported. Dense, attention,
router, shared-expert and MTP tensors retain the carrier's quantization.
"""

import regex as re
import torch

from vllm import envs
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4FusedMoE
from vllm.utils.b12x import B12xWarmupUnit
from vllm.utils.trellismx import load_overlay, routed_layer

logger = init_logger(__name__)


def maybe_trellismx_method(config, layer, prefix):
    directory = envs.VLLM_TRELLISMX_CHECKPOINT
    index = routed_layer(prefix)
    if not directory:
        return None
    text_config = get_current_vllm_config().model_config.hf_text_config
    if text_config.model_type != "glm5_next_text":
        raise ValueError(
            "TrellisMX overlay currently requires the GLM5Next text adapter"
        )
    if index is None:
        if re.fullmatch(
            r"(?:model\.language_model|language_model\.model|model)"
            r"\.layers\.45\.(?:mtp_block\.)?mlp\.experts",
            prefix,
        ):
            return None
        raise ValueError(f"Unrecognized TrellisMX routed-expert prefix: {prefix}")
    if getattr(config, "quant_method", None) != "NVFP4":
        raise ValueError("TrellisMX requires the pinned ModelOpt NVFP4 carrier")
    return TrellisMXMoEMethod(config, layer.moe_config, directory, index)


class TrellisMXMoEMethod(ModelOptNvFp4FusedMoE):
    """Keep the carrier's weight-loader ABI; execute routed weights using P8."""

    def __init__(self, config, moe_config, directory, layer_index):
        # Do not select or compile an NVFP4 expert backend we never execute.
        FusedMoEMethodBase.__init__(self, moe_config)
        self.quant_config = config
        self.use_a16 = False
        self.use_global_sf = False
        parallel = moe_config.moe_parallel_config
        if (
            parallel.tp_size != 4
            or parallel.ep_size != 1
            or moe_config.hidden_dim != 4096
            or moe_config.intermediate_size_per_partition != 512
            or moe_config.num_experts != 288
            or moe_config.experts_per_token != 8
            or moe_config.has_bias
            or moe_config.is_lora_enabled
            or moe_config.activation != MoEActivation.SILU
            or moe_config.in_dtype != torch.bfloat16
            or moe_config.swiglu_limit != 10.0
        ):
            raise ValueError(
                "TrellisMX GLM adapter requires TP4, EP1 and GLM Flash shapes"
            )
        self.layer_index = layer_index
        self.rank = parallel.tp_rank
        self.overlay = load_overlay(directory)
        self.runtime = None

    @property
    def is_monolithic(self):
        return False

    @property
    def supports_eplb(self):
        return False

    def get_fused_moe_quant_config(self, layer):
        return None

    def process_weights_after_loading(self, layer):
        if self.runtime is not None:
            raise RuntimeError("TrellisMX hot weight replacement is unsupported")
        from b12x.moe._shared.trellismx.p8_native_kernel import P8NativeTPMoE

        device = layer.w13_weight.device
        if device.type != "cuda" or torch.cuda.get_device_capability(device) != (12, 0):
            raise ValueError("This TrellisMX runtime requires SM120 CUDA")
        sidecar = self.overlay.sidecar(self.layer_index, self.rank)
        record = self.overlay.records[self.layer_index, self.rank]
        runtime = P8NativeTPMoE(
            sidecar,
            device=device,
            tp_rank=self.rank,
            world_size=4,
            layer=self.layer_index,
            expected_design_sha256=record["source_design_sha256"],
            expected_transform_sha256=self.overlay.transform_hash,
            topk=8,
            hidden=4096,
            intermediate=512,
            swiglu_limit=10.0,
            small_m_scheduler=True,
            fc1_tile_n=128,
            fuse_scratch_zero=True,
            prefill_chunk_tokens=0,
            grid_policy=True,
            fc1_warp_quant=False,
            fc1_broadcast_a=True,
        )
        # Release only replaced routed storage, after a successful load. Never
        # touch the runner's router/shared experts or the separately owned MTP.
        released = 0
        for name in (
            "w13_weight",
            "w2_weight",
            "w13_weight_scale",
            "w2_weight_scale",
            "w13_weight_scale_2",
            "w2_weight_scale_2",
            "w13_input_scale",
            "w2_input_scale",
        ):
            value = getattr(layer, name)
            released += value.numel() * value.element_size()
            setattr(
                layer,
                name,
                torch.nn.Parameter(
                    torch.empty(0, dtype=value.dtype, device=value.device),
                    requires_grad=False,
                ),
            )
        self.runtime = runtime
        layer.b12x_warmup_provider = self
        logger.info(
            "TrellisMX layer=%d rank=%d K%d E4M3/UE8M0-32 "
            "coupled-h512-h128 released_carrier_bytes=%d",
            self.layer_index,
            self.rank,
            record["bits"],
            released,
        )

    def apply(
        self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
    ):
        if self.runtime is None:
            raise RuntimeError("TrellisMX routed weights were not loaded")
        if x.shape[0] == 0:
            return torch.empty_like(x)
        return self.runtime(x, topk_weights, topk_ids)

    def apply_monolithic(self, *args, **kwargs):
        raise RuntimeError("TrellisMX routing belongs to the Jovian MoE runner")

    def get_b12x_warmup_unit(self, layer, token_counts, output_dtype):
        runtime = self.runtime
        if runtime is None:
            raise RuntimeError("TrellisMX warmup before weight load")

        def compile():
            for tokens in token_counts:
                x = torch.zeros(
                    (tokens, 4096), dtype=output_dtype, device=runtime.device
                )
                ids = torch.arange(8, dtype=torch.int32, device=runtime.device)
                ids = ids.expand(tokens, 8).contiguous()
                weights = torch.full((tokens, 8), 0.125, device=runtime.device)
                runtime(x, weights, ids)

        # Each layer owns compiled descriptors and scratch; do not deduplicate
        # across layers solely because K/shape match.
        return B12xWarmupUnit(
            name="TrellisMX",
            key=(type(self), self.layer_index, self.rank),
            compile=compile,
        )
