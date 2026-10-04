# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Losslessly compressed DeepSeek-V4-Flash routed experts with native dense tensors.

Serves DeepSeek-V4-Flash and DeepSeek-V4-Flash-Vision-Exp. Only the target
model's routed experts read compressed scales; dense, attention, shared-expert,
vision and MTP/DSpark draft tensors keep the source configuration.
"""

from typing import cast

import regex as re
import torch

import vllm.envs as envs
from vllm.config import get_current_vllm_config, get_current_vllm_config_or_none
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.models.deepseek_v4_1.mxfp4_csf import Mxfp4CsfMoEMethod
from vllm.models.deepseek_v41.quant_config import DeepseekV4FP8Config


class DeepseekV4Mxfp4CsfConfig(DeepseekV4FP8Config):
    """MXFP4-CSF target experts; native block-FP8 dense and draft weights.

    The base is the class CUDA registers as ``deepseek_v4_fp8``, which serves
    the uncompressed DeepSeek-V4-Flash checkpoints.
    """

    checkpoint_root: str
    scale_scratch: tuple[torch.Tensor, ...] | None

    @classmethod
    def get_name(cls):
        return "mxfp4_csf"

    @classmethod
    def get_min_capability(cls):
        return 120

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        if (
            user_quant in (None, cls.get_name())
            and hf_quant_cfg is not None
            and hf_quant_cfg.get("quant_method") == cls.get_name()
            and getattr(hf_config, "model_type", None) == "deepseek_v4"
        ):
            return cls.get_name()
        return None

    @classmethod
    def from_config(cls, config):
        if config.get("format_version") != 1 or not config.get("checkpoint_root"):
            raise ValueError("MXFP4-CSF requires format_version=1 and checkpoint_root")
        if config.get("weight_block_size") != [128, 128]:
            raise ValueError(
                "DeepSeek-V4-Flash MXFP4-CSF requires the source block-FP8 "
                "[128, 128] quantization config"
            )
        # The serving config keeps the source fields and replaces the method.
        source = {
            key: value
            for key, value in config.items()
            if key not in ("format_version", "checkpoint_root")
        }
        result = cast(
            "DeepseekV4Mxfp4CsfConfig",
            super().from_config({**source, "quant_method": "fp8"}),
        )
        result.checkpoint_root = config["checkpoint_root"]
        result.scale_scratch = None
        return result

    def get_quant_method(self, layer, prefix):
        current = get_current_vllm_config_or_none()
        if current is not None and current.parallel_config.enable_expert_parallel:
            # MegaMoE experts bypass quantization methods and would never read
            # the compressed experts.
            raise NotImplementedError(
                "MXFP4-CSF DeepSeek-V4-Flash supports TP without expert parallelism"
            )
        if isinstance(layer, RoutedExperts):
            match = re.search(r"(?:^|\.)layers\.(\d+)\.", prefix)
            if match is None:
                raise ValueError("DeepSeek-V4 routed experts require a numbered layer")
            config = get_current_vllm_config()
            # MTP and DSpark draft layers follow the target layers.
            if int(match.group(1)) < config.model_config.hf_config.num_hidden_layers:
                if self.expert_dtype != "fp4" or self.moe_quant_algo:
                    raise ValueError("MXFP4-CSF requires MXFP4 routed experts")
                if (
                    config.parallel_config.pipeline_parallel_size != 1
                    or config.parallel_config.use_ubatching
                ):
                    raise NotImplementedError(
                        "MXFP4-CSF shared scale scratch requires PP1 without ubatching"
                    )
                if config.load_config.load_format not in ("mxfp4_csf",):
                    raise ValueError("MXFP4-CSF requires --load-format mxfp4_csf ")
                return Mxfp4CsfMoEMethod(
                    layer.moe_config,
                    self,
                    activation_mode="a16" if envs.VLLM_B12X_MOE_FP4_FORCE_A16 else "a8",
                )
        return super().get_quant_method(layer, prefix)
