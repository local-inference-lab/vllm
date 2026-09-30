# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 lossless MXFP4 expert scales with unchanged BF16 dense weights."""

import torch

from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config
from vllm.models.deepseek_v4_1.exact_mxfp4 import ExactMXFP4MoEMethod


class KimiX4TConfig(Mxfp4Config):
    checkpoint_root: str
    scale_scratch: tuple[torch.Tensor, ...] | None

    @classmethod
    def get_name(cls):
        return "kimi_x4t"

    @classmethod
    def get_min_capability(cls):
        return 120

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        if (
            user_quant in (None, cls.get_name())
            and hf_quant_cfg is not None
            and hf_quant_cfg.get("quant_method") == cls.get_name()
            and getattr(hf_config, "model_type", None) in ("kimi_k3", "kimi_linear")
        ):
            return cls.get_name()
        return None

    @classmethod
    def from_config(cls, config):
        if config.get("format_version") != 1 or not config.get("checkpoint_root"):
            raise ValueError("Kimi X4T requires format_version=1 and checkpoint_root")
        result = cls()
        result.checkpoint_root = config["checkpoint_root"]
        result.scale_scratch = None
        return result

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, RoutedExperts):
            config = get_current_vllm_config()
            if (
                config.parallel_config.pipeline_parallel_size != 1
                or config.parallel_config.use_ubatching
            ):
                raise NotImplementedError(
                    "X4T shared scale scratch requires PP1 without ubatching"
                )
            if config.load_config.load_format != "exact_mxfp4":
                raise ValueError("Kimi X4T requires --load-format exact_mxfp4")
            return ExactMXFP4MoEMethod(layer.moe_config, self)
        return super().get_quant_method(layer, prefix)
