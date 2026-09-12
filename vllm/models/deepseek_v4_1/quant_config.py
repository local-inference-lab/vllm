# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 checkpoint quantization with fail-closed native dispatch."""

from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding


class DeepseekV41FP8Config(Fp8Config):
    is_scale_e8m0 = True

    @classmethod
    def get_name(cls):
        return "deepseek_v41_fp8"

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        if getattr(hf_config, "model_type", None) in (
            "deepseek_v41",
            "deepseek_v41_text",
        ) and hf_quant_cfg.get("quant_method") in ("fp8", "deepseek_v41_fp8"):
            return cls.get_name()
        return None

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase):
            from .b12x_layers import B12xFP8LinearMethod, B12xLinearMethod

            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
                match_mode=self.ignored_layers_match_mode,
            ):
                return B12xLinearMethod()
            return B12xFP8LinearMethod(self)
        if isinstance(layer, VocabParallelEmbedding):
            from .b12x_layers import B12xEmbeddingMethod

            return B12xEmbeddingMethod()
        if isinstance(layer, RoutedExperts):
            return Mxfp4MoEMethod(layer.moe_config, numerical_recipe="deepseek_v41")
        return None
