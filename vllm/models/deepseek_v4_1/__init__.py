# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 native b12x model entry points."""

from .quant_config import DeepseekV41FP8Config


def __getattr__(name):
    if name == "DSparkDeepseekV4ForCausalLM":
        from .nvidia.dspark import DSparkDeepseekV4ForCausalLM

        return DSparkDeepseekV4ForCausalLM
    if name == "DeepseekV41ForCausalLM":
        from .nvidia.vl_model import DeepseekV41ForCausalLM

        return DeepseekV41ForCausalLM
    raise AttributeError(name)


__all__ = [
    "DeepseekV41FP8Config",
    "DSparkDeepseekV4ForCausalLM",
    "DeepseekV41ForCausalLM",
]
