# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 hardware-isolated model entry point."""

from vllm.platforms import current_platform
from vllm.utils.b12x import b12x_native_device

from .quant_config import DeepseekV4FP8Config


def __getattr__(name):
    if name not in ("DeepseekV41ForCausalLM", "DSparkDeepseekV4ForCausalLM"):
        raise AttributeError(name)
    if current_platform.is_rocm():
        from .amd.dspark import DSparkDeepseekV4ForCausalLM
        from .amd.vl_model import DeepseekV41ForCausalLM
    elif b12x_native_device():
        from vllm.models.deepseek_v4_1 import (
            DeepseekV41ForCausalLM,
            DSparkDeepseekV4ForCausalLM,
        )
    else:
        from .nvidia.dspark import DSparkDeepseekV4ForCausalLM
        from .nvidia.vl_model import DeepseekV41ForCausalLM
    return (
        DeepseekV41ForCausalLM
        if name == "DeepseekV41ForCausalLM"
        else DSparkDeepseekV4ForCausalLM
    )


__all__ = [
    "DSparkDeepseekV4ForCausalLM",
    "DeepseekV4FP8Config",
    "DeepseekV41ForCausalLM",
]
