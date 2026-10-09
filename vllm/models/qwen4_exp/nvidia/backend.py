# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execution-component selection for the shared Qwen4Exp model."""

from vllm.config import VllmConfig


def uses_b12x(vllm_config: VllmConfig) -> bool:
    kernel_config = vllm_config.kernel_config
    return kernel_config.linear_backend == "b12x" or kernel_config.moe_backend == "b12x"
