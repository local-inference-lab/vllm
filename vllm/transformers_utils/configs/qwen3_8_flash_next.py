# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility exports for Qwen3.8-Flash-Next configuration classes."""

from vllm.models.qwen3_8_flash_next.config import (
    Qwen3_8FlashNextConfig,
    Qwen3_8FlashNextTextConfig,
    Qwen3_8FlashNextVisionConfig,
    Qwen4ExpConfig,
    Qwen4ExpTextConfig,
    Qwen4ExpVisionConfig,
)

__all__ = [
    "Qwen3_8FlashNextConfig",
    "Qwen3_8FlashNextTextConfig",
    "Qwen3_8FlashNextVisionConfig",
    "Qwen4ExpConfig",
    "Qwen4ExpTextConfig",
    "Qwen4ExpVisionConfig",
]
