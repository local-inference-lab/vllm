# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8-Flash-Next model configuration."""

from typing import ClassVar

from transformers import (
    PreTrainedConfig,
    Qwen4ExpConfig,
    Qwen4ExpTextConfig,
    Qwen4ExpVisionConfig,
)


class Qwen3_8FlashNextVisionConfig(Qwen4ExpVisionConfig):
    model_type = "qwen3_8_flash_next"


class Qwen3_8FlashNextTextConfig(Qwen4ExpTextConfig):
    model_type = "qwen3_8_flash_next_text"


class Qwen3_8FlashNextConfig(Qwen4ExpConfig):
    model_type = "qwen3_8_flash_next"
    sub_configs: ClassVar[dict[str, type[PreTrainedConfig]]] = {
        "vision_config": Qwen3_8FlashNextVisionConfig,
        "text_config": Qwen3_8FlashNextTextConfig,
    }


__all__ = [
    "Qwen3_8FlashNextConfig",
    "Qwen3_8FlashNextTextConfig",
    "Qwen3_8FlashNextVisionConfig",
    "Qwen4ExpConfig",
    "Qwen4ExpTextConfig",
    "Qwen4ExpVisionConfig",
]
