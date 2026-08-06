# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Text-only Kimi K3 configuration."""

from transformers.configuration_utils import PretrainedConfig

from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig


class KimiK3Config(PretrainedConfig):
    model_type = "kimi_k3"

    def __init__(
        self,
        text_config: dict | KimiLinearConfig | None = None,
        vision_config: dict | PretrainedConfig | None = None,
        ignore_index: int = -100,
        media_placeholder_token_id: int = 163605,
        pad_token_id: int = 0,
        image_placeholder: str = "<|kimi_image_placeholder|>",
        **kwargs,
    ) -> None:
        if text_config is None:
            self.text_config = KimiLinearConfig()
        elif isinstance(text_config, dict):
            self.text_config = KimiLinearConfig(**text_config)
        else:
            self.text_config = text_config

        self.vision_config = vision_config
        self.ignore_index = ignore_index
        self.media_placeholder_token_id = media_placeholder_token_id
        self.image_placeholder = image_placeholder

        text_quantization_config = getattr(
            self.text_config, "quantization_config", None
        )

        super().__init__(pad_token_id=pad_token_id, **kwargs)
        if text_quantization_config is not None:
            self.quantization_config = text_quantization_config

    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.text_config.vocab_size
