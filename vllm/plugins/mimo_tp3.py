# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.models import mimo_v2_omni
from vllm.model_executor.models.config import (
    MODELS_CONFIG_MAP,
    Qwen3DSparkConfig,
    VerifyAndUpdateConfig,
)


class MiMoTP3Config(VerifyAndUpdateConfig):
    @staticmethod
    def update_model_config_for_parallelism(model_config, parallel_config):
        if parallel_config.tensor_parallel_size != 3:
            return
        from vllm.model_executor.models.mimo_v2 import _padded_head_counts

        config = model_config.hf_text_config
        for prefix in ("", "swa_"):
            query_name = f"{prefix}num_attention_heads"
            kv_name = f"{prefix}num_key_value_heads"
            query_heads = getattr(
                config, f"original_{query_name}", getattr(config, query_name)
            )
            kv_heads = getattr(config, f"original_{kv_name}", getattr(config, kv_name))
            if query_heads != 64 or kv_heads != (8 if prefix else 4):
                raise ValueError(
                    "MiMo TP3 padding requires the 64-head MOPD checkpoint"
                )
            setattr(config, f"original_{query_name}", query_heads)
            setattr(config, f"original_{kv_name}", kv_heads)
            padded_query, padded_kv = _padded_head_counts(query_heads, kv_heads, 3)
            setattr(config, query_name, padded_query)
            setattr(config, kv_name, padded_kv)
        model_config.model_arch_config = model_config.get_model_arch_config()
        if model_config.multimodal_config is not None:
            model_config.multimodal_config.mm_encoder_tp_mode = "data"


class ReplicatedMiMoVisionAttention(mimo_v2_omni.MiMoVisionAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.tp_size == 1:
            self.tp_rank = 0


def register():
    mimo_v2_omni.MiMoVisionAttention = ReplicatedMiMoVisionAttention
    for architecture in (
        "MiMoV2ForCausalLM",
        "MiMoV2FlashForCausalLM",
        "MiMoV2OmniForCausalLM",
    ):
        MODELS_CONFIG_MAP[architecture] = MiMoTP3Config
    MODELS_CONFIG_MAP["DFlashDraftModel"] = Qwen3DSparkConfig
