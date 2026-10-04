# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP4 lossless scale compression with model-specific retained precision."""

from vllm.models.deepseek_v4_1.mxfp4_csf import DeepseekV41Mxfp4CsfConfig

from .kimi_mxfp4_csf import KimiMxfp4CsfConfig


class Mxfp4CsfConfig(KimiMxfp4CsfConfig):
    """Select Kimi, DeepSeek-V4.1 or DeepSeek-V4 dense precision by family."""

    @classmethod
    def get_name(cls):
        return "mxfp4_csf"

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        stored = (hf_quant_cfg or {}).get("quant_method")
        if user_quant in (None, cls.get_name()) and stored == cls.get_name():
            return cls.get_name()
        return None

    @classmethod
    def from_config(cls, config):
        from vllm.model_executor.model_loader.mxfp4_csf_loader import (
            checkpoint_contract,
        )

        if config.get("format_version") != 1 or not config.get("checkpoint_root"):
            raise ValueError("MXFP4-CSF requires format_version=1 and checkpoint_root")
        family = checkpoint_contract(config["checkpoint_root"])["family"]
        if family == "deepseek_v41":
            return DeepseekV41Mxfp4CsfConfig.from_config(config)
        if family == "deepseek_v4_flash":
            # Deferred: the DeepSeek-V4 package imports its platform model.
            from vllm.models.deepseek_v4.mxfp4_csf import DeepseekV4Mxfp4CsfConfig

            return DeepseekV4Mxfp4CsfConfig.from_config(config)
        if family == "kimi_k3":
            return super().from_config(config)
        raise ValueError(f"Unsupported MXFP4-CSF model family: {family}")
