# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4-CSF storage with unchanged native ModelOpt activation calibration."""

from .nvfp4_lsc import Nvfp4LscConfig


class Nvfp4CsfConfig(Nvfp4LscConfig):
    @classmethod
    def get_name(cls):
        return "nvfp4_csf"

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        stored = (hf_quant_cfg or {}).get("quant_method")
        if stored == cls.get_name() or (
            stored == "nvfp4_lsc" and user_quant == cls.get_name()
        ):
            return cls.get_name()
        return None
