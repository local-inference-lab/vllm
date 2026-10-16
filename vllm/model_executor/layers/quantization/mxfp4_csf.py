# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP4 lossless scale compression with model-specific retained precision.

[heo 2026-10-05, turin local overlay — MiMo-V2.6-Pro support, private]
Additions vs kk-beta 6008f020 (base classes unchanged):
  * `mimo_v26_pro_mxfp4` family branch in Mxfp4CsfConfig.from_config.
  * MimoMxfp4CsfConfig(Fp8Config): MiMo's source checkpoint is hybrid —
    FP8 dense (with ignored_layers) + MXFP4 routed experts — so the dense
    side must stay on Fp8Config dispatch (DS4.1 pattern: their CSF config
    also extends the model's FP8 config). Only RoutedExperts re-routes to
    the shared in-image Mxfp4CsfMoEMethod (Kimi pattern: the DS4.1 method
    is model-agnostic; all model specifics live in the CSF loader).
  * checkpoint_root may come from MIMO_CSF_ROOT when the HF quant dict has
    no checkpoint_root, so the SOURCE model dir serves unmodified; the
    container contract (family/schema/codec) still gates everything.
"""

import os

from vllm.logger import init_logger

from vllm.models.deepseek_v4_1.mxfp4_csf import DeepseekV41Mxfp4CsfConfig

from .fp8 import Fp8Config
from .kimi_mxfp4_csf import KimiMxfp4CsfConfig

logger = init_logger(__name__)


class MimoMxfp4CsfConfig(Fp8Config):
    """MiMo routed experts via CSF scales; FP8 dense semantics preserved."""

    checkpoint_root: str
    scale_scratch: tuple | None
    # Owner capability attribute for the X4T scale prefetch (Codex §9.1-3):
    # the shared Mxfp4CsfMoEMethod arms only when this is True AND the env gate
    # is on. Non-MiMo owners (DS4.1/DS4-Flash/Kimi) never set it, so they are
    # bit-identical with the env off and unarmed even with it on. Set on the
    # CLASS (not instances) so every per-layer method sees one shared owner.
    x4t_scale_prefetch = True
    # Shared prefetch state (scale_layers/scale_stream/scale_prefetch), one per
    # owner; created in from_config. The method registers into it at load.
    scale_prefetch_state = None

    @classmethod
    def get_name(cls):
        return "mxfp4_csf"

    @classmethod
    def get_min_capability(cls):
        return 120

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        stored = (hf_quant_cfg or {}).get("quant_method")
        is_mimo_hybrid = (
            stored == "fp8"
            and (hf_quant_cfg or {}).get("store_dtype") == "mxfp4"
            and getattr(hf_config, "model_type", "") == "mimo_v2"
        )
        if user_quant in (None, cls.get_name()) and (
            stored == cls.get_name() or is_mimo_hybrid
        ):
            return cls.get_name()
        return None

    @classmethod
    def from_config(cls, config):
        root = config.get("checkpoint_root") or os.environ.get("MIMO_CSF_ROOT")
        if not root:
            raise ValueError(
                "MiMo MXFP4-CSF requires checkpoint_root (config) or "
                "MIMO_CSF_ROOT (env)"
            )
        result = cls(
            is_checkpoint_fp8_serialized=True,
            activation_scheme=config.get("activation_scheme", "dynamic"),
            ignored_layers=config.get("ignored_layers"),
            weight_block_size=config.get("weight_block_size", [128, 128]),
        )
        result.checkpoint_root = root
        result.scale_scratch = None
        # One prefetch state per owner (per model load): scale_layers, side
        # stream and the pending (layer, token, event) triple live here.
        # Codex review fix (P1): the state is owner-wide and the constructor
        # takes no arguments -- the reviewed commit passed layer_index=-1 and
        # broke every MiMo load (from_config raised TypeError with the
        # prefetch gate OFF or ON).
        from vllm.models.deepseek_v4_1.mxfp4_csf import MimoMxfp4CsfScalePrefetch

        result.scale_prefetch_state = MimoMxfp4CsfScalePrefetch()
        return result

    def get_quant_method(self, layer, prefix):
        from vllm.config import get_current_vllm_config
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
        from vllm.models.deepseek_v4_1.mxfp4_csf import Mxfp4CsfMoEMethod

        if isinstance(layer, RoutedExperts):
            config = get_current_vllm_config()
            if (
                config.parallel_config.pipeline_parallel_size != 1
                or config.parallel_config.use_ubatching
            ):
                raise NotImplementedError(
                    "MXFP4-CSF shared scale scratch requires PP1 without ubatching"
                )
            if config.load_config.load_format not in ("mxfp4_csf",):
                raise ValueError("MiMo MXFP4-CSF requires --load-format mxfp4_csf")
            # [heo §23-C] EXPLICIT activation wiring: read the operator's mode
            # once, pass it to the method ctor, and log it. The env alone
            # (VLLM_B12X_MOE_FP4_FORCE_A16=0) does NOT reconfigure this
            # adapter — the ctor default is A16. Validated + logged at boot.
            mode = os.environ.get("MIMO_CSF_ACTIVATION", "a16").strip().lower()
            if mode not in ("a16", "a8"):
                raise ValueError(
                    f"MIMO_CSF_ACTIVATION must be 'a16' or 'a8', got {mode!r}"
                )
            logger.info(
                "[mimo-csf] RoutedExperts activation_mode=%s (MIMO_CSF_ACTIVATION)",
                mode,
            )
            return Mxfp4CsfMoEMethod(layer.moe_config, self, activation_mode=mode)
        return super().get_quant_method(layer, prefix)


class Mxfp4CsfConfig(KimiMxfp4CsfConfig):
    """Select Kimi, DeepSeek-V4.1, DeepSeek-V4 or MiMo precision by family."""

    @classmethod
    def get_name(cls):
        return "mxfp4_csf"

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        # [heo overlay §23-P1] EXPLICIT-INTENT gate: the hybrid MiMo header is
        # only re-routed when the operator asked for CSF (--quantization, a
        # checkpoint_root in the dict, or MIMO_CSF_ROOT). A plain launch of
        # the source checkpoint with none of those MUST stay plain FP8.
        stored = (hf_quant_cfg or {}).get("quant_method")
        has_intent = (
            user_quant == cls.get_name()
            or os.environ.get("MIMO_CSF_ROOT")
            or "checkpoint_root" in (hf_quant_cfg or {})
        )
        if not has_intent:
            return None
        if stored == cls.get_name():
            return cls.get_name()
        is_mimo_hybrid = (
            stored == "fp8"
            and (hf_quant_cfg or {}).get("store_dtype") == "mxfp4"
            and getattr(hf_config, "model_type", "") == "mimo_v2"
        )
        if is_mimo_hybrid:
            return cls.get_name()
        # No match: return None and let vLLM's ordered-selection loop keep
        # probing (config/model.py). An explicit --quantization mxfp4_csf on
        # a non-CSF checkpoint then fails loudly at the post-loop config/model
        # mismatch check ("Quantization method specified in the model config
        # ... does not match ..."), never silently degrading to plain FP8.
        # Raising here would break method enumeration: the loop calls every
        # registered override unconditionally (tests/quantization/
        # test_mxfp4_csf.py:391 asserts None for exact_mxfp4/kimi_x4t).
        return None

    @classmethod
    def from_config(cls, config):
        from vllm.model_executor.model_loader.mxfp4_csf_loader import (
            checkpoint_contract,
        )

        root = config.get("checkpoint_root") or os.environ.get("MIMO_CSF_ROOT")
        if not root:
            raise ValueError(
                "MXFP4-CSF requires checkpoint_root (config) or MIMO_CSF_ROOT (env)"
            )
        family = checkpoint_contract(root)["family"]
        if family == "deepseek_v41":
            return DeepseekV41Mxfp4CsfConfig.from_config(config)
        if family == "deepseek_v4_flash":
            # Deferred: the DeepSeek-V4 package imports its platform model.
            from vllm.models.deepseek_v4.mxfp4_csf import DeepseekV4Mxfp4CsfConfig

            return DeepseekV4Mxfp4CsfConfig.from_config(config)
        if family == "kimi_k3":
            return super().from_config(config)
        if family == "mimo_v26_pro_mxfp4":
            return MimoMxfp4CsfConfig.from_config(config)
        raise ValueError(f"Unsupported MXFP4-CSF model family: {family}")
