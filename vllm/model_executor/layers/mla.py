# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import math
import os
import re
from dataclasses import dataclass
from functools import cache

import torch

from vllm.config import CacheConfig, get_current_vllm_config
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.quantization import QuantizationConfig


_NVFP4_MLA_SCALES_ENV = "VLLM_NVFP4_MLA_SCALES_FILE"
_NVFP4_MLA_SCALES_FORMAT = "nvfp4_ds_mla_outer_scale_v1"
_NVFP4_MLA_NUM_LAYERS = 78
_NVFP4_MLA_LATENT_DIM = 512
_NVFP4_MLA_SCALE_DENOMINATOR = 6.0 * 448.0
_NVFP4_MLA_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_KV_FP8_ROPE_ENABLED = os.getenv("KV_FP8_ROPE", "0") == "1"


_IS_GLM_MOE_DSA_CACHE: bool | None = None


def _is_glm_moe_dsa_model() -> bool:
    # Robust to being called before the vLLM config context is established
    # (cudagraph compilation in a worker): fall back to the explicit request and
    # re-resolve once the config is available. Only reached when KV_FP8_ROPE=1.
    global _IS_GLM_MOE_DSA_CACHE
    if _IS_GLM_MOE_DSA_CACHE is not None:
        return _IS_GLM_MOE_DSA_CACHE
    try:
        vllm_config = get_current_vllm_config()
    except Exception:
        return _KV_FP8_ROPE_ENABLED
    model_config = vllm_config.model_config
    if model_config is None:
        return False
    model_type = getattr(model_config.hf_config, "model_type", None)
    if model_type == "glm_moe_dsa":
        _IS_GLM_MOE_DSA_CACHE = True
        return True
    speculative_config = getattr(vllm_config, "speculative_config", None)
    target_model_config = getattr(
        speculative_config, "target_model_config", None
    )
    target_model_type = (
        getattr(target_model_config.hf_config, "model_type", None)
        if target_model_config is not None
        else None
    )
    result = model_type == "deepseek_mtp" and target_model_type == "glm_moe_dsa"
    _IS_GLM_MOE_DSA_CACHE = result
    return result


@cache
def _load_nvfp4_mla_outer_scales(path: str) -> tuple[float, ...]:
    """Load and validate the calibrated, zero-based per-layer outer scales."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{_NVFP4_MLA_SCALES_ENV} must contain a JSON object")
    if payload.get("format") != _NVFP4_MLA_SCALES_FORMAT:
        raise ValueError(
            f"{_NVFP4_MLA_SCALES_ENV} has unsupported format "
            f"{payload.get('format')!r}"
        )
    if type(payload.get("num_layers")) is not int or (
        payload["num_layers"] != _NVFP4_MLA_NUM_LAYERS
    ):
        raise ValueError(
            f"{_NVFP4_MLA_SCALES_ENV} must declare "
            f"num_layers={_NVFP4_MLA_NUM_LAYERS}"
        )
    if type(payload.get("latent_dim")) is not int or (
        payload["latent_dim"] != _NVFP4_MLA_LATENT_DIM
    ):
        raise ValueError(
            f"{_NVFP4_MLA_SCALES_ENV} must declare "
            f"latent_dim={_NVFP4_MLA_LATENT_DIM}"
        )
    denominator = payload.get("denominator")
    if isinstance(denominator, bool) or not isinstance(
        denominator, (int, float)
    ) or not math.isclose(
        float(denominator), _NVFP4_MLA_SCALE_DENOMINATOR, rel_tol=0.0, abs_tol=0.0
    ):
        raise ValueError(
            f"{_NVFP4_MLA_SCALES_ENV} must declare "
            f"denominator={_NVFP4_MLA_SCALE_DENOMINATOR}"
        )
    raw_scales = payload.get("scales")
    if not isinstance(raw_scales, list):
        raise ValueError(f"{_NVFP4_MLA_SCALES_ENV} must contain a scales list")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in raw_scales
    ):
        raise ValueError(f"{_NVFP4_MLA_SCALES_ENV} scales must be JSON numbers")
    scales = tuple(float(value) for value in raw_scales)
    if len(scales) != _NVFP4_MLA_NUM_LAYERS or any(
        not math.isfinite(value) or value <= 0.0 for value in scales
    ):
        raise ValueError(
            f"{_NVFP4_MLA_SCALES_ENV} must contain exactly "
            f"{_NVFP4_MLA_NUM_LAYERS} finite positive scales"
        )
    return scales


@dataclass
class MLAModules:
    """Modules used in MLA."""

    kv_a_layernorm: torch.nn.Module
    kv_b_proj: torch.nn.Module
    rotary_emb: torch.nn.Module
    o_proj: torch.nn.Module
    fused_qkv_a_proj: torch.nn.Module | None
    kv_a_proj_with_mqa: torch.nn.Module | None
    q_a_layernorm: torch.nn.Module | None
    q_b_proj: torch.nn.Module | None
    q_proj: torch.nn.Module | None
    indexer: torch.nn.Module | None
    is_sparse: bool
    topk_indices_buffer: torch.Tensor | None
    indexer_rotary_emb: torch.nn.Module | None = None


# --8<-- [start:multi_head_latent_attention]
@PluggableLayer.register("multi_head_latent_attention")
class MultiHeadLatentAttentionWrapper(PluggableLayer):
    """Pluggable MLA layer which allows OOT backends to add
    custom implementations of the outer MLA layer (including rope & o_proj).
    Note that currently oot platforms can still use CustomOp.register_oot to
    replace MLA layer entirely, although we use PluggableLayer to register
    this layer now.

    This class takes positions and hidden_states as input.
    The input tensors can either contain prefill tokens or decode tokens.
    The class does the following:

    1. MLA Preprocess.
    2. Perform multi-head attention to prefill tokens and
       multi-query attention to decode tokens separately.
    3. Return the output tensor.
    """

    # --8<-- [end:multi_head_latent_attention]

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        mla_modules: MLAModules,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        skip_topk: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        self.fused_qkv_a_proj = mla_modules.fused_qkv_a_proj
        self.kv_a_proj_with_mqa = mla_modules.kv_a_proj_with_mqa
        self.q_a_layernorm = mla_modules.q_a_layernorm
        self.q_b_proj = mla_modules.q_b_proj
        self.q_proj = mla_modules.q_proj
        self.kv_a_layernorm = mla_modules.kv_a_layernorm
        self.kv_b_proj = mla_modules.kv_b_proj
        self.rotary_emb = mla_modules.rotary_emb
        self.o_proj = mla_modules.o_proj
        self.indexer = mla_modules.indexer
        self.indexer_rope_emb = mla_modules.indexer_rotary_emb
        self.is_sparse = mla_modules.is_sparse

        # Whether to skip top-k token selection computation in this layer.
        # When True, the indexer will not be called, and the layer will reuse
        # the topk_tokens buffer written by a previous layer in the same pass.
        # Refer: https://arxiv.org/abs/2603.12201 for more details.
        self.skip_topk = skip_topk
        if self.indexer is not None:
            assert hasattr(self.indexer, "topk_tokens")
            self.topk_tokens = self.indexer.topk_tokens
            self.topk_indices_buffer = mla_modules.topk_indices_buffer

        self.mla_attn = MLAAttention(
            num_heads=self.num_heads,
            scale=scale,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            kv_b_proj=self.kv_b_proj,
            use_sparse=self.is_sparse,
            indexer=self.indexer,
            topk_indices_buffer=mla_modules.topk_indices_buffer,
        )

        # The deployed NVFP4 writer accepts a scale tensor but discards it and
        # quantizes with outer scale 1.0.  Feeding x/s_l is exactly the missing
        # writer-side normalization; the CuTe readers restore s_l in-kernel.
        self._nvfp4_mla_outer_scale = 1.0
        scale_file = os.getenv(_NVFP4_MLA_SCALES_ENV, "").strip()
        if scale_file and (
            self.mla_attn.kv_cache_dtype == "nvfp4_ds_mla"
            and self.mla_attn.attn_backend.get_name() == "B12X_MLA_SPARSE"
        ):
            match = _NVFP4_MLA_LAYER_RE.search(prefix)
            if match is None:
                raise ValueError(
                    f"Cannot derive decoder layer index from MLA prefix {prefix!r}"
                )
            layer_idx = int(match.group(1))
            # Layers 0..77 are the calibrated main decoder stack. Higher indices
            # (e.g. the MTP / draft layer 78 under speculative decode) are NOT in
            # the calibration set and keep s_l=1.0 (identity) rather than raising.
            # That layer is deep/late (not underflowing) and its KV is transient,
            # so identity there is a safe no-op for KLD.
            if 0 <= layer_idx < _NVFP4_MLA_NUM_LAYERS:
                self._nvfp4_mla_outer_scale = _load_nvfp4_mla_outer_scales(
                    scale_file
                )[layer_idx]
        # forward_mqa receives this MLAAttention object as ``layer``.  Keep a
        # host float here so no device .item() or per-call scale tensor is needed.
        self.mla_attn._nvfp4_mla_outer_scale = self._nvfp4_mla_outer_scale

        # Runtime cache-format gate only.  This deliberately does not alter the
        # checkpoint tensors or the 512-D latent outer-scale path above.  The
        # cache writer receives k_pe after rotary_emb in forward(), making this
        # the requested POST-RoPE variant.
        self._kv_fp8_rope = bool(
            _KV_FP8_ROPE_ENABLED
            and _is_glm_moe_dsa_model()
            and self.mla_attn.kv_cache_dtype == "nvfp4_ds_mla"
            and self.mla_attn.attn_backend.get_name() == "B12X_MLA_SPARSE"
        )
        if self._kv_fp8_rope and self.rotary_emb is None:
            raise RuntimeError("KV_FP8_ROPE=1 requires GLM rotary_emb")

        self.prefix = prefix

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_c = None
        kv_lora = None

        if self.q_lora_rank is not None:
            assert self.fused_qkv_a_proj is not None, (
                "fused_qkv_a_proj is required when q_lora_rank is not None"
            )
            assert self.q_a_layernorm is not None, (
                "q_a_layernorm is required when q_lora_rank is not None"
            )
            assert self.q_b_proj is not None, (
                "q_b_proj is required when q_lora_rank is not None"
            )

            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            q_c = self.q_a_layernorm(q_c)
            q = self.q_b_proj(q_c)[0]
        else:
            assert self.kv_a_proj_with_mqa is not None, (
                "kv_a_proj_with_mqa is required when q_lora_rank is None"
            )
            assert self.q_proj is not None, (
                "q_proj is required when q_lora_rank is None"
            )
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
            q = self.q_proj(hidden_states)[0]

        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c)
        # Normalize only the 512-D compressed latent before the cache writer.
        # k_pe is a separate 64-D BF16 tensor and remains bit-for-bit unscaled.
        kv_c_for_cache = kv_c_normed
        if self._nvfp4_mla_outer_scale != 1.0:
            kv_c_for_cache = kv_c_normed / self._nvfp4_mla_outer_scale

        q = q.view(-1, self.num_heads, self.qk_head_dim)
        # Add head dim of 1 to k_pe
        k_pe = k_pe.unsqueeze(1)

        if self.rotary_emb is not None:
            q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
                positions, q[..., self.qk_nope_head_dim :], k_pe
            )
        if self._kv_fp8_rope and (
            k_pe.dtype != torch.bfloat16 or k_pe.shape[-1] != 64
        ):
            raise RuntimeError(
                "KV_FP8_ROPE POST-RoPE writer requires BF16 k_pe[...,64], got "
                f"dtype={k_pe.dtype}, shape={tuple(k_pe.shape)}"
            )

        if self.indexer and self.is_sparse and not self.skip_topk:
            self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)

        if llama_4_scaling is not None:
            q *= llama_4_scaling

        attn_out = self.mla_attn(
            q,
            kv_c_for_cache,
            k_pe,
            output_shape=(hidden_states.shape[0], self.num_heads * self.v_head_dim),
        )

        return self.o_proj(attn_out)[0]
