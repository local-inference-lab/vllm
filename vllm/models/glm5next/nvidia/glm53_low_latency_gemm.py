# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash TP3 decode GEMM selection for unquantized BF16 on SM120.

Mirrors ``deepseek_v32/nvidia/glm52_low_latency_gemm.py``. cuBLAS serves the
small per-rank BF16 projections at decode batch sizes with a split-K pair
(GEMM plus ``splitKreduce``) or an SM80 WMMA fallback. For the measured
(N, K, M) combinations where the CuTe shape-dynamic skinny GEMM beat cuBLAS by
more than 3% on RTX PRO 6000 at TP3 (+1.3-1.7% decode), route them to it;
every other shape and batch size keeps cuBLAS.

The plan is keyed on TP3 per-rank shapes, and the replicated projections
(``fused_qkv_a_proj``, indexer ``wk``/``wq_b``/``weights_proj``) have the same
shape at every TP size, so the swap only happens at tensor_parallel_size == 3.
``VLLM_GLM53_LOW_LATENCY_GEMM=0`` disables it there; it is never enabled at
other TP sizes.
"""

from __future__ import annotations

import os

import torch
from torch import nn

import vllm.envs as envs
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    shape_dynamic_skinny_gemm,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.platforms import current_platform

logger = init_logger(__name__)

SUPPORTED_TP_SIZE = 3

# (N, K) per TP3 rank -> decode M values where the skinny GEMM wins.
GLM53_TP3_PLAN: dict[tuple[int, int], frozenset[int]] = {
    (8598, 4096): frozenset({1, 2, 4, 8}),  # KDA in_proj_qkvgfab
    (128, 4096): frozenset({1, 2, 4}),  # KDA g_a_proj, indexer wk
    (4096, 2816): frozenset({1, 2}),  # KDA o_proj
    (2048, 4096): frozenset({1, 2}),  # DSA fused_qkv_a_proj
    (6144, 1536): frozenset({1}),  # DSA q_b_proj
    (4096, 6144): frozenset({1, 2, 4}),  # DSA o_proj
    (4096, 1536): frozenset({1, 2, 4}),  # indexer wq_b
    (32, 4096): frozenset({1, 2, 4, 8}),  # indexer weights_proj
}


def _config_fits(config: SkinnyGemmConfig, m: int, n: int, k: int) -> bool:
    """Whether the skinny GEMM accepts this shape (mirrors its call checks)."""
    return (
        1 <= m <= 16
        and config.num_rows == m
        and n % config.outputs_per_block == 0
        and k % (config.block_size * config.vector_width) == 0
        and (config.static_k is None or config.static_k == k)
    )


def plan_for(n: int, k: int) -> dict[int, SkinnyGemmConfig]:
    """Measured M -> config for a TP3 (N, K) shape, dropping unusable Ms."""
    plan: dict[int, SkinnyGemmConfig] = {}
    for m in sorted(GLM53_TP3_PLAN.get((n, k), ())):
        config = shape_dynamic_skinny_gemm._config(m, n, k)
        if _config_fits(config, m, n, k):
            plan[m] = config
    return plan


def requested() -> bool:
    """``VLLM_GLM53_LOW_LATENCY_GEMM``: unset or nonzero means on (at TP3)."""
    return os.getenv("VLLM_GLM53_LOW_LATENCY_GEMM", "1").strip() not in ("", "0")


def _enabled(dtype: torch.dtype | None) -> bool:
    # A pure-performance path: any unexpected configuration keeps cuBLAS
    # rather than aborting model construction.
    if not requested() or dtype != torch.bfloat16:
        return False
    if get_tensor_model_parallel_world_size() != SUPPORTED_TP_SIZE:
        return False
    if not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(120)
    ):
        return False
    return shape_dynamic_skinny_gemm.is_available()


def _row_major(t: torch.Tensor) -> bool:
    return t.dim() == 2 and t.stride() == (t.shape[1], 1)


class GLM53LowLatencyLinearMethod(UnquantizedLinearMethod):
    def __init__(self, plan: dict[int, SkinnyGemmConfig]) -> None:
        super().__init__()
        self._plan = plan

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight
        config = self._plan.get(x.shape[0]) if x.dim() == 2 else None
        if (
            config is not None
            and bias is None
            and not envs.VLLM_BATCH_INVARIANT
            and x.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16
            and _row_major(x)
            and _row_major(weight)
            and x.shape[1] == weight.shape[1]
        ):
            return shape_dynamic_skinny_gemm(x, weight, config)
        return super().apply(layer, x, bias)


def enable_glm53_low_latency_gemm(module: nn.Module, dtype: torch.dtype | None) -> int:
    """Swap measured TP3 projections to the skinny GEMM; returns the count."""
    if not _enabled(dtype):
        return 0
    warmup: set[SkinnyGemmConfig] = set()
    swapped = 0
    for child in module.modules():
        if (
            not isinstance(child, LinearBase)
            or type(child.quant_method) is not UnquantizedLinearMethod
        ):
            continue
        weight = getattr(child, "weight", None)
        if weight is None or weight.dim() != 2 or weight.dtype != torch.bfloat16:
            continue
        plan = plan_for(int(weight.shape[0]), int(weight.shape[1]))
        if not plan:
            continue
        child.quant_method = GLM53LowLatencyLinearMethod(plan)
        warmup.update(plan.values())
        swapped += 1
    if warmup:
        # Compile before CUDA graph capture.
        shape_dynamic_skinny_gemm.request_warmup_configs(torch.bfloat16, warmup)
    logger.info_once(
        "GLM-5.3 low-latency GEMM: %d BF16 projections routed to the CuTe skinny "
        "GEMM for their measured decode batch sizes (%d warmup configs)",
        swapped,
        len(warmup),
    )
    return swapped
