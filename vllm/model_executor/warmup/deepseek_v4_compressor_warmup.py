# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up DeepSeek V4 compressor Triton kernels before KV allocation."""

import gc
from types import SimpleNamespace

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_dcp_group
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec


def _find_deepseek_v4_compressors(model: nn.Module) -> list[nn.Module]:
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None) if config is not None else None
    if model_type is not None and model_type != "deepseek_v4":
        return []

    return [
        module
        for module in model.modules()
        if module.__class__.__name__ == "DeepseekCompressor"
    ]


def _needs_triton_warmup(compressor: nn.Module, dcp_world_size: int) -> bool:
    # Mirrors DeepseekCompressor.forward: CUDA head=512 uses CuTe unless DCP
    # needs Triton's mapped state-cache lookup. The indexer compressor
    # (head=128) always uses this Triton launcher.
    return int(compressor.head_dim) != 512 or dcp_world_size > 1


def _kv_cache_last_dim(spec: MLAAttentionSpec) -> int:
    if spec.cache_dtype_str == "fp8_ds_mla" and spec.model_version == "deepseek_v4":
        return 584
    return spec.head_size


def _make_dummy_kv_cache(spec: MLAAttentionSpec, device: torch.device) -> torch.Tensor:
    dtype_size = get_dtype_size(spec.dtype)
    page_elements = spec.page_size_bytes // dtype_size
    block_size = spec.storage_block_size
    last_dim = _kv_cache_last_dim(spec)
    raw = torch.empty(page_elements, dtype=spec.dtype, device=device)
    return torch.as_strided(
        raw,
        size=(1, block_size, last_dim),
        stride=(page_elements, last_dim, 1),
    )


def _warmup_one_signature(
    compressor: nn.Module,
    spec: MLAAttentionSpec,
    vllm_config: VllmConfig,
    dcp_rank: int,
) -> None:
    from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
        compress_norm_rope_store_triton,
    )

    device = compressor.norm.weight.device
    head_dim = int(compressor.head_dim)
    compress_ratio = int(compressor.compress_ratio)
    overlap = bool(compressor.overlap)
    state_width = int(compressor.coff) * head_dim
    state_block_size = int(compressor.state_cache.block_size)

    dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
    cp_interleave = vllm_config.parallel_config.cp_kv_cache_interleave_size
    history_tokens = (1 + int(overlap)) * compress_ratio
    virtual_block_size = state_block_size * dcp_world_size
    num_state_blocks = max(1, cdiv(history_tokens, virtual_block_size))

    state_cache = torch.zeros(
        (num_state_blocks, state_block_size, 2 * state_width),
        dtype=torch.float32,
        device=device,
    )
    token_to_req_indices = torch.zeros(1, dtype=torch.int32, device=device)
    position = torch.tensor([compress_ratio - 1], dtype=torch.int64, device=device)
    slot_mapping = torch.zeros(1, dtype=torch.int64, device=device)
    block_table = torch.arange(
        num_state_blocks,
        dtype=torch.int32,
        device=device,
    ).view(1, num_state_blocks)

    rope_head_dim = int(compressor.rope_head_dim)
    cos_sin_cache = torch.zeros(
        (1, rope_head_dim),
        dtype=torch.float32,
        device=device,
    )
    cos_sin_cache[:, : rope_head_dim // 2] = 1.0

    kv_cache = _make_dummy_kv_cache(spec, device)
    k_cache_metadata = SimpleNamespace(slot_mapping=slot_mapping)
    pdl_kwargs = (
        {}
        if current_platform.is_rocm() or current_platform.is_xpu()
        else {"launch_pdl": False}
    )

    compress_norm_rope_store_triton(
        state_cache=state_cache,
        num_actual=1,
        token_to_req_indices=token_to_req_indices,
        positions=position,
        slot_mapping=slot_mapping,
        block_table=block_table,
        block_size=state_block_size,
        state_width=state_width,
        cos_sin_cache=cos_sin_cache,
        kv_cache=kv_cache,
        k_cache_metadata=k_cache_metadata,
        pdl_kwargs=pdl_kwargs,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        overlap=overlap,
        use_fp4_cache=bool(compressor.use_fp4_cache),
        rms_norm_weight=compressor.norm.weight,
        rms_norm_eps=float(compressor.rms_norm_eps),
        quant_block=int(compressor._quant_block),
        token_stride=int(compressor._token_stride),
        scale_dim=int(compressor._scale_dim),
        dcp_world_size=dcp_world_size,
        dcp_rank=dcp_rank,
        cp_kv_cache_interleave_size=cp_interleave,
    )


@torch.inference_mode()
def deepseek_v4_compressor_triton_warmup(
    model: nn.Module,
    kv_cache_specs: dict[str, KVCacheSpec],
    vllm_config: VllmConfig,
) -> None:
    if not current_platform.is_cuda():
        return

    dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
    compressors = [
        compressor
        for compressor in _find_deepseek_v4_compressors(model)
        if _needs_triton_warmup(compressor, dcp_world_size)
    ]
    if not compressors:
        return

    dcp_rank = get_dcp_group().rank_in_group if dcp_world_size > 1 else 0
    warmed_signatures: set[tuple[object, ...]] = set()
    for compressor in compressors:
        spec = kv_cache_specs.get(compressor.k_cache_prefix)
        if not isinstance(spec, MLAAttentionSpec) or spec.dtype != torch.uint8:
            continue

        signature = (
            int(compressor.head_dim),
            int(compressor.compress_ratio),
            bool(compressor.overlap),
            bool(compressor.use_fp4_cache),
            int(compressor._quant_block),
            int(compressor._token_stride),
            int(compressor._scale_dim),
            spec.storage_block_size,
            spec.page_size_bytes // get_dtype_size(spec.dtype),
            spec.head_size,
            spec.cache_dtype_str,
            spec.model_version,
            dcp_world_size,
            dcp_rank,
            vllm_config.parallel_config.cp_kv_cache_interleave_size,
        )
        if signature in warmed_signatures:
            continue
        warmed_signatures.add(signature)
        _warmup_one_signature(compressor, spec, vllm_config, dcp_rank)

    torch.accelerator.synchronize()
    gc.collect()
    torch.accelerator.empty_cache()
