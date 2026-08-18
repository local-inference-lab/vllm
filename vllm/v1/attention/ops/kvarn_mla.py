# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVarN K5 cache operations for MLA latent caches."""

from __future__ import annotations

import os

import torch

from vllm.model_executor.layers.quantization.kvarn.config import KVarNMLAConfig
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.kvarn_store import (
    kvarn_store_tile_k_batch_from_sinkhorn,
)
from vllm.v1.attention.ops.triton_kvarn_sinkhorn import kvarn_sinkhorn_triton


@triton.jit
def _unpack_dense_bits(payload_ptr, value_indices, mask, bits: tl.constexpr):
    bit_offsets = value_indices * bits
    byte_offsets = bit_offsets // 8
    shifts = bit_offsets % 8
    lo = tl.load(payload_ptr + byte_offsets, mask=mask, other=0).to(tl.uint32)
    hi = tl.load(payload_ptr + byte_offsets + 1, mask=mask, other=0).to(tl.uint32)
    return ((lo | (hi << 8)) >> shifts) & ((1 << bits) - 1)


@triton.jit
def _scatter_kvarn_mla_exact_kernel(
    latent_ptr,
    rope_ptr,
    slot_mapping_ptr,
    block_to_slot_ptr,
    latent_pool_ptr,
    rope_pool_ptr,
    latent_stride_t: tl.constexpr,
    rope_stride_t: tl.constexpr,
    latent_pool_stride_s: tl.constexpr,
    latent_pool_stride_t: tl.constexpr,
    rope_pool_stride_s: tl.constexpr,
    rope_pool_stride_t: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    NUM_POOL_SLOTS: tl.constexpr,
    GROUP: tl.constexpr,
    LATENT_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
):
    token = tl.program_id(0)
    physical_slot = tl.load(slot_mapping_ptr + token)
    valid_slot = physical_slot >= 0
    block = physical_slot // GROUP
    offset = physical_slot % GROUP
    pool_slot = tl.load(
        block_to_slot_ptr + block,
        mask=valid_slot & (block < NUM_BLOCKS),
        other=-1,
    )
    valid = (
        valid_slot
        & (block < NUM_BLOCKS)
        & (pool_slot >= 0)
        & (pool_slot < NUM_POOL_SLOTS)
    )
    safe_pool_slot = tl.where(valid, pool_slot, 0)

    latent_cols = tl.arange(0, LATENT_DIM)
    latent = tl.load(latent_ptr + token * latent_stride_t + latent_cols)
    tl.store(
        latent_pool_ptr
        + safe_pool_slot * latent_pool_stride_s
        + offset * latent_pool_stride_t
        + latent_cols,
        latent,
        mask=valid,
    )

    rope_cols = tl.arange(0, ROPE_DIM)
    rope = tl.load(rope_ptr + token * rope_stride_t + rope_cols)
    tl.store(
        rope_pool_ptr
        + safe_pool_slot * rope_pool_stride_s
        + offset * rope_pool_stride_t
        + rope_cols,
        rope,
        mask=valid,
    )


def scatter_kvarn_mla_exact(
    latent_rotated: torch.Tensor,
    rope: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_to_slot: torch.Tensor,
    latent_pool: torch.Tensor,
    rope_pool: torch.Tensor,
) -> None:
    num_tokens = slot_mapping.shape[0]
    if num_tokens == 0:
        return
    if latent_rotated.shape[0] < num_tokens or rope.shape[0] < num_tokens:
        raise ValueError("KVarN MLA exact scatter inputs have too few rows")
    if latent_pool.shape[0] != rope_pool.shape[0]:
        raise ValueError("KVarN MLA exact latent/RoPE pools must have equal slots")
    if not slot_mapping.is_contiguous() or not block_to_slot.is_contiguous():
        raise ValueError("KVarN MLA exact scatter index buffers must be contiguous")
    _scatter_kvarn_mla_exact_kernel[(num_tokens,)](
        latent_rotated,
        rope,
        slot_mapping,
        block_to_slot,
        latent_pool,
        rope_pool,
        latent_rotated.stride(0),
        rope.stride(0),
        latent_pool.stride(0),
        latent_pool.stride(1),
        rope_pool.stride(0),
        rope_pool.stride(1),
        NUM_BLOCKS=block_to_slot.shape[0],
        NUM_POOL_SLOTS=latent_pool.shape[0],
        GROUP=latent_pool.shape[1],
        LATENT_DIM=latent_pool.shape[2],
        ROPE_DIM=rope_pool.shape[2],
        num_warps=4,
    )


@triton.jit
def _round_to_even(values):
    lower = tl.floor(values)
    fraction = values - lower
    lower_int = lower.to(tl.int32)
    ties_up = (lower_int & 1) != 0
    return tl.where(
        fraction > 0.5,
        lower + 1.0,
        tl.where(fraction < 0.5, lower, lower + ties_up),
    )


@triton.jit
def _quantize_rtn(values, lower, scale, qmax: tl.constexpr):
    return tl.maximum(
        tl.minimum(_round_to_even((values - lower) / scale), qmax),
        0.0,
    )


@triton.jit
def _serialize_kvarn_mla_blocks_kernel(
    balanced_ptr,
    s_col_ptr,
    s_row_ptr,
    rope_pool_ptr,
    block_ids_ptr,
    pool_slots_ptr,
    cache_ptr,
    balanced_stride_n: tl.constexpr,
    balanced_stride_r: tl.constexpr,
    s_col_stride_n: tl.constexpr,
    s_row_stride_n: tl.constexpr,
    rope_pool_stride_s: tl.constexpr,
    rope_pool_stride_t: tl.constexpr,
    cache_stride_b: tl.constexpr,
    GROUP: tl.constexpr,
    LATENT_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BITS: tl.constexpr,
    PACKED_ROW_BYTES: tl.constexpr,
    S_COL_OFFSET: tl.constexpr,
    ZP_OFFSET: tl.constexpr,
    S_ROW_OFFSET: tl.constexpr,
    ROPE_OFFSET: tl.constexpr,
    AFFINE_REFIT: tl.constexpr,
):
    program = tl.program_id(0)
    tile = program // LATENT_DIM
    row = program % LATENT_DIM
    tokens = tl.arange(0, GROUP)
    values = tl.load(
        balanced_ptr + tile * balanced_stride_n + row * balanced_stride_r + tokens
    ).to(tl.float32)

    qmax: tl.constexpr = (1 << BITS) - 1
    lower = tl.min(values, axis=0)
    upper = tl.max(values, axis=0)
    quant_scale = tl.maximum((upper - lower) / qmax, 1e-10)
    scale = quant_scale
    zero = lower
    codes = _quantize_rtn(values, lower, quant_scale, qmax)

    if AFFINE_REFIT:
        token_scales = tl.load(s_col_ptr + tile * s_col_stride_n + tokens).to(
            tl.float32
        )
        weights = token_scales * token_scales
        weight_sum = tl.maximum(tl.sum(weights, axis=0), 1e-20)
        code_mean = tl.sum(weights * codes, axis=0) / weight_sum
        value_mean = tl.sum(weights * values, axis=0) / weight_sum
        centered_codes = codes - code_mean
        denominator = tl.sum(weights * centered_codes * centered_codes, axis=0)
        numerator = tl.sum(
            weights * centered_codes * (values - value_mean),
            axis=0,
        )
        fitted_scale = tl.maximum(numerator / tl.maximum(denominator, 1e-20), 1e-10)
        fitted_zero = value_mean - fitted_scale * code_mean
        usable = denominator > 1e-20
        scale = tl.where(usable, fitted_scale, scale)
        zero = tl.where(usable, fitted_zero, zero)

    physical_block = tl.load(block_ids_ptr + tile)
    record = cache_ptr + physical_block * cache_stride_b
    row_scale = tl.load(s_row_ptr + tile * s_row_stride_n + row).to(tl.float32)
    fp16_record = record.to(tl.pointer_type(tl.float16))
    tl.store(fp16_record + S_COL_OFFSET // 2 + row, row_scale * scale)
    tl.store(fp16_record + ZP_OFFSET // 2 + row, row_scale * zero)

    packed_offsets = tl.arange(0, 64)
    packed_mask = packed_offsets < PACKED_ROW_BYTES
    bit_offsets = packed_offsets * 8
    source = bit_offsets // BITS
    shifts = bit_offsets % BITS

    source0 = tl.load(
        balanced_ptr + tile * balanced_stride_n + row * balanced_stride_r + source,
        mask=packed_mask & (source < GROUP),
        other=lower,
    ).to(tl.float32)
    source1 = tl.load(
        balanced_ptr + tile * balanced_stride_n + row * balanced_stride_r + source + 1,
        mask=packed_mask & (source + 1 < GROUP),
        other=lower,
    ).to(tl.float32)
    source2 = tl.load(
        balanced_ptr + tile * balanced_stride_n + row * balanced_stride_r + source + 2,
        mask=packed_mask & (source + 2 < GROUP),
        other=lower,
    ).to(tl.float32)
    code0 = _quantize_rtn(source0, lower, quant_scale, qmax).to(tl.uint32)
    code1 = _quantize_rtn(source1, lower, quant_scale, qmax).to(tl.uint32)
    code2 = _quantize_rtn(source2, lower, quant_scale, qmax).to(tl.uint32)
    words = code0 | (code1 << BITS) | (code2 << (2 * BITS))
    packed = (words >> shifts) & 0xFF
    tl.store(
        record + row * PACKED_ROW_BYTES + packed_offsets,
        packed.to(tl.uint8),
        mask=packed_mask,
    )

    shared_offsets = tl.arange(0, 64)
    shared_mask = shared_offsets < GROUP
    token_scales = tl.load(
        s_col_ptr + tile * s_col_stride_n + shared_offsets,
        mask=shared_mask,
    )
    tl.store(
        fp16_record + S_ROW_OFFSET // 2 + shared_offsets,
        token_scales,
        mask=(row == 0) & shared_mask,
    )

    pool_slot = tl.load(pool_slots_ptr + tile)
    rope = tl.load(
        rope_pool_ptr
        + pool_slot * rope_pool_stride_s
        + row * rope_pool_stride_t
        + shared_offsets,
        mask=(row < GROUP) & (shared_offsets < ROPE_DIM),
    )
    rope_record = (record + ROPE_OFFSET).to(tl.pointer_type(tl.bfloat16))
    tl.store(
        rope_record + row * ROPE_DIM + shared_offsets,
        rope,
        mask=(row < GROUP) & (shared_offsets < ROPE_DIM),
    )


def _serialize_kvarn_mla_blocks(
    kv_cache: torch.Tensor,
    rope_pool: torch.Tensor,
    block_ids: torch.Tensor,
    pool_slots: torch.Tensor,
    balanced: torch.Tensor,
    s_col: torch.Tensor,
    s_row: torch.Tensor,
    config: KVarNMLAConfig,
) -> None:
    packed_row_bytes = config.latent_packed_bytes // config.latent_dim
    _serialize_kvarn_mla_blocks_kernel[(block_ids.numel() * config.latent_dim,)](
        balanced,
        s_col,
        s_row,
        rope_pool,
        block_ids,
        pool_slots,
        kv_cache.view(torch.uint8),
        balanced.stride(0),
        balanced.stride(1),
        s_col.stride(0),
        s_row.stride(0),
        rope_pool.stride(0),
        rope_pool.stride(1),
        kv_cache.view(torch.uint8).stride(0),
        GROUP=config.group,
        LATENT_DIM=config.latent_dim,
        ROPE_DIM=config.rope_dim,
        BITS=config.bits,
        PACKED_ROW_BYTES=packed_row_bytes,
        S_COL_OFFSET=config.latent_s_col_offset,
        ZP_OFFSET=config.latent_zp_offset,
        S_ROW_OFFSET=config.latent_s_row_offset,
        ROPE_OFFSET=config.rope_offset,
        AFFINE_REFIT=os.environ.get("KVARN_AFFINE_REFIT", "1") == "1",
        num_warps=4,
    )


def pack_kvarn_mla_blocks(
    kv_cache: torch.Tensor,
    latent_pool: torch.Tensor,
    rope_pool: torch.Tensor,
    block_ids: torch.Tensor,
    pool_slots: torch.Tensor,
    config: KVarNMLAConfig,
) -> None:
    """Quantize complete latent tiles and serialize exact BF16 RoPE tiles."""
    if block_ids.numel() == 0:
        return
    latent = latent_pool.index_select(0, pool_slots).float()
    latent_tiles = latent.transpose(1, 2).contiguous()
    balanced, s_col, s_row = kvarn_sinkhorn_triton(
        latent_tiles, iterations=config.sinkhorn_iters
    )
    quantile = float(os.environ.get("KVARN_RTN_QUANTILE", "") or 0.0)
    if config.bits == 5 and quantile <= 0.0:
        _serialize_kvarn_mla_blocks(
            kv_cache,
            rope_pool,
            block_ids,
            pool_slots,
            balanced,
            s_col,
            s_row,
            config,
        )
        return
    packed = kvarn_store_tile_k_batch_from_sinkhorn(
        balanced, s_col, s_row, bits=config.bits
    )

    num_blocks = block_ids.numel()
    records = torch.zeros(
        (num_blocks, config.tile_bytes), dtype=torch.uint8, device=kv_cache.device
    )
    records[:, : config.latent_packed_bytes].copy_(
        packed["q_packed_uint8"].reshape(num_blocks, -1)
    )
    records[
        :,
        config.latent_s_col_offset : config.latent_zp_offset,
    ].copy_(packed["s_col_K"].contiguous().view(torch.uint8))
    records[
        :,
        config.latent_zp_offset : config.latent_s_row_offset,
    ].copy_(packed["zp_K"].contiguous().view(torch.uint8))
    records[:, config.latent_s_row_offset : config.rope_offset].copy_(
        packed["s_row_K"].contiguous().view(torch.uint8)
    )
    rope = rope_pool.index_select(0, pool_slots).to(torch.bfloat16).contiguous()
    records[:, config.rope_offset :].copy_(
        rope.view(torch.uint8).reshape(num_blocks, -1)
    )
    kv_cache.view(torch.uint8).reshape(kv_cache.shape[0], -1).index_copy_(
        0, block_ids, records
    )


@triton.jit
def _materialize_selected_kvarn_mla_kernel(
    selected_ptr,
    cache_ptr,
    block_to_slot_ptr,
    latent_pool_ptr,
    rope_pool_ptr,
    output_ptr,
    selected_stride: tl.constexpr,
    cache_stride_b: tl.constexpr,
    latent_pool_stride_s: tl.constexpr,
    latent_pool_stride_t: tl.constexpr,
    rope_pool_stride_s: tl.constexpr,
    rope_pool_stride_t: tl.constexpr,
    output_stride: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    NUM_POOL_SLOTS: tl.constexpr,
    GROUP: tl.constexpr,
    LATENT_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BITS: tl.constexpr,
    PACKED_BYTES: tl.constexpr,
    S_COL_OFFSET: tl.constexpr,
    ZP_OFFSET: tl.constexpr,
    S_ROW_OFFSET: tl.constexpr,
    ROPE_OFFSET: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    width = LATENT_DIM + ROPE_DIM
    col_mask = cols < width
    physical_slot = tl.load(selected_ptr + row * selected_stride)
    valid_slot = physical_slot >= 0
    block = physical_slot // GROUP
    token = physical_slot % GROUP
    valid_block = valid_slot & (block < NUM_BLOCKS)
    pool_slot = tl.load(block_to_slot_ptr + block, mask=valid_block, other=-1)
    exact = valid_block & (pool_slot >= 0) & (pool_slot < NUM_POOL_SLOTS)
    safe_pool_slot = tl.where(exact, pool_slot, 0)
    safe_block = tl.where(valid_block, block, 0)
    record = cache_ptr + safe_block * cache_stride_b

    latent_mask = col_mask & (cols < LATENT_DIM) & valid_block
    body_latent_mask = latent_mask & ~exact
    latent_indices = cols * GROUP + token
    q = _unpack_dense_bits(record, latent_indices, body_latent_mask, BITS).to(
        tl.float32
    )
    fp16_record = record.to(tl.pointer_type(tl.float16))
    s_col = tl.load(
        fp16_record + S_COL_OFFSET // 2 + cols,
        mask=body_latent_mask,
        other=0.0,
    )
    zp = tl.load(
        fp16_record + ZP_OFFSET // 2 + cols,
        mask=body_latent_mask,
        other=0.0,
    )
    s_row = tl.load(
        fp16_record + S_ROW_OFFSET // 2 + token,
        mask=valid_block & ~exact,
        other=0.0,
    )
    body_latent = (q * s_col + zp) * s_row
    exact_latent = tl.load(
        latent_pool_ptr
        + safe_pool_slot * latent_pool_stride_s
        + token * latent_pool_stride_t
        + cols,
        mask=latent_mask & exact,
        other=0.0,
    ).to(tl.float32)
    latent = tl.where(exact, exact_latent, body_latent)

    rope_cols = cols - LATENT_DIM
    rope_mask = col_mask & (cols >= LATENT_DIM)
    body_rope_ptr = (record + ROPE_OFFSET).to(tl.pointer_type(tl.bfloat16))
    body_rope = tl.load(
        body_rope_ptr + token * ROPE_DIM + rope_cols,
        mask=rope_mask & valid_block & ~exact,
        other=0.0,
    ).to(tl.float32)
    exact_rope = tl.load(
        rope_pool_ptr
        + safe_pool_slot * rope_pool_stride_s
        + token * rope_pool_stride_t
        + rope_cols,
        mask=rope_mask & exact,
        other=0.0,
    ).to(tl.float32)
    rope_value = tl.where(exact, exact_rope, body_rope)
    value = tl.where(cols < LATENT_DIM, latent, rope_value)
    tl.store(
        output_ptr + row * output_stride + cols,
        value,
        mask=col_mask & valid_block,
    )


@triton.jit
def _linearize_selected_kernel(
    selected_ptr,
    remapped_ptr,
    n_elements: tl.constexpr,
    max_physical_slots: tl.constexpr,
):
    offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
    mask = offsets < n_elements
    selected = tl.load(selected_ptr + offsets, mask=mask, other=-1)
    valid = (selected >= 0) & (selected < max_physical_slots)
    tl.store(remapped_ptr + offsets, tl.where(valid, offsets, -1), mask=mask)


def materialize_selected_kvarn_mla(
    selected_indices: torch.Tensor,
    kv_cache: torch.Tensor,
    block_to_slot: torch.Tensor,
    latent_pool: torch.Tensor,
    rope_pool: torch.Tensor,
    output: torch.Tensor,
    remapped: torch.Tensor | None,
    config: KVarNMLAConfig,
) -> None:
    if selected_indices.dtype != torch.int32 or not selected_indices.is_contiguous():
        raise ValueError("KVarN MLA selected indices must be contiguous int32")
    if not output.is_contiguous():
        raise ValueError("KVarN MLA dense workspace must be contiguous")
    if latent_pool.shape[0] != rope_pool.shape[0]:
        raise ValueError("KVarN MLA exact latent/RoPE pools must have equal slots")
    if block_to_slot.numel() < kv_cache.shape[0]:
        raise ValueError("KVarN MLA block-to-slot map is smaller than the cache")
    if block_to_slot.dtype != torch.int32:
        raise ValueError("KVarN MLA block-to-slot map must be int32")

    flat_selected = selected_indices.view(-1)
    flat_output = output.view(-1, config.latent_dim + config.rope_dim)
    rows = flat_selected.numel()
    if rows > flat_output.shape[0]:
        raise ValueError(
            f"KVarN MLA dense workspace has {flat_output.shape[0]} rows, "
            f"requires {rows}"
        )
    flat_remapped = None
    if remapped is not None:
        if remapped.dtype != torch.int32 or not remapped.is_contiguous():
            raise ValueError("KVarN MLA remap workspace must be contiguous int32")
        flat_remapped = remapped.view(-1)
        if rows > flat_remapped.numel():
            raise ValueError(
                f"KVarN MLA remap workspace has {flat_remapped.numel()} entries, "
                f"requires {rows}"
            )
    if rows == 0:
        return

    grid = (rows, triton.cdiv(config.latent_dim + config.rope_dim, 64))
    _materialize_selected_kvarn_mla_kernel[grid](
        flat_selected,
        kv_cache.view(torch.uint8),
        block_to_slot,
        latent_pool,
        rope_pool,
        flat_output,
        1,
        kv_cache.view(torch.uint8).stride(0),
        latent_pool.stride(0),
        latent_pool.stride(1),
        rope_pool.stride(0),
        rope_pool.stride(1),
        flat_output.stride(0),
        NUM_BLOCKS=kv_cache.shape[0],
        NUM_POOL_SLOTS=latent_pool.shape[0],
        GROUP=config.group,
        LATENT_DIM=config.latent_dim,
        ROPE_DIM=config.rope_dim,
        BITS=config.bits,
        PACKED_BYTES=config.latent_packed_bytes,
        S_COL_OFFSET=config.latent_s_col_offset,
        ZP_OFFSET=config.latent_zp_offset,
        S_ROW_OFFSET=config.latent_s_row_offset,
        ROPE_OFFSET=config.rope_offset,
        BLOCK_D=64,
        num_warps=4,
    )
    if flat_remapped is not None:
        _linearize_selected_kernel[(triton.cdiv(rows, 256),)](
            flat_selected,
            flat_remapped,
            n_elements=rows,
            max_physical_slots=kv_cache.shape[0] * config.group,
        )


@triton.jit
def _copy_bounded_physical_indices_kernel(
    selected_ptr,
    remapped_ptr,
    n_elements: tl.constexpr,
    max_physical_slots: tl.constexpr,
):
    offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
    mask = offsets < n_elements
    physical = tl.load(selected_ptr + offsets, mask=mask, other=-1)
    valid = (physical >= 0) & (physical < max_physical_slots)
    tl.store(remapped_ptr + offsets, tl.where(valid, physical, -1), mask=mask)


def remap_kvarn_mla_physical_indices(
    selected_indices: torch.Tensor,
    remapped: torch.Tensor,
    *,
    max_physical_slots: int,
) -> None:
    if (
        selected_indices.dtype != torch.int32
        or remapped.dtype != torch.int32
        or not selected_indices.is_contiguous()
        or not remapped.is_contiguous()
    ):
        raise ValueError("KVarN MLA physical index buffers must be contiguous int32")
    rows = selected_indices.numel()
    if rows > remapped.numel():
        raise ValueError(
            f"KVarN MLA remap workspace has {remapped.numel()} entries, requires {rows}"
        )
    if rows == 0:
        return
    _copy_bounded_physical_indices_kernel[(triton.cdiv(rows, 256),)](
        selected_indices.view(-1),
        remapped.view(-1),
        n_elements=rows,
        max_physical_slots=max_physical_slots,
    )


def materialize_physical_kvarn_mla(
    physical_slots: torch.Tensor,
    selected_indices: torch.Tensor,
    kv_cache: torch.Tensor,
    block_to_slot: torch.Tensor,
    latent_pool: torch.Tensor,
    rope_pool: torch.Tensor,
    output: torch.Tensor,
    remapped: torch.Tensor,
    config: KVarNMLAConfig,
) -> None:
    """Materialize the local page arena at its stable physical row indices."""
    page_rows = kv_cache.shape[0] * config.group
    if (
        physical_slots.dtype != torch.int32
        or not physical_slots.is_contiguous()
        or physical_slots.numel() != page_rows
    ):
        raise ValueError(
            "KVarN MLA physical-slot workspace must be contiguous int32 "
            f"with {page_rows} entries"
        )
    materialize_selected_kvarn_mla(
        physical_slots,
        kv_cache,
        block_to_slot,
        latent_pool,
        rope_pool,
        output,
        None,
        config,
    )
    remap_kvarn_mla_physical_indices(
        selected_indices,
        remapped,
        max_physical_slots=page_rows,
    )


def stage_physical_kvarn_mla_fp8(
    physical_slots: torch.Tensor,
    selected_indices: torch.Tensor,
    kv_cache: torch.Tensor,
    block_to_slot: torch.Tensor,
    latent_pool: torch.Tensor,
    rope_pool: torch.Tensor,
    output_records: torch.Tensor,
    remapped: torch.Tensor,
    config: KVarNMLAConfig,
) -> None:
    """Stage the local page arena in SparkInfer's physical token order."""
    try:
        from b12x.attention.kvarn_mla import stage_k5_as_fp8_records
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "KVarN MLA requires a b12x build with "
            "b12x.attention.kvarn_mla.stage_k5_as_fp8_records"
        ) from exc

    page_rows = kv_cache.shape[0] * config.group
    if (
        physical_slots.dtype != torch.int32
        or not physical_slots.is_contiguous()
        or physical_slots.numel() != page_rows
    ):
        raise ValueError(
            "KVarN MLA physical-slot workspace must be contiguous int32 "
            f"with {page_rows} entries"
        )
    if (
        output_records.dtype != torch.uint8
        or not output_records.is_contiguous()
        or output_records.shape != (page_rows, 656)
    ):
        raise ValueError(f"KVarN MLA FP8 workspace must have shape ({page_rows}, 656)")
    if output_records.data_ptr() % 16:
        raise ValueError("KVarN MLA FP8 workspace must be 16-byte aligned")
    if block_to_slot.numel() < kv_cache.shape[0]:
        raise ValueError("KVarN MLA block-to-slot map is smaller than the cache")
    if latent_pool.shape[0] != rope_pool.shape[0]:
        raise ValueError("KVarN MLA exact latent/RoPE pools must have equal slots")

    stage_k5_as_fp8_records(
        physical_slots,
        kv_cache,
        block_to_slot,
        latent_pool,
        rope_pool,
        output_records,
    )
    remap_kvarn_mla_physical_indices(
        selected_indices,
        remapped,
        max_physical_slots=page_rows,
    )
