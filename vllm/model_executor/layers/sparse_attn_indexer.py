# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers."""

import os
from importlib import import_module
from typing import Any

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    has_deep_gemm,
)
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
)
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024
_B12X_COMPRESSED_INDEX_PAGE_SIZE = 64
_B12X_COMPRESSED_INDEX_HEAD_DIM = 128
_B12X_COMPRESSED_INDEX_SCALE_BYTES = 4
_B12X_COMPRESSED_INDEX_PAGE_WIDTH = _B12X_COMPRESSED_INDEX_PAGE_SIZE * (
    _B12X_COMPRESSED_INDEX_HEAD_DIM + _B12X_COMPRESSED_INDEX_SCALE_BYTES
)
_B12X_EXTEND_TOPK_SUPERTILE_K = int(
    os.getenv("VLLM_B12X_NSA_EXTEND_TOPK_SUPERTILE_K", "32768")
)

# MXFP4 layout: 2 values packed per byte, ue8m0 (1-byte) scale per block of 32.
MXFP4_BLOCK_SIZE = 32


def _b12x_sparse_indexer_requested(enabled: bool | None = None) -> bool:
    return bool(envs.VLLM_USE_B12X_SPARSE_INDEXER if enabled is None else enabled)


def _ensure_b12x_sparse_indexer_supported() -> None:
    if not current_platform.is_cuda():
        raise RuntimeError("VLLM_USE_B12X_SPARSE_INDEXER requires CUDA.")
    if not current_platform.is_device_capability_family(120):
        raise RuntimeError(
            "VLLM_USE_B12X_SPARSE_INDEXER currently requires an SM120 GPU."
        )


def _use_b12x_sparse_indexer(enabled: bool | None = None) -> bool:
    if not _b12x_sparse_indexer_requested(enabled):
        return False
    _ensure_b12x_sparse_indexer_supported()
    return True


def _gather_workspace_shapes(
    total_seq_lens: int,
    head_dim: int,
    fp8_dtype: torch.dtype,
    use_fp4_cache: bool,
) -> tuple[tuple[tuple[int, int], torch.dtype], tuple[tuple[int, int], torch.dtype]]:
    """Return ((values_shape, values_dtype), (scales_shape, scales_dtype)) for
    the K-gather workspace. FP8 path: (T, head_dim) fp8 + (T, 4) uint8 fp32
    scales. MXFP4 path: (T, head_dim // 2) uint8 packed mxfp4 +
    (T, head_dim // MXFP4_BLOCK_SIZE) uint8 ue8m0 scales."""
    if use_fp4_cache:
        return (
            ((total_seq_lens, head_dim // 2), torch.uint8),
            ((total_seq_lens, head_dim // MXFP4_BLOCK_SIZE), torch.uint8),
        )
    return (
        ((total_seq_lens, head_dim), fp8_dtype),
        ((total_seq_lens, 4), torch.uint8),
    )


def kv_cache_as_quant_view(
    kv_cache: torch.Tensor,
    head_dim: int,
    use_fp4_cache: bool,
) -> torch.Tensor:
    """4D ``[num_blocks, block_size, 1, head_width]`` view expected by
    DeepGEMM, from the 3D indexer kv-cache allocation."""
    if use_fp4_cache:
        assert kv_cache.ndim == 3 and kv_cache.dtype == torch.uint8
        num_blocks, block_size, _ = kv_cache.shape
        page_bytes = int(kv_cache.stride(0))
        fp4_bytes = head_dim // 2 + head_dim // MXFP4_BLOCK_SIZE
        return torch.as_strided(
            kv_cache,
            size=(num_blocks, block_size, 1, fp4_bytes),
            stride=(page_bytes, fp4_bytes, fp4_bytes, 1),
        )
    return kv_cache.unsqueeze(-2)


def _normalize_prefill_topk_to_req_relative(
    chunk: object, topk_indices: torch.Tensor
) -> None:
    """Convert packed prefill offsets to per-request token offsets."""
    cu_seq_lens = getattr(chunk, "cu_seq_lens", None)
    token_to_seq = getattr(chunk, "token_to_seq", None)
    if cu_seq_lens is None or token_to_seq is None or cu_seq_lens.numel() <= 2:
        return

    valid = topk_indices >= 0
    safe_indices = topk_indices.clamp(min=0, max=int(token_to_seq.numel()) - 1)
    seq_ids = token_to_seq[safe_indices]
    seq_starts = cu_seq_lens[seq_ids]
    normalized = topk_indices - seq_starts
    topk_indices.copy_(torch.where(valid, normalized, topk_indices))


def _flatten_b12x_compressed_index_cache(kv_cache: torch.Tensor) -> torch.Tensor:
    expected_shape_tail = (
        _B12X_COMPRESSED_INDEX_PAGE_SIZE,
        _B12X_COMPRESSED_INDEX_HEAD_DIM + _B12X_COMPRESSED_INDEX_SCALE_BYTES,
    )

    if kv_cache.ndim != 3 or kv_cache.dtype != torch.uint8:
        raise RuntimeError(
            "b12x C4 compressed indexer cache must be rank-3 uint8 with "
            f"shape [num_blocks, {expected_shape_tail[0]}, "
            f"{expected_shape_tail[1]}], got shape={tuple(kv_cache.shape)} "
            f"dtype={kv_cache.dtype}."
        )
    if tuple(kv_cache.shape[1:]) != expected_shape_tail:
        raise RuntimeError(
            "b12x C4 compressed indexer cache has an unsupported shape, "
            f"got {tuple(kv_cache.shape)}; expected tail {expected_shape_tail}."
        )
    if kv_cache.stride(1) != expected_shape_tail[1] or kv_cache.stride(2) != 1:
        raise RuntimeError(
            "b12x C4 compressed indexer cache has an unsupported layout, "
            f"shape={tuple(kv_cache.shape)} stride={tuple(kv_cache.stride())}; "
            f"expected inner strides ({expected_shape_tail[1]}, 1)."
        )

    return kv_cache.as_strided(
        (int(kv_cache.shape[0]), _B12X_COMPRESSED_INDEX_PAGE_WIDTH),
        (int(kv_cache.stride(0)), 1),
    )


def _run_b12x_compressed_decode_topk(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    kv_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_metadata: torch.Tensor | None,
    topk_indices: torch.Tensor,
    topk_tokens: int,
) -> torch.Tensor:
    from b12x.attention.indexer.tiled_topk import run_row_topk
    from b12x.integration.compressed_indexer import (
        COMPRESSED_INDEX_PAGE_SIZE,
        compressed_index_decode_logits_fp8,
        prepare_compressed_indexer_metadata,
    )

    if int(COMPRESSED_INDEX_PAGE_SIZE) != _B12X_COMPRESSED_INDEX_PAGE_SIZE:
        raise RuntimeError(
            "b12x compressed indexer page-size contract changed, got "
            f"{COMPRESSED_INDEX_PAGE_SIZE}; expected "
            f"{_B12X_COMPRESSED_INDEX_PAGE_SIZE}."
        )

    index_k_cache = _flatten_b12x_compressed_index_cache(kv_cache)
    expected_num_q_heads = int(q_fp8.shape[1])
    metadata = prepare_compressed_indexer_metadata(
        real_page_table=block_table,
        cache_seqlens_int32=seq_lens,
        page_size=COMPRESSED_INDEX_PAGE_SIZE,
        expected_num_q_heads=expected_num_q_heads,
        schedule_metadata=schedule_metadata,
        validate_raw_lengths=False,
    )
    logits = compressed_index_decode_logits_fp8(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        metadata=metadata,
        page_size=COMPRESSED_INDEX_PAGE_SIZE,
        expected_num_q_heads=expected_num_q_heads,
        preinitialize_invalid_logits=False,
    )
    topk_values = torch.empty(
        (int(q_fp8.shape[0]), int(topk_tokens)),
        dtype=torch.float32,
        device=q_fp8.device,
    )
    _, result = run_row_topk(
        row_logits=logits,
        lengths=metadata.cache_seqlens_int32,
        topk=topk_tokens,
        output_values=topk_values,
        output_indices=topk_indices,
    )
    return result


@eager_break_during_capture
def sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
    use_b12x_sparse_indexer: bool = False,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)

    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        if _b12x_sparse_indexer_requested(use_b12x_sparse_indexer):
            _ensure_b12x_sparse_indexer_supported()
            _ = torch.empty(
                values_spec[0],
                dtype=values_spec[1],
                device=hidden_states.device,
            )
            _ = torch.empty(
                scales_spec[0],
                dtype=scales_spec[1],
                device=hidden_states.device,
            )
            _ = torch.empty(
                (RADIX_TOPK_WORKSPACE_SIZE,),
                dtype=torch.uint8,
                device=hidden_states.device,
            )
        else:
            # Reserve workspace for indexer during profiling run.
            current_workspace_manager().get_simultaneous(
                values_spec, scales_spec, ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8)
            )

        # Dummy allocation to simulate for peak logits tensor memory during inference.
        # FP8 elements so elements == bytes
        max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        _ = torch.empty(
            max_logits_elems, dtype=torch.uint8, device=hidden_states.device
        )

        return sparse_attn_indexer_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_quant,
            q_scale,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
            skip_k_cache_insert,
            use_fp4_cache,
            use_b12x_sparse_indexer,
        )
    attn_metadata_narrowed = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata_narrowed, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata_narrowed.slot_mapping
    has_decode = attn_metadata_narrowed.num_decodes > 0
    has_prefill = attn_metadata_narrowed.num_prefills > 0
    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens

    # q_scale is required iff the FP4 cache path is enabled; the FP8 path
    # folds the Q scale into `weights` inside fused_indexer_q_rope_quant.
    if use_fp4_cache:
        assert q_scale is not None, "use_fp4_cache=True requires q_scale"
    else:
        assert q_scale is None, "q_scale must be None when use_fp4_cache=False"

    # During speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens. Truncate k to avoid
    # out-of-bounds reads in the kernel.
    num_tokens = slot_mapping.shape[0]
    if k is not None:
        k = k[:num_tokens]

    if not skip_k_cache_insert:
        # scale_fmt can be None, but the function expects str
        assert scale_fmt is not None
        assert not use_fp4_cache, "Unfused FP4 Insert is not supported yet"
        ops.indexer_k_quant_and_cache(
            k,
            kv_cache,
            slot_mapping,
            quant_block_size,
            scale_fmt,
        )

    topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata_narrowed.prefill
        assert prefill_metadata is not None

        # Layout switches between FP8 (head_dim bytes + 4-byte fp32 scale) and
        # MXFP4 (head_dim/2 bytes packed + head_dim/MXFP4_BLOCK_SIZE ue8m0
        # scales) based on use_fp4_cache.
        use_b12x_indexer = _use_b12x_sparse_indexer(use_b12x_sparse_indexer)
        if use_b12x_indexer and use_fp4_cache:
            raise RuntimeError(
                "b12x sparse indexer currently requires the FP8 indexer cache; "
                "disable use_fp4_indexer_cache or disable b12x sparse indexer."
            )
        b12x_indexer: Any = None
        if use_b12x_indexer:
            b12x_indexer = import_module("b12x.integration.indexer")
        else:
            workspace_manager = current_workspace_manager()
            values_spec, scales_spec = _gather_workspace_shapes(
                total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
            )
            k_quant_full, k_scale_full = workspace_manager.get_simultaneous(
                values_spec,
                scales_spec,
            )
        for chunk in prefill_metadata.chunks:
            q_slice = q_quant[chunk.token_start : chunk.token_end]
            q_scale_slice = (
                q_scale[chunk.token_start : chunk.token_end]
                if q_scale is not None
                else None
            )
            weights_slice = weights[chunk.token_start : chunk.token_end]
            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]
            if chunk.total_seq_lens <= 0:
                topk_indices.fill_(-1)
                continue

            if use_b12x_indexer:
                row_has_no_kv = chunk.cu_seqlen_ke <= chunk.cu_seqlen_ks
                b12x_cu_seqlen_ks = torch.where(
                    row_has_no_kv,
                    torch.zeros_like(chunk.cu_seqlen_ks),
                    chunk.cu_seqlen_ks,
                )
                b12x_cu_seqlen_ke = torch.where(
                    row_has_no_kv,
                    torch.ones_like(chunk.cu_seqlen_ke),
                    chunk.cu_seqlen_ke,
                )
                values_spec, scales_spec = _gather_workspace_shapes(
                    chunk.total_seq_lens,
                    head_dim,
                    fp8_dtype,
                    use_fp4_cache=False,
                )
                k_quant = torch.empty(
                    values_spec[0],
                    dtype=values_spec[1],
                    device=hidden_states.device,
                )
                k_scale = torch.empty(
                    scales_spec[0],
                    dtype=scales_spec[1],
                    device=hidden_states.device,
                )
            else:
                k_quant = k_quant_full[: chunk.total_seq_lens]
                k_scale = k_scale_full[: chunk.total_seq_lens]

            if not chunk.skip_kv_gather:
                ops.cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_quant,
                    k_scale,
                    chunk.block_table,
                    chunk.cu_seq_lens,
                )

            if use_b12x_indexer:
                assert b12x_indexer is not None
                k_scale_f32 = k_scale.view(torch.float32).flatten()
                k_fp8_b12x = (
                    k_quant.view(torch.float8_e4m3fn)
                    if k_quant.dtype == torch.uint8
                    else k_quant
                )
                topk_indices.copy_(
                    b12x_indexer.extend_tiled_topk(
                        q_fp8=q_slice,
                        weights=weights_slice,
                        kv_fp8=(k_fp8_b12x, k_scale_f32),
                        metadata=b12x_indexer.IndexerExtendMetadata(
                            k_start=b12x_cu_seqlen_ks,
                            k_end=b12x_cu_seqlen_ke,
                        ),
                        topk=topk_tokens,
                        supertile_k=_B12X_EXTEND_TOPK_SUPERTILE_K,
                    )
                )
                topk_indices.masked_fill_(row_has_no_kv[:, None], -1)
                _normalize_prefill_topk_to_req_relative(chunk, topk_indices)
                continue

            # DeepGEMM scalar-type tags (zero-copy): MXFP4 values → int8
            # (kPackedFP4), scales → int32 squeezed to 1-D kv_sf / 2-D q_sf.
            if use_fp4_cache:
                q_slice_cast = q_slice.view(torch.int8)
                k_quant_cast = k_quant.view(torch.int8)
                k_scale_cast = k_scale.view(torch.int32).squeeze(-1)
            else:
                q_slice_cast = q_slice
                k_quant_cast = k_quant
                k_scale_cast = k_scale.view(torch.float32).squeeze(-1)
            if current_platform.is_xpu():
                if q_scale_slice is not None:
                    raise RuntimeError("XPU fp8_mqa_logits does not support FP4 Q")
                logits = torch.ops.vllm.xpu_fp8_mqa_logits(
                    q_slice_cast,
                    k_quant_cast,
                    k_scale_cast,
                    weights_slice,
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                )
            else:
                logits = fp8_fp4_mqa_logits(
                    (q_slice_cast, q_scale_slice),
                    (k_quant_cast, k_scale_cast),
                    weights_slice,
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    clean_logits=False,
                )
            num_rows = logits.shape[0]

            ops.top_k_per_row_prefill(
                logits,
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )

    if has_decode:
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None
        use_b12x_indexer = _use_b12x_sparse_indexer(use_b12x_sparse_indexer)
        if use_b12x_indexer and use_fp4_cache:
            raise RuntimeError(
                "b12x sparse indexer currently requires the FP8 indexer cache; "
                "disable use_fp4_indexer_cache or disable b12x sparse indexer."
            )

        b12x_seq_lens = decode_metadata.seq_lens
        b12x_block_table = decode_metadata.block_table
        if b12x_seq_lens.dim() == 2:
            b12x_batch_size, b12x_next_n = b12x_seq_lens.shape
            if num_decode_tokens == b12x_batch_size * b12x_next_n:
                b12x_seq_lens = b12x_seq_lens.reshape(-1).contiguous()
                b12x_block_table = b12x_block_table.repeat_interleave(
                    b12x_next_n, dim=0
                ).contiguous()
        b12x_decode_supported = (
            use_b12x_indexer
            and not decode_metadata.requires_padding
            and b12x_seq_lens.dim() == 1
        )
        if use_b12x_indexer and (
            decode_metadata.requires_padding or b12x_seq_lens.dim() != 1
        ):
            raise RuntimeError(
                "b12x sparse indexer decode requires an unpadded rank-1 "
                "seq_lens contract after native-spec normalization; refusing "
                "to fall back to DeepGEMM. "
                f"requires_padding={decode_metadata.requires_padding}, "
                f"seq_lens_shape={tuple(decode_metadata.seq_lens.shape)}, "
                f"normalized_seq_lens_shape={tuple(b12x_seq_lens.shape)}, "
                f"num_decode_tokens={num_decode_tokens}."
            )

        if b12x_decode_supported:
            seq_lens = b12x_seq_lens[:num_decode_tokens].contiguous()
            block_table = b12x_block_table[:num_decode_tokens].contiguous()
            topk_indices = topk_indices_buffer[:num_decode_tokens, :topk_tokens]
            _run_b12x_compressed_decode_topk(
                q_fp8=q_quant[:num_decode_tokens].contiguous(),
                weights=weights[:num_decode_tokens].contiguous(),
                kv_cache=kv_cache,
                seq_lens=seq_lens,
                block_table=block_table,
                schedule_metadata=decode_metadata.schedule_metadata,
                topk_indices=topk_indices,
                topk_tokens=topk_tokens,
            )
            return topk_indices_buffer

        kv_cache = kv_cache_as_quant_view(kv_cache, head_dim, use_fp4_cache)
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens).
            # FP8 Q is float8_e4m3fn (pack_seq_triton's fp32 pad path is OK —
            # downstream context_lens masks stale slots). MXFP4 Q is two
            # uint8 tensors (values + ue8m0 scales) — use the dedicated uint8
            # packer with pad_byte=0 so padded slots dequantize to 0 and
            # can't produce NaN/Inf in the logits kernel.
            if q_scale is not None:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens, pad_value=0
                )
                padded_q_scale = pack_seq_triton(
                    q_scale[:num_decode_tokens], decode_lens, pad_value=0
                )
            else:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens
                )
                padded_q_scale = None
        else:
            padded_q_quant_decode_tokens = q_quant[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_quant.shape[1:]
            )
            if q_scale is not None:
                padded_q_scale = q_scale[:num_decode_tokens].reshape(
                    decode_lens.shape[0], -1, *q_scale.shape[1:]
                )
            else:
                padded_q_scale = None
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_quant_decode_tokens.shape[0]
        next_n = padded_q_quant_decode_tokens.shape[1]
        num_padded_tokens = batch_size * next_n
        seq_lens = decode_metadata.seq_lens[:batch_size]
        # seq_lens is always 2D: (B, next_n) for native spec decode, (B, 1)
        # otherwise. deep_gemm fp8_fp4_paged_mqa_logits requires 2D context_lens;
        # the downstream topk kernels accept both 1D and 2D.
        padded_q_quant_cast = (
            padded_q_quant_decode_tokens.view(torch.int8)
            if use_fp4_cache
            else padded_q_quant_decode_tokens
        )
        if current_platform.is_xpu():
            if padded_q_scale is not None:
                raise RuntimeError("XPU fp8_paged_mqa_logits does not support FP4 Q")
            seq_lens_xpu = (
                seq_lens[:, -1].contiguous() if seq_lens.ndim == 2 else seq_lens
            )
            logits = torch.ops.vllm.xpu_fp8_paged_mqa_logits(
                padded_q_quant_cast,
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens_xpu,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len,
            )
        else:
            logits = fp8_fp4_paged_mqa_logits(
                (padded_q_quant_cast, padded_q_scale),
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len=max_model_len,
                clean_logits=False,
            )
        num_rows = logits.shape[0]
        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        if current_platform.is_cuda() and topk_tokens in (512, 1024, 2048):
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.persistent_topk(
                logits,
                seq_lens,
                topk_indices,
                topk_workspace,
                topk_tokens,
                attn_metadata_narrowed.max_seq_len,
            )
        else:
            ops.top_k_per_row_decode(
                logits,
                next_n,
                seq_lens,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[: topk_indices.shape[0], : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


def sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
    use_b12x_sparse_indexer: bool = False,
) -> torch.Tensor:
    return topk_indices_buffer


direct_register_custom_op(
    op_name="sparse_attn_indexer",
    op_func=sparse_attn_indexer,
    mutates_args=["topk_indices_buffer"],
    fake_impl=sparse_attn_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)


@CustomOp.register("sparse_attn_indexer")
class SparseAttnIndexer(CustomOp):
    """Sparse Attention Indexer Custom Op Layer. This layer is extracted as a
    separate custom op since it involves heavy custom kernels like `mqa_logits`,
    `paged_mqa_logits` and `top_k_per_row`, etc. Those kernels maybe requires
    specific memory layout or implementation for different hardware backends to
    achieve optimal performance.

    For now, the default native path will use CUDA backend path. Other platform
    may requires add the corresponding Custom Op name `sparse_attn_indexer` to
    `custom_ops` in `CompilationConfig` to enable the platform specific path.
    """

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.skip_k_cache_insert = skip_k_cache_insert
        self.use_fp4_cache = use_fp4_cache
        self.use_b12x_sparse_indexer = bool(envs.VLLM_USE_B12X_SPARSE_INDEXER)
        if self.use_b12x_sparse_indexer:
            _ensure_b12x_sparse_indexer_supported()
            if self.use_fp4_cache:
                raise RuntimeError(
                    "VLLM_USE_B12X_SPARSE_INDEXER requires the FP8/C4 indexer "
                    "cache; disable use_fp4_indexer_cache."
                )
        elif current_platform.is_cuda() and not has_deep_gemm():
            raise RuntimeError(
                "Sparse Attention Indexer CUDA op requires DeepGEMM to be installed."
            )

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        if current_platform.is_cuda() or current_platform.is_xpu():
            return self.forward_cuda(hidden_states, q_quant, k, weights)
        elif current_platform.is_rocm():
            return self.forward_hip(hidden_states, q_quant, k, weights)
        else:
            raise NotImplementedError(
                "SparseAttnIndexer native forward is only implemented for "
                "CUDA, ROCm and XPU platforms."
            )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        # FP8 path: single tensor (per-token scale is folded into `weights`).
        # FP4 path: (values, scales) tuple with scales required by the kernel.
        if isinstance(q_quant, tuple):
            q_values, q_scale = q_quant
        else:
            q_values, q_scale = q_quant, None
        return torch.ops.vllm.sparse_attn_indexer(
            hidden_states,
            _encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache,
            q_values,
            q_scale,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            self.skip_k_cache_insert,
            self.use_fp4_cache,
            self.use_b12x_sparse_indexer,
        )

    def forward_xpu(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        return self.forward_cuda(hidden_states, q_fp8, k, weights)

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        assert not self.use_fp4_cache, "AMD platform doesn't support fp4 cache yet"
        assert isinstance(q_quant, torch.Tensor), (
            "AMD sparse_attn_indexer expects a single FP8 q_quant tensor"
        )
        if rocm_aiter_ops.is_enabled():
            return torch.ops.vllm.rocm_aiter_sparse_attn_indexer(
                hidden_states,
                _encode_layer_name(self.k_cache.prefix),
                self.k_cache.kv_cache,
                q_quant,
                k,
                weights,
                self.quant_block_size,
                self.scale_fmt,
                self.topk_tokens,
                self.head_dim,
                self.max_model_len,
                self.max_total_seq_len,
                self.topk_indices_buffer,
                skip_k_cache_insert=self.skip_k_cache_insert,
            )
        raise RuntimeError(
            "Sparse attention indexer ROCm path is only supported on AITER. "
            "Please enable aiter with VLLM_ROCM_USE_AITER=1"
        )
