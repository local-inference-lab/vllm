# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers."""

import os

import torch

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import get_current_vllm_config_or_none
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.b12x_contract import (
    b12x_sparse_indexer_active_for_config,
)
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

if current_platform.is_cuda_alike():
    from vllm import _custom_ops as ops
elif current_platform.is_xpu():
    from vllm._xpu_ops import xpu_ops

logger = init_logger(__name__)

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024


def fp8_mqa_logits(
    q_fp8: torch.Tensor,
    kv_fp8: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    *,
    clean_logits: bool,
) -> torch.Tensor:
    return fp8_fp4_mqa_logits(
        (q_fp8, None),
        kv_fp8,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        clean_logits=clean_logits,
    )


def fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_metadata,
    *,
    max_model_len: int,
    clean_logits: bool,
) -> torch.Tensor:
    return fp8_fp4_paged_mqa_logits(
        (q_fp8, None),
        kv_cache,
        weights,
        seq_lens,
        block_table,
        schedule_metadata,
        max_model_len=max_model_len,
        clean_logits=clean_logits,
    )

_SGL_KERNEL_FAST_TOPK_AVAILABLE: bool | None = None
_SGL_KERNEL_FAST_TOPK_V2_AVAILABLE: bool | None = None
_SGL_KERNEL_FAST_TOPK_TRANSFORM_AVAILABLE: bool | None = None
_B12X_INDEXER_ARENAS: dict[tuple[object, ...], object] = {}
_B12X_INDEXER_WORKSPACES: dict[tuple[object, ...], object] = {}
_B12X_INDEXER_PHANTOMS: dict[tuple[object, ...], dict[str, object]] = {}
_B12X_INDEXER_EXTEND_ARENAS: dict[tuple[object, ...], object] = {}
_B12X_INDEXER_EXTEND_WORKSPACES: dict[tuple[object, ...], object] = {}
_USE_SGL_KERNEL_FAST_TOPK = bool(int(os.getenv("VLLM_USE_SGL_KERNEL_FAST_TOPK", "0")))
_USE_SGL_KERNEL_FAST_TOPK_V2 = bool(
    int(os.getenv("VLLM_USE_SGL_KERNEL_FAST_TOPK_V2", "0"))
)
_USE_SGL_KERNEL_FAST_TOPK_TRANSFORM = bool(
    int(os.getenv("VLLM_USE_SGL_KERNEL_FAST_TOPK_TRANSFORM", "0"))
)
_B12X_EXTEND_TOPK_SUPERTILE_K = int(
    os.getenv(
        "VLLM_B12X_NSA_EXTEND_TOPK_SUPERTILE_K",
        "32768",
    )
)
_B12X_INDEXER_EXTEND_MAX_Q = 8192
_B12X_INDEXER_EXTEND_MAX_BATCH = 64
_B12X_INDEXER_EXTEND_MAX_KV_ROWS = 0
_B12X_INDEXER_EXTEND_TILE_LOGITS_K_ROWS = int(
    os.getenv(
        "VLLM_B12X_INDEXER_EXTEND_TILE_LOGITS_K_ROWS",
        str(_B12X_EXTEND_TOPK_SUPERTILE_K),
    )
)
_B12X_INDEXER_DECODE_MAX_Q_ROWS = 0
_DEBUG_NSA_INDEXER = os.getenv("VLLM_DEBUG_NSA_INDEXER", "0") == "1"
_DEBUG_NSA_INDEXER_FILE = os.getenv(
    "VLLM_NSA_INDEXER_DEBUG_FILE", "/diag/nsa_indexer_debug.log"
)
_DEBUG_NSA_INDEXER_MARKER = os.getenv("VLLM_NSA_INDEXER_DEBUG_MARKER", "")
_DEBUG_NSA_INDEXER_MAX = int(os.getenv("VLLM_NSA_INDEXER_DEBUG_MAX", "512"))
_debug_nsa_indexer_count = 0


def _debug_nsa_indexer_prefill(stage: str, chunk, topk_indices: torch.Tensor) -> None:
    if not _DEBUG_NSA_INDEXER:
        return
    if _DEBUG_NSA_INDEXER_MARKER and not os.path.exists(_DEBUG_NSA_INDEXER_MARKER):
        return

    global _debug_nsa_indexer_count
    if _debug_nsa_indexer_count >= _DEBUG_NSA_INDEXER_MAX:
        return

    try:
        rows = min(int(topk_indices.shape[0]), 16)
        cols = min(int(topk_indices.shape[1]), 16)
        topk_sample = topk_indices[:rows, :cols].detach().cpu().tolist()
        cu_seq_lens = getattr(chunk, "cu_seq_lens", None)
        cu_seq_lens_cpu = (
            cu_seq_lens.detach().cpu().tolist() if cu_seq_lens is not None else None
        )
        unequal_pairs = None
        first_pair = None
        if cu_seq_lens_cpu is not None and len(cu_seq_lens_cpu) >= 3:
            q0 = int(cu_seq_lens_cpu[1]) - int(cu_seq_lens_cpu[0])
            q1 = int(cu_seq_lens_cpu[2]) - int(cu_seq_lens_cpu[1])
            if q0 == q1 and q0 > 0 and topk_indices.shape[0] >= 2 * q0:
                left = topk_indices[:q0, :cols]
                right = topk_indices[q0 : 2 * q0, :cols]
                unequal = (left != right).any(dim=1)
                unequal_pairs = int(unequal.sum().item())
                first_pair = [
                    left[: min(q0, 4)].detach().cpu().tolist(),
                    right[: min(q0, 4)].detach().cpu().tolist(),
                ]

        payload = (
            f"stage={stage} rows={tuple(topk_indices.shape)} "
            f"token=({getattr(chunk, 'token_start', None)},"
            f"{getattr(chunk, 'token_end', None)}) "
            f"total_seq_lens={getattr(chunk, 'total_seq_lens', None)} "
            f"cu_seq_lens={cu_seq_lens_cpu} unequal_pairs={unequal_pairs} "
            f"first_pair={first_pair} topk_sample={topk_sample}"
        )
        with open(_DEBUG_NSA_INDEXER_FILE, "a", encoding="utf-8") as f:
            f.write(f"NSA_INDEXER_DEBUG[{_debug_nsa_indexer_count}] {payload}\n")
        _debug_nsa_indexer_count += 1
    except Exception as e:
        try:
            with open(_DEBUG_NSA_INDEXER_FILE, "a", encoding="utf-8") as f:
                f.write(
                    "NSA_INDEXER_DEBUG_ERROR "
                    f"stage={stage} error={type(e).__name__}: {e}\n"
                )
        except Exception:
            pass


def _normalize_prefill_topk_to_req_relative(chunk, topk_indices: torch.Tensor) -> None:
    """Convert packed prefill workspace offsets to per-request token offsets."""
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


def _get_runtime_vllm_config() -> object | None:
    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is not None:
        return vllm_config
    if is_forward_context_available():
        return getattr(get_forward_context(), "vllm_config", None)
    return None


def _use_b12x_sparse_indexer() -> bool:
    return (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(120)
        and (
            envs.VLLM_USE_B12X_SPARSE_INDEXER
            or b12x_sparse_indexer_active_for_config(_get_runtime_vllm_config())
        )
    )


def _has_sgl_kernel_fast_topk() -> bool:
    if not _USE_SGL_KERNEL_FAST_TOPK:
        return False
    global _SGL_KERNEL_FAST_TOPK_AVAILABLE
    if _SGL_KERNEL_FAST_TOPK_AVAILABLE is not None:
        return _SGL_KERNEL_FAST_TOPK_AVAILABLE
    try:
        import sgl_kernel  # noqa: F401

        # Importing sgl_kernel registers torch.ops.sgl_kernel.fast_topk.
        _SGL_KERNEL_FAST_TOPK_AVAILABLE = hasattr(torch.ops, "sgl_kernel") and hasattr(
            torch.ops.sgl_kernel, "fast_topk"
        )
    except Exception as exc:
        logger.info("sgl_kernel fast_topk unavailable, using vLLM topk: %s", exc)
        _SGL_KERNEL_FAST_TOPK_AVAILABLE = False
    return _SGL_KERNEL_FAST_TOPK_AVAILABLE


def _has_sgl_kernel_fast_topk_v2() -> bool:
    if not _USE_SGL_KERNEL_FAST_TOPK_V2:
        return False
    global _SGL_KERNEL_FAST_TOPK_V2_AVAILABLE
    if _SGL_KERNEL_FAST_TOPK_V2_AVAILABLE is not None:
        return _SGL_KERNEL_FAST_TOPK_V2_AVAILABLE
    try:
        import sgl_kernel  # noqa: F401

        _SGL_KERNEL_FAST_TOPK_V2_AVAILABLE = hasattr(sgl_kernel, "fast_topk_v2")
    except Exception as exc:
        logger.info("sgl_kernel fast_topk_v2 unavailable, using vLLM topk: %s", exc)
        _SGL_KERNEL_FAST_TOPK_V2_AVAILABLE = False
    return _SGL_KERNEL_FAST_TOPK_V2_AVAILABLE


def _has_sgl_kernel_fast_topk_transform() -> bool:
    if not _USE_SGL_KERNEL_FAST_TOPK_TRANSFORM:
        return False
    global _SGL_KERNEL_FAST_TOPK_TRANSFORM_AVAILABLE
    if _SGL_KERNEL_FAST_TOPK_TRANSFORM_AVAILABLE is not None:
        return _SGL_KERNEL_FAST_TOPK_TRANSFORM_AVAILABLE
    try:
        import sgl_kernel

        _SGL_KERNEL_FAST_TOPK_TRANSFORM_AVAILABLE = hasattr(
            sgl_kernel, "fast_topk_transform_fused"
        )
    except Exception as exc:
        logger.info(
            "sgl_kernel fast_topk_transform_fused unavailable, using vLLM topk: %s",
            exc,
        )
        _SGL_KERNEL_FAST_TOPK_TRANSFORM_AVAILABLE = False
    return _SGL_KERNEL_FAST_TOPK_TRANSFORM_AVAILABLE


def _b12x_prefill_max_num_reqs(attn_metadata, chunk, q_rows: int) -> int:
    num_reqs = getattr(attn_metadata, "num_reqs", None)
    if num_reqs is not None:
        return max(int(num_reqs), 1)

    cu_seq_lens = getattr(chunk, "cu_seq_lens", None)
    if cu_seq_lens is not None:
        return max(int(cu_seq_lens.numel()) - 1, 1)

    return max(int(q_rows), 1)


def _get_b12x_indexer_workspace(
    *,
    q_fp8: torch.Tensor,
    index_k_cache: torch.Tensor,
    topk_tokens: int,
    max_num_reqs: int,
    max_model_len: int,
):
    from b12x.attention.workspace import (
        B12XAttentionArena,
        B12XAttentionArenaCaps,
        B12XAttentionWorkspaceContract,
    )

    page_size = int(index_k_cache.shape[1])
    max_page_table_width = max(1, (int(max_model_len) + page_size - 1) // page_size)
    paged_max_q_rows = max(
        _B12X_INDEXER_DECODE_MAX_Q_ROWS,
        int(max_num_reqs),
        int(q_fp8.shape[0]),
        1,
    )
    indexer_num_q_heads = int(q_fp8.shape[1])
    head_dim = 576
    v_head_dim = 512
    key = (
        q_fp8.device.type,
        q_fp8.device.index,
        q_fp8.dtype,
        index_k_cache.dtype,
        indexer_num_q_heads,
        int(q_fp8.shape[2]),
        topk_tokens,
        max_page_table_width,
        paged_max_q_rows,
        page_size,
    )
    workspace = _B12X_INDEXER_WORKSPACES.get(key)
    if workspace is not None:
        return workspace

    arena = _B12X_INDEXER_ARENAS.get(key)
    if arena is None:
        caps = B12XAttentionArenaCaps(
            device=q_fp8.device,
            dtype=torch.bfloat16,
            kv_dtype=index_k_cache.dtype,
            num_q_heads=1,
            indexer_num_q_heads=indexer_num_q_heads,
            head_dim=head_dim,
            max_v_head_dim=v_head_dim,
            topk=topk_tokens,
            max_page_table_width=max_page_table_width,
            extend_max_total_q=1,
            extend_max_batch=1,
            extend_max_kv_rows=0,
            paged_max_q_rows=paged_max_q_rows,
            paged_max_batch=paged_max_q_rows,
            page_size=page_size,
        )
        arena = B12XAttentionArena.allocate(caps)
        _B12X_INDEXER_ARENAS[key] = arena

    contract = B12XAttentionWorkspaceContract(
        mode="decode",
        max_total_q=paged_max_q_rows,
        max_batch=paged_max_q_rows,
        max_paged_q_rows=paged_max_q_rows,
        max_kv_rows=0,
        v_head_dim=v_head_dim,
        indexer_num_q_heads=indexer_num_q_heads,
        max_page_table_width=max_page_table_width,
    )
    workspace = arena.make_workspace(contract, use_cuda_graph=True)
    _B12X_INDEXER_WORKSPACES[key] = workspace
    return workspace


def _get_b12x_indexer_phantoms(
    *,
    q_fp8: torch.Tensor,
    index_k_cache: torch.Tensor,
    max_num_reqs: int,
    max_model_len: int,
):
    """Build (and cache) phantom tensors for stable NSA-indexer host-launcher
    cache keys (#87 — Path C).

    Without phantoms, ``paged_decode_logits()`` keys its
    compiled-kernel cache by the actual q_fp8 / page_table / etc. shapes,
    so every cudagraph capture-size variant (1, 4, 8, ...) and every
    eager-mode batch size triggers a fresh CUTLASS compile (~5-15 min
    per shape). On cold cache that compounds into the 60-90 min wall
    that killed alice's DCP=8 boots even after #83 v2 lifted the NCCL
    watchdog.

    With phantoms (zero-strided base tensors registered as the contract
    via ``make_indexer_contract_phantoms``), the cache key is fixed
    to the (max_q_rows, num_heads, max_pages, page_size) tuple — all
    capture sizes ≤ max_q_rows hit the same compiled kernel. The phantom
    tensors are tiny (zero-stride views of `torch.empty(1, ...)`), so
    the cache footprint is negligible.

    See `b12x.attention.indexer.api.make_indexer_contract_phantoms`
    docstring: "avoid CUTLASS recompilation when batch size varies".
    """
    from b12x.integration.indexer import make_indexer_contract_phantoms

    page_size = int(index_k_cache.shape[1])
    max_page_table_width = max(1, (int(max_model_len) + page_size - 1) // page_size)
    max_q_rows = max(
        _B12X_INDEXER_DECODE_MAX_Q_ROWS,
        int(max_num_reqs),
        int(q_fp8.shape[0]),
        1,
    )
    indexer_num_q_heads = int(q_fp8.shape[1])
    key = (
        q_fp8.device.type,
        q_fp8.device.index,
        indexer_num_q_heads,
        max_page_table_width,
        max_q_rows,
        page_size,
    )
    phantoms = _B12X_INDEXER_PHANTOMS.get(key)
    if phantoms is not None:
        return phantoms
    phantoms = make_indexer_contract_phantoms(
        max_q_rows=max_q_rows,
        num_heads=indexer_num_q_heads,
        max_pages=max_page_table_width,
        page_size=page_size,
        device=q_fp8.device,
    )
    _B12X_INDEXER_PHANTOMS[key] = phantoms
    return phantoms


def _get_b12x_indexer_extend_workspace(
    *,
    q_fp8: torch.Tensor,
    index_k_cache: torch.Tensor,
    topk_tokens: int,
    max_num_reqs: int,
    max_model_len: int,
    total_seq_lens: int,
    head_dim: int,
):
    from b12x.attention.workspace import (
        B12XAttentionArena,
        B12XAttentionArenaCaps,
        B12XAttentionWorkspaceContract,
    )

    page_size = int(index_k_cache.shape[1])
    q_rows = max(1, int(q_fp8.shape[0]))
    k_rows = max(1, int(total_seq_lens))
    indexer_num_q_heads = int(q_fp8.shape[1])
    v_head_dim = 512
    extend_topk_supertile_k = _B12X_EXTEND_TOPK_SUPERTILE_K
    from b12x.attention.indexer import (
        resolve_extend_prefill_block_k,
    )

    prefill_block_k = resolve_extend_prefill_block_k(
        valid_q_rows=q_rows,
        k_rows=k_rows,
        num_heads=indexer_num_q_heads,
    )
    if prefill_block_k is None:
        # tiled_topk explicitly forces the prefill scorer for small Q.
        prefill_block_k = 256
    # Match SGLang's model: compile against a stable capacity contract and pass
    # live sequence lengths as runtime metadata. If q/k capacity follows the
    # current chunk, b12x/CUTE sees different tensor metadata and compiles
    # another host launcher during long prefill.
    capacity_q_rows = max(q_rows, _B12X_INDEXER_EXTEND_MAX_Q)
    capacity_k_rows = max(k_rows, int(prefill_block_k))
    if _B12X_INDEXER_EXTEND_MAX_KV_ROWS > 0:
        capacity_k_rows = max(capacity_k_rows, _B12X_INDEXER_EXTEND_MAX_KV_ROWS)
    else:
        capacity_k_rows = max(capacity_k_rows, int(max_model_len))
    capacity_batch = max(
        1,
        min(
            max(int(max_num_reqs), _B12X_INDEXER_EXTEND_MAX_BATCH),
            capacity_q_rows,
        ),
    )
    tile_logits_k_rows = min(
        max(capacity_k_rows, 0),
        max(_B12X_INDEXER_EXTEND_TILE_LOGITS_K_ROWS, 0),
    )
    key = (
        q_fp8.device.type,
        q_fp8.device.index,
        q_fp8.dtype,
        index_k_cache.dtype,
        indexer_num_q_heads,
        int(q_fp8.shape[2]),
        int(head_dim),
        topk_tokens,
        capacity_q_rows,
        capacity_k_rows,
        capacity_batch,
        page_size,
        extend_topk_supertile_k,
        tile_logits_k_rows,
    )
    workspace = _B12X_INDEXER_EXTEND_WORKSPACES.get(key)
    if workspace is not None:
        return workspace

    arena = _B12X_INDEXER_EXTEND_ARENAS.get(key)
    if arena is None:
        caps = B12XAttentionArenaCaps(
            device=q_fp8.device,
            dtype=torch.bfloat16,
            kv_dtype=index_k_cache.dtype,
            num_q_heads=1,
            indexer_num_q_heads=indexer_num_q_heads,
            head_dim=max(int(head_dim), v_head_dim + 64),
            max_v_head_dim=v_head_dim,
            topk=topk_tokens,
            max_page_table_width=max(1, topk_tokens),
            extend_max_total_q=capacity_q_rows,
            extend_max_batch=capacity_batch,
            extend_max_kv_rows=capacity_k_rows,
            paged_max_q_rows=1,
            paged_max_batch=1,
            page_size=page_size,
            reserve_extend_indexer_logits=False,
            extend_indexer_tile_logits_k_rows=tile_logits_k_rows,
        )
        arena = B12XAttentionArena.allocate(caps)
        _B12X_INDEXER_EXTEND_ARENAS[key] = arena

    contract = B12XAttentionWorkspaceContract(
        mode="extend",
        max_total_q=capacity_q_rows,
        max_batch=capacity_batch,
        max_paged_q_rows=1,
        max_kv_rows=capacity_k_rows,
        v_head_dim=v_head_dim,
        indexer_num_q_heads=indexer_num_q_heads,
        max_page_table_width=max(1, topk_tokens),
    )
    workspace = arena.make_workspace(contract, use_cuda_graph=True)
    _B12X_INDEXER_EXTEND_WORKSPACES[key] = workspace
    return workspace


@eager_break_during_capture
def sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)

    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        # Reserve workspace for indexer during profiling run
        max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        current_workspace_manager().get_simultaneous(
            ((total_seq_lens, head_dim), torch.float8_e4m3fn),
            ((total_seq_lens, 4), torch.uint8),
            ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            ((max_logits_elems,), torch.uint8),
            ((hidden_states.shape[0],), torch.int32),
            ((hidden_states.shape[0], topk_tokens), torch.float32),
        )

        return sparse_attn_indexer_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_fp8,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
        )
    attn_metadata_narrowed = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata_narrowed, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata_narrowed.slot_mapping
    has_decode = attn_metadata_narrowed.num_decodes > 0
    has_prefill = attn_metadata_narrowed.num_prefills > 0
    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens

    # During speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens. Truncate k to avoid
    # out-of-bounds reads in the kernel.
    num_tokens = slot_mapping.shape[0]
    k = k[:num_tokens]

    # scale_fmt can be None, but the function expects str
    assert scale_fmt is not None
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

        # Get the full shared workspace buffers once (will allocate on first use)
        workspace_manager = current_workspace_manager()
        use_b12x_indexer = _use_b12x_sparse_indexer()
        b12x_indexer = None
        if use_b12x_indexer:
            from b12x.integration import indexer as b12x_indexer

        if not use_b12x_indexer:
            k_fp8_full, k_scale_full = workspace_manager.get_simultaneous(
                ((total_seq_lens, head_dim), fp8_dtype),
                ((total_seq_lens, 4), torch.uint8),
            )
        for chunk in prefill_metadata.chunks:
            q_chunk = q_fp8[chunk.token_start : chunk.token_end]
            weights_chunk = weights[chunk.token_start : chunk.token_end]
            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]
            if use_b12x_indexer and chunk.total_seq_lens <= 0:
                _debug_nsa_indexer_prefill(
                    "b12x_extend_empty_local_kv", chunk, topk_indices
                )
                continue

            tile_logits = lengths = topk_values = None
            candidate_values = candidate_indices = None
            b12x_extend_phantoms = None
            if use_b12x_indexer:
                b12x_max_num_reqs = _b12x_prefill_max_num_reqs(
                    attn_metadata_narrowed, chunk, int(q_chunk.shape[0])
                )
                b12x_workspace = _get_b12x_indexer_extend_workspace(
                    q_fp8=q_chunk,
                    index_k_cache=kv_cache,
                    topk_tokens=topk_tokens,
                    max_num_reqs=b12x_max_num_reqs,
                    max_model_len=max_model_len,
                    total_seq_lens=chunk.total_seq_lens,
                    head_dim=head_dim,
                )
                k_fp8, k_scale = b12x_workspace.get_indexer_gather_outputs(
                    row_count=chunk.total_seq_lens
                )
                tile_logits = b12x_workspace.get_indexer_extend_tile_logits()
                lengths = b12x_workspace.get_indexer_extend_lengths(
                    row_count=q_chunk.shape[0]
                )
                topk_values, topk_indices_out = (
                    b12x_workspace.get_indexer_extend_topk_buffers(
                        row_count=q_chunk.shape[0]
                    )
                )
                candidate_values, candidate_indices = (
                    b12x_workspace.get_indexer_extend_candidate_buffers()
                )
                # Thread workspace-side phantoms so the extend NSA indexer
                # kernels compile against the stable arena contract.
                b12x_extend_phantoms = (
                    b12x_workspace.get_indexer_contract_phantoms()
                )
            else:
                k_fp8 = k_fp8_full[: chunk.total_seq_lens]
                k_scale = k_scale_full[: chunk.total_seq_lens]
                topk_indices_out = topk_indices

            if not chunk.skip_kv_gather:
                ops.cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_fp8,
                    k_scale,
                    chunk.block_table,
                    chunk.cu_seq_lens,
                )

            k_scale_f32 = k_scale.view(torch.float32).flatten()
            k_fp8_b12x = (
                k_fp8.view(torch.float8_e4m3fn)
                if k_fp8.dtype == torch.uint8
                else k_fp8
            )
            if use_b12x_indexer:
                assert b12x_indexer is not None
                row_has_no_kv = chunk.cu_seqlen_ke <= chunk.cu_seqlen_ks
                # DCP can give a rank no local KV for early global query rows.
                # b12x tiled top-k expects a non-empty range, so score a dummy
                # local row and restore the invalid result after the call.
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
                topk_indices.copy_(
                    b12x_indexer.extend_tiled_topk(
                        q_fp8=q_chunk,
                        weights=weights_chunk,
                        kv_fp8=(k_fp8_b12x, k_scale_f32),
                        metadata=(
                            b12x_indexer.IndexerExtendMetadata(
                                k_start=b12x_cu_seqlen_ks,
                                k_end=b12x_cu_seqlen_ke,
                            )
                        ),
                        topk=topk_tokens,
                        contract_phantoms=b12x_extend_phantoms,
                        tile_logits=tile_logits,
                        lengths=lengths,
                        output_values=topk_values,
                        output_indices=topk_indices_out,
                        candidate_values=candidate_values,
                        candidate_indices=candidate_indices,
                        supertile_k=_B12X_EXTEND_TOPK_SUPERTILE_K,
                    )
                )
                topk_indices.masked_fill_(row_has_no_kv[:, None], -1)
                _normalize_prefill_topk_to_req_relative(chunk, topk_indices)
                _debug_nsa_indexer_prefill(
                    "b12x_extend_tiled_topk", chunk, topk_indices
                )
                continue
            else:
                logits = fp8_mqa_logits(
                    q_chunk,
                    (k_fp8, k_scale_f32),
                    weights_chunk,
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    clean_logits=False,
                )
            num_rows = logits.shape[0]

            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]

            if current_platform.is_xpu():
                xpu_ops.top_k_per_row_prefill(  # type: ignore[attr-defined]
                    logits,
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )
            else:
                torch.ops._C.top_k_per_row_prefill(
                    logits,
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )
            _normalize_prefill_topk_to_req_relative(chunk, topk_indices)
            _debug_nsa_indexer_prefill("fallback_prefill_topk", chunk, topk_indices)

    if has_decode:
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None
        decode_lens = decode_metadata.decode_lens
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
            _use_b12x_sparse_indexer()
            and not decode_metadata.requires_padding
            and b12x_seq_lens.dim() == 1
        )
        if b12x_decode_supported:
            from b12x.integration.indexer import (
                IndexerPagedDecodeMetadata,
                paged_decode_logits,
            )

            seq_lens = b12x_seq_lens[:num_decode_tokens].contiguous()
            block_table = b12x_block_table[:num_decode_tokens].contiguous()
            index_k_cache = kv_cache.view(kv_cache.shape[0], -1)
            workspace = _get_b12x_indexer_workspace(
                q_fp8=q_fp8[:num_decode_tokens],
                index_k_cache=kv_cache,
                topk_tokens=topk_tokens,
                max_num_reqs=attn_metadata_narrowed.num_reqs,
                max_model_len=max_model_len,
            )
            # Pass contract phantoms so the b12x indexer caches its compiled
            # kernel by the stable arena contract instead of per-call shape.
            contract_phantoms = _get_b12x_indexer_phantoms(
                q_fp8=q_fp8[:num_decode_tokens],
                index_k_cache=kv_cache,
                max_num_reqs=attn_metadata_narrowed.num_reqs,
                max_model_len=max_model_len,
            )
            logits = paged_decode_logits(
                q_fp8=q_fp8[:num_decode_tokens],
                weights=weights[:num_decode_tokens],
                index_k_cache=index_k_cache,
                metadata=IndexerPagedDecodeMetadata(
                    real_page_table=block_table,
                    cache_seqlens_int32=seq_lens,
                    paged_mqa_schedule_metadata=decode_metadata.schedule_metadata,
                ),
                page_size=kv_cache.shape[1],
                contract_phantoms=contract_phantoms,
                workspace=workspace,
            )
            next_n = 1
            num_padded_tokens = num_decode_tokens

        if not b12x_decode_supported:
            # kv_cache shape requirement for DeepGEMM:
            # [num_block, block_size, n_head, head_dim]. We only have
            # [num_block, block_size, head_dim].
            kv_cache = kv_cache.unsqueeze(-2)
            if decode_metadata.requires_padding:
                # pad in edge case where we have short chunked prefill length <
                # decode_threshold since we unstrictly split
                # prefill and decode by decode_threshold
                # (currently set to 1 + speculative tokens)
                padded_q_fp8_decode_tokens = pack_seq_triton(
                    q_fp8[:num_decode_tokens], decode_lens
                )
            else:
                padded_q_fp8_decode_tokens = q_fp8[:num_decode_tokens].reshape(
                    decode_lens.shape[0], -1, *q_fp8.shape[1:]
                )
            # TODO: move and optimize below logic with triton kernels
            batch_size = padded_q_fp8_decode_tokens.shape[0]
            next_n = padded_q_fp8_decode_tokens.shape[1]
            num_padded_tokens = batch_size * next_n
            seq_lens = decode_metadata.seq_lens[:batch_size]
            # DeepGEMM 2.5+sm120 hard-requires `context_lens` to be 2D
            # `(B, next_n)` per `csrc/apis/attention.hpp:352`. The older comment
            # below (kept for context) claimed the kernel accepted 1D too;
            # that's no longer true — coerce to 2D here to match the kernel
            # contract. For non-spec decode (next_n=1) this is a free reshape;
            # for spec decode (next_n>1) we replicate the per-batch context
            # length across all next_n query positions so the scheduler sees
            # the right work shape.
            # (Was: "seq_lens is (B, next_n) for native spec decode, (B,)
            # otherwise. fp8_paged_mqa_logits and all topk kernels accept
            # both shapes.")
            if seq_lens.dim() == 1:
                seq_lens_2d = seq_lens.unsqueeze(-1).expand(-1, next_n).contiguous()
            else:
                seq_lens_2d = seq_lens
            logits = fp8_paged_mqa_logits(
                padded_q_fp8_decode_tokens,
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens_2d,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len=max_model_len,
                clean_logits=False,
            )
        num_rows = logits.shape[0]
        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        if (
            b12x_decode_supported
            and current_platform.is_cuda()
            and topk_tokens == 2048
            and decode_metadata.page_table_1 is not None
            and decode_metadata.cu_seqlens_q is not None
            and seq_lens.dim() == 1
            and _has_sgl_kernel_fast_topk_transform()
        ):
            import sgl_kernel

            topk_indices = sgl_kernel.fast_topk_transform_fused(
                logits,
                seq_lens,
                decode_metadata.page_table_1,
                decode_metadata.cu_seqlens_q,
                topk_tokens,
                row_starts=None,
            )
            topk_indices_buffer[:num_padded_tokens, :topk_tokens] = topk_indices
        elif (
            b12x_decode_supported
            and current_platform.is_cuda()
            and topk_tokens == 2048
            and _has_sgl_kernel_fast_topk_v2()
        ):
            import sgl_kernel

            topk_indices = sgl_kernel.fast_topk_v2(
                logits, seq_lens, topk_tokens, row_starts=None
            )
            topk_indices_buffer[:num_padded_tokens, :topk_tokens] = topk_indices
        elif (
            b12x_decode_supported
            and current_platform.is_cuda()
            and topk_tokens == 2048
            and _has_sgl_kernel_fast_topk()
        ):
            torch.ops.sgl_kernel.fast_topk(logits, topk_indices, seq_lens, None)
        elif current_platform.is_cuda():
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
            if current_platform.is_xpu():
                xpu_ops.top_k_per_row_decode(  # type: ignore[attr-defined]
                    logits,
                    next_n,
                    seq_lens,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )
            else:
                torch.ops._C.top_k_per_row_decode(
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
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
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
        if (
            current_platform.is_cuda()
            and not has_deep_gemm()
            and not _use_b12x_sparse_indexer()
        ):
            raise RuntimeError(
                "Sparse Attention Indexer CUDA op requires DeepGEMM to be installed."
            )

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        if current_platform.is_cuda() or current_platform.is_xpu():
            return self.forward_cuda(hidden_states, q_fp8, k, weights)
        elif current_platform.is_rocm():
            return self.forward_hip(hidden_states, q_fp8, k, weights)
        else:
            raise NotImplementedError(
                "SparseAttnIndexer native forward is only implemented for "
                "CUDA, ROCm and XPU platforms."
            )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        return torch.ops.vllm.sparse_attn_indexer(
            hidden_states,
            _encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache,
            q_fp8,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
        )

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        if rocm_aiter_ops.is_enabled():
            return torch.ops.vllm.rocm_aiter_sparse_attn_indexer(
                hidden_states,
                _encode_layer_name(self.k_cache.prefix),
                self.k_cache.kv_cache,
                q_fp8,
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
        else:
            raise RuntimeError(
                "Sparse attention indexer ROCm custom op requires ROCm "
                "Aiter ops to be enabled."
            )
