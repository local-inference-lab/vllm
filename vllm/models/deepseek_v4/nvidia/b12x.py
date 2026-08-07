# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""b12x compressed sparse-MLA impl for DeepSeek-V4 (consumer Blackwell SM120).

The DSV4 sparse-MLA path uses global top-k slot ids from
``compute_global_topk_indices_and_lens``, a SWA + indexed dual cache, paged
per-chunk prefill, and ``attn_sink``. The leaf call goes through b12x's
``compressed_mla_decode_forward`` binding API (``plan_compressed_mla_scratch``
-> vLLM workspace-manager scratch -> ``plan.bind`` in ordinary Python, then one
``compressed_mla_decode_forward`` leaf call). No persistent workspace object is
held.

DSV4 compressed-MLA contract (== upstream/DeepGEMM): q_head_dim = 448 NoPE +
64 RoPE = 512, V = 512; the ``fp8_ds_mla`` 584 B/token page (448 NoPE fp8 +
128 RoPE bf16 + 8-byte UE8M0 footer) is read directly.
"""

import os
from typing import TYPE_CHECKING, ClassVar, Literal, cast

import torch

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.common.ops import (
    compute_dcp_global_topk_indices_and_lens,
    compute_global_topk_indices_and_lens,
)
from vllm.models.deepseek_v4.nvidia.flashmla import (
    DeepseekV4FlashMLAAttention,
    DeepseekV4FlashMLABackend,
)
from vllm.models.deepseek_v4.sparse_mla import DeepseekV4FlashMLAMetadata
from vllm.v1.attention.backends.mla.compressor_utils import (
    get_c128a_topk_width,
    get_compressed_mla_max_q_chunks,
    get_compressed_mla_split_cap,
    get_dspark_swa_index_width,
)
from vllm.v1.attention.ops.common import cp_lse_ag_out_rs
from vllm.v1.attention.ops.dcp_alltoall import (
    dcp_a2a_lse_reduce,
    dcp_b12x_all_gather_heads,
)
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

# DSV4 compressed-MLA dims (q_head_dim = 448 NoPE + 64 RoPE = 512; V = 512).
_DSV4_HEAD_DIM = 512
_DSV4_V_HEAD_DIM = 512
_DSV4_CACHE_BYTES_PER_TOKEN = 584
_VALIDATE_DCP_INDICES_ENV = "VLLM_DSV4_DCP_VALIDATE_INDICES"


def _get_dspark_decode_row_capacity(vllm_config: VllmConfig) -> int | None:
    """Return the largest target-verifier row count the scheduler can emit."""
    speculative_config = vllm_config.speculative_config
    if speculative_config is None or not speculative_config.use_dspark():
        return None
    num_speculative_tokens = int(speculative_config.num_speculative_tokens or 0)
    if num_speculative_tokens <= 0:
        return None
    scheduler_config = vllm_config.scheduler_config
    return min(
        int(scheduler_config.max_num_batched_tokens),
        int(scheduler_config.max_num_seqs) * (1 + num_speculative_tokens),
    )


def _validate_compressed_mla_decode_row_capacity(
    *,
    rows: int,
    mode: Literal["decode", "extend"],
    decode_row_capacity: int | None,
) -> None:
    if mode != "decode" or decode_row_capacity is None:
        return
    if int(rows) > int(decode_row_capacity):
        raise ValueError(
            f"compressed MLA decode rows {rows} exceed the declared "
            f"capacity {decode_row_capacity}"
        )


def _cdiv(x: int, y: int) -> int:
    return (int(x) + int(y) - 1) // int(y)


def _dsv4_b12x_page_nbytes(page_size: int) -> int:
    """Return the logical DSV4 payload bytes in one cache page.

    vLLM may place padding between physical pages, but the padding is not part
    of the compressed-MLA payload.  Keeping it out of the exported view also
    supports standalone contiguous allocations, whose physical page stride is
    exactly ``page_size * 584`` bytes.

    Args:
        page_size: Number of tokens stored in one cache page.

    Returns:
        The logical compressed-MLA payload size in bytes.
    """
    return int(page_size) * _DSV4_CACHE_BYTES_PER_TOKEN


def _b12x_cache_page_view(
    cache: torch.Tensor,
    page_size: int,
    name: str,
) -> torch.Tensor:
    """Return a uint8 ``[pages, payload_bytes]`` view for SparkInfer kernels.

    Preserve the physical page stride so both padded packed allocations and
    contiguous per-layer allocations follow the same runtime contract.

    Args:
        cache: Source paged cache tensor.
        page_size: Number of tokens stored in one cache page.
        name: Cache name used in validation errors.

    Returns:
        A logical byte view that preserves the source physical page stride.

    Raises:
        ValueError: If ``page_size`` is not positive.
        RuntimeError: If the cache is not paged, a page cannot contain the
            logical payload, pages overlap, or the payload within a page is
            not contiguous.
    """
    page_nbytes = _dsv4_b12x_page_nbytes(page_size)
    if page_nbytes <= 0:
        raise ValueError(f"{name} page_size must be positive, got {page_size}")

    byte_cache = cache if cache.dtype == torch.uint8 else cache.view(torch.uint8)
    if byte_cache.ndim == 2:
        if int(byte_cache.shape[1]) < page_nbytes:
            raise RuntimeError(
                f"{name} page width {int(byte_cache.shape[1])} is smaller than "
                f"DSV4 payload width {page_nbytes}"
            )
        if int(byte_cache.stride(1)) != 1:
            raise RuntimeError(
                f"{name} page payload must be contiguous, got stride "
                f"{tuple(byte_cache.stride())}"
            )
        if int(byte_cache.stride(0)) < page_nbytes:
            raise RuntimeError(
                f"{name} page stride {int(byte_cache.stride(0))} is smaller than "
                f"DSV4 page payload {page_nbytes}"
            )
        return byte_cache[:, :page_nbytes]

    if byte_cache.ndim < 2:
        raise RuntimeError(
            f"{name} expected a paged cache tensor, got shape {tuple(cache.shape)}"
        )

    pages = int(byte_cache.shape[0])
    page_stride_nbytes = int(byte_cache.stride(0))
    if page_stride_nbytes < page_nbytes:
        raise RuntimeError(
            f"{name} page stride {page_stride_nbytes} is smaller than DSV4 page "
            f"payload {page_nbytes}"
        )

    expected_stride = 1
    for dim in range(byte_cache.ndim - 1, 0, -1):
        if int(byte_cache.stride(dim)) != expected_stride:
            raise RuntimeError(
                f"{name} page payload must be contiguous, got stride "
                f"{tuple(byte_cache.stride())}"
            )
        expected_stride *= int(byte_cache.shape[dim])

    # Packed DS4 KV cache views have a storage offset for this layer and a
    # larger per-block stride for the whole packed block. Expose only the
    # logical payload while preserving stride(0), so SparkInfer can use the
    # physical block stride without materializing/copying.
    page_view = torch.as_strided(
        byte_cache,
        size=(pages, page_nbytes),
        stride=(page_stride_nbytes, 1),
    )
    return page_view


def _b12x_cache_page_view_key(
    cache: torch.Tensor,
    page_size: int,
) -> tuple[int, int, torch.dtype, int, tuple[int, ...], tuple[int, ...]]:
    """Stable key for cached KV page views.

    Packed DS4 KV tensors are stable after vLLM cache initialization. B12X only
    needs a static view mapping for a layer's cache tensor; the actual contents
    mutate in-place under that view. Avoid rebuilding the same as_strided view
    for every layer forward.
    """
    return (
        int(cache.untyped_storage().data_ptr()),
        int(cache.storage_offset()),
        cache.dtype,
        int(page_size),
        tuple(int(dim) for dim in cache.shape),
        tuple(int(stride) for stride in cache.stride()),
    )


def _validate_index_matrix_for_b12x(
    *,
    name: str,
    indices: torch.Tensor,
    lens: torch.Tensor,
    rows: int,
    page_size: int,
    num_pages: int,
) -> None:
    if indices.ndim == 3 and indices.shape[1] == 1:
        indices_2d = indices[:, 0]
    elif indices.ndim == 2:
        indices_2d = indices
    else:
        raise RuntimeError(
            f"{name} indices must be [rows, width] or [rows, 1, width], "
            f"got {tuple(indices.shape)}"
        )

    if int(indices_2d.shape[0]) != rows:
        raise RuntimeError(
            f"{name} row count {int(indices_2d.shape[0])} != q rows {rows}"
        )
    if tuple(lens.shape) != (rows,):
        raise RuntimeError(f"{name} lens shape {tuple(lens.shape)} != ({rows},)")

    width = int(indices_2d.shape[1])
    max_slots = int(page_size) * int(num_pages)
    lens_min = int(lens.min().item()) if rows else 0
    lens_max = int(lens.max().item()) if rows else 0
    if lens_min < 0 or lens_max > width:
        raise RuntimeError(
            f"{name} lens out of range: min={lens_min}, max={lens_max}, width={width}"
        )

    if rows == 0 or width == 0:
        return

    offsets = torch.arange(width, device=indices_2d.device)
    prefix_mask = offsets.unsqueeze(0) < lens.unsqueeze(1)
    if not bool(prefix_mask.any().item()):
        return

    prefix_values = indices_2d[prefix_mask]
    bad_mask = (prefix_values < 0) | (prefix_values >= max_slots)
    if bool(bad_mask.any().item()):
        bad_value = int(prefix_values[bad_mask][0].item())
        raise RuntimeError(
            f"{name} contains out-of-cache slot {bad_value}; "
            f"valid range is [0, {max_slots}) with page_size={page_size}, "
            f"num_pages={num_pages}"
        )


def _maybe_validate_compressed_mla_inputs(
    *,
    q: torch.Tensor,
    swa_k_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    swa_page_size: int,
    indexed_k_cache: torch.Tensor | None,
    indexed_indices: torch.Tensor | None,
    indexed_lens: torch.Tensor | None,
    indexed_page_size: int | None,
) -> None:
    if os.environ.get(_VALIDATE_DCP_INDICES_ENV) != "1":
        return
    if torch.cuda.is_current_stream_capturing():
        return

    rows = int(q.shape[0])
    _validate_index_matrix_for_b12x(
        name="swa",
        indices=swa_indices,
        lens=swa_lens,
        rows=rows,
        page_size=swa_page_size,
        num_pages=int(swa_k_cache.shape[0]),
    )
    if indexed_k_cache is None:
        return
    assert indexed_indices is not None
    assert indexed_lens is not None
    assert indexed_page_size is not None
    _validate_index_matrix_for_b12x(
        name="indexed",
        indices=indexed_indices,
        lens=indexed_lens,
        rows=rows,
        page_size=indexed_page_size,
        num_pages=int(indexed_k_cache.shape[0]),
    )


def _run_compressed_mla(
    *,
    q: torch.Tensor,
    output: torch.Tensor,
    attn_sink: torch.Tensor | None,
    scale: float,
    swa_k_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    swa_page_size: int,
    indexed_k_cache: torch.Tensor | None,
    indexed_indices: torch.Tensor | None,
    indexed_lens: torch.Tensor | None,
    indexed_page_size: int | None,
    mode: Literal["decode", "extend"] = "decode",
    decode_row_capacity: int | None = None,
    return_lse: bool = False,
    lse_scale: Literal["natural", "base2"] = "natural",
) -> torch.Tensor | None:
    """Plan, bind, and call b12x compressed MLA in plain eager Python.

    ``q`` is ``[tokens, padded_heads, 512]`` (heads pre-padded to
    {16,32,64,128} by the outer wrapper). Indices are global slot ids, so no
    indexed page table is needed.
    """
    from sparkinfer.attention.compressed_mla import (
        Caps as B12XCompressedMLAScratchCaps,
    )
    from sparkinfer.attention.compressed_mla import (
        plan as plan_compressed_mla_scratch,
    )
    from sparkinfer.attention.compressed_mla import (
        run as compressed_mla_decode_forward,
    )
    from sparkinfer.attention.compressed_mla import (
        split_chunks_for_contract as compressed_mla_split_chunks_for_contract,
    )

    rows, heads = int(q.shape[0]), int(q.shape[1])
    _validate_compressed_mla_decode_row_capacity(
        rows=rows,
        mode=mode,
        decode_row_capacity=decode_row_capacity,
    )
    q = q.contiguous()
    swa_indices = swa_indices.contiguous()
    swa_lens = swa_lens.contiguous()
    if indexed_indices is not None:
        indexed_indices = indexed_indices.contiguous()
    if indexed_lens is not None:
        indexed_lens = indexed_lens.contiguous()

    _maybe_validate_compressed_mla_inputs(
        q=q,
        swa_k_cache=swa_k_cache,
        swa_indices=swa_indices,
        swa_lens=swa_lens,
        swa_page_size=swa_page_size,
        indexed_k_cache=indexed_k_cache,
        indexed_indices=indexed_indices,
        indexed_lens=indexed_lens,
        indexed_page_size=indexed_page_size,
    )

    # b12x checks total_width = swa_width + indexed_width against scratch.topk,
    # so the scratch must be planned for the combined dual-cache width.
    width = int(swa_indices.shape[-1])
    if indexed_indices is not None:
        width += int(indexed_indices.shape[-1])
    decode_split_cap = get_compressed_mla_split_cap(width)
    # Keep the legacy 64-wide split cap for decode, but let the b12x contract
    # select the smaller batched-prefill split count when rows > decode max.
    num_splits_cap = compressed_mla_split_chunks_for_contract(
        rows=max(1, rows),
        width=width,
        max_chunks=decode_split_cap,
        decode_row_capacity=decode_row_capacity,
    )

    plan = plan_compressed_mla_scratch(
        B12XCompressedMLAScratchCaps(
            device=q.device,
            num_q_heads=heads,
            max_q_rows=max(1, rows),
            max_width=width,
            head_dim=_DSV4_HEAD_DIM,
            v_head_dim=_DSV4_V_HEAD_DIM,
            page_size=int(swa_page_size),
            max_chunks_per_row=num_splits_cap,
            decode_row_capacity=decode_row_capacity,
        )
    )
    scratch = current_workspace_manager().get_simultaneous(*plan.shapes_and_dtypes())

    binding = plan.bind(
        scratch=scratch,
        q=q,
        swa_indices=swa_indices,
        swa_lengths=swa_lens,
        indexed_indices=indexed_indices,
        indexed_lengths=indexed_lens,
    )
    binding.scratch.mode = mode

    # attn_sink is sized to the model's padded_heads (max(n_local,64)); b12x
    # wants it at the real local head count (== q heads). Under DCP, only one
    # rank contributes the gathered sink so the LSE reducer does not count it
    # once per DCP rank.
    sink = attn_sink[:heads].contiguous() if attn_sink is not None else None
    # b12x writes the final O directly into `output` (no workspace-to-caller
    # DtoD copy); the [tokens, heads, 512] bf16 contiguous contract is enforced
    # by b12x and raises on mismatch.
    result = compressed_mla_decode_forward(
        binding=binding,
        swa_k_cache=swa_k_cache,
        swa_page_size=int(swa_page_size),
        indexed_k_cache=indexed_k_cache,
        indexed_page_size=indexed_page_size,
        attn_sink=sink,
        sm_scale=scale,
        expected_num_q_heads=heads,
        return_lse=return_lse,
        lse_scale=lse_scale,
        out=output,
    )
    if return_lse:
        _, lse = cast(tuple[torch.Tensor, torch.Tensor], result)
        return lse
    return None


def _run_dcp_compressed_mla(
    *,
    q: torch.Tensor,
    output: torch.Tensor,
    attn_sink: torch.Tensor,
    scale: float,
    dcp_comm_backend: str,
    dcp_max_batch_size: int,
    swa_k_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    swa_page_size: int,
    indexed_k_cache: torch.Tensor | None,
    indexed_indices: torch.Tensor | None,
    indexed_lens: torch.Tensor | None,
    indexed_page_size: int | None,
    mode: Literal["decode", "extend"] = "decode",
    decode_row_capacity: int | None = None,
) -> None:
    from vllm.distributed.parallel_state import get_dcp_group

    dcp_group = get_dcp_group()
    if dcp_comm_backend == "a2a":
        q_all = dcp_b12x_all_gather_heads(
            q,
            dcp_group,
            max_batch_size=dcp_max_batch_size,
            output_head_dim=_DSV4_V_HEAD_DIM,
        )
    else:
        q_all = dcp_group.all_gather(q.contiguous(), dim=1)
    local_heads = int(q.shape[1])
    gathered_sink = dcp_group.all_gather(attn_sink[:local_heads].contiguous(), dim=0)
    sink = gathered_sink if dcp_group.rank_in_group == 0 else None

    partial_output = q_all.new_empty((q_all.shape[0], q_all.shape[1], _DSV4_V_HEAD_DIM))
    lse = _run_compressed_mla(
        q=q_all,
        output=partial_output,
        attn_sink=sink,
        scale=scale,
        swa_k_cache=swa_k_cache,
        swa_indices=swa_indices,
        swa_lens=swa_lens,
        swa_page_size=swa_page_size,
        indexed_k_cache=indexed_k_cache,
        indexed_indices=indexed_indices,
        indexed_lens=indexed_lens,
        indexed_page_size=indexed_page_size,
        mode=mode,
        decode_row_capacity=decode_row_capacity,
        return_lse=True,
        lse_scale="natural",
    )
    assert lse is not None

    if dcp_comm_backend == "a2a":
        reduced = dcp_a2a_lse_reduce(
            partial_output,
            lse,
            dcp_group,
            is_lse_base_on_e=True,
            use_b12x=True,
            b12x_max_batch_size=dcp_max_batch_size,
            b12x_query_head_dim=int(q.shape[-1]),
        )
    else:
        reduced = cp_lse_ag_out_rs(
            partial_output,
            lse,
            dcp_group,
            is_lse_base_on_e=True,
        )
    output.copy_(reduced)


class DeepseekV4B12xMLASparseBackend(DeepseekV4FlashMLABackend):
    """b12x compressed sparse-MLA backend for DeepSeek-V4 (SM120 / SM121).

    Geometry is identical to the FlashMLA parent (``fp8_ds_mla`` 584 B page,
    head 512, block 64) -- it inherits ``get_kv_cache_shape`` /
    ``get_supported_head_sizes``; only the impl class differs.
    """

    @staticmethod
    def get_name() -> str:
        return "V4_B12X_SPARSE"


class DeepseekV4B12xMLAAttention(DeepseekV4FlashMLAAttention):
    """b12x compressed sparse-MLA attention layer for DeepSeek-V4 (SM120)."""

    backend_cls: ClassVar[type[DeepseekV4FlashMLABackend]] = (
        DeepseekV4B12xMLASparseBackend
    )
    enqueue_default_before_indexer: ClassVar[bool] = True
    enable_post_gemm_aux_streams: ClassVar[bool] = True

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        if num_heads <= 16:
            return 16
        if num_heads <= 32:
            return 32
        if num_heads <= 64:
            return 64
        if num_heads <= 128:
            return 128
        raise ValueError(
            f"DeepseekV4 b12x sparse MLA does not support {num_heads} heads "
            "(kernel requires h_q in {16, 32, 64, 128})."
        )

    def _get_b12x_cache_page_view(
        self,
        cache: torch.Tensor,
        page_size: int,
        name: str,
    ) -> torch.Tensor:
        views = getattr(self, "_b12x_cache_page_views", None)
        if views is None:
            views = {}
            self._b12x_cache_page_views = views
        key = _b12x_cache_page_view_key(cache, page_size)
        view = views.get(key)
        if view is None:
            view = _b12x_cache_page_view(cache, page_size, name)
            views[key] = view
        return view

    def _reserve_dummy_compressed_mla_scratch(self, q: torch.Tensor) -> None:
        from sparkinfer.attention.compressed_mla import (
            Caps as B12XCompressedMLAScratchCaps,
        )
        from sparkinfer.attention.compressed_mla import (
            plan as plan_compressed_mla_scratch,
        )
        from sparkinfer.attention.compressed_mla import (
            split_chunks_for_contract as compressed_mla_split_chunks_for_contract,
        )

        indexed_width = 0
        if self.compress_ratio == 4:
            if self.topk_indices_buffer is not None:
                indexed_width = int(self.topk_indices_buffer.shape[-1])
            elif self.indexer is not None:
                indexed_width = int(self.indexer.topk_tokens)
        elif self.compress_ratio > 1:
            indexed_width = get_c128a_topk_width(
                self.max_model_len,
                self.compress_ratio,
            )

        swa_width = int(self.window_size)
        speculative_config = self.vllm_config.speculative_config
        decode_row_capacity = _get_dspark_decode_row_capacity(self.vllm_config)
        if speculative_config is not None and speculative_config.use_dspark():
            swa_width = max(
                swa_width,
                get_dspark_swa_index_width(
                    swa_width,
                    speculative_config.num_speculative_tokens or 0,
                ),
            )

        width = max(swa_width + indexed_width, 1)
        rows = max(int(self.max_num_batched_tokens), 1)
        decode_split_cap = get_compressed_mla_split_cap(width)
        num_splits_cap = compressed_mla_split_chunks_for_contract(
            rows=rows,
            width=width,
            max_chunks=decode_split_cap,
            decode_row_capacity=decode_row_capacity,
        )
        max_q_chunks = get_compressed_mla_max_q_chunks(
            rows,
            width,
            decode_split_cap,
            compressed_mla_split_chunks_for_contract,
            decode_row_capacity=decode_row_capacity,
        )
        dcp_world_size = max(
            int(self.vllm_config.parallel_config.decode_context_parallel_size),
            1,
        )
        # max_q_rows covers final row-sized buffers; max_q_chunks covers split
        # intermediates whose peak can occur below max_q_rows.
        plan = plan_compressed_mla_scratch(
            B12XCompressedMLAScratchCaps(
                device=q.device,
                num_q_heads=int(q.shape[1]) * dcp_world_size,
                max_q_rows=rows,
                max_width=width,
                head_dim=_DSV4_HEAD_DIM,
                v_head_dim=_DSV4_V_HEAD_DIM,
                page_size=int(self.swa_cache_layer.block_size),
                max_chunks_per_row=num_splits_cap,
                max_q_chunks=max_q_chunks,
                decode_row_capacity=decode_row_capacity,
            )
        )
        current_workspace_manager().get_simultaneous(*plan.shapes_and_dtypes())

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        del kv, positions
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
        assert output.dtype == q.dtype, (
            f"output buffer dtype {output.dtype} must match q dtype {q.dtype}"
        )

        prefix = self.prefix
        swa_cache_prefix = self.swa_cache_layer.prefix
        compress_ratio = self.compress_ratio
        compressed_kv_cache = self.kv_cache
        swa_kv_cache = self.swa_cache_layer.kv_cache
        topk_indices_buffer = self.topk_indices_buffer
        attn_sink = self.attn_sink
        scale = self.scale
        vllm_config = self.vllm_config

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            # Warmup dummy run: no metadata, so reserve the largest compressed
            # MLA scratch this layer can request before vLLM locks workspace.
            # Its config-derived geometry is identical on later dummy calls.
            output.zero_()
            self._reserve_dummy_compressed_mla_scratch(q)
            return

        assert isinstance(attn_metadata, dict)
        flashmla_metadata = cast(
            DeepseekV4FlashMLAMetadata | None, attn_metadata.get(prefix)
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(swa_cache_prefix),
        )
        assert swa_metadata is not None

        swa_only = compress_ratio <= 1
        self_kv_cache = compressed_kv_cache if not swa_only else None

        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens
        num_prefill_tokens = swa_metadata.num_prefill_tokens

        if num_prefills > 0:
            prefill_end = num_decode_tokens + num_prefill_tokens
            self._forward_b12x_prefill(
                q=q[num_decode_tokens:prefill_end],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:prefill_end],
                attn_metadata=flashmla_metadata,
                swa_metadata=swa_metadata,
                compress_ratio=compress_ratio,
                topk_indices_buffer=topk_indices_buffer,
                attn_sink=attn_sink,
                scale=scale,
                vllm_config=vllm_config,
            )
        if num_decodes > 0:
            self._forward_b12x_decode(
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_kv_cache=swa_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata,
                swa_only=swa_only,
                compress_ratio=compress_ratio,
                topk_indices_buffer=topk_indices_buffer,
                attn_sink=attn_sink,
                scale=scale,
                output=output[:num_decode_tokens],
                vllm_config=vllm_config,
            )

    def _forward_b12x_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,  # only used when compress_ratio > 1
        swa_kv_cache: torch.Tensor,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        compress_ratio: int,
        topk_indices_buffer: torch.Tensor | None,
        attn_sink: torch.Tensor,
        scale: float,
        output: torch.Tensor,
        vllm_config,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        decode_row_capacity = _get_dspark_decode_row_capacity(vllm_config)
        dcp_rank = 0
        if dcp_world_size > 1:
            from vllm.distributed.parallel_state import get_dcp_group

            dcp_rank = get_dcp_group().rank_in_group

        # Indexed (compressed) region global top-k.
        topk_indices = None
        topk_lens = None
        indexed_k_cache = None
        indexed_page_size = None
        if not swa_only:
            assert attn_metadata is not None
            assert kv_cache is not None
            assert swa_metadata.is_valid_token is not None
            block_size = attn_metadata.block_size // compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if compress_ratio == 4:
                assert topk_indices_buffer is not None
                if dcp_world_size > 1:
                    topk_indices, topk_lens = compute_dcp_global_topk_indices_and_lens(
                        topk_indices_buffer[:num_decode_tokens],
                        swa_metadata.token_to_req_indices,
                        attn_metadata.block_table[:num_decodes],
                        block_size,
                        is_valid,
                        dcp_world_size,
                        dcp_rank,
                        vllm_config.parallel_config.cp_kv_cache_interleave_size,
                    )
                else:
                    topk_indices, topk_lens = compute_global_topk_indices_and_lens(
                        topk_indices_buffer[:num_decode_tokens],
                        swa_metadata.token_to_req_indices,
                        attn_metadata.block_table[:num_decodes],
                        block_size,
                        is_valid,
                    )
            else:
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens
            indexed_page_size = block_size
            indexed_k_cache = self._get_b12x_cache_page_view(
                kv_cache,
                indexed_page_size,
                "indexed_k_cache",
            )

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens
        assert swa_indices is not None
        assert swa_lens is not None
        swa_k_cache = self._get_b12x_cache_page_view(
            swa_kv_cache,
            swa_metadata.block_size,
            "swa_k_cache",
        )

        if dcp_world_size > 1:
            _run_dcp_compressed_mla(
                q=q,
                output=output,
                attn_sink=attn_sink,
                scale=scale,
                dcp_comm_backend=vllm_config.parallel_config.dcp_comm_backend,
                dcp_max_batch_size=(
                    vllm_config.scheduler_config.max_num_batched_tokens
                ),
                swa_k_cache=swa_k_cache,
                swa_indices=swa_indices,
                swa_lens=swa_lens,
                swa_page_size=swa_metadata.block_size,
                indexed_k_cache=indexed_k_cache,
                indexed_indices=topk_indices,
                indexed_lens=topk_lens,
                indexed_page_size=indexed_page_size,
                mode="decode",
                decode_row_capacity=decode_row_capacity,
            )
        else:
            _run_compressed_mla(
                q=q,
                output=output,
                attn_sink=attn_sink,
                scale=scale,
                swa_k_cache=swa_k_cache,
                swa_indices=swa_indices,
                swa_lens=swa_lens,
                swa_page_size=swa_metadata.block_size,
                indexed_k_cache=indexed_k_cache,
                indexed_indices=topk_indices,
                indexed_lens=topk_lens,
                indexed_page_size=indexed_page_size,
                mode="decode",
                decode_row_capacity=decode_row_capacity,
            )

    def _forward_b12x_prefill(
        self,
        q: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        compress_ratio: int,
        topk_indices_buffer: torch.Tensor | None,
        attn_sink: torch.Tensor,
        scale: float,
        vllm_config,
    ) -> None:
        swa_only = compress_ratio <= 1
        num_prefills = swa_metadata.num_prefills
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        dcp_rank = 0
        if dcp_world_size > 1:
            from vllm.distributed.parallel_state import get_dcp_group

            dcp_rank = get_dcp_group().rank_in_group

        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        assert query_start_loc_cpu is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        # Indexed (compressed) region global top-k for the prefill rows.
        extra_topk_indices = None
        extra_topk_lens = None
        indexed_k_cache = None
        indexed_page_size = None
        if not swa_only:
            assert attn_metadata is not None
            assert compressed_k_cache is not None
            if compress_ratio == 4:
                assert topk_indices_buffer is not None
                local_topk_indices = topk_indices_buffer[
                    num_decode_tokens : num_decode_tokens + num_prefill_tokens
                ]
            else:
                local_topk_indices = attn_metadata.c128a_prefill_topk_indices
            assert swa_metadata.token_to_req_indices is not None
            assert swa_metadata.is_valid_token is not None
            prefill_slice = slice(
                num_decode_tokens, num_decode_tokens + num_prefill_tokens
            )
            block_size = attn_metadata.block_size // compress_ratio
            if dcp_world_size > 1:
                extra_topk_indices, extra_topk_lens = (
                    compute_dcp_global_topk_indices_and_lens(
                        local_topk_indices,
                        swa_metadata.token_to_req_indices[prefill_slice],
                        attn_metadata.block_table,
                        block_size,
                        swa_metadata.is_valid_token[prefill_slice],
                        dcp_world_size,
                        dcp_rank,
                        vllm_config.parallel_config.cp_kv_cache_interleave_size,
                    )
                )
            else:
                extra_topk_indices, extra_topk_lens = (
                    compute_global_topk_indices_and_lens(
                        local_topk_indices,
                        swa_metadata.token_to_req_indices[prefill_slice],
                        attn_metadata.block_table,
                        block_size,
                        swa_metadata.is_valid_token[prefill_slice],
                    )
                )
            indexed_page_size = block_size
            indexed_k_cache = self._get_b12x_cache_page_view(
                compressed_k_cache,
                indexed_page_size,
                "indexed_k_cache",
            )

        assert swa_metadata.prefill_swa_indices is not None
        assert swa_metadata.prefill_swa_lens is not None
        swa_k_cache = self._get_b12x_cache_page_view(
            swa_k_cache,
            swa_metadata.block_size,
            "swa_k_cache",
        )

        num_chunks = (
            num_prefills + self.PREFILL_CHUNK_SIZE - 1
        ) // self.PREFILL_CHUNK_SIZE
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * self.PREFILL_CHUNK_SIZE
            chunk_end = min(chunk_start + self.PREFILL_CHUNK_SIZE, num_prefills)
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            idx_chunk = (
                extra_topk_indices[query_start:query_end]
                if extra_topk_indices is not None
                else None
            )
            idx_lens_chunk = (
                extra_topk_lens[query_start:query_end]
                if extra_topk_lens is not None
                else None
            )
            if dcp_world_size > 1:
                _run_dcp_compressed_mla(
                    q=q[query_start:query_end],
                    output=output[query_start:query_end],
                    attn_sink=attn_sink,
                    scale=scale,
                    dcp_comm_backend=vllm_config.parallel_config.dcp_comm_backend,
                    dcp_max_batch_size=(
                        vllm_config.scheduler_config.max_num_batched_tokens
                    ),
                    swa_k_cache=swa_k_cache,
                    swa_indices=swa_metadata.prefill_swa_indices[query_start:query_end],
                    swa_lens=swa_metadata.prefill_swa_lens[query_start:query_end],
                    swa_page_size=swa_metadata.block_size,
                    indexed_k_cache=indexed_k_cache,
                    indexed_indices=idx_chunk,
                    indexed_lens=idx_lens_chunk,
                    indexed_page_size=indexed_page_size,
                    mode="extend",
                )
            else:
                _run_compressed_mla(
                    q=q[query_start:query_end],
                    output=output[query_start:query_end],
                    attn_sink=attn_sink,
                    scale=scale,
                    swa_k_cache=swa_k_cache,
                    swa_indices=swa_metadata.prefill_swa_indices[query_start:query_end],
                    swa_lens=swa_metadata.prefill_swa_lens[query_start:query_end],
                    swa_page_size=swa_metadata.block_size,
                    indexed_k_cache=indexed_k_cache,
                    indexed_indices=idx_chunk,
                    indexed_lens=idx_lens_chunk,
                    indexed_page_size=indexed_page_size,
                    mode="extend",
                )
