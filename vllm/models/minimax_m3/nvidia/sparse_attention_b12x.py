# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12X MSA-backed MiniMax M3 block-sparse attention.

This path keeps vLLM ownership of scratch memory: decode uses prepared graph
replay plans in both eager prewarm and CUDA graph capture, extend/prefill stays
on the eager planner, and every launch binds runtime device metadata plus the
Triton indexer's q2k indices into caller scratch borrowed from vLLM's workspace
manager.
"""

from __future__ import annotations

import math
import os
from typing import Any, Literal

import torch

from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.minimax_m3.common.ops.sparse_attn import (
    SPARSE_BLOCK_SIZE,
    minimax_m3_sparse_attn,
    minimax_m3_sparse_attn_decode,
)
from vllm.models.minimax_m3.common.sparse_attention import (
    MiniMaxM3SparseImpl,
    MiniMaxM3SparseMetadata,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import canonicalize_singleton_dim_strides
from vllm.v1.attention.backend import AttentionLayer
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

_B12X_MINIMAX_MSA_TOPK = 16
_B12X_MINIMAX_MSA_PAGE_SIZE = 128
_B12X_FP8_KV_CACHE_DTYPES = ("fp8", "fp8_e4m3")
_B12X_MSA_COMPARE_TRITON = "VLLM_B12X_MINIMAX_M3_MSA_COMPARE_TRITON"
_B12X_MSA_COMPARE_AFTER_ENGINE_START = (
    "VLLM_B12X_MINIMAX_M3_MSA_COMPARE_AFTER_ENGINE_START"
)
_B12X_MSA_COMPARE_LOG_ALL = "VLLM_B12X_MINIMAX_M3_MSA_COMPARE_LOG_ALL"
_B12X_MSA_COMPARE_MAX_REPORTS = "VLLM_B12X_MINIMAX_M3_MSA_COMPARE_MAX_REPORTS"
_B12X_MSA_COMPARE_ATOL = "VLLM_B12X_MINIMAX_M3_MSA_COMPARE_ATOL"
_B12X_MSA_DUMP_DIR = "VLLM_B12X_MINIMAX_M3_MSA_DUMP_DIR"
_B12X_MSA_SYNC_AFTER = "VLLM_B12X_MINIMAX_M3_MSA_SYNC_AFTER"
_B12X_MSA_ZERO_OUTPUT_BEFORE = "VLLM_B12X_MINIMAX_M3_MSA_ZERO_OUTPUT_BEFORE"
_B12X_MSA_TRITON_COMPARE_REPORTS = 0


def _debug_rank_label() -> str:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return f"rank{torch.distributed.get_rank()}"
    except RuntimeError:
        pass
    return f"pid{os.getpid()}"


def _tensor_addr_range(tensor: torch.Tensor) -> tuple[int, int]:
    start = int(tensor.data_ptr())
    return start, start + int(tensor.numel() * tensor.element_size())


def _ranges_overlap(lhs: tuple[int, int], rhs: tuple[int, int]) -> bool:
    return lhs[0] < rhs[1] and rhs[0] < lhs[1]


def _is_b12x_fp8_kv_cache(kv_cache_dtype: str) -> bool:
    return kv_cache_dtype in _B12X_FP8_KV_CACHE_DTYPES


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %d", name, value, default)
        return default
    if parsed <= 0:
        logger.warning("Ignoring non-positive %s=%r; using %d", name, value, default)
        return default
    return parsed


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %.6g", name, value, default)
        return default


def _claim_triton_compare_report(max_reports: int) -> bool:
    global _B12X_MSA_TRITON_COMPARE_REPORTS
    if max_reports <= _B12X_MSA_TRITON_COMPARE_REPORTS:
        return False
    _B12X_MSA_TRITON_COMPARE_REPORTS += 1
    return True


def _capture_alloc_forbidden() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except RuntimeError:
        return False


def _ensure_i32_contiguous(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.dtype != torch.int32:
        if _capture_alloc_forbidden():
            raise RuntimeError(
                "B12X MiniMax M3 MSA would convert "
                f"{name} to int32 during CUDA graph capture."
            )
        tensor = tensor.to(torch.int32)
    if not tensor.is_contiguous():
        if _capture_alloc_forbidden():
            raise RuntimeError(
                "B12X MiniMax M3 MSA would make "
                f"{name} contiguous during CUDA graph capture."
            )
        tensor = tensor.contiguous()
    return tensor


def _cu_seqlens_total_q(cu_seqlens_q: torch.Tensor) -> int:
    if _capture_alloc_forbidden():
        raise RuntimeError(
            "B12X MiniMax M3 MSA cannot read cu_seqlens_q on host during "
            "CUDA graph capture."
        )
    return int(cu_seqlens_q[-1].item())


def _kv_dtype_from_cache_config(kv_cache_dtype: str) -> torch.dtype:
    if _is_b12x_fp8_kv_cache(kv_cache_dtype):
        fp8_dtype = current_platform.fp8_dtype()
        if fp8_dtype != torch.float8_e4m3fn:
            raise NotImplementedError(
                "B12X MiniMax M3 MSA supports only fp8_e4m3 KV caches, got "
                f"platform fp8 dtype {fp8_dtype}."
            )
        return fp8_dtype
    if kv_cache_dtype == "bfloat16":
        return torch.bfloat16
    if kv_cache_dtype == "auto":
        return get_current_vllm_config().model_config.dtype
    raise NotImplementedError(
        "B12X MiniMax M3 MSA supports auto, bfloat16, fp8, and fp8_e4m3 "
        f"KV caches, got {kv_cache_dtype!r}."
    )


class MiniMaxM3SparseB12XImpl(MiniMaxM3SparseImpl):
    """B12X paged MSA attend over MiniMax's Triton-selected sparse blocks."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        kv_cache_dtype: str = "auto",
        *,
        topk_blocks: int,
        sparse_block_size: int,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            kv_cache_dtype,
            topk_blocks=topk_blocks,
            sparse_block_size=sparse_block_size,
        )
        if self.head_size != 128:
            raise NotImplementedError(
                f"B12X MiniMax M3 MSA requires head_size=128, got {self.head_size}."
            )
        if self.block_size != _B12X_MINIMAX_MSA_PAGE_SIZE:
            raise NotImplementedError(
                "B12X MiniMax M3 MSA requires page/block size 128, "
                f"got {self.block_size}."
            )
        if self.block_size != SPARSE_BLOCK_SIZE:
            raise NotImplementedError(
                "B12X MiniMax M3 MSA expects the sparse block size to match "
                f"the KV page size, got {SPARSE_BLOCK_SIZE} vs {self.block_size}."
            )
        if self.topk_blocks != _B12X_MINIMAX_MSA_TOPK:
            raise NotImplementedError(
                "B12X MiniMax M3 MSA requires sparse_topk_blocks=16, "
                f"got {self.topk_blocks}."
            )
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                "B12X MiniMax M3 MSA requires q heads divisible by kv heads."
            )
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        if self.num_queries_per_kv != 16:
            raise NotImplementedError(
                "B12X MiniMax M3 MSA requires GQA group size 16, got "
                f"{self.num_queries_per_kv}."
            )
        expected_scale = self.head_size**-0.5
        if not math.isclose(float(scale), expected_scale, rel_tol=1e-5, abs_tol=1e-7):
            raise NotImplementedError(
                "B12X MiniMax M3 MSA requires canonical softmax scale "
                f"head_dim**-0.5={expected_scale}, got {scale}."
            )

        self.kv_torch_dtype = _kv_dtype_from_cache_config(kv_cache_dtype)
        if self.kv_torch_dtype not in (torch.bfloat16, torch.float8_e4m3fn):
            raise NotImplementedError(
                "B12X MiniMax M3 MSA page-128 plans support only bf16 or "
                f"fp8_e4m3 KV caches, got {self.kv_torch_dtype}."
            )

        self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self._unit_fp8_descale = torch.ones((), dtype=torch.float32, device=self.device)

        from vllm.v1.attention.backends.b12x_attn import (
            _disable_cutlass_memory_debug_snapshot_if_off,
        )

        _disable_cutlass_memory_debug_snapshot_if_off()

        from sparkinfer.attention.paged import (
            Caps as B12XPagedAttentionScratchCaps,
        )
        from sparkinfer.attention.paged import (
            plan as plan_paged_attention_scratch,
        )
        from sparkinfer.attention.paged import (
            run as paged_attention_forward,
        )

        self._paged_attention_forward = paged_attention_forward
        self._scratch_caps_type = B12XPagedAttentionScratchCaps
        self._plan_paged_attention_scratch = plan_paged_attention_scratch

        vllm_config = get_current_vllm_config()
        scheduler_config = vllm_config.scheduler_config
        model_config = vllm_config.model_config
        self.dtype = torch.get_default_dtype()
        if self.dtype not in (torch.float16, torch.bfloat16):
            self.dtype = model_config.dtype
        max_batched = int(scheduler_config.max_num_batched_tokens)
        max_num_seqs = int(scheduler_config.max_num_seqs)
        max_model_len = int(model_config.max_model_len)
        max_page_table_width = max(
            cdiv(max(max_model_len, 1), _B12X_MINIMAX_MSA_PAGE_SIZE),
            1,
        )
        extend_q_tiles = cdiv(max_batched, 8)
        extend_work_items = _env_int(
            "VLLM_B12X_MINIMAX_M3_MSA_EXTEND_MAX_WORK_ITEMS",
            extend_q_tiles + max_num_seqs,
        )
        self._extend_scratch_plan = self._plan_paged_attention_scratch(
            self._scratch_caps_type(
                device=self.device,
                mode="extend",
                dtype=self.dtype,
                kv_dtype=self.kv_torch_dtype,
                num_q_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.head_size,
                head_dim_vo=self.head_size,
                page_size=_B12X_MINIMAX_MSA_PAGE_SIZE,
                max_total_q=max_batched,
                max_batch=max_num_seqs,
                max_page_table_width=max_page_table_width,
                max_work_items=extend_work_items,
                max_partial_rows=0,
                num_cache_pages=1,
                use_cuda_graph=False,
                msa_block_sparse=True,
            )
        )
        self._decode_graph_scratch_plans: dict[tuple[Any, ...], Any] = {}
        self._triton_compare_reports = 0
        self._triton_compare_dumps = 0
        self._debug_reports = 0

        current_workspace_manager().get_simultaneous(
            ((int(self._extend_scratch_plan.layout.nbytes),), torch.uint8),
        )

        logger.info_once(
            "Using B12X MiniMax M3 MSA attention: heads=%d kv_heads=%d "
            "head_dim=%d kv_dtype=%s extend_work_items=%d",
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            self.kv_torch_dtype,
            extend_work_items,
        )

    def _kv_cache_views(
        self,
        kv_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_fp8_kv:
            kv_cache = kv_cache.view(self.kv_cache_fp8_dtype)
        key_cache, value_cache = kv_cache.unbind(1)
        key_cache = canonicalize_singleton_dim_strides(key_cache)
        value_cache = canonicalize_singleton_dim_strides(value_cache)
        if (
            key_cache.dtype != self.kv_torch_dtype
            or value_cache.dtype != self.kv_torch_dtype
        ):
            raise TypeError(
                "B12X MiniMax M3 MSA plan expects KV dtype "
                f"{self.kv_torch_dtype}, got {key_cache.dtype}/{value_cache.dtype}."
            )
        return key_cache, value_cache

    def _prepare_fp8_descales(
        self,
        num_reqs: int,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not _is_b12x_fp8_kv_cache(self.kv_cache_dtype):
            return None, None
        if num_reqs <= 0:
            raise ValueError(
                "B12X MiniMax M3 MSA fp8 descale request count must be positive."
            )
        if self._unit_fp8_descale.device != device:
            raise RuntimeError(
                "B12X MiniMax M3 MSA fp8 descales must be on the query device."
            )
        descale = self._unit_fp8_descale.expand(num_reqs)
        return descale, descale

    def _validate_q2k_indices(
        self,
        q2k_indices: torch.Tensor,
        total_q: int,
    ) -> None:
        if q2k_indices.ndim != 3:
            raise ValueError(
                "B12X MiniMax M3 MSA q2k_indices must have shape "
                "[kv_heads, total_q, 16]."
            )
        if q2k_indices.dtype != torch.int32:
            raise TypeError("B12X MiniMax M3 MSA q2k_indices must be int32.")
        if not q2k_indices.is_contiguous():
            raise ValueError("B12X MiniMax M3 MSA q2k_indices must be contiguous.")
        if (
            int(q2k_indices.shape[0]) != self.num_kv_heads
            or int(q2k_indices.shape[1]) < total_q
            or int(q2k_indices.shape[2]) != _B12X_MINIMAX_MSA_TOPK
        ):
            raise ValueError(
                "B12X MiniMax M3 MSA q2k_indices must have shape "
                f"({self.num_kv_heads}, >={total_q}, 16), got "
                f"{tuple(q2k_indices.shape)}."
            )

    def _make_scratch_caps(
        self,
        *,
        q: torch.Tensor,
        key_cache: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        mode: Literal["decode", "extend"],
        max_work_items: int,
        max_partial_rows: int,
        use_cuda_graph: bool,
    ) -> Any:
        return self._scratch_caps_type(
            device=q.device,
            mode=mode,
            dtype=q.dtype,
            kv_dtype=self.kv_torch_dtype,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim_qk=self.head_size,
            head_dim_vo=self.head_size,
            page_size=_B12X_MINIMAX_MSA_PAGE_SIZE,
            max_total_q=max(int(q.shape[0]), 1),
            max_batch=max(int(cache_seqlens.shape[0]), 1),
            max_page_table_width=max(int(page_table.shape[1]), 1),
            max_work_items=max(int(max_work_items), 1),
            max_partial_rows=max(int(max_partial_rows), 0),
            num_cache_pages=max(int(key_cache.shape[0]), 1),
            use_cuda_graph=use_cuda_graph,
            msa_block_sparse=True,
        )

    def _decode_graph_scratch_key(
        self,
        *,
        q: torch.Tensor,
        key_cache: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> tuple[Any, ...]:
        return (
            q.device.index,
            q.dtype,
            self.kv_torch_dtype,
            int(q.shape[0]),
            int(cache_seqlens.shape[0]),
            int(page_table.shape[1]),
            int(key_cache.shape[0]),
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
        )

    def _prepare_decode_graph_scratch_plan(
        self,
        *,
        q: torch.Tensor,
        key_cache: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> Any:
        key = self._decode_graph_scratch_key(
            q=q,
            key_cache=key_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
        )
        cached = self._decode_graph_scratch_plans.get(key)
        if cached is not None:
            return cached
        batch = int(cache_seqlens.shape[0])
        page_table_width = int(page_table.shape[1])
        max_decode_work_items = max(batch * 32, 1)
        graph_scratch_plan = self._plan_paged_attention_scratch(
            self._make_scratch_caps(
                q=q,
                key_cache=key_cache,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                mode="decode",
                max_work_items=max_decode_work_items,
                max_partial_rows=max_decode_work_items,
                use_cuda_graph=True,
            )
        )
        graph_scratch_plan.prepare_decode_graph_replay_state(
            batch=batch,
            max_page_table_width=page_table_width,
            total_q_capacity=max(int(q.shape[0]), 1),
            max_cache_page_count=page_table_width,
            fixed_split_size=-1,
            window_left=-1,
        )
        self._decode_graph_scratch_plans[key] = graph_scratch_plan
        return graph_scratch_plan

    def _select_scratch_plan(
        self,
        *,
        q: torch.Tensor,
        key_cache: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        mode: Literal["decode", "extend"],
    ) -> Any:
        if mode == "decode":
            return self._prepare_decode_graph_scratch_plan(
                q=q,
                key_cache=key_cache,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
            )
        return self._extend_scratch_plan

    def _run_b12x_slice(
        self,
        *,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        q2k_indices: torch.Tensor,
        mode: Literal["decode", "extend"],
        prefix_lens: torch.Tensor | None = None,
        max_query_len: int | None = None,
        layer_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        total_q_capacity = int(q.shape[0])
        if total_q_capacity <= 0:
            return output, None
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError(f"B12X MiniMax M3 MSA does not support q dtype {q.dtype}.")
        if output.dtype != q.dtype:
            raise TypeError(
                "B12X MiniMax M3 MSA expects output dtype to match q dtype, "
                f"got {output.dtype} vs {q.dtype}."
            )

        page_table = _ensure_i32_contiguous(page_table, "page_table")
        cache_seqlens = _ensure_i32_contiguous(cache_seqlens, "seq_lens")
        cu_seqlens_q = _ensure_i32_contiguous(cu_seqlens_q, "cu_seqlens_q")

        capturing = _capture_alloc_forbidden()
        if mode == "extend":
            if capturing:
                raise RuntimeError(
                    "B12X MiniMax M3 MSA can only use a pre-planned decode "
                    "graph scratch path during CUDA graph capture."
                )
            total_q = _cu_seqlens_total_q(cu_seqlens_q)
            if total_q < 0 or total_q > total_q_capacity:
                raise ValueError(
                    "B12X MiniMax M3 MSA cu_seqlens_q total query length "
                    f"{total_q} is outside q capacity {total_q_capacity}."
                )
            if total_q < total_q_capacity:
                output[total_q:].zero_()
                if total_q == 0:
                    return output, None
                q = q[:total_q]
                output = output[:total_q]
        else:
            total_q = total_q_capacity

        self._validate_q2k_indices(q2k_indices, total_q)

        scratch_plan = self._select_scratch_plan(
            q=q,
            key_cache=key_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            mode=mode,
        )
        (scratch_storage,) = current_workspace_manager().get_simultaneous(
            ((int(scratch_plan.layout.nbytes),), torch.uint8),
        )
        num_reqs = int(cache_seqlens.shape[0])
        compare_after_engine_start = (
            os.getenv(_B12X_MSA_COMPARE_AFTER_ENGINE_START, "0") == "1"
        )
        engine_started = os.getenv("B12X_VLLM_ENGINE_STARTED", "0") == "1"
        compare_max_reports = _env_int(_B12X_MSA_COMPARE_MAX_REPORTS, 8)
        compare_requested = (
            os.getenv(_B12X_MSA_COMPARE_TRITON, "0") == "1"
            and not capturing
            and (not compare_after_engine_start or engine_started)
        )
        compare_triton = compare_requested and _claim_triton_compare_report(
            compare_max_reports
        )
        triton_f32_cpu: torch.Tensor | None = None
        q_finite_before: bool | None = None
        q2k_range_before: tuple[int, int] | None = None
        q_f32_cpu_before: torch.Tensor | None = None
        key_pages_f32_cpu_before: torch.Tensor | None = None
        value_pages_f32_cpu_before: torch.Tensor | None = None
        active_page_ids_cpu: torch.Tensor | None = None
        if compare_triton:
            q_finite_before = bool(torch.isfinite(q.float()).all().item())
            q2k_range_before = (
                int(q2k_indices.min().item()),
                int(q2k_indices.max().item()),
            )
            active_page_ids_cpu = self._active_page_ids_cpu(
                page_table,
                cache_seqlens,
            )
            q_f32_cpu_before = q.float().cpu()
            (
                key_pages_f32_cpu_before,
                value_pages_f32_cpu_before,
            ) = self._active_kv_pages_f32_cpu(
                key_cache,
                value_cache,
                active_page_ids_cpu,
            )
            triton_output = self._run_triton_reference_slice(
                q=q,
                kv_cache=kv_cache,
                q2k_indices=q2k_indices,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                mode=mode,
                prefix_lens=prefix_lens,
                max_query_len=max_query_len,
            )
            torch.cuda.synchronize()
            triton_f32_cpu = triton_output.float().cpu()
        if (
            os.getenv("VLLM_DEBUG_B12X_MINIMAX_M3_MSA", "0") == "1"
            and self._debug_reports < 12
        ):
            plan = getattr(scratch_plan, "_plan", None)
            logger.warning(
                "B12X MiniMax M3 MSA bind layer=%s mode=%s capturing=%s "
                "q=%s out=%s page_table=%s cache_seqlens=%s cu=%s "
                "q2k=%s scratch=%d plan_total_q=%s plan_split=%s "
                "plan_kv_chunk=%s",
                layer_name,
                mode,
                capturing,
                tuple(q.shape),
                tuple(output.shape),
                tuple(page_table.shape),
                tuple(cache_seqlens.shape),
                tuple(cu_seqlens_q.shape),
                tuple(q2k_indices.shape),
                int(scratch_plan.layout.nbytes),
                getattr(plan, "total_q", None),
                getattr(plan, "split_kv", None),
                getattr(plan, "kv_chunk_size", None),
            )
            self._debug_reports += 1
        k_descale, v_descale = self._prepare_fp8_descales(num_reqs, q.device)
        if not capturing:
            scratch_plan._q2k_indices_data_ptr = None
        binding = scratch_plan.bind(
            scratch=scratch_storage,
            q=q,
            k_cache=key_cache,
            v_cache=value_cache,
            output=output,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            active_total_q=total_q,
            q2k_indices=q2k_indices,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        bound_scratch = getattr(binding, "scratch", None)
        bound_plan = getattr(bound_scratch, "_plan", None)
        plan_debug = self._plan_debug_dict(
            plan=bound_plan,
            scratch=bound_scratch,
            scratch_storage=scratch_storage,
        )
        if (
            os.getenv("VLLM_DEBUG_B12X_MINIMAX_M3_MSA", "0") == "1"
            and self._debug_reports < 12
        ):
            logger.warning(
                "B12X MiniMax M3 MSA bound layer=%s mode=%s rank=%s plan=%s scratch=%s",
                layer_name,
                mode,
                _debug_rank_label(),
                plan_debug["plan"],
                plan_debug["scratch"],
            )
            self._debug_reports += 1
        if os.getenv(_B12X_MSA_ZERO_OUTPUT_BEFORE, "0") == "1":
            output.zero_()
        returned_output, returned_lse = self._paged_attention_forward(binding=binding)
        if os.getenv(_B12X_MSA_SYNC_AFTER, "0") == "1":
            torch.cuda.synchronize()
        if not capturing:
            scratch_plan._q2k_indices_data_ptr = None
        if compare_triton:
            assert triton_f32_cpu is not None
            self._compare_triton_slice(
                q=q,
                key_cache=key_cache,
                value_cache=value_cache,
                q2k_indices=q2k_indices,
                b12x_output=output,
                triton_f32_cpu=triton_f32_cpu,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                k_descale=k_descale,
                v_descale=v_descale,
                scratch_storage=scratch_storage,
                returned_output=returned_output,
                q_f32_cpu_before=q_f32_cpu_before,
                key_pages_f32_cpu_before=key_pages_f32_cpu_before,
                value_pages_f32_cpu_before=value_pages_f32_cpu_before,
                active_page_ids_cpu=active_page_ids_cpu,
                mode=mode,
                num_reqs=num_reqs,
                q_finite_before=q_finite_before,
                q2k_range_before=q2k_range_before,
                plan_debug=plan_debug,
                layer_name=layer_name,
            )
            self._triton_compare_reports += 1
        return returned_output, returned_lse

    def _plan_debug_dict(
        self,
        *,
        plan: Any,
        scratch: Any,
        scratch_storage: torch.Tensor,
    ) -> dict[str, Any]:
        scratch_ptr = int(scratch_storage.data_ptr())
        scratch_bytes = int(scratch_storage.numel() * scratch_storage.element_size())
        union_blocks = getattr(scratch, "msa_union_blocks", None)
        union_masks = getattr(scratch, "msa_union_masks", None)
        union_counts = getattr(scratch, "msa_union_counts", None)
        lse = getattr(scratch, "lse", None)
        request_indices = getattr(plan, "request_indices", ())
        qo_tile_indices = getattr(plan, "qo_tile_indices", ())
        kv_tile_indices = getattr(plan, "kv_tile_indices", ())
        plan_info = {
            "total_q": getattr(plan, "total_q", None),
            "mode": getattr(plan, "mode", None),
            "cta_tile_q": getattr(plan, "cta_tile_q", None),
            "kv_chunk_size": getattr(plan, "kv_chunk_size", None),
            "split_kv": getattr(plan, "split_kv", None),
            "new_batch_size": getattr(plan, "new_batch_size", None),
            "padded_batch_size": getattr(plan, "padded_batch_size", None),
            "num_qo_tiles": getattr(plan, "num_qo_tiles", None),
            "total_num_partial_rows": getattr(plan, "total_num_partial_rows", None),
            "page_size": getattr(plan, "page_size", None),
            "gqa_group_size": getattr(plan, "gqa_group_size", None),
            "msa_block_sparse": getattr(plan, "msa_block_sparse", None),
            "msa_union_tile": getattr(plan, "msa_union_tile", None),
            "request_first": tuple(request_indices[:8]),
            "request_last": tuple(request_indices[-8:]),
            "qo_first": tuple(qo_tile_indices[:8]),
            "qo_last": tuple(qo_tile_indices[-8:]),
            "kv_first": tuple(kv_tile_indices[:8]),
            "kv_last": tuple(kv_tile_indices[-8:]),
        }
        scratch_info = {
            "ptr": scratch_ptr,
            "bytes": scratch_bytes,
            "align256": scratch_ptr % 256,
            "union_blocks": None if union_blocks is None else tuple(union_blocks.shape),
            "union_masks": None if union_masks is None else tuple(union_masks.shape),
            "union_counts": None if union_counts is None else tuple(union_counts.shape),
            "lse": None if lse is None else tuple(lse.shape),
        }
        return {"plan": plan_info, "scratch": scratch_info}

    def _run_triton_reference_slice(
        self,
        *,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        q2k_indices: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        mode: Literal["decode", "extend"],
        prefix_lens: torch.Tensor | None,
        max_query_len: int | None,
    ) -> torch.Tensor:
        triton_output = torch.empty_like(q)
        triton_kv_cache = (
            kv_cache.view(self.kv_cache_fp8_dtype) if self.use_fp8_kv else kv_cache
        )
        if mode == "decode":
            num_reqs = int(cache_seqlens.shape[0])
            decode_query_len = max(int(q.shape[0]) // max(num_reqs, 1), 1)
            minimax_m3_sparse_attn_decode(
                q,
                triton_kv_cache,
                q2k_indices,
                page_table,
                cache_seqlens,
                self.num_kv_heads,
                self.scale,
                triton_output,
                decode_query_len,
            )
        else:
            if prefix_lens is None:
                query_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
                prefix_lens = cache_seqlens - query_lens
            if max_query_len is None:
                max_query_len = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max())
            minimax_m3_sparse_attn(
                q,
                triton_kv_cache,
                q2k_indices,
                page_table,
                cu_seqlens_q,
                cache_seqlens,
                prefix_lens,
                max_query_len,
                self.num_kv_heads,
                self.scale,
                triton_output,
            )
        return triton_output

    def _compare_triton_slice(
        self,
        *,
        q: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        q2k_indices: torch.Tensor,
        b12x_output: torch.Tensor,
        triton_f32_cpu: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        k_descale: torch.Tensor | None,
        v_descale: torch.Tensor | None,
        scratch_storage: torch.Tensor,
        returned_output: torch.Tensor,
        q_f32_cpu_before: torch.Tensor,
        key_pages_f32_cpu_before: torch.Tensor,
        value_pages_f32_cpu_before: torch.Tensor,
        active_page_ids_cpu: torch.Tensor,
        mode: Literal["decode", "extend"],
        num_reqs: int,
        q_finite_before: bool | None,
        q2k_range_before: tuple[int, int] | None,
        plan_debug: dict[str, Any],
        layer_name: str,
    ) -> None:
        b12x_f32 = b12x_output.float().cpu()
        triton_f32 = triton_f32_cpu
        q_finite_after = bool(torch.isfinite(q.float()).all().item())
        q2k_range_after = (
            int(q2k_indices.min().item()),
            int(q2k_indices.max().item()),
        )
        finite_b12x = bool(torch.isfinite(b12x_f32).all().item())
        finite_triton = bool(torch.isfinite(triton_f32).all().item())
        b12x_absmax = float(b12x_f32.abs().max().item())
        triton_absmax = float(triton_f32.abs().max().item())
        q_f32_cpu_after = q.float().cpu()
        key_pages_f32_cpu_after, value_pages_f32_cpu_after = (
            self._active_kv_pages_f32_cpu(
                key_cache,
                value_cache,
                active_page_ids_cpu,
            )
        )
        q_absmax = float(q_f32_cpu_after.abs().max().item())
        active_k_absmax, active_v_absmax = self._active_kv_absmax(
            key_cache,
            value_cache,
            active_page_ids_cpu,
        )
        q_mut_max = float((q_f32_cpu_after - q_f32_cpu_before).abs().max().item())
        k_mut_max = float(
            (key_pages_f32_cpu_after - key_pages_f32_cpu_before).abs().max().item()
        )
        v_mut_max = float(
            (value_pages_f32_cpu_after - value_pages_f32_cpu_before).abs().max().item()
        )
        q_storage_ptr = int(q.untyped_storage().data_ptr())
        out_storage_ptr = int(b12x_output.untyped_storage().data_ptr())
        returned_out_storage_ptr = int(returned_output.untyped_storage().data_ptr())
        q2k_storage_ptr = int(q2k_indices.untyped_storage().data_ptr())
        scratch_storage_ptr = int(scratch_storage.untyped_storage().data_ptr())
        scratch_range = _tensor_addr_range(scratch_storage)
        abs_diff = (b12x_f32 - triton_f32).abs()
        max_abs = float(abs_diff.max().item())
        mean_abs = float(abs_diff.mean().item())
        cos = float(
            torch.nn.functional.cosine_similarity(
                b12x_f32.reshape(-1), triton_f32.reshape(-1), dim=0
            ).item()
        )
        compare_atol = _env_float(_B12X_MSA_COMPARE_ATOL, 1e-2)
        log_all = os.getenv(_B12X_MSA_COMPARE_LOG_ALL, "0") == "1"
        mismatch = (not finite_b12x) or (not finite_triton) or max_abs > compare_atol
        if mismatch or log_all:
            logger.warning(
                "B12X MiniMax M3 MSA compare layer=%s mode=%s q=%s reqs=%d "
                "q2k=%s finite=%s/%s q_finite=%s/%s q2k_range=%s/%s "
                "max_abs=%.6g mean_abs=%.6g cos=%.8f "
                "absmax b12x/triton/q/k/v=%.6g/%.6g/%.6g/%.6g/%.6g "
                "mut q/k/v=%.6g/%.6g/%.6g "
                "stride q/out/ret/k/v=%s/%s/%s/%s/%s active_pages=%d "
                "descale_stride=%s/%s alias q/out/ret/q2k/scratch=%s/%s/%s/%s "
                "overlap q/out/ret/q2k/scratch=%s/%s/%s/%s "
                "plan=%s scratch=%s",
                layer_name,
                mode,
                tuple(q.shape),
                num_reqs,
                tuple(q2k_indices.shape),
                finite_b12x,
                finite_triton,
                q_finite_before,
                q_finite_after,
                q2k_range_before,
                q2k_range_after,
                max_abs,
                mean_abs,
                cos,
                b12x_absmax,
                triton_absmax,
                q_absmax,
                active_k_absmax,
                active_v_absmax,
                q_mut_max,
                k_mut_max,
                v_mut_max,
                tuple(q.stride()),
                tuple(b12x_output.stride()),
                tuple(returned_output.stride()),
                tuple(key_cache.stride()),
                tuple(value_cache.stride()),
                int(active_page_ids_cpu.numel()),
                None if k_descale is None else tuple(k_descale.stride()),
                None if v_descale is None else tuple(v_descale.stride()),
                q_storage_ptr == scratch_storage_ptr,
                out_storage_ptr == scratch_storage_ptr,
                returned_out_storage_ptr == scratch_storage_ptr,
                q2k_storage_ptr == scratch_storage_ptr,
                _ranges_overlap(_tensor_addr_range(q), scratch_range),
                _ranges_overlap(_tensor_addr_range(b12x_output), scratch_range),
                _ranges_overlap(_tensor_addr_range(returned_output), scratch_range),
                _ranges_overlap(_tensor_addr_range(q2k_indices), scratch_range),
                plan_debug["plan"],
                plan_debug["scratch"],
            )
        self._dump_first_mismatch(
            q=q,
            key_cache=key_cache,
            value_cache=value_cache,
            q2k_indices=q2k_indices,
            b12x_f32=b12x_f32,
            triton_f32=triton_f32,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            active_page_ids_cpu=active_page_ids_cpu,
            q_f32_cpu_before=q_f32_cpu_before,
            key_pages_f32_cpu_before=key_pages_f32_cpu_before,
            value_pages_f32_cpu_before=value_pages_f32_cpu_before,
            q_f32_cpu_after=q_f32_cpu_after,
            key_pages_f32_cpu_after=key_pages_f32_cpu_after,
            value_pages_f32_cpu_after=value_pages_f32_cpu_after,
            k_descale=k_descale,
            v_descale=v_descale,
            layer_name=layer_name,
            mode=mode,
            max_abs=max_abs,
            compare_atol=compare_atol,
            finite_b12x=finite_b12x,
            finite_triton=finite_triton,
            plan_debug=plan_debug,
        )

    def _active_page_ids_cpu(
        self,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        page_table_cpu = page_table.detach().cpu()
        cache_lens_cpu = cache_seqlens.detach().cpu()
        active_pages: list[torch.Tensor] = []
        for req_idx, cache_len_tensor in enumerate(cache_lens_cpu):
            cache_len = int(cache_len_tensor.item())
            num_pages = (cache_len + _B12X_MINIMAX_MSA_PAGE_SIZE - 1) // (
                _B12X_MINIMAX_MSA_PAGE_SIZE
            )
            if num_pages > 0:
                active_pages.append(page_table_cpu[req_idx, :num_pages])
        if not active_pages:
            return torch.empty((0,), dtype=torch.long)
        return torch.cat(active_pages).to(torch.long).unique(sorted=True)

    def _active_kv_absmax(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        active_page_ids_cpu: torch.Tensor,
    ) -> tuple[float, float]:
        if active_page_ids_cpu.numel() == 0:
            return 0.0, 0.0
        active_page_ids = active_page_ids_cpu.to(
            device=key_cache.device,
            dtype=torch.long,
        )
        key_pages = key_cache.index_select(0, active_page_ids)
        value_pages = value_cache.index_select(0, active_page_ids)
        return (
            float(key_pages.float().abs().max().item()),
            float(value_pages.float().abs().max().item()),
        )

    def _active_kv_pages_f32_cpu(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        active_page_ids_cpu: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if active_page_ids_cpu.numel() == 0:
            empty = torch.empty((0,), dtype=torch.float32)
            return empty, empty
        active_page_ids = active_page_ids_cpu.to(
            device=key_cache.device,
            dtype=torch.long,
        )
        key_pages = key_cache.index_select(0, active_page_ids).float().cpu()
        value_pages = value_cache.index_select(0, active_page_ids).float().cpu()
        return key_pages, value_pages

    def _dump_first_mismatch(
        self,
        *,
        q: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        q2k_indices: torch.Tensor,
        b12x_f32: torch.Tensor,
        triton_f32: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        active_page_ids_cpu: torch.Tensor,
        q_f32_cpu_before: torch.Tensor,
        key_pages_f32_cpu_before: torch.Tensor,
        value_pages_f32_cpu_before: torch.Tensor,
        q_f32_cpu_after: torch.Tensor,
        key_pages_f32_cpu_after: torch.Tensor,
        value_pages_f32_cpu_after: torch.Tensor,
        k_descale: torch.Tensor | None,
        v_descale: torch.Tensor | None,
        layer_name: str,
        mode: Literal["decode", "extend"],
        max_abs: float,
        compare_atol: float,
        finite_b12x: bool,
        finite_triton: bool,
        plan_debug: dict[str, Any],
    ) -> None:
        dump_dir = os.getenv(_B12X_MSA_DUMP_DIR)
        if not dump_dir or self._triton_compare_dumps > 0:
            return
        if finite_b12x and finite_triton and max_abs <= compare_atol:
            return
        os.makedirs(dump_dir, exist_ok=True)
        active_page_ids = active_page_ids_cpu.to(
            device=key_cache.device,
            dtype=torch.long,
        )
        safe_layer = layer_name.replace("/", "_").replace(".", "_")
        rank_label = _debug_rank_label()
        device_index = q.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        path = os.path.join(
            dump_dir,
            "b12x_minimax_m3_msa_"
            f"{rank_label}_dev{device_index}_{mode}_{safe_layer}.pt",
        )
        torch.save(
            {
                "rank": rank_label,
                "device_index": device_index,
                "layer_name": layer_name,
                "mode": mode,
                "plan_debug": plan_debug,
                "q": q.detach().cpu(),
                "q_before": q_f32_cpu_before,
                "q_after": q_f32_cpu_after,
                "q_stride": tuple(q.stride()),
                "key_pages": key_cache.index_select(0, active_page_ids).cpu(),
                "key_pages_before": key_pages_f32_cpu_before,
                "key_pages_after": key_pages_f32_cpu_after,
                "key_cache_stride": tuple(key_cache.stride()),
                "value_pages": value_cache.index_select(0, active_page_ids).cpu(),
                "value_pages_before": value_pages_f32_cpu_before,
                "value_pages_after": value_pages_f32_cpu_after,
                "value_cache_stride": tuple(value_cache.stride()),
                "active_page_ids": active_page_ids_cpu,
                "q2k_indices": q2k_indices.detach().cpu(),
                "page_table": page_table.detach().cpu(),
                "cache_seqlens": cache_seqlens.detach().cpu(),
                "cu_seqlens_q": cu_seqlens_q.detach().cpu(),
                "k_descale": None if k_descale is None else k_descale.detach().cpu(),
                "v_descale": None if v_descale is None else v_descale.detach().cpu(),
                "b12x_output": b12x_f32,
                "triton_output": triton_f32,
                "max_abs": max_abs,
                "finite_b12x": finite_b12x,
                "finite_triton": finite_triton,
            },
            path,
        )
        self._triton_compare_dumps += 1
        logger.warning("Dumped B12X MiniMax M3 MSA mismatch tensors to %s", path)

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        topk_idx: tuple[torch.Tensor | None, torch.Tensor | None],
        output: torch.Tensor,
    ) -> torch.Tensor:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return output
        main_md = attn_metadata[layer.layer_name]  # type: ignore[attr-defined]
        assert isinstance(main_md, MiniMaxM3SparseMetadata)
        decode_topk, prefill_topk = topk_idx

        nd = main_md.num_decode_tokens
        num_tokens = main_md.num_actual_tokens
        hd = self.head_size
        q = query[:num_tokens].view(-1, self.num_heads, hd)
        out = output[:num_tokens].view(-1, self.num_heads, hd)
        key_cache, value_cache = self._kv_cache_views(kv_cache)

        if main_md.num_decodes > 0:
            d = main_md.decode
            assert d is not None and decode_topk is not None
            decode_mode: Literal["decode", "extend"] = (
                "decode" if d.decode_query_len == 1 else "extend"
            )
            self._run_b12x_slice(
                q=q[:nd],
                kv_cache=kv_cache,
                key_cache=key_cache,
                value_cache=value_cache,
                output=out[:nd],
                page_table=d.block_table,
                cache_seqlens=d.seq_lens,
                cu_seqlens_q=d.cu_seqlens_q,
                q2k_indices=decode_topk,
                mode=decode_mode,
                layer_name=layer.layer_name,
            )
            self._maybe_log_sparse_stats(
                layer_name=layer.layer_name,
                mode=decode_mode,
                q=q[:nd],
                out=out[:nd],
                topk=decode_topk,
                seq_lens=d.seq_lens,
            )

        if main_md.num_prefills > 0:
            p = main_md.prefill
            assert p is not None and prefill_topk is not None
            self._run_b12x_slice(
                q=q[nd:],
                kv_cache=kv_cache,
                key_cache=key_cache,
                value_cache=value_cache,
                output=out[nd:],
                page_table=p.block_table,
                cache_seqlens=p.seq_lens,
                cu_seqlens_q=p.cu_seqlens_q,
                q2k_indices=prefill_topk,
                mode="extend",
                prefix_lens=p.context_lens,
                max_query_len=p.max_query_len,
                layer_name=layer.layer_name,
            )
            self._maybe_log_sparse_stats(
                layer_name=layer.layer_name,
                mode="extend",
                q=q[nd:],
                out=out[nd:],
                topk=prefill_topk,
                seq_lens=p.seq_lens,
            )

        return output
