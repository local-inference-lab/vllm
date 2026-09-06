# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor-parallel collectives for Kimi-K3 projection outputs."""

import os
from functools import lru_cache

import torch

import vllm.envs as envs
from vllm.distributed import (
    get_dcp_group,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_all_reduce_in_place,
    tensor_model_parallel_is_borrowed_storage,
    tensor_model_parallel_pcie_all_gather_pair,
    tensor_model_parallel_pcie_reduce_scatter_columns,
    tensor_model_parallel_prepare_pcie_reduce_scatter,
)
from vllm.v1.attention.ops import dcp_alltoall
from vllm.v1.attention.ops.dcp_alltoall import (
    dcp_b12x_all_gather_heads,
    dcp_b12x_all_gather_pair,
)

_KIMI_B12X_PAIRED_PROJECTION_MAX_TOKENS = 8
_KIMI_INPLACE_REDUCTION_MIN_TOKENS = 1024


@lru_cache(maxsize=1)
def kimi_ring_static_io_enabled() -> bool:
    """``VLLM_K3_RING_STATIC_IO=1``: prefill in-place reductions borrow the
    B12X DMA ring's static output instead of copying it out.

    The reduced tensor then aliases ring memory that the next reduction of
    the same shape overwrites. Kimi-K3 consumes each such tensor before that
    point (attention output -> post-attention norm and MoE input; MoE output
    -> next layer's pre-attention norm) and materializes the two tensors it
    retains longer (``materialize_kimi_reduction``: the AttnRes block-write
    prefix and the model output). Off by default.
    """
    return os.getenv("VLLM_K3_RING_STATIC_IO", "0") == "1"


def reduce_kimi_full_width_projection(
    output_parallel: torch.Tensor,
    tp_size: int,
) -> torch.Tensor:
    """Reduce a Kimi full-width row-parallel projection.

    A projection with at least 1,024 rows is a prefill intermediate whose
    rank-local value is dead after reduction. NCCL may therefore overwrite
    that storage and avoid an equally sized output allocation. Smaller
    projections retain the ordinary functional collective used by decode and
    CUDA Graph capture. With ``kimi_ring_static_io_enabled()`` the result may
    be the DMA ring's static output (see there).
    """
    if tp_size <= 1:
        return output_parallel
    if (
        output_parallel.ndim == 2
        and output_parallel.shape[0] >= _KIMI_INPLACE_REDUCTION_MIN_TOKENS
        and output_parallel.is_contiguous()
    ):
        if kimi_ring_static_io_enabled():
            return tensor_model_parallel_all_reduce_in_place(
                output_parallel, borrow_output=True
            )
        return tensor_model_parallel_all_reduce_in_place(output_parallel)
    return tensor_model_parallel_all_reduce(output_parallel)


def materialize_kimi_reduction(tensor: torch.Tensor) -> torch.Tensor:
    """Return ``tensor`` in caller-owned storage.

    A tensor a borrowed reduction returned is copied out; any other tensor
    is returned as is. Call this before retaining a reduction result across
    the next same-shape reduction.
    """
    if kimi_ring_static_io_enabled() and tensor_model_parallel_is_borrowed_storage(
        tensor
    ):
        return tensor.clone()
    return tensor


def kimi_reduction_is_borrowed(tensor: torch.Tensor) -> bool:
    """Whether ``tensor`` is a borrowed reduction result (ring-owned)."""
    return kimi_ring_static_io_enabled() and tensor_model_parallel_is_borrowed_storage(
        tensor
    )


_KIMI_PROJECTION_GATHER_MODES = ("nccl", "dma_pair")
KIMI_DMA_PAIR_GATHER_MIN_TOKENS = 1024
_KIMI_LATENT_REDUCE_MODES = ("allreduce", "rs_fp32", "rs_bf16")
KIMI_LATENT_REDUCE_SCATTER_MIN_TOKENS = 1024


@lru_cache(maxsize=1)
def kimi_latent_reduce_mode() -> str:
    """``VLLM_K3_LATENT_REDUCE``: how a prefill layer reduces the TP-partial
    routed latent before its RMSNorm and row-parallel up-projection.

    ``allreduce`` (default): the full-width all-reduce (DMA ring, eight
    bf16 roundings at TP9), then each rank normalizes the full latent and
    reads its input shard. ``rs_fp32`` / ``rs_bf16``: a column reduce-scatter
    on the DMA ring returning only this rank's input shard, with an fp32
    running sum on the wire (one bf16 rounding) or bf16 hops (eight); the
    RMSNorm variance then comes from fp64 per-shard sums of squares combined
    across ranks. Falls back to ``allreduce`` per call when the ring
    declines or the call is decode-sized.
    """
    mode = os.getenv("VLLM_K3_LATENT_REDUCE", "allreduce").strip().lower()
    if mode not in _KIMI_LATENT_REDUCE_MODES:
        raise ValueError(
            "VLLM_K3_LATENT_REDUCE must be one of "
            f"{_KIMI_LATENT_REDUCE_MODES}, got {mode!r}"
        )
    return mode


def kimi_latent_reduce_scatter_wire() -> str | None:
    """The reduce-scatter wire of ``kimi_latent_reduce_mode()``, or ``None``
    for the all-reduce."""
    mode = kimi_latent_reduce_mode()
    return None if mode == "allreduce" else mode.removeprefix("rs_")


def prepare_kimi_latent_reduce_scatter() -> bool:
    """Compile the ring's reduce-scatter kernels for the configured wire at
    model build time, before any kernel freeze or graph capture."""
    wire = kimi_latent_reduce_scatter_wire()
    if wire is None:
        return False
    return tensor_model_parallel_prepare_pcie_reduce_scatter(wire)


def try_reduce_scatter_kimi_latent(
    partial: torch.Tensor, *, cols: int
) -> torch.Tensor | None:
    """Reduce a prefill TP-partial latent to this rank's ``[rows, cols]``
    column block on the DMA ring; ``None`` keeps the all-reduce path."""
    wire = kimi_latent_reduce_scatter_wire()
    if wire is None:
        return None
    if (
        partial.ndim != 2
        or partial.shape[0] < KIMI_LATENT_REDUCE_SCATTER_MIN_TOKENS
        or partial.dtype != torch.bfloat16
        or not partial.is_contiguous()
        or get_tensor_model_parallel_world_size() <= 1
    ):
        return None
    return tensor_model_parallel_pcie_reduce_scatter_columns(
        partial, wire=wire, cols=cols
    )


@lru_cache(maxsize=1)
def kimi_projection_gather_mode() -> str:
    """``VLLM_K3_PROJECTION_GATHER``: how a prefill layer gathers its router
    logits and routed latent across TP ranks.

    ``nccl`` (default): two PyNCCL all-gathers on the main stream.
    ``dma_pair``: one B12X DMA ring pass carrying both blocks, issued on the
    ring's side stream so the shared experts run underneath; falls back to
    ``nccl`` per call when the ring is unavailable or the call is decode-sized.
    """
    mode = os.getenv("VLLM_K3_PROJECTION_GATHER", "nccl").strip().lower()
    if mode not in _KIMI_PROJECTION_GATHER_MODES:
        raise ValueError(
            "VLLM_K3_PROJECTION_GATHER must be one of "
            f"{_KIMI_PROJECTION_GATHER_MODES}, got {mode!r}"
        )
    return mode


def assemble_rank_major_blocks(blocks: torch.Tensor, width: int) -> torch.Tensor:
    """Concatenate rank-major ``[world, rows, c]`` blocks into ``[rows, width]``.

    Rank ``r`` owns logical columns ``[r*c, (r+1)*c)``; columns past ``width``
    (the last rank's zero-filled padding) are dropped. One pass: the full
    blocks are copied through a ``[rows, full, c]`` view, the partial last
    block through a narrow copy.
    """
    world, rows, cols = blocks.shape
    if width <= 0 or width > world * cols:
        raise ValueError(f"width {width} does not fit {world} blocks of {cols} columns")
    out = blocks.new_empty((rows, width))
    full = width // cols
    if full:
        out[:, : full * cols].view(rows, full, cols).copy_(
            blocks[:full].permute(1, 0, 2)
        )
    rest = width - full * cols
    if rest:
        out[:, full * cols :].copy_(blocks[full, :, :rest])
    return out


class PendingProjectionGather:
    """A paired projection gather in flight on the ring's side stream.

    ``wait`` orders the current stream after the gather and assembles the
    logical ``[rows, width]`` tensors from the rank-major blocks (the same
    values the NCCL path produces, since an all-gather only copies).
    """

    def __init__(
        self,
        out_first: torch.Tensor,
        out_second: torch.Tensor,
        done,
        first_width: int,
        second_width: int,
    ) -> None:
        self._out_first = out_first
        self._out_second = out_second
        self._done = done
        self._first_width = first_width
        self._second_width = second_width

    def wait(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._done is not None:
            torch.cuda.current_stream().wait_event(self._done)
        first = assemble_rank_major_blocks(self._out_first, self._first_width)
        second = assemble_rank_major_blocks(self._out_second, self._second_width)
        return first, second


class CompletedProjectionGather:
    """Gathered projections that need no wait (the NCCL fallback)."""

    def __init__(self, first: torch.Tensor, second: torch.Tensor) -> None:
        self._first = first
        self._second = second

    def wait(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._first, self._second


def gather_kimi_projection_pair_prefill(
    local_first: torch.Tensor,
    first_width: int,
    local_second: torch.Tensor,
    second_width: int,
) -> PendingProjectionGather | CompletedProjectionGather:
    """Gather two prefill projection shards, on the DMA ring when possible.

    Returns a pending gather (ring side stream) or, when the ring declines,
    the completed NCCL gathers, each sliced to its logical width as
    ``KimiPaddedColumnParallelLinear.forward`` would.
    """
    pending = try_gather_kimi_projection_pair_async(
        local_first, first_width, local_second, second_width
    )
    if pending is not None:
        return pending
    first = gather_kimi_sharded_projection(local_first)[..., :first_width]
    second = gather_kimi_sharded_projection(local_second)[..., :second_width]
    return CompletedProjectionGather(first.contiguous(), second.contiguous())


def try_gather_kimi_projection_pair_async(
    local_first: torch.Tensor,
    first_width: int,
    local_second: torch.Tensor,
    second_width: int,
) -> PendingProjectionGather | None:
    """Start a prefill-size paired gather on the DMA ring's side stream.

    ``local_first`` / ``local_second`` are this rank's ``[rows, c]`` shards of
    two column-parallel projections whose logical widths are ``first_width``
    and ``second_width``. Returns ``None`` (caller uses the NCCL path) unless
    ``kimi_projection_gather_mode()`` is ``dma_pair``, the rows are prefill
    sized and the TP group's ring accepts the pair.
    """
    if kimi_projection_gather_mode() != "dma_pair":
        return None
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size <= 1:
        return None
    if not (
        local_first.ndim == local_second.ndim == 2
        and local_first.shape[0] == local_second.shape[0]
        and local_first.shape[0] >= KIMI_DMA_PAIR_GATHER_MIN_TOKENS
        and local_first.is_contiguous()
        and local_second.is_contiguous()
        and local_first.shape[1] * tp_size >= first_width
        and local_second.shape[1] * tp_size >= second_width
    ):
        return None
    if _get_kimi_projection_group().world_size != tp_size:
        return None
    gathered = tensor_model_parallel_pcie_all_gather_pair(local_first, local_second)
    if gathered is None:
        return None
    out_first, out_second, done = gathered
    return PendingProjectionGather(
        out_first, out_second, done, first_width, second_width
    )


def _get_kimi_projection_group():
    """Return the coordinator that spans every projection weight shard.

    Projection weights are sharded across the full tensor-parallel group. The
    DCP coordinator is valid only when its ordered rank list matches the
    tensor-parallel coordinator.
    """
    tp_size = get_tensor_model_parallel_world_size()
    dcp_group = get_dcp_group()
    tp_group = get_tp_group()
    if tp_group.world_size != tp_size:
        raise RuntimeError(
            "Kimi projection group does not span tensor-parallel ranks: "
            f"group={tp_group.world_size}, TP={tp_size}"
        )
    if dcp_group.world_size == tp_size and list(dcp_group.ranks) == list(
        tp_group.ranks
    ):
        return dcp_group
    return tp_group


def _try_b12x_kimi_projection_gather(
    output_parallel: torch.Tensor,
) -> torch.Tensor | None:
    """Gather one decode projection over the lossless B12X copy channel."""
    if (
        not envs.VLLM_USE_B12X_DCP_A2A
        or output_parallel.ndim != 2
        or output_parallel.shape[0] != 1
        or not output_parallel.is_cuda
        or not output_parallel.is_contiguous()
    ):
        return None

    tp_size = get_tensor_model_parallel_world_size()
    if tp_size <= 1:
        return None
    projection_group = _get_kimi_projection_group()

    local_width = output_parallel.shape[1]
    restore_dtype: torch.dtype | None = None
    strip_local_width: int | None = None
    if output_parallel.dtype in (torch.float16, torch.bfloat16):
        if local_width % 8 == 0:
            transport = output_parallel.view(1, 1, local_width)
        else:
            padded_width = (local_width + 7) // 8 * 8
            transport = torch.nn.functional.pad(
                output_parallel, (0, padded_width - local_width)
            ).view(1, 1, padded_width)
            strip_local_width = local_width
    elif output_parallel.dtype == torch.float32:
        raw_width = local_width * output_parallel.element_size()
        if raw_width % 8 != 0:
            return None
        # The FP8 view exposes one-byte transport lanes without converting the
        # FP32 payload. The gathered result is restored to the original dtype.
        transport = output_parallel.view(torch.float8_e4m3fn).view(1, 1, raw_width)
        restore_dtype = torch.float32
    elif output_parallel.dtype == torch.float8_e4m3fn:
        if local_width % 16 != 0:
            return None
        transport = output_parallel.view(1, 1, local_width)
    else:
        return None

    gathered = dcp_b12x_all_gather_heads(
        transport,
        projection_group,
        max_batch_size=1,
    )
    if strip_local_width is not None:
        gathered = gathered.narrow(-1, 0, strip_local_width).contiguous()
    gathered = gathered.flatten(1)
    if restore_dtype is not None:
        gathered = gathered.view(restore_dtype)
    return gathered


def gather_kimi_sharded_projection(output_parallel: torch.Tensor) -> torch.Tensor:
    """Gather a rank-major Kimi-K3 projection through a lossless fast path."""
    if get_tensor_model_parallel_world_size() <= 1:
        return output_parallel
    gathered = _try_b12x_kimi_projection_gather(output_parallel)
    if gathered is not None:
        return gathered
    return tensor_model_parallel_all_gather(output_parallel, dim=-1)


def gather_kimi_sharded_projection_pair(
    local_first: torch.Tensor,
    local_second: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather two decode projections behind one lossless B12X barrier."""
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size <= 1:
        return local_first, local_second
    if (
        local_first.ndim == local_second.ndim == 2
        and local_first.shape[0] == local_second.shape[0]
        and 0 < local_first.shape[0] <= _KIMI_B12X_PAIRED_PROJECTION_MAX_TOKENS
        and local_first.is_cuda
        and local_second.is_cuda
        and local_first.is_contiguous()
        and local_second.is_contiguous()
    ):
        projection_group = _get_kimi_projection_group()
        if projection_group.world_size == tp_size:
            return dcp_b12x_all_gather_pair(
                local_first,
                local_second,
                projection_group,
                max_batch_size=_KIMI_B12X_PAIRED_PROJECTION_MAX_TOKENS,
            )
    return (
        gather_kimi_sharded_projection(local_first),
        gather_kimi_sharded_projection(local_second),
    )


def try_gather_kimi_sharded_projection_pair_topk(
    local_down: torch.Tensor,
    local_router: torch.Tensor,
    correction_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Use B12X Kimi projection transport and precomputed expert selection.

    A missing or ineligible binding returns ``None``. The model caller must
    then use the exact paired gather and ordinary router operations.
    """
    if get_tensor_model_parallel_world_size() <= 1:
        return None
    pair_topk = getattr(
        dcp_alltoall,
        "try_dcp_b12x_all_gather_pair_kimi_topk",
        None,
    )
    if pair_topk is None:
        return None
    return pair_topk(
        local_down,
        local_router,
        correction_bias,
        _get_kimi_projection_group(),
    )


def try_select_kimi_routed_experts(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Use B12X CuTeDSL expert selection for assembled Kimi router logits."""
    select_topk = getattr(dcp_alltoall, "try_b12x_kimi_topk16", None)
    if select_topk is None:
        return None
    return select_topk(
        router_logits,
        correction_bias,
    )
