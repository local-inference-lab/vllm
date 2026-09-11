# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic ordering of the B12X sparse-indexer selection.

The B12X DSA indexer packs the selected KV rows of a query in the order its
CTAs reserve output slots, an atomic arrival order that follows kernel
timing. The sparse MLA decode kernel sums the selected rows in that order in
fp32, so a change in the timing of the kernels around the indexer (a
concurrent weight prefetch, a different launch sequence) moves the summation
order and, through the BF16 rounding of the layer output, the deep-tail log
probabilities of long contexts. Sorting each row makes the order a function
of the selected set alone.

The sort is the b12x op ``attention.topk_sort`` (``sort_convert``): the
indexer emits logical positions, and the sort rewrites each row ascending and
converts the positions to physical cache slots in place. This module owns
the dispatch: the run-time gates, the side stream inside full CUDA graphs and
the join before the first consumer.

Inside full CUDA graphs the decode-path sort runs on its own stream
(``sort_convert_async``: the stream waits for the indexer and the sort is
enqueued there; ``join`` makes the main stream wait for it). The join sits in
the sparse MLA backend's ``forward_mqa`` right before the first read of the
selection. The KV-cache write and the query projections of this model run
before the indexer, so the main-stream work between the indexer and the
attention kernels is the latent query projection (the W_UK batched matmul,
which the attention module issues after the indexer launch on the bf16
query path for this purpose) and the query concatenation that
``forward_mqa`` runs; those two kernels are what hides the sort. A join
placed before the attention op would leave nothing to overlap: the captured
chain would be linear and the driver would replay it on one stream. The sort
stream is not the L2-prefetch side
stream: that one carries the output-projection prefetch, issued at the start
of the attention block, and a sort enqueued behind it would make the
attention kernels wait for the prefetch. Eager forwards sort in line: a fork
records an event per call, and events pending behind a host that runs ahead
retain device memory.

The fork retains its input tensors until the join. The indexer hands the
sort a per-token page table that is a temporary of the indexer call (the
per-request table repeated once per query token); the caching allocator
returns that memory to the pool when the last Python reference drops, at the
indexer's return, and orders its reuse only against the allocating stream.
Without the retention, the next main-stream allocation between the fork and
the join (the latent query projection's output) reuses the memory while the
sort stream still reads it, and the sort emits slots from an overwritten
table that the attention kernel then dereferences.

``VLLM_TOPK_SORT=0`` disables the sort and ``VLLM_TOPK_SORT_MAX_TOKENS``
(default 8) bounds the row count that sorts. Both are read once per process
and decide, at model construction, which indexer plans emit logical
positions; they never enter a compile or cache key.
"""

from __future__ import annotations

import os

import torch

from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger

logger = init_logger(__name__)

ENABLED = os.environ.get("VLLM_TOPK_SORT", "1") != "0"
#: Largest row count (tokens of the step) that is sorted. The sort exists so
#: that loads overlapping the indexer and the attention kernels (the L2
#: prefetch attention window, gated by ``VLLM_L2_PREFETCH_ATTN_MAX_TOKENS``)
#: cannot change the attention's summation order; steps beyond that window
#: do not pay for it. 0 sorts every row count.
MAX_TOKENS = int(os.environ.get("VLLM_TOPK_SORT_MAX_TOKENS", "8"))

#: Inputs of a forked sort, per device, held until ``join``; see the module
#: docstring for why the references must outlive the caller's temporaries.
_pending: dict[int, tuple[torch.Tensor, ...]] = {}
_streams: dict[int, torch.cuda.Stream] = {}


def active(rows: int) -> bool:
    """Whether a step of ``rows`` tokens sorts its selection."""
    return ENABLED and (MAX_TOKENS <= 0 or rows <= MAX_TOKENS)


def _op():
    from b12x.attention import topk_sort

    return topk_sort


def is_supported(device: torch.device) -> bool:
    """Whether the sort serves ``device``; False, with one warning, when the
    installed b12x has no ``attention.topk_sort`` op, so the indexer keeps
    its unsorted selection instead of failing at model construction."""
    if not ENABLED:
        return False
    try:
        return bool(_op().is_supported(device))
    except (ImportError, AttributeError) as exc:
        logger.warning_once("B12X selection sort disabled: %s", exc)
        return False


def precompile(max_positions: int, device: torch.device) -> None:
    """Compile and warm-run the sort before any CUDA-graph capture."""
    _op().precompile(max_positions, device)


def _device_index(device: torch.device) -> int:
    return (
        device.index
        if device.index is not None
        else torch.accelerator.current_device_index()
    )


def _stream(device: torch.device) -> torch.cuda.Stream:
    index = _device_index(device)
    stream = _streams.get(index)
    if stream is None:
        stream = torch.cuda.Stream(device=device)
        _streams[index] = stream
    return stream


def _graph_mode() -> bool:
    if torch.cuda.is_current_stream_capturing():
        return True
    if not is_forward_context_available():
        return False
    return get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL


def sort_convert(
    indices: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    max_positions: int,
) -> torch.Tensor:
    """Sort each row of logical positions ascending and convert them in place
    to physical slots, on the current stream."""
    _op().sort_convert(
        indices,
        seq_lens.to(torch.int32),
        block_table,
        block_size,
        max_positions,
    )
    return indices


def sort_convert_async(
    indices: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    max_positions: int,
) -> torch.Tensor:
    """``sort_convert`` on the sort stream inside full CUDA graphs, in line
    otherwise. The caller must run ``join`` on the main stream before the
    first consumer of ``indices``."""
    if not _graph_mode():
        return sort_convert(indices, seq_lens, block_table, block_size, max_positions)
    device = indices.device
    main = torch.cuda.current_stream(device)
    side = _stream(device)
    side.wait_stream(main)
    with torch.cuda.stream(side):
        sort_convert(indices, seq_lens, block_table, block_size, max_positions)
    _pending[_device_index(device)] = (indices, seq_lens, block_table)
    return indices


def join(device: torch.device) -> None:
    """Main-stream wait for a pending ``sort_convert_async``; a no-op when
    nothing is pending."""
    index = _device_index(device)
    if index not in _pending:
        return
    torch.cuda.current_stream(device).wait_stream(_stream(device))
    # Released after the wait: a reuse of the temporaries by a later
    # main-stream allocation is then ordered behind the sort.
    del _pending[index]
