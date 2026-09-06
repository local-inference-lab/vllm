# SPDX-License-Identifier: Apache-2.0
"""Split one prefill chunk into two row halves and run them as consecutive
sub-steps (Kimi-K3, eager chunked prefill).

Stage 1 of the TP all-reduce / compute overlap design
(research/prefill-w4a16-20260902/DESIGN-intra-request-ubatch-prefill-20260902.md,
research/prefill-campaign-20260906/r1-split-prefill.md): the halves run
sequentially on the current stream, so every cross-half dependency is
satisfied by stream order. Cross-half state and its exactness:

- KDA recurrent/conv state: carried in the request's mamba slot; exact when
  the boundary is a FlashKDA tile boundary (``SPLIT_ALIGNMENT``).
- MLA keys of the first half: bf16 stash consumed by the second half
  (``mla.py``, ``k3_split_exact``); the fp8 cache path is not exact.
- TP all-reduces: the B12X DMA ring orders its reduction by the row block
  ``row // (rows / world)``, which depends on the row count, so the halves
  are not bit-identical to the unsplit chunk unless the ring uses its
  row-count-invariant granule mapping (``B12X_PCIE_RING_GRANULE_ROWS=g``).
  With that mapping the boundary must fall on a ``world * g`` row period
  and the chunk must be a whole number of periods (``split_point``);
  chunks that are not run whole.

The overlap (two threads, yield points at the collectives) comes on top of
the split (``_run_overlapped``).

Eligibility: a single request, no draft tokens, eager (non-graph) dispatch,
at least ``VLLM_K3_UBATCH_PREFILL_MIN_TOKENS`` scheduled tokens, enabled
with ``VLLM_K3_UBATCH_PREFILL=1``. Anything else runs the normal path.
"""
from __future__ import annotations

import dataclasses
import os
import threading
import time

import numpy as np
import torch

from vllm.forward_context import (
    create_forward_context,
    override_forward_context,
    set_forward_context,
)
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.input_batch import InputBatch


_MODE_CACHE: list = [0.0, None]


def runtime_mode() -> str | None:
    """Operator override of the split mode without a restart.

    When ``VLLM_K3_UBATCH_MODE_FILE`` names a file, its first word selects
    the mode for the next split-eligible forwards: ``off`` (no split),
    ``stage1`` (sequential halves, one thread), ``noyield`` (two threads
    and ubatch ids, collectives on the compute stream, no hand-off),
    ``lockstep`` (hand-offs with the device drained around each),
    ``overlap``, or ``inexact`` (overlap with the second half reading the
    first half through the KV cache instead of the bf16 stash). The file is
    re-read at most once per second; a missing file leaves the
    environment-derived behaviour in place.
    """
    path = os.getenv("VLLM_K3_UBATCH_MODE_FILE", "")
    if not path:
        return None
    now = time.time()
    if now - _MODE_CACHE[0] < 1.0:
        return _MODE_CACHE[1]
    try:
        with open(path) as fh:
            mode = fh.read().split()[0].strip().lower()
    except Exception:
        mode = None
    _MODE_CACHE[0] = now
    _MODE_CACHE[1] = mode
    return mode


def ubatch_prefill_enabled() -> bool:
    mode = runtime_mode()
    if mode is not None:
        return mode != "off"
    return os.getenv("VLLM_K3_UBATCH_PREFILL", "0") == "1"


def ubatch_prefill_min_tokens() -> int:
    return int(os.getenv("VLLM_K3_UBATCH_PREFILL_MIN_TOKENS", "1024"))


def ubatch_prefill_overlap() -> bool:
    """Stage 2: run the halves on two threads with the TP all-reduces on a
    shared comm stream (see communication_op._ubatch_comm_region)."""
    mode = runtime_mode()
    if mode is not None:
        return mode in ("noyield", "lockstep", "overlap", "inexact")
    return os.getenv("VLLM_K3_UBATCH_PREFILL_OVERLAP", "0") == "1"


_COMM_STREAM: torch.cuda.Stream | None = None
_READY_BARRIER: threading.Barrier | None = None


def _comm_resources(device: torch.device):
    global _COMM_STREAM, _READY_BARRIER
    if _COMM_STREAM is None:
        _COMM_STREAM = torch.cuda.Stream(device=device)
        _READY_BARRIER = threading.Barrier(3)
    return _COMM_STREAM, _READY_BARRIER


def prime_workspaces() -> None:
    """Give the second half its own workspace slots before the manager locks.

    ``vllm.v1.worker.workspace`` keys its scratch buffers by
    ``(ubatch id, lane)``; the worker sizes the manager for one ubatch unless
    DBO is enabled, and after warm-up the manager is locked against growth.
    The split's second thread runs as ubatch 1, so without this it either
    indexes past the slot list or hits the growth lock on its first KDA/MLA
    workspace request. Each ubatch-1 lane gets a buffer as large as the
    corresponding ubatch-0 lane after warm-up (sized for the full chunk, so
    a half fits). No-op when the split is disabled or the slots exist.
    """
    if not ubatch_prefill_enabled():
        return
    from vllm.v1.worker.workspace import current_workspace_manager

    mgr = current_workspace_manager()
    lanes = mgr._num_lanes
    if mgr._num_ubatches < 2:
        mgr._num_ubatches = 2
        mgr._current_workspaces.extend([None] * lanes)
    for lane in range(lanes):
        base = mgr._current_workspaces[lane]
        slot = 1 * lanes + lane
        if base is None or mgr._current_workspaces[slot] is not None:
            continue
        mgr._current_workspaces[slot] = torch.empty(
            (base.numel(),), dtype=torch.uint8, device=base.device
        )


# FlashKDA advances its recurrent state in 16-token tiles and the state at a
# tile boundary is a bf16 value that the fp32 mamba slot carries losslessly,
# so a half boundary on a tile boundary hands the second half exactly the
# state the unsplit chunk holds there. An unaligned boundary would make the
# kernel re-tile the second half (different intra-tile terms).
SPLIT_ALIGNMENT = 16

# Granules per ring chunk that the B12X ring accepts before it falls back to
# the served contiguous mapping (b12x.comm.pcie.pcie_dma.MAX_PIECES): one
# granule is one copy piece of a reduce-scatter step.
_RING_MAX_GRANULES_PER_CHUNK = 8


def ring_granule_rows() -> int:
    """Rows per granule of the B12X ring's row-count-invariant chunk mapping.

    ``B12X_PCIE_RING_GRANULE_ROWS=g`` (``g > 0``) makes the ring assign row
    granules of ``g`` rows round-robin to its ``world`` reduction chunks, so
    an element's summation order depends only on its position inside a
    ``world * g`` row period instead of on the tensor's row count. 0 (the
    default) selects the served contiguous mapping, whose order is
    row-count-relative.
    """
    try:
        return max(0, int(os.getenv("B12X_PCIE_RING_GRANULE_ROWS", "0") or 0))
    except ValueError:
        return 0


def _tp_world_size() -> int:
    from vllm.distributed.parallel_state import (
        get_tensor_model_parallel_world_size,
    )

    return int(get_tensor_model_parallel_world_size())


def split_point(rows: int, block_rows: int = 0) -> int:
    """Row index where the second half starts, or 0 to run the chunk whole.

    ``block_rows`` is the row period over which the TP all-reduce's element
    ordering repeats (``world * granule`` for the ring's row-count-invariant
    mapping, 0 when the ring orders its reduction relative to the row count).

    With a period, both the chunk and the boundary must be whole multiples of
    it, because the halves reduce to the same bits as the unsplit chunk only
    when each half's rows start and end on a period boundary; a chunk that is
    not a multiple of the period runs whole. Without a period the boundary is
    rounded up to ``SPLIT_ALIGNMENT`` and the halves' all-reduces differ from
    the unsplit chunk's whatever the boundary is.

    The first half is the larger one so the second half fits the upper half
    of the retained AttnRes workspace (see KimiLinearModel).
    """
    if rows <= 0:
        return 0
    if block_rows <= 0:
        split = rows - rows // 2
        split = -(-split // SPLIT_ALIGNMENT) * SPLIT_ALIGNMENT
        return split if split < rows else 0
    blocks, tail = divmod(rows, block_rows)
    if tail or blocks < 2 or blocks > _RING_MAX_GRANULES_PER_CHUNK:
        return 0
    return (blocks - blocks // 2) * block_rows


def current_split_point(rows: int) -> int:
    """``split_point`` for the ring this process runs; 0 to run the chunk
    whole.

    A configured granule whose period cannot be determined (no
    tensor-parallel group) refuses the split rather than falling back to the
    row-count-relative boundary: the granule is configured precisely to make
    the halves reduce like the unsplit chunk.
    """
    granule = ring_granule_rows()
    if granule <= 0:
        return split_point(rows)
    try:
        world = _tp_world_size()
    except Exception:
        return 0
    if world <= 1:
        return split_point(rows)
    return split_point(rows, world * granule)


def eligible(input_batch: InputBatch) -> bool:
    rows = input_batch.num_tokens
    return (
        input_batch.num_reqs == 1
        and input_batch.num_draft_tokens == 0
        and rows == input_batch.num_tokens_after_padding
        and rows >= ubatch_prefill_min_tokens()
        and 0 < current_split_point(rows) < rows
        and bool(input_batch.is_prefilling_np[0])
    )


def _half_batch(
    runner,
    input_batch: InputBatch,
    row_start: int,
    row_end: int,
    computed_offset: int | None = None,
) -> InputBatch:
    """The rows [row_start, row_end) of a single-request batch as a batch of
    their own, positioned as if ``computed_offset`` earlier rows (default:
    ``row_start``) were already computed. Positions and slot mappings always
    follow the true rows; ``computed_offset=0`` builds MLA metadata whose
    chunked context excludes the first half (its keys are supplied in bf16
    by the attention layer itself, see KimiK3 mla.py split stash)."""
    rows = row_end - row_start
    if computed_offset is None:
        computed_offset = row_start
    device = input_batch.query_start_loc.device
    computed = input_batch.num_computed_tokens_np.copy()
    computed[0] += computed_offset
    computed_prefill = input_batch.num_computed_prefill_tokens_np.copy()
    computed_prefill[0] += computed_offset
    seq_len = int(computed[0] + rows)
    seq_lens = torch.full((input_batch.num_reqs_after_padding,), seq_len,
                          dtype=torch.int32, device=device)
    seq_lens_cpu = torch.full((input_batch.num_reqs_after_padding,), seq_len,
                              dtype=torch.int32)
    qsl_np = np.array([0, rows], dtype=np.int32)
    qsl = torch.from_numpy(qsl_np).to(device, non_blocking=True)
    dcp_local = None
    if input_batch.dcp_local_seq_lens is not None:
        dcp_local = torch.empty_like(input_batch.dcp_local_seq_lens)
        prepare_dcp_local_seq_lens(
            dcp_local,
            seq_lens,
            input_batch.num_reqs,
            runner.dcp_size,
            runner.dcp_rank,
            runner.cp_interleave,
        )
    return dataclasses.replace(
        input_batch,
        num_scheduled_tokens=np.array([rows], dtype=np.int32),
        max_query_len=rows,
        num_tokens=rows,
        num_tokens_after_padding=rows,
        query_start_loc=qsl,
        query_start_loc_np=qsl_np,
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens_cpu,
        max_seq_len_upper_bound=seq_len,
        dcp_local_seq_lens=dcp_local,
        num_computed_tokens_np=computed,
        num_computed_prefill_tokens_np=computed_prefill,
        input_ids=input_batch.input_ids[row_start:row_end],
        positions=input_batch.positions[row_start:row_end],
        is_padding=input_batch.is_padding[row_start:row_end],
        max_req_tokens=None,
    )


def run_split_prefill(
    runner,
    scheduler_output,
    input_batch: InputBatch,
    model_inputs: dict,
    *,
    cudagraph_runtime_mode,
    num_tokens_across_dp,
    batch_descriptor,
    skip_compiled: bool,
):
    """Run the chunk as two consecutive half forwards; returns the
    concatenated model output in the same form as one forward."""
    rows = input_batch.num_tokens
    split = current_split_point(rows)
    if not 0 < split < rows:
        raise ValueError(f"{rows} rows do not split (boundary {split})")
    halves = ((0, split), (split, rows))
    outputs = []
    prepared = []
    num_computed_gpu = runner.req_states.num_computed_tokens.gpu
    req_state_idx = int(input_batch.idx_mapping_np[0])
    # Exact split: the second half's MLA layers take the first half's keys
    # in bf16 from a per-layer stash and their chunked context excludes it
    # (see mla.py, `k3_split_exact`); `inexact` mode (mode file) keeps the
    # cache path for the first half instead.
    exact_mla = (
        os.getenv("VLLM_K3_UBATCH_PREFILL_EXACT", "1") == "1"
        and runtime_mode() != "inexact"
    )
    for start, end in halves:
        hb = _half_batch(runner, input_batch, start, end)
        block_tables, slot_mappings = runner.prepare_attn(hb)
        # The recurrent-state pre-copy reads the request's computed-token
        # count from the GPU request table; the second half starts where the
        # first ended, so hand it a shifted copy instead of the step-start
        # value.
        if start == 0:
            computed_for_half = num_computed_gpu
        else:
            computed_for_half = num_computed_gpu.clone()
            computed_for_half[req_state_idx] += start
        runner.model_state.preprocess_state(
            hb, block_tables, runner.kv_cache_config, computed_for_half,
        )
        slot_mappings_by_layer = build_slot_mappings_by_layer(
            slot_mappings, runner.kv_cache_config
        )
        attn_metadata = runner.model_state.prepare_attn(
            hb, cudagraph_runtime_mode, block_tables, slot_mappings,
            runner.attn_groups, runner.kv_cache_config, for_capture=False,
        )
        half_index = 0 if start == 0 else 1
        if exact_mla:
            # Both halves' MLA layers take the exact path: the first half
            # stashes its bf16 keys, the second consumes them. The half
            # index travels in the metadata because the sequential
            # (single-thread) mode has no ubatch context to read it from.
            for md in attn_metadata.values():
                if hasattr(md, "prefill") and hasattr(md, "num_decode_tokens"):
                    md.k3_split_exact = True
                    md.k3_split_half = half_index
        if start > 0 and exact_mla:
            # MLA layers of the second half: context = earlier chunks only;
            # the first half's keys come from the layer's bf16 stash.
            hb_mla = _half_batch(runner, input_batch, start, end, computed_offset=0)
            mla_metadata = runner.model_state.prepare_attn(
                hb_mla, cudagraph_runtime_mode, block_tables, slot_mappings,
                runner.attn_groups, runner.kv_cache_config, for_capture=False,
            )
            merged = dict(attn_metadata)
            for name, md in mla_metadata.items():
                if hasattr(md, "prefill") and hasattr(md, "num_decode_tokens"):
                    md.k3_split_exact = True
                    md.k3_split_half = half_index
                    if os.getenv("VLLM_K3_UBATCH_STASH_CHECK", "0") == "1":
                        # Diagnostics (mla.py _split_cache_reference): the
                        # half's cache-path metadata, context through the
                        # first half's cache entries.
                        try:
                            md.k3_split_cache_metadata = attn_metadata[name]
                        except Exception as exc:  # frozen/slots dataclass
                            from vllm.logger import init_logger

                            init_logger(__name__).warning(
                                "split check: cannot attach cache metadata: %s", exc
                            )
                    merged[name] = md
            attn_metadata = merged
        half_inputs = dict(model_inputs)
        half_inputs["input_ids"] = (
            None if model_inputs.get("input_ids") is None
            else model_inputs["input_ids"][start:end]
        )
        half_inputs["positions"] = model_inputs["positions"][start:end]
        if model_inputs.get("inputs_embeds") is not None:
            half_inputs["inputs_embeds"] = model_inputs["inputs_embeds"][start:end]
        half_inputs.update(runner.model_state.prepare_inputs(hb, runner.req_states))
        prepared.append((hb, attn_metadata, slot_mappings_by_layer, half_inputs))

    if not ubatch_prefill_overlap():
        for hb, attn_metadata, slot_mappings_by_layer, half_inputs in prepared:
            with set_forward_context(
                attn_metadata,
                runner.vllm_config,
                num_tokens=hb.num_tokens_after_padding,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                num_tokens_across_dp=num_tokens_across_dp,
                batch_descriptor=dataclasses.replace(
                    batch_descriptor, num_tokens=hb.num_tokens_after_padding
                ),
                slot_mapping=slot_mappings_by_layer,
                skip_compiled=skip_compiled,
                is_padding=hb.is_padding,
            ):
                outputs.append(runner.model(**half_inputs))
    else:
        outputs = _run_overlapped(runner, prepared, cudagraph_runtime_mode,
                                  batch_descriptor, skip_compiled)
    first = outputs[0]
    if isinstance(first, tuple):
        hidden = torch.cat([o[0] for o in outputs], dim=0)
        aux_lists = [o[1] for o in outputs]
        aux = [torch.cat(parts, dim=0) for parts in zip(*aux_lists)]
        return hidden, aux
    return torch.cat(outputs, dim=0)


def _run_overlapped(runner, prepared, cudagraph_runtime_mode, batch_descriptor,
                    skip_compiled):
    """Two threads, one per half, alternating at every TP all-reduce.

    Both halves compute on the current (compute) stream in CPU-issue order,
    which keeps every cross-half dependency (K/V and recurrent state written
    by the first half) satisfied by stream order; the all-reduces go to one
    shared comm stream (communication_op._ubatch_comm_region), so each
    half's collective overlaps the other half's compute.
    """
    from vllm.v1.worker.ubatching import make_ubatch_contexts

    device = prepared[0][0].query_start_loc.device
    compute_stream = torch.cuda.current_stream(device)
    comm_stream, ready_barrier = _comm_resources(device)
    forward_contexts = [
        create_forward_context(
            attn_metadata,
            runner.vllm_config,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=dataclasses.replace(
                batch_descriptor, num_tokens=hb.num_tokens_after_padding
            ),
            slot_mapping=slot_mappings_by_layer,
            skip_compiled=skip_compiled,
            is_padding=hb.is_padding,
        )
        for hb, attn_metadata, slot_mappings_by_layer, _ in prepared
    ]
    ctxs = make_ubatch_contexts(
        num_micro_batches=len(prepared),
        compute_stream=compute_stream,
        comm_stream=comm_stream,
        forward_contexts=forward_contexts,
        ready_barrier=ready_barrier,
    )
    results: list = []
    errors: list = []

    @torch.inference_mode()
    def _thread(ctx, half_inputs):
        try:
            # vLLM's thread-local current_stream() would otherwise create a
            # fresh CUDA stream for this thread on first use (a device
            # allocation that fails when the device is full); bind the
            # thread to the step's compute stream up front.
            torch.cuda.set_stream(compute_stream)
            with ctx:
                out = runner.model(**half_inputs)
            results.append((ctx.id, out))
        except BaseException as exc:  # surfaced after join
            import traceback

            from vllm.logger import init_logger

            init_logger(__name__).error(
                "split-prefill half %d failed: %s\n%s", ctx.id, exc, traceback.format_exc()
            )
            errors.append(exc)
            ctx.cpu_signal_event.set()

    with override_forward_context(None):
        threads = [
            threading.Thread(target=_thread, args=(ctx, item[3]))
            for ctx, item in zip(ctxs, prepared)
        ]
        for th in threads:
            th.start()
        ready_barrier.wait()
        ctxs[0].cpu_wait_event.set()
        try:
            _join_with_watchdog(threads, runner)
        finally:
            # A half that failed inside UBatchContext.__enter__ never ran
            # __exit__, leaving its thread registered; a later forward on the
            # main thread would then take the ubatch path of
            # dbo_current_ubatch_id() and fail with KeyError.
            from vllm.v1.worker import ubatching

            for th in threads:
                ubatching._THREAD_ID_TO_CONTEXT.pop(th.ident, None)
            for i in range(len(ubatching._CURRENT_CONTEXTS)):
                ubatching._CURRENT_CONTEXTS[i] = None
    if errors:
        raise errors[0]
    return [out for _, out in sorted(results, key=lambda r: r[0])]


def _join_with_watchdog(threads, runner) -> None:
    """Join the half threads; if they do not finish within
    ``VLLM_K3_UBATCH_WATCHDOG_S`` seconds (0 = off), log this rank's
    collective trace and the halves' Python stacks every few seconds so a
    cross-rank stall can be located from the worker logs."""
    import sys
    import traceback

    timeout = float(os.getenv("VLLM_K3_UBATCH_WATCHDOG_S", "0") or 0)
    if timeout <= 0:
        for th in threads:
            th.join()
        return
    from vllm.distributed.communication_op import _UbatchTrace
    from vllm.logger import init_logger

    logger = init_logger(__name__)
    deadline = time.time() + timeout
    reported = 0
    while any(th.is_alive() for th in threads):
        for th in threads:
            th.join(timeout=0.5)
        if time.time() > deadline and time.time() - reported > 5:
            reported = time.time()
            frames = sys._current_frames()
            stacks = []
            for th in threads:
                fr = frames.get(th.ident)
                if fr is not None:
                    stacks.append(
                        f"{th.name}: " + " <- ".join(
                            f"{f.name}:{f.lineno}" for f in traceback.extract_stack(fr)[-6:]
                        )
                    )
            logger.error(
                "split-prefill watchdog (rank %s): %s | %s",
                getattr(runner, "rank", "?"), _UbatchTrace.report(), " || ".join(stacks),
            )
    _UbatchTrace.reset()
