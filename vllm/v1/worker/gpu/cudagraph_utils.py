# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
import itertools
import os
from collections import defaultdict
from collections.abc import Callable, Iterable
from contextlib import ExitStack
from dataclasses import dataclass, replace
from itertools import groupby, product
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

import torch
import torch.nn as nn
from tqdm import tqdm

from vllm.compilation.breakable_cudagraph import (
    BreakableCUDAGraphWrapper,
    is_breakable_cudagraph_enabled,
)
from vllm.compilation.counter import compilation_counter
from vllm.compilation.cuda_graph import CUDAGraphStat, CUDAGraphWrapper
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.device_communicators.pynccl_allocator import set_graph_pool_id
from vllm.distributed.parallel_state import (
    get_pp_group,
    graph_capture,
    is_global_first_rank,
)
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import requires_raw_input_tokens
from vllm.model_executor.offloader.base import get_offloader
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import current_stream
from vllm.v1.hisparse.binding import release_hisparse_profiling_cache
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cp_utils import maybe_prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.ubatch_utils import check_ubatch_thresholds, get_num_ubatches
from vllm.v1.worker.utils import AttentionGroup, clear_layer_kv_caches
from vllm.v1.worker.workspace import collect_cuda_graph_capture_resources

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.pcp_manager import PCPManager
    from vllm.v1.worker.gpu.ubatch_utils import UBatchRunner

logger = init_logger(__name__)
_DENSE_VARLEN_DECODE_MAX_REQS = 2


def dense_varlen_decode_shapes(
    max_num_reqs: int, decode_query_len: int, max_capture_tokens: int
) -> tuple[tuple[int, int], ...]:
    """Return request and token counts of exact variable-length decode graphs."""
    return tuple(
        (num_reqs, num_reqs * query_len)
        for num_reqs in range(1, min(_DENSE_VARLEN_DECODE_MAX_REQS, max_num_reqs) + 1)
        for query_len in range(1, decode_query_len + 1)
        if num_reqs * query_len <= max_capture_tokens
    )


_DEBUG_GRAPH_MEMORY_ACCOUNTING = (
    os.getenv("VLLM_DEBUG_GRAPH_MEMORY_ACCOUNTING", "0") == "1"
)


def _graph_pool_snapshot_totals() -> tuple[int, int, int, int]:
    segments = [
        segment
        for segment in torch.cuda.memory_snapshot()
        if tuple(segment["segment_pool_id"]) != (0, 0)
    ]
    return (
        sum(segment["total_size"] for segment in segments),
        sum(segment["allocated_size"] for segment in segments),
        sum(segment["active_size"] for segment in segments),
        sum(
            block["size"]
            for segment in segments
            for block in segment["blocks"]
            if block["state"] == "inactive"
        ),
    )


def _log_graph_pool_growth(
    progress_bar_desc: str,
    desc: "BatchExecutionDescriptor",
    before: tuple[int, int, int, int] | None,
) -> None:
    if before is None:
        return
    after = _graph_pool_snapshot_totals()
    delta = tuple(end - start for start, end in zip(before, after))
    mib = 1 << 20
    logger.info(
        "[CG MEM] %s %s tokens=%d reqs=%s: "
        "pool=%+.1f MiB allocated=%+.1f MiB active=%+.1f MiB "
        "inactive=%+.1f MiB",
        progress_bar_desc,
        desc.cg_mode.name,
        desc.num_tokens,
        desc.num_reqs,
        *(value / mib for value in delta),
    )


def _memory_frame(frames: list[dict[str, Any]]) -> str:
    for frame in frames:
        filename = frame.get("filename", "")
        if "/vllm/" in filename and "/site-packages/" not in filename:
            relative_filename = filename.rsplit("/vllm/", 1)[-1]
            return f"{relative_filename}:{frame.get('line')}:{frame.get('name')}"
    if frames:
        frame = frames[0]
        return f"{frame.get('filename')}:{frame.get('line')}:{frame.get('name')}"
    return "<no Python frame>"


def _log_graph_pool_snapshot() -> None:
    snapshot = torch.cuda.memory._snapshot()
    segments = [
        segment
        for segment in snapshot["segments"]
        if tuple(segment["segment_pool_id"]) != (0, 0)
    ]
    mib = 1 << 20
    by_pool: defaultdict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for segment in segments:
        by_pool[tuple(segment["segment_pool_id"])].append(segment)
    for pool_id, pool_segments in sorted(by_pool.items()):
        total = sum(segment["total_size"] for segment in pool_segments)
        allocated = sum(segment["allocated_size"] for segment in pool_segments)
        active = sum(segment["active_size"] for segment in pool_segments)
        requested = sum(segment["requested_size"] for segment in pool_segments)
        inactive = sum(
            block["size"]
            for segment in pool_segments
            for block in segment["blocks"]
            if block["state"] == "inactive"
        )
        logger.info(
            "[CG MEM] pool=%s segments=%d total=%.1f MiB allocated=%.1f MiB "
            "active=%.1f MiB requested=%.1f MiB inactive=%.1f MiB",
            pool_id,
            len(pool_segments),
            total / mib,
            allocated / mib,
            active / mib,
            requested / mib,
            inactive / mib,
        )

    active_sites: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for segment in segments:
        for block in segment["blocks"]:
            if block["state"] == "inactive":
                continue
            site = _memory_frame(block.get("frames", []))
            totals = active_sites[site]
            totals[0] += block["size"]
            totals[1] += block["requested_size"]
            totals[2] += 1
    for site, (size, requested, count) in sorted(
        active_sites.items(), key=lambda item: item[1][0], reverse=True
    )[:30]:
        logger.info(
            "[CG MEM] active site=%s blocks=%d size=%.1f MiB requested=%.1f MiB",
            site,
            count,
            size / mib,
            requested / mib,
        )

    segment_sites: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
    for device_trace in snapshot["device_traces"]:
        for event in device_trace:
            if event["action"] != "segment_alloc":
                continue
            if tuple(event.get("pool_id", (0, 0))) == (0, 0):
                continue
            site = _memory_frame(event.get("frames", []))
            totals = segment_sites[site]
            totals[0] += event["size"]
            totals[1] += 1
    for site, (size, count) in sorted(
        segment_sites.items(), key=lambda item: item[1][0], reverse=True
    )[:30]:
        logger.info(
            "[CG MEM] segment site=%s segments=%d size=%.1f MiB",
            site,
            count,
            size / mib,
        )


def normalize_model_token_inputs(
    model: nn.Module,
    model_inputs: dict[str, Any],
) -> None:
    """Keep token-input arguments identical between graph capture and replay.

    Models that receive prepared embeddings normally omit ``input_ids``. Models
    declaring ``requires_raw_input_tokens`` are the exception and receive both.
    CUDA graph capture and ordinary execution must apply the same rule because
    breakable graphs require an invariant set of tensor arguments and addresses.
    """
    if model_inputs.get("inputs_embeds") is not None and not (
        requires_raw_input_tokens(model)
    ):
        model_inputs["input_ids"] = None


class AttentionState(NamedTuple):
    attn_metadata: dict[str, Any] | None
    slot_mappings: dict[str, torch.Tensor]


@dataclass(frozen=True)
class BatchExecutionDescriptor:
    """Describes the shape of the batch and CG mode to run; this is used to make shape
    matches between the capture and runtime."""

    cg_mode: CUDAGraphMode
    num_tokens: int
    num_reqs: int | None  # None means no request padding is needed (PIECEWISE graphs)
    uniform_token_count: int | None = None
    # Upper bound on per-request query length. Varlen decode graphs leave
    # uniform_token_count unset, so this is what keeps a prefill batch out of one.
    max_query_len: int | None = None
    num_active_loras: int = 0
    exact_num_tokens: bool = False
    # Number of microbatches the batch is split into (DBO). 1 means no splitting.
    num_ubatches: int = 1


def make_cudagraph_stats(
    batch_desc: BatchExecutionDescriptor, num_tokens: int
) -> CUDAGraphStat:
    return CUDAGraphStat(
        num_unpadded_tokens=num_tokens,
        num_padded_tokens=batch_desc.num_tokens,
        num_paddings=batch_desc.num_tokens - num_tokens,
        runtime_mode=str(batch_desc.cg_mode),
    )


class CreateForwardFn(Protocol):
    """Factory that prepares inputs (OUTSIDE the graph) and returns a
    forward_fn. Called with warmup=True for the warmup pass and warmup=False
    for the captured pass."""

    def __call__(
        self,
        desc: BatchExecutionDescriptor,
        warmup: bool,
    ) -> Callable[[CUDAGraphMode], None]: ...


def _is_compatible(
    desc: BatchExecutionDescriptor,
    num_reqs: int,
    num_tokens: int,
    uniform_token_count: int | None,
    num_active_loras: int,
    max_query_len: int | None,
    num_ubatches: int,
) -> bool:
    # desc.uniform_token_count=None (PIECEWISE) can handle any uniform_token_count
    # desc.num_reqs=None means no request padding needed (PIECEWISE)
    # desc.max_query_len=None means the graph does not constrain query length; a
    # caller that does not track max_query_len must not match one that does
    # A graph captured for N microbatches can only serve a batch split N ways.
    return (
        (not desc.exact_num_tokens or num_tokens == desc.num_tokens)
        and (
            desc.uniform_token_count is None
            or desc.uniform_token_count == uniform_token_count
        )
        and (
            desc.max_query_len is None
            or (max_query_len is not None and desc.max_query_len >= max_query_len)
        )
        and (desc.num_reqs is None or desc.num_reqs >= num_reqs)
        and desc.num_tokens >= num_tokens
        and desc.num_active_loras == num_active_loras
        and desc.num_ubatches == num_ubatches
    )


def has_compiled_submodule(model: nn.Module) -> bool:
    """Whether any submodule is an active @support_torch_compile module."""
    return any(
        isinstance(m, TorchCompileWithNoGuardsWrapper)
        and not getattr(m, "do_not_compile", True)
        for m in model.modules()
    )


class CudaGraphManager:
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_len: int,
        lora_capture_cases: list[int] | None = None,
        varlen_decode: bool = False,
        ubatch_runner: "UBatchRunner | None" = None,
        full_capture_request_sizes: frozenset[int] | None = None,
        specialize_full_decode: bool = False,
        single_request_prefill_tokens: int = 0,
    ):
        self.vllm_config = vllm_config
        self.device = device
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.compilation_config = vllm_config.compilation_config
        assert self.compilation_config is not None
        self.cudagraph_mode = cudagraph_mode
        self.decode_query_len = decode_query_len
        self.varlen_decode = varlen_decode
        # DBO supports FULL CUDA graphs only.
        self.ubatch_runner = ubatch_runner
        self.full_capture_request_sizes = full_capture_request_sizes
        self.specialize_full_decode = specialize_full_decode
        self.single_request_prefill_tokens = single_request_prefill_tokens

        self.dp_size = vllm_config.parallel_config.data_parallel_size
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.is_first_pp_rank = get_pp_group().is_first_rank
        self.is_last_pp_rank = get_pp_group().is_last_rank
        self.lora_capture_cases = lora_capture_cases or [0]
        # Precompute actual num_active_loras -> captured case mapping so that
        # dispatch() is a plain dict lookup instead of a per-call bisect.
        self._lora_dispatch_map, self._max_lora_case = self._build_lora_dispatch_map()

        self.graphs: dict[BatchExecutionDescriptor, torch.cuda.CUDAGraph] = {}
        self.graph_capture_resources: dict[BatchExecutionDescriptor, list[Any]] = {}
        self.pool = current_platform.get_global_graph_pool() if cudagraph_mode else None

        self._graphs_captured = False

        # Profiling hooks, set only by profile_cudagraph_memory() below: cap
        # FULL-mode capture at the N largest descriptors and record each
        # captured FULL graph's memory delta for extrapolation.
        self._max_full_descs_to_capture: int | None = None
        self._capture_mem_samples: list[int] | None = None

        self._candidates: dict[tuple[int, int], list[BatchExecutionDescriptor]] = {}
        self._capture_descs: dict[CUDAGraphMode, list[BatchExecutionDescriptor]] = {}

        # Breakable CUDA graph (PW CUDA graph without torch.compile)
        self.use_breakable_cg = (
            is_breakable_cudagraph_enabled()
            and self.cudagraph_mode.has_piecewise_cudagraphs()
        )
        self.breakable_cg_runner: BreakableCUDAGraphWrapper | None = None

        self._init_candidates()

    def _build_lora_dispatch_map(self) -> tuple[dict[int, int], int]:
        """Precompute actual num_active_loras -> effective captured case.

        Mirrors the num_tokens candidate expansion in ``_init_candidates``:
        every possible active-LoRA count is mapped ahead of time to the
        smallest captured case that can serve it, so ``dispatch`` is a plain
        dict lookup instead of a per-call bisect.
        """
        captured_with_lora = sorted(c for c in self.lora_capture_cases if c > 0)
        if not captured_with_lora:
            return {}, 0
        dispatch_map: dict[int, int] = {}
        case_idx = 0
        for n in range(1, captured_with_lora[-1] + 1):
            while captured_with_lora[case_idx] < n:
                case_idx += 1
            dispatch_map[n] = captured_with_lora[case_idx]
        return dispatch_map, captured_with_lora[-1]

    def _resolve_effective_loras(self, num_active_loras: int) -> int:
        """Map an actual active-LoRA count to its captured graph case."""
        if num_active_loras <= 0 or not self._lora_dispatch_map:
            return num_active_loras
        # Counts above the largest captured case clamp to it.
        return self._lora_dispatch_map.get(num_active_loras, self._max_lora_case)

    def _maybe_ubatch_twin(
        self, desc: BatchExecutionDescriptor
    ) -> BatchExecutionDescriptor | None:
        """Return a microbatched capture candidate when eligible.

        Uniform query lengths preserve the captured request split. Use the DP
        dispatch thresholds so all ranks generate the same candidates.
        """
        if self.ubatch_runner is None or desc.cg_mode != CUDAGraphMode.FULL:
            return None
        if desc.num_reqs is None:
            return None
        uniform_token_count, remainder = divmod(desc.num_tokens, desc.num_reqs)
        if remainder or desc.uniform_token_count not in (None, uniform_token_count):
            return None
        parallel_config = self.vllm_config.parallel_config
        num_ubatches = get_num_ubatches(parallel_config)
        if desc.num_tokens < num_ubatches:
            return None
        if not check_ubatch_thresholds(
            parallel_config, desc.num_tokens, uniform_decode=True
        ):
            return None
        return replace(
            desc, num_ubatches=num_ubatches, uniform_token_count=uniform_token_count
        )

    def _init_candidates(self) -> None:
        """Build priority-ordered candidate lists for each token count."""
        capture_sizes = self.compilation_config.cudagraph_capture_sizes
        if not (self.cudagraph_mode and capture_sizes):
            return

        capture_sizes = sorted(capture_sizes)
        max_decode_tokens = self.max_num_reqs * self.decode_query_len
        decode_mode = self.cudagraph_mode.decode_mode()
        mixed_mode = self.cudagraph_mode.mixed_mode()
        separate_decode_routine = self.cudagraph_mode.separate_routine() or (
            self.cudagraph_mode == CUDAGraphMode.FULL and self.specialize_full_decode
        )
        max_cg_capture_size = self.compilation_config.max_cudagraph_capture_size

        descs_by_mode: defaultdict[CUDAGraphMode, list[BatchExecutionDescriptor]] = (
            defaultdict(list)
        )

        # When using Dynamic SD, num_speculative_tokens is the max number of
        # draft tokens. The scheduler might use a smaller number so we need
        # to capture graphs for all possible values during decode.
        speculative_config = self.vllm_config.speculative_config
        if (
            speculative_config
            and speculative_config.uses_acceptance_length_adaptation()
            and self.decode_query_len >= self.vllm_config.num_speculative_tokens
        ):
            # decode_query_len = num_speculative_steps + num_new_sampled_tokens
            # _per_step. Recover num_new_sampled_tokens_per_step
            # from the values the manager already has.
            num_new_sampled_tokens_per_step = (
                self.decode_query_len - self.vllm_config.num_speculative_tokens
            )
            num_spec_per_batch_size = (
                speculative_config.num_speculative_tokens_per_batch_size
            )
            if num_spec_per_batch_size is None:
                reachable_depths = range(1, self.vllm_config.num_speculative_tokens + 1)
            else:
                caps = {entry[2] for entry in num_spec_per_batch_size}
                max_cap = min(max(caps), self.vllm_config.num_speculative_tokens)
                reachable_depths = range(0 if 0 in caps else 1, max_cap + 1)
            decode_query_lens = [
                depth + num_new_sampled_tokens_per_step for depth in reachable_depths
            ]
        elif (
            speculative_config
            and speculative_config.uses_batch_size_dynamic_speculative_decoding()
            and self.decode_query_len >= self.vllm_config.num_speculative_tokens
        ):
            num_spec_per_batch_size = (
                speculative_config.num_speculative_tokens_per_batch_size
            )
            assert num_spec_per_batch_size is not None
            num_new_sampled_tokens_per_step = (
                self.decode_query_len - self.vllm_config.num_speculative_tokens
            )
            dense_schedule = build_dynamic_sd_schedule_lookup(
                num_spec_per_batch_size,
                vllm_max_batch_size=self.max_num_reqs,
                vllm_num_speculative_tokens=self.vllm_config.num_speculative_tokens,
            )
            decode_query_lens = sorted(
                {
                    depth + num_new_sampled_tokens_per_step
                    for depth in dense_schedule[1:]
                }
            )
        else:
            decode_query_lens = [self.decode_query_len]

        capture_varlen_decode = (
            separate_decode_routine and bool(decode_mode) and self.varlen_decode
        )
        if capture_varlen_decode:
            # Keep exact low-concurrency FULL graphs for every possible
            # per-request speculative width. The ordinary token ladder below
            # remains as the padded fallback for larger request counts and
            # heterogeneous low-concurrency totals.
            for num_active_loras, (dense_num_reqs, num_tokens) in product(
                self.lora_capture_cases,
                dense_varlen_decode_shapes(
                    self.max_num_reqs, self.decode_query_len, max_cg_capture_size
                ),
            ):
                desc = BatchExecutionDescriptor(
                    cg_mode=decode_mode,
                    num_tokens=num_tokens,
                    num_reqs=dense_num_reqs,
                    max_query_len=self.decode_query_len,
                    num_active_loras=num_active_loras,
                )
                descs_by_mode[decode_mode].append(desc)
        for num_tokens, num_active_loras in product(
            capture_sizes, self.lora_capture_cases
        ):
            # Varlen decode graphs take any mix of 1..decode_query_len tokens per
            # request, worst case 1 token per request (or max_num_reqs)
            if capture_varlen_decode and num_tokens <= max_decode_tokens:
                desc = BatchExecutionDescriptor(
                    cg_mode=decode_mode,
                    num_tokens=num_tokens,
                    num_reqs=min(num_tokens, self.max_num_reqs),
                    max_query_len=self.decode_query_len,
                    num_active_loras=num_active_loras,
                )
                if desc not in descs_by_mode[decode_mode]:
                    descs_by_mode[decode_mode].append(desc)
            # Capture uniform decode specfifc graphs if required
            #  (i.e. separate decode routine)
            elif separate_decode_routine and decode_mode and not self.varlen_decode:
                for decode_query_len in decode_query_lens:
                    rounded_num_tokens = round_up(num_tokens, decode_query_len)
                    rounded_num_reqs = rounded_num_tokens // decode_query_len

                    if (
                        rounded_num_tokens > max_decode_tokens
                        or rounded_num_tokens > max_cg_capture_size
                        or rounded_num_reqs > self.max_num_reqs
                    ):
                        continue
                    if (
                        self.full_capture_request_sizes is not None
                        and rounded_num_reqs not in self.full_capture_request_sizes
                    ):
                        continue

                    desc = BatchExecutionDescriptor(
                        cg_mode=decode_mode,
                        num_tokens=rounded_num_tokens,
                        num_reqs=rounded_num_reqs,
                        uniform_token_count=decode_query_len,
                        num_active_loras=num_active_loras,
                    )

                    # avoid duplicate graphs
                    if desc not in descs_by_mode[decode_mode]:
                        descs_by_mode[decode_mode].append(desc)

                    ubatch_desc = self._maybe_ubatch_twin(desc)
                    if ubatch_desc is not None and (
                        ubatch_desc not in descs_by_mode[decode_mode]
                    ):
                        descs_by_mode[decode_mode].append(ubatch_desc)

            # recoverSSM cannot capture a dummy query wider than its workspace.
            if mixed_mode and (
                not self.vllm_config.cache_config.use_kda_recoverssm
                or num_tokens <= max_decode_tokens
            ):
                # for PIECEWISE graphs there is no limit on requests when replaying
                # i.e. no request padding is needed, so we leave it as None.
                # For breakable PW graphs, break-point kernels read the real batch
                # from the forward context; in-graph kernels handle the token padding
                # themselves from the padded slot_mapping (rows with slot == -1).
                num_reqs = None
                if mixed_mode == CUDAGraphMode.FULL:
                    num_reqs = min(num_tokens, self.max_num_reqs)
                desc = BatchExecutionDescriptor(
                    cg_mode=mixed_mode,
                    num_tokens=num_tokens,
                    num_reqs=num_reqs,
                    num_active_loras=num_active_loras,
                )
                descs_by_mode[mixed_mode].append(desc)

                ubatch_desc = self._maybe_ubatch_twin(desc)
                if ubatch_desc is not None:
                    descs_by_mode[mixed_mode].append(ubatch_desc)

        if self.single_request_prefill_tokens and self.use_breakable_cg:
            descs_by_mode[CUDAGraphMode.PIECEWISE].append(
                BatchExecutionDescriptor(
                    cg_mode=CUDAGraphMode.PIECEWISE,
                    num_tokens=self.single_request_prefill_tokens,
                    num_reqs=1,
                    max_query_len=self.single_request_prefill_tokens,
                    exact_num_tokens=True,
                )
            )

        for mode, descs in descs_by_mode.items():
            descs.sort(key=lambda d: d.num_tokens, reverse=True)
            self._capture_descs[mode] = descs

        for mode in (CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE):
            mode_descs = tuple(reversed(descs_by_mode.get(mode, [])))
            for num_active_loras in self.lora_capture_cases:
                lora_descs = [
                    d for d in mode_descs if d.num_active_loras == num_active_loras
                ]
                # Keep every graph large enough for a token count. A denser
                # descriptor at the nearest size can still reject the runtime
                # request count, in which case dispatch must continue to the
                # ordinary larger padded graph rather than fall to PIECEWISE.
                for num_tokens, group in groupby(lora_descs, lambda d: d.num_tokens):
                    matching = sorted(
                        group,
                        key=lambda d: (
                            d.uniform_token_count is None,
                            d.max_query_len is None,
                            d.num_reqs is None,
                            d.num_reqs or 0,
                        ),
                    )
                    for i in range(1, num_tokens + 1):
                        key = (i, num_active_loras)
                        self._candidates.setdefault(key, []).extend(matching)

    def needs_capture(self) -> bool:
        return len(self._capture_descs) > 0

    def _capture_stream(self, desc: BatchExecutionDescriptor) -> torch.cuda.Stream:
        """Capture on the stream used by the microbatch threads."""
        if desc.num_ubatches > 1:
            assert self.ubatch_runner is not None
            return self.ubatch_runner.capture_stream
        return current_stream()

    def planned_token_counts(self) -> list[int]:
        """Return model-row counts staged for decoder graph capture.

        Returns:
            Sorted, unique ``num_tokens`` values from the capture descriptors.
        """
        return sorted(
            {
                desc.num_tokens
                for descs in self._capture_descs.values()
                for desc in descs
            }
        )

    def captured_full_batch_shapes(self) -> list[tuple[int, int]]:
        """Sorted ``(num_tokens, num_reqs)`` shapes retained for FULL replay."""
        return sorted(
            {
                (desc.num_tokens, desc.num_reqs)
                for desc in self.graphs
                if desc.cg_mode == CUDAGraphMode.FULL
                and desc.num_reqs is not None
                and desc.num_active_loras == 0
            }
        )

    def reset_graphs(self) -> None:
        """Destroy FULL graph executables while retaining captured resources."""
        for graph in self.graphs.values():
            graph.reset()

    @torch.inference_mode()
    def capture(
        self,
        create_forward_fn: CreateForwardFn,
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        """Capture CUDA graphs.

        Args:
            create_forward_fn: Factory that prepares inputs (OUTSIDE graph) and
                returns a forward_fn. For FULL and breakable PIECEWISE modes,
                it is invoked once with warmup=True and again with warmup=False
                because attention backends may mutate or lazily initialize
                metadata during warmup.
        """
        with graph_capture(device=self.device), ExitStack() as stack:
            if self.ubatch_runner is not None:
                # Join parked threads on failure to avoid blocking later captures.
                stack.callback(self.ubatch_runner.abort_pending_run)
            # Capture in order: PIECEWISE first, then FULL. PIECEWISE has larger
            # activations so FULL activations should fit in already allocated
            # buffers in the graph pool.
            for mode in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]:
                if mode not in self._capture_descs:
                    continue

                descs = self._capture_descs[mode]
                if (
                    mode == CUDAGraphMode.FULL
                    and self._max_full_descs_to_capture is not None
                ):
                    # Profiling only: capture a sample of the largest FULL
                    # graphs; the total cost is extrapolated from their
                    # per-graph memory deltas.
                    descs = descs[: self._max_full_descs_to_capture]
                if is_global_first_rank():
                    descs = tqdm(descs, desc=f"{progress_bar_desc} ({mode.name})")
                for desc in descs:
                    # Prepare inputs and get forward function
                    forward_fn = create_forward_fn(desc, warmup=True)

                    # Warmup
                    forward_fn(CUDAGraphMode.NONE)
                    # A model forward may fork work onto auxiliary streams and
                    # join them with events queued on the compute stream.  CUDA
                    # graph capture must not begin while those warmup kernels
                    # are still executing, even though the queued event waits
                    # preserve normal stream ordering.
                    torch.accelerator.synchronize()

                    # Capture
                    logger.debug(
                        "CG Capture: mode=%s, batch_desc=%s",
                        desc.cg_mode.name,
                        desc,
                    )
                    pool_before = (
                        _graph_pool_snapshot_totals()
                        if _DEBUG_GRAPH_MEMORY_ACCOUNTING and is_global_first_rank()
                        else None
                    )
                    if (
                        desc.cg_mode == CUDAGraphMode.PIECEWISE
                        and not self.use_breakable_cg
                    ):
                        forward_fn(CUDAGraphMode.PIECEWISE)
                    else:
                        # Capture with fresh attention state.
                        forward_fn = create_forward_fn(desc, warmup=False)
                        if desc.cg_mode == CUDAGraphMode.PIECEWISE:
                            forward_fn(CUDAGraphMode.PIECEWISE)
                            _log_graph_pool_growth(progress_bar_desc, desc, pool_before)
                            continue
                        assert desc not in self.graphs, (
                            f"Graph already captured for {desc}"
                        )
                        graph = torch.cuda.CUDAGraph()
                        # Sync offloader's copy stream before capture.
                        # Ensure any pre-capture prefetches from offloader are complete.
                        get_offloader().sync_prev_onload()
                        if self.pool is not None:
                            set_graph_pool_id(self.pool)
                        else:
                            set_graph_pool_id(current_platform.graph_pool_handle())
                        if self._capture_mem_samples is not None:
                            torch.accelerator.synchronize()
                            free_before = torch.accelerator.get_memory_info()[0]
                        with (
                            collect_cuda_graph_capture_resources() as resources,
                            torch.cuda.graph(
                                graph, self.pool, stream=self._capture_stream(desc)
                            ),
                        ):
                            forward_fn(CUDAGraphMode.NONE)
                            # Join the offloader copy stream because the last layer
                            # can leave a prefetch pending at capture end.
                            get_offloader().join_after_forward()
                        if self._capture_mem_samples is not None:
                            torch.accelerator.synchronize()
                            free_after = torch.accelerator.get_memory_info()[0]
                            self._capture_mem_samples.append(free_before - free_after)
                        self.graphs[desc] = graph
                        self.graph_capture_resources[desc] = resources
                        compilation_counter.num_cudagraph_captured += 1
                    _log_graph_pool_growth(progress_bar_desc, desc, pool_before)
        self._graphs_captured = True

    def captured_token_counts(self) -> list[int]:
        """Sorted token counts with a captured graph, ignoring LoRA variants."""
        return sorted(
            {desc.num_tokens for desc in self.graphs if desc.num_active_loras == 0}
        )

    def dispatch(
        self,
        num_reqs: int,
        num_tokens: int,
        uniform_token_count: int | None,
        num_active_loras: int,
        max_query_len: int | None = None,
        num_ubatches: int = 1,
    ) -> BatchExecutionDescriptor:
        """Find matching cudagraph descriptor from priority-ordered candidates."""

        effective_loras = self._resolve_effective_loras(num_active_loras)
        key = (num_tokens, effective_loras)
        if self._graphs_captured and num_tokens > 0 and key in self._candidates:
            for desc in self._candidates[key]:
                if _is_compatible(
                    desc,
                    num_reqs,
                    num_tokens,
                    uniform_token_count,
                    effective_loras,
                    max_query_len,
                    num_ubatches,
                ):
                    return desc
        return BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            num_active_loras=effective_loras,
            num_ubatches=num_ubatches,
        )

    def run_fullgraph(self, desc: BatchExecutionDescriptor):
        """Replay a captured FULL cudagraph."""
        assert desc.cg_mode == CUDAGraphMode.FULL, (
            f"Expected FULL mode, got {desc.cg_mode}"
        )
        assert desc in self.graphs, f"No cudagraph for {desc}"
        # Sync offloader before replay - needed when transitioning from
        # eager/piecewise to full cudagraph (e.g., prefill → decode).
        # The previous eager iteration's start_prefetch may have queued
        # H2D copies on copy_stream that the graph's captured events
        # cannot see. Without this, replay could overwrite static buffers
        # while those copies are still in flight.
        get_offloader().sync_prev_onload()
        self.graphs[desc].replay()

    def init_breakable_cg_runner(self, model: nn.Module) -> None:
        if self.breakable_cg_runner is None:
            self.breakable_cg_runner = BreakableCUDAGraphWrapper(
                model, self.vllm_config
            )

    def run_pw_graph(self, model: nn.Module, model_inputs: dict[str, Any]) -> Any:
        if not self.use_breakable_cg:
            # Default: Use torch-compiled piecewise cudagraph.
            return model(**model_inputs)
        assert self.breakable_cg_runner is not None
        return self.breakable_cg_runner(**model_inputs)


class ModelCudaGraphManager(CudaGraphManager):
    """CudaGraphManager with model-specific capture and hidden state management."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_len: int,
        lora_capture_cases: list[int] | None = None,
        varlen_decode: bool = False,
        ubatch_runner: "UBatchRunner | None" = None,
        specialize_full_decode: bool = False,
        single_request_prefill_tokens: int = 0,
    ):
        super().__init__(
            vllm_config,
            device,
            cudagraph_mode,
            decode_query_len,
            lora_capture_cases=lora_capture_cases,
            varlen_decode=varlen_decode,
            ubatch_runner=ubatch_runner,
            specialize_full_decode=specialize_full_decode,
            single_request_prefill_tokens=single_request_prefill_tokens,
        )
        self.hidden_states: torch.Tensor | None = None
        self.aux_hidden_states: list[torch.Tensor] = []
        self.use_aux_hidden_state_outputs = False
        self.intermediate_tensors: IntermediateTensors | None = None

    def capture(
        self,
        model: nn.Module,
        model_state: ModelState,
        input_buffers: InputBuffers,
        intermediate_tensors: IntermediateTensors | None,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        pcp_manager: "PCPManager | None" = None,
        has_lora: bool = False,
        use_aux_hidden_state_outputs: bool = False,
        lora_capture_hook: Callable[[int, int, int], None] | None = None,
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        """Capture CUDA graphs for model forward pass."""
        self.use_aux_hidden_state_outputs = use_aux_hidden_state_outputs
        if self.use_breakable_cg:
            self.init_breakable_cg_runner(model)

        if self.cudagraph_mode.has_piecewise_cudagraphs() and not (
            self.use_breakable_cg or has_compiled_submodule(model)
        ):
            raise RuntimeError(
                f"{type(model).__name__}: piecewise CUDA graphs "
                f"(cudagraph_mode={self.cudagraph_mode.name}) unavailable, "
                "model is not torch-compiled and breakable CUDA graph is off. "
                "Set VLLM_USE_BREAKABLE_CUDAGRAPH=1 or cudagraph_mode=NONE/FULL."
            )

        def store_capture_output(num_tokens: int, model_output: Any) -> None:
            """Copy outputs to persistent buffers, allocating on first use."""
            if self.is_last_pp_rank:
                # Last PP rank (common case).
                if self.use_aux_hidden_state_outputs:
                    hidden_states, aux_hidden_states = model_output
                else:
                    hidden_states = model_output
                    aux_hidden_states = []
                if self.hidden_states is None:
                    self.hidden_states = torch.empty_like(hidden_states)
                self.hidden_states[:num_tokens] = hidden_states
                if self.use_aux_hidden_state_outputs and not self.aux_hidden_states:
                    self.aux_hidden_states = [
                        torch.empty_like(x) for x in aux_hidden_states
                    ]
                for i, aux in enumerate(aux_hidden_states):
                    self.aux_hidden_states[i][:num_tokens] = aux
            else:
                # Non-last PP rank.
                assert isinstance(model_output, IntermediateTensors)
                intermediate_tensors = model_output
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = IntermediateTensors.empty_like(
                        intermediate_tensors
                    )
                for k, v in intermediate_tensors.tensors.items():
                    self.intermediate_tensors[k][:num_tokens] = v

        def create_forward_fn(
            desc: BatchExecutionDescriptor,
            warmup: bool,
        ) -> Callable[[CUDAGraphMode], None]:
            num_tokens = desc.num_tokens
            num_reqs = desc.num_reqs or min(num_tokens, self.max_num_reqs)

            # Set LoRA state before capture so kernels see correct adapters.
            if lora_capture_hook is not None:
                lora_capture_hook(desc.num_active_loras, num_reqs, num_tokens)

            num_tokens_across_dp = (
                torch.full((self.dp_size,), num_tokens, dtype=torch.int32, device="cpu")
                if self.dp_size > 1
                else None
            )

            model_inputs = {
                "input_ids": input_buffers.input_ids[:num_tokens],
                "positions": input_buffers.positions[:num_tokens],
                **model_state.prepare_dummy_inputs(num_reqs, num_tokens),
            }
            normalize_model_token_inputs(model, model_inputs)
            if not self.is_first_pp_rank:
                # Update for non-first PP ranks.
                model_inputs["input_ids"] = None
                model_inputs["inputs_embeds"] = None
                assert intermediate_tensors is not None
                model_inputs["intermediate_tensors"] = intermediate_tensors[:num_tokens]

            if desc.num_ubatches > 1:
                # Prepare and park threads before capture; finish runs inside it.
                assert self.ubatch_runner is not None
                ubatch_state = self.ubatch_runner.prepare(
                    InputBatch.make_dummy(num_reqs, num_tokens, input_buffers),
                    block_tables.get_dummy_block_tables(num_reqs),
                    block_tables.get_dummy_slot_mappings(num_tokens),
                    cg_mode=CUDAGraphMode.FULL,
                    for_capture=True,
                )
                # Capture with dummy rows marked as padding.
                input_buffers.is_padding.fill_(True)
                finish = self.ubatch_runner.begin_capturable_run(
                    model, model_inputs, ubatch_state, for_capture=True
                )

                def ubatch_forward_fn(cg_mode: CUDAGraphMode) -> None:
                    assert cg_mode != CUDAGraphMode.PIECEWISE, (
                        "DBO does not support PIECEWISE cudagraphs"
                    )
                    store_capture_output(num_tokens, finish())

                return ubatch_forward_fn

            attn_metadata, slot_mappings = prepare_inputs_to_capture(
                num_reqs,
                num_tokens,
                model_state,
                input_buffers,
                block_tables,
                attn_groups,
                kv_cache_config,
                full_cudagraph=desc.cg_mode == CUDAGraphMode.FULL,
                uniform_decode_graph=desc.uniform_token_count is not None,
                max_query_len=desc.max_query_len,
                pcp_manager=pcp_manager,
            )
            model_state.finalize_cudagraph_inputs(model_inputs, desc.cg_mode)

            # Capture with dummy rows marked as padding.
            input_buffers.is_padding.fill_(True)

            def forward_fn(cg_mode: CUDAGraphMode) -> None:
                batch_descriptor = None
                if (
                    cg_mode == CUDAGraphMode.PIECEWISE
                    or desc.cg_mode == CUDAGraphMode.FULL
                ):
                    batch_descriptor = BatchDescriptor(
                        num_tokens=num_tokens,
                        num_reqs=desc.num_reqs,
                        uniform=desc.uniform_token_count is not None,
                        has_lora=has_lora,
                        num_active_loras=desc.num_active_loras,
                    )
                with set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens,
                    cudagraph_runtime_mode=cg_mode,
                    num_tokens_across_dp=num_tokens_across_dp,
                    slot_mapping=slot_mappings,
                    batch_descriptor=batch_descriptor,
                    is_padding=input_buffers.is_padding[:num_tokens],
                ):
                    if cg_mode == CUDAGraphMode.PIECEWISE:
                        # PIECEWISE graph (compiled PW or breakable, chosen inside
                        # run_pw_graph).
                        model_output = self.run_pw_graph(model, model_inputs)
                    else:
                        model_output = model(**model_inputs)

                if cg_mode == CUDAGraphMode.PIECEWISE:
                    # PW CUDA graph (compiled or breakable) internally handles the
                    # model outputs. No need to keep track of the hidden states.
                    return None

                store_capture_output(num_tokens, model_output)

            return forward_fn

        super().capture(create_forward_fn, progress_bar_desc)

    def run_fullgraph(
        self, desc: BatchExecutionDescriptor
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]] | IntermediateTensors:
        """Replay a captured FULL cudagraph and return hidden states."""
        super().run_fullgraph(desc)
        if not self.is_last_pp_rank:
            assert self.intermediate_tensors is not None
            return self.intermediate_tensors[: desc.num_tokens]

        assert self.hidden_states is not None
        hidden_states = self.hidden_states[: desc.num_tokens]
        if not self.use_aux_hidden_state_outputs:
            return hidden_states
        return hidden_states, [x[: desc.num_tokens] for x in self.aux_hidden_states]


def prepare_inputs_to_capture(
    num_reqs: int,
    num_tokens: int,
    model_state: ModelState,
    input_buffers: InputBuffers,
    block_tables: BlockTables,
    attn_groups: list[list[AttentionGroup]],
    kv_cache_config: KVCacheConfig,
    full_cudagraph: bool,
    max_query_len: int | None = None,
    pcp_manager: "PCPManager | None" = None,
    uniform_decode_graph: bool = False,
) -> AttentionState:
    input_batch = InputBatch.make_dummy(
        num_reqs, num_tokens, input_buffers, max_query_len=max_query_len
    )
    if pcp_manager is not None:
        input_batch = pcp_manager.prepare_inputs_to_capture(input_batch)

    input_batch.uniform_decode_graph = full_cudagraph and uniform_decode_graph

    block_table_provider = pcp_manager or block_tables
    input_block_tables = block_table_provider.get_dummy_block_tables(num_reqs)
    slot_mappings = block_table_provider.get_dummy_slot_mappings(num_tokens)
    slot_mappings_by_layer = build_slot_mappings_by_layer(
        slot_mappings, kv_cache_config
    )

    input_batch.dcp_local_seq_lens = maybe_prepare_dcp_local_seq_lens(
        input_buffers.dcp_local_seq_lens,
        input_batch.seq_lens,
        input_batch.num_reqs,
        block_tables.cp_size,
        block_tables.cp_rank,
        block_tables.cp_interleave,
        num_reqs_padded=input_batch.num_reqs_after_padding,
    )

    # NOTE(woosuk): Attention metadata is required not just by standard attention
    # kernels, but also by specialized attention-like operations (e.g., Inkling's sconv,
    # DSV4 compressor), which maintain their own states and require special metadata
    # such as block tables.
    # During CUDA graph capture:
    # - For FULL CUDA graphs: We set for_capture=True so that both attention and
    #   attention-like ops produce capturable metadata compatible with CUDA graphs.
    # - For PIECEWISE CUDA graphs: We still build attention metadata, but set
    #   for_capture=False. This is because:
    #     * Attention-like ops (such as sconv or DSV4 compressor) may not be used as
    #       breakpoints in PIECEWISE CUDA graphs, so we must generate their attention
    #       metadata so they can execute and be captured during graph capture.
    #     * Standard attention ops that are treated as breakpoints will be executed
    #       eagerly at capture time (not included in the graph itself), and for these,
    #       setting for_capture=False is essential. Some attention backends
    #       (like linear attention) cannot generate capturable metadata for prefill,
    #       so for_capture=False ensures they execute without issue.
    #     * We assume that attention-like operations intended for capture will still
    #       produce capturable metadata, even when for_capture=False. While this
    #       assumption is brittle, it currently works in practice.
    # In summary: We always generate attention metadata for both FULL and PIECEWISE
    # CUDA graphs, setting for_capture=True for FULL graphs, and for_capture=False
    # for PIECEWISE graphs, to ensure correct execution and capture.
    attn_metadata = model_state.prepare_attn(
        input_batch,
        CUDAGraphMode.NONE,
        input_block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture=full_cudagraph,
    )
    return AttentionState(attn_metadata, slot_mappings_by_layer)


# ---------------------------------------------------------------------------
# CUDA graph memory profiling
# ---------------------------------------------------------------------------

# Number of FULL graphs captured during profiling; the total FULL capture
# cost is extrapolated from this sample to avoid a second full capture.
_FULL_GRAPH_PROFILING_SAMPLES = 2
# Floor for the extrapolated per-graph cost (driver overhead per graph).
_MIN_PER_GRAPH_BYTES = 1 << 20


def _profiling_cudagraph_managers(runner: "GPUModelRunner") -> list[CudaGraphManager]:
    managers: list[CudaGraphManager] = []
    if isinstance(runner.cudagraph_manager, CudaGraphManager):
        managers.append(runner.cudagraph_manager)
    speculator = runner.speculator
    if speculator is not None:
        for candidate in vars(speculator).values():
            if isinstance(candidate, CudaGraphManager) and all(
                candidate is not manager for manager in managers
            ):
                managers.append(candidate)
    return managers


@torch.inference_mode()
def profile_cudagraph_memory(
    runner: "GPUModelRunner",
    prepare_profile_state: Callable[[], None] | None = None,
) -> int:
    """Estimate the GPU memory needed for CUDA graph capture.

    Called during memory profiling, *before* the real KV cache is allocated,
    so that ``Worker.determine_available_memory`` can reserve headroom for
    graph capture. Bootstraps a minimal KV cache, runs ``capture_model()``
    once, then releases everything so the real init/capture path starts clean.

    FULL graphs bake in KV cache pointers, so only the largest few are
    captured (into a throwaway pool) and their total cost is extrapolated.
    PIECEWISE, encoder and speculator graphs are measured in full. All
    profiling captures are discarded afterwards: replaying graphs recorded
    against the throwaway profiling state is unsafe (e.g. inductor graph
    partition reclaims the storages of earlier cudagraph recordings once the
    real capture records new ones, leading to use-after-free crashes).
    """
    runner.cudagraph_native_memory_profile = None
    if runner.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
        return 0

    gc.collect()
    torch.accelerator.empty_cache()

    # Run the whole profiling phase against a throwaway CUDA graph pool by
    # pointing the global graph pool singleton at it: objects that bind the
    # pool lazily during profiling (speculator cudagraph managers, breakable
    # runners created mid-capture) then land on the throwaway pool too. Pools
    # bound before profiling (piecewise wrappers) are swapped explicitly in
    # the inner block. Profiling graphs captured into the persistent global
    # pool and then discarded would drop its use_count to 0, tripping the c10
    # allocator's create_or_incref_pool assert when the real capture reuses
    # that pool ("use_count > 0 INTERNAL ASSERT FAILED").
    platform_cls = type(current_platform)
    saved_global_pool = current_platform.get_global_graph_pool()
    throwaway_pool = current_platform.graph_pool_handle()
    platform_cls._global_graph_pool = throwaway_pool

    try:
        try:
            with set_current_vllm_config(runner.vllm_config):
                _init_minimal_kv_cache_for_profiling(runner)
                if prepare_profile_state is not None:
                    prepare_profile_state()
        except BaseException:
            _teardown_profiling_state(runner)
            raise

        manager = runner.cudagraph_manager
        assert manager is not None

        # Don't count profiling captures; the real capture_model() runs later.
        saved_num_cudagraph_captured = compilation_counter.num_cudagraph_captured
        saved_capture_triggers = compilation_counter.num_gpu_runner_capture_triggers
        all_wrappers: list[Any] = []
        original_pools: dict[int, Any] = {}
        original_manager_pools: dict[int, Any] = {}
        speculator = getattr(runner, "speculator", None)
        spec_manager_names: list[str] = []
        try:
            if not manager.needs_capture():
                return 0
            for graph_manager in _profiling_cudagraph_managers(runner):
                original_manager_pools[id(graph_manager)] = graph_manager.pool
                graph_manager.pool = throwaway_pool
            del graph_manager
            if manager.use_breakable_cg:
                # Create the breakable runner before the wrapper pool swap so
                # its pool is covered as well.
                manager.init_breakable_cg_runner(runner.model)
            all_wrappers = list(CUDAGraphWrapper._all_instances) + list(
                BreakableCUDAGraphWrapper._all_instances
            )
            for wrapper in all_wrappers:
                original_pools[id(wrapper)] = wrapper.graph_pool
                wrapper.graph_pool = throwaway_pool
            if speculator is not None:
                spec_manager_names = [
                    name
                    for name, value in vars(speculator).items()
                    if isinstance(value, CudaGraphManager)
                ]
            manager._max_full_descs_to_capture = (
                None
                if _DEBUG_GRAPH_MEMORY_ACCOUNTING
                else _FULL_GRAPH_PROFILING_SAMPLES
            )
            mem_samples: list[int] = []
            manager._capture_mem_samples = mem_samples

            native_before_capture = _non_torch_memory_bytes()
            measured = int(runner.capture_model(profile_only=True))
            native_after_capture = _non_torch_memory_bytes()
            runner.cudagraph_native_memory_profile = (
                native_before_capture,
                native_after_capture,
                measured,
            )

            # The measured delta covers PIECEWISE, encoder and speculator graphs
            # plus the sampled FULL graphs; swap the sampled FULL cost for the
            # extrapolated total. FULL and PIECEWISE share one pool here just as
            # they share the global pool at runtime, so the overlap is not
            # double-counted.
            full_descs = manager._capture_descs.get(CUDAGraphMode.FULL, [])
            full_estimate = _extrapolate_full_graph_memory(mem_samples, full_descs)
            return max(measured - sum(mem_samples) + full_estimate, 0)
        finally:
            compilation_counter.num_cudagraph_captured = saved_num_cudagraph_captured
            compilation_counter.num_gpu_runner_capture_triggers = saved_capture_triggers
            graph_managers = _profiling_cudagraph_managers(runner)
            torch.accelerator.synchronize()
            CUDAGraphWrapper.reset_all_graphs()
            BreakableCUDAGraphWrapper.reset_all_graphs()
            for graph_manager in graph_managers:
                graph_manager.reset_graphs()
            torch.accelerator.synchronize()
            CUDAGraphWrapper.clear_all_graphs()
            BreakableCUDAGraphWrapper.clear_all_graphs()
            for graph_manager in graph_managers:
                graph_manager.graphs.clear()
                graph_manager.graph_capture_resources.clear()
                graph_manager._graphs_captured = False
                graph_manager.pool = original_manager_pools.get(
                    id(graph_manager), saved_global_pool
                )
            for wrapper in list(CUDAGraphWrapper._all_instances) + list(
                BreakableCUDAGraphWrapper._all_instances
            ):
                wrapper.graph_pool = original_pools.get(id(wrapper), saved_global_pool)
            if graph_managers:
                del graph_manager
            del graph_managers

            # Drop the speculator's cudagraph managers; the real
            # initialize_kv_cache re-creates them. Their profiling graphs
            # release the throwaway pool here rather than after the real init.
            for name in spec_manager_names:
                setattr(speculator, name, None)
            # Drop local references before teardown detaches the runner's
            # manager and flushes the allocator.
            del manager
            _teardown_profiling_state(runner)
    finally:
        platform_cls._global_graph_pool = saved_global_pool


def _non_torch_memory_bytes() -> int:
    """Measure device memory not owned by the Torch caching allocator."""
    torch.accelerator.synchronize()
    free, total = torch.accelerator.get_memory_info()
    return total - free - torch.accelerator.memory_reserved()


def _extrapolate_full_graph_memory(
    mem_samples: list[int],
    graph_descs: list[BatchExecutionDescriptor],
) -> int:
    """Project unsampled FULL graph costs from their descriptor token counts."""
    if not mem_samples or not graph_descs:
        return 0
    assert len(mem_samples) <= len(graph_descs)

    estimate = mem_samples[0] + sum(
        max(sample, _MIN_PER_GRAPH_BYTES) for sample in mem_samples[1:]
    )
    if len(mem_samples) == len(graph_descs) or len(mem_samples) < 2:
        return estimate

    reference_desc = graph_descs[len(mem_samples) - 1]
    reference_cost = max(mem_samples[-1], _MIN_PER_GRAPH_BYTES)
    reference_tokens = reference_desc.num_tokens
    for desc in graph_descs[len(mem_samples) :]:
        scaled_cost = (
            reference_cost * desc.num_tokens + reference_tokens - 1
        ) // reference_tokens
        estimate += max(scaled_cost, _MIN_PER_GRAPH_BYTES)
    return estimate


def _init_minimal_kv_cache_for_profiling(
    runner: "GPUModelRunner", *, num_blocks: int | None = None
) -> None:
    """Allocate the smallest KV cache required by a profiling workload.

    Graph capture needs one block per padded sequence. A caller that executes
    one eager request can request one block explicitly, avoiding temporary KV
    storage that would otherwise be included in activation headroom.
    """
    from vllm.v1.core.kv_cache_utils import (
        get_kv_cache_config_from_groups,
        get_kv_cache_groups,
    )

    kv_cache_spec = runner.get_kv_cache_spec()
    kv_cache_groups = get_kv_cache_groups(runner.vllm_config, kv_cache_spec)
    # At least one block per sequence is required to capture the graphs.
    if num_blocks is None:
        num_blocks = (
            min(
                runner.max_num_reqs,
                runner.compilation_config.max_cudagraph_capture_size,
            )
            or 1
        )
    if num_blocks < 1:
        raise ValueError("Profiling KV cache requires at least one block")
    saved_override = runner.cache_config.num_gpu_blocks_override
    runner.cache_config.num_gpu_blocks_override = num_blocks
    try:
        minimal_config = get_kv_cache_config_from_groups(
            runner.vllm_config, kv_cache_groups, available_memory=0
        )
    finally:
        runner.cache_config.num_gpu_blocks_override = saved_override

    runner.initialize_kv_cache(minimal_config, is_profiling=True)
    runner.cache_config.num_gpu_blocks = minimal_config.num_blocks


def _teardown_profiling_state(runner: "GPUModelRunner") -> None:
    """Release the profiling KV cache and captured graphs while keeping model
    weights, so the real ``initialize_kv_cache`` starts from a clean slate."""
    ubatch_runner = getattr(runner, "ubatch_runner", None)
    if ubatch_runner is not None:
        ubatch_runner.abort_pending_run()
        runner.ubatch_runner = None
    del ubatch_runner
    torch.accelerator.synchronize()
    if hasattr(runner.model_state, "_mamba_ctx"):
        runner.model_state._mamba_ctx = None
    # Invalidate the align-mode Mamba group metadata cached from the
    # profiling KVCacheConfig: the real (e.g. PP-projected) config may
    # place Mamba layers into a different group layout, so it must be
    # re-derived from the real config.
    if hasattr(runner.model_state, "_mamba_group_ids"):
        runner.model_state._mamba_group_ids = []
    if hasattr(runner.model_state, "_mamba_spec"):
        runner.model_state._mamba_spec = None
    if hasattr(runner, "kv_caches"):
        runner.kv_caches.clear()
    if hasattr(runner, "attn_groups"):
        runner.attn_groups.clear()
    if hasattr(runner, "kv_cache_config"):
        del runner.kv_cache_config
    if hasattr(runner, "block_tables"):
        del runner.block_tables
    runner.pcp_manager = None
    runner.adaptive_verification = None
    # Dropping the manager releases the profiling graphs and throwaway pool.
    runner.cudagraph_manager = None
    # Release encoder graphs captured during profiling; the real
    # capture_model() re-captures them.
    if runner.model_state.supports_mm_inputs:
        runner.model_state.encoder_runner.clear()
    # Detach profiling KV tensors and every layer-derived cache view/binding.
    layers: Iterable[Any] = runner.compilation_config.static_forward_context.values()
    if (model := getattr(runner, "model", None)) is not None:
        layers = itertools.chain(layers, model.modules())
    clear_layer_kv_caches(layers)
    reset_model_state = getattr(runner.model_state, "reset_kv_cache_state", None)
    if callable(reset_model_state):
        reset_model_state()
    speculator = getattr(runner, "speculator", None)
    if speculator is not None:
        speculator.reset_attn()

    release_hisparse_profiling_cache(runner.compilation_config.static_forward_context)
    runner.cache_config.num_gpu_blocks = None
    runner.maybe_remove_all_loras(runner.lora_config)
    gc.collect()
    torch.accelerator.synchronize()
    torch.accelerator.empty_cache()
