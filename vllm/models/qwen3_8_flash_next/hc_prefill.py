# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 token-row ownership for Qwen's pure eager prefills."""

from dataclasses import dataclass
from typing import Any

import torch

from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger

logger = init_logger(__name__)


def configure(model, config, mode: str) -> None:
    """Agree on ownership admission before compiling any model forward."""
    if mode not in ("off", "control", "shard"):
        raise ValueError("Qwen HC prefill mode must be off, control, or shard")
    model.hc_prefill_mode = mode
    model.hc_prefill_reports = 0
    if not torch.distributed.is_initialized():
        if mode == "off":
            return
        raise RuntimeError("Qwen HC ownership requires initialized TP ranks")
    group = get_tp_group()
    error = None
    parallel = config.parallel_config
    if mode != "off" and (
        group.world_size != 4
        or parallel.pipeline_parallel_size != 1
        or parallel.data_parallel_size != 1
        or parallel.decode_context_parallel_size != 1
        or parallel.prefill_context_parallel_size != 1
        or parallel.enable_expert_parallel
        or parallel.enable_eplb
        or parallel.use_sequence_parallel_moe
        or parallel.enable_dbo
        or config.model_config.dtype != torch.bfloat16
    ):
        error = "Qwen HC ownership requires BF16 TP4/PP1/DP1/DCP1/PCP1"
    votes = [None] * group.world_size
    torch.distributed.all_gather_object(votes, (mode, error), group=group.cpu_group)
    if any(vote != (mode, None) for vote in votes):
        raise RuntimeError(f"Qwen HC ownership configuration mismatch: {votes}")


def eligible(model, rows: int) -> bool:
    """Only real pure prefills with evenly owned rows bypass compiled decode."""
    if model.hc_prefill_mode == "off" or torch.compiler.is_compiling():
        return False
    if rows < 1024 or rows % 4:
        return False
    if not is_forward_context_available() or torch.cuda.is_current_stream_capturing():
        return False
    context = get_forward_context()
    if (
        getattr(context, "is_dummy_run", False)
        or context.cudagraph_runtime_mode.name != "NONE"
        or context.ubatch_slices is not None
        or not isinstance(context.attn_metadata, dict)
    ):
        return False
    # GDN metadata has CPU counts for prefill, ordinary decode and speculative
    # decode. Check every recurrent layer; never infer purity from row count.
    metadata: list[Any] = [
        item
        for item in context.attn_metadata.values()
        if hasattr(item, "num_prefill_tokens") and hasattr(item, "num_spec_decodes")
    ]
    return bool(metadata) and all(
        type(item.num_prefill_tokens) is int
        and item.num_prefill_tokens == rows
        and item.num_prefills > 0
        and item.num_decodes == 0
        and item.num_spec_decodes == 0
        for item in metadata
    )


@dataclass
class RowOwnership:
    """Keep HC state local and exchange only complete TP block boundaries."""

    rows: int
    rank: int
    group: Any
    reductions: int = 0
    gathers: int = 0

    def local(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[0] != self.rows:
            raise ValueError("Qwen HC full tensor has the wrong row count")
        count = self.rows // 4
        return tensor.narrow(0, self.rank * count, count)

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[0] != self.rows // 4:
            raise ValueError("Qwen HC owned tensor has the wrong row count")
        self.gathers += 1
        source = tensor.contiguous()
        output = source.new_empty((self.rows, *source.shape[1:]))
        self.group.all_gather(output, source)
        return output

    def reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[0] != self.rows:
            raise ValueError("Qwen HC reduction requires full TP partial rows")
        self.reductions += 1
        source = tensor.contiguous()
        output = source.new_empty((self.rows // 4, *source.shape[1:]))
        self.group.reduce_scatter(output, source)
        return output


def create(model, rows: int) -> RowOwnership | None:
    if model.hc_prefill_mode != "shard":
        return None
    group = get_tp_group()
    comm = getattr(group.device_communicator, "pynccl_comm", None)
    if comm is None or not comm.available or comm.disabled:
        raise RuntimeError("Qwen HC ownership requires the TP NCCL communicator")
    return RowOwnership(rows, group.rank_in_group, comm)


def report(model, owner: RowOwnership | None, rows: int) -> None:
    if model.hc_prefill_reports < 8:
        logger.info(
            "QWEN_HC_PREFILL mode=%s rows=%d owner_rows=%d rs=%d ag=%d",
            model.hc_prefill_mode,
            rows,
            rows if owner is None else rows // 4,
            0 if owner is None else owner.reductions,
            0 if owner is None else owner.gathers,
        )
        model.hc_prefill_reports += 1
