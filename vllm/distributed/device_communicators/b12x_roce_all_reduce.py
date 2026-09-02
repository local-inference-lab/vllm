# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RoCEnante: adapter for the b12x one-shot RoCE collectives (multi-node DGX Spark TP).

Thin shim: capability voting, construction, and size gating live here; the
protocol lives in ``b12x.comm.roce``.  Enabled with
``VLLM_ENABLE_ROCE_ALLREDUCE=1`` for tensor-parallel groups whose ranks span
nodes; single-node groups keep their existing backends.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm.distributed.device_communicators.b12x_pcie_all_reduce import (
    _parse_byte_size,
)
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import init_logger

logger = init_logger(__name__)


class B12xRoceAllReduce:
    """Route eligible tensor-parallel all-reduces to ``b12x.comm.roce``."""

    backend_name = "B12X_ROCENANTE"

    def __init__(
        self,
        group: ProcessGroup,
        device_group: ProcessGroup | None,
        device: torch.device,
    ) -> None:
        self.disabled = True
        self.group = group
        self.device_group = device_group
        self.device = device
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        self._runtime = None
        self._announced = False
        self._announced_gather = False

        if device_group is None:
            logger.warning("RoCEnante requires a CUDA process group.")
            return
        if all(in_the_same_node_as(group, source_rank=0)):
            logger.info("RoCEnante skipped: group is single-node.")
            return

        # Vote before the collective constructor so a rank that cannot take
        # part fails every rank cleanly instead of leaving peers in a gather.
        reason = self._local_capability()
        if not self._all_ranks_succeeded(reason):
            if reason is not None:
                logger.warning(
                    "RoCEnante unavailable on rank %d: %s", self.rank, reason
                )
            else:
                logger.warning("RoCEnante unavailable on another rank.")
            return

        from b12x.comm import roce

        max_size = _parse_byte_size(envs.VLLM_ROCE_ALLREDUCE_MAX_SIZE)
        max_gather = _parse_byte_size(envs.VLLM_ROCE_ALLGATHER_MAX_SIZE)
        try:
            # Exchange setup over the CPU (gloo) group: using the torch NCCL
            # group would create a torch NCCL communicator that vLLM otherwise
            # never needs, costing ~3.4 GB of unified memory per rank on Spark.
            self._runtime = roce.AllReduce.from_exchange_group(
                exchange_group=group,
                device=device,
                max_size=max_size,
                max_gather_bytes=max_gather,
            )
        except Exception as exc:  # noqa: BLE001 - the runtime already coordinated ranks
            logger.warning("RoCEnante initialization failed: %s", exc)
            return
        self.disabled = False
        if self.rank == 0:
            logger.info(
                "Using RoCEnante (b12x one-shot RoCE collectives): world=%d, hcas=%s, "
                "all-reduce max=%d bytes, all-gather shard max=%d bytes.",
                self.world_size,
                ",".join(self._runtime.hca_names),
                max_size,
                max_gather,
            )

    def _local_capability(self) -> str | None:
        try:
            from b12x.comm import roce
        except ModuleNotFoundError:
            return "b12x.comm.roce is not installed"
        if not roce.is_supported(self.device):
            return "needs an integrated GPU with an active RDMA device"
        return None

    def _all_ranks_succeeded(self, reason: str | None) -> bool:
        failed = torch.tensor([int(reason is not None)], dtype=torch.int32)
        dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=self.group)
        return int(failed.item()) == 0

    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        return not self.disabled and self._runtime.should_allreduce(inp)

    def custom_all_reduce(self, inp: torch.Tensor) -> torch.Tensor | None:
        if not self.should_custom_ar(inp):
            return None
        if not self._announced:
            self._announced = True
            # One confirmation line, rank 0 only; workers keep it at debug.
            log = logger.info if self.rank == 0 else logger.debug
            log(
                "RoCEnante all-reduce is live: first routed all-reduce is %d bytes "
                "(%s); NCCL remains the fallback above %s.",
                inp.numel() * inp.element_size(),
                str(inp.dtype).replace("torch.", ""),
                envs.VLLM_ROCE_ALLREDUCE_MAX_SIZE,
            )
        return self._runtime.all_reduce(inp)

    def should_all_gather(self, inp: torch.Tensor, dim: int) -> bool:
        return not self.disabled and self._runtime.should_all_gather(inp, dim)

    def all_gather(self, inp: torch.Tensor, dim: int) -> torch.Tensor:
        """Concatenate along ``dim`` (0 or last) directly in the output layout."""

        if not self._announced_gather:
            self._announced_gather = True
            log = logger.info if self.rank == 0 else logger.debug
            log(
                "RoCEnante all-gather is live: first routed shard is %s %s "
                "along dim %d.",
                tuple(inp.shape),
                str(inp.dtype).replace("torch.", ""),
                dim,
            )
        return self._runtime.all_gather(inp, dim=dim)

    def supports_fused_add_rms_norm(self) -> bool:
        return False

    @contextmanager
    def capture(self, stream: torch.cuda.Stream | None = None):
        if self.disabled:
            yield
            return
        # Compile before capture; the runtime refuses to compile inside a graph.
        self._runtime.prepare((torch.bfloat16, torch.float16, torch.float32))
        with self._runtime.capture(stream=stream):
            yield

    def close(self) -> None:
        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None
        self.disabled = True
