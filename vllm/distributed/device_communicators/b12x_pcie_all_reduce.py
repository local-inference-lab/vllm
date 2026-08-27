# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

import regex as re
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.distributed.utils import is_weak_contiguous
from vllm.logger import init_logger

logger = init_logger(__name__)


def _parse_byte_size(value: str) -> int:
    match = re.fullmatch(r"\s*([+-]?\d+)\s*([kmgt]?i?b?)?\s*", value.lower())
    if match is None:
        raise ValueError(f"invalid byte size: {value!r}")
    amount = int(match.group(1))
    suffix = match.group(2) or ""
    multipliers = {
        "": 1,
        "b": 1,
        "k": 1 << 10,
        "kb": 1 << 10,
        "kib": 1 << 10,
        "m": 1 << 20,
        "mb": 1 << 20,
        "mib": 1 << 20,
        "g": 1 << 30,
        "gb": 1 << 30,
        "gib": 1 << 30,
        "t": 1 << 40,
        "tb": 1 << 40,
        "tib": 1 << 40,
    }
    try:
        return amount * multipliers[suffix]
    except KeyError as exc:
        raise ValueError(f"invalid byte-size suffix: {suffix!r}") from exc


@lru_cache(maxsize=1)
def _load_b12x_pcie() -> tuple[Any, Any, Any] | None:
    try:
        from b12x.comm.pcie import AllReduce, DmaAllReduce, is_supported
    except ModuleNotFoundError as exc:
        if exc.name != "b12x":
            raise
        return None
    return AllReduce, DmaAllReduce, is_supported


@lru_cache(maxsize=1)
def _load_b12x_recommended_max_bytes() -> Any | None:
    try:
        from b12x.comm.pcie.pcie_allreduce import recommended_max_bytes
    except ModuleNotFoundError as exc:
        if exc.name != "b12x":
            raise
        return None
    return recommended_max_bytes


def _allreduce_max_bytes(world_size: int) -> int:
    configured = os.getenv("VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE")
    if configured is not None:
        return _parse_byte_size(configured)

    default = _parse_byte_size(envs.VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE)
    recommender = _load_b12x_recommended_max_bytes()
    if recommender is None:
        return default
    return int(recommender(world_size, default=default))


def _oneshot_limits(world_size: int) -> tuple[int, int, int]:
    allreduce_max = _allreduce_max_bytes(world_size)
    fused_max = _parse_byte_size(envs.VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE)
    if allreduce_max < 0 or fused_max < 0:
        raise ValueError("B12X PCIe one-shot size limits must be non-negative")
    return allreduce_max, fused_max, max(allreduce_max, fused_max, 16)


def _dma_min_bytes() -> int | None:
    configured = envs.VLLM_PCIE_DMA_MIN_BYTES.strip().lower()
    if configured in {"off", "disabled", "none"}:
        return None
    value = _parse_byte_size(configured)
    if value < 0:
        raise ValueError("B12X PCIe DMA minimum size must be non-negative")
    return value


def _dma_capacity_bytes() -> int | None:
    from vllm.config import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    if config is None or config.model_config is None:
        return None

    model_configs = [config.model_config]
    speculative_config = config.speculative_config
    draft_config = (
        getattr(speculative_config, "draft_model_config", None)
        if speculative_config is not None
        else None
    )
    if draft_config is not None:
        model_configs.append(draft_config)

    max_row_bytes = max(
        model_config.get_hidden_size()
        * torch.empty((), dtype=model_config.dtype).element_size()
        for model_config in model_configs
    )
    return config.scheduler_config.max_num_batched_tokens * max_row_bytes


def _is_piecewise_cudagraph_runtime() -> bool:
    try:
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )
    except ImportError:
        return False
    return (
        is_forward_context_available()
        and get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
    )


class B12xPcieAllReduce:
    """Adapter for B12X PCIe one-shot and DMA all-reduce runtimes."""

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
        self._runtime: Any | None = None
        self._dma: Any | None = None
        self._is_capturing = False
        self._capture_stream: torch.cuda.Stream | None = None

        if device_group is None:
            logger.warning("B12X PCIe all-reduce requires a CUDA process group.")
            return
        if not all(in_the_same_node_as(group, source_rank=0)):
            logger.warning("B12X PCIe all-reduce supports only single-node groups.")
            return

        b12x_pcie = _load_b12x_pcie()
        if b12x_pcie is None:
            logger.warning(
                "B12X PCIe all-reduce was requested, but b12x is not installed."
            )
            return
        allreduce_cls, dma_cls, is_supported = b12x_pcie
        if not is_supported(device):
            logger.warning("B12X PCIe all-reduce is unsupported on device %s.", device)
            return

        self.allreduce_max_bytes, self.fused_max_bytes, buffer_bytes = _oneshot_limits(
            self.world_size
        )
        runtime: Any | None = None
        init_error: Exception | None = None
        try:
            runtime = allreduce_cls.from_exchange_group(
                exchange_group=device_group,
                device=device,
                eager_buffer_bytes=buffer_bytes,
                max_size=buffer_bytes,
                single_channel=True,
                max_concurrent_channels=1,
            )
        except Exception as exc:
            init_error = exc

        if not self._all_ranks_succeeded(init_error):
            if runtime is not None:
                runtime.close()
            if init_error is not None:
                logger.warning(
                    "B12X PCIe all-reduce initialization failed on rank %d: %s",
                    self.rank,
                    init_error,
                )
            else:
                logger.warning(
                    "B12X PCIe all-reduce initialization failed on another rank."
                )
            return

        assert runtime is not None
        self._runtime = runtime
        self._initialize_dma(dma_cls)
        self.disabled = False

        if self.rank == 0:
            logger.info(
                "Using B12X PCIe all-reduce (algorithm=%s, one-shot max=%d, "
                "fused max=%d, DMA min=%s).",
                getattr(runtime, "algorithm", "oneshot"),
                self.allreduce_max_bytes,
                self.fused_max_bytes,
                getattr(self._dma, "min_bytes", "off"),
            )

    def _all_ranks_succeeded(self, error: Exception | None) -> bool:
        failed = torch.tensor([int(error is not None)], dtype=torch.int32)
        dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=self.group)
        return int(failed.item()) == 0

    def _initialize_dma(self, dma_cls: Any) -> None:
        assert self._runtime is not None
        min_bytes = _dma_min_bytes()
        if min_bytes is None or not bool(
            getattr(self._runtime, "supports_all_peer_auxiliary", True)
        ):
            return

        capacity = _dma_capacity_bytes()
        if capacity is None:
            logger.warning(
                "B12X PCIe DMA all-reduce requires an active vLLM model and "
                "scheduler configuration; large tensors will use PyNCCL."
            )
            return

        dma: Any | None = None
        init_error: Exception | None = None
        try:
            dma = dma_cls(
                exchange_group=self.device_group,
                device=self.device,
                max_bytes=capacity,
                fp8=envs.VLLM_PCIE_DMA_FP8,
            )
        except Exception as exc:
            init_error = exc

        if not self._all_ranks_succeeded(init_error):
            if dma is not None:
                dma.close()
            logger.warning(
                "B12X PCIe DMA all-reduce initialization failed on rank %d: %s; "
                "large tensors will use PyNCCL.",
                self.rank,
                init_error,
            )
            return

        assert dma is not None
        dma.min_bytes = min_bytes
        self._dma = dma

    def _runtime_stream(self) -> torch.cuda.Stream | None:
        stream = self._capture_stream
        if stream is None:
            return None
        if not (self._is_capturing or torch.cuda.is_current_stream_capturing()):
            return None
        if torch.cuda.current_stream().cuda_stream != stream.cuda_stream:
            return None
        return stream

    def _oneshot_accepts(self, inp: torch.Tensor) -> bool:
        runtime = self._runtime
        return bool(
            runtime is not None
            and inp.nbytes <= self.allreduce_max_bytes
            and runtime.for_stream(self._runtime_stream()).should_allreduce(inp)
        )

    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        if self.disabled:
            return False
        if self._oneshot_accepts(inp):
            return True
        return self._dma is not None and self._dma.should_allreduce(inp)

    def custom_all_reduce(self, inp: torch.Tensor) -> torch.Tensor | None:
        if not self.should_custom_ar(inp):
            return None

        runtime = self._runtime
        assert runtime is not None
        use_oneshot = self._oneshot_accepts(inp)
        if self._is_capturing and not torch.cuda.is_current_stream_capturing():
            if _is_piecewise_cudagraph_runtime():
                return self._all_reduce(inp, use_oneshot=use_oneshot)
            if use_oneshot:
                prepare = getattr(runtime, "prepare_graph_all_reduce", None)
                if prepare is not None:
                    prepare(inp, stream=self._runtime_stream())
            return torch.empty_like(inp)
        return self._all_reduce(inp, use_oneshot=use_oneshot)

    def _all_reduce(self, inp: torch.Tensor, *, use_oneshot: bool) -> torch.Tensor:
        if use_oneshot:
            assert self._runtime is not None
            return self._runtime.all_reduce(inp, stream=self._runtime_stream())
        assert self._dma is not None
        stream = self._runtime_stream()
        if stream is None:
            return self._dma.all_reduce(inp)
        with torch.cuda.stream(stream):
            return self._dma.all_reduce(inp)

    def supports_fused_add_rms_norm(self) -> bool:
        return bool(
            not self.disabled
            and self._runtime is not None
            and self.fused_max_bytes > 0
            and hasattr(self._runtime, "all_reduce_fused_add_rms_norm")
        )

    def try_fused_add_rms_norm(
        self,
        inp: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        epsilon: float,
    ) -> bool:
        if (
            not self.supports_fused_add_rms_norm()
            or inp.nbytes > self.fused_max_bytes
            or inp.ndim == 0
            or residual.shape != inp.shape
            or residual.dtype != inp.dtype
            or residual.device != inp.device
            or not is_weak_contiguous(residual)
            or weight.shape != (inp.shape[-1],)
            or weight.dtype != inp.dtype
            or weight.device != inp.device
            or not weight.is_contiguous()
            or inp.shape[-1] * inp.element_size() % 16 != 0
            or inp.data_ptr() == residual.data_ptr()
            or epsilon < 0
        ):
            return False

        runtime = self._runtime
        assert runtime is not None
        stream = self._runtime_stream()
        if not runtime.for_stream(stream).should_allreduce(inp):
            return False
        if self._is_capturing and not torch.cuda.is_current_stream_capturing():
            prepare = getattr(runtime, "prepare_graph_fused_add_rms_norm", None)
            if prepare is not None:
                prepare(inp, stream=stream)
        runtime.all_reduce_fused_add_rms_norm(
            inp,
            residual,
            weight,
            epsilon,
            out=inp,
            residual_out=residual,
            stream=stream,
        )
        return True

    @contextmanager
    def capture(self, stream: torch.cuda.Stream | None = None):
        if self.disabled or self._runtime is None:
            yield
            return

        old_stream = self._capture_stream
        old_capturing = self._is_capturing
        self._capture_stream = stream
        self._is_capturing = True
        try:
            with self._runtime.capture(stream=stream):
                yield
        finally:
            self._capture_stream = old_stream
            self._is_capturing = old_capturing

    def close(self) -> None:
        if self._dma is not None:
            self._dma.close()
            self._dma = None
        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None
        self.disabled = True


def get_b12x_pcie_allreduce() -> B12xPcieAllReduce | None:
    """Return the active tensor-parallel B12X communicator, if available."""
    try:
        from vllm.distributed.parallel_state import get_tp_group

        device_communicator = get_tp_group().device_communicator
    except (AssertionError, RuntimeError):
        return None
    communicator = getattr(device_communicator, "b12x_ar_comm", None)
    if (
        isinstance(communicator, B12xPcieAllReduce)
        and communicator.supports_fused_add_rms_norm()
    ):
        return communicator
    return None
