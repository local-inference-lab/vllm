# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import regex as re
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.distributed.utils import is_weak_contiguous
from vllm.logger import init_logger
from vllm.utils.b12x import B12xPreparationUnit, B12xWorkload

logger = init_logger(__name__)


@dataclass(frozen=True)
class B12xPcieInvocation:
    """Exact model-owned native collective ABI declared before preparation."""

    name: str
    operation: Literal["all_reduce", "all_reduce_fused_add_rms_norm"]
    shape: tuple[int, ...]
    dtype: torch.dtype
    strides: tuple[int, ...] | None = None
    input_alignment: int = 16
    norm_weight: torch.Tensor | None = None
    epsilon: float | None = None
    channel_id: str | None = None
    persistent_input: torch.Tensor | None = None

    def __post_init__(self):
        if self.operation not in (
            "all_reduce",
            "all_reduce_fused_add_rms_norm",
        ):
            raise ValueError(f"unsupported PCIe collective operation {self.operation!r}")
        if not self.name or any(type(value) is not int or value <= 0 for value in self.shape):
            raise ValueError("PCIe invocation requires a nonempty name and positive shape")
        if self.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("PCIe invocation dtype is unsupported")
        if self.strides is not None and (
            len(self.strides) != len(self.shape)
            or any(type(value) is not int or value <= 0 for value in self.strides)
        ):
            raise ValueError("PCIe invocation strides must match its shape")
        if self.input_alignment < 16 or self.input_alignment & (self.input_alignment - 1):
            raise ValueError("PCIe invocation alignment must be a power of two >= 16")
        fused = self.operation == "all_reduce_fused_add_rms_norm"
        if fused != (self.norm_weight is not None):
            raise ValueError("fused PCIe invocation requires exactly one norm weight")
        if fused and (self.epsilon is None or self.epsilon < 0):
            raise ValueError("fused PCIe invocation requires a nonnegative epsilon")
        if not fused and (self.epsilon is not None or self.norm_weight is not None):
            raise ValueError("plain PCIe invocation cannot carry RMSNorm controls")



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


def _twoshot_max_bytes() -> int:
    """Largest all-reduce routed to the opt-in lossless BF16 two-shot."""
    raw = envs.VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE.strip().lower()
    if raw in ("", "0", "off", "none", "disabled"):
        return 0
    return _parse_byte_size(raw)


@lru_cache(maxsize=1)
def _load_b12x_twoshot_bf16() -> Any | None:
    try:
        from b12x.comm.pcie import PCIeTwoShotBF16
    except ModuleNotFoundError as exc:
        if exc.name != "b12x":
            raise
        return None
    return PCIeTwoShotBF16


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


def _dma_capacity_plan() -> dict[torch.dtype, int] | None:
    """Plan static per-dtype element bounds, including FP32 reductions."""
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

    max_tokens = config.scheduler_config.max_num_batched_tokens
    capacities: dict[torch.dtype, int] = {}
    for model_config in model_configs:
        elements = max_tokens * model_config.get_hidden_size()
        dtype = model_config.dtype
        capacities[dtype] = max(capacities.get(dtype, 0), elements)
        capacities[torch.float32] = max(capacities.get(torch.float32, 0), elements)
    return capacities


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
    """Adapter for B12X PCIe one-shot, two-shot, and DMA runtimes."""

    def __init__(
        self,
        group: ProcessGroup,
        device_group: ProcessGroup | None,
        device: torch.device,
        *,
        global_ranks: Sequence[int] | None = None,
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
        self.global_ranks = tuple(
            int(rank)
            for rank in (
                global_ranks if global_ranks is not None else range(self.world_size)
            )
        )
        self._describers: list[
            tuple[object, Callable[[object], Sequence[B12xPcieInvocation]]]
        ] = []
        self._plans: dict[str, object] = {}
        self._invocations: dict[str, B12xPcieInvocation] = {}
        self._routes: dict[str, str] = {}
        if len(self.global_ranks) != self.world_size:
            raise ValueError("PCIe global ranks must match the process group")

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
        self._twoshot: Any | None = None
        self.twoshot_max_bytes = 0
        self._initialize_twoshot()
        self.disabled = False
        from vllm.utils.b12x import register_b12x_unit_provider

        register_b12x_unit_provider(self)

        if self.rank == 0:
            logger.info(
                "Using B12X PCIe all-reduce (algorithm=%s, one-shot max=%d, "
                "fused max=%d, two-shot bf16 max=%s, DMA min=%s).",
                getattr(runtime, "algorithm", "oneshot"),
                self.allreduce_max_bytes,
                self.fused_max_bytes,
                self.twoshot_max_bytes if self._twoshot is not None else "off",
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

        capacity_plan = _dma_capacity_plan()
        if capacity_plan is None:
            logger.warning(
                "B12X PCIe DMA all-reduce requires an active vLLM model and "
                "scheduler configuration; large tensors will use PyNCCL."
            )
            return
        capacity = max(
            dtype.itemsize * elements for dtype, elements in capacity_plan.items()
        )

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

    def _initialize_twoshot(self) -> None:
        """Lossless bf16 two-shot for payloads above the one-shot ceiling."""
        max_bytes = _twoshot_max_bytes()
        if max_bytes <= 0 or self.world_size != 4:
            return
        twoshot_cls = _load_b12x_twoshot_bf16()
        if twoshot_cls is None:
            logger.warning("B12X PCIe two-shot bf16 requested but unavailable.")
            return
        row_elems = int(os.getenv("VLLM_PCIE_TWOSHOT_ROW_ELEMS", "4096"))
        # Rows are counted in row_elems-wide bf16 rows; keep a multiple of the
        # world size and enough capacity for the configured byte ceiling.
        max_rows = max_bytes // (row_elems * 2)
        max_rows -= max_rows % self.world_size
        if max_rows < self.world_size:
            return
        twoshot: Any | None = None
        init_error: Exception | None = None
        try:
            twoshot = twoshot_cls.from_exchange_group(
                exchange_group=self.device_group,
                device=self.device,
                max_rows=max_rows,
                row_elems=row_elems,
            )
        except Exception as exc:
            init_error = exc
        if not self._all_ranks_succeeded(init_error):
            if twoshot is not None:
                twoshot.close()
            logger.warning(
                "B12X PCIe two-shot bf16 initialization failed on rank %d: %s; "
                "mid-size tensors will use PyNCCL.",
                self.rank,
                init_error,
            )
            return
        assert twoshot is not None
        self._twoshot = twoshot
        self.twoshot_max_bytes = max_rows * row_elems * 2

    def _twoshot_accepts(self, inp: torch.Tensor) -> bool:
        twoshot = self._twoshot
        return bool(
            twoshot is not None
            and inp.nbytes > self.allreduce_max_bytes
            and inp.nbytes <= self.twoshot_max_bytes
            and twoshot.accepts(inp)
        )

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

    def register_describer(
        self,
        owner: object,
        describe: Callable[[object], Sequence[B12xPcieInvocation]],
    ) -> None:
        if not callable(describe):
            raise TypeError("PCIe collective describer must be callable")
        for index, (existing_owner, _) in enumerate(self._describers):
            if existing_owner is owner:
                self._describers[index] = (owner, describe)
                return
        self._describers.append((owner, describe))

    @staticmethod
    def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
        stride = 1
        values = []
        for extent in reversed(shape):
            values.append(stride)
            stride *= extent
        return tuple(reversed(values))

    def _route_invocation(self, invocation: B12xPcieInvocation) -> str | None:
        """Apply the native routing arithmetic to producer-owned metadata."""
        nbytes = invocation.dtype.itemsize
        for extent in invocation.shape:
            nbytes *= extent
        strides = (
            self._contiguous_strides(invocation.shape)
            if invocation.strides is None
            else invocation.strides
        )
        contiguous = strides == self._contiguous_strides(invocation.shape)
        if invocation.operation == "all_reduce":
            if nbytes <= self.allreduce_max_bytes:
                return "oneshot"
            if (
                self._twoshot is not None
                and invocation.dtype is torch.bfloat16
                and nbytes <= self.twoshot_max_bytes
                and nbytes > self.allreduce_max_bytes
                and contiguous
                and (nbytes // invocation.dtype.itemsize)
                % (self.world_size * self._twoshot.row_elems) == 0
            ):
                return "twoshot"
            if (
                self._dma is not None
                and nbytes >= self._dma.min_bytes
                and nbytes <= self._dma.max_bytes
                and contiguous
                and (nbytes // invocation.dtype.itemsize) % (self.world_size * 8) == 0
            ):
                return "dma"
        elif invocation.operation == "all_reduce_fused_add_rms_norm":
            if nbytes <= self.fused_max_bytes:
                return "oneshot_fused"
        # The enclosing CUDA communicator retains its existing backend dispatch
        # for sizes this transport does not accept. They are not native obligations.
        return None

    def _request_input(
        self, invocation: B12xPcieInvocation, *, registered: bool
    ) -> torch.Tensor:
        if registered:
            if invocation.persistent_input is None:
                from vllm.utils.b12x import PreparationResourceUnavailableError

                raise PreparationResourceUnavailableError(
                    "direct PCIe oneshot preparation requires the producer's "
                    "persistent input buffer"
                )
            return invocation.persistent_input
        strides = (
            self._contiguous_strides(invocation.shape)
            if invocation.strides is None
            else invocation.strides
        )
        return torch.empty_strided(
            invocation.shape, strides, dtype=invocation.dtype, device=self.device
        )

    @staticmethod
    def _oneshot_prime_input(
        inp: torch.Tensor, *, registered: bool
    ) -> tuple[
        Callable[[], None] | None,
        Callable[[], None],
        Callable[[], None] | None,
        tuple[torch.Tensor, ...],
    ]:
        """Supply finite activation data and restore a borrowed direct buffer."""
        snapshot = torch.empty_like(inp) if registered else None

        def reset() -> None:
            assert snapshot is not None
            snapshot.copy_(inp)

        def produce() -> None:
            inp.fill_(1)

        def restore() -> None:
            assert snapshot is not None
            inp.copy_(snapshot)

        return (
            reset if snapshot is not None else None,
            produce,
            restore if snapshot is not None else None,
            () if snapshot is None else (snapshot,),
        )

    def get_b12x_preparation_units(
        self, owner: object, workload: B12xWorkload
    ) -> Sequence[B12xPreparationUnit]:
        if owner is not self:
            raise ValueError("PCIe preparation owner mismatch")
        if workload.stage != "weights":
            return ()
        from b12x.preparation import CollectiveRequirement
        from b12x.comm.pcie import (
            _dma_preparation,
            _oneshot_preparation,
            _twoshot_preparation,
        )

        invocations = [
            invocation
            for _, describe in self._describers
            for invocation in describe(workload)
        ]
        names = [invocation.name for invocation in invocations]
        if len(names) != len(set(names)):
            from collections import Counter
            duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
            raise ValueError(
                "PCIe collective describers produced duplicate names: "
                + ", ".join(duplicates)
            )
        requests = []
        plans: dict[str, object] = {}
        routes: dict[str, str] = {}
        collective_ranks = tuple(sorted(self.global_ranks))
        declarations = {}
        for invocation in invocations:
            route = self._route_invocation(invocation)
            if route is None:
                continue
            routes[invocation.name] = route
            target = self._runtime._prepared_channel_for_stream(
                None, invocation.channel_id
            )
            if route.startswith("oneshot"):
                surface = (
                    "OneshotAllReduce.all_reduce_fused_add_rms_norm"
                    if route == "oneshot_fused"
                    else "OneshotAllReduce.all_reduce"
                )
                query = _oneshot_preparation.query_from_metadata(
                    target, surface=surface, shape=invocation.shape,
                    dtype=invocation.dtype, strides=invocation.strides,
                    alignment=invocation.input_alignment,
                )
                binding = (
                    id(invocation.persistent_input)
                    if query.setup["registered"] else None
                )
                key = (id(target), query, binding)
                plan = declarations.get(key)
                if plan is None:
                    plan = _oneshot_preparation.plan(query, runtime=target)
                    declarations[key] = plan

                def prepare(state, invocation=invocation, query=query):
                    registered = bool(query.setup["registered"])
                    inp = self._request_input(invocation, registered=registered)
                    reset, produce, restore, owners = self._oneshot_prime_input(
                        inp, registered=registered
                    )
                    if invocation.operation == "all_reduce":
                        return _oneshot_preparation._prepare_plain_call(
                            state, inp=inp, out=torch.empty_like(inp), produce=produce,
                            reset=reset, restore=restore, owners=owners,
                        )
                    residual = torch.empty_like(inp)

                    def produce_fused() -> None:
                        produce()
                        residual.fill_(1)

                    return _oneshot_preparation._prepare_fused_call(
                        state, inp=inp, residual=residual,
                        weight=invocation.norm_weight, out=torch.empty_like(inp),
                        residual_out=residual, epsilon=invocation.epsilon,
                        produce=produce_fused, reset=reset, restore=restore, owners=owners,
                    )
            elif route == "twoshot":
                assert self._twoshot is not None
                query = _twoshot_preparation.query_from_metadata(
                    self._twoshot, surface="PCIeTwoShotBF16.all_reduce",
                    shape=invocation.shape, dtype=invocation.dtype,
                    strides=invocation.strides, alignment=invocation.input_alignment,
                )
                key = (id(self._twoshot), query, None)
                plan = declarations.get(key)
                if plan is None:
                    plan = _twoshot_preparation.plan(query, runtime=self._twoshot)
                    declarations[key] = plan

                def prepare(state, invocation=invocation):
                    inp = self._request_input(invocation, registered=False)
                    assert inp is not None
                    inp.fill_(1)
                    return _twoshot_preparation.prepared_call(
                        state, payload=inp, out=torch.empty_like(inp)
                    )
            else:
                assert self._dma is not None
                query = _dma_preparation.query_from_metadata(
                    self._dma, shape=invocation.shape, dtype=invocation.dtype,
                    strides=invocation.strides, alignment=invocation.input_alignment,
                )
                key = (id(self._dma), query, None)
                plan = declarations.get(key)
                if plan is None:
                    plan = _dma_preparation.plan(query, runtime=self._dma)
                    declarations[key] = plan

                def prepare(state, invocation=invocation):
                    inp = self._request_input(invocation, registered=False)
                    assert inp is not None
                    inp.fill_(1)
                    return _dma_preparation.prepared_call(
                        state, inp=inp, out=torch.empty_like(inp)
                    )
            plans[invocation.name] = plan
            requests.append(
                plan.request(
                    name=invocation.name,
                    prepare_call=prepare,
                    collective=CollectiveRequirement(
                        key=(
                            f"pcie:{','.join(map(str, collective_ranks))}:"
                            f"{invocation.name}"
                        ),
                        ranks=collective_ranks,
                    ),
                )
            )
        self._invocations = {
            invocation.name: invocation for invocation in invocations
            if invocation.name in routes
        }
        self._plans = plans
        self._routes = routes
        if not requests:
            return ()
        return (
            B12xPreparationUnit(
                name="PCIE_ALL_REDUCE",
                key=(self.global_ranks, tuple(sorted(routes.items()))),
                requests=tuple(requests),
                stage="weights",
                autotune=not workload.eager_only,
            ),
        )

    def _plan_for(
        self, inp: torch.Tensor, *, operation: str = "all_reduce",
        weight: torch.Tensor | None = None, epsilon: float | None = None,
    ):
        strides = tuple(inp.stride())
        for name, invocation in self._invocations.items():
            if (
                invocation.operation == operation
                and tuple(inp.shape) == invocation.shape
                and inp.dtype == invocation.dtype
                and strides
                == (
                    self._contiguous_strides(invocation.shape)
                    if invocation.strides is None
                    else invocation.strides
                )
                and (
                    operation == "all_reduce"
                    or (
                        invocation.norm_weight is weight
                        and invocation.epsilon == epsilon
                    )
                )
            ):
                return self._plans[name]
        from vllm.utils.b12x import PreparationResourceUnavailableError

        raise PreparationResourceUnavailableError(
            f"PCIe collective has no declared plan for {operation} on shape "
            f"{tuple(inp.shape)} {inp.dtype} strides {strides}; declared: "
            + ", ".join(
                f"{invocation.operation}{invocation.shape}"
                for invocation in self._invocations.values()
            )
        )

    def _has_plan_for(self, inp: torch.Tensor, **kwargs) -> bool:
        """True when a prepared plan exists for this exact call shape.

        Only declared shapes take the b12x path; any other shape is served by
        the next all-reduce backend in the dispatch order. Declared shapes are
        the captured and planned serving shapes, so an undeclared shape is a
        prefill chunk or warm-up batch outside the graph-replayed set.
        """
        from vllm.utils.b12x import PreparationResourceUnavailableError

        try:
            self._plan_for(inp, **kwargs)
        except PreparationResourceUnavailableError:
            logger.debug(
                "b12x PCIe all-reduce declines undeclared shape %s %s",
                tuple(inp.shape), inp.dtype,
            )
            return False
        return True

    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        if self.disabled:
            return False
        if not self._has_plan_for(inp):
            return False
        if self._oneshot_accepts(inp):
            return True
        if self._twoshot_accepts(inp):
            return True
        return self._dma is not None and self._dma.should_allreduce(inp)

    def custom_all_reduce(self, inp: torch.Tensor) -> torch.Tensor | None:
        if not self.should_custom_ar(inp):
            return None
        use_oneshot = self._oneshot_accepts(inp)
        use_twoshot = (not use_oneshot) and self._twoshot_accepts(inp)
        if use_twoshot:
            twoshot = self._twoshot
            assert twoshot is not None
            out = torch.empty_like(inp)
            return twoshot.all_reduce(
                inp, out=out, plan=self._plan_for(inp)
            )
        return self._all_reduce(inp, use_oneshot=use_oneshot)

    def _all_reduce(self, inp: torch.Tensor, *, use_oneshot: bool) -> torch.Tensor:
        plan = self._plan_for(inp)
        if use_oneshot:
            assert self._runtime is not None
            return self._runtime.all_reduce(
                inp, stream=self._runtime_stream(), plan=plan
            )
        assert self._dma is not None
        stream = self._runtime_stream()
        if stream is None:
            return self._dma.all_reduce(inp, plan=plan)
        with torch.cuda.stream(stream):
            return self._dma.all_reduce(inp, plan=plan)

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

        if not self._has_plan_for(
            inp, operation="all_reduce_fused_add_rms_norm", weight=weight, epsilon=epsilon,
        ):
            return False
        runtime = self._runtime
        assert runtime is not None
        stream = self._runtime_stream()
        if not runtime.for_stream(stream).should_allreduce(inp):
            return False
        runtime.all_reduce_fused_add_rms_norm(
            inp,
            residual,
            weight,
            epsilon,
            plan=self._plan_for(
                inp,
                operation="all_reduce_fused_add_rms_norm",
                weight=weight,
                epsilon=epsilon,
            ),
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
        if not self._plans:
            from vllm.utils.b12x import PreparationResourceUnavailableError

            raise PreparationResourceUnavailableError(
                "PCIe capture requires a declared native preparation plan"
            )

        old_stream = self._capture_stream
        old_capturing = self._is_capturing
        self._capture_stream = stream
        self._is_capturing = True
        try:
            twoshot_plan = None
            if self._twoshot is not None:
                twoshot_plan = next(
                    (self._plans[name] for name, route in self._routes.items()
                     if route == "twoshot"),
                    None,
                )
            with self._runtime.capture(stream=stream):
                if twoshot_plan is not None:
                    assert self._twoshot is not None
                    with self._twoshot.capture(plan=twoshot_plan):
                        yield
                else:
                    yield
        finally:
            self._capture_stream = old_stream
            self._is_capturing = old_capturing

    def close(self) -> None:
        twoshot = getattr(self, "_twoshot", None)
        if twoshot is not None:
            twoshot.close()
            self._twoshot = None
        if self._dma is not None:
            self._dma.close()
            self._dma = None
        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None
        self._plans.clear()
        self._invocations.clear()
        self._routes.clear()
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
