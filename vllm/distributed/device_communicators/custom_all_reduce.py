# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from functools import lru_cache
import os
from pathlib import Path
from typing import Any
from typing import cast

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.distributed.device_communicators.all_reduce_utils import (
    CUSTOM_ALL_REDUCE_MAX_SIZES,
    gpu_p2p_access_check,
)
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import init_logger
from vllm.platforms import current_platform

try:
    ops.meta_size()
    custom_ar = True
except Exception:
    # For CPUs
    custom_ar = False

logger = init_logger(__name__)


def _get_pcie_allreduce_backend() -> str:
    backend = os.getenv("VLLM_PCIE_ALLREDUCE_BACKEND", "cpp").lower()
    if backend not in {"b12x", "cpp"}:
        raise ValueError(
            "Invalid VLLM_PCIE_ALLREDUCE_BACKEND: "
            f"{backend!r}. Valid values: b12x, cpp."
        )
    return backend


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _is_piecewise_cudagraph_runtime() -> bool:
    try:
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )
    except Exception:
        return False
    return (
        is_forward_context_available()
        and get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
    )


def _parse_byte_size(value: str) -> int:
    normalized = value.upper().strip()
    suffixes = {
        "KB": 1024,
        "K": 1024,
        "MB": 1024 * 1024,
        "M": 1024 * 1024,
    }
    for suffix, multiplier in sorted(suffixes.items(), key=lambda item: -len(item[0])):
        if normalized.endswith(suffix):
            return int(normalized[: -len(suffix)]) * multiplier
    return int(value)


@lru_cache(maxsize=1)
def _load_b12x_pcie_oneshot_pool() -> Any | None:
    try:
        from b12x.distributed import PCIeOneshotAllReducePool
    except Exception:
        return None
    return PCIeOneshotAllReducePool


def _get_physical_device_numa_node(physical_device_id: int) -> int | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
        try:
            numa_node = pynvml.nvmlDeviceGetNumaNodeId(handle)
            if numa_node >= 0 and _numa_node_has_cpus(numa_node):
                return int(numa_node)
        except Exception:
            pass

        for cpu_id in _get_device_cpu_affinity(pynvml, handle):
            numa_node = _get_numa_node_for_cpu(cpu_id)
            if numa_node is not None:
                return numa_node
    except Exception:
        return None
    return None


def _numa_node_has_cpus(node_id: int) -> bool:
    try:
        return Path(f"/sys/devices/system/node/node{node_id}/cpulist").read_text(
            encoding="utf-8"
        ).strip() != ""
    except (OSError, ValueError):
        return False


def _get_device_cpu_affinity(pynvml: Any, handle: Any) -> list[int]:
    cpu_count = os.cpu_count()
    if cpu_count is None:
        return []

    cpu_set_size = (cpu_count + 63) // 64
    cpu_affinity_mask = pynvml.nvmlDeviceGetCpuAffinity(handle, cpu_set_size)

    cpu_ids = []
    for i, mask in enumerate(cpu_affinity_mask):
        for bit in range(64):
            cpu_id = i * 64 + bit
            if cpu_id >= cpu_count:
                break
            if mask & (1 << bit):
                cpu_ids.append(cpu_id)
    return cpu_ids


def _get_numa_node_for_cpu(cpu_id: int) -> int | None:
    node_path = Path("/sys/devices/system/node")
    if not node_path.exists():
        return None

    for node_dir in node_path.iterdir():
        if not node_dir.name.startswith("node"):
            continue
        try:
            node_id = int(node_dir.name[4:])
            cpulist_file = node_dir / "cpulist"
            if cpulist_file.exists() and _cpu_in_cpulist(
                cpu_id, cpulist_file.read_text(encoding="utf-8").strip()
            ):
                return node_id
        except (ValueError, OSError):
            continue
    return None


def _cpu_in_cpulist(cpu_id: int, cpulist: str) -> bool:
    for part in cpulist.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            if int(start) <= cpu_id <= int(end):
                return True
        elif part and cpu_id == int(part):
            return True
    return False


def _is_cross_numa_topology(physical_device_ids: list[int]) -> bool:
    numa_nodes: list[int] = []
    for physical_device_id in physical_device_ids:
        numa_node = _get_physical_device_numa_node(physical_device_id)
        if numa_node is not None:
            numa_nodes.append(numa_node)

    return len(set(numa_nodes)) > 1


def _can_p2p(rank: int, world_size: int) -> bool:
    for i in range(world_size):
        if i == rank:
            continue
        if envs.VLLM_SKIP_P2P_CHECK:
            logger.debug("Skipping P2P check and trusting the driver's P2P report.")
            return torch.cuda.can_device_access_peer(rank, i)
        if not gpu_p2p_access_check(rank, i):
            return False
    return True


def is_weak_contiguous(inp: torch.Tensor):
    return inp.is_contiguous() or (
        inp.storage().nbytes() - inp.storage_offset() * inp.element_size()
        == inp.numel() * inp.element_size()
    )


class CustomAllreduce:
    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]

    # max_size: max supported allreduce size
    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
        max_size=8192 * 1024,
        symm_mem_enabled=False,
        nccl_group: ProcessGroup | None = None,
    ) -> None:
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the CustomAllreduce to. If None,
                it will be bound to f"cuda:{local_rank}".
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device, and all communicators in this group
        are in the same node.
        """
        self._IS_CAPTURING = False
        self.disabled = True
        self._pcie_runtime = None
        self._pcie_capture_stream: torch.cuda.Stream | None = None
        self._cpp_ar_cutoff_size: int | None = None
        self._cpp_ar_ignore_cutoff_max_rows = 0
        self._cpp_ar_shape_log = False
        self._cpp_ar_logged_shapes: set[tuple[tuple[int, ...], torch.dtype, str]] = set()
        self._pcie_cpp_backend = False
        self._pcie_logged_first_accept = False
        self._pcie_logged_first_reject = False
        self._pcie_logged_first_allreduce = False
        self._ptr = 0

        if not custom_ar:
            # disable because of missing custom allreduce library
            # e.g. in a non-GPU environment
            logger.info(
                "Custom allreduce is disabled because "
                "of missing custom allreduce library"
            )
            return

        self.group = group
        self.nccl_group = nccl_group

        assert dist.get_backend(group) != dist.Backend.NCCL, (
            "CustomAllreduce should be attached to a non-NCCL group."
        )

        if not all(in_the_same_node_as(group, source_rank=0)):
            # No need to initialize custom allreduce for multi-node case.
            logger.warning(
                "Custom allreduce is disabled because this process group"
                " spans across nodes."
            )
            return

        rank = dist.get_rank(group=self.group)
        self.rank = rank
        world_size = dist.get_world_size(group=self.group)
        if world_size == 1:
            # No need to initialize custom allreduce for single GPU case.
            return

        if world_size not in CustomAllreduce._SUPPORTED_WORLD_SIZES:
            logger.warning(
                "Custom allreduce is disabled due to an unsupported world"
                " size: %d. Supported world sizes: %s. To silence this "
                "warning, specify disable_custom_all_reduce=True explicitly.",
                world_size,
                str(CustomAllreduce._SUPPORTED_WORLD_SIZES),
            )
            return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device
        device_capability = current_platform.get_device_capability()
        if (
            current_platform.is_cuda()
            and symm_mem_enabled
            and device_capability is not None
        ):
            device_capability_str = device_capability.as_version_str()
            if device_capability_str in CUSTOM_ALL_REDUCE_MAX_SIZES:
                max_size = min(
                    CUSTOM_ALL_REDUCE_MAX_SIZES[device_capability_str][world_size],
                    max_size,
                )
        cuda_visible_devices = envs.CUDA_VISIBLE_DEVICES
        if cuda_visible_devices:
            device_ids = list(map(int, cuda_visible_devices.split(",")))
        else:
            device_ids = list(range(current_platform.device_count()))

        physical_device_id = device_ids[device.index]
        tensor = torch.tensor([physical_device_id], dtype=torch.int, device="cpu")
        gather_list = [
            torch.tensor([0], dtype=torch.int, device="cpu") for _ in range(world_size)
        ]
        dist.all_gather(gather_list, tensor, group=self.group)
        physical_device_ids = [t.item() for t in gather_list]

        # test nvlink first, this will filter out most of the cases
        # where custom allreduce is not supported
        # this checks hardware and driver support for NVLink
        assert current_platform.is_cuda_alike()
        fully_connected = current_platform.is_fully_connected(physical_device_ids)
        use_pcie_oneshot = False
        if not fully_connected:
            if envs.VLLM_ENABLE_PCIE_ALLREDUCE:
                pcie_backend = _get_pcie_allreduce_backend()
                if pcie_backend == "b12x":
                    logger.info(
                        "b12x PCIe oneshot allreduce requested for PCIe-only "
                        "topology (world_size=%d, physical_device_ids=%s).",
                        world_size,
                        physical_device_ids,
                    )
                    use_pcie_oneshot = current_platform.is_cuda()
                elif world_size > 2:
                    logger.info(
                        "PCIe custom allreduce enabled via "
                        "VLLM_ENABLE_PCIE_ALLREDUCE=1 "
                        "(backend=cpp, using vLLM C++ custom allreduce)."
                    )
                    # Preserve the legacy PCIe opt-in behavior: allow the same
                    # small-tensor C++ custom allreduce path as fully-connected
                    # topologies once the user explicitly enables it.
                    self._pcie_cpp_backend = True
                    fully_connected = True
            elif world_size > 2:
                logger.warning(
                    "Custom allreduce is disabled for >2 PCIe-only GPUs. "
                    "Set VLLM_ENABLE_PCIE_ALLREDUCE=1 to enable P2P custom "
                    "allreduce on PCIe topology (requires P2P-capable driver, "
                    "see PR #39040 for details)."
                )
                return
        # test P2P capability, this checks software/cudaruntime support
        # this is expensive to compute at the first time
        # then we cache the result
        # On AMD GPU, p2p is always enabled between XGMI connected GPUs
        if not current_platform.is_rocm() and not _can_p2p(rank, world_size):
            logger.warning(
                "Custom allreduce is disabled because your platform lacks "
                "GPU P2P capability or P2P test failed. To silence this "
                "warning, specify disable_custom_all_reduce=True explicitly."
            )
            return

        if use_pcie_oneshot:
            allow_cross_numa = os.getenv(
                "VLLM_PCIE_ONESHOT_ALLOW_CROSS_NUMA", "1"
            ) != "0"
            if _is_cross_numa_topology(physical_device_ids) and not allow_cross_numa:
                logger.warning(
                    "Custom allreduce is disabled because b12x PCIe oneshot "
                    "allreduce was requested on a cross-NUMA PCIe topology "
                    "(physical_device_ids=%s). Set "
                    "VLLM_PCIE_ONESHOT_ALLOW_CROSS_NUMA=1 or unset it to force it.",
                    physical_device_ids,
                )
                return
            pool_cls = _load_b12x_pcie_oneshot_pool()
            if pool_cls is None:
                logger.warning(
                    "PCIe custom allreduce was requested, but "
                    "b12x.distributed.PCIeOneshotAllReducePool is unavailable."
                )
                return
            pcie_max_size = min(
                max_size,
                _parse_byte_size(
                    os.getenv("VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE", "64KB")
                ),
            )
            if self.nccl_group is None:
                logger.warning(
                    "Custom allreduce is disabled because b12x PCIe oneshot "
                    "allreduce requires a CUDA/NCCL device process group."
                )
                return
            self.max_size = pcie_max_size
            self.rank = rank
            self.world_size = world_size
            self.fully_connected = False
            self._pcie_runtime = pool_cls.from_exchange_group(
                exchange_group=self.nccl_group,
                device=self.device,
                eager_buffer_bytes=pcie_max_size,
                max_size=pcie_max_size,
            )
            if _env_flag("VLLM_PCIE_ONESHOT_AUTOTUNE"):
                autotune_kwargs = {}
                ceiling = os.getenv("VLLM_PCIE_ONESHOT_AUTOTUNE_CEILING")
                if ceiling is not None:
                    autotune_kwargs["ceiling_bytes"] = _parse_byte_size(ceiling)
                fine_step = os.getenv("VLLM_PCIE_ONESHOT_AUTOTUNE_FINE_STEP")
                if fine_step is not None:
                    autotune_kwargs["fine_step_bytes"] = _parse_byte_size(fine_step)
                warmup = os.getenv("VLLM_PCIE_ONESHOT_AUTOTUNE_WARMUP")
                if warmup is not None:
                    autotune_kwargs["warmup"] = int(warmup)
                iters = os.getenv("VLLM_PCIE_ONESHOT_AUTOTUNE_ITERS")
                if iters is not None:
                    autotune_kwargs["iters"] = int(iters)
                autotune_group = self.nccl_group if self.nccl_group is not None else group
                default_channel = self._pcie_runtime.for_stream()
                tuned_size = default_channel.find_crossover_size(
                    autotune_group, **autotune_kwargs
                )
                self._pcie_runtime.max_size = default_channel.max_size
                self.max_size = default_channel.max_size
                logger.info(
                    "Autotuned b12x PCIe oneshot allreduce max_size=%d "
                    "(requested=%d, crossover=%d).",
                    self.max_size,
                    pcie_max_size,
                    tuned_size,
                )
            self.disabled = False
            logger.info(
                "Using b12x PCIe oneshot allreduce backend "
                "(world_size=%d, max_size=%d).",
                world_size,
                self.max_size,
            )
            return

        if world_size > 2 and not fully_connected:
            logger.warning(
                "Custom allreduce is disabled because this PCIe topology is not "
                "fully connected and b12x PCIe oneshot is unavailable."
            )
            return

        self.disabled = False
        # Buffers memory are owned by this Python class and passed to C++.
        # Metadata composes of two parts: metadata for synchronization and a
        # temporary buffer for storing intermediate allreduce results.
        self.meta_ptrs = self.create_shared_buffer(
            ops.meta_size() + max_size, group=group, uncached=True
        )
        # This is a pre-registered IPC buffer. In eager mode, input tensors
        # are first copied into this buffer before allreduce is performed
        self.buffer_ptrs = self.create_shared_buffer(max_size, group=group)
        # This is a buffer for storing the tuples of pointers pointing to
        # IPC buffers from all ranks. Each registered tuple has size of
        # 8*world_size bytes where world_size is at most 8. Allocating 8MB
        # is enough for 131072 such tuples. The largest model I've seen only
        # needs less than 10000 of registered tuples.
        self.rank_data = torch.empty(
            8 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        self.max_size = max_size
        self.rank = rank
        self.world_size = world_size
        self.fully_connected = fully_connected
        default_cutoff = "56KB" if self._pcie_cpp_backend else None
        cpp_ar_cutoff = os.getenv(
            "VLLM_CPP_AR_1STAGE_NCCL_CUTOFF", default_cutoff or ""
        )
        if cpp_ar_cutoff:
            self._cpp_ar_cutoff_size = _parse_byte_size(cpp_ar_cutoff)
        cpp_ar_ignore_rows = os.getenv(
            "VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS",
            "1" if self._pcie_cpp_backend else "0",
        ) or "0"
        self._cpp_ar_ignore_cutoff_max_rows = int(cpp_ar_ignore_rows)
        self._cpp_ar_shape_log = _env_flag("VLLM_CPP_AR_SHAPE_LOG")
        if (
            self._cpp_ar_cutoff_size is not None
            and self._cpp_ar_ignore_cutoff_max_rows > 0
        ):
            logger.info(
                "Using dynamic C++ custom allreduce cutoff "
                "(cutoff=%d bytes, ignore_cutoff_max_rows=%d).",
                self._cpp_ar_cutoff_size,
                self._cpp_ar_ignore_cutoff_max_rows,
            )
        self._ptr = ops.init_custom_ar(
            self.meta_ptrs, self.rank_data, rank, self.fully_connected
        )
        ops.register_buffer(self._ptr, self.buffer_ptrs)

    @contextmanager
    def capture(self, stream: torch.cuda.Stream | None = None):
        """
        The main responsibility of this context manager is the
        `register_graph_buffers` call at the end of the context.
        It records all the buffer addresses used in the CUDA graph.
        """
        old_pcie_capture_stream = self._pcie_capture_stream
        try:
            self._IS_CAPTURING = True
            if self._pcie_runtime is None:
                yield
            else:
                self._pcie_capture_stream = stream
                with self._pcie_runtime.capture(stream=stream):
                    yield
        finally:
            self._pcie_capture_stream = old_pcie_capture_stream
            self._IS_CAPTURING = False
            if not self.disabled and self._pcie_runtime is None:
                self.register_graph_buffers()

    def _pcie_runtime_stream(self) -> torch.cuda.Stream | None:
        pinned = self._pcie_capture_stream
        if pinned is None:
            return None
        if not (self._IS_CAPTURING or torch.cuda.is_current_stream_capturing()):
            return None
        # Only pin the all-reduce onto the stored capture stream when that
        # stream is the one actually being captured (the full-CUDA-graph
        # path, where vLLM's graph_capture() makes _pcie_capture_stream the
        # current stream). Piecewise / inductor CUDA graphs (e.g. MTP or
        # spec-decode) capture on a torch-owned stream that is *not* our
        # stored stream; redirecting onto the stored stream there would
        # launch and allocate on a non-capturing stream and raise
        # cudaErrorStreamCaptureUnsupported. In that case return None so the
        # runtime runs inline on the current (capturing) stream.
        if torch.cuda.current_stream().cuda_stream != pinned.cuda_stream:
            return None
        return pinned

    def register_graph_buffers(self):
        if self._pcie_runtime is not None:
            self._pcie_runtime.for_stream(
                self._pcie_runtime_stream()
            ).register_graph_buffers()
            return
        handle, offset = ops.get_graph_buffer_ipc_meta(self._ptr)
        logger.info("Registering %d cuda graph addresses", len(offset))
        # We cannot directly use `dist.all_gather_object` here
        # because it is incompatible with `gloo` backend under inference mode.
        # see https://github.com/pytorch/pytorch/issues/126032 for details.
        all_data: list[list[list[int] | None]]
        all_data = [[None, None] for _ in range(dist.get_world_size(group=self.group))]
        all_data[self.rank] = [handle, offset]
        ranks = sorted(dist.get_process_group_ranks(group=self.group))
        for i, rank in enumerate(ranks):
            dist.broadcast_object_list(
                all_data[i], src=rank, group=self.group, device="cpu"
            )
        # Unpack list of tuples to tuple of lists.
        handles = cast(list[list[int]], [d[0] for d in all_data])
        offsets = cast(list[list[int]], [d[1] for d in all_data])
        ops.register_graph_buffers(self._ptr, handles, offsets)

    def should_custom_ar(self, inp: torch.Tensor):
        if self.disabled:
            return False
        if self._pcie_runtime is not None:
            use_custom = self._pcie_runtime.for_stream(
                self._pcie_runtime_stream()
            ).should_allreduce(inp)
            if use_custom and not self._pcie_logged_first_accept:
                self._pcie_logged_first_accept = True
                logger.info(
                    "b12x PCIe oneshot allreduce accepted tensor: "
                    "shape=%s dtype=%s bytes=%d max_size=%d.",
                    tuple(inp.shape),
                    inp.dtype,
                    inp.numel() * inp.element_size(),
                    self.max_size,
                )
            elif not use_custom and not self._pcie_logged_first_reject:
                self._pcie_logged_first_reject = True
                logger.info(
                    "b12x PCIe oneshot allreduce active but rejected tensor: "
                    "shape=%s dtype=%s bytes=%d max_size=%d.",
                    tuple(inp.shape),
                    inp.dtype,
                    inp.numel() * inp.element_size(),
                    self.max_size,
                )
            return use_custom
        inp_size = inp.numel() * inp.element_size()
        rows = int(inp.shape[0]) if inp.ndim >= 2 else 1
        cutoff_applies = not (
            self._cpp_ar_ignore_cutoff_max_rows > 0
            and rows <= self._cpp_ar_ignore_cutoff_max_rows
        )
        if (
            cutoff_applies
            and self._cpp_ar_cutoff_size is not None
            and inp_size > self._cpp_ar_cutoff_size
        ):
            if self._cpp_ar_shape_log:
                self._log_cpp_ar_shape(inp, rows, inp_size, "nccl_cutoff")
            return False
        # custom allreduce requires input byte size to be multiples of 16
        if inp_size % 16 != 0:
            if self._cpp_ar_shape_log:
                self._log_cpp_ar_shape(inp, rows, inp_size, "nccl_unaligned")
            return False
        if not is_weak_contiguous(inp):
            if self._cpp_ar_shape_log:
                self._log_cpp_ar_shape(inp, rows, inp_size, "nccl_noncontiguous")
            return False
        # Keep the runtime guard aligned with the initialization contract
        # above. For >2 PCIe GPUs we only use custom allreduce when the
        # topology is explicitly opted in and treated as fully connected.
        if self.world_size == 2 or self.fully_connected:
            use_custom = inp_size < self.max_size
            if self._cpp_ar_shape_log:
                self._log_cpp_ar_shape(
                    inp,
                    rows,
                    inp_size,
                    "custom" if use_custom else "nccl_max_size",
                )
            return use_custom
        if self._cpp_ar_shape_log:
            self._log_cpp_ar_shape(inp, rows, inp_size, "nccl_topology")
        return False

    def _log_cpp_ar_shape(
        self,
        inp: torch.Tensor,
        rows: int,
        inp_size: int,
        decision: str,
    ) -> None:
        key = (tuple(inp.shape), inp.dtype, decision)
        if key in self._cpp_ar_logged_shapes:
            return
        self._cpp_ar_logged_shapes.add(key)
        logger.info(
            "C++ custom allreduce selector shape=%s dtype=%s rows=%d "
            "bytes=%d cutoff=%s ignore_cutoff_max_rows=%d decision=%s.",
            tuple(inp.shape),
            inp.dtype,
            rows,
            inp_size,
            self._cpp_ar_cutoff_size,
            self._cpp_ar_ignore_cutoff_max_rows,
            decision,
        )

    def backend_name(self) -> str:
        if self._pcie_runtime is not None:
            return "B12X_PCIE_ONESHOT"
        if self._pcie_cpp_backend:
            return "CUSTOM_CPP_PCIE"
        return "CUSTOM"

    def all_reduce(
        self, inp: torch.Tensor, *, out: torch.Tensor = None, registered: bool = False
    ):
        """Performs an out-of-place all reduce.

        If registered is True, this assumes inp's pointer is already
        IPC-registered. Otherwise, inp is first copied into a pre-registered
        buffer.
        """
        if self._pcie_runtime is not None:
            if not self._pcie_logged_first_allreduce:
                self._pcie_logged_first_allreduce = True
                logger.info(
                    "b12x PCIe oneshot allreduce first dispatch: "
                    "shape=%s dtype=%s bytes=%d capture_stream=%s.",
                    tuple(inp.shape),
                    inp.dtype,
                    inp.numel() * inp.element_size(),
                    self._pcie_runtime_stream() is not None,
                )
            return self._pcie_runtime.all_reduce(
                inp, out=out, stream=self._pcie_runtime_stream()
            )
        if out is None:
            out = torch.empty_like(inp)
        if registered:
            ops.all_reduce(self._ptr, inp, out, 0, 0)
        else:
            ops.all_reduce(
                self._ptr, inp, out, self.buffer_ptrs[self.rank], self.max_size
            )
        return out

    def custom_all_reduce(self, input: torch.Tensor) -> torch.Tensor | None:
        """The main allreduce API that provides support for cuda graph."""
        # When custom allreduce is disabled, this will be None.
        if self.disabled or not self.should_custom_ar(input):
            return None
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self.all_reduce(input, registered=True)
            else:
                # Piecewise CUDA graph execution can run split ops eagerly while
                # graph capture bookkeeping is active. Those ops need a real
                # all-reduce; returning a placeholder is only valid for warmup.
                if _is_piecewise_cudagraph_runtime():
                    return self.all_reduce(input, registered=False)
                # If warm up, mimic the allocation pattern since custom
                # allreduce is out-of-place.
                return torch.empty_like(input)
        else:
            # Note: outside of cuda graph context, custom allreduce incurs a
            # cost of cudaMemcpy, which should be small (<=1% of overall
            # latency) compared to the performance gain of using custom kernels
            return self.all_reduce(input, registered=False)

    def close(self):
        if self._pcie_runtime is not None:
            self._pcie_runtime.close()
            self._pcie_runtime = None
        if not self.disabled and self._ptr:
            if ops is not None:
                ops.dispose(self._ptr)
            self._ptr = 0
            self.free_shared_buffer(self.meta_ptrs, rank=self.rank)
            self.free_shared_buffer(self.buffer_ptrs, rank=self.rank)

    def __del__(self):
        self.close()

    @staticmethod
    def create_shared_buffer(
        size_in_bytes: int,
        group: ProcessGroup | None = None,
        uncached: bool | None = False,
    ) -> list[int]:
        pointer, handle = ops.allocate_shared_buffer_and_handle(size_in_bytes)

        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        handles = [None] * world_size
        dist.all_gather_object(handles, handle, group=group)

        pointers: list[int] = []
        for i, h in enumerate(handles):
            if i == rank:
                pointers.append(pointer)  # type: ignore
            else:
                pointers.append(ops.open_mem_handle(h))
        return pointers

    @staticmethod
    def free_shared_buffer(
        pointers: list[int],
        group: ProcessGroup | None = None,
        rank: int | None = None,
    ) -> None:
        if rank is None:
            rank = dist.get_rank(group=group)
        if ops is not None:
            ops.free_shared_buffer(pointers[rank])
