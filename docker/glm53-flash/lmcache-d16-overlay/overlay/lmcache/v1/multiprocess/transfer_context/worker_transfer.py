# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transfer context abstractions for LMCache multiprocess worker adapters."""

# Standard
import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.utils import EngineType, init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.multiprocess.custom_types import (
    EngineDrivenGroupLayout,
    RegisterEngineDrivenContextPayload,
)
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.multiprocess.protocols.engine import RegisterEngineDrivenContextResponse
from lmcache.v1.multiprocess.transfer_context.base import (
    EngineDrivenContext,
    EngineDrivenContextMetadata,
    StoreAdmissionRejected,
    compute_kv_layout,
    create_engine_driven_context,
    gather_paged_kv_to_cpu,
    scatter_cpu_to_paged_kv,
)
from lmcache.v1.platform import get_device_spec, resolve_kv_wrapper_factory
from lmcache.v1.platform.base.event_ipc import (
    EventIPCBackend,
    get_event_ipc_backend,
)
from lmcache.v1.platform.kv_wrap import wrap_kv_caches

logger = init_logger(__name__)

# Environment variable that lets the user override the default routing
# performed by :func:`create_transfer_context`. Accepted values match the
# string values of :class:`MPTransferMode` (``auto`` / ``engine_driven`` /
# ``lmcache_driven``); ``auto`` reproduces the historical device-type-based
# dispatch.
ENV_MP_TRANSFER_MODE = "LMCACHE_MP_TRANSFER_MODE"


# Helper functions
def _supports_async_primitives() -> bool:
    """Probe whether the worker device supports the async store primitives.

    The async engine-driven store path needs a stream, an event exposing
    ``record``/``synchronize``/``wait``, and pinned (page-locked) host memory.
    When any of these is unavailable (e.g. a CPU-only backend), the factory
    falls back to the synchronous :class:`EngineDrivenTransferContext`. This
    dispatch is internal and capability-based; there is no user-facing
    async/sync flag.

    Returns:
        True if all required async primitives are available, else False.
    """
    if not hasattr(torch_dev, "Stream") or not hasattr(torch_dev, "Event"):
        return False
    # CPU-only stub exposes Stream/Event but has no real async capability.
    if hasattr(torch_dev, "is_available") and not torch_dev.is_available():
        return False
    try:
        stream = torch_dev.Stream()
        event = torch_dev.Event()
    except Exception:
        return False
    for attr in ("record", "synchronize", "wait"):
        if not callable(getattr(event, attr, None)):
            del stream, event
            return False
    del stream, event
    try:
        probe = torch.empty(1, dtype=torch.uint8, device="cpu", pin_memory=True)
        del probe
    except (RuntimeError, TypeError):
        return False
    return True


def _build_engine_driven_context() -> "TransferContext":
    """Build the engine-driven context, async when device-capable else sync.

    Routes the ``ENGINE_DRIVEN`` and AUTO branches through a single capability
    check. ``AsyncEngineDrivenTransferContext`` is imported lazily to avoid an
    import cycle and to keep the synchronous path free of stream/event
    dependencies.

    Returns:
        ``AsyncEngineDrivenTransferContext`` when async primitives are
        available, otherwise ``EngineDrivenTransferContext``.
    """
    if _supports_async_primitives():
        # First Party
        from lmcache.v1.multiprocess.transfer_context.async_engine_driven import (
            AsyncEngineDrivenTransferContext,
        )

        logger.info("Using AsyncEngineDrivenTransferContext for store path")
        return AsyncEngineDrivenTransferContext()

    logger.info("Using EngineDrivenTransferContext (sync) for store path")
    return EngineDrivenTransferContext()


class MPTransferMode(str, Enum):
    """Routing mode used by :func:`create_transfer_context`.

    * ``AUTO``: dispatch by ``tensor.device.type`` (CUDA -> lmcache-driven,
      others -> engine-driven). Preserves the historical behaviour.
    * ``ENGINE_DRIVEN``: force :class:`EngineDrivenTransferContext`
      (worker-side gather / scatter copy path).
    * ``LMCACHE_DRIVEN``: force :class:`LMCacheDrivenTransferContext`
      (IPC / SHM zero-copy path). Requires a registered KV-wrapper factory
      for the device.
    """

    AUTO = "auto"
    ENGINE_DRIVEN = "engine_driven"
    LMCACHE_DRIVEN = "lmcache_driven"


def _resolve_mode(mode: "str | MPTransferMode | None") -> MPTransferMode:
    """Coerce ``mode`` into :class:`MPTransferMode`, falling back to env."""
    raw = (
        mode
        if mode is not None
        else os.environ.get(ENV_MP_TRANSFER_MODE, MPTransferMode.AUTO.value)
    )
    if isinstance(raw, MPTransferMode):
        return raw
    try:
        return MPTransferMode(str(raw).lower())
    except ValueError as exc:
        valid = ", ".join(m.value for m in MPTransferMode)
        raise ValueError(
            "Invalid MP transfer mode {!r} (valid: {})".format(raw, valid)
        ) from exc


def _build_lmcache_driven_context(device_type: str) -> "TransferContext":
    """Build a :class:`LMCacheDrivenTransferContext` after capability check."""
    try:
        resolve_kv_wrapper_factory(device_type)
    except ValueError as exc:
        raise ValueError(
            "MP transfer mode 'lmcache_driven' is not supported for device type "
            "{!r}: no KV-cache wrapper factory is registered. "
            "Use mode 'engine_driven' or 'auto' instead.".format(device_type)
        ) from exc
    device_spec = get_device_spec(device_type)
    if device_spec and not device_spec.is_handle_transfer_available():
        raise ValueError(
            "MP transfer mode 'lmcache_driven' is not available for device type "
            "{!r}: required platform capability checks failed. "
            "Use mode 'engine_driven' or 'auto' instead.".format(device_type)
        )
    return LMCacheDrivenTransferContext()


class IPCEvent(Protocol):
    """Protocol for device events used by transport operations."""

    def wait(self, stream: object | None = None) -> None:
        """Make ``stream`` wait for this event (async ordering primitive)."""


SendRequest = Callable[[MessageQueueClient, RequestType, list[object]], MessagingFuture]


def _single_group_block_ids(block_ids: list[list[int]]) -> list[int]:
    """Return the flat block-id list for transports without HMA support."""
    if len(block_ids) != 1:
        raise RuntimeError(
            "engine-driven transfer does not support hybrid KV cache groups"
        )
    return block_ids[0]


@dataclass(frozen=True)
class _EngineDrivenWorkerGroup:
    """Worker-only transfer state for one wire-visible object group."""

    layout: EngineDrivenGroupLayout
    layer_names: tuple[str, ...]
    engine_kv_format: Any


def _validate_group_block_ids(
    groups: Sequence[_EngineDrivenWorkerGroup],
    block_ids: list[list[int]],
) -> int:
    """Validate complete group coverage and return the common chunk count."""
    chunk_counts: list[int] = []
    for group in groups:
        engine_group_idx = group.layout.engine_group_idx
        if engine_group_idx >= len(block_ids):
            raise ValueError(f"missing block IDs for engine group {engine_group_idx}")
        group_ids = block_ids[engine_group_idx]
        blocks_per_chunk = group.layout.blocks_per_chunk
        if blocks_per_chunk <= 0 or len(group_ids) % blocks_per_chunk:
            raise ValueError(
                f"engine group {engine_group_idx} has {len(group_ids)} block IDs; "
                f"expected a multiple of {blocks_per_chunk}"
            )
        chunk_counts.append(len(group_ids) // blocks_per_chunk)
    if not chunk_counts or len(set(chunk_counts)) != 1:
        raise ValueError(
            f"engine-driven groups do not cover one common chunk prefix: {chunk_counts}"
        )
    return chunk_counts[0]


def _get_kv_device(kv_caches: dict[str, torch.Tensor]) -> torch.device:
    """Return the device shared by a non-empty KV-cache mapping.

    Args:
        kv_caches: Worker KV-cache tensors keyed by layer name.

    Returns:
        The device of the first KV-cache tensor.

    Raises:
        ValueError: If ``kv_caches`` is empty.
    """
    if not kv_caches:
        raise ValueError("LMCache-driven transfer requires at least one KV cache")
    return next(iter(kv_caches.values())).device


class TransferContext(ABC):
    """Abstract transport layer for worker-side KV transfer.

    Concrete implementations encapsulate how worker-side store/retrieve
    operations are transmitted to the multiprocess server. Device-handle paths
    return event-aware futures backed by MQ requests, while CPU paths may perform
    gather/scatter synchronously and return already-resolved futures.
    """

    @abstractmethod
    def register(
        self,
        instance_id: int,
        _kv_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        blocks_in_chunk: int,
        mq_client: MessageQueueClient,
        mq_timeout: float,
        send_request: SendRequest,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
        engine_type: EngineType = EngineType.VLLM,
    ) -> None:
        """Register KV caches with the server and wait for ACK.

        Args:
            instance_id: Worker process instance identifier.
            kv_caches: Worker KV cache tensors keyed by layer name.
            model_name: Model name used by cache keys.
            world_size: KV world size.
            blocks_in_chunk: Number of vLLM blocks per LMCache chunk.
            mq_client: Message queue client used to communicate with server.
            mq_timeout: Timeout in seconds for synchronous request wait.
            send_request: Request sender callable used to issue MQ requests.
            layout_hints: Optional inference-engine-provided layout hints.
            engine_group_infos: LMCache-owned engine KV cache group metadata.
            engine_type: Serving engine that produced the caches. Only
                consumed by the handle path; adapters should pass their
                own :class:`EngineType` so this transport stays engine-
                neutral. Defaults to :attr:`EngineType.VLLM` for
                backwards compatibility.

        Raises:
            TimeoutError: If server registration does not complete before
                ``mq_timeout``.
            RuntimeError: If a concrete context cannot initialize.
        """

    def register_q(
        self,
        instance_id: int,
        q_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        blocks_in_chunk: int,
        mq_client: MessageQueueClient,
        mq_timeout: float,
        send_request: SendRequest,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
    ) -> None:
        """Register the paged Q ring with the server under the same worker
        instance_id but different model_name (model_name##query).

        Args:
            instance_id: Worker process instance identifier.
            q_caches: Worker Q cache tensors keyed by layer name.
            model_name: Model name used by cache keys (model_name##query).
            world_size: KV world size.
            blocks_in_chunk: Number of Q ring blocks per LMCache chunk.
            mq_client: Message queue client used to communicate with server.
            mq_timeout: Timeout in seconds for synchronous request wait.
            send_request: Request sender callable used to issue MQ requests.
            layout_hints: Optional inference-engine-provided layout hints.
            engine_group_infos: LMCache-owned engine KV cache group metadata.

        Raises:
            NotImplementedError: If the concrete transport does not support the
                Q ring (now only lmcache-driven).
            TimeoutError: If server registration does not complete before
                ``mq_timeout``.
            RuntimeError: If a concrete context cannot initialize.
        """
        raise NotImplementedError(
            "Q ring registration is not supported by this transfer context"
        )

    def submit_q_store(
        self,
        request_id: str,
        key: Any,
        instance_id: int,
        q_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent,
        blocks_in_chunk: int,
    ) -> MessagingFuture:
        """Submit a Q ring store request and return a completion future.

        Args:
            request_id: External request identifier.
            key: LMCache key for the Q store range (query-specific model_name).
            instance_id: Worker process instance identifier (shared with KV).
            q_caches: Q ring tensors keyed by layer name.
            block_ids: Q ring block IDs to store, indexed by LMCache KV group id.
            event: Synchronization event object.
            blocks_in_chunk: Number of Q ring blocks per LMCache chunk.

        Returns:
            A future compatible with adapter-side ``query()``/``result()`` flow.

        Raises:
            NotImplementedError: If the concrete transport does not support the
                Q ring (only the lmcache-driven path does).
            RuntimeError: If register_q() was not called first.
        """
        raise NotImplementedError(
            "Q ring store is not supported by this transfer context"
        )

    @abstractmethod
    def submit_store(
        self,
        request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent,
        blocks_in_chunk: int,
    ) -> MessagingFuture:
        """Submit a store request and return a completion future.

        Args:
            request_id: External request identifier.
            key: LMCache key object for the store range.
            instance_id: Worker process instance identifier.
            kv_caches: Worker KV cache tensors keyed by layer name.
            block_ids: vLLM block IDs to store, indexed by LMCache KV group id.
            event: Synchronization event object.
            blocks_in_chunk: Number of vLLM blocks per LMCache chunk.

        Returns:
            A future compatible with adapter-side ``query()``/``result()`` flow.

        Raises:
            RuntimeError: If register() was not called first.
        """

    @abstractmethod
    def submit_retrieve(
        self,
        request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent,
        blocks_in_chunk: int,
        skip_first_n_tokens: int = 0,
    ) -> MessagingFuture:
        """Submit a retrieve request and return a completion future.

        Args:
            request_id: External request identifier.
            key: LMCache key object for the retrieve range.
            instance_id: Worker process instance identifier.
            kv_caches: Worker KV cache tensors keyed by layer name.
            block_ids: vLLM block IDs to retrieve into, indexed by LMCache KV
                group id.
            event: Synchronization event object.
            blocks_in_chunk: Number of vLLM blocks per LMCache chunk.
            skip_first_n_tokens: Number of initial tokens to skip when writing.

        Returns:
            A future compatible with adapter-side ``query()``/``result()`` flow.

        Raises:
            RuntimeError: If register() was not called first.
        """

    @abstractmethod
    def close(self) -> None:
        """Release resources held by this context."""

    @abstractmethod
    def flush_inflight_stores(self) -> None:
        """Synchronize any in-flight gather operations.

        Subclasses must implement this method. Contexts with no deferred
        operations should implement it as a no-op. Async contexts that
        defer GPU->CPU gather work must block until all in-flight stores
        have completed, so that vLLM cannot overwrite paged KV blocks
        before they are read.
        """


class LMCacheDrivenTransferContext(TransferContext):
    """LMCache-driven IPC + MQ future transport context.

    In this mode the serving engine provides device handles (accelerator IPC,
    or SHM wrappers for CPU with IPC-like semantics) and the LMCache server
    performs direct device-side data transfer.
    """

    def __init__(self) -> None:
        self._mq_client: MessageQueueClient | None = None
        self._send_request: SendRequest | None = None
        self._device: torch.device | None = None
        self._event_backend: EventIPCBackend | None = None
        self._ipc_wrappers: list[Any] = []

    def register(
        self,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        _blocks_in_chunk: int,
        mq_client: MessageQueueClient,
        mq_timeout: float,
        send_request: SendRequest,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
        engine_type: EngineType = EngineType.VLLM,
    ) -> None:
        """Register the worker KV cache with the LMCache server.

        Args:
            instance_id: Worker process instance identifier.
            kv_caches: Worker KV-cache tensors keyed by layer name.
            model_name: Model identifier used by the server.
            world_size: Tensor-parallel world size.
            _blocks_in_chunk: Engine blocks per LMCache chunk.
            mq_client: Message-queue client used for requests.
            mq_timeout: Timeout for the registration response.
            send_request: Request sender used by this context.
            layout_hints: Optional KV-layout metadata.
            engine_group_infos: Optional engine KV-group metadata.
            engine_type: Serving engine that produced the caches.

        Raises:
            RuntimeError: If event IPC is unsupported for the KV-cache device.
            ValueError: If ``kv_caches`` is empty.
        """
        device = _get_kv_device(kv_caches)
        event_backend = get_event_ipc_backend(device)
        event_backend.check_event_support(device)

        self._mq_client = mq_client
        self._send_request = send_request
        wrappers = wrap_kv_caches(kv_caches)
        if any(type(wrapper).__name__ == "CuMemCudaIPCWrapper" for wrapper in wrappers):
            logger.info("LMCache MP runtime mode verified: cumem_cuda_ipc")
        try:
            future = send_request(
                mq_client,
                RequestType.REGISTER_KV_CACHE,
                [
                    instance_id,
                    wrappers,
                    model_name,
                    world_size,
                    engine_type,
                    layout_hints,
                    list(engine_group_infos),
                ],
            )
            future.result(timeout=mq_timeout)
        except BaseException:
            for wrapper in wrappers:
                close = getattr(wrapper, "close", None)
                if close is not None:
                    close()
            raise
        self._ipc_wrappers = wrappers
        self._device = device
        self._event_backend = event_backend

    def register_q(
        self,
        instance_id: int,
        q_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        _blocks_in_chunk: int,
        mq_client: MessageQueueClient,
        mq_timeout: float,
        send_request: SendRequest,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
    ) -> None:
        self._mq_client = mq_client
        self._send_request = send_request
        future = send_request(
            mq_client,
            RequestType.REGISTER_Q_CACHE,
            [
                instance_id,
                wrap_kv_caches(q_caches),
                model_name,
                world_size,
                EngineType.VLLM,
                layout_hints,
                list(engine_group_infos),
            ],
        )
        future.result(timeout=mq_timeout)

    def submit_store(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent,
        _blocks_in_chunk: int,
    ) -> MessagingFuture:
        """Submit a handle-based store ordered by ``event``.

        Args:
            _request_id: External request identifier (unused by this transport).
            key: LMCache key for the store range.
            instance_id: Worker process instance identifier.
            _kv_caches: Worker KV-cache tensors accepted for interface
                consistency; the registered device is reused.
            block_ids: Engine block IDs indexed by LMCache KV group.
            event: Producer event that orders reads of the engine KV cache.
            _blocks_in_chunk: Engine blocks per chunk (unused by this transport).

        Returns:
            A device-event-aware future for the server response.

        Raises:
            RuntimeError: If the context is not registered or event IPC is
                unsupported.
        """
        if (
            self._mq_client is None
            or self._send_request is None
            or self._device is None
            or self._event_backend is None
        ):
            raise RuntimeError(
                "LMCache-driven transfer context is not registered. "
                "Call register() before submit_store()."
            )
        event_ipc_handle = self._event_backend.export_event(event, self._device)
        return self._send_request(
            self._mq_client,
            RequestType.STORE,
            [key, instance_id, block_ids, event_ipc_handle],
        ).to_device_future(device=self._device)

    def submit_q_store(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        _q_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent,
        _blocks_in_chunk: int,
    ) -> MessagingFuture:
        if (
            self._mq_client is None
            or self._send_request is None
            or self._device is None
            or self._event_backend is None
        ):
            raise RuntimeError(
                "LMCache-driven transfer context is not registered. "
                "Call register() before submit_q_store()."
            )
        event_ipc_handle = self._event_backend.export_event(event, self._device)
        return self._send_request(
            self._mq_client,
            RequestType.STORE_Q,
            [key, instance_id, block_ids, event_ipc_handle],
        ).to_device_future(device=self._device)

    def submit_retrieve(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        _kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent,
        _blocks_in_chunk: int,
        skip_first_n_tokens: int = 0,
    ) -> MessagingFuture:
        """Submit a handle-based retrieve ordered by ``event``.

        Args:
            _request_id: External request identifier (unused by this transport).
            key: LMCache key for the retrieve range.
            instance_id: Worker process instance identifier.
            _kv_caches: Worker KV-cache tensors accepted for interface
                consistency; the registered device is reused.
            block_ids: Engine block IDs indexed by LMCache KV group.
            event: Producer event that orders writes to the engine KV cache.
            _blocks_in_chunk: Engine blocks per chunk (unused by this transport).
            skip_first_n_tokens: Initial tokens the server must not overwrite.

        Returns:
            A device-event-aware future for the server response.

        Raises:
            RuntimeError: If the context is not registered or event IPC is
                unsupported.
        """
        if (
            self._mq_client is None
            or self._send_request is None
            or self._device is None
            or self._event_backend is None
        ):
            raise RuntimeError(
                "LMCache-driven transfer context is not registered. "
                "Call register() before submit_retrieve()."
            )
        event_ipc_handle = self._event_backend.export_event(event, self._device)
        return self._send_request(
            self._mq_client,
            RequestType.RETRIEVE,
            [key, instance_id, block_ids, event_ipc_handle, skip_first_n_tokens],
        ).to_device_future(device=self._device)

    def close(self) -> None:
        """Release the message queue and cached event-backend state."""
        for wrapper in self._ipc_wrappers:
            close = getattr(wrapper, "close", None)
            if close is not None:
                close()
        self._ipc_wrappers.clear()
        self._mq_client = None
        self._send_request = None
        self._device = None
        self._event_backend = None

    def flush_inflight_stores(self) -> None:
        pass


class EngineDrivenTransferContext(TransferContext):
    """Engine-driven transfer context for non-CUDA workers.

    In this mode the engine (worker side) owns the data movement: the
    worker adapter gathers/packs KV into CPU buffers, commits via
    message-queue, and the server side persists/rehydrates from storage.
    """

    def __init__(self) -> None:
        self._engine_driven_context: EngineDrivenContext | None = None
        self._layout_hints: LayoutHints | None = None
        self._engine_kv_format: Any = None
        self._worker_groups: tuple[_EngineDrivenWorkerGroup, ...] = ()

    def _gather_group_payloads(
        self,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        out_buffers: list[list[torch.Tensor]] | None = None,
        group_chunk_indices: list[list[int]] | None = None,
    ) -> list[list[torch.Tensor]]:
        """Gather every object group in stable registration order."""
        num_chunks = _validate_group_block_ids(self._worker_groups, block_ids)
        if out_buffers is not None and len(out_buffers) != len(self._worker_groups):
            raise ValueError("store buffers do not cover all groups")
        if group_chunk_indices is not None and len(group_chunk_indices) != len(
            self._worker_groups
        ):
            raise ValueError("store chunk indices do not cover all groups")

        payloads: list[list[torch.Tensor]] = []
        for group_idx, group in enumerate(self._worker_groups):
            ids = block_ids[group.layout.engine_group_idx]
            group_kv = {name: kv_caches[name] for name in group.layer_names}
            out = out_buffers[group_idx] if out_buffers is not None else None
            indices = (
                group_chunk_indices[group_idx]
                if group_chunk_indices is not None
                else None
            )
            if indices is not None and any(i < 0 or i >= num_chunks for i in indices):
                raise ValueError(f"group {group_idx} has invalid chunk indices")
            payloads.append(
                gather_paged_kv_to_cpu(
                    group_kv,
                    ids,
                    group.layout.blocks_per_chunk,
                    layout_hints=self._layout_hints,
                    engine_kv_format=group.engine_kv_format,
                    out=out,
                    chunk_indices=indices,
                )
            )
        return payloads

    def _scatter_group_payloads(
        self,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        group_payloads: list[list[torch.Tensor]],
        skip_first_n_tokens: int,
    ) -> None:
        """Validate all groups before scattering any retrieved state."""
        num_chunks = _validate_group_block_ids(self._worker_groups, block_ids)
        if len(group_payloads) != len(self._worker_groups):
            raise ValueError("retrieve payload does not cover all groups")
        for group_idx, (group, payloads) in enumerate(
            zip(self._worker_groups, group_payloads, strict=True)
        ):
            if len(payloads) != num_chunks:
                raise ValueError(
                    f"group {group_idx} returned {len(payloads)} chunks; "
                    f"expected {num_chunks}"
                )
            expected_shape = torch.Size(group.layout.shape)
            expected_dtype = getattr(torch, group.layout.dtype_str)
            if any(
                payload.shape != expected_shape or payload.dtype != expected_dtype
                for payload in payloads
            ):
                raise ValueError(
                    f"group {group_idx} retrieve payload layout does not match "
                    "registration"
                )
            group_kv = {name: kv_caches[name] for name in group.layer_names}
            # First Party
            from lmcache.v1.gpu_connector.utils import (
                get_num_blocks,
                normalize_kv_and_discover_format,
            )

            _, normalized = normalize_kv_and_discover_format(
                list(group_kv.values()),
                EngineType.VLLM,
                layout_hints=self._layout_hints,
            )
            num_blocks = get_num_blocks(normalized, group.engine_kv_format)
            ids = block_ids[group.layout.engine_group_idx]
            if any(block_id < 0 or block_id >= num_blocks for block_id in ids):
                raise ValueError(
                    f"group {group_idx} destination block ID is out of range"
                )

        for group, payloads in zip(self._worker_groups, group_payloads, strict=True):
            group_kv = {name: kv_caches[name] for name in group.layer_names}
            physical_block_size = (
                group.layout.shape[-2] // group.layout.blocks_per_chunk
            )
            group_skip = (
                skip_first_n_tokens
                * physical_block_size
                // group.layout.tokens_per_block
            )
            scatter_cpu_to_paged_kv(
                group_kv,
                block_ids[group.layout.engine_group_idx],
                payloads,
                group.layout.blocks_per_chunk,
                skip_first_n_tokens=group_skip,
                layout_hints=self._layout_hints,
                engine_kv_format=group.engine_kv_format,
            )

    @property
    def engine_driven_context(self) -> EngineDrivenContext:
        """Return the underlying SHM/pickle context created by ``register``.

        Raises:
            RuntimeError: If accessed before ``register`` has run.
        """
        if self._engine_driven_context is None:
            raise RuntimeError(
                "EngineDrivenTransferContext is not registered, call register() first."
            )
        return self._engine_driven_context

    def register(
        self,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        blocks_in_chunk: int,
        mq_client: MessageQueueClient,
        mq_timeout: float,
        send_request: SendRequest,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
        engine_type: EngineType = EngineType.VLLM,
    ) -> None:
        """Register KV caches with the non-GPU context server.

        Each ``engine_group_infos`` entry becomes one stable wire/object group.
        The object carries only kernel-equivalent layers and selects its own
        engine block-ID address space.
        """
        del engine_type
        layer_names = tuple(kv_caches)
        if not layer_names:
            raise ValueError("kv_caches is empty")
        group_infos = list(engine_group_infos) or [
            EngineGroupInfo(
                engine_group_id=0,
                layer_indices=tuple(range(len(layer_names))),
            )
        ]
        representative_indices = tuple(group_infos[0].layer_indices)
        if not representative_indices:
            raise ValueError("engine-driven group 0 has no layers")
        representative_kv = {
            layer_names[i]: kv_caches[layer_names[i]] for i in representative_indices
        }
        (
            block_size,
            num_layers,
            hidden_dim_size,
            dtype_str,
            engine_kv_format,
            kv_size,
        ) = compute_kv_layout(representative_kv, layout_hints=layout_hints)
        self._layout_hints = layout_hints
        self._engine_kv_format = engine_kv_format

        logical_chunk_tokens = blocks_in_chunk * (
            engine_group_infos[0].tokens_per_block
            if engine_group_infos and engine_group_infos[0].tokens_per_block > 0
            else block_size
        )
        worker_groups: list[_EngineDrivenWorkerGroup] = []
        for object_group_id, info in enumerate(group_infos):
            indices = tuple(info.layer_indices)
            if not indices:
                raise ValueError(f"engine-driven group {object_group_id} has no layers")
            try:
                group_kv = {layer_names[i]: kv_caches[layer_names[i]] for i in indices}
            except IndexError as exc:
                raise ValueError(
                    f"engine-driven group {object_group_id} has invalid layer indices"
                ) from exc
            (
                group_block_size,
                group_num_layers,
                group_hidden_dim,
                group_dtype_str,
                group_format,
                group_kv_size,
            ) = compute_kv_layout(group_kv, layout_hints=layout_hints)
            tokens_per_block = (
                info.tokens_per_block if info.tokens_per_block > 0 else group_block_size
            )
            if logical_chunk_tokens % tokens_per_block:
                raise ValueError(
                    f"LMCache chunk size {logical_chunk_tokens} does not align to "
                    f"group {object_group_id} tokens_per_block {tokens_per_block}"
                )
            group_blocks_per_chunk = logical_chunk_tokens // tokens_per_block
            group_slots_per_chunk = group_blocks_per_chunk * group_block_size
            group_shape = (
                (group_num_layers, group_slots_per_chunk, group_hidden_dim)
                if group_kv_size == 1
                else (2, group_num_layers, group_slots_per_chunk, group_hidden_dim)
            )
            group_layout = EngineDrivenGroupLayout(
                object_group_id=object_group_id,
                engine_group_idx=info.engine_group_id,
                layer_indices=indices,
                tokens_per_block=tokens_per_block,
                blocks_per_chunk=group_blocks_per_chunk,
                shape=group_shape,
                dtype_str=group_dtype_str,
            )
            worker_groups.append(
                _EngineDrivenWorkerGroup(
                    layout=group_layout,
                    layer_names=tuple(layer_names[i] for i in indices),
                    engine_kv_format=group_format,
                )
            )
        self._worker_groups = tuple(worker_groups)

        # The wire field is named use_mla but only drives the object plane
        # count: single-plane (kv_size == 1) covers MLA and fused-K/V formats.
        use_mla_flag = kv_size == 1
        shape = (
            torch.Size([num_layers, blocks_in_chunk * block_size, hidden_dim_size])
            if use_mla_flag
            else torch.Size(
                [2, num_layers, blocks_in_chunk * block_size, hidden_dim_size]
            )
        )
        dtype = getattr(torch, dtype_str)
        layout_desc = MemoryLayoutDesc(shapes=[shape], dtypes=[dtype])

        future = send_request(
            mq_client,
            RequestType.REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT,
            [
                RegisterEngineDrivenContextPayload(
                    instance_id=instance_id,
                    model_name=model_name,
                    world_size=world_size,
                    block_size=block_size,
                    num_layers=num_layers,
                    hidden_dim_size=hidden_dim_size,
                    dtype_str=dtype_str,
                    use_mla=use_mla_flag,
                    group_layouts=tuple(group.layout for group in worker_groups),
                )
            ],
        )
        response = future.result(timeout=mq_timeout)
        shm_name = ""
        pool_size = 0
        if isinstance(response, RegisterEngineDrivenContextResponse):
            shm_name = response.shm_name
            pool_size = response.pool_size

        metadata = EngineDrivenContextMetadata(
            layout_desc=layout_desc,
            block_size=block_size,
            use_mla=use_mla_flag,
            group_layouts=tuple(group.layout for group in worker_groups),
        )
        self._engine_driven_context = create_engine_driven_context(
            metadata,
            mq_client,
            mq_timeout,
            shm_name=shm_name,
            pool_size=pool_size,
        )
        supported_transfer_mode = "SHM" if shm_name and pool_size > 0 else "pickle"
        logger.info(
            "Worker non-GPU transfer context registered (instance_id=%d, mode=%s)",
            instance_id,
            supported_transfer_mode,
        )

    def submit_store(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        _event: IPCEvent,
        blocks_in_chunk: int,
    ) -> MessagingFuture:
        if self._engine_driven_context is None:
            raise RuntimeError(
                "Engine-driven transfer context is not registered. "
                "Call register() before submit_store()."
            )

        torch_dev.synchronize()
        try:
            result = self._engine_driven_context.prepare_store(key, instance_id)
        except StoreAdmissionRejected as exc:
            logger.warning(
                "Skipping synchronous engine-driven cache store: reason=%s",
                exc.reason,
            )
            future: MessagingFuture[bool] = MessagingFuture()
            future.set_result(False)
            return future
        out_buffers, group_chunk_indices = (
            result if result is not None else (None, None)
        )
        # All chunks already in cache — nothing to gather or commit.
        if group_chunk_indices is not None and not any(group_chunk_indices):
            future: MessagingFuture[bool] = MessagingFuture()
            future.set_result(True)
            return future
        aborted = False
        try:
            cpu_chunks = self._gather_group_payloads(
                kv_caches,
                block_ids,
                out_buffers=out_buffers,
                group_chunk_indices=group_chunk_indices,
            )
            if out_buffers is not None:
                # SHM path uses async device->CPU copies; complete them before commit.
                torch_dev.synchronize()
            ok = self._engine_driven_context.commit_store(key, instance_id, cpu_chunks)
        except Exception:
            if out_buffers is not None:
                # A gather can fail after enqueuing D2H work. Drain it before
                # allowing vLLM to reuse the source blocks or SHM slots.
                torch_dev.synchronize()
                self._engine_driven_context.abort_store(key, instance_id)
                aborted = True
            logger.exception("Synchronous engine-driven store failed")
            ok = False
        if out_buffers is not None and not ok and not aborted:
            self._engine_driven_context.abort_store(key, instance_id)

        future = MessagingFuture()
        future.set_result(ok)
        return future

    def submit_retrieve(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        _event: IPCEvent,
        blocks_in_chunk: int,
        skip_first_n_tokens: int = 0,
    ) -> MessagingFuture:
        if self._engine_driven_context is None:
            raise RuntimeError(
                "Engine-driven transfer context is not registered. "
                "Call register() before submit_retrieve()."
            )

        src_buffers = self._engine_driven_context.prepare_retrieve(key, instance_id)
        ok = src_buffers is not None
        if src_buffers is not None:
            try:
                self._scatter_group_payloads(
                    kv_caches,
                    block_ids,
                    src_buffers,
                    skip_first_n_tokens=skip_first_n_tokens,
                )
            except (RuntimeError, ValueError, TypeError, IndexError):
                logger.exception("Failed to scatter retrieved CPU context chunks")
                ok = False
            # SHM path: ensure all device writes are complete before releasing
            # the SHM slot (server may immediately reuse it after commit_retrieve).
            torch_dev.synchronize()
        self._engine_driven_context.commit_retrieve(key, instance_id)

        future: MessagingFuture[bool] = MessagingFuture()
        future.set_result(ok)
        return future

    def close(self) -> None:
        if self._engine_driven_context is not None:
            self._engine_driven_context.close()
            self._engine_driven_context = None

    def flush_inflight_stores(self) -> None:
        pass


def create_transfer_context(
    kv_caches: dict[str, torch.Tensor],
    mode: "str | MPTransferMode | None" = None,
    **_kwargs: Any,
) -> TransferContext:
    """Create a transfer context from KV cache device type.

    The device check is intentionally centralized here. Routing can be
    overridden via the ``mode`` argument or the ``LMCACHE_MP_TRANSFER_MODE``
    environment variable; see :class:`MPTransferMode` for accepted values.

    Args:
        kv_caches: Worker KV cache tensors keyed by layer name.
        mode: Optional routing override. When ``None`` the value of
            ``LMCACHE_MP_TRANSFER_MODE`` is consulted, defaulting to
            :attr:`MPTransferMode.AUTO`.
        **kwargs: Unused placeholder for forward-compatible factory extension.

    Returns:
        A concrete :class:`TransferContext` implementation.

    Raises:
        ValueError: If ``kv_caches`` is empty, has mixed device types, the
            requested mode string is unknown, or the requested mode is not
            supported for the worker device.
    """
    if not kv_caches:
        raise ValueError("kv_caches is empty")
    device_types = {tensor.device.type for tensor in kv_caches.values()}
    if len(device_types) != 1:
        raise ValueError(
            f"All KV cache tensors must share one device type, got {device_types}"
        )
    device_type = next(iter(device_types))
    resolved_mode = _resolve_mode(mode)
    logger.info(
        "Creating transfer context (device_type=%s, mode=%s)",
        device_type,
        resolved_mode.value,
    )
    if resolved_mode is MPTransferMode.LMCACHE_DRIVEN:
        return _build_lmcache_driven_context(device_type)
    if resolved_mode is MPTransferMode.ENGINE_DRIVEN:
        return _build_engine_driven_context()
    # AUTO: dispatch by device type (CUDA -> handle path, else -> data path).
    if device_type == "cuda":
        return LMCacheDrivenTransferContext()
    return _build_engine_driven_context()
