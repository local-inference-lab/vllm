# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA IPC wrapper implementations.

:class:`CudaIPCWrapper` handles tensors backed by PyTorch's caching
allocator (vLLM default).  :class:`RawCudaIPCWrapper` handles tensors
allocated outside PyTorch (e.g. TRT-LLM's ``cudaMalloc``'d pool).

:class:`CudaIPCWrapper` is bound to ``device_type="cuda"`` via
:attr:`~lmcache.v1.platform.cuda.CudaDeviceSpec.ipc_wrapper_cls`, so the
multiprocess adapter dispatches to it via
:func:`~lmcache.v1.platform.resolve_kv_wrapper_factory`.
:class:`RawCudaIPCWrapper` is not exposed on the spec -- callers (the
TRT-LLM adapter) instantiate it directly.
"""

# Future
from __future__ import annotations

# Standard
from builtins import ExceptionGroup
from typing import ClassVar

# Third Party
import torch

# First Party
from lmcache import torch_device_type
from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper


class CudaIPCWrapper(DeviceIPCWrapper):
    #: ``torch.device.type`` this wrapper handles. Kept as a class-level
    #: constant so external tooling / tests can introspect the binding.
    device_type: ClassVar[str] = "cuda"

    @classmethod
    def wrap(cls, tensor: torch.Tensor) -> DeviceIPCWrapper:
        """Factory used by
        :func:`~lmcache.v1.platform.resolve_kv_wrapper_factory`.

        Args:
            tensor: A CUDA tensor backed by PyTorch's caching allocator.

        Returns:
            A new :class:`CudaIPCWrapper` wrapping ``tensor`` for the
            multiprocess wire.
        """
        # vLLM's sleep-mode allocator uses CUDA VMM allocations. PyTorch does
        # not own those allocations, so _share_cuda_ is invalid; export the
        # underlying CUmemGenericAllocationHandle instead.
        from lmcache.v1.platform.cuda.cumem_ipc import find_cumem_allocation

        allocation = find_cumem_allocation(tensor)
        if allocation is not None:
            return CuMemCudaIPCWrapper(tensor, allocation)
        return cls(tensor)

    def __init__(self, tensor: torch.Tensor) -> None:
        # First Party
        from lmcache.v1.gpu_connector.kv_format.contiguity import (
            attempt_permute_to_contiguous_view,
        )

        # Permute any non-contiguous view (e.g. vLLM's NHD-over-HND) so the
        # shape/stride we encode across IPC reflects the physical layout.
        # Offset is preserved by the wrapper's storage_offset field.
        tensor = attempt_permute_to_contiguous_view(tensor)

        storage = tensor.untyped_storage()
        handle = storage._share_cuda_()

        self.handle = handle
        self.dtype = tensor.dtype
        self.shape = tuple(tensor.shape)
        self.stride = tuple(tensor.stride())
        self.storage_offset = int(tensor.storage_offset())

        device_index = tensor.device.index
        self.device_uuid = self._get_device_uuid(device_index)

    def to_tensor(self) -> torch.Tensor:
        """
        Note:
            This function may break if the accelerator is not initialized.
            We should call ``torch_dev.init()`` before using this function
            (guarded by hasattr since not all backends expose init()).
        """
        device_index = self._get_device_index_from_uuid(self.device_uuid)

        storage = torch.UntypedStorage._new_shared_cuda(  # noqa: SLF001
            device_index, *self.handle[1:]
        )

        t = torch.empty(
            (), device=f"{torch_device_type}:{device_index}", dtype=self.dtype
        )
        t.set_(storage, self.storage_offset, self.shape, self.stride)
        return t


class CuMemCudaIPCWrapper(DeviceIPCWrapper):
    """Serializable view over a vLLM cuMem/VMM allocation.

    The allocation's POSIX shareable handle is passed out-of-band with
    SCM_RIGHTS. The wire descriptor carries only a capability token and view
    metadata, so aliases map one physical allocation exactly once.
    """

    device_type: ClassVar[str] = "cuda"

    def __init__(self, tensor: torch.Tensor, allocation) -> None:
        from lmcache.v1.platform.cuda.cumem_ipc import (
            export_cumem_allocation,
            validate_tensor_view,
        )

        descriptor, lease, allocation_storage_offset = export_cumem_allocation(
            tensor, allocation
        )
        try:
            self.handle = descriptor
            self.dtype = tensor.dtype
            self.shape = tuple(tensor.shape)
            self.stride = tuple(tensor.stride())
            self.storage_offset = int(tensor.storage_offset())
            self.device_uuid = descriptor.device_uuid
            self.physical_storage_nbytes = int(tensor.untyped_storage().nbytes())
            self.allocation_storage_offset_bytes = allocation_storage_offset
            self._lease = lease
            self._mapping = None
            self._tensor = None
            self._closed = False
            validate_tensor_view(
                allocation_size=descriptor.allocation_size,
                storage_offset_bytes=allocation_storage_offset,
                storage_nbytes=self.physical_storage_nbytes,
                shape=self.shape,
                stride=self.stride,
                itemsize=tensor.element_size(),
            )
            self._validate_view_inside_storage(tensor.element_size())
        except BaseException:
            lease.close()
            raise

    def _validate_view_inside_storage(self, itemsize: int) -> None:
        if self.storage_offset < 0:
            raise ValueError("negative tensor storage offset")
        if not self.shape or any(dim == 0 for dim in self.shape):
            last = self.storage_offset
        else:
            last = self.storage_offset + sum(
                (dim - 1) * step
                for dim, step in zip(self.shape, self.stride, strict=True)
            )
        if (last + 1) * itemsize > self.physical_storage_nbytes:
            raise ValueError("tensor view exceeds physical storage extent")

    def __getstate__(self):
        state = self.__dict__.copy()
        # Exporter-only resources never cross the pickle/msgspec wire.
        state["_lease"] = None
        state["_mapping"] = None
        state["_tensor"] = None
        state["_closed"] = False
        return state

    def to_tensor(self) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("cuMem IPC wrapper is closed")
        if self._tensor is not None:
            return self._tensor
        from lmcache.v1.platform.cuda.cumem_ipc import (
            acquire_imported_mapping,
            validate_tensor_view,
        )

        itemsize = torch.empty((), dtype=self.dtype).element_size()
        validate_tensor_view(
            allocation_size=self.handle.allocation_size,
            storage_offset_bytes=self.allocation_storage_offset_bytes,
            storage_nbytes=self.physical_storage_nbytes,
            shape=self.shape,
            stride=self.stride,
            itemsize=itemsize,
        )
        self._validate_view_inside_storage(itemsize)
        mapping = acquire_imported_mapping(self.handle)
        try:
            raw = mapping.as_torch_bytes()
            typed = raw.view(self.dtype)
            absolute_offset = (
                self.allocation_storage_offset_bytes + self.storage_offset * itemsize
            )
            tensor = torch.as_strided(
                typed,
                self.shape,
                self.stride,
                storage_offset=absolute_offset // itemsize,
            )
        except BaseException:
            from lmcache.v1.platform.cuda.cumem_ipc import release_imported_mapping

            release_imported_mapping(self.handle)
            raise
        self._mapping = mapping
        self._tensor = tensor
        return tensor

    def close(self) -> None:
        if self._closed:
            return
        self._tensor = None
        close_errors: list[BaseException] = []
        if self._mapping is not None:
            from lmcache.v1.platform.cuda.cumem_ipc import release_imported_mapping

            try:
                release_imported_mapping(self.handle)
            except BaseException as exc:
                close_errors.append(exc)
            else:
                self._mapping = None
        if self._lease is not None:
            try:
                self._lease.close()
            except BaseException as exc:
                close_errors.append(exc)
            else:
                self._lease = None
        self._closed = self._mapping is None and self._lease is None
        if close_errors:
            raise ExceptionGroup("CuMemCudaIPCWrapper.close failed", close_errors)


class RawCudaIPCWrapper(DeviceIPCWrapper):
    """IPC wrapper for CUDA tensors allocated outside PyTorch's caching
    allocator.

    PyTorch's ``UntypedStorage._share_cuda_()`` only works for tensors
    backed by its own caching allocator. TRT-LLM publishes its KV pool
    via ``at::for_blob`` over a ``cudaMalloc``'d buffer, which raises in
    ``_share_cuda_()``. This subclass bypasses that path: it calls
    ``cudaIpcGetMemHandle`` on the raw data pointer, then reconstructs
    the tensor on the receiving side via ``cudaIpcOpenMemHandle`` plus
    a CuPy ``UnownedMemory`` → DLPack → ``torch`` round-trip.

    Sharing the ``DeviceIPCWrapper`` base (rather than introducing a
    parallel class with its own msgspec ext code) is load-bearing —
    msgspec does not support unions of custom ext-encoded types. With a
    common base, ``KVCache = list[DeviceIPCWrapper]`` type-checks, the
    single ext code 1 round-trips every wrapper, and pickle preserves
    the concrete subclass identity through the wire so ``to_tensor``
    dispatches correctly.
    """

    #: Same ``torch.device.type`` as ``CudaIPCWrapper``, but not exposed
    #: on :attr:`~lmcache.v1.platform.cuda.CudaDeviceSpec.ipc_wrapper_cls`
    #: -- callers (TRT-LLM adapter) instantiate it directly.
    device_type: ClassVar[str] = "cuda"

    def __init__(self, tensor: torch.Tensor) -> None:
        # First Party
        from lmcache.v1.gpu_connector.utils import assert_contiguous

        assert_contiguous(tensor)

        try:
            # Third Party
            from cuda.bindings import runtime as cudart
        except ImportError:
            # Third Party
            from cuda import cudart

        data_ptr = tensor.data_ptr()
        err, ipc_handle = cudart.cudaIpcGetMemHandle(data_ptr)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(
                f"cudaIpcGetMemHandle failed: {err} (ptr=0x{data_ptr:x})"
            )

        # Store only what's needed for reconstruction.
        self._ipc_handle_reserved = bytes(ipc_handle.reserved)
        self._nbytes = tensor.untyped_storage().nbytes()

        # DeviceIPCWrapper interface fields. ``handle`` is unused —
        # ``to_tensor`` is overridden to bypass it — but kept (None) so
        # the base-class equality check has a value to compare.
        self.handle = None
        self.dtype = tensor.dtype
        self.shape = tuple(tensor.shape)
        self.stride = tuple(tensor.stride())
        self.storage_offset = int(tensor.storage_offset())

        device_index = tensor.device.index
        self.device_uuid = self._get_device_uuid(device_index)

    def to_tensor(self) -> torch.Tensor:
        """Reconstruct the tensor in this process via raw CUDA IPC."""
        # Third Party
        import cupy

        try:
            # Third Party
            from cuda.bindings import runtime as cudart
        except ImportError:
            # Third Party
            from cuda import cudart

        device_index = self._get_device_index_from_uuid(self.device_uuid)

        handle = cudart.cudaIpcMemHandle_t()
        handle.reserved = self._ipc_handle_reserved
        err, ptr = cudart.cudaIpcOpenMemHandle(
            handle, cudart.cudaIpcMemLazyEnablePeerAccess
        )
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaIpcOpenMemHandle failed: {err}")

        # Wrap as a flat ``uint8`` CuPy array, DLPack to torch, then view
        # as the original dtype/shape. ``uint8`` avoids dtype-conversion
        # gaps (bfloat16, fp8 have no direct CuPy/NumPy equivalent without
        # ml_dtypes).
        with cupy.cuda.Device(device_index):
            mem = cupy.cuda.UnownedMemory(ptr, self._nbytes, owner=self)
            memptr = cupy.cuda.MemoryPointer(mem, 0)
            cp_flat = cupy.ndarray(self._nbytes, dtype=cupy.uint8, memptr=memptr)

        raw = torch.from_dlpack(cp_flat)
        return raw.view(self.dtype).reshape(self.shape)
