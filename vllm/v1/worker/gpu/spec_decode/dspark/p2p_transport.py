# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Peer-memory transport between the verifier and a draft server on another GPU.

The draft server exports CUDA IPC handles for a context ring (rows the
verifier appends to the draft KV cache) and for reply slots (the top-k draft
logits of one proposal). The verifier's rank 0 opens the handles once, then
moves the bulk payloads with device-to-device copies over the PCIe fabric:
context rows are pushed into a ring slot before the proposal header is sent,
and the reply logits are pulled from a reply slot after the reply header
arrived. The ZMQ channel carries only headers, positions and tokens.

Both processes must enumerate the other's GPU (``CUDA_VISIBLE_DEVICES``);
``cudaIpcOpenMemHandle`` maps the remote allocation into this process with
lazy peer access.
"""

from __future__ import annotations

import base64
import contextlib
import ctypes
from dataclasses import dataclass
from typing import Any

import torch

from vllm.distributed.device_communicators.cuda_wrapper import find_loaded_library

P2P_CAPABILITY = "dflash_p2p_v1"
TOPK_LOGITS_P2P_CAPABILITY = "dflash_logits_topk_p2p_v1"

_CUDA_IPC_HANDLE_BYTES = 64
_CUDA_MEMCPY_DEFAULT = 4
_CUDA_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 1


class _CudaIpcMemHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_byte * _CUDA_IPC_HANDLE_BYTES)]


def _ipc_handle_from_share(blob: bytes) -> bytes:
    """Extract the ``cudaIpcMemHandle_t`` from a storage's share blob.

    ``UntypedStorage._share_cuda_`` returns a two-byte header (format
    version, allocation kind) followed by the handle; only allocations the
    CUDA caching allocator made with ``cudaMalloc`` (kind ``c``) carry an IPC
    memory handle. Expandable segments (kind ``e``) export a file descriptor
    instead and cannot be mapped with ``cudaIpcOpenMemHandle``.
    """
    if len(blob) == _CUDA_IPC_HANDLE_BYTES:
        return bytes(blob)
    if len(blob) == _CUDA_IPC_HANDLE_BYTES + 2 and blob[1:2] == b"c":
        return bytes(blob[2:])
    raise ValueError(
        "exported storage is not a cudaMalloc allocation (set "
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False in the exporting "
        f"process); share blob of {len(blob)} bytes, kind {blob[1:2]!r}"
    )


class _Cudart:
    """The CUDA runtime entry points the transport needs, bound with ctypes."""

    def __init__(self) -> None:
        path = find_loaded_library("libcudart")
        if path is None:
            raise RuntimeError("libcudart is not loaded")
        lib = ctypes.CDLL(path)
        lib.cudaIpcOpenMemHandle.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            _CudaIpcMemHandle,
            ctypes.c_uint,
        ]
        lib.cudaIpcOpenMemHandle.restype = ctypes.c_int
        lib.cudaIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
        lib.cudaIpcCloseMemHandle.restype = ctypes.c_int
        lib.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.cudaMemcpyAsync.restype = ctypes.c_int
        lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        lib.cudaGetErrorString.restype = ctypes.c_char_p
        self._lib = lib

    def check(self, err: int, what: str) -> None:
        if err != 0:
            message = self._lib.cudaGetErrorString(err).decode()
            raise RuntimeError(f"{what} failed: {message} ({err})")

    def open(self, handle: bytes) -> int:
        if len(handle) != _CUDA_IPC_HANDLE_BYTES:
            raise ValueError(f"CUDA IPC handle must be {_CUDA_IPC_HANDLE_BYTES} bytes")
        raw = _CudaIpcMemHandle()
        ctypes.memmove(raw.reserved, handle, _CUDA_IPC_HANDLE_BYTES)
        ptr = ctypes.c_void_p()
        self.check(
            self._lib.cudaIpcOpenMemHandle(
                ctypes.byref(ptr), raw, _CUDA_IPC_MEM_LAZY_ENABLE_PEER_ACCESS
            ),
            "cudaIpcOpenMemHandle",
        )
        assert ptr.value is not None
        return int(ptr.value)

    def close(self, ptr: int) -> None:
        self.check(self._lib.cudaIpcCloseMemHandle(ctypes.c_void_p(ptr)), "close")

    def memcpy_async(self, dst: int, src: int, nbytes: int, stream: int) -> None:
        if nbytes == 0:
            return
        self.check(
            self._lib.cudaMemcpyAsync(
                ctypes.c_void_p(dst),
                ctypes.c_void_p(src),
                ctypes.c_size_t(nbytes),
                _CUDA_MEMCPY_DEFAULT,
                ctypes.c_void_p(stream),
            ),
            "cudaMemcpyAsync",
        )


def export_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    """Describe a contiguous CUDA tensor for a peer process.

    Returns the base64 IPC handle of the tensor's allocation, the byte
    offset of the tensor within it, and the tensor's geometry. The exporting
    process must keep the tensor alive while a peer uses the handle.
    """
    if not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError("only contiguous CUDA tensors can be exported")
    storage = tensor.untyped_storage()
    shared = storage._share_cuda_()
    handle = _ipc_handle_from_share(shared[1])
    storage_offset_bytes = int(shared[3])
    offset = storage_offset_bytes + tensor.storage_offset() * tensor.element_size()
    return {
        "handle": base64.b64encode(handle).decode("ascii"),
        "offset_bytes": offset,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "nbytes": tensor.numel() * tensor.element_size(),
    }


@dataclass
class PeerTensor:
    """A peer-process tensor mapped into this process."""

    base_ptr: int
    offset_bytes: int
    shape: tuple[int, ...]
    dtype: torch.dtype
    nbytes: int

    @property
    def data_ptr(self) -> int:
        return self.base_ptr + self.offset_bytes

    def element_size(self) -> int:
        return torch.empty(0, dtype=self.dtype).element_size()

    def row_stride_bytes(self) -> int:
        """Bytes of one leading-dimension slice."""
        inner = 1
        for size in self.shape[1:]:
            inner *= size
        return inner * self.element_size()


class DraftPeerBuffers:
    """The draft server's context ring and reply slots, mapped for rank 0."""

    def __init__(self, info: dict[str, Any]) -> None:
        self._cudart = _Cudart()
        self._opened: list[int] = []
        try:
            self.context = self._open(info["context"])
            self.values = self._open(info["values"])
            self.indices = self._open(info["indices"])
        except Exception:
            self.close()
            raise
        self.context_slots = int(self.context.shape[0])
        self.context_rows = int(self.context.shape[1])
        self.context_width = int(self.context.shape[2])
        self.reply_slots = int(self.values.shape[0])
        self.max_requests = int(self.values.shape[1])
        self.num_steps = int(self.values.shape[2])
        self.topk = int(self.values.shape[3])
        if tuple(self.indices.shape) != tuple(self.values.shape):
            raise ValueError("reply value and index slots must share a shape")

    def _open(self, spec: dict[str, Any]) -> PeerTensor:
        handle = base64.b64decode(spec["handle"])
        base = self._cudart.open(handle)
        self._opened.append(base)
        dtype = getattr(torch, spec["dtype"])
        return PeerTensor(
            base_ptr=base,
            offset_bytes=int(spec["offset_bytes"]),
            shape=tuple(int(v) for v in spec["shape"]),
            dtype=dtype,
            nbytes=int(spec["nbytes"]),
        )

    def close(self) -> None:
        for base in self._opened:
            with contextlib.suppress(Exception):
                self._cudart.close(base)
        self._opened.clear()

    def push_context(self, slot: int, context: torch.Tensor, stream: int) -> None:
        """Copy ``context`` rows into ring slot ``slot`` on the caller's stream."""
        rows, width = context.shape
        if not 0 <= slot < self.context_slots:
            raise ValueError(f"context slot {slot} out of range")
        if rows > self.context_rows or width != self.context_width:
            raise ValueError(
                f"context of {rows}x{width} does not fit the ring slot "
                f"{self.context_rows}x{self.context_width}"
            )
        if context.dtype != self.context.dtype:
            raise ValueError("context dtype does not match the ring")
        context = context.contiguous()
        dst = self.context.data_ptr + slot * self.context.row_stride_bytes()
        self._cudart.memcpy_async(
            dst, context.data_ptr(), rows * width * context.element_size(), stream
        )

    def pull_reply(
        self,
        slot: int,
        rows: int,
        num_steps: int,
        values_out: torch.Tensor,
        indices_out: torch.Tensor,
        stream: int,
    ) -> None:
        """Copy reply slot ``slot`` into ``values_out``/``indices_out``.

        Both outputs are ``[rows, num_steps, topk]`` contiguous device tensors
        of the slot's dtypes; the slot rows are read as ``[rows, K, topk]``
        with the slot's own ``K`` stride, so ``num_steps`` may be smaller
        than the slot depth.
        """
        if not 0 <= slot < self.reply_slots:
            raise ValueError(f"reply slot {slot} out of range")
        if rows > self.max_requests or num_steps > self.num_steps:
            raise ValueError("reply geometry exceeds the reply slot")
        for out, src in ((values_out, self.values), (indices_out, self.indices)):
            if (
                tuple(out.shape) != (rows, num_steps, self.topk)
                or out.dtype != src.dtype
            ):
                raise ValueError("reply output tensor does not match the slot")
            if not out.is_contiguous():
                raise ValueError("reply output tensor must be contiguous")
            row_bytes = self.num_steps * self.topk * out.element_size()
            step_bytes = self.topk * out.element_size()
            base = src.data_ptr + slot * src.row_stride_bytes()
            if num_steps == self.num_steps:
                self._cudart.memcpy_async(
                    out.data_ptr(), base, rows * row_bytes, stream
                )
            else:
                for row in range(rows):
                    self._cudart.memcpy_async(
                        out.data_ptr() + row * num_steps * step_bytes,
                        base + row * row_bytes,
                        num_steps * step_bytes,
                        stream,
                    )
