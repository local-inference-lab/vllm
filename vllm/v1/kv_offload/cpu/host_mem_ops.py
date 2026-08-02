# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-level host-memory registration helpers for CPU KV offload."""

import torch

from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
from vllm.platforms import current_platform


def _get_cuda_driver():
    from cuda.bindings import driver

    return driver


def register_host_memory(ptr: int, size: int) -> int:
    """Register host memory and return a CUDA/HIP error code.

    CUDA uses the driver API deliberately. A failed runtime
    ``cudaHostRegister`` updates the calling thread's last-error state, which
    can make a later unrelated PyTorch operation fail even when offload falls
    back to pageable memory. Driver API failures are returned directly and do
    not contaminate that runtime state.
    """
    if current_platform.is_cuda():
        (result,) = _get_cuda_driver().cuMemHostRegister(ptr, size, 0)
        return int(result.value)

    cudart = torch.cuda.cudart()
    result = cudart.cudaHostRegister(ptr, size, 0)
    code = int(result.value)
    if code != 0:
        # HIP uses the runtime API, so consume its expected registration error
        # before the caller continues with pageable memory.
        CudaRTLibrary().cudaGetLastError()
    return code


def unregister_host_memory(ptr: int) -> int:
    """Unregister host memory and return a CUDA/HIP error code."""
    if current_platform.is_cuda():
        (result,) = _get_cuda_driver().cuMemHostUnregister(ptr)
        return int(result.value)

    cudart = torch.cuda.cudart()
    result = cudart.cudaHostUnregister(ptr)
    code = int(result.value)
    if code != 0:
        CudaRTLibrary().cudaGetLastError()
    return code
