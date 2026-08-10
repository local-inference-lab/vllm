# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import vllm.envs as envs


def b12x_cuda_graph_prewarm_enabled() -> bool:
    return (
        envs.VLLM_USE_B12X_SPARSE_INDEXER
        or envs.VLLM_USE_B12X_MHC
        or envs.VLLM_USE_B12X_FP8_GEMM
        or envs.VLLM_USE_B12X_WO_PROJECTION
        or envs.VLLM_USE_B12X_MOE
        or envs.VLLM_USE_B12X_MINIMAX_M3_MSA
    )


def b12x_cuda_graph_wrapper_prewarm_enabled(is_piecewise: bool) -> bool:
    if not b12x_cuda_graph_prewarm_enabled():
        return False
    if is_piecewise:
        return envs.VLLM_B12X_CUDAGRAPH_PIECEWISE_PREWARM
    return True


def _kernel_resolution_api() -> (
    tuple[
        Callable[[str], None],
        Callable[[], bool],
        Callable[[], None],
    ]
    | None
):
    for namespace in ("b12x", "sparkinfer"):
        try:
            module = importlib.import_module(namespace)
        except ModuleNotFoundError as exc:
            if exc.name != namespace:
                raise
            continue
        freeze = getattr(module, "freeze_kernel_resolution", None)
        frozen = getattr(module, "kernel_resolution_frozen", None)
        unfreeze = getattr(module, "unfreeze_kernel_resolution", None)
        if all(callable(item) for item in (freeze, frozen, unfreeze)):
            return freeze, frozen, unfreeze
    return None


@contextmanager
def guard_b12x_kernel_resolution(reason: str) -> Iterator[None]:
    if not b12x_cuda_graph_prewarm_enabled():
        yield
        return

    resolution_api = _kernel_resolution_api()
    if resolution_api is None:
        yield
        return
    (
        freeze_kernel_resolution,
        kernel_resolution_frozen,
        unfreeze_kernel_resolution,
    ) = resolution_api

    if kernel_resolution_frozen():
        yield
        return

    freeze_kernel_resolution(reason)
    try:
        yield
    finally:
        unfreeze_kernel_resolution()
