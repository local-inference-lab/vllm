# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm b12x JIT kernels used by a loaded model."""

from collections import Counter
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.b12x import B12xWarmupUnit, b12x_warmup_token_counts

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

_WarmupSignature = tuple[tuple[int, ...], torch.dtype]


def _collect_warmup_units(
    model: torch.nn.Module,
    token_counts: tuple[int, ...],
    output_dtype: torch.dtype,
) -> Iterable[B12xWarmupUnit]:
    units: dict[object, B12xWarmupUnit] = {}
    for layer in model.modules():
        provider = getattr(layer, "b12x_warmup_provider", None)
        get_unit = getattr(provider, "get_b12x_warmup_unit", None)
        if not callable(get_unit):
            continue
        unit = get_unit(layer, token_counts, output_dtype)
        assert isinstance(unit, B12xWarmupUnit)
        units.setdefault(unit.key, unit)
    return units.values()


def _compile_warmup_units(
    units: Iterable[B12xWarmupUnit],
) -> Counter[str]:
    warmed: Counter[str] = Counter()
    with torch.inference_mode():
        for unit in units:
            unit.compile()
            warmed[unit.name] += 1
        if warmed:
            torch.accelerator.synchronize()
    return warmed


def b12x_warmup(worker: "Worker", cudagraph_capture_sizes: list[int]) -> bool:
    """Resolve the B12X kernels the loaded model will launch when serving.

    Args:
        worker: The worker holding the loaded model and its configuration.
        cudagraph_capture_sizes: The CUDA-graph capture sizes; with the
            batched-token limit and the compile sizes they define the
            capacities to resolve.

    Returns:
        Whether this call resolved kernels: ``False`` off CUDA SM120-class
        devices, when the model holds no B12X warm-up units, or when the
        same capacities and dtype were already resolved in this worker.
    """
    if not current_platform.is_cuda():
        return False
    if not current_platform.is_device_capability_family(120):
        return False

    output_dtype = getattr(
        getattr(worker, "model_config", None),
        "dtype",
        torch.bfloat16,
    )
    if output_dtype not in (torch.bfloat16, torch.float16):
        output_dtype = torch.bfloat16
    compile_sizes = worker.vllm_config.compilation_config.compile_sizes or []
    max_num_batched_tokens = worker.scheduler_config.max_num_batched_tokens
    serving_sizes = [
        max_num_batched_tokens,
        *cudagraph_capture_sizes,
        *(size for size in compile_sizes if isinstance(size, int)),
    ]
    max_tokens = max_num_batched_tokens
    max_num_scheduled_tokens = worker.scheduler_config.max_num_scheduled_tokens
    if max_num_scheduled_tokens is not None:
        max_tokens = max(max_tokens, max_num_scheduled_tokens)
    num_speculative_tokens = int(
        getattr(worker.vllm_config, "num_speculative_tokens", 0) or 0
    )
    if 0 < num_speculative_tokens < max_tokens:
        serving_sizes.append(max_tokens - num_speculative_tokens)
    token_counts = b12x_warmup_token_counts(
        max_tokens=max_tokens,
        cudagraph_capture_sizes=serving_sizes,
    )
    # Memory profiling resolves B12X kernels before sizing the KV cache. The
    # regular pre-capture warmup later requests the same static capacities in
    # the same worker. Repeating those launches after KV allocation can require
    # cold-start scratch that was intentionally excluded from the repeatable
    # serving peak. The model is fully loaded before either call, so its exact
    # capacity set and output dtype identify the worker-local request even when
    # a provider resolves internal kernel state during the first warmup.
    signature: _WarmupSignature = (token_counts, output_dtype)
    completed: set[_WarmupSignature] = getattr(
        worker, "_b12x_completed_warmup_signatures", set()
    )
    if signature in completed:
        logger.info_once(
            "Skipping repeated B12X warmup for capacities=%s and dtype=%s.",
            token_counts,
            output_dtype,
        )
        return False

    units = tuple(
        _collect_warmup_units(
            worker.get_model(),
            token_counts,
            output_dtype,
        )
    )
    if not units:
        return False

    for name, count in _compile_warmup_units(units).items():
        logger.info_once(
            "Warmed up %d b12x %s kernel signature(s).",
            count,
            name,
        )
    completed.add(signature)
    vars(worker)["_b12x_completed_warmup_signatures"] = completed
    return True
