# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Accessors for the optional ``b12x`` package."""

import importlib
import importlib.util
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass, fields, is_dataclass
from types import ModuleType
from typing import Any

import torch


@dataclass(frozen=True)
class B12xWarmupUnit:
    name: str
    key: Hashable
    compile: Callable[[], None]


_HAS_B12X = importlib.util.find_spec("b12x") is not None


def _import_submodule(module_name: str) -> ModuleType | None:
    if not _HAS_B12X:
        return None
    try:
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError):
        return None


_B12X_SUBMODULES = {
    module_name: _import_submodule(module_name)
    for module_name in (
        "b12x.attention.paged",
        "b12x.attention.sparse_mla",
        "b12x.attention.compressed_sparse_mla",
        "b12x.attention.dsa_indexer",
        "b12x.attention.qsa",
        "b12x.gemm.bf16_vocab_projection",
        "b12x.gemm.blockscaled",
        "b12x.gemm.mla_query_projection",
        "b12x.gemm.wo_projection",
        "b12x.norm.mhc",
        # TODO: Remove once B12X exposes the scale-swizzle API publicly.
        "b12x._lib.intrinsics",
        "b12x.gemm.mxfp8_linear",
        "b12x.gemm.tensor_fp8_linear",
        "b12x.moe.fused_moe",
        "b12x.norm.hyperconnection",
        "b12x.sequence.gdn_decode",
        "b12x.sequence.kda_prefill",
        "b12x.sequence.mtp_feedback",
        "b12x.sequence.ple",
        "b12x.sequence.ple_embedding",
        "b12x.sequence.ple_hash",
    )
}


def has_b12x() -> bool:
    """Return whether the B12X package is installed."""
    return _HAS_B12X


def _get_submodule(module_name: str) -> ModuleType | None:
    return _B12X_SUBMODULES.get(module_name)


def get_b12x_blockscaled() -> ModuleType | None:
    return _get_submodule("b12x.gemm.blockscaled")


def get_b12x_bf16_vocab_projection() -> ModuleType | None:
    return _get_submodule("b12x.gemm.bf16_vocab_projection")


def get_b12x_mla_query_projection() -> ModuleType | None:
    return _get_submodule("b12x.gemm.mla_query_projection")


def get_b12x_wo_projection() -> ModuleType | None:
    return _get_submodule("b12x.gemm.wo_projection")


def get_b12x_mhc() -> ModuleType | None:
    return _get_submodule("b12x.norm.mhc")


def get_b12x_compressed_sparse_mla() -> ModuleType | None:
    return _get_submodule("b12x.attention.compressed_sparse_mla")


def get_b12x_sparse_mla() -> ModuleType | None:
    return _get_submodule("b12x.attention.sparse_mla")


def get_b12x_dsa_indexer() -> ModuleType | None:
    return _get_submodule("b12x.attention.dsa_indexer")


def get_b12x_intrinsics() -> ModuleType | None:
    return _get_submodule("b12x._lib.intrinsics")


def get_b12x_mxfp8_linear() -> ModuleType | None:
    return _get_submodule("b12x.gemm.mxfp8_linear")


def get_b12x_tensor_fp8_linear() -> ModuleType | None:
    return _get_submodule("b12x.gemm.tensor_fp8_linear")


def get_b12x_fused_moe() -> ModuleType | None:
    return _get_submodule("b12x.moe.fused_moe")


def get_b12x_paged_attention() -> ModuleType | None:
    return _get_submodule("b12x.attention.paged")


def get_b12x_qsa() -> ModuleType | None:
    return _get_submodule("b12x.attention.qsa")


def get_b12x_hyperconnection() -> ModuleType | None:
    return _get_submodule("b12x.norm.hyperconnection")


def get_b12x_gdn_decode() -> ModuleType | None:
    return _get_submodule("b12x.sequence.gdn_decode")


def get_b12x_kda_prefill() -> ModuleType | None:
    return _get_submodule("b12x.sequence.kda_prefill")


def get_b12x_mtp_feedback() -> ModuleType | None:
    return _get_submodule("b12x.sequence.mtp_feedback")


def get_b12x_ple() -> ModuleType | None:
    return _get_submodule("b12x.sequence.ple")


def get_b12x_ple_embedding() -> ModuleType | None:
    return _get_submodule("b12x.sequence.ple_embedding")


def get_b12x_ple_hash() -> ModuleType | None:
    return _get_submodule("b12x.sequence.ple_hash")


def b12x_warmup_token_counts(
    *,
    max_tokens: int,
    cudagraph_capture_sizes: Iterable[int] = (),
) -> tuple[int, ...]:
    # B12X deduplicates shapes that select the same internal kernel policy.
    # Keep the complete serving shape set here rather than duplicating its
    # policy-selection heuristics in vLLM.
    counts = {1}
    counts.update(int(size) for size in cudagraph_capture_sizes if int(size) > 0)
    if int(max_tokens) > 0:
        counts.add(int(max_tokens))
    return tuple(sorted(counts))


def get_b12x_scratch_buffers(plan: Any) -> list[torch.Tensor]:
    """Return caller-owned scratch buffers for a planned b12x operation."""
    specs = tuple(plan.scratch_specs())
    if not specs:
        return []

    from vllm.v1.worker.workspace import (
        current_workspace_manager,
        is_workspace_manager_initialized,
    )

    if is_workspace_manager_initialized():
        return current_workspace_manager().get_simultaneous(
            *((spec.shape, spec.dtype) for spec in specs)
        )
    return [
        torch.empty(spec.shape, dtype=spec.dtype, device=spec.device) for spec in specs
    ]


def _same_packed_layout(current: Any, replacement: Any) -> bool:
    if type(current) is not type(replacement):
        return False
    if isinstance(current, torch.Tensor):
        return (
            current.shape == replacement.shape
            and current.stride() == replacement.stride()
            and current.dtype == replacement.dtype
            and current.device == replacement.device
        )
    if is_dataclass(current):
        return all(
            _same_packed_layout(
                getattr(current, field.name),
                getattr(replacement, field.name),
            )
            for field in fields(current)
        )
    return bool(current == replacement)


def _copy_packed_tensors(current: Any, replacement: Any) -> None:
    if isinstance(current, torch.Tensor):
        current.copy_(replacement)
    elif is_dataclass(current):
        for field in fields(current):
            _copy_packed_tensors(
                getattr(current, field.name),
                getattr(replacement, field.name),
            )


@torch.no_grad()
def reuse_packed_weight_storage(current: Any, replacement: Any) -> Any:
    """Reuse packed tensor addresses when a compatible weight is reloaded."""
    if current is None or not _same_packed_layout(current, replacement):
        return replacement
    _copy_packed_tensors(current, replacement)
    return current
