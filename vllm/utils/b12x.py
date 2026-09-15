# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Accessors and preparation types for the optional ``b12x`` package.

A native kernel family declares its work as a b12x ``Plan`` that the layer
holds for its lifetime. Preparation fills the plan in place. Custom ops carry
tensors, primitives, and a layer name; their bodies resolve the layer and run
the family's plain-Python path. Nothing here owns memory or graph lifetime.
"""

from __future__ import annotations

import importlib
import importlib.util
import weakref
from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from types import ModuleType
from typing import Any, Literal

import torch

import vllm.envs as envs
from vllm.utils.torch_utils import (
    LayerNameType,
    _resolve_layer_name,
    direct_register_custom_op,
)


class PreparationResourceUnavailableError(RuntimeError):
    """A native owner cannot describe or run its prepared work."""


@dataclass(frozen=True)
class B12xWorkload:
    """Serving shapes one preparation pass prepares for.

    ``stage`` selects the lifecycle point: ``weights`` runs after model load
    and before memory profiling; ``state`` runs after the KV and state pools
    exist. ``eager_only`` marks multimodal-encoder shapes that are executed
    eagerly, never captured, and prepared with their default configuration.
    """

    stage: Literal["weights", "state"]
    token_counts: tuple[int, ...]
    fixed_token_counts: tuple[int, ...]
    output_dtype: torch.dtype
    max_tokens: int
    max_seqs: int
    max_model_len: int
    speculative_tokens: int = 0
    lane: int = 0
    eager_only: bool = False

    def __post_init__(self):
        if self.stage not in ("weights", "state"):
            raise ValueError("invalid native preparation stage")
        for name in ("max_tokens", "max_seqs", "max_model_len"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("speculative_tokens", "lane"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        counts = tuple(self.token_counts)
        if not counts or any(type(count) is not int or count <= 0 for count in counts):
            raise ValueError("native preparation requires positive token counts")
        if counts != tuple(sorted(set(counts))) or counts[-1] > self.max_tokens:
            raise ValueError("token counts must be sorted, unique, and within capacity")
        fixed = tuple(self.fixed_token_counts)
        if fixed != tuple(sorted(set(fixed))) or not set(fixed) <= set(counts):
            raise ValueError("fixed token counts must be a sorted subset of token counts")
        if any(count >= self.max_tokens for count in fixed):
            raise ValueError("fixed token counts must be below capacity")
        if type(self.eager_only) is not bool:
            raise TypeError("eager_only must be boolean")
        object.__setattr__(self, "token_counts", counts)
        object.__setattr__(self, "fixed_token_counts", fixed)


@dataclass(frozen=True)
class B12xPreparationUnit:
    """One layer's preparation requests for one stage."""

    name: str
    key: Hashable
    requests: tuple[object, ...]
    stage: Literal["weights", "state"]
    autotune: bool = True

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("preparation units require a family name")
        hash(self.key)
        if self.stage not in ("weights", "state"):
            raise ValueError("invalid native preparation stage")
        if type(self.autotune) is not bool:
            raise TypeError("autotune must be boolean")
        requests = tuple(self.requests)
        names = [request.name for request in requests]
        if len(names) != len(set(names)):
            raise ValueError("preparation unit declares duplicate request names")
        object.__setattr__(self, "requests", requests)


_B12X_LAYERS: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
_B12X_UNIT_PROVIDERS: list[weakref.ref] = []


def b12x_layer_prefix(layer: torch.nn.Module) -> str:
    """A stable, process-unique name for a layer's custom-op lookups."""
    prefix = getattr(layer, "prefix", None) or ""
    existing = _B12X_LAYERS.get(prefix) if prefix else None
    if not prefix or (existing is not None and existing is not layer):
        prefix = f"{prefix or type(layer).__name__}#{id(layer):x}"
    return prefix


def register_b12x_layer(name: str, layer: torch.nn.Module) -> None:
    """Make ``layer`` reachable from custom-op bodies by ``name``."""
    existing = _B12X_LAYERS.get(name)
    if existing is not None and existing is not layer:
        raise ValueError(f"b12x layer name {name!r} is already registered")
    _B12X_LAYERS[name] = layer


def b12x_layer(name: str) -> torch.nn.Module:
    layer = _B12X_LAYERS.get(name)
    if layer is None:
        raise PreparationResourceUnavailableError(f"no live b12x layer named {name!r}")
    return layer


def set_b12x_preparation_provider(layer: object, provider: object) -> None:
    """Attach the preparation provider the driver discovers on ``layer``.

    A provider is often the layer itself or another module. Stored through
    ``object.__setattr__`` so ``torch.nn.Module`` never registers it as a
    child, which would make a layer its own submodule and turn every walk
    with ``remove_duplicate=False`` into infinite recursion.
    """
    object.__setattr__(layer, "b12x_preparation_provider", provider)


def register_b12x_unit_provider(provider: object) -> None:
    """Register a non-module owner, such as a communicator, for preparation."""
    _B12X_UNIT_PROVIDERS.append(weakref.ref(provider))


def b12x_unit_providers() -> list[object]:
    live = []
    for reference in list(_B12X_UNIT_PROVIDERS):
        provider = reference()
        if provider is None:
            _B12X_UNIT_PROVIDERS.remove(reference)
        else:
            live.append(provider)
    return live


def scope_b12x_unit_calls(unit: B12xPreparationUnit, lane: int) -> B12xPreparationUnit:
    """Run every callback of a unit's requests inside one workspace lane."""
    from vllm.v1.worker.workspace import use_workspace_lane

    def wrap_factory(factory):
        def wrapped(state):
            with use_workspace_lane(lane):
                call = factory(state)

            def scoped(callback):
                if callback is None:
                    return None

                def invoke():
                    with use_workspace_lane(lane):
                        return callback()

                return invoke

            return replace(
                call, run=scoped(call.run), produce=scoped(call.produce),
                reset=scoped(call.reset), restore=scoped(call.restore), close=scoped(call.close),
            )

        return wrapped

    def wrap_calls(calls):
        if calls is None:
            return None
        if isinstance(calls, Mapping):
            return {count: wrap_factory(factory) for count, factory in calls.items()}
        return wrap_factory(calls)

    requests = tuple(
        replace(
            request,
            prepare_call=wrap_calls(request.prepare_call),
            benchmark_call=wrap_calls(request.benchmark_call),
        )
        for request in unit.requests
    )
    return replace(unit, requests=requests)


def b12x_preparation_token_counts(
    *, max_tokens: int, cudagraph_capture_sizes: Iterable[int] = (),
    compile_sizes: Iterable[int] = (), compile_range_endpoints: Iterable[int] = (),
    speculative_tokens: int = 0,
) -> tuple[int, ...]:
    """Collect every exact serving specialization required by native owners."""
    counts = {1, int(max_tokens)}
    counts.update(int(value) for value in cudagraph_capture_sizes if int(value) > 0)
    counts.update(int(value) for value in compile_sizes if int(value) > 0)
    counts.update(int(value) for value in compile_range_endpoints if int(value) > 0)
    if speculative_tokens > 0:
        counts.add(max(1, int(max_tokens) - int(speculative_tokens)))
    return tuple(sorted(counts))


def get_b12x_dense_activation_mode(recipe: Literal["nvfp4", "mxfp8"]) -> str:
    """Resolve the dense precision override once when loading a layer."""
    override = getattr(envs, f"VLLM_B12X_{recipe.upper()}_ACTIVATION_MODE")
    return override if override is not None else envs.VLLM_B12X_DENSE_ACTIVATION_MODE


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
        "b12x.sequence.gdn_prefill",
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


def get_b12x_gdn_prefill() -> ModuleType | None:
    return _get_submodule("b12x.sequence.gdn_prefill")


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


def _b12x_blockscaled_linear(
    source: torch.Tensor,
    bias: torch.Tensor | None,
    out_features: int,
    layer_name: LayerNameType,
) -> torch.Tensor:
    layer = b12x_layer(_resolve_layer_name(layer_name))
    return layer.b12x_linear.run(source, bias)


def _b12x_blockscaled_linear_fake(
    source: torch.Tensor,
    bias: torch.Tensor | None,
    out_features: int,
    layer_name: LayerNameType,
) -> torch.Tensor:
    return source.new_empty((*source.shape[:-1], out_features))


direct_register_custom_op(
    op_name="b12x_blockscaled_linear",
    op_func=_b12x_blockscaled_linear,
    fake_impl=_b12x_blockscaled_linear_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def run_b12x_blockscaled_linear(
    source: torch.Tensor,
    bias: torch.Tensor | None,
    out_features: int,
    layer_name: LayerNameType,
) -> torch.Tensor:
    """Run a layer's prepared block-scaled linear through the opaque op."""
    return torch.ops.vllm.b12x_blockscaled_linear(source, bias, out_features, layer_name)


def get_b12x_projection_workspaces(
    rows: int, *layers: torch.nn.Module
) -> tuple[torch.Tensor | None, ...]:
    """Reserve disjoint scratch for projections that can run concurrently."""
    sizes = tuple(
        holder.get_workspace_size(rows)
        if (holder := getattr(layer, "b12x_linear", None)) is not None
        else 0
        for layer in layers
    )
    if not any(sizes):
        return (None,) * len(layers)

    from vllm.v1.worker.workspace import current_workspace_manager

    buffers = current_workspace_manager().get_simultaneous(
        *(((size,), torch.uint8) for size in sizes)
    )
    return tuple(buffer if size else None for buffer, size in zip(buffers, sizes))


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
