# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CUDA-graph-safe EXL3 cartridge runtime for MSRT additive quantization.

MSRT (Multi-Stage Rescaled Trellis) cartridges contain full-rank,
trellis-quantized residuals for the gate, up, and down expert projections.
Applying those residuals projection-by-projection is exact but cannot be added
after a fused MoE call: gate and up residuals must be present before SiLU.

For rank-sliced EXL3, this module keeps cartridge trellises compressed on the
GPU and builds stable pointer tables for the additive routed-expert kernel.
Cartridge load quiesces the engine, drops the base CUDA graphs, allocates the
packed tensors, and captures cartridge graphs. Deactivation performs the
inverse transition and releases the cartridge tensors before recapturing the
compressed base path.

The runtime is opt-in. Inactive cartridge support has no cartridge buffers,
alternate graph path, or runtime overhead. Set
``VLLM_ENABLE_EXL3_CARTRIDGE=1`` on a trusted-admin deployment to allow the
quiescent recapture transaction. Use
``await async_llm.load_exl3_cartridge(path)`` and
``await async_llm.deactivate_exl3_cartridge()``; both dispatch through the worker
control plane to every model worker. The synchronous ``LLMEngine`` interface is
intentionally unsupported because it cannot serialize request admission with a
model-wide graph transition.

Tensor parallel workers either slice a manifest-declared full-rank ``rank0``
cartridge or select their matching rank from a cartridge containing every TP
rank. The runtime exposes one model-wide slot; per-request selection is not
claimed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
import time
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, TypedDict

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.exl3 import (
    Exl3LinearMethod,
    _load_exl3_ext,
)

logger = init_logger(__name__)


def _load_additive_exl3_ext() -> Any:
    ext = _load_exl3_ext()
    if getattr(ext, "EXL3_MOE_ADDITIVE_ABI_VERSION", None) != 1:
        raise RuntimeError(
            "packed EXL3 cartridges require EXL3_MOE_ADDITIVE_ABI_VERSION=1"
        )
    if not hasattr(ext, "exl3_moe_additive_fused"):
        raise RuntimeError(
            "packed EXL3 cartridges require an ExLlamaV3 extension that "
            "exports exl3_moe_additive_fused"
        )
    return ext


_CARTRIDGE_KEY_RE = re.compile(
    r"^(?P<layer>.+\.experts)\."
    r"(?P<expert>[0-9]+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\."
    r"rank(?P<rank>[0-9]+)\.trellis_(?P<label>.+)$"
)
_CARTRIDGE_COMPANION_KEY_RE = re.compile(
    r"^(?P<layer>.+\.experts)\."
    r"(?P<expert>[0-9]+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\."
    r"rank(?P<rank>[0-9]+)\.scale_(?P<label>.+)$"
)
_ADAPTER_SCHEMA = "fq-cartridge-adapter/3"
_MCG_MULTIPLIER = 3417055213
_RUNTIME_OPERATION = "base_exl3_gemm + sum(stage_exl3_gemm / stage_scale)"
_RUNTIME_PROFILE = "exl3-msrt-additive/1"
_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
_LAYER_NAME_RE = re.compile(r"^model\.layers\.([0-9]+)\.mlp\.experts$")
_LIVE_LAYER_INDEX_RE = re.compile(r"(?:^|\.)layers\.([0-9]+)(?=\.|$)")
_ADAPTER_FIELDS = frozenset(
    {
        "schema",
        "assembly",
        "base",
        "chain",
        "format",
        "runtime_profile",
        "rotation_ownership",
        "standard_lora_compatible",
        "runtime_operation",
        "codebook",
        "mcg_multiplier",
        "mcg_ownership",
        "scale_shape",
        "tensor_parallel",
        "shards",
        "num_tensors",
        "selected_experts",
        "selected_layers",
        "coverage",
        "producer_verified_signer",
        "campaign",
        "source_assembly",
        "tool_version",
        "created_utc",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SHARD_MAP = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}
_HADAMARD_BLOCK = 128
_TP_AXES = {
    "gate_proj": "output",
    "up_proj": "output",
    "down_proj": "input",
}
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_SHARD_BYTES = 64 * 1024 * 1024 * 1024
_MAX_ADAPTER_BYTES = 1024 * 1024 * 1024 * 1024
_STAGING_TIMEOUT_SECONDS = 15 * 60
_COPY_CHUNK_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class _AdapterState:
    config: dict[str, Any]
    stage_labels: tuple[str, ...]
    stage_bits: dict[str, int]
    paths: tuple[Path, ...]
    by_layer: dict[str, list[tuple[str, re.Match[str]]]]
    key_count: int


class StagedExl3Cartridge:
    """Verified private shard copies ready for the drained transaction."""

    def __init__(self, temporary: tempfile.TemporaryDirectory[str]) -> None:
        self._temporary = temporary
        self.state: _AdapterState | None = None
        self.local_layer_names: tuple[str, ...] = ()

    def close(self) -> None:
        self._temporary.cleanup()

    def __enter__(self) -> StagedExl3Cartridge:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class _StageTensors(TypedDict):
    trellis: torch.Tensor
    scale: float


class Exl3LoraCartridge:
    """MSRT residual tensors for one routed-expert layer.

    Each stage is keyed by ``(expert_id, shard_id)`` and stores packed trellis
    indices and the positive rescaling factor used by the MCG-only rank-sliced
    encoder; residuals reuse the base layer's own suh/svh.
    """

    def __init__(self, num_stages: int, num_experts: int, device: torch.device):
        if num_stages < 1:
            raise ValueError(f"num_stages must be positive, got {num_stages}")
        if num_experts < 1:
            raise ValueError(f"num_experts must be positive, got {num_experts}")
        self.num_stages = num_stages
        self.num_experts = num_experts
        self.device = device
        self.stages: list[dict[tuple[int, str], _StageTensors]] = [
            {} for _ in range(num_stages)
        ]
        self.active = False

    def set_stage_tensors(
        self,
        stage_idx: int,
        expert_id: int,
        shard_id: str,
        trellis: torch.Tensor,
        scale: float,
    ) -> None:
        """Set one stage of one expert projection."""
        if not 0 <= stage_idx < self.num_stages:
            raise IndexError(f"stage index {stage_idx} is out of range")
        if not 0 <= expert_id < self.num_experts:
            raise IndexError(f"expert index {expert_id} is out of range")
        if shard_id not in {"w1", "w2", "w3"}:
            raise ValueError(f"unsupported EXL3 shard {shard_id!r}")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(
                f"cartridge scale must be finite and positive, got {scale}"
            )
        inverse_scale = 1.0 / scale
        if (
            not math.isfinite(inverse_scale)
            or inverse_scale > torch.finfo(torch.float32).max
        ):
            raise ValueError(
                "cartridge inverse scale is not finite in FP32: "
                f"scale={scale}, inverse={inverse_scale}"
            )
        self.stages[stage_idx][(expert_id, shard_id)] = {
            "trellis": trellis.to(self.device).contiguous(),
            "scale": float(scale),
        }

    def get_stage_tensors(
        self, stage_idx: int, expert_id: int, shard_id: str
    ) -> _StageTensors | None:
        """Return one stage of one expert projection, if present."""
        return self.stages[stage_idx].get((expert_id, shard_id))

    def to(self, device: torch.device) -> None:
        """Move source tensors to one device without changing stage metadata."""
        for stage in self.stages:
            for tensors in stage.values():
                tensor = tensors["trellis"]
                tensors["trellis"] = tensor.to(device).contiguous()
        self.device = device

    def clear(self) -> None:
        """Release source cartridge tensors."""
        self.stages = [{} for _ in range(self.num_stages)]
        self.active = False


class Exl3CUDAGraphCartridgeRuntime:
    """Fixed-address packed cartridge tensors and additive MoE workspaces."""

    def __init__(self, layer: Any, workspace: dict[str, torch.Tensor] | None = None):
        self.tp_rank = int(getattr(layer, "exl3_tp_rank", 0))
        self.tp_size = int(getattr(layer, "exl3_tp_size", 1))
        if self.tp_size < 1 or not 0 <= self.tp_rank < self.tp_size:
            raise ValueError(
                "invalid EXL3 cartridge tensor-parallel configuration: "
                f"rank={self.tp_rank}, size={self.tp_size}"
            )
        self.num_experts = int(layer.local_num_experts)
        self.hidden_size = int(layer.exl3_hidden_size)
        self.intermediate_size = int(layer.exl3_intermediate_size_per_partition)
        self.topk = int(layer.top_k)
        self.dtype = torch.float16
        self.device = layer.w13_trellis.device
        self.chunk = min(128, int(layer.exl3_max_num_batched_tokens))
        self.ext = _load_additive_exl3_ext() if self.device.type == "cuda" else None
        if self.ext is not None:
            if not hasattr(self.ext, "exl3_moe_max_concurrency"):
                raise RuntimeError(
                    "The EXL3 extension lacks packed cartridge entry point "
                    "exl3_moe_max_concurrency"
                )
            concurrency = int(self.ext.exl3_moe_max_concurrency(self.device.index or 0))
        else:
            concurrency = 1
        workspace = {} if workspace is None else workspace
        if not workspace:
            workspace.update(
                xh=torch.empty(
                    (self.chunk, self.hidden_size),
                    dtype=self.dtype,
                    device=self.device,
                ),
                out32=torch.empty(
                    (self.chunk, self.hidden_size),
                    dtype=torch.float32,
                    device=self.device,
                ),
                tg=torch.empty(
                    (concurrency, self.chunk, self.hidden_size),
                    dtype=self.dtype,
                    device=self.device,
                ),
                ig=torch.empty(
                    (concurrency, self.chunk, self.intermediate_size),
                    dtype=self.dtype,
                    device=self.device,
                ),
                expert_count=torch.empty(
                    self.num_experts + 1,
                    dtype=torch.int64,
                    device=self.device,
                ),
                token_sorted=torch.empty(
                    self.chunk * self.topk,
                    dtype=torch.int64,
                    device=self.device,
                ),
                weight_sorted=torch.empty(
                    self.chunk * self.topk,
                    dtype=self.dtype,
                    device=self.device,
                ),
                expert_map=torch.arange(
                    self.num_experts,
                    dtype=torch.int64,
                    device=self.device,
                ),
            )
            workspace["tu"] = torch.empty_like(workspace["tg"])
            workspace["iu"] = torch.empty_like(workspace["ig"])
            workspace["expert_offsets"] = torch.empty_like(workspace["expert_count"])
        for name, tensor in workspace.items():
            setattr(self, name, tensor)
        self._active = False
        self._materialized = False
        self._packed_tensors: tuple[torch.Tensor, ...] = ()
        self.pointer_args: tuple[torch.Tensor, ...] = ()
        self.max_residual_bits = 0
        layer_bitrates = tuple(getattr(layer, "exl3_layer_bitrates", ()))
        if not layer_bitrates or len(set(layer_bitrates)) != 1:
            raise ValueError("packed cartridge runtime requires a uniform base bitrate")
        self.base_bits = int(next(iter(layer_bitrates)))

    @staticmethod
    def _pointer_table(tensors: list[torch.Tensor]) -> torch.Tensor:
        return torch.tensor(
            [tensor.data_ptr() for tensor in tensors],
            dtype=torch.int64,
            device=tensors[0].device,
        )

    def deactivate(self) -> None:
        """Select the rank-sliced base path."""
        self._active = False

    def activate(self) -> None:
        """Select the prepared packed cartridge path."""
        if not self._materialized:
            raise RuntimeError("cannot activate an unmaterialized EXL3 cartridge")
        self._active = True

    @torch.inference_mode()
    def materialize(self, layer: Any, cartridge: Exl3LoraCartridge) -> None:
        """Retain packed stages and construct fixed-address kernel metadata."""
        del layer
        if cartridge.num_experts != self.num_experts:
            raise ValueError(
                "cartridge expert count does not match runtime: "
                f"{cartridge.num_experts} != {self.num_experts}"
            )
        if not cartridge.active:
            raise ValueError("cannot materialize an inactive cartridge")

        self._materialized = False
        self.deactivate()
        projection_tensors: dict[str, list[torch.Tensor]] = {
            "w1": [],
            "w3": [],
            "w2": [],
        }
        projection_scales: dict[str, list[float]] = {
            "w1": [],
            "w3": [],
            "w2": [],
        }
        projection_bits: dict[str, list[int]] = {
            "w1": [],
            "w3": [],
            "w2": [],
        }
        retained: list[torch.Tensor] = []
        for stage_idx in range(cartridge.num_stages):
            for shard_id in ("w1", "w3", "w2"):
                stage_tensors = [
                    cartridge.get_stage_tensors(stage_idx, expert_id, shard_id)
                    for expert_id in range(self.num_experts)
                ]
                present_trellises = [
                    tensors["trellis"]
                    for tensors in stage_tensors
                    if tensors is not None
                ]
                if present_trellises:
                    fallback = present_trellises[0]
                    stage_bits = {
                        trellis.shape[2] // 16 for trellis in present_trellises
                    }
                    if len(stage_bits) != 1:
                        raise ValueError(
                            "packed cartridge stage bitrate must be uniform across "
                            f"experts: stage={stage_idx}, projection={shard_id}, "
                            f"bitrates={sorted(stage_bits)}"
                        )
                    stage_bit = stage_bits.pop()
                elif projection_tensors[shard_id]:
                    fallback = projection_tensors[shard_id][0]
                    stage_bit = fallback.shape[2] // 16
                else:
                    raise ValueError(
                        "packed cartridge stage has no fallback tensor: "
                        f"stage={stage_idx}, projection={shard_id}"
                    )

                for expert_id, tensors in enumerate(stage_tensors):
                    if tensors is None:
                        projection_tensors[shard_id].append(fallback)
                        projection_scales[shard_id].append(0.0)
                        continue
                    trellis = tensors["trellis"]
                    scale = tensors["scale"]
                    assert isinstance(trellis, torch.Tensor)
                    assert isinstance(scale, float)
                    projection_tensors[shard_id].append(trellis)
                    inverse_scale = 1.0 / scale
                    if (
                        not math.isfinite(inverse_scale)
                        or inverse_scale > torch.finfo(torch.float32).max
                    ):
                        raise ValueError(
                            "packed cartridge inverse scale is not finite in FP32: "
                            f"stage={stage_idx}, expert={expert_id}, "
                            f"projection={shard_id}, scale={scale}"
                        )
                    projection_scales[shard_id].append(inverse_scale)
                    retained.append(trellis)
                projection_bits[shard_id].append(stage_bit)

        table_shape = (cartridge.num_stages, self.num_experts)
        pointer_tables = tuple(
            self._pointer_table(projection_tensors[shard_id]).view(table_shape)
            for shard_id in ("w1", "w3", "w2")
        )
        scale_tables = tuple(
            torch.tensor(
                projection_scales[shard_id],
                dtype=torch.float32,
                device=self.device,
            ).view(table_shape)
            for shard_id in ("w1", "w3", "w2")
        )
        bit_tables = tuple(
            torch.tensor(
                projection_bits[shard_id],
                dtype=torch.int32,
                device=self.device,
            )
            for shard_id in ("w1", "w3", "w2")
        )
        self.pointer_args = pointer_tables + scale_tables + bit_tables
        self.max_residual_bits = max(
            bits for projection in projection_bits.values() for bits in projection
        )
        computed_max_residual_bits = max(
            int(table.max().item()) for table in bit_tables
        )
        if computed_max_residual_bits != self.max_residual_bits:
            raise ValueError(
                "packed cartridge max_residual_bits does not match its "
                f"per-stage bit tables: {self.max_residual_bits} != "
                f"{computed_max_residual_bits}"
            )
        self._packed_tensors = tuple(retained) + self.pointer_args
        self._materialized = True

    def apply(
        self,
        layer: Any,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        if not self._materialized or not self._active:
            raise RuntimeError("packed EXL3 cartridge runtime is not active")
        if self.ext is None:
            raise RuntimeError("packed EXL3 cartridge execution requires CUDA")
        if x.shape[0] == 0:
            return torch.empty_like(x)
        outputs: list[torch.Tensor] = []
        for start in range(0, x.shape[0], self.chunk):
            rows = min(self.chunk, x.shape[0] - start)
            chunk_x = x[start : start + rows]
            if x.dtype == self.dtype and chunk_x.is_contiguous():
                # Under CUDA-graph capture/replay this chunk's slice of `x`
                # already sits at a fixed address (it is itself the output of
                # an earlier op in the same captured graph); copying it into
                # the fixed self.xh landing buffer is then pure overhead.
                xh = chunk_x
            else:
                xh = self.xh[:rows]
                xh.copy_(chunk_x)
            out32 = self.out32[:rows]
            out32.zero_()
            route_count = rows * self.topk
            # Upper bound on distinct experts this chunk can hit. The kernel's
            # ticket scheduler already loops over every expert with tokens
            # regardless of grid shape (correctness never depends on this
            # value), so an upper bound is always safe; it only narrows the
            # host-side group_size/num_groups launch split to avoid
            # over-subscribing SMs when few experts can possibly be active,
            # e.g. every single-token decode step.
            num_active = min(route_count, self.num_experts)
            self.ext.exl3_moe_additive_fused(
                xh,
                out32,
                topk_ids[start : start + rows],
                topk_weights[start : start + rows],
                self.expert_map,
                self.expert_count,
                self.expert_offsets,
                self.token_sorted[:route_count],
                self.weight_sorted[:route_count],
                self.tg,
                self.tu,
                self.ig,
                self.iu,
                0,
                self.base_bits,
                self.base_bits,
                self.base_bits,
                *layer.exl3_pointer_tables,
                *self.pointer_args,
                self.max_residual_bits,
                True,
                False,
                True,
                False,
                True,
                False,
                0.0,
                num_active,
            )
            # out32 is fp32 because the kernel's multi-expert scatter-add
            # (top_k > 1 contributions to one token) uses atomicAdd, which
            # requires fp32; this cast-copy is architecturally required, not
            # an avoidable inefficiency like the input copy above.
            outputs.append(out32.to(x.dtype))
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs)


def prepare_exl3_cudagraph_cartridge_runtime(
    layer: Any,
    workspace: dict[str, torch.Tensor] | None = None,
) -> Exl3CUDAGraphCartridgeRuntime:
    """Allocate a layer's fixed-address cartridge buffers before capture."""
    runtime = getattr(layer, "_exl3_cartridge_runtime", None)
    if runtime is None:
        runtime = Exl3CUDAGraphCartridgeRuntime(layer, workspace)
        layer._exl3_cartridge_runtime = runtime
    return runtime


def _model_workspace(
    model: torch.nn.Module,
    layer: Any,
) -> dict[str, torch.Tensor]:
    cache = getattr(model, "_exl3_cartridge_workspaces", None)
    if cache is None:
        cache = {}
        model._exl3_cartridge_workspaces = cache
    device = layer.w13_trellis.device
    key = (
        device.type,
        device.index,
        torch.float16,
        int(layer.exl3_hidden_size),
        int(layer.exl3_intermediate_size_per_partition),
        int(layer.local_num_experts),
        int(layer.top_k),
        min(128, int(layer.exl3_max_num_batched_tokens)),
    )
    return cache.setdefault(key, {})


def apply_exl3_cudagraph_cartridge(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    layer: Any,
) -> torch.Tensor:
    """Run the packed additive path captured for the active topology."""
    runtime = getattr(layer, "_exl3_cartridge_runtime", None)
    if not isinstance(runtime, Exl3CUDAGraphCartridgeRuntime):
        raise RuntimeError(
            "EXL3 cartridge runtime was not prepared before CUDA graph capture"
        )
    return runtime.apply(layer, x, topk_weights, topk_ids)


def _validate_stage_trellis_shape(
    trellis: torch.Tensor,
    *,
    input_size: int,
    output_size: int,
) -> None:
    shape_valid = trellis.ndim == 3
    packed_k = trellis.shape[0] * 16 if shape_valid else 0
    packed_n = trellis.shape[1] * 16 if shape_valid else 0
    expected_packed_k = ((input_size + 127) // 128) * 128
    expected_packed_n = ((output_size + 127) // 128) * 128
    if (
        trellis.dtype != torch.int16
        or not shape_valid
        or packed_k != expected_packed_k
        or packed_n != expected_packed_n
        or trellis.shape[2] % 16
        or trellis.shape[2] // 16 not in (1, 2, 3, 4, 5, 6)
    ):
        raise ValueError(
            "Invalid MSRT cartridge trellis: "
            f"shape={tuple(trellis.shape)}, dtype={trellis.dtype}; "
            f"packed K/N must equal the 128-aligned logical shape "
            f"({expected_packed_k}, {expected_packed_n}) for "
            f"K={input_size}, N={output_size}; use K in "
            "{1,2,3,4,5,6} and dtype=torch.int16"
        )


def _validate_stage_trellis(layer: Any, shard_id: str, trellis: torch.Tensor) -> None:
    """Validate one rank-local MSRT residual against its base shard."""
    hidden_size = int(layer.exl3_hidden_size)
    intermediate_size = int(layer.exl3_intermediate_size_per_partition)
    input_size, output_size = (
        (intermediate_size, hidden_size)
        if shard_id == "w2"
        else (hidden_size, intermediate_size)
    )
    _validate_stage_trellis_shape(
        trellis,
        input_size=input_size,
        output_size=output_size,
    )


def _slice_full_stage_trellis_for_tp(
    layer: Any,
    shard_id: str,
    trellis: torch.Tensor,
) -> torch.Tensor:
    """Select this TP worker's shard from a full-rank cartridge."""
    tp_rank = int(getattr(layer, "exl3_tp_rank", 0))
    tp_size = int(getattr(layer, "exl3_tp_size", 1))
    local_intermediate = int(layer.exl3_intermediate_size_per_partition)
    full_intermediate = local_intermediate * tp_size
    hidden_size = int(layer.exl3_hidden_size)
    input_size, output_size = (
        (full_intermediate, hidden_size)
        if shard_id == "w2"
        else (hidden_size, full_intermediate)
    )
    _validate_stage_trellis_shape(
        trellis,
        input_size=input_size,
        output_size=output_size,
    )
    if tp_size == 1:
        return trellis

    return Exl3LinearMethod._slice_exl3_tensor(
        trellis,
        dim=0 if shard_id == "w2" else 1,
        start=tp_rank * local_intermediate,
        size=local_intermediate,
    )


def _select_rank_entries(
    entries: list[tuple[str, re.Match[str]]],
    layer: Any,
    *,
    layout: str,
    manifest_ranks: tuple[int, ...],
) -> list[tuple[str, re.Match[str]]]:
    """Select local tensors according to the manifest-declared TP layout."""
    if not entries:
        return entries

    tp_rank = int(getattr(layer, "exl3_tp_rank", 0))
    tp_size = int(getattr(layer, "exl3_tp_size", 1))
    entries_by_rank: dict[int, list[tuple[str, re.Match[str]]]] = {}
    for entry in entries:
        entries_by_rank.setdefault(int(entry[1].group("rank")), []).append(entry)

    ranks = set(entries_by_rank)
    expected_ranks = set(manifest_ranks)
    if ranks != expected_ranks:
        raise ValueError(
            "MSRT cartridge tensor ranks do not match adapter_config.json: "
            f"tensors={sorted(ranks)}, manifest={sorted(expected_ranks)}"
        )
    if layout == "full":
        return entries_by_rank[0]
    if layout != "rank-sharded":
        raise ValueError(f"unsupported MSRT cartridge TP layout {layout!r}")

    def signature(
        rank_entries: list[tuple[str, re.Match[str]]],
    ) -> set[tuple[str, int, str]]:
        return {
            (
                match.group("label"),
                int(match.group("expert")),
                match.group("projection"),
            )
            for _, match in rank_entries
        }

    expected_signature = signature(entries_by_rank[0])
    inconsistent = [
        rank
        for rank, rank_entries in entries_by_rank.items()
        if signature(rank_entries) != expected_signature
    ]
    if inconsistent:
        raise ValueError(
            "MSRT cartridge TP ranks have inconsistent stage topology: "
            f"ranks={inconsistent}"
        )
    if tp_size != len(manifest_ranks) or tp_rank not in entries_by_rank:
        raise ValueError(
            "MSRT cartridge TP ranks do not match the runtime: "
            f"cartridge={list(manifest_ranks)}, runtime=0..{tp_size - 1}"
        )
    return entries_by_rank[tp_rank]


def _index_cartridge_keys(
    keys: tuple[str, ...],
) -> dict[str, list[tuple[str, re.Match[str]]]]:
    """Group trellises by layer and reject orphaned or unknown companions."""
    key_set = set(keys)
    by_layer: dict[str, list[tuple[str, re.Match[str]]]] = {}
    for key in keys:
        match = _CARTRIDGE_KEY_RE.fullmatch(key)
        if match is not None:
            scale_key = (
                f"{match.group('layer')}.{match.group('expert')}."
                f"{match.group('projection')}.rank{match.group('rank')}."
                f"scale_{match.group('label')}"
            )
            if scale_key not in key_set:
                raise ValueError(
                    f"MSRT cartridge trellis {key!r} has no scalar scale companion"
                )
            by_layer.setdefault(match.group("layer"), []).append((key, match))
            continue
        companion = _CARTRIDGE_COMPANION_KEY_RE.fullmatch(key)
        if companion is None:
            raise ValueError(f"Malformed MSRT cartridge key {key!r}")
        trellis_key = (
            f"{companion.group('layer')}.{companion.group('expert')}."
            f"{companion.group('projection')}.rank{companion.group('rank')}."
            f"trellis_{companion.group('label')}"
        )
        if trellis_key not in key_set:
            raise ValueError(
                f"Orphaned MSRT cartridge scale {key!r}; missing {trellis_key!r}"
            )
    return by_layer


def _validate_manifest_tensor_coverage(
    config: dict[str, Any],
    by_layer: dict[str, list[tuple[str, re.Match[str]]]],
) -> None:
    observed: dict[str, dict[str, set[int]]] = {}
    observed_layers: set[int] = set()
    observed_experts: set[int] = set()
    for layer_name, entries in by_layer.items():
        layer_match = _LAYER_NAME_RE.fullmatch(layer_name)
        if layer_match is None:
            raise ValueError(f"Invalid MSRT cartridge layer name {layer_name!r}")
        layer_id = int(layer_match.group(1))
        observed_layers.add(layer_id)
        for _, match in entries:
            expert_id = int(match.group("expert"))
            observed_experts.add(expert_id)
            by_stage = observed.setdefault(match.group("label"), {})
            by_stage.setdefault(str(layer_id), set()).add(expert_id)

    declared = {
        label: {layer_id: set(experts) for layer_id, experts in layers.items()}
        for label, layers in config["coverage"].items()
    }
    if observed != declared:
        raise ValueError(
            "MSRT cartridge tensors do not match adapter_config.json coverage"
        )
    if observed_layers != set(config["selected_layers"]):
        raise ValueError(
            "MSRT cartridge tensors do not match adapter_config.json selected_layers"
        )
    if observed_experts != set(config["selected_experts"]):
        raise ValueError(
            "MSRT cartridge tensors do not match adapter_config.json selected_experts"
        )
    stages = config["chain"]
    previous: dict[str, set[int]] | None = None
    for stage in stages:
        label = stage["label"]
        current = declared[label]
        if previous is not None:
            child_only = {
                layer_id: sorted(experts - previous.get(layer_id, set()))
                for layer_id, experts in current.items()
                if experts - previous.get(layer_id, set())
            }
            if child_only:
                raise ValueError(
                    "MSRT cartridge coverage is not parent-closed at "
                    f"stage {label!r}: {child_only}"
                )
        stage_experts = stage["experts"]
        covered_experts = set().union(*current.values())
        if stage_experts != "all" and covered_experts != set(stage_experts):
            raise ValueError(f"MSRT stage {label!r} experts do not match its coverage")
        if stage_experts == "all" and covered_experts != set(
            config["selected_experts"]
        ):
            raise ValueError(
                f"MSRT stage {label!r} declares all experts but coverage is sparse"
            )
        previous = current


def _valid_id_list(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(
            not isinstance(item, bool) and isinstance(item, int) and item >= 0
            for item in value
        )
        and len(set(value)) == len(value)
    )


def _manifest_layer_name_for_live_layer(layer_name: str) -> tuple[str, str]:
    layer_ids = _LIVE_LAYER_INDEX_RE.findall(layer_name)
    if len(layer_ids) != 1:
        raise ValueError(
            "Invalid live EXL3 layer name; expected exactly one numeric "
            f"layers.<index> segment, got {layer_name!r}"
        )
    layer_id = str(int(layer_ids[0]))
    return f"model.layers.{layer_id}.mlp.experts", layer_id


def _read_regular_file(path: Path, *, max_bytes: int, description: str) -> bytes:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"cannot open {description}") from exc
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{description} must be a regular file")
        if metadata.st_size > max_bytes:
            raise ValueError(
                f"{description} exceeds the {max_bytes}-byte verification limit"
            )
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(fd, min(_COPY_CHUNK_BYTES, remaining))
            if not chunk:
                raise ValueError(f"{description} changed while it was read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise ValueError(f"{description} grew while it was read")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _copy_verified_regular_file(
    source: Path,
    destination: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    deadline: float,
) -> None:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        source_fd = os.open(source, flags)
    except OSError as exc:
        raise ValueError("cannot open manifest-listed cartridge shard") from exc
    try:
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("manifest-listed cartridge shard must be a regular file")
        if before.st_size != expected_size:
            raise ValueError(
                "manifest-listed cartridge shard size does not match "
                "adapter_config.json"
            )
        digest = hashlib.sha256()
        with destination.open("xb") as output:
            remaining = expected_size
            while remaining:
                if time.monotonic() > deadline:
                    raise TimeoutError("EXL3 cartridge shard verification timed out")
                chunk = os.read(source_fd, min(_COPY_CHUNK_BYTES, remaining))
                if not chunk:
                    raise ValueError("cartridge shard changed during verification")
                output.write(chunk)
                digest.update(chunk)
                remaining -= len(chunk)
            if os.read(source_fd, 1):
                raise ValueError("cartridge shard grew during verification")
        after = os.fstat(source_fd)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after:
            raise ValueError("cartridge shard changed during verification")
        if digest.hexdigest() != expected_sha256:
            raise ValueError(
                "cartridge shard sha256 does not match adapter_config.json"
            )
    finally:
        os.close(source_fd)


def _validate_adapter_manifest_shape(
    config: dict[str, Any],
    config_path: Path,
) -> None:
    fields = set(config)
    if fields != _ADAPTER_FIELDS:
        missing = sorted(_ADAPTER_FIELDS - fields)
        unexpected = sorted(fields - _ADAPTER_FIELDS)
        raise ValueError(
            f"{config_path}: manifest fields do not match {_ADAPTER_SCHEMA}; "
            f"missing={missing}, unexpected={unexpected}"
        )
    if _LABEL_RE.fullmatch(str(config.get("assembly"))) is None:
        raise ValueError(f"{config_path}: invalid assembly label")
    for field in ("selected_experts", "selected_layers"):
        if not _valid_id_list(config.get(field)):
            raise ValueError(f"{config_path}: {field} must be unique non-negative IDs")
    if not isinstance(config.get("coverage"), dict) or not config["coverage"]:
        raise ValueError(f"{config_path}: coverage must be a non-empty object")
    for label, layers in config["coverage"].items():
        if _LABEL_RE.fullmatch(str(label)) is None or not isinstance(layers, dict):
            raise ValueError(f"{config_path}: invalid coverage entry {label!r}")
        for layer_id, experts in layers.items():
            if (
                not isinstance(layer_id, str)
                or re.fullmatch(r"[0-9]+", layer_id) is None
                or not _valid_id_list(experts)
            ):
                raise ValueError(
                    f"{config_path}: invalid coverage for {label!r}/{layer_id!r}"
                )
    signer = config.get("producer_verified_signer")
    if signer is not None and _SHA256_RE.fullmatch(str(signer)) is None:
        raise ValueError(
            f"{config_path}: producer_verified_signer must be a SHA256 or null"
        )
    campaign = config.get("campaign")
    campaign_fields = {
        "recipe_sha256",
        "base_model",
        "base_revision",
        "encoder_sha256",
        "signer_pubkey",
        "block_size",
        "moe_layers",
    }
    if (
        not isinstance(campaign, dict)
        or set(campaign) != campaign_fields
        or _SHA256_RE.fullmatch(str(campaign.get("recipe_sha256"))) is None
        or not isinstance(campaign.get("base_model"), str)
        or not campaign["base_model"]
        or not isinstance(campaign.get("base_revision"), str)
        or not campaign["base_revision"]
        or (
            campaign.get("encoder_sha256") is not None
            and _SHA256_RE.fullmatch(str(campaign["encoder_sha256"])) is None
        )
        or _SHA256_RE.fullmatch(str(campaign.get("signer_pubkey"))) is None
        or isinstance(campaign.get("block_size"), bool)
        or not isinstance(campaign.get("block_size"), int)
        or campaign["block_size"] < 1
        or not _valid_id_list(campaign.get("moe_layers"))
    ):
        raise ValueError(f"{config_path}: invalid campaign provenance")
    source = config.get("source_assembly")
    if (
        not isinstance(source, dict)
        or set(source) != {"path", "sha256"}
        or not isinstance(source.get("path"), str)
        or not source["path"]
        or Path(source["path"]).is_absolute()
        or ".." in Path(source["path"]).parts
        or _SHA256_RE.fullmatch(str(source.get("sha256"))) is None
    ):
        raise ValueError(f"{config_path}: invalid source_assembly identity")
    if not isinstance(config.get("tool_version"), str) or not config["tool_version"]:
        raise ValueError(f"{config_path}: tool_version must be a non-empty string")
    created_utc = config.get("created_utc")
    if not isinstance(created_utc, str):
        raise ValueError(f"{config_path}: created_utc must be an RFC 3339 time")
    normalized_created_utc = (
        created_utc[:-1] + "+00:00" if created_utc.endswith("Z") else created_utc
    )
    try:
        parsed_created_utc = datetime.fromisoformat(normalized_created_utc)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{config_path}: created_utc must be an RFC 3339 time"
        ) from exc
    if parsed_created_utc.tzinfo is None:
        raise ValueError(f"{config_path}: created_utc must include a timezone")
    num_tensors = config.get("num_tensors")
    if (
        isinstance(num_tensors, bool)
        or not isinstance(num_tensors, int)
        or num_tensors < 1
    ):
        raise ValueError(f"{config_path}: num_tensors must be a positive integer")


def _stage_adapter_contract(
    adapter_path: str,
    layer: Any,
) -> StagedExl3Cartridge:
    requested_input = Path(adapter_path).expanduser()
    if requested_input.is_symlink() or not requested_input.exists():
        raise ValueError("EXL3 cartridge path must exist and must not be a symlink")
    if requested_input.absolute() != requested_input.resolve():
        raise ValueError("EXL3 cartridge path must not traverse symlinks")
    if requested_input.is_dir():
        directory = requested_input.resolve()
        requested: Path | None = None
    elif requested_input.is_file():
        requested = requested_input.resolve()
        directory = requested.parent
    else:
        raise ValueError("EXL3 cartridge path must be a directory or regular file")
    config_path = directory / "adapter_config.json"
    try:
        config = json.loads(
            _read_regular_file(
                config_path,
                max_bytes=_MAX_MANIFEST_BYTES,
                description="EXL3 cartridge adapter_config.json",
            )
        )
    except json.JSONDecodeError as exc:
        raise ValueError("EXL3 cartridge adapter_config.json is invalid JSON") from exc
    if not isinstance(config, dict) or config.get("schema") != _ADAPTER_SCHEMA:
        raise ValueError(
            f"{config_path}: schema must be {_ADAPTER_SCHEMA!r}; "
            "unversioned cartridge tensors are not supported"
        )
    _validate_adapter_manifest_shape(config, config_path)
    if config.get("format") != "exl3-msrt-packed":
        raise ValueError(f"{config_path}: format must be 'exl3-msrt-packed'")
    if config.get("runtime_profile") != _RUNTIME_PROFILE:
        raise ValueError(f"{config_path}: unsupported runtime_profile")
    if config.get("rotation_ownership") != "base":
        raise ValueError(f"{config_path}: rotation_ownership must be 'base'")
    if config.get("standard_lora_compatible") is not False:
        raise ValueError(f"{config_path}: standard_lora_compatible must be false")
    for field, expected in (
        ("runtime_operation", _RUNTIME_OPERATION),
        ("codebook", "mcg"),
        ("mcg_ownership", "adapter-config"),
    ):
        if config.get(field) != expected:
            raise ValueError(f"{config_path}: {field} must be {expected!r}")
    if config.get("mcg_multiplier") != _MCG_MULTIPLIER:
        raise ValueError(f"{config_path}: mcg_multiplier must be {_MCG_MULTIPLIER}")
    if config.get("scale_shape") != []:
        raise ValueError(f"{config_path}: scale_shape must be []")

    base = config.get("base")
    expected_compatibility = getattr(
        layer,
        "exl3_base_compatibility_sha256",
        None,
    )
    if (
        not isinstance(base, dict)
        or set(base) != {"label", "k", "compatibility_sha256", "compatibility_by_layer"}
        or _LABEL_RE.fullmatch(str(base.get("label"))) is None
        or isinstance(base.get("k"), bool)
        or not isinstance(base.get("k"), int)
        or base["k"] not in (2, 3, 4, 5, 6)
        or _SHA256_RE.fullmatch(str(base.get("compatibility_sha256"))) is None
        or not isinstance(base.get("compatibility_by_layer"), dict)
        or set(base["compatibility_by_layer"])
        != {str(index) for index in config["selected_layers"]}
        or any(
            _SHA256_RE.fullmatch(str(digest)) is None
            for digest in base["compatibility_by_layer"].values()
        )
    ):
        raise ValueError(f"{config_path}: invalid base compatibility identity")
    if (
        not isinstance(expected_compatibility, str)
        or _SHA256_RE.fullmatch(expected_compatibility) is None
        or getattr(layer, "exl3_base_compatibility_verified", False) is not True
    ):
        raise ValueError(
            "Loaded EXL3 base has no verified compatibility_sha256; use a "
            "checkpoint emitted for fq-cartridge-adapter/3"
        )
    if base["compatibility_sha256"] != expected_compatibility:
        raise ValueError(
            "EXL3 cartridge was encoded for a different base checkpoint: "
            f"adapter={base['compatibility_sha256']}, "
            f"loaded={expected_compatibility}"
        )

    chain = config.get("chain")
    if not isinstance(chain, list) or not chain:
        raise ValueError(f"{config_path}: chain must be a non-empty list")
    parent = base.get("label")
    labels: list[str] = []
    for stage in chain:
        if (
            not isinstance(stage, dict)
            or set(stage) != {"label", "k", "parent", "experts"}
            or _LABEL_RE.fullmatch(str(stage.get("label"))) is None
            or stage.get("parent") != parent
            or isinstance(stage.get("k"), bool)
            or not isinstance(stage.get("k"), int)
            or stage["k"] not in (1, 2, 3, 4, 5, 6)
        ):
            raise ValueError(f"{config_path}: chain is not an ordered MSRT stage path")
        experts = stage.get("experts")
        if experts != "all" and (
            not isinstance(experts, list)
            or not experts
            or any(
                isinstance(expert, bool) or not isinstance(expert, int) or expert < 0
                for expert in experts
            )
            or len(set(experts)) != len(experts)
        ):
            raise ValueError(
                f"{config_path}: stage {stage.get('label')!r} has an invalid expert set"
            )
        label = stage["label"]
        if label in labels:
            raise ValueError(f"{config_path}: duplicate stage label {label!r}")
        labels.append(label)
        parent = label
    if set(config["coverage"]) != set(labels):
        raise ValueError(f"{config_path}: coverage must describe every chain stage")
    stage_bits = {stage["label"]: stage["k"] for stage in chain}

    tp = config.get("tensor_parallel")
    tp_size = int(getattr(layer, "exl3_tp_size", 1))
    if (
        not isinstance(tp, dict)
        or set(tp) != {"layout", "world_size", "ranks", "axis_by_projection"}
        or tp.get("layout") not in {"full", "rank-sharded"}
        or isinstance(tp.get("world_size"), bool)
        or not isinstance(tp.get("world_size"), int)
        or tp["world_size"] < 1
        or tp.get("ranks") != list(range(tp["world_size"]))
        or tp.get("axis_by_projection") != _TP_AXES
    ):
        raise ValueError(f"{config_path}: invalid tensor_parallel contract")
    if tp["layout"] == "full" and (tp["world_size"], tp["ranks"]) != (1, [0]):
        raise ValueError(f"{config_path}: full layout must contain only rank0")
    if tp["layout"] == "rank-sharded" and tp["world_size"] != tp_size:
        raise ValueError(
            f"{config_path}: cartridge TP topology {tp!r} does not match "
            f"runtime TP={tp_size}"
        )

    shard_entries = config.get("shards")
    if not isinstance(shard_entries, list) or not shard_entries:
        raise ValueError(f"{config_path}: shards must be a non-empty list")
    temporary = tempfile.TemporaryDirectory(prefix="vllm-exl3-cartridge-")
    staged = StagedExl3Cartridge(temporary)
    try:
        staged_root = Path(temporary.name)
        shards: list[Path] = []
        source_shards: list[Path] = []
        total_size = 0
        deadline = time.monotonic() + _STAGING_TIMEOUT_SECONDS
        for entry in shard_entries:
            if (
                not isinstance(entry, dict)
                or set(entry) != {"path", "size", "sha256"}
                or not isinstance(entry.get("path"), str)
                or not entry["path"]
                or Path(entry["path"]).is_absolute()
                or ".." in Path(entry["path"]).parts
                or isinstance(entry.get("size"), bool)
                or not isinstance(entry.get("size"), int)
                or not 0 < entry["size"] <= _MAX_SHARD_BYTES
                or _SHA256_RE.fullmatch(str(entry.get("sha256"))) is None
            ):
                raise ValueError(f"{config_path}: invalid shard entry {entry!r}")
            source = directory / entry["path"]
            if source.is_symlink():
                raise ValueError(
                    "manifest-listed cartridge shards must not be symlinks"
                )
            shard = source.resolve()
            if source.absolute() != shard:
                raise ValueError(
                    "manifest-listed cartridge shard paths must not traverse symlinks"
                )
            if directory != shard and directory not in shard.parents:
                raise ValueError(
                    f"{config_path}: shard path {entry['path']!r} escapes "
                    "the adapter directory"
                )
            if shard == directory:
                raise ValueError(
                    f"{config_path}: shard path names the adapter directory"
                )
            if shard in source_shards:
                raise ValueError(f"{config_path}: duplicate shard {entry['path']!r}")
            total_size += entry["size"]
            if total_size > _MAX_ADAPTER_BYTES:
                raise ValueError("EXL3 cartridge exceeds the total staging size limit")
            destination = staged_root / f"{len(shards):06d}.safetensors"
            _copy_verified_regular_file(
                shard,
                destination,
                expected_size=entry["size"],
                expected_sha256=entry["sha256"],
                deadline=deadline,
            )
            source_shards.append(shard)
            shards.append(destination)
        if requested is not None and requested not in source_shards:
            raise ValueError(f"{requested}: not listed by adjacent adapter_config.json")
        with _SafeTensorCollection(shards) as handle:
            keys = handle.keys()
            by_layer = _index_cartridge_keys(keys)
            _validate_manifest_tensor_coverage(config, by_layer)
            if config["tensor_parallel"]["layout"] == "rank-sharded":
                _validate_rank_scales(
                    handle,
                    by_layer,
                    tuple(config["tensor_parallel"]["ranks"]),
                )
            if len(keys) != config["num_tensors"]:
                raise ValueError(
                    "MSRT cartridge tensor count does not match adapter_config.json"
                )
        staged.state = _AdapterState(
            config=config,
            stage_labels=tuple(labels),
            stage_bits=stage_bits,
            paths=tuple(shards),
            by_layer=by_layer,
            key_count=len(keys),
        )
    except Exception:
        staged.close()
        raise
    return staged


class _SafeTensorCollection:
    """One duplicate-free tensor namespace spanning manifest-listed shards."""

    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths
        self.stack = ExitStack()
        self.key_to_handle: dict[str, Any] = {}

    def __enter__(self) -> _SafeTensorCollection:
        from safetensors import safe_open

        try:
            for path in self.paths:
                handle = self.stack.enter_context(safe_open(str(path), framework="pt"))
                for key in tuple(handle.keys()):
                    if key in self.key_to_handle:
                        raise ValueError(f"Duplicate MSRT cartridge tensor {key!r}")
                    self.key_to_handle[key] = handle
        except Exception:
            self.stack.close()
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stack.__exit__(exc_type, exc_value, traceback)

    def keys(self) -> tuple[str, ...]:
        return tuple(self.key_to_handle)

    def get_tensor(self, key: str) -> torch.Tensor:
        return self.key_to_handle[key].get_tensor(key)


def _validate_rank_scales(
    handle: _SafeTensorCollection,
    by_layer: dict[str, list[tuple[str, re.Match[str]]]],
    ranks: tuple[int, ...],
) -> None:
    if len(ranks) < 2:
        return
    for entries in by_layer.values():
        signatures_by_rank: dict[int, set[tuple[str, int, str]]] = {}
        for _, match in entries:
            signatures_by_rank.setdefault(int(match.group("rank")), set()).add(
                (
                    match.group("label"),
                    int(match.group("expert")),
                    match.group("projection"),
                )
            )
        expected_signature = signatures_by_rank.get(ranks[0], set())
        inconsistent = [
            rank
            for rank in ranks
            if signatures_by_rank.get(rank, set()) != expected_signature
        ]
        if inconsistent:
            raise ValueError(
                "MSRT cartridge TP ranks have inconsistent stage topology: "
                f"ranks={inconsistent}"
            )
        grouped: dict[tuple[str, int, str], dict[int, bytes]] = {}
        for _, match in entries:
            scale_key = (
                f"{match.group('layer')}.{match.group('expert')}."
                f"{match.group('projection')}.rank{match.group('rank')}."
                f"scale_{match.group('label')}"
            )
            scale = handle.get_tensor(scale_key)
            if (
                scale.dtype != torch.float32
                or scale.ndim != 0
                or not math.isfinite(float(scale.item()))
                or float(scale.item()) <= 0
            ):
                raise ValueError(
                    "rank-sharded MSRT scales must be finite positive float32 scalars"
                )
            signature = (
                match.group("label"),
                int(match.group("expert")),
                match.group("projection"),
            )
            grouped.setdefault(signature, {})[int(match.group("rank"))] = (
                scale.view(-1).view(torch.uint8).numpy().tobytes()
            )
        for signature, values in grouped.items():
            if set(values) != set(ranks) or len(set(values.values())) != 1:
                raise ValueError(
                    f"rank-sharded MSRT scales differ across TP ranks: {signature}"
                )


def _build_cartridge_from_entries(
    handle: Any,
    entries: list[tuple[str, re.Match[str]]],
    layer: Any,
    num_experts: int,
    device: torch.device,
    *,
    expected_stage_labels: tuple[str, ...],
    expected_stage_bits: dict[str, int],
    tp_layout: str,
    tp_ranks: tuple[int, ...],
) -> Exl3LoraCartridge | None:
    """Construct one layer's cartridge and select this worker's TP shard."""
    if not entries:
        return None

    entries = _select_rank_entries(
        entries,
        layer,
        layout=tp_layout,
        manifest_ranks=tp_ranks,
    )
    key_set = set(handle.keys())
    observed_labels = {match.group("label") for _, match in entries}
    unknown_labels = observed_labels - set(expected_stage_labels)
    if unknown_labels:
        raise ValueError(
            "MSRT cartridge tensor stages are absent from adapter_config.json: "
            f"{sorted(unknown_labels)}"
        )
    labels = [label for label in expected_stage_labels if label in observed_labels]
    label_to_stage = {label: index for index, label in enumerate(labels)}
    cartridge = Exl3LoraCartridge(len(labels), num_experts, device)
    shards_by_stage_expert: dict[tuple[str, int], set[str]] = {}

    for trellis_key, match in entries:
        expert_id = int(match.group("expert"))
        projection = match.group("projection")
        label = match.group("label")
        if not 0 <= expert_id < num_experts:
            raise ValueError(
                f"MSRT cartridge expert {expert_id} is outside [0, {num_experts})"
            )
        shard_id = _SHARD_MAP[projection]
        scale_key = (
            f"{match.group('layer')}.{match.group('expert')}.{projection}."
            f"rank{match.group('rank')}.scale_{label}"
        )
        trellis = handle.get_tensor(trellis_key)
        if tp_layout == "rank-sharded":
            _validate_stage_trellis(layer, shard_id, trellis)
        else:
            trellis = _slice_full_stage_trellis_for_tp(
                layer,
                shard_id,
                trellis,
            )
        if trellis.shape[2] // 16 != expected_stage_bits[label]:
            raise ValueError(
                f"MSRT stage {label!r} declares K"
                f"{expected_stage_bits[label]} but tensor {trellis_key!r} "
                f"contains K{trellis.shape[2] // 16}"
            )
        if scale_key not in key_set:
            raise ValueError(
                f"MSRT cartridge trellis {trellis_key!r} has no scalar scale companion"
            )
        scale_tensor = handle.get_tensor(scale_key)
        if scale_tensor.dtype != torch.float32 or scale_tensor.ndim != 0:
            raise ValueError(
                "MSRT cartridge scale must be a float32 scalar, got "
                f"shape={tuple(scale_tensor.shape)}, dtype={scale_tensor.dtype}"
            )
        cartridge.set_stage_tensors(
            label_to_stage[label],
            expert_id,
            shard_id,
            trellis,
            float(scale_tensor.item()),
        )
        shards_by_stage_expert.setdefault((label, expert_id), set()).add(shard_id)

    required_shards = {"w1", "w2", "w3"}
    incomplete = {
        key: sorted(required_shards - shards)
        for key, shards in shards_by_stage_expert.items()
        if shards != required_shards
    }
    if incomplete:
        raise ValueError(
            f"Incomplete MSRT cartridge expert stages; missing shards: {incomplete}"
        )
    cartridge.active = True
    return cartridge


def _load_cartridge_from_staged_adapter(
    staged: StagedExl3Cartridge,
    adapter_path: str,
    layer: Any,
    num_experts: int,
    device: torch.device,
) -> Exl3LoraCartridge | None:
    """Load one routed-expert layer from one already-verified adapter."""
    if staged.state is None:
        raise RuntimeError("EXL3 cartridge adapter has not been staged")
    state = staged.state
    config = state.config
    live_layer_name = str(layer.layer_name)
    manifest_layer_name, _ = _manifest_layer_name_for_live_layer(live_layer_name)
    tp = config["tensor_parallel"]
    with _SafeTensorCollection(list(state.paths)) as handle:
        cartridge = _build_cartridge_from_entries(
            handle,
            state.by_layer.get(manifest_layer_name, []),
            layer,
            num_experts,
            device,
            expected_stage_labels=state.stage_labels,
            expected_stage_bits=state.stage_bits,
            tp_layout=tp["layout"],
            tp_ranks=tuple(tp["ranks"]),
        )
    if cartridge is None:
        logger.warning(
            "No MSRT cartridge tensors for %s in %s",
            live_layer_name,
            adapter_path,
        )
        return None
    logger.info(
        "Loaded %d-stage MSRT cartridge for %s from %s",
        cartridge.num_stages,
        live_layer_name,
        adapter_path,
    )
    return cartridge


@torch.inference_mode()
def stage_exl3_cartridge_adapter(
    model: torch.nn.Module, adapter_path: str
) -> StagedExl3Cartridge:
    """Verify and privately stage one adapter without mutating model state."""
    layers = [
        layer
        for layer in model.modules()
        if isinstance(
            getattr(layer, "_exl3_cartridge_runtime", None),
            Exl3CUDAGraphCartridgeRuntime,
        )
        or bool(getattr(layer, "exl3_cartridge_capable", False))
    ]
    if not layers:
        raise RuntimeError("Model is not compatible with EXL3 cartridges")
    staged = _stage_adapter_contract(adapter_path, layers[0])
    assert staged.state is not None
    state = staged.state
    try:
        live_layers = []
        for layer in layers:
            live_layer_name = str(layer.layer_name)
            manifest_layer_name, layer_id = _manifest_layer_name_for_live_layer(
                live_layer_name
            )
            live_layers.append((layer, live_layer_name, manifest_layer_name, layer_id))
        model_layer_names = {entry[2] for entry in live_layers}
        selected = set(state.by_layer)
        local_selected = selected & model_layer_names
        if not local_selected:
            raise ValueError(
                "MSRT cartridge does not select any layer in this local model"
            )
        checkpoint_compatibility = getattr(
            layers[0], "exl3_base_compatibility_by_layer", None
        )
        if not isinstance(checkpoint_compatibility, dict):
            raise ValueError("Loaded EXL3 base has no per-layer compatibility identity")
        for selected_layer_name in selected:
            selected_match = _LAYER_NAME_RE.fullmatch(selected_layer_name)
            if selected_match is None:
                raise ValueError(
                    f"Invalid MSRT cartridge layer name {selected_layer_name!r}"
                )
            layer_id = selected_match.group(1)
            if checkpoint_compatibility.get(layer_id) != state.config["base"][
                "compatibility_by_layer"
            ].get(layer_id):
                raise ValueError(
                    "EXL3 cartridge base compatibility identity differs at "
                    f"{selected_layer_name}"
                )
        staged.local_layer_names = tuple(sorted(local_selected))
        tp = state.config["tensor_parallel"]
        with _SafeTensorCollection(list(state.paths)) as handle:
            for layer, live_layer_name, manifest_layer_name, layer_id in live_layers:
                if manifest_layer_name not in local_selected:
                    continue
                if (
                    getattr(layer, "exl3_base_compatibility_verified", False)
                    is not True
                    or getattr(layer, "exl3_base_compatibility_sha256", None)
                    != state.config["base"]["compatibility_sha256"]
                    or getattr(layer, "exl3_base_layer_compatibility_sha256", None)
                    != state.config["base"]["compatibility_by_layer"].get(layer_id)
                ):
                    raise ValueError(
                        "EXL3 cartridge base compatibility identity differs "
                        f"at {live_layer_name}"
                    )
                layer_bitrates = tuple(
                    int(value) for value in getattr(layer, "exl3_layer_bitrates", ())
                )
                if (
                    not layer_bitrates
                    or len(set(layer_bitrates)) != 1
                    or layer_bitrates[0] != state.config["base"]["k"]
                ):
                    raise ValueError(
                        f"EXL3 cartridge base.k does not match {live_layer_name}"
                    )
                cartridge = _build_cartridge_from_entries(
                    handle,
                    state.by_layer[manifest_layer_name],
                    layer,
                    int(layer.local_num_experts),
                    torch.device("cpu"),
                    expected_stage_labels=state.stage_labels,
                    expected_stage_bits=state.stage_bits,
                    tp_layout=tp["layout"],
                    tp_ranks=tuple(tp["ranks"]),
                )
                if cartridge is None:
                    raise ValueError(f"Cartridge has no tensors for {live_layer_name}")
                cartridge.clear()
        return staged
    except Exception:
        staged.close()
        raise


@torch.inference_mode()
def prepare_staged_exl3_cartridge_into_model(
    model: torch.nn.Module, staged: StagedExl3Cartridge
) -> int:
    """Materialize one verified adapter during the drained transaction."""

    if staged.state is None:
        raise RuntimeError("EXL3 cartridge adapter has not been staged")
    state = staged.state
    config = state.config
    layers = [
        layer
        for layer in model.modules()
        if isinstance(
            getattr(layer, "_exl3_cartridge_runtime", None),
            Exl3CUDAGraphCartridgeRuntime,
        )
        or bool(getattr(layer, "exl3_cartridge_capable", False))
    ]
    if not layers:
        return 0
    selected = set(state.by_layer)
    for layer in layers:
        live_layer_name = str(layer.layer_name)
        manifest_layer_name, _ = _manifest_layer_name_for_live_layer(live_layer_name)
        runtime = getattr(layer, "_exl3_cartridge_runtime", None)
        if (
            isinstance(runtime, Exl3CUDAGraphCartridgeRuntime)
            and manifest_layer_name not in selected
        ):
            runtime.deactivate()
            layer.exl3_cartridge_enabled = False
            del layer._exl3_cartridge_runtime
    prepared_count = 0
    try:
        tp = config["tensor_parallel"]
        with _SafeTensorCollection(list(state.paths)) as handle:
            for layer in layers:
                live_layer_name = str(layer.layer_name)
                manifest_layer_name, _ = _manifest_layer_name_for_live_layer(
                    live_layer_name
                )
                if manifest_layer_name not in state.by_layer:
                    continue
                runtime = getattr(layer, "_exl3_cartridge_runtime", None)
                if not isinstance(runtime, Exl3CUDAGraphCartridgeRuntime):
                    runtime = prepare_exl3_cudagraph_cartridge_runtime(
                        layer, _model_workspace(model, layer)
                    )
                cartridge = _build_cartridge_from_entries(
                    handle,
                    state.by_layer[manifest_layer_name],
                    layer,
                    runtime.num_experts,
                    torch.device("cpu"),
                    expected_stage_labels=state.stage_labels,
                    expected_stage_bits=state.stage_bits,
                    tp_layout=tp["layout"],
                    tp_ranks=tuple(tp["ranks"]),
                )
                if cartridge is None:
                    raise ValueError(f"Cartridge has no tensors for {live_layer_name}")
                logger.info(
                    "Loaded %d-stage MSRT cartridge for %s from %s",
                    cartridge.num_stages,
                    live_layer_name,
                    "verified private staging",
                )
                runtime.deactivate()
                cartridge.to(runtime.device)
                try:
                    runtime.materialize(layer, cartridge)
                finally:
                    cartridge.clear()
                prepared_count += 1
        return prepared_count
    except Exception:
        deactivate_exl3_cartridge(model)
        raise


@torch.inference_mode()
def activate_exl3_cartridge(model: torch.nn.Module) -> int:
    """Activate the fully prepared cartridge on every local layer."""
    updated = 0
    for layer in model.modules():
        runtime = getattr(layer, "_exl3_cartridge_runtime", None)
        if isinstance(runtime, Exl3CUDAGraphCartridgeRuntime):
            runtime.activate()
            layer.exl3_cartridge_enabled = True
            updated += 1
    return updated


@torch.inference_mode()
def load_exl3_cartridge_into_model(model: torch.nn.Module, adapter_path: str) -> int:
    """Prepare and activate a cartridge in a quiescent single-worker model."""
    with stage_exl3_cartridge_adapter(model, adapter_path) as staged:
        updated = prepare_staged_exl3_cartridge_into_model(model, staged)
    if updated == 0:
        raise RuntimeError("Model has no prepared EXL3 cartridge runtime")
    activated = activate_exl3_cartridge(model)
    if activated != updated:
        deactivate_exl3_cartridge(model)
        raise RuntimeError(
            f"Prepared {updated} EXL3 cartridge layers but activated {activated}"
        )
    return updated


def has_exl3_cartridge(model: torch.nn.Module) -> bool:
    """Return whether this worker owns any packed cartridge runtime."""
    return any(
        isinstance(
            getattr(layer, "_exl3_cartridge_runtime", None),
            Exl3CUDAGraphCartridgeRuntime,
        )
        for layer in model.modules()
    )


@torch.inference_mode()
def deactivate_exl3_cartridge(model: torch.nn.Module) -> int:
    """Select the compressed base path and release every cartridge runtime."""
    updated = 0
    for layer in model.modules():
        runtime = getattr(layer, "_exl3_cartridge_runtime", None)
        if isinstance(runtime, Exl3CUDAGraphCartridgeRuntime):
            runtime.deactivate()
            layer.exl3_cartridge_enabled = False
            del layer._exl3_cartridge_runtime
            updated += 1
    if hasattr(model, "_exl3_cartridge_workspaces"):
        del model._exl3_cartridge_workspaces
    return updated


__all__ = [
    "Exl3CUDAGraphCartridgeRuntime",
    "Exl3LoraCartridge",
    "activate_exl3_cartridge",
    "apply_exl3_cudagraph_cartridge",
    "deactivate_exl3_cartridge",
    "has_exl3_cartridge",
    "load_exl3_cartridge_into_model",
    "prepare_exl3_cudagraph_cartridge_runtime",
    "prepare_staged_exl3_cartridge_into_model",
    "stage_exl3_cartridge_adapter",
]
