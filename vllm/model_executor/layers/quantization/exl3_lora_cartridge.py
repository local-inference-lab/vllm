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
alternate graph path, or runtime overhead. Set ``cartridge_runtime: true`` in
the EXL3 quantization configuration to allow the quiescent recapture
transaction. Use
``llm_engine.load_exl3_cartridge(path)`` and
``llm_engine.deactivate_exl3_cartridge()``; both dispatch through the worker
control plane to every model worker.

The initial implementation supports tensor parallel size one and one
model-wide slot; per-request cartridge selection is intentionally not claimed.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

import torch

from vllm.model_executor.layers.quantization.exl3 import _load_exl3_ext

logger = logging.getLogger(__name__)


def _load_additive_exl3_ext() -> Any:
    ext = _load_exl3_ext()
    if not hasattr(ext, "exl3_moe_additive_fused"):
        raise RuntimeError(
            "packed EXL3 cartridges require an ExLlamaV3 extension that "
            "exports exl3_moe_additive_fused"
        )
    return ext


_CARTRIDGE_KEY_RE = re.compile(
    r"^(?P<layer>.+\.experts)\."
    r"(?P<expert>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\."
    r"rank(?P<rank>\d+)\.trellis_(?P<label>.+)$"
)
_CARTRIDGE_COMPANION_KEY_RE = re.compile(
    r"^(?P<layer>.+\.experts)\."
    r"(?P<expert>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\."
    r"rank(?P<rank>\d+)\.(?:suh|svh|scale|mcg|mul1)_(?P<label>.+)$"
)
_SHARD_MAP = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}


class Exl3LoraCartridge:
    """MSRT residual tensors for one routed-expert layer.

    Each stage is keyed by ``(expert_id, shard_id)`` and stores packed trellis
    indices, input/output Hadamard vectors, and the positive rescaling factor
    used by the MCG-only rank-sliced encoder.
    """

    def __init__(self, num_stages: int, num_experts: int, device: torch.device):
        if num_stages < 1:
            raise ValueError(f"num_stages must be positive, got {num_stages}")
        if num_experts < 1:
            raise ValueError(f"num_experts must be positive, got {num_experts}")
        self.num_stages = num_stages
        self.num_experts = num_experts
        self.device = device
        self.stages: list[dict[tuple[int, str], dict[str, torch.Tensor | float]]] = [
            {} for _ in range(num_stages)
        ]
        self.active = False

    def set_stage_tensors(
        self,
        stage_idx: int,
        expert_id: int,
        shard_id: str,
        trellis: torch.Tensor,
        suh: torch.Tensor,
        svh: torch.Tensor,
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
        self.stages[stage_idx][(expert_id, shard_id)] = {
            "trellis": trellis.to(self.device).contiguous(),
            "suh": suh.to(self.device).contiguous(),
            "svh": svh.to(self.device).contiguous(),
            "scale": float(scale),
        }

    def get_stage_tensors(
        self, stage_idx: int, expert_id: int, shard_id: str
    ) -> dict[str, torch.Tensor | float] | None:
        """Return one stage of one expert projection, if present."""
        return self.stages[stage_idx].get((expert_id, shard_id))

    def to(self, device: torch.device) -> None:
        """Move source tensors to one device without changing stage metadata."""
        for stage in self.stages:
            for tensors in stage.values():
                for name in ("trellis", "suh", "svh"):
                    tensor = tensors[name]
                    assert isinstance(tensor, torch.Tensor)
                    tensors[name] = tensor.to(device).contiguous()
        self.device = device

    def clear(self) -> None:
        """Release source cartridge tensors."""
        self.stages = [{} for _ in range(self.num_stages)]
        self.active = False


class Exl3CUDAGraphCartridgeRuntime:
    """Fixed-address packed cartridge tensors and additive MoE workspaces."""

    def __init__(self, layer: Any):
        tp_size = int(getattr(layer, "exl3_tp_size", 1))
        if tp_size != 1:
            raise NotImplementedError(
                "EXL3 cartridge runtime currently requires tensor_parallel_size=1"
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
        self.xh = torch.empty(
            (self.chunk, self.hidden_size), dtype=self.dtype, device=self.device
        )
        self.out32 = torch.empty(
            (self.chunk, self.hidden_size), dtype=torch.float32, device=self.device
        )
        self.tg = torch.empty(
            (concurrency, self.chunk, self.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        self.tu = torch.empty_like(self.tg)
        self.ig = torch.empty(
            (concurrency, self.chunk, self.intermediate_size),
            dtype=self.dtype,
            device=self.device,
        )
        self.iu = torch.empty_like(self.ig)
        self.expert_count = torch.empty(
            self.num_experts + 1, dtype=torch.int64, device=self.device
        )
        self.expert_offsets = torch.empty_like(self.expert_count)
        self.token_sorted = torch.empty(
            self.chunk * self.topk, dtype=torch.int64, device=self.device
        )
        self.weight_sorted = torch.empty(
            self.chunk * self.topk, dtype=self.dtype, device=self.device
        )
        self.expert_map = torch.arange(
            self.num_experts, dtype=torch.int64, device=self.device
        )
        self._active = False
        self._materialized = False
        self._packed_tensors: tuple[torch.Tensor, ...] = ()
        self.pointer_args: tuple[torch.Tensor, ...] = ()
        self.max_residual_bits = 0
        layer_bitrates = tuple(getattr(layer, "exl3_layer_bitrates", ()))
        if len(set(layer_bitrates)) != 1:
            raise ValueError("packed cartridge runtime requires a uniform base bitrate")
        self.base_bits = int(layer_bitrates[0])

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
            xh = self.xh[:rows]
            xh.copy_(x[start : start + rows])
            out32 = self.out32[:rows]
            out32.zero_()
            route_count = rows * self.topk
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
            )
            outputs.append(out32.to(x.dtype))
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs)


def prepare_exl3_cudagraph_cartridge_runtime(
    layer: Any,
) -> Exl3CUDAGraphCartridgeRuntime:
    """Allocate a layer's fixed-address cartridge buffers before capture."""
    runtime = getattr(layer, "_exl3_cartridge_runtime", None)
    if runtime is None:
        runtime = Exl3CUDAGraphCartridgeRuntime(layer)
        layer._exl3_cartridge_runtime = runtime
    return runtime


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


def _stage_sort_key(label: str) -> tuple[str, int]:
    prefix, separator, suffix = label.rpartition("_")
    if separator and suffix.isdigit():
        return prefix, int(suffix)
    match = re.match(r"^(.*?)(\d+)$", label)
    if match is not None:
        return match.group(1), int(match.group(2))
    return label, -1


def _validate_stage_tensors(
    layer: Any,
    shard_id: str,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
) -> None:
    hidden_size = int(layer.exl3_hidden_size)
    intermediate_size = int(layer.exl3_intermediate_size_per_partition)
    input_size, output_size = (
        (intermediate_size, hidden_size)
        if shard_id == "w2"
        else (hidden_size, intermediate_size)
    )
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
    for name, tensor, expected_size in (
        ("suh", suh, packed_k),
        ("svh", svh, packed_n),
    ):
        if tensor.dtype != torch.float16 or tensor.shape != (expected_size,):
            raise ValueError(
                f"Invalid MSRT cartridge {name}: shape={tuple(tensor.shape)}, "
                f"dtype={tensor.dtype}, expected=({expected_size},), "
                "dtype=torch.float16"
            )
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"MSRT cartridge {name} contains non-finite values")


def _validate_base_rotations(
    layer: Any,
    expert_id: int,
    shard_id: str,
    suh: torch.Tensor,
    svh: torch.Tensor,
) -> None:
    group = "w2" if shard_id == "w2" else "w13"
    key = (expert_id, shard_id)
    for name, cartridge_tensor in (("suh", suh), ("svh", svh)):
        parameter = getattr(layer, f"{group}_{name}")
        base_tensor = parameter.exl3_tensors[key]
        if cartridge_tensor.device != base_tensor.device:
            cartridge_tensor = cartridge_tensor.to(base_tensor.device)
        if not torch.equal(cartridge_tensor, base_tensor):
            raise ValueError(
                "MSRT cartridge rotations must match the rank-sliced EXL3 base: "
                f"expert={expert_id}, projection={shard_id}, tensor={name}"
            )


def load_cartridge_from_adapter(
    adapter_path: str,
    layer: Any,
    num_experts: int,
    device: torch.device,
) -> Exl3LoraCartridge | None:
    """Load the stages belonging to one routed-expert layer.

    Expected tensor names are
    ``{layer}.<expert>.<projection>.rank0.{trellis,suh,svh,scale}_<label>``.
    Keys for other layers are ignored rather than accidentally overwriting the
    same expert/shard entries.
    """
    from safetensors import safe_open

    layer_name = str(layer.layer_name)
    entries: list[tuple[str, re.Match[str]]] = []
    with safe_open(adapter_path, framework="pt") as handle:
        keys = tuple(handle.keys())
        key_set = set(keys)
        for key in keys:
            match = _CARTRIDGE_KEY_RE.match(key)
            if match is not None and match.group("layer") == layer_name:
                entries.append((key, match))
                continue
            companion = _CARTRIDGE_COMPANION_KEY_RE.match(key)
            if key.startswith(f"{layer_name}.") and (
                companion is None or companion.group("layer") != layer_name
            ):
                raise ValueError(f"Malformed MSRT cartridge key {key!r}")

        if not entries:
            logger.warning(
                "No MSRT cartridge tensors for %s in %s",
                layer_name,
                adapter_path,
            )
            return None

        labels = sorted(
            {match.group("label") for _, match in entries},
            key=_stage_sort_key,
        )
        label_to_stage = {label: index for index, label in enumerate(labels)}
        cartridge = Exl3LoraCartridge(len(labels), num_experts, device)
        shards_by_stage_expert: dict[tuple[str, int], set[str]] = {}

        for trellis_key, match in entries:
            rank = int(match.group("rank"))
            if rank != 0:
                raise ValueError(
                    f"MSRT cartridge only supports rank0 tensors, got rank{rank}"
                )
            expert_id = int(match.group("expert"))
            projection = match.group("projection")
            label = match.group("label")
            if not 0 <= expert_id < num_experts:
                raise ValueError(
                    f"MSRT cartridge expert {expert_id} is outside [0, {num_experts})"
                )
            shard_id = _SHARD_MAP[projection]
            prefix = trellis_key[: -len(f"trellis_{label}")]
            companion_keys = {
                "suh": f"{prefix}suh_{label}",
                "svh": f"{prefix}svh_{label}",
                "scale": f"{prefix}scale_{label}",
            }
            missing = [
                key_name
                for key_name in (companion_keys["suh"], companion_keys["svh"])
                if key_name not in key_set
            ]
            if missing:
                raise ValueError(
                    f"Incomplete MSRT cartridge entry {trellis_key}: missing {missing}"
                )
            trellis = handle.get_tensor(trellis_key)
            suh = handle.get_tensor(companion_keys["suh"])
            svh = handle.get_tensor(companion_keys["svh"])
            _validate_stage_tensors(layer, shard_id, trellis, suh, svh)
            _validate_base_rotations(layer, expert_id, shard_id, suh, svh)
            scale = 1.0
            if companion_keys["scale"] in key_set:
                scale_tensor = handle.get_tensor(companion_keys["scale"])
                if scale_tensor.numel() != 1:
                    raise ValueError(
                        f"MSRT cartridge scale must be scalar, got "
                        f"shape={tuple(scale_tensor.shape)}"
                    )
                scale = float(scale_tensor.item())
            cartridge.set_stage_tensors(
                label_to_stage[label],
                expert_id,
                shard_id,
                trellis,
                suh,
                svh,
                scale,
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
    logger.info(
        "Loaded %d-stage MSRT cartridge for %s from %s",
        cartridge.num_stages,
        layer_name,
        adapter_path,
    )
    return cartridge


@torch.inference_mode()
def prepare_exl3_cartridge_into_model(model: torch.nn.Module, adapter_path: str) -> int:
    """Load each layer into packed storage without a dense or staging peak."""
    prepared_count = 0
    try:
        for layer in model.modules():
            runtime = getattr(layer, "_exl3_cartridge_runtime", None)
            if not isinstance(runtime, Exl3CUDAGraphCartridgeRuntime):
                if not bool(getattr(layer, "exl3_cartridge_capable", False)):
                    continue
                runtime = prepare_exl3_cudagraph_cartridge_runtime(layer)
            cartridge = load_cartridge_from_adapter(
                adapter_path,
                layer,
                runtime.num_experts,
                torch.device("cpu"),
            )
            if cartridge is None:
                raise ValueError(
                    f"Cartridge {adapter_path} has no tensors for {layer.layer_name}"
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
    updated = prepare_exl3_cartridge_into_model(model, adapter_path)
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
    return updated


__all__ = [
    "Exl3CUDAGraphCartridgeRuntime",
    "Exl3LoraCartridge",
    "activate_exl3_cartridge",
    "apply_exl3_cudagraph_cartridge",
    "deactivate_exl3_cartridge",
    "has_exl3_cartridge",
    "load_cartridge_from_adapter",
    "load_exl3_cartridge_into_model",
    "prepare_exl3_cartridge_into_model",
    "prepare_exl3_cudagraph_cartridge_runtime",
]
