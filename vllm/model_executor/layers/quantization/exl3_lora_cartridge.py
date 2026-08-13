# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CUDA-graph-safe EXL3 cartridge runtime for MSRT additive quantization.

MSRT (Multi-Stage Rescaled Trellis) cartridges contain full-rank,
trellis-quantized residuals for the gate, up, and down expert projections.
Applying those residuals projection-by-projection is exact but cannot be added
after a fused MoE call: gate and up residuals must be present before SiLU.

For rank-sliced EXL3, this module lazily materializes the combined
base+cartridge expert matrices into fixed-address GPU buffers. Cartridge load
quiesces the engine, drops the base CUDA graphs, allocates the dense buffers,
and captures cartridge graphs. Deactivation performs the inverse transition
and releases the buffers before recapturing the compressed base path.

The runtime is opt-in because an active cartridge's dense shadow weights
consume substantially more GPU memory. Inactive cartridge support has no
shadow buffers, dense-MoE work, or alternate graph path. Set
``cartridge_runtime: true`` in the EXL3 quantization configuration to allow
the quiescent recapture transaction. Use
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

from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.quantization.exl3 import Exl3MoEMethod, _exl3_gemm

logger = logging.getLogger(__name__)

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
    """Fixed-address dense shadow weights used by active CUDA graphs.

    There is one runtime per routed-expert layer while a cartridge is loaded.
    ``w13`` and ``w2`` remain fixed for the lifetime of those captured graphs.
    """

    def __init__(self, layer: Any):
        tp_size = int(getattr(layer, "exl3_tp_size", 1))
        if tp_size != 1:
            raise NotImplementedError(
                "EXL3 cartridge runtime currently requires tensor_parallel_size=1"
            )
        num_experts = int(layer.local_num_experts)
        hidden_size = int(layer.exl3_hidden_size)
        intermediate_size = int(layer.exl3_intermediate_size_per_partition)
        dtype = torch.float16
        device = layer.w13_trellis.device

        self.w13 = torch.zeros(
            (num_experts, 2 * intermediate_size, hidden_size),
            dtype=dtype,
            device=device,
        )
        self.w2 = torch.zeros(
            (num_experts, hidden_size, intermediate_size),
            dtype=dtype,
            device=device,
        )
        self.active = torch.zeros((), dtype=torch.bool, device=device)
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dtype = dtype
        self.device = device
        self._materialized = False

    def deactivate(self) -> None:
        """Select the rank-sliced base path without changing graph pointers."""
        self.active.zero_()

    def activate(self) -> None:
        """Select the fully materialized dense cartridge path."""
        if not self._materialized:
            raise RuntimeError("cannot activate an unmaterialized EXL3 cartridge")
        self.active.fill_(True)

    @torch.inference_mode()
    def materialize(self, layer: Any, cartridge: Exl3LoraCartridge) -> None:
        """Decode exact base+cartridge matrices into the shadow buffers.

        Identity activations turn each EXL3 GEMM into a matrix materialization.
        Base and residual outputs are summed before transposition, preserving
        the exact projection-level MSRT semantics across the SiLU boundary.
        """
        if cartridge.num_experts != self.num_experts:
            raise ValueError(
                "cartridge expert count does not match runtime: "
                f"{cartridge.num_experts} != {self.num_experts}"
            )
        if not cartridge.active:
            raise ValueError("cannot materialize an inactive cartridge")

        self._materialized = False
        self.deactivate()
        hidden_identity = torch.eye(
            self.hidden_size, dtype=torch.float16, device=self.device
        )
        intermediate_identity = torch.eye(
            self.intermediate_size, dtype=torch.float16, device=self.device
        )

        for expert_id in range(self.num_experts):
            for shard_id, offset in (("w1", 0), ("w3", self.intermediate_size)):
                base = Exl3MoEMethod._apply_expert(
                    layer, "w13", hidden_identity, expert_id, shard_id
                )
                combined = apply_exl3_cartridge(
                    base,
                    hidden_identity,
                    layer,
                    "w13",
                    expert_id,
                    shard_id,
                    cartridge,
                )
                self.w13[expert_id, offset : offset + self.intermediate_size].copy_(
                    combined.T.to(self.dtype)
                )

            base = Exl3MoEMethod._apply_expert(
                layer, "w2", intermediate_identity, expert_id, "w2"
            )
            combined = apply_exl3_cartridge(
                base,
                intermediate_identity,
                layer,
                "w2",
                expert_id,
                "w2",
                cartridge,
            )
            self.w2[expert_id].copy_(combined.T.to(self.dtype))

        for weights in (self.w13, self.w2):
            minimum, maximum = torch.aminmax(weights)
            if not torch.isfinite(minimum).item() or not torch.isfinite(maximum).item():
                raise ValueError(
                    "materialized EXL3 cartridge contains non-finite weights"
                )
        self._materialized = True

        # Activation is a separate model-wide commit step. Keeping this runtime
        # inactive until every layer materializes prevents mixed-model states.


def prepare_exl3_cudagraph_cartridge_runtime(
    layer: Any,
) -> Exl3CUDAGraphCartridgeRuntime:
    """Allocate a layer's fixed-address cartridge buffers before capture."""
    runtime = getattr(layer, "_exl3_cartridge_runtime", None)
    if runtime is None:
        runtime = Exl3CUDAGraphCartridgeRuntime(layer)
        layer._exl3_cartridge_runtime = runtime
    return runtime


def apply_exl3_cartridge(
    base_output: torch.Tensor,
    x: torch.Tensor,
    layer: Any,
    group: str,
    expert_id: int,
    shard_id: str,
    cartridge: Exl3LoraCartridge | None,
) -> torch.Tensor:
    """Add every residual stage to one projection output."""
    del layer, group
    if cartridge is None or not cartridge.active:
        return base_output

    output = base_output.float()
    for stage_idx in range(cartridge.num_stages):
        stage_tensors = cartridge.get_stage_tensors(stage_idx, expert_id, shard_id)
        if stage_tensors is None:
            continue

        trellis = stage_tensors["trellis"]
        suh = stage_tensors["suh"]
        svh = stage_tensors["svh"]
        scale = stage_tensors["scale"]
        assert isinstance(trellis, torch.Tensor)
        assert isinstance(suh, torch.Tensor)
        assert isinstance(svh, torch.Tensor)
        assert isinstance(scale, float)

        packed_k = trellis.shape[0] * 16
        if x.shape[-1] > packed_k:
            raise ValueError(
                f"MSRT cartridge input width {x.shape[-1]} exceeds packed K={packed_k}"
            )
        padded_x = x
        if x.shape[-1] < packed_k:
            padded_x = torch.nn.functional.pad(x, (0, packed_k - x.shape[-1]))
        # Rank-sliced EXL3 metadata and the fq-cartridge/1 producer both
        # require MCG; cartridge files do not override the base codebook.
        correction = _exl3_gemm(padded_x, trellis, suh, svh, True, False)
        logical_n = base_output.shape[-1]
        if correction.shape[-1] < logical_n:
            raise ValueError(
                "MSRT cartridge packed output width "
                f"{correction.shape[-1]} is below logical N={logical_n}"
            )
        output = output + correction[..., :logical_n].float() * (1.0 / scale)

    return output


def apply_exl3_cudagraph_cartridge(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    layer: Any,
) -> torch.Tensor:
    """Run the dense cartridge path captured for the active topology."""
    runtime = getattr(layer, "_exl3_cartridge_runtime", None)
    if not isinstance(runtime, Exl3CUDAGraphCartridgeRuntime):
        raise RuntimeError(
            "EXL3 cartridge runtime was not prepared before CUDA graph capture"
        )

    return fused_experts(
        x.to(runtime.dtype),
        runtime.w13,
        runtime.w2,
        topk_weights,
        topk_ids,
        activation=layer.activation,
        apply_router_weight_on_input=False,
        global_num_experts=runtime.num_experts,
    ).to(x.dtype)


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
    """Lazily allocate and materialize a cartridge without activating it."""
    prepared: list[tuple[Any, Exl3CUDAGraphCartridgeRuntime, Exl3LoraCartridge]] = []
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
            prepared.append((layer, runtime, cartridge))
        if not prepared:
            return 0

        prepared_count = len(prepared)
        for _, runtime, _ in prepared:
            runtime.deactivate()
        for layer, runtime, cartridge in prepared:
            cartridge.to(runtime.device)
            try:
                runtime.materialize(layer, cartridge)
            finally:
                cartridge.clear()
        return prepared_count
    except Exception:
        deactivate_exl3_cartridge(model)
        raise
    finally:
        prepared.clear()


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
    """Return whether this worker owns any dense cartridge runtime."""
    return any(
        isinstance(
            getattr(layer, "_exl3_cartridge_runtime", None),
            Exl3CUDAGraphCartridgeRuntime,
        )
        for layer in model.modules()
    )


@torch.inference_mode()
def deactivate_exl3_cartridge(model: torch.nn.Module) -> int:
    """Select the compressed base path and release every dense runtime."""
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
    "apply_exl3_cartridge",
    "apply_exl3_cudagraph_cartridge",
    "deactivate_exl3_cartridge",
    "has_exl3_cartridge",
    "load_cartridge_from_adapter",
    "load_exl3_cartridge_into_model",
    "prepare_exl3_cartridge_into_model",
    "prepare_exl3_cudagraph_cartridge_runtime",
]
