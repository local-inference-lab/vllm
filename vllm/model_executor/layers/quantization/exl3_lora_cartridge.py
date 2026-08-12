# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""EXL3 LoRA cartridge support for MSRT additive quantization.

This module extends EXL3 MoE quantization to support MSRT (Multi-Stage Rescaled
Trellis) additive cartridge adapters. Unlike standard LoRA (low-rank A·B
decomposition), MSRT cartridges contain full-rank trellis-quantized residual
weights that are applied as additional exl3_gemm passes summed with the base
output.

The cartridge is loaded via vLLM's LoRA hot-swap API (add_lora / remove_lora)
and applied at runtime without model reload. Each cartridge stage adds one
exl3_gemm pass per expert; the outputs are summed with per-stage rescaling.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from vllm.model_executor.layers.quantization.exl3 import (
    Exl3Config,
    Exl3MoEMethod,
    _exl3_gemm,
)
from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase

logger = logging.getLogger(__name__)


def exl3_get_supported_lora_modules() -> list[str]:
    """Return the module names that support EXL3 LoRA cartridges.

    EXL3 quantizes MoE expert projections (gate_proj, up_proj, down_proj).
    These are the modules that can receive additive cartridge corrections.
    """
    return ["gate_proj", "up_proj", "down_proj"]


def exl3_lora_can_replace_layer(
    source_layer: Any,
    lora_config: Any,
    packed_modules_list: list[str],
    model_config: Any,
) -> bool:
    """Check if a layer can be replaced with an EXL3 LoRA-wrapped version.

    This replaces the standard FusedMoEWithLoRA check for EXL3 layers.
    Returns True when the source layer uses EXL3 MoE quantization.
    """
    # EXL3 LoRA works with the monolithic per-expert path, not the modular kernel
    # path. We check if the layer's quant method is Exl3MoEMethod.
    quant_method = getattr(getattr(source_layer, "routed_experts", source_layer),
                          "quant_method", None)
    if isinstance(quant_method, Exl3MoEMethod):
        return True
    return False


class Exl3LoraCartridge:
    """Holds MSRT cartridge tensors for one MoE layer.

    Each cartridge stage contains:
    - trellis: int16 packed trellis indices (k//16, n//16, K*16)
    - suh: float16 row-side Hadamard scales
    - svh: float16 column-side Hadamard scales
    - scale: float32 rescaling factor (codebook_scale / RMS(residual))

    The cartridge is applied as:
      output += (1/scale) * exl3_gemm(x, trellis, suh, svh, mcg=True)
    """

    def __init__(self, num_stages: int, num_experts: int, device: torch.device):
        self.num_stages = num_stages
        self.num_experts = num_experts
        self.device = device
        # Per-stage storage: list of dicts keyed by (expert_id, shard_id)
        self.stages: list[dict[tuple[int, str], dict[str, torch.Tensor]]] = [
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
        """Set cartridge tensors for one expert's projection at one stage."""
        self.stages[stage_idx][(expert_id, shard_id)] = {
            "trellis": trellis.to(self.device),
            "suh": suh.to(self.device),
            "svh": svh.to(self.device),
            "scale": scale,
        }

    def get_stage_tensors(
        self, stage_idx: int, expert_id: int, shard_id: str
    ) -> dict[str, torch.Tensor] | None:
        """Get cartridge tensors for one expert's projection at one stage."""
        return self.stages[stage_idx].get((expert_id, shard_id))

    def clear(self) -> None:
        """Remove all cartridge tensors."""
        self.stages = [{} for _ in range(self.num_stages)]
        self.active = False


def apply_exl3_cartridge(
    base_output: torch.Tensor,
    x: torch.Tensor,
    layer: Any,
    group: str,
    expert_id: int,
    shard_id: str,
    cartridge: Exl3LoraCartridge | None,
) -> torch.Tensor:
    """Apply MSRT cartridge correction to a base EXL3 GEMM output.

    Args:
        base_output: Output from the base EXL3 GEMM (K2 or K3 tier)
        x: Input activations (float16, padded to packed_k)
        layer: The RoutedExperts layer (for accessing trellis tensors)
        group: Weight group ("w13" or "w2")
        expert_id: Expert index
        shard_id: Projection shard ("w1", "w3", or "w2")
        cartridge: The cartridge to apply, or None

    Returns:
        Corrected output: base_output + sum of cartridge stage corrections
    """
    if cartridge is None or not cartridge.active:
        return base_output

    key = (expert_id, shard_id)
    output = base_output

    for stage_idx in range(cartridge.num_stages):
        stage_tensors = cartridge.get_stage_tensors(stage_idx, expert_id, shard_id)
        if stage_tensors is None:
            continue

        trellis = stage_tensors["trellis"]
        suh = stage_tensors["suh"]
        svh = stage_tensors["svh"]
        scale = stage_tensors["scale"]

        # Run additional EXL3 GEMM for this cartridge stage
        cartridge_output = _exl3_gemm(x, trellis, suh, svh, True, False)
        # Sum with rescaling: output += cartridge_output / scale
        output = output + cartridge_output * (1.0 / scale)

    return output


# Patch Exl3MoEMethod._apply_expert to support cartridges
_original_apply_expert = Exl3MoEMethod._apply_expert


@staticmethod
def _apply_expert_with_cartridge(
    layer: Any,
    group: str,
    x: torch.Tensor,
    expert_id: int,
    shard_id: str,
) -> torch.Tensor:
    """Extended _apply_expert that applies MSRT cartridge after base GEMM."""
    # Run the original base EXL3 GEMM
    output = _original_apply_expert(layer, group, x, expert_id, shard_id)

    # Apply cartridge if present
    cartridge = getattr(layer, "_exl3_cartridge", None)
    if cartridge is not None and cartridge.active:
        key = (expert_id, shard_id)
        trellis = getattr(layer, f"{group}_trellis").exl3_tensors[key]
        packed_k = trellis.shape[0] * 16
        x_padded = x
        if x.shape[-1] < packed_k:
            x_padded = torch.nn.functional.pad(x, (0, packed_k - x.shape[-1]))
        output = apply_exl3_cartridge(
            output, x_padded, layer, group, expert_id, shard_id, cartridge)

    return output


# Install the patched method
Exl3MoEMethod._apply_expert = _apply_expert_with_cartridge


def load_cartridge_from_adapter(
    adapter_path: str,
    layer: Any,
    num_experts: int,
    device: torch.device,
) -> Exl3LoraCartridge | None:
    """Load an MSRT cartridge adapter from a safetensors file.

    The adapter contains tensors named:
      model.layers.{L}.mlp.experts.{E}.{proj}.rank0.trellis_{label}
      model.layers.{L}.mlp.experts.{E}.{proj}.rank0.suh_{label}
      model.layers.{L}.mlp.experts.{E}.{proj}.rank0.svh_{label}
      model.layers.{L}.mlp.experts.{E}.{proj}.rank0.scale_{label}

    Returns an Exl3LoraCartridge loaded onto the device.
    """
    from safetensors import safe_open

    # Detect number of stages from tensor names
    stage_labels = set()
    with safe_open(adapter_path, framework="pt") as f:
        keys = list(f.keys())
        for key in keys:
            if ".trellis_" in key:
                label = key.split(".trellis_")[-1]
                stage_labels.add(label)

    if not stage_labels:
        logger.warning("No MSRT cartridge stages found in %s", adapter_path)
        return None

    # Sort labels to maintain stage order
    sorted_labels = sorted(stage_labels)
    num_stages = len(sorted_labels)
    cartridge = Exl3LoraCartridge(num_stages, num_experts, device)

    shard_map = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}

    with safe_open(adapter_path, framework="pt") as f:
        for stage_idx, label in enumerate(sorted_labels):
            for key in keys:
                if f".trellis_{label}" not in key:
                    continue
                # Parse: model.layers.{L}.mlp.experts.{E}.{proj}.rank{R}.trellis_{label}
                parts = key.split(".")
                # Find expert ID and projection
                expert_id = int(parts[5])
                proj = parts[6]
                rank = int(parts[7].replace("rank", ""))
                shard_id = shard_map.get(proj, proj)

                prefix = ".".join(parts[:-1])  # without .trellis_{label}
                trellis_key = f"{prefix}.trellis_{label}"
                suh_key = f"{prefix}.suh_{label}"
                svh_key = f"{prefix}.svh_{label}"
                scale_key = f"{prefix}.scale_{label}"

                if trellis_key in f.keys() and suh_key in f.keys():
                    trellis = f.get_tensor(trellis_key)
                    suh = f.get_tensor(suh_key)
                    svh = f.get_tensor(svh_key) if svh_key in f.keys() else suh
                    scale_val = 1.0
                    if scale_key in f.keys():
                        scale_val = float(f.get_tensor(scale_key).item())

                    cartridge.set_stage_tensors(
                        stage_idx, expert_id, shard_id,
                        trellis, suh, svh, scale_val)

    cartridge.active = True
    logger.info("Loaded MSRT cartridge: %d stages, %d experts from %s",
                num_stages, num_experts, adapter_path)
    return cartridge
