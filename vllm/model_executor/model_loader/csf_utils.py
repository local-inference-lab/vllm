# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block-scale checkpoint metadata and bounded safetensors access."""

import json
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import torch
from safetensors import safe_open

if TYPE_CHECKING:
    from b12x.loader._csf_blocks import CsfBlockPlane
    from b12x.loader._csf_source import CsfSource

CSF_ENCODING = "csf_32x2_v1"
CSF_COMPONENTS = ("bases", "codes", "page_bases", "offsets", "exceptions")


class PackedTensorView(Protocol):
    """CPU tensor slicing interface provided by safetensors and tensor stores."""

    def get_shape(self) -> list[int]: ...

    def get_dtype(self) -> str: ...

    def __getitem__(self, extent: tuple[slice, slice]) -> torch.Tensor: ...


@dataclass(frozen=True)
class CsfMatrix:
    """One projection's lazy FP4 bytes and resident CPU scale components.

    NVFP4 also supplies the original FP32 scalar weight and activation calibration.
    MXFP4 leaves those fields unset. Keep the backing tensor store open until the
    layer reader returns.
    """

    weight: PackedTensorView
    scales: "CsfBlockPlane | CsfSource"
    global_scale: torch.Tensor | None = None
    input_scale: torch.Tensor | None = None


def tp_extent(intermediate_size: int, tp_rank: int, tp_size: int, alignment: int):
    """Validate an equal TP partition and return its intermediate-axis bounds."""
    if (
        tp_size <= 0
        or not 0 <= tp_rank < tp_size
        or intermediate_size <= 0
        or intermediate_size % (alignment * tp_size)
    ):
        raise ValueError(
            f"CSF requires a valid TP rank and extents aligned to {alignment} "
            "intermediate channels"
        )
    local = intermediate_size // tp_size
    return tp_rank * local, (tp_rank + 1) * local


def read_csf_contract(root, *, families):
    """Read the scale encoding and standard tensor-to-shard index."""
    root = Path(root)
    config = json.loads((root / "config.json").read_text())
    text = config.get("text_config", config)
    quant = config.get("quantization_config") or text.get("quantization_config", {})
    if quant.get("weight_scale_encoding") not in ("csf", CSF_ENCODING):
        raise ValueError(
            f"CSF requires weight_scale_encoding='csf' or {CSF_ENCODING!r}"
        )
    # A multimodal config names the model at the top (deepseek_v41) and its
    # text tower in text_config (deepseek_v41_text); the family is the model's.
    model_type = config.get("model_type") or text.get("model_type")
    family = {
        "deepseek_v4_1": "deepseek_v41",
        "deepseek_v4": "deepseek_v4_flash",
    }.get(model_type, model_type)
    if family not in families:
        raise ValueError("CSF config must name a supported model family")
    names = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    if not names or any(
        Path(name).is_absolute()
        or ".." in Path(name).parts
        or not name.endswith(".safetensors")
        for name in names.values()
    ):
        raise ValueError("CSF index must reference checkpoint-local safetensors files")
    return {"family": family, "source_names": names}


class CsfTensorReader(ExitStack):
    """Keep a layer's lazily opened shards alive while slicing its tensors."""

    def __init__(self, root, source_names, codec):
        super().__init__()
        self.root = Path(root)
        self.source_names = source_names
        self.codec = codec
        self.handles = {}

    def _handle(self, name):
        filename = self.source_names[name]
        if filename not in self.handles:
            self.handles[filename] = self.enter_context(
                safe_open(self.root / filename, framework="pt", device="cpu")
            )
        return self.handles[filename]

    def matrix(self, weight, scale, *, global_scale=None, input_scale=None):
        """Expose a lazy packed weight view and one projection's CPU scale data."""
        from b12x.loader._csf_blocks import CsfBlockPlane

        view = self._handle(weight).get_slice(weight)
        rows, packed_columns = view.get_shape()
        columns = packed_columns * 2 // (16 if self.codec == "nvfp4" else 32)
        if f"{scale}.csf32_bases" in self.source_names:
            components = {
                component: self._handle(f"{scale}.csf32_{component}")
                .get_tensor(f"{scale}.csf32_{component}")
                .numpy()
                for component in CSF_COMPONENTS
            }
            plane = CsfBlockPlane(rows, columns, self.codec, **components)
        else:
            from b12x.loader._csf_source import CsfSource

            prefix = f"{scale}.{self.codec}_csf_"
            fixed, exceptions = (
                self._handle(prefix + name).get_tensor(prefix + name)
                for name in ("fixed", "exceptions")
            )
            # One slab segment: 16-row groups of 16 base bytes plus the packed
            # scale bits, stored flat in the checkpoint.
            plane = CsfSource(
                (fixed.view(1, rows // 16, -1), exceptions),
                torch.tensor([0, exceptions.numel()], dtype=torch.int64),
                rows,
                columns,
                self.codec,
                1,
                "csf",
            )
            plane.validate()
        return CsfMatrix(
            weight=view,
            scales=plane,
            global_scale=(
                self._handle(global_scale).get_tensor(global_scale)
                if global_scale is not None
                else None
            ),
            input_scale=(
                self._handle(input_scale).get_tensor(input_scale)
                if input_scale is not None
                else None
            ),
        )
