# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read retained native tensors from a lossless MXFP4 container.

Compressed routed experts are owned by the model's quantization method.
Engram tensors retain their immutable file-range descriptors, avoiding full
CPU/GPU staging allocations for the host-mapped embedding tables.
"""

import time
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import numpy as np
import regex as re
import torch
from safetensors import safe_open

from vllm.model_executor.model_loader.csf_utils import (
    CsfMatrix,
    CsfTensorReader,
    read_csf_contract,
    tp_extent,
)
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import (
    file_source_tensor,
    safetensors_file_sources,
)

SCHEMA = "lil-mxfp4-csf-checkpoint/1"
CODEC = "row-base-offset1-u24-exceptions/1"
FAMILIES = {
    "deepseek_v41": (384, 5120, 2304),
    "kimi_k3": (896, 3584, 3072),
}


@lru_cache(maxsize=4)
def checkpoint_contract(root: str) -> dict:
    """Validate the MXFP4-CSF container before loading model tensors."""
    return read_csf_contract(root, schema=SCHEMA, codec=CODEC, families=FAMILIES)


def read_mxfp4_csf_layer(
    root,
    layer_index,
    *,
    num_experts,
    hidden_size,
    intermediate_size,
    tp_rank,
    tp_size,
    device,
    w13_scale_scratch,
    w2_scale_scratch,
):
    """Read rank-local gate/up/down tensors and unprepared compressed scales."""
    contract = checkpoint_contract(str(Path(root).resolve()))
    family = contract["family"]
    if (num_experts, hidden_size, intermediate_size) != FAMILIES[family]:
        raise ValueError("MXFP4-CSF expert geometry differs from the checkpoint family")
    supported_tp = (1, 2, 4, 8) if family == "deepseek_v41" else (1, 2, 4, 8, 12, 16)
    if tp_size not in supported_tp or not 0 <= tp_rank < tp_size:
        raise ValueError(
            f"MXFP4-CSF {family} supports TP {supported_tp} with a valid rank"
        )
    with CsfTensorReader(root, contract["source_names"], "mxfp4") as reader:

        def experts():
            for expert in range(num_experts):
                if family == "kimi_k3":
                    prefix = (
                        f"language_model.model.layers.{layer_index}."
                        f"block_sparse_moe.experts.{expert}"
                    )
                    weight, scale = "weight_packed", "weight_scale"
                else:
                    prefix = f"layers.{layer_index}.ffn.experts.{expert}"
                    weight, scale = "weight", "scale"
                yield tuple(
                    reader.matrix(f"{prefix}.{p}.{weight}", f"{prefix}.{p}.{scale}")
                    for p in ("w1", "w3", "w2")
                )

        return _load_mxfp4_csf_weights(
            experts(),
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            tp_rank=tp_rank,
            tp_size=tp_size,
            device=device,
            w13_scale_scratch=w13_scale_scratch,
            w2_scale_scratch=w2_scale_scratch,
        )


POSITION_MASK = (1 << 24) - 1


def _slice_scale_plane(fixed, exceptions, rows, columns, row_slice, column_slice):
    """Slice compressed bytes without floating-point reconstruction or fitting."""
    r0, r1 = row_slice
    c0, c1 = column_slice
    if not (0 <= r0 < r1 <= rows and r0 % 16 == r1 % 16 == 0):
        raise ValueError("MXFP4-CSF row slices must contain complete 16-row slabs")
    if not 0 <= c0 < c1 <= columns:
        raise ValueError("MXFP4-CSF column slice is outside the scale plane")
    if fixed.dtype != torch.uint8 or exceptions.dtype != torch.uint32:
        raise TypeError("MXFP4-CSF requires uint8 fixed bytes and uint32 exceptions")
    selectors = (columns + 7) // 8
    stream = fixed.numpy().reshape(rows // 16, 16 * (1 + selectors))
    bases = stream[:, :16]
    if (bases > 254).any():
        raise ValueError("MXFP4-CSF palette bases must be in 0..254")
    bits = np.unpackbits(
        stream[:, 16:].reshape(rows, selectors), axis=1, bitorder="little"
    )
    if bits[:, columns:].any():
        raise ValueError("MXFP4-CSF unused selector bits must be zero")
    selected = np.packbits(bits[r0:r1, c0:c1], axis=1, bitorder="little")
    result = np.concatenate(
        (bases[r0 // 16 : r1 // 16], selected.reshape((r1 - r0) // 16, -1)), 1
    )
    words = exceptions.numpy().reshape(-1)
    positions = words & POSITION_MASK
    if len(words) and (
        positions[-1] >= rows * columns or (positions[1:] <= positions[:-1]).any()
    ):
        raise ValueError(
            "MXFP4-CSF exception positions must be unique, sorted and in range"
        )
    rr, cc = positions // columns, positions % columns
    keep = (rr >= r0) & (rr < r1) & (cc >= c0) & (cc < c1)
    positions = (rr[keep] - r0) * (c1 - c0) + cc[keep] - c0
    words = (words[keep] & np.uint32(0xFF000000)) | positions
    return torch.from_numpy(result.copy()), torch.from_numpy(words.astype(np.uint32))


def _load_mxfp4_csf_weights(
    experts: Iterable[tuple[CsfMatrix, CsfMatrix, CsfMatrix]],
    *,
    num_experts,
    hidden_size,
    intermediate_size,
    tp_rank,
    tp_size,
    device,
    w13_scale_scratch,
    w2_scale_scratch,
):
    """Slice and upload gate/up/down-ordered expert projections.

    The iterable must yield exactly ``num_experts`` projection triples. Tensor
    stores, manifests and model-specific tensor names belong to the caller.
    Expanded scale buffers remain caller-owned for serialized layer execution.
    """
    from b12x.moe.fused_moe import CsfScalePlanes, Mxfp4CsfWeights

    if num_experts <= 0 or hidden_size <= 0 or hidden_size % 64:
        raise ValueError("MXFP4-CSF requires experts and 64-aligned hidden channels")
    first, last = tp_extent(intermediate_size, tp_rank, tp_size, 32)
    local = last - first
    w13 = torch.empty(
        (num_experts, 2 * local, hidden_size // 2), dtype=torch.uint8, device="cpu"
    )
    w2 = torch.empty(
        (num_experts, hidden_size, local // 2), dtype=torch.uint8, device="cpu"
    )
    fixed13, fixed2, exceptions13, exceptions2 = [], [], [], []
    for expert, (first_projection, second_projection, down) in zip(
        range(num_experts), experts, strict=True
    ):
        f13, e13 = [], []
        for matrix, projection in enumerate(
            (first_projection, second_projection, down)
        ):
            view = projection.weight
            expected = (
                [intermediate_size, hidden_size // 2]
                if matrix < 2
                else [hidden_size, intermediate_size // 2]
            )
            if view.get_shape() != expected or view.get_dtype() not in ("I8", "U8"):
                raise ValueError(
                    f"MXFP4-CSF nibble geometry/dtype mismatch: "
                    f"expert={expert}, projection={matrix}"
                )
            if matrix < 2:
                w13[expert, matrix * local : (matrix + 1) * local].copy_(
                    view[first:last, :].view(torch.uint8)
                )
                rows, columns = intermediate_size, hidden_size // 32
                row_slice, column_slice = (first, last), (0, columns)
            else:
                w2[expert].copy_(view[:, first // 2 : last // 2].view(torch.uint8))
                rows, columns = hidden_size, intermediate_size // 32
                row_slice, column_slice = (0, rows), (first // 32, last // 32)
            fixed, exceptions = _slice_scale_plane(
                projection.fixed,
                projection.exceptions,
                rows,
                columns,
                row_slice,
                column_slice,
            )
            if matrix < 2:
                f13.append(fixed)
                if matrix:
                    words = exceptions.numpy().copy()
                    words += np.uint32(local * columns)
                    exceptions = torch.from_numpy(words)
                e13.append(exceptions)
            else:
                fixed2.append(fixed)
                exceptions2.append(exceptions)
        fixed13.append(torch.cat(f13))
        exceptions13.append(torch.cat(e13))
    return Mxfp4CsfWeights(
        w13=w13.to(device),
        w2=w2.to(device),
        w13_scales=CsfScalePlanes(tuple(fixed13), tuple(exceptions13)),
        w2_scales=CsfScalePlanes(tuple(fixed2), tuple(exceptions2)),
        w13_scale_scratch=w13_scale_scratch,
        w2_scale_scratch=w2_scale_scratch,
    )


class Mxfp4CsfModelLoader(DefaultModelLoader):
    def _root(self, model_config):
        text_config = getattr(model_config, "hf_text_config", model_config.hf_config)
        quant = getattr(model_config.hf_config, "quantization_config", None)
        quant = quant or text_config.quantization_config
        if quant.get("quant_method") not in ("mxfp4_csf",):
            raise ValueError(
                "MXFP4-CSF loading requires a compressed-scale model config"
            )
        root = Path(quant["checkpoint_root"])
        if not root.is_absolute():
            raise ValueError("MXFP4-CSF checkpoint_root must be an absolute local path")
        return root, checkpoint_contract(str(root.resolve()))

    def download_model(self, model_config):
        self._root(model_config)

    def get_all_weights(self, model_config, model):
        root, contract = self._root(model_config)
        if getattr(model, "secondary_weights", ()):
            raise NotImplementedError(
                "MXFP4-CSF does not support secondary weight sources"
            )
        file_filter = getattr(model, "checkpoint_file_weight_filter", None)
        prefixes = getattr(model, "checkpoint_weight_name_prefixes", None)
        text_config = getattr(model_config, "hf_text_config", model_config.hf_config)
        layers = text_config.num_hidden_layers
        self.counter_before_loading_weights = time.perf_counter()
        for filename in sorted(set(contract["source_names"].values())):
            path = str(root / "tensors" / filename)
            descriptors = safetensors_file_sources(path)
            with safe_open(path, framework="pt", device="cpu") as handle:
                for name in sorted(descriptors):
                    if prefixes is not None and not name.startswith(prefixes):
                        continue
                    match = re.search(
                        r"(?:^|\.)layers\.(\d+)\.(?:ffn|block_sparse_moe)\.experts\.",
                        name,
                    )
                    if match and int(match.group(1)) < layers:
                        continue
                    if callable(file_filter) and file_filter(name):
                        yield name, file_source_tensor(descriptors[name])
                    else:
                        yield name, handle.get_tensor(name)
