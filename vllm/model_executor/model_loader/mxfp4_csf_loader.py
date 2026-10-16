# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read retained native tensors from a lossless MXFP4 container.

Compressed routed experts are owned by the model's quantization method.
Engram tensors retain their immutable file-range descriptors, avoiding full
CPU/GPU staging allocations for the host-mapped embedding tables.

[heo 2026-10-05, turin local overlay — MiMo-V2.6-Pro support, private]
Additions vs kk-beta 6008f020 (everything else verbatim):
  * FAMILIES += "mimo_v26_pro_mxfp4": (384, 6144, 2048, range(1, 70))
    (69 MoE layers; layer 0 dense per moe_layer_freq)
  * expert_tensor_names: MiMo branch — model.layers.N.mlp.experts.E with
    {gate,up,down}_proj.{weight,weight_scale} names (gate-first order to
    match W13Layout.W31, proven byte-exact by the TP8 harness 2026-10-05)
  * Mxfp4CsfModelLoader._root: accept the MiMo source config's
    quant_method="fp8" + store_dtype="mxfp4" (hybrid FP8 dense / MXFP4
    experts checkpoint) and resolve checkpoint_root from MIMO_CSF_ROOT when
    the config omits it; read_csf_contract still validates family/schema/
    codec, so acceptance is not unconditional.
  * get_all_weights: the expert-skip regex also matches `mlp.experts`
    (MiMo namespace) alongside ffn/block_sparse_moe.
  * [csfdelta 2026-10-12] `_load_mxfp4_csf_weights` constructs legacy-encoding
    (`encoding="csf"`) `b12x.loader._csf_source.CsfSource` planes instead of
    the pre-97fde4bd5 `CsfScalePlanes` type: FlashInfer 97fde4bd5/7e5a142e9
    replaced per-expert host planes with CsfSource (batched components +
    per-segment exception starts) and moved scale expansion onto CUDA, so
    the old symbol no longer exists on the paired flashinfer kk-mimo branch
    and its import failed at weight-load time. The sliced slab layout and
    u24 exception words are byte-identical under both spellings.
"""

import hashlib
import os
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
# Routed experts, hidden size, MoE intermediate size and compressed layers.
FAMILIES = {
    "deepseek_v41": (384, 5120, 2304, range(40)),
    "deepseek_v4_flash": (256, 4096, 2048, range(43)),
    "kimi_k3": (896, 3584, 3072, range(1, 93)),
    "mimo_v26_pro_mxfp4": (384, 6144, 2048, range(1, 70)),
}


@lru_cache(maxsize=4)
def checkpoint_contract(root: str) -> dict:
    """Validate the MXFP4-CSF container before loading model tensors."""
    return read_csf_contract(root, schema=SCHEMA, codec=CODEC, families=FAMILIES)


def expert_tensor_names(family, layer_index, expert):
    """Source weight and scale names of an expert's gate, up and down matrices."""
    if family == "kimi_k3":
        prefix = (
            f"language_model.model.layers.{layer_index}."
            f"block_sparse_moe.experts.{expert}"
        )
        weight, scale = "weight_packed", "weight_scale"
        order = ("w1", "w3", "w2")
    elif family == "mimo_v26_pro_mxfp4":
        prefix = f"model.layers.{layer_index}.mlp.experts.{expert}"
        weight, scale = "weight", "weight_scale"
        # gate-first: W13Layout.W31 fusion order (vLLM fuses [gate; up])
        order = ("gate_proj", "up_proj", "down_proj")
    else:
        # DeepSeek-V4.1-Flash and DeepSeek-V4-Flash, including its vision
        # variant, keep DeepSeek's native routed-expert names.
        prefix = f"layers.{layer_index}.ffn.experts.{expert}"
        weight, scale = "weight", "scale"
        order = ("w1", "w3", "w2")
    return tuple(
        (f"{prefix}.{p}.{weight}", f"{prefix}.{p}.{scale}") for p in order
    )


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
    e, h, n, layers = FAMILIES[family]
    if (num_experts, hidden_size, intermediate_size) != (e, h, n):
        raise ValueError("MXFP4-CSF expert geometry differs from the checkpoint family")
    if layer_index not in layers:
        raise ValueError("MXFP4-CSF layer is outside the compressed expert inventory")
    supported_tp = {
        "kimi_k3": (1, 2, 4, 8, 12, 16),
        # 2304 / 3 = 768 local channels: 32-aligned, whole 16-row scale slabs.
        "deepseek_v41": (1, 2, 3, 4, 8),
    }.get(family, (1, 2, 4, 8))
    if tp_size not in supported_tp or not 0 <= tp_rank < tp_size:
        raise ValueError(
            f"MXFP4-CSF {family} supports TP {supported_tp} with a valid rank"
        )
    with CsfTensorReader(root, contract["source_names"], "mxfp4") as reader:

        def experts():
            for expert in range(num_experts):
                yield tuple(
                    reader.matrix(weight, scale)
                    for weight, scale in expert_tensor_names(
                        family, layer_index, expert
                    )
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
    from b12x.loader._csf_source import CsfSource
    from b12x.moe.fused_moe import Mxfp4CsfWeights

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

    def csf_source(fixed, exceptions, rows, columns):
        """Batch sliced per-expert slab planes into one legacy ``CsfSource``.

        The sliced planes already carry the checkpoint's row-base/selector slab
        layout (16 base bytes followed by packed one-bit selector bytes per
        16-row slab) and u24-position exception words; the FlashInfer
        ``"csf"`` CsfSource encoding is byte-identical, so the slices upload
        verbatim and the b12x preparation expands them into the native scale
        layout on CUDA (FlashInfer 7e5a142e9) instead of the caller decoding
        scale values on the host. ``starts`` segments the concatenated
        exception words per expert.
        """
        sizes = [int(t.numel()) for t in exceptions]
        starts = torch.zeros(len(sizes) + 1, dtype=torch.int64)
        torch.cumsum(torch.tensor(sizes, dtype=torch.int64), 0, out=starts[1:])
        return CsfSource(
            (torch.stack(fixed), torch.cat(exceptions)),
            starts,
            rows,
            columns,
            "mxfp4",
            len(sizes),
            encoding="csf",
        )

    return Mxfp4CsfWeights(
        w13=w13.to(device),
        w2=w2.to(device),
        w13_scales=csf_source(
            fixed13, exceptions13, 2 * local, hidden_size // 32
        ),
        w2_scales=csf_source(fixed2, exceptions2, hidden_size, local // 32),
        w13_scale_scratch=w13_scale_scratch,
        w2_scale_scratch=w2_scale_scratch,
    )


class Mxfp4CsfModelLoader(DefaultModelLoader):
    # [heo overlay] structural geometry + identity gate: the container contract
    # must match the model config's expert geometry AND be pinned to the exact
    # source checkpoint identity (review §21-P1.4), otherwise this is NOT the
    # CSF main model (e.g. the DFlash draft) and must not be CSF-loaded.
    @staticmethod
    def _text_geometry(text_config):
        return (
            getattr(text_config, "n_routed_experts", None),
            getattr(text_config, "hidden_size", None),
            getattr(text_config, "moe_intermediate_size", None),
        )

    def _resolve_csf_root(self, quant, text_config, model_root=None):
        """Validated CSF container root for THIS model, or None (pass-through).

        None means: load this model with a DEFAULT-format loader (draft models,
        non-MiMo models served under the same load format).
        """
        if not isinstance(quant, dict):
            return None
        stored = quant.get("quant_method")
        root = None
        if stored == "mxfp4_csf":
            root = quant.get("checkpoint_root") or os.environ.get("MIMO_CSF_ROOT")
        elif (
            stored == "fp8"
            and quant.get("store_dtype") == "mxfp4"
            and getattr(text_config, "model_type", "").startswith("mimo_v2")
        ):
            # Hybrid MiMo source checkpoint: FP8 dense + MXFP4 routed experts.
            root = quant.get("checkpoint_root") or os.environ.get("MIMO_CSF_ROOT")
        if not root:
            return None
        root_path = Path(root)
        if not root_path.is_absolute():
            raise ValueError("MXFP4-CSF checkpoint_root must be an absolute local path")
        contract = checkpoint_contract(str(root_path.resolve()))
        e, h, n, _ = FAMILIES[contract["family"]]
        # Strict geometry + identity contract is scoped to the MiMo family
        # only (Codex §12.2): upstream families keep their stock pass-through
        # behavior for geometry-mismatched configs, unchanged by this commit.
        if contract["family"] == "mimo_v26_pro_mxfp4":
            if self._text_geometry(text_config) != (e, h, n):
                # Geometry mismatch: not the model this container was built for.
                if stored == "mxfp4_csf":
                    raise ValueError(
                        "MXFP4-CSF container geometry does not match the model config"
                    )
                return None
        # [heo overlay] Identity pin (§21-P1.4, hardened §23-P1): MANDATORY
        # for the CSF main target, scoped to the MiMo family (Codex §12.2):
        # our containers record source_index_sha256 and the serving tree must
        # carry a matching index. Upstream families keep stock behavior
        # (no identity pin) — their fixtures and containers predate it.
        if contract["family"] == "mimo_v26_pro_mxfp4":
            if model_root is None:
                raise ValueError(
                    "MXFP4-CSF loading requires the model path for identity pinning"
                )
            index_path = Path(model_root) / "model.safetensors.index.json"
            if not index_path.is_file():
                raise ValueError(
                    f"MXFP4-CSF identity pin requires {index_path} (not found)"
                )
            actual = hashlib.sha256(index_path.read_bytes()).hexdigest()
            recorded = contract.get("source_index_sha256")
            if not recorded:
                raise ValueError(
                    "MXFP4-CSF container contract lacks source_index_sha256 - "
                    "identity pin is mandatory (§23-P1)"
                )
            if actual != recorded:
                raise ValueError(
                    "MXFP4-CSF container was built from a different source "
                    f"checkpoint (index sha {actual[:16]}… != container "
                    f"record {recorded[:16]}…) - refusing to load"
                )
        # Path, not str: get_all_weights() applies `/` to this value (§23-P0).
        return root_path, contract

    def _resolve_for(self, model_config):
        """(resolved-root-or-None, text_config, quant, model_root)."""
        text_config = getattr(model_config, "hf_text_config", model_config.hf_config)
        hf_config = model_config.hf_config
        quant = getattr(hf_config, "quantization_config", None)
        if not isinstance(quant, dict):
            # Draft models may have NO quantization_config at all (the real
            # DFlash draft does not) - plain pass-through (review §21-P0.2).
            quant = getattr(text_config, "quantization_config", None)
        model_root = getattr(model_config, "model", None) or getattr(
            model_config, "model_path", None
        )
        resolved = self._resolve_csf_root(quant, text_config, model_root)
        return resolved, text_config, quant, model_root

    @staticmethod
    def _default_loader_for(model_config):
        """A DefaultModelLoader with a SUPPORTED load format for pass-through
        models (review §21-P0.2: DefaultModelLoader rejects load_format
        mxfp4_csf; never delegate on the CSF LoadConfig)."""
        from vllm.config import LoadConfig

        load_config = model_config.load_config if hasattr(model_config, "load_config") else None
        fmt = getattr(load_config, "load_format", None) if load_config else None
        if fmt in (None, "auto", "mxfp4_csf"):
            fmt = "auto"
        fresh = LoadConfig(load_format=fmt)
        return DefaultModelLoader(fresh)

    def _root(self, model_config):
        resolved, _, _, _ = self._resolve_for(model_config)
        if resolved is None:
            raise ValueError(
                "MXFP4-CSF loading requires a compressed-scale model config"
            )
        return resolved

    def download_model(self, model_config):
        resolved, _, _, _ = self._resolve_for(model_config)
        if resolved is None:
            # [heo overlay] pass-through with a SUPPORTED load format.
            self._default_loader_for(model_config).download_model(model_config)

    def get_all_weights(self, model_config, model):
        resolved, _, _, _ = self._resolve_for(model_config)
        if resolved is None:
            # [heo overlay] pass-through with a SUPPORTED load format.
            yield from self._default_loader_for(model_config).get_all_weights(
                model_config, model
            )
            return
        root, contract = resolved
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
                        r"(?:^|\.)layers\.(\d+)\.(?:ffn|block_sparse_moe|mlp)\.experts\.",
                        name,
                    )
                    if match and int(match.group(1)) < layers:
                        continue
                    if callable(file_filter) and file_filter(name):
                        yield name, file_source_tensor(descriptors[name])
                    else:
                        yield name, handle.get_tensor(name)
