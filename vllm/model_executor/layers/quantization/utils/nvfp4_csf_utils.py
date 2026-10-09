# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor-only slicing and preparation of NVFP4 compressed scale planes."""

from collections.abc import Iterable

import numpy as np
import torch

from vllm.model_executor.model_loader.csf_utils import CsfMatrix, tp_extent

_PADDED_SCALE = 0x38


class TensorView:
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def get_shape(self):
        return list(self.tensor.shape)

    def get_dtype(self):
        return "U8" if self.tensor.dtype == torch.uint8 else str(self.tensor.dtype)

    def __getitem__(self, extent):
        return self.tensor[extent]


def _slice_scale_plane(
    fixed,
    exceptions,
    rows,
    columns,
    row_slice,
    column_slice,
    out_rows=None,
    out_columns=None,
):
    """Copy aligned compressed extents and rebase exception positions.

    ``out_rows``/``out_columns`` extend the slice to a padded TP shard whose
    weights are zero there: appended rows get the E4M3 scale 1.0 (a zero scale
    would be stored as a replacement word in every packed word), appended
    columns zero codes over their row's base.
    """
    r0, r1 = row_slice
    c0, c1 = column_slice
    if not (0 <= r0 < r1 <= rows and r0 % 16 == r1 % 16 == 0):
        raise ValueError("NVFP4-CSF row slices require complete 16-row slabs")
    if not (0 <= c0 < c1 <= columns and c0 % 2 == c1 % 2 == columns % 2 == 0):
        raise ValueError("NVFP4-CSF column slices require complete nibble pairs")
    if fixed.dtype != torch.uint8 or exceptions.dtype != torch.uint32:
        raise TypeError("NVFP4-CSF requires uint8 fixed and uint32 exception tensors")
    stream = fixed.numpy().reshape(rows // 16, 16 * (1 + columns // 2))
    bases = stream[:, :16]
    if np.any(bases > 240):
        raise ValueError("NVFP4-CSF row bases must be in 0..240")
    packed = stream[:, 16:].reshape(rows, columns // 2)
    out_rows = r1 - r0 if out_rows is None else out_rows
    out_columns = c1 - c0 if out_columns is None else out_columns
    if out_rows < r1 - r0 or out_rows % 16 or out_columns < c1 - c0 or out_columns % 2:
        raise ValueError("NVFP4-CSF padded slices must extend whole slabs and pairs")
    selected = np.zeros((out_rows, out_columns // 2), dtype=np.uint8)
    selected[: r1 - r0, : (c1 - c0) // 2] = packed[r0:r1, c0 // 2 : c1 // 2]
    slab_bases = np.full((out_rows // 16, 16), _PADDED_SCALE, dtype=np.uint8)
    slab_bases[: (r1 - r0) // 16] = bases[r0 // 16 : r1 // 16]
    result = np.concatenate((slab_bases, selected.reshape(out_rows // 16, -1)), 1)
    words = exceptions.numpy().reshape(-1)
    positions = words & np.uint32(0xFFFFFF)
    if len(words) and (
        positions[-1] >= rows * columns or np.any(positions[1:] <= positions[:-1])
    ):
        raise ValueError("NVFP4-CSF exception positions must be sorted and unique")
    rr, cc = positions // columns, positions % columns
    keep = (rr >= r0) & (rr < r1) & (cc >= c0) & (cc < c1)
    local_positions = (rr[keep] - r0) * out_columns + cc[keep] - c0
    selected_words = (words[keep] & np.uint32(0xFF000000)) | local_positions
    return result.copy(), selected_words.astype(np.uint32)


def prepare_nvfp4_csf_weights(
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
    local_size=None,
):
    """Slice and upload up/gate/down-ordered expert projections.

    The iterable must yield exactly ``num_experts`` projection triples. Tensor
    stores, manifests and model-specific tensor names belong to the caller.
    Expanded scale buffers remain caller-owned for serialized layer execution.
    ``local_size`` is the per-rank width of a TP-padded expert (GLM-5.3 at
    TP6: 2048 channels padded to 6 x 352): rank r holds checkpoint channels
    [r * local_size, (r + 1) * local_size) and zeros past the checkpoint.
    """
    from b12x.moe.fused_moe import CsfScalePlanes, Nvfp4CsfWeights, PackedWeights

    if num_experts <= 0 or hidden_size <= 0 or hidden_size % 128:
        raise ValueError("NVFP4-CSF requires experts and 128-aligned hidden channels")
    if local_size is None:
        first, last = tp_extent(intermediate_size, tp_rank, tp_size, 64)
        local = last - first
    else:
        local = int(local_size)
        if local <= 0 or local % 32 or local * tp_size < intermediate_size:
            raise ValueError(
                "NVFP4-CSF padded shards need 32-aligned widths covering the expert"
            )
        first = min(tp_rank * local, intermediate_size)
        last = min(first + local, intermediate_size)
    real = last - first
    if real <= 0:
        raise ValueError("NVFP4-CSF requires checkpoint channels on every TP rank")
    w13 = torch.zeros(
        (num_experts, 2 * local, hidden_size // 2), dtype=torch.uint8, device="cpu"
    )
    w2 = torch.zeros(
        (num_experts, hidden_size, local // 2), dtype=torch.uint8, device="cpu"
    )
    # Model loaders may set BF16 as the default dtype. Calibration belongs to
    # the source FP32 contract and must not be rounded with the model weights.
    g13, g2 = (
        torch.empty(num_experts, dtype=torch.float32, device="cpu") for _ in range(2)
    )
    a13, a2 = (
        torch.empty(num_experts, dtype=torch.float32, device="cpu") for _ in range(2)
    )
    fixed13, fixed2, exceptions13, exceptions2 = [], [], [], []

    def scalar(value, role):
        if value is None or value.numel() != 1 or value.dtype != torch.float32:
            raise ValueError(f"NVFP4 {role} must be one FP32 value")
        if not bool(torch.isfinite(value).all() and (value > 0).all()):
            raise ValueError(f"NVFP4 {role} must be positive and finite")
        return value.reshape(())

    for expert, (first_projection, second_projection, down) in zip(
        range(num_experts), experts, strict=True
    ):
        f13, e13, global13, input13 = [], [], [], []
        for matrix, projection in enumerate(
            (first_projection, second_projection, down)
        ):
            view = projection.weight
            expected = (
                [intermediate_size, hidden_size // 2]
                if matrix < 2
                else [hidden_size, intermediate_size // 2]
            )
            if view.get_shape() != expected or view.get_dtype() != "U8":
                raise ValueError(
                    f"NVFP4 nibble geometry/dtype mismatch: "
                    f"expert={expert}, projection={matrix}"
                )
            global_scale, input_scale = (
                scalar(projection.global_scale, "global_scale"),
                scalar(projection.input_scale, "input_scale"),
            )
            if matrix < 2:
                if real:
                    w13[expert, matrix * local : matrix * local + real].copy_(
                        view[first:last, :]
                    )
                rows, columns = intermediate_size, hidden_size // 16
                row_slice, column_slice = (first, last), (0, columns)
                out_rows, out_columns = local, columns
                global13.append(global_scale)
                input13.append(input_scale)
            else:
                if real:
                    w2[expert, :, : real // 2].copy_(view[:, first // 2 : last // 2])
                rows, columns = hidden_size, intermediate_size // 16
                row_slice, column_slice = (0, rows), (first // 16, last // 16)
                out_rows, out_columns = rows, local // 16
                g2[expert], a2[expert] = global_scale, input_scale
            fixed, exceptions = _slice_scale_plane(
                projection.fixed,
                projection.exceptions,
                rows,
                columns,
                row_slice,
                column_slice,
                out_rows,
                out_columns,
            )
            if matrix < 2:
                f13.append(fixed)
                if matrix:
                    exceptions += np.uint32(local * columns)
                e13.append(exceptions)
            else:
                fixed2.append(fixed)
                exceptions2.append(exceptions)
        if not torch.equal(global13[0], global13[1]):
            raise ValueError("NVFP4-CSF requires equal gate/up global weight scales")
        g13[expert] = global13[0]
        a13[expert] = torch.stack(input13).amax()
        fixed13.append(np.concatenate(f13))
        exceptions13.append(np.concatenate(e13))
    return Nvfp4CsfWeights(
        packed=PackedWeights(
            w13=w13.to(device),
            w2=w2.to(device),
            w13_block_scales=w13_scale_scratch,
            w2_block_scales=w2_scale_scratch,
            w13_global_scales=g13.to(device),
            w2_global_scales=g2.to(device),
            input_scale=a13.to(device).reciprocal(),
            intermediate_scale=a2.to(device).reciprocal(),
            immutable_input_scales=True,
        ),
        w13_scales=CsfScalePlanes(
            tuple(torch.from_numpy(p) for p in fixed13),
            tuple(torch.from_numpy(p) for p in exceptions13),
        ),
        w2_scales=CsfScalePlanes(
            tuple(torch.from_numpy(p) for p in fixed2),
            tuple(torch.from_numpy(p) for p in exceptions2),
        ),
    )
