# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small compressed matrices with known bytes across TP partition boundaries."""

import numpy as np
import torch

from vllm.model_executor.model_loader.csf_utils import CsfMatrix


class TensorView:
    def __init__(self, tensor):
        self.tensor = tensor

    def get_shape(self):
        return list(self.tensor.shape)

    def get_dtype(self):
        return "I8" if self.tensor.dtype == torch.int8 else "U8"

    def __getitem__(self, extent):
        return self.tensor[extent]


def matrix(rows, channels, *, group_size, seed):
    """Return a source projection and its uncompressed reference scale bytes."""
    columns = channels // group_size
    bases = ((np.arange(rows) + seed) % 190 + 20).astype(np.uint8)
    offsets = (np.arange(rows * columns) + seed) % (2 if group_size == 32 else 16)
    offsets = offsets.reshape(rows, columns).astype(np.uint8)
    logical = bases[:, None] + offsets
    positions = np.unique(
        [
            0,
            columns - 1,
            rows // 2 * columns - 1,
            rows // 2 * columns,
            rows * columns - 1,
        ]
    ).astype(np.uint32)
    values = (np.arange(len(positions), dtype=np.uint32) * 7 + seed) % 256
    logical.flat[positions] = values.astype(np.uint8)
    packed = (
        np.packbits(offsets, axis=1, bitorder="little")
        if group_size == 32
        else offsets[:, ::2] | (offsets[:, 1::2] << 4)
    )
    fixed = np.concatenate(
        (bases.reshape(rows // 16, 16), packed.reshape(rows // 16, -1)), axis=1
    )
    weight = (torch.arange(rows * channels // 2) + seed).to(torch.uint8)
    source = CsfMatrix(
        weight=TensorView(weight.reshape(rows, channels // 2)),
        fixed=torch.from_numpy(fixed),
        exceptions=torch.from_numpy(positions | (values << 24)),
    )
    return source, torch.from_numpy(logical.copy())


def decode_planes(planes, rows, columns, group_size):
    """Reconstruct CPU scale bytes independently of B12X preparation."""
    result = []
    for fixed, exceptions in zip(planes.fixed, planes.exceptions, strict=True):
        selectors = (columns + 7) // 8 if group_size == 32 else columns // 2
        slabs = fixed.numpy().reshape(rows // 16, 16 * (1 + selectors))
        packed = slabs[:, 16:].reshape(rows, selectors)
        if group_size == 32:
            offsets = np.unpackbits(packed, axis=1, bitorder="little")[:, :columns]
        else:
            offsets = np.stack((packed & 15, packed >> 4), axis=-1).reshape(
                rows, columns
            )
        logical = (slabs[:, :16].reshape(rows, 1) + offsets).astype(np.uint8)
        words = exceptions.numpy()
        logical.flat[words & 0xFFFFFF] = (words >> 24).astype(np.uint8)
        result.append(torch.from_numpy(logical.copy()))
    return torch.stack(result)
