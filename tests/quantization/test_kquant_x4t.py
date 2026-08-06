# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import zlib

import pytest
import torch
from safetensors.torch import save_file

from vllm.model_executor.layers.quantization import kquant_kimi_k3_qsrt_tp12 as qsrt
from vllm.model_executor.layers.quantization import kquant_x4t as x4t


def _constant_x4t_scale(rows: int, columns: int, value: int) -> bytes:
    return x4t.pack_x4t_scale_plane(
        torch.full((rows, columns), value, dtype=torch.uint8)
    )


def _rank_striped_packed(matrix: str) -> bytes:
    rows, features = x4t._MATRIX_SHAPES[matrix]
    packed_columns = features // 2
    if matrix in ("w1", "w3"):
        return b"".join(bytes([row // 256]) * packed_columns for row in range(rows))
    stripe = b"".join(bytes([rank]) * 128 for rank in range(12))
    assert len(stripe) == packed_columns
    return stripe * rows


def _record(matrix: str, scale_value: int) -> bytes:
    rows, features = x4t._MATRIX_SHAPES[matrix]
    packed = _rank_striped_packed(matrix)
    scale = _constant_x4t_scale(rows, features // x4t.MXFP4_BLOCK, scale_value)
    return (
        x4t._RECORD_HEADER.pack(
            x4t.X4T_RECORD_MAGIC,
            x4t.X4T_VERSION,
            x4t._RECORD_HEADER.size,
            x4t.X4T_MATRIX_ORDER.index(matrix),
            x4t.X4T_TILE_ROWS,
            0,
            rows,
            features,
            len(packed),
            len(scale),
            bytes(28),
        )
        + packed
        + scale
    )


def _patterned_scale(matrix: str, scale_value: int) -> torch.Tensor:
    rows, features = x4t._MATRIX_SHAPES[matrix]
    columns = features // x4t.MXFP4_BLOCK
    row = torch.arange(rows, dtype=torch.int16)[:, None]
    column = torch.arange(columns, dtype=torch.int16)[None, :]
    scale = (scale_value + (row % 4) + ((row + 3 * column) & 1)).to(torch.uint8)
    for position, value in (
        (3, 97),
        (columns * (rows // 2) + columns // 2, 151),
        (rows * columns - 1, 211),
    ):
        scale.view(-1)[position] = value
    return scale.contiguous()


def _patterned_record(matrix: str, scale_value: int) -> bytes:
    rows, features = x4t._MATRIX_SHAPES[matrix]
    packed = _rank_striped_packed(matrix)
    scale = x4t.pack_x4t_scale_plane(_patterned_scale(matrix, scale_value))
    return (
        x4t._RECORD_HEADER.pack(
            x4t.X4T_RECORD_MAGIC,
            x4t.X4T_VERSION,
            x4t._RECORD_HEADER.size,
            x4t.X4T_MATRIX_ORDER.index(matrix),
            x4t.X4T_TILE_ROWS,
            0,
            rows,
            features,
            len(packed),
            len(scale),
            bytes(28),
        )
        + packed
        + scale
    )


def _write_layer(
    path, *, layer: int, expert: int, patterned_scale: bool = False
) -> None:
    entries = [
        x4t.X4TDirectoryEntry(0, 0, 0, 0)
        for _ in range(x4t.X4T_EXPERTS_PER_LAYER * len(x4t.X4T_MATRIX_ORDER))
    ]
    records: list[tuple[int, bytes]] = []
    cursor = x4t.X4T_DATA_OFFSET
    for matrix_index, matrix in enumerate(x4t.X4T_MATRIX_ORDER):
        record = (
            _patterned_record(matrix, 120 + matrix_index)
            if patterned_scale
            else _record(matrix, 120 + matrix_index)
        )
        index = expert * len(x4t.X4T_MATRIX_ORDER) + matrix_index
        entries[index] = x4t.X4TDirectoryEntry(
            cursor, len(record), zlib.crc32(record), 1
        )
        records.append((cursor, record))
        cursor = x4t._align_up(cursor + len(record))
    directory_payload = b"".join(
        x4t._DIRECTORY_ENTRY.pack(
            entry.offset,
            entry.length,
            entry.crc32,
            entry.flags,
        )
        for entry in entries
    )
    directory = directory_payload + bytes(
        x4t.X4T_DIRECTORY_BYTES - len(directory_payload)
    )
    directory_crc = zlib.crc32(directory)
    zero_header = x4t._canonical_layer_header(
        layer=layer,
        file_bytes=cursor,
        record_count=3,
        directory_crc32=directory_crc,
        header_crc32=0,
    )
    header = x4t._canonical_layer_header(
        layer=layer,
        file_bytes=cursor,
        record_count=3,
        directory_crc32=directory_crc,
        header_crc32=zlib.crc32(zero_header),
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.ftruncate(descriptor, cursor)
        os.pwrite(descriptor, header, 0)
        os.pwrite(descriptor, directory, x4t.X4T_LAYER_HEADER_BYTES)
        for offset, record in records:
            os.pwrite(descriptor, record, offset)
    finally:
        os.close(descriptor)


def _decode_components(components: x4t.X4TScaleComponents) -> torch.Tensor:
    rows = components.rows
    columns = components.columns
    selector_bytes = (columns + 7) // 8
    tile_bytes = x4t.X4T_TILE_ROWS * (1 + selector_bytes)
    fixed = components.fixed.reshape(rows // x4t.X4T_TILE_ROWS, tile_bytes)
    bases = fixed[:, : x4t.X4T_TILE_ROWS].reshape(rows)
    selectors = fixed[:, x4t.X4T_TILE_ROWS :].reshape(rows, selector_bytes)
    column = torch.arange(columns, dtype=torch.int64)
    selected = (
        selectors[:, column // 8].to(torch.int16) >> (column % 8).to(torch.int16)
    ) & 1
    result = (bases.to(torch.int16)[:, None] + selected).to(torch.uint8)
    for word in components.exceptions.to(torch.int64).tolist():
        position = word & x4t.X4T_POSITION_MASK
        result.view(-1)[position] = word >> x4t.X4T_POSITION_BITS
    return result


def _flatten_components(
    components: tuple[x4t.X4TScaleComponents, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fixed = torch.stack([component.fixed for component in components])
    counts = torch.tensor(
        [component.exceptions.numel() for component in components],
        dtype=torch.int64,
    )
    offsets = torch.empty((len(components) + 1,), dtype=torch.int64)
    offsets[0] = 0
    torch.cumsum(counts, 0, out=offsets[1:])
    exceptions = (
        torch.cat([component.exceptions for component in components])
        if int(offsets[-1])
        else torch.empty((0,), dtype=torch.uint32)
    )
    return fixed, offsets, exceptions


def _write_qsrt_slab(path, *, layer: int, kept_expert: int) -> None:
    layout = qsrt.TP12SlabLayout(
        qsrt.EXPERTS - 1,
        1,
        keep_storage=qsrt.KEEP_STORAGE_EXTERNAL_X4T,
    )
    prefix = qsrt._QSRT_HEADER.pack(
        qsrt.QSRT_MAGIC,
        qsrt.QSRT_HEADER_VERSION,
        qsrt.HEADER_BYTES,
        qsrt.TP_SIZE,
        layer,
        qsrt.EXPERTS,
        layout.compressed_experts,
        layout.kept_experts,
        8,
        qsrt.CODEBOOK_IDS[qsrt.CODEBOOK_SQG_NORMAL_E4M3],
        qsrt.KEEP_STORAGE_IDS[qsrt.KEEP_STORAGE_EXTERNAL_X4T],
        qsrt.CODEBOOK_MULTIPLIERS[qsrt.CODEBOOK_SQG_NORMAL_E4M3],
        qsrt.ALIGNMENT,
        qsrt.HEADER_BYTES,
        qsrt.HEADER_BYTES + qsrt.FORMAT_BYTES,
        layout.rank_sections_offset,
        layout.rank_stride,
        layout.disk_bytes,
    )
    formats = bytearray(qsrt.EXPERTS)
    formats[kept_expert] = qsrt.FORMAT_MXFP4
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.ftruncate(descriptor, layout.disk_bytes)
        os.pwrite(
            descriptor,
            prefix + bytes(qsrt.HEADER_BYTES - len(prefix)),
            0,
        )
        os.pwrite(
            descriptor,
            bytes(formats) + bytes(qsrt.FORMAT_BYTES - len(formats)),
            qsrt.HEADER_BYTES,
        )
    finally:
        os.close(descriptor)


def test_x4t_reader_recovers_exact_tp12_rank_shards(tmp_path) -> None:
    path = tmp_path / "x4t-layer-00024.bin"
    _write_layer(path, layer=24, expert=17)

    reader = x4t.X4TLayerReader(path)
    w1 = reader.read_rank(17, "w1", 5)
    w3 = reader.read_rank(17, "w3", 5)
    w2 = reader.read_rank(17, "w2", 5)

    assert reader.layer == 24
    assert reader.record_count == 3
    assert not reader.has(16, "w1")
    assert tuple(w1.packed.shape) == (256, 1792)
    assert tuple(w1.scale.shape) == (256, 112)
    assert bool(torch.all(w1.packed == 5))
    assert bool(torch.all(w1.scale == 120))
    assert bool(torch.all(w3.packed == 5))
    assert bool(torch.all(w3.scale == 121))
    assert tuple(w2.packed.shape) == (3584, 128)
    assert tuple(w2.scale.shape) == (3584, 8)
    assert bool(torch.all(w2.packed == 5))
    assert bool(torch.all(w2.scale == 122))


def test_x4t_direct_components_match_strict_rank_decode(tmp_path) -> None:
    path = tmp_path / "x4t-layer-00024.bin"
    _write_layer(path, layer=24, expert=17, patterned_scale=True)
    reader = x4t.X4TLayerReader(path)

    for rank in (0, 5, 11):
        w1, w3, w2 = reader.read_rank_triplet(17, rank)
        for direct in (w1, w3, w2):
            strict = reader.read_rank(17, direct.matrix, rank)
            assert torch.equal(direct.packed, strict.packed)
            assert torch.equal(_decode_components(direct.scale), strict.scale)

        fused = w1.scale.concatenate_rows(w3.scale)
        strict_w1 = reader.read_rank(17, "w1", rank)
        strict_w3 = reader.read_rank(17, "w3", rank)
        assert torch.equal(
            _decode_components(fused),
            torch.cat((strict_w1.scale, strict_w3.scale)),
        )


def test_x4t_tp12_cache_maps_exact_gpu_facing_rank(tmp_path) -> None:
    layer = 24
    rank = 5
    expert = 17
    source = tmp_path / "x4t-layer-00024.bin"
    cache_path = tmp_path / x4t.x4t_tp12_rank_filename(layer, rank)
    _write_layer(source, layer=layer, expert=expert, patterned_scale=True)
    reader = x4t.X4TLayerReader(source)
    w1, w3, w2 = reader.read_rank_triplet(expert, rank)
    w13_scale = w1.scale.concatenate_rows(w3.scale)
    w13_fixed, w13_offsets, w13_exceptions = _flatten_components((w13_scale,))
    w2_fixed, w2_offsets, w2_exceptions = _flatten_components((w2.scale,))
    save_file(
        {
            "expert_ids": torch.tensor([expert], dtype=torch.int32),
            "w13_packed": torch.cat((w1.packed, w3.packed)).unsqueeze(0),
            "w2_packed": w2.packed.unsqueeze(0),
            "w13_fixed": w13_fixed,
            "w13_exception_offsets": w13_offsets,
            "w13_exceptions": w13_exceptions,
            "w2_fixed": w2_fixed,
            "w2_exception_offsets": w2_offsets,
            "w2_exceptions": w2_exceptions,
        },
        cache_path,
        metadata={
            "format": x4t.X4T_TP12_CACHE_FORMAT,
            "version": str(x4t.X4T_TP12_CACHE_VERSION),
            "layer": str(layer),
            "rank": str(rank),
        },
    )

    cache = x4t.read_x4t_tp12_rank_shard(
        cache_path,
        layer=layer,
        rank=rank,
        expected_expert_ids=[expert],
    )

    assert cache.expert_ids.tolist() == [expert]
    assert torch.equal(cache.w13_packed[0, :256], w1.packed)
    assert torch.equal(cache.w13_packed[0, 256:], w3.packed)
    assert torch.equal(cache.w2_packed[0], w2.packed)
    assert torch.equal(
        _decode_components(cache.w13_scales[0]), _decode_components(w13_scale)
    )
    assert torch.equal(
        _decode_components(cache.w2_scales[0]), _decode_components(w2.scale)
    )


def test_x4t_reader_detects_record_corruption_after_directory_load(tmp_path) -> None:
    path = tmp_path / "x4t-layer-00024.bin"
    _write_layer(path, layer=24, expert=17)
    reader = x4t.X4TLayerReader(path)
    entry = reader.entries[17 * len(x4t.X4T_MATRIX_ORDER)]
    descriptor = os.open(path, os.O_RDWR)
    try:
        original = os.pread(descriptor, 1, entry.offset + x4t._RECORD_HEADER.size)
        os.pwrite(descriptor, bytes([original[0] ^ 1]), entry.offset + 64)
    finally:
        os.close(descriptor)

    with pytest.raises(ValueError, match="checksum"):
        reader.read(17, "w1")


def test_qsrt_v5_reader_joins_external_x4t_by_exact_inventory(tmp_path) -> None:
    """Guard the complete slab/X4T boundary consumed by one TP12 process."""

    layer = 24
    expert = 17
    rank = 5
    slab = tmp_path / "mixed-exl3-tp12-layer-00024.bin"
    sidecar = tmp_path / "x4t-layer-00024.bin"
    _write_qsrt_slab(slab, layer=layer, kept_expert=expert)
    _write_layer(sidecar, layer=layer, expert=expert)
    expected_bits = [3] * qsrt.EXPERTS
    expected_bits[expert] = 4

    payload = qsrt.read_tp12_rank_payload(
        slab,
        layer=layer,
        rank=rank,
        x4t_path=sidecar,
        expected_bits=expected_bits,
        expected_codebook=qsrt.CODEBOOK_SQG_NORMAL_E4M3,
        selected_experts=[expert],
    )

    assert payload.compressed_expert_ids.numel() == 0
    assert payload.kept_expert_ids.tolist() == [expert]
    assert tuple(payload.w13_mxfp4.shape) == (1, 512, 1792)
    assert tuple(payload.w13_mxfp4_scale.shape) == (1, 512, 112)
    assert tuple(payload.w2_mxfp4.shape) == (1, 3584, 128)
    assert tuple(payload.w2_mxfp4_scale.shape) == (1, 3584, 8)
    assert bool(torch.all(payload.w13_mxfp4 == rank))
    assert bool(torch.all(payload.w13_mxfp4_scale[:, :256] == 120))
    assert bool(torch.all(payload.w13_mxfp4_scale[:, 256:] == 121))
    assert bool(torch.all(payload.w2_mxfp4 == rank))
    assert bool(torch.all(payload.w2_mxfp4_scale == 122))
