# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the per-token KVarN MLA record codec (layout, pack/unpack)."""

import pytest
import torch

from vllm.v1.attention.ops.kvarn_mla import (
    KvarnMlaLayout,
    hadamard_matrix,
    kvarn_cache_dtype_enabled,
    kvarn_mla_layout_from_dtype,
    pack_bits,
    pack_kvarn_mla_records,
    rotate_latent,
    unpack_bits,
    unpack_kvarn_mla_records,
)

DTYPES = [
    "kvarn_mla_k4v4_g256",
    "kvarn_mla_k5v5_g256",
    "kvarn_mla_k4v2_g256",
    "kvarn_mla_k4_g256",
]


@pytest.mark.parametrize("dtype", DTYPES)
def test_dtype_dispatch(dtype: str):
    layout = kvarn_mla_layout_from_dtype(dtype)
    assert (
        layout.record_bytes
        == KvarnMlaLayout(
            k_bits=layout.k_bits, v_bits=layout.v_bits, group=layout.group
        ).record_bytes
    )
    # Mod-33 rule: dual variants keep 0 < r % 33 <= 21 (reserved per-tile
    # tail fits the C4 gap); single-plane variants end on the C4 boundary.
    if layout.v_bits == 0:
        assert layout.record_bytes % 33 == 0
    else:
        assert 0 < layout.record_bytes % 33 <= 21
    # Offsets must be non-overlapping, ordered, inside the record.
    offs = [
        layout.k_code_offset,
        layout.v_code_offset,
        layout.k_scale_offset,
        layout.k_zp_offset,
        layout.v_scale_offset,
        layout.v_zp_offset,
    ]
    assert offs == sorted(offs)
    assert offs[0] == 0
    assert offs[2] == layout.k_code_bytes + layout.v_code_bytes
    assert layout.v_zp_offset + 2 <= layout.content_bytes
    assert layout.content_bytes <= layout.record_bytes


def test_record_widths():
    # Content (codes + 4 fp16) 8-byte aligned, then the mod-33 rule.
    assert kvarn_mla_layout_from_dtype("kvarn_mla_k4v4_g256").record_bytes == 536
    assert kvarn_mla_layout_from_dtype("kvarn_mla_k5v5_g256").record_bytes == 648
    assert kvarn_mla_layout_from_dtype("kvarn_mla_k4v2_g256").record_bytes == 400
    assert kvarn_mla_layout_from_dtype("kvarn_mla_k4_g256").record_bytes == 264


@pytest.mark.parametrize("dtype", DTYPES)
def test_tile_stride_and_page_geometry(dtype: str):
    """The rounded tile stride + FP8 index tail must fit the allocated page
    for every supported dtype (the _index_cache_view contract)."""
    layout = kvarn_mla_layout_from_dtype(dtype)
    index_page = 8448  # 64 pools x 132 bytes C4 page
    # tile_stride_bytes is a whole number of C4 index pages.
    assert layout.tile_stride_bytes % index_page == 0
    # The page holds the rounded tile span plus one index tail page.
    assert layout.page_size_bytes == layout.tile_stride_bytes + index_page
    # Parent-page stride stays an exact C4-page multiple.
    assert layout.page_size_bytes % index_page == 0


@pytest.mark.parametrize("bits", [2, 4, 5])
@pytest.mark.parametrize("cols", [512, 517])
def test_pack_bits_roundtrip(bits: int, cols: int):
    gen = torch.Generator(device="cpu").manual_seed(0)
    codes = torch.randint(0, 2**bits, (7, cols), generator=gen, dtype=torch.int32)
    packed = pack_bits(codes, bits)
    assert packed.shape[-1] == (cols * bits + 7) // 8
    assert (unpack_bits(packed, bits, cols) == codes).all()


@pytest.mark.parametrize("dtype", DTYPES)
def test_pack_unpack_parity(dtype: str):
    """Per-token RTN parity: score-path cosine high, value-path bounded."""
    layout = kvarn_mla_layout_from_dtype(dtype)
    gen = torch.Generator(device="cpu").manual_seed(2)
    latent = torch.randn(layout.group, layout.latent_dim, generator=gen)

    records = pack_kvarn_mla_records(latent, layout)
    assert records.shape == (layout.group, layout.record_bytes)
    # Pad region is zero.
    assert int(records[:, layout.content_bytes :].sum()) == 0

    k_hat, v_hat = unpack_kvarn_mla_records(records, layout)
    assert k_hat.shape == latent.shape
    assert v_hat.shape == latent.shape
    # Dequantized scores preserve cosine structure of the Hadamard-rotated
    # latent (what attention logits see): mean cosine similarity must be high.
    k_hat_n = k_hat / k_hat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    lat_n = latent / latent.norm(dim=-1, keepdim=True)
    cos = (k_hat_n * lat_n).sum(-1)
    assert cos.mean() > 0.97, f"score-path cosine degraded: {cos.mean():.4f}"
    # Value path relative error bound for the variant's bits.
    rel = (v_hat - latent).norm() / latent.norm()
    if layout.v_bits >= 4:
        assert rel < 0.12, f"value-path relative error too high: {rel:.4f}"
    else:  # 2-bit value path
        assert rel < 0.65, f"value-path relative error too high: {rel:.4f}"


def test_single_plane_shared_code():
    """Single-plane (v_bits=0): one RTN grid, both paths read k_hat; the
    stored k-plane is bit-identical to the dual k4v4 record's k-plane."""
    single = kvarn_mla_layout_from_dtype("kvarn_mla_k4_g256")
    dual = kvarn_mla_layout_from_dtype("kvarn_mla_k4v4_g256")
    gen = torch.Generator(device="cpu").manual_seed(4)
    latent = torch.randn(64, single.latent_dim, generator=gen)

    rec_s = pack_kvarn_mla_records(latent, single)
    rec_d = pack_kvarn_mla_records(latent, dual)
    k_s, v_s = unpack_kvarn_mla_records(rec_s, single)
    k_d, _ = unpack_kvarn_mla_records(rec_d, dual)
    # Same grid/bits on the k plane -> identical dequantized codes.
    assert torch.equal(k_s, k_d)
    # The value path reads the shared k_hat.
    assert torch.equal(v_s, k_s)
    # Reserved v_scale/v_zp fields are zero.
    assert int(rec_s[:, single.v_scale_offset : single.v_zp_offset + 2].sum()) == 0
    # Geometry: 256B codes + 8B footer, records end exactly on the C4
    # boundary (264 = 8 C4 pages); no per-tile slack, index tail flush.
    assert single.record_bytes == 264
    assert single.record_bytes % 33 == 0
    assert single.tile_stride_bytes == 8 * 8448
    assert single.page_tail_bytes_per_token == 33
    assert single.page_size_bytes == 9 * 8448


def test_pack_unpack_vs_fp16_reference_cosine():
    """k5v5 must beat k4v2 on score-path fidelity (monotone in bits)."""
    gen = torch.Generator(device="cpu").manual_seed(3)
    latent = torch.randn(256, 512, generator=gen)

    def mean_cos(dtype):
        layout = kvarn_mla_layout_from_dtype(dtype)
        recs = pack_kvarn_mla_records(latent, layout)
        k_hat, _ = unpack_kvarn_mla_records(recs, layout)
        n = k_hat / k_hat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        ref = latent / latent.norm(dim=-1, keepdim=True)
        return (n * ref).sum(-1).mean().item()

    assert mean_cos("kvarn_mla_k5v5_g256") >= mean_cos("kvarn_mla_k4v2_g256")


def test_per_token_independence():
    """Records quantize per token: appending tokens never changes others."""
    gen = torch.Generator(device="cpu").manual_seed(7)
    layout = kvarn_mla_layout_from_dtype("kvarn_mla_k4v4_g256")
    base = torch.randn(64, 512, generator=gen)
    extra = torch.randn(8, 512, generator=gen)
    r1 = pack_kvarn_mla_records(base, layout)
    r2 = pack_kvarn_mla_records(torch.cat([base, extra]), layout)
    assert torch.equal(r1, r2[:64])


def test_hadamard_orthonormal():
    h = hadamard_matrix(512, torch.device("cpu"))
    torch.testing.assert_close(h @ h.T, torch.eye(512), rtol=1e-5, atol=1e-6)


def test_rotation_invariance_math():
    """H·q · H·k == q·k and H^T recovers the original frame (exact math)."""
    gen = torch.Generator(device="cpu").manual_seed(4)
    h = hadamard_matrix(512, torch.device("cpu"))
    h_t = h.transpose(-2, -1)
    q = torch.randn(8, 512, generator=gen)
    k = torch.randn(8, 512, generator=gen)
    v = torch.randn(8, 512, generator=gen)
    qr = rotate_latent(q, h_t)  # H q
    kr = rotate_latent(k, h_t)  # H k
    torch.testing.assert_close((qr * kr).sum(-1), (q * k).sum(-1), rtol=1e-4, atol=1e-4)
    # output side: attention accumulates H v; H^T (H v) == v
    out_rot = rotate_latent(v, h_t)
    out = rotate_latent(out_rot, h_t.transpose(-2, -1))  # H^T (H v)
    torch.testing.assert_close(out, v, rtol=1e-4, atol=1e-4)


def test_kvarn_cache_dtype_enabled():
    assert kvarn_cache_dtype_enabled("kvarn_mla_k4v2_g256")
    assert not kvarn_cache_dtype_enabled("nvfp4_ds_mla")
    assert not kvarn_cache_dtype_enabled(None)
