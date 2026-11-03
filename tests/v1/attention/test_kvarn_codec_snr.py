# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end torch A/B: kvarn pipeline vs exact MLA attention.

Verifies the DESIGN math assembly (rotation + folds + codec) on GPU:
  reference:  score = (q_nope @ W_UK_T) . kv_c ;  o = P @ (kv_c @ W_UV)
  kvarn:      l = kv_c @ H^T (store frame)
              q_abs' = q_nope @ (W_UK_T @ H^T)
              score' = q_abs' . dequant(pack(l))
              o'     = P @ dequant(l) @ (H @ W_UV)
  H orthonormal => o' ~= o up to quantization noise.
"""

import math

import pytest
import torch

from vllm.v1.attention.ops.kvarn_mla import (
    hadamard_matrix,
    kvarn_mla_layout_from_dtype,
    pack_kvarn_mla_records,
    unpack_kvarn_mla_records,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

GROUP, L, N, V = 256, 512, 4, 512
S_QUERY = 64  # separate query axis so softmax spans cache tokens


@requires_cuda
def test_kvarn_codec_snr():
    """The kvarn codec must keep signal-to-noise above a sane floor."""
    torch.manual_seed(7)
    dev = torch.device("cuda")
    layout = kvarn_mla_layout_from_dtype("kvarn_mla_k4v4_g256")

    # Cache tokens (one tile) and a separate query axis.
    kv_c = torch.randn(GROUP, L, device=dev, dtype=torch.float32) * 0.5
    q_nope = torch.randn(S_QUERY, N, 128, device=dev, dtype=torch.float32) * 0.5
    W_UK = torch.randn(L, N, 128, device=dev, dtype=torch.float32) * 0.05
    W_UV = torch.randn(L, N, V, device=dev, dtype=torch.float32) * 0.05

    # ---- reference (exact) ----
    q_c = torch.einsum("snp,lnp->snl", q_nope, W_UK)  # [Sq,N,L]
    score = torch.einsum("qnl,cl->qnc", q_c, kv_c) * (1.0 / math.sqrt(L))
    P = torch.softmax(score, dim=-1)  # normalize over cache tokens
    W_UV_flat = W_UV.reshape(L, N * V)  # [L, N*V]
    v = (kv_c @ W_UV_flat).view(GROUP, N, V)  # [Sc,N,V]
    o_ref = torch.einsum("qnc,cnv->qnv", P, v)  # [Sq,N,V]

    # ---- kvarn pipeline ----
    h = hadamard_matrix(L, dev).to(torch.float32)
    h_t = h.transpose(-2, -1)
    l_rot = kv_c @ h_t  # stored frame [Sc, L]

    records = pack_kvarn_mla_records(l_rot, layout)
    l_hat, v2_hat = unpack_kvarn_mla_records(records, layout)

    # folded weights
    W_UK_T = W_UK.permute(1, 2, 0)  # [N, P, L]
    W_UK_T_folded = W_UK_T @ h_t  # q_abs' = q_nope @ W_UK_T' (rotated frame)
    W_UV_n = W_UV.permute(1, 0, 2)  # [N, L, V]
    W_UV_folded = h @ W_UV_n  # o = P l @ (H W_UV)

    q_abs = torch.einsum("snp,npl->snl", q_nope, W_UK_T_folded)
    score2 = torch.einsum("qnl,cl->qnc", q_abs, l_hat) * (1.0 / math.sqrt(L))
    P2 = torch.softmax(score2, dim=-1)  # normalize over cache tokens
    # Output uses the decoded V plane, not the K plane.
    v2 = torch.einsum("cl,nlv->cnv", v2_hat, W_UV_folded)  # [Sc,N,V]
    o2 = torch.einsum("qnc,cnv->qnv", P2, v2)  # [Sq,N,V]

    lsnr = 10 * torch.log10((l_rot**2).mean() / ((l_hat - l_rot) ** 2).mean())
    # Output SNR: end-to-end attention output vs exact reference.
    snr = 10 * torch.log10((o_ref**2).mean() / ((o2 - o_ref) ** 2).mean())

    assert lsnr.item() >= 15.0, f"codec SNR too low: {lsnr.item():.1f} dB"
    assert snr.item() >= 15.0, f"output SNR too low: {snr.item():.1f} dB"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
