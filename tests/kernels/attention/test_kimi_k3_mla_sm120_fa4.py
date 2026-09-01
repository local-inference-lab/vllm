# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exactness oracles for Kimi-K3's opt-in SM120 FA4 prefill path."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.prefill.flash_attn import (
    FlashAttnPrefillBackend,
)
from vllm.vllm_flash_attn import flash_attn_varlen_func

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(120),
    reason="requires consumer Blackwell SM120/SM121",
)

_HEADS = 12
_QK_DIM = 192
_V_DIM = 128


def _run(
    version: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indptr_q: torch.Tensor,
    indptr_k: torch.Tensor,
    max_q: int,
    max_k: int,
    *,
    causal: bool,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if version == 2:
        v = torch.nn.functional.pad(v, (0, _QK_DIM - _V_DIM))
    output, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=indptr_q,
        cu_seqlens_k=indptr_k,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        softmax_scale=_QK_DIM**-0.5,
        causal=causal,
        return_softmax_lse=True,
        out=out,
        num_splits=1 if version == 4 else 0,
        fa_version=version,
    )
    return output[..., :_V_DIM], lse


@torch.inference_mode()
def test_sm120_fa4_kimi_context_matches_fa2_and_honors_out() -> None:
    torch.manual_seed(23)
    q_len, kv_len = 128, 2048
    q = torch.randn(q_len, _HEADS, _QK_DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(kv_len, _HEADS, _QK_DIM, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(kv_len, _HEADS, _V_DIM, device="cuda", dtype=torch.bfloat16)
    qo = torch.tensor([0, q_len], device="cuda", dtype=torch.int32)
    kv = torch.tensor([0, kv_len], device="cuda", dtype=torch.int32)

    expected, expected_lse = _run(2, q, k, v, qo, kv, q_len, kv_len, causal=False)
    out = torch.empty_like(expected)
    actual, actual_lse = _run(4, q, k, v, qo, kv, q_len, kv_len, causal=False, out=out)

    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=0, atol=3e-6)


@torch.inference_mode()
def test_sm120_fa4_kimi_varlen_causal_matches_fa2() -> None:
    torch.manual_seed(29)
    lengths = (1024, 512)
    total = sum(lengths)
    q = torch.randn(total, _HEADS, _QK_DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(total, _HEADS, _V_DIM, device="cuda", dtype=torch.bfloat16)
    indptr = torch.tensor([0, lengths[0], total], device="cuda", dtype=torch.int32)

    expected, expected_lse = _run(
        2, q, k, v, indptr, indptr, max(lengths), max(lengths), causal=True
    )
    actual, actual_lse = _run(
        4, q, k, v, indptr, indptr, max(lengths), max(lengths), causal=True
    )

    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=0, atol=3e-6)


@torch.inference_mode()
def test_sm120_fa4_kimi_backend_context_wiring(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_MLA_SM120_FA4_PREFILL", "1")
    q_len, kv_len = 64, 1024
    backend = FlashAttnPrefillBackend(
        num_heads=_HEADS,
        scale=_QK_DIM**-0.5,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=_V_DIM,
        vllm_config=MagicMock(),
    )
    assert backend.vllm_flash_attn_version == 4
    assert backend.supports_out()

    q = torch.randn(q_len, _HEADS, _QK_DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(kv_len, _HEADS, _QK_DIM, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(kv_len, _HEADS, _V_DIM, device="cuda", dtype=torch.bfloat16)
    qo = torch.tensor([0, q_len], device="cuda", dtype=torch.int32)
    kv = torch.tensor([0, kv_len], device="cuda", dtype=torch.int32)
    chunk = SimpleNamespace(
        query_start_loc=qo,
        cu_seq_lens=kv,
        max_query_len=q_len,
        max_seq_len=kv_len,
    )
    out = torch.empty(q_len, _HEADS, _V_DIM, device="cuda", dtype=torch.bfloat16)

    actual, actual_lse = backend.run_prefill_context_chunk(chunk, q, k, v, out=out)
    expected, expected_lse = _run(2, q, k, v, qo, kv, q_len, kv_len, causal=False)

    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=0, atol=3e-6)
