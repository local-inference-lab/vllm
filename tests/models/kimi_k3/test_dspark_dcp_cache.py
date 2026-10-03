# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.models.kimi_k3.nvidia import mla


def _cache_spec(
    monkeypatch,
    *,
    non_causal: bool,
    dcp_size: int,
    draft_kv_window: int = 0,
):
    attention = object.__new__(mla.MultiHeadLatentAttention)
    attention.kv_cache_dtype = "auto"
    attention.head_size = 640
    attention.non_causal_multi_token_decode = non_causal
    attention.draft_kv_window = draft_kv_window
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=64),
        model_config=object(),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp_size,
        ),
    )
    monkeypatch.setattr(
        mla,
        "kv_cache_dtype_str_to_dtype",
        lambda _cache_dtype, _model_config: torch.bfloat16,
    )
    monkeypatch.setattr(mla, "get_kv_quant_mode", lambda _cache_dtype: None)
    return mla.MultiHeadLatentAttention.get_kv_cache_spec(attention, config)


def test_external_k3_dspark_cache_is_replicated_under_target_dcp(
    monkeypatch,
) -> None:
    monkeypatch.delenv("VLLM_DCP_SHARD_DRAFT", raising=False)

    spec = _cache_spec(monkeypatch, non_causal=True, dcp_size=16)

    assert spec.dcp_replicated


def test_explicit_draft_sharding_disables_cache_replication(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_DCP_SHARD_DRAFT", "1")

    spec = _cache_spec(monkeypatch, non_causal=True, dcp_size=16)

    assert not spec.dcp_replicated


def test_target_k3_cache_is_not_marked_as_draft_replication(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_DCP_SHARD_DRAFT", raising=False)

    spec = _cache_spec(monkeypatch, non_causal=False, dcp_size=16)

    assert not spec.dcp_replicated


def test_bounded_dspark_cache_preserves_non_causal_mode(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_DCP_SHARD_DRAFT", raising=False)

    spec = _cache_spec(
        monkeypatch,
        non_causal=True,
        dcp_size=16,
        draft_kv_window=65_536,
    )

    assert isinstance(spec, mla.SlidingWindowMLASpec)
    assert spec.sliding_window == 65_536
    assert spec.non_causal_multi_token_decode
    assert spec.dcp_replicated
