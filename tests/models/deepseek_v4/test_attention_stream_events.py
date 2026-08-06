# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import torch

from vllm.models.deepseek_v4 import attention as attention_module
from vllm.models.deepseek_v4.nvidia.b12x import DeepseekV4B12xMLAAttention
from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
    DeepseekV4FlashInferSM120Attention,
)


def test_sm120_backends_enable_post_gemm_aux_streams_for_dspark() -> None:
    assert DeepseekV4B12xMLAAttention.enable_post_gemm_aux_streams is True
    assert DeepseekV4FlashInferSM120Attention.enable_post_gemm_aux_streams is True


def test_post_gemm_aux_stream_gate_covers_every_attention_path() -> None:
    streams = [object(), object(), object()]
    layer = SimpleNamespace(
        aux_stream_list=streams,
        enable_post_gemm_aux_streams=False,
    )

    for index in range(len(streams)):
        assert (
            attention_module.DeepseekV4Attention._post_gemm_aux_stream(layer, index)
            is None
        )

    layer.enable_post_gemm_aux_streams = True
    for index, stream in enumerate(streams):
        assert (
            attention_module.DeepseekV4Attention._post_gemm_aux_stream(layer, index)
            is stream
        )


def test_gemm_and_attention_overlap_use_distinct_event_sets(monkeypatch) -> None:
    calls = []

    def fake_execute_in_parallel(
        default_fn,
        aux_fns,
        start_event,
        done_events,
        aux_streams,
        enable,
        **kwargs,
    ):
        del default_fn, aux_streams, enable, kwargs
        calls.append((start_event, tuple(done_events)))
        return torch.empty(1), [None] * len(aux_fns)

    monkeypatch.setattr(
        attention_module, "execute_in_parallel", fake_execute_in_parallel
    )
    monkeypatch.setattr(
        attention_module,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=None),
    )

    ln_events = [object() for _ in range(4)]
    attn_events = [object() for _ in range(3)]
    layer = SimpleNamespace(
        aux_stream_list=[object(), object(), object()],
        compressor=object(),
        indexer=object(),
        fused_wqa_wkv=object(),
        ln_events=ln_events,
        attn_events=attn_events,
        _post_gemm_event_lease=lambda: nullcontext(attn_events),
        enqueue_default_before_indexer=True,
        enable_post_gemm_aux_streams=True,
        indexer_rotary_emb=object(),
        rotary_emb=object(),
        forward_mqa=lambda *args: None,
    )
    tensor = torch.empty(1)

    attention_module.DeepseekV4Attention.attn_gemm_parallel_execute(layer, tensor)
    attention_module.DeepseekV4Attention.attention_impl(
        layer,
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
    )

    assert calls == [
        (ln_events[0], tuple(ln_events[1:4])),
        (attn_events[0], tuple(attn_events[1:3])),
    ]
    assert set(map(id, ln_events)).isdisjoint(map(id, attn_events))


def test_attention_overlap_uses_capture_private_events(monkeypatch) -> None:
    calls = []
    captured_events = [object() for _ in range(3)]

    def fake_execute_in_parallel(
        default_fn,
        aux_fns,
        start_event,
        done_events,
        aux_streams,
        enable,
        **kwargs,
    ):
        del default_fn, aux_streams, enable, kwargs
        calls.append((start_event, tuple(done_events)))
        return torch.empty(1), [None] * len(aux_fns)

    monkeypatch.setattr(
        attention_module, "execute_in_parallel", fake_execute_in_parallel
    )
    monkeypatch.setattr(
        attention_module,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=None),
    )

    layer = SimpleNamespace(
        aux_stream_list=[object(), object(), object()],
        compressor=object(),
        indexer=object(),
        attn_events=[object() for _ in range(3)],
        attn_event_pool=SimpleNamespace(
            lease=lambda **_kwargs: nullcontext(captured_events)
        ),
        _post_gemm_event_lease=lambda: nullcontext(captured_events),
        enqueue_default_before_indexer=True,
        enable_post_gemm_aux_streams=True,
        indexer_rotary_emb=object(),
        rotary_emb=object(),
        forward_mqa=lambda *args: None,
    )
    tensor = torch.empty(1)

    attention_module.DeepseekV4Attention.attention_impl(
        layer,
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
    )

    assert calls == [(captured_events[0], tuple(captured_events[1:3]))]
