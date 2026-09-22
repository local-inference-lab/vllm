# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regression checks for mixed DiffKV request partitioning."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.v1.attention.backend import AttentionCGSupport, CommonAttentionMetadata
from vllm.v1.attention.backends import triton_attn
from vllm.v1.attention.backends import triton_attn_diffkv as backend

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def builder(monkeypatch):
    monkeypatch.setattr(
        triton_attn, "get_num_attention_heads_from_layers", lambda *a: 16
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            get_num_kv_heads=lambda _: 1,
            get_head_size=lambda: 192,
            rswa_window=None,
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
        speculative_config=SimpleNamespace(
            num_speculative_tokens=7, parallel_drafting=False
        ),
    )
    return backend.TritonAttentionDiffKVMetadataBuilder(
        SimpleNamespace(block_size=16), [], config, torch.device("cpu")
    )


def common(q_lens):
    starts = torch.tensor(
        [0, *torch.tensor(q_lens).cumsum(0).tolist()], dtype=torch.int32
    )
    lengths = torch.tensor([2048 + n for n in q_lens], dtype=torch.int32)
    return CommonAttentionMetadata(
        query_start_loc=starts,
        query_start_loc_cpu=starts.clone(),
        seq_lens=lengths,
        num_reqs=len(q_lens),
        num_actual_tokens=sum(q_lens),
        max_query_len=max(q_lens),
        max_seq_len=int(lengths.max()),
        block_table_tensor=torch.arange(len(q_lens) * 256, dtype=torch.int32).reshape(
            -1, 256
        ),
        slot_mapping=torch.arange(sum(q_lens)),
    )


def run_forward(monkeypatch, metadata):
    calls = []
    n = metadata.num_actual_tokens
    q = torch.arange((n + 3) * 16 * 192, dtype=torch.float32).reshape(n + 3, 16, 192)
    out = torch.full((n + 3, 16, 128), -17.0)

    def attention(**kw):
        calls.append(kw)
        # Observable writes also check output placement and untouched padding.
        kw["out"].copy_(kw["q"][..., :128])

    monkeypatch.setattr(backend, "unified_attention_diffkv", attention)
    impl = object.__new__(backend.TritonAttentionDiffKVImpl)
    impl.head_size, impl.scale = 192, 192**-0.5
    impl.kv_cache_dtype = "bfloat16"
    impl.alibi_slopes, impl.use_alibi_sqrt = None, False
    impl.sliding_window, impl.logits_soft_cap = (127, 0), 0
    impl.sinks = torch.arange(16, dtype=torch.float32)
    layer = SimpleNamespace(_k_scale=torch.tensor(0.5), _v_scale=torch.tensor(1.25))
    cache = torch.empty(1, 1, 16, 320)
    impl.forward(layer, q, None, None, cache, metadata, out)
    torch.testing.assert_close(out[:n], q[:n, :, :128])
    assert (out[n:] == -17).all()
    for call in calls:
        assert call["window_size"] == (127, 0)
        assert call["sinks"] is impl.sinks
        assert (
            call["k"].untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
        )
    return calls


@pytest.mark.parametrize("decode_lens", [[1], [8], [8, 3, 1], [8] * 16])
def test_mixed_batch_separates_short_queries(builder, monkeypatch, decode_lens):
    c = common([*decode_lens, 257, 129])
    calls = run_forward(monkeypatch, builder.build(0, c))
    assert len(calls) == 2
    decode, prefill = calls
    n, r = sum(decode_lens), len(decode_lens)
    assert decode["q"].shape[0] == n
    assert prefill["q"].shape[0] == 386
    torch.testing.assert_close(decode["cu_seqlens_q"], c.query_start_loc[: r + 1])
    torch.testing.assert_close(
        prefill["cu_seqlens_q"], torch.tensor([0, 257, 386], dtype=torch.int32)
    )
    torch.testing.assert_close(decode["seqused_k"], c.seq_lens[:r])
    torch.testing.assert_close(prefill["seqused_k"], c.seq_lens[r:])
    torch.testing.assert_close(decode["block_table"], c.block_table_tensor[:r])
    torch.testing.assert_close(prefill["block_table"], c.block_table_tensor[r:])
    assert decode["max_seqlen_q"] == max(decode_lens)
    assert prefill["max_seqlen_q"] == 257


@pytest.mark.parametrize("q_lens", [[8, 3, 1], [257, 129], [8, 16], [0, 8, 0]])
def test_single_launch_when_partition_cannot_help(builder, monkeypatch, q_lens):
    calls = run_forward(monkeypatch, builder.build(0, common(q_lens)))
    assert len(calls) == 1


@pytest.mark.parametrize(
    "buffer", ["softmax_segm_output", "softmax_segm_max", "softmax_segm_expsum"]
)
def test_small_workspace_does_not_split(builder, monkeypatch, buffer):
    setattr(builder, buffer, getattr(builder, buffer)[:4])
    calls = run_forward(monkeypatch, builder.build(0, common([8, 257])))
    assert len(calls) == 1


def test_builder_preserves_graph_safe_decode_and_requests_reordering(builder):
    assert builder.reorder_batch_threshold == 8
    assert (
        builder.get_cudagraph_support(builder.vllm_config, builder.kv_cache_spec)
        == AttentionCGSupport.UNIFORM_BATCH
    )


def test_rebuilt_metadata_tracks_changed_request_boundary(builder, monkeypatch):
    for q_lens, expected_decode in [([8, 8, 256], 16), ([8, 264], 8), ([272], 272)]:
        calls = run_forward(monkeypatch, builder.build(0, common(q_lens)))
        assert calls[0]["q"].shape[0] == expected_decode


def test_single_token_decode_dispatches_split_kv_in_mixed_batch(builder, monkeypatch):
    from vllm.v1.attention.ops import triton_unified_attention_diffkv as ops

    launches = []

    class KernelSpy:
        def __getitem__(self, grid):
            def launch(**kwargs):
                launches.append(kwargs.get("IS_3D"))

            return launch

    monkeypatch.setattr(ops, "kernel_unified_attention_diffkv", KernelSpy())
    monkeypatch.setattr(ops, "kernel_reduce_segments_diffkv", KernelSpy())
    real_attention = backend.unified_attention_diffkv
    metadata = builder.build(0, common([1, 257]))
    # run_forward owns the output-placement spy. Invoke the real dispatcher
    # separately on each observed call, with every GPU launch intercepted.
    calls = run_forward(monkeypatch, metadata)
    for call in calls:
        real_attention(**call)
    assert launches == [True, None, False]


def test_draft_metadata_update_keeps_live_single_launch_views(builder):
    c = common([1, 1, 1])
    metadata = builder.build(0, c)
    assert not getattr(metadata, "partitions", ())
    c.seq_lens.add_(1)
    builder.update_draft_decode_metadata(metadata)
    assert metadata.seq_lens.data_ptr() == c.seq_lens.data_ptr()
    torch.testing.assert_close(metadata.seq_lens, c.seq_lens)
