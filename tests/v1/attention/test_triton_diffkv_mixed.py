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
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def builder_factory(monkeypatch):
    monkeypatch.setattr(
        triton_attn, "get_num_attention_heads_from_layers", lambda *a: 16
    )

    def make(
        max_seqs=16,
        spec_tokens=7,
        token_budget=4096,
        captures=(),
        parallel_drafting=False,
        dcp=1,
        kv_heads=1,
        spec_kind="full",
    ):
        config = SimpleNamespace(
            model_config=SimpleNamespace(
                get_num_kv_heads=lambda _: kv_heads,
                get_head_size=lambda: 192,
                rswa_window=None,
            ),
            parallel_config=SimpleNamespace(decode_context_parallel_size=dcp),
            compilation_config=SimpleNamespace(
                cudagraph_mode=CUDAGraphMode.FULL if captures else CUDAGraphMode.NONE,
                cudagraph_capture_sizes=list(captures),
            ),
            scheduler_config=SimpleNamespace(
                max_num_seqs=max_seqs, max_num_batched_tokens=token_budget
            ),
            speculative_config=SimpleNamespace(
                num_speculative_tokens=spec_tokens, parallel_drafting=parallel_drafting
            )
            if spec_tokens
            else None,
        )
        spec_args = dict(
            block_size=16,
            num_kv_heads=kv_heads,
            head_size=192,
            head_size_v=128,
            dtype=torch.bfloat16,
        )
        if spec_kind == "sliding":
            spec = SlidingWindowSpec(**spec_args, sliding_window=128)
        else:
            spec = FullAttentionSpec(
                **spec_args,
                sliding_window=128 if spec_kind == "full-sliding" else None,
                attention_chunk_size=128 if spec_kind == "full-chunked" else None,
            )
        return backend.TritonAttentionDiffKVMetadataBuilder(
            spec, [], config, torch.device("cpu")
        )

    return make


@pytest.fixture
def builder(builder_factory):
    return builder_factory()


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
        if "fp8_e4m3" in backend.TritonAttentionDiffKVBackend.supported_kv_cache_dtypes:
            # Also guards merging this change with the E4M3 backend: omitting
            # either descale in the new loop must fail, not silently pass.
            assert call["k_descale"] is layer._k_scale
            assert call["v_descale"] is layer._v_scale
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


@pytest.mark.parametrize(
    "q_lens", [[8, 3, 1], [257, 129], [8, 16], [0, 8, 0], [257, 8], [8] * 17 + [257]]
)
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


@pytest.fixture
def dispatcher_spy(monkeypatch):
    from vllm.v1.attention.ops import triton_unified_attention_diffkv as ops

    launches = []

    class KernelSpy:
        def __getitem__(self, grid):
            def launch(**kwargs):
                launches.append(kwargs.get("IS_3D"))

            return launch

    monkeypatch.setattr(ops, "kernel_unified_attention_diffkv", KernelSpy())
    monkeypatch.setattr(ops, "kernel_reduce_segments_diffkv", KernelSpy())
    return backend.unified_attention_diffkv, launches


def test_single_token_decode_dispatches_split_kv_in_mixed_batch(
    builder, monkeypatch, dispatcher_spy
):
    real_attention, launches = dispatcher_spy
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


def test_uniform_capture_has_no_host_partitions(builder):
    c = common([8] * 16)
    metadata = builder.build_for_cudagraph_capture(c)
    assert not getattr(metadata, "partitions", ())
    assert metadata.seq_lens.data_ptr() == c.seq_lens.data_ptr()
    assert (metadata.seq_lens == 1).all()


@pytest.mark.parametrize(
    "config,capacity",
    [
        pytest.param(dict(max_seqs=32), 256, id="c32-eager"),
        pytest.param(dict(max_seqs=32, spec_kind="sliding"), 128, id="swa-eager"),
        pytest.param(
            dict(max_seqs=32, spec_kind="sliding", captures=(1, 128, 256, 512)),
            128,
            id="swa-graphs",
        ),
        pytest.param(
            dict(max_seqs=32, spec_kind="full-sliding"),
            128,
            id="swa-with-full-allocation",
        ),
        pytest.param(
            dict(max_seqs=32, spec_kind="full-chunked"),
            128,
            id="chunked-with-full-allocation",
        ),
        pytest.param(
            dict(max_seqs=32, captures=(1, 128, 256, 512)), 256, id="c32-graphs"
        ),
        pytest.param(dict(max_seqs=24), 192, id="actual-concurrency"),
        pytest.param(dict(max_seqs=32, spec_tokens=3), 128, id="shorter-verification"),
        pytest.param(dict(max_seqs=32, spec_tokens=0), 128, id="no-speculation"),
        pytest.param(dict(max_seqs=32, token_budget=200), 200, id="token-budget"),
        pytest.param(
            dict(max_seqs=32, captures=(1, 128, 192)), 192, id="capture-ceiling"
        ),
        pytest.param(
            dict(max_seqs=32, captures=(1, 64)), 64, id="small-capture-ceiling"
        ),
        pytest.param(
            dict(max_seqs=32, parallel_drafting=True), 480, id="parallel-drafting"
        ),
        pytest.param(dict(max_seqs=32, dcp=2), 128, id="unsupported-dcp"),
        pytest.param(dict(max_seqs=32, kv_heads=2), 256, id="two-kv-heads"),
    ],
)
def test_speculative_workspace_capacity(builder_factory, config, capacity):
    builder = builder_factory(**config)
    assert builder.seq_threshold_3D == capacity
    for tensor, shape in (
        (builder.softmax_segm_output, (capacity, 16, 16, 128)),
        (builder.softmax_segm_max, (capacity, 16, 16)),
        (builder.softmax_segm_expsum, (capacity, 16, 16)),
    ):
        assert tensor.shape == shape
        assert tensor.dtype == torch.float32
        assert tensor.device.type == "cpu"


@pytest.mark.parametrize(
    "captures,q_lens,expected_rows",
    [
        ((1, 128, 256, 512), [8] * 32, [256]),
        ((1, 128, 256, 512), [8, 7, 3, 1] * 8, [152]),
        ((1, 128, 192), [8] * 24, [192]),
        ((1, 128, 192), [8] * 24 + [1], [193]),
        ((1, 128, 192), [8] * 32, [256]),
        ((1, 128, 256, 512), [8] * 31 + [1024], [248, 1024]),
        ((1, 128, 256, 512), [1024], [1024]),
    ],
)
def test_speculative_workspace_forward_metadata(
    builder_factory, monkeypatch, captures, q_lens, expected_rows
):
    builder = builder_factory(max_seqs=32, captures=captures)
    metadata = builder.build(0, common(q_lens))
    calls = run_forward(monkeypatch, metadata)
    assert [call["q"].shape[0] for call in calls] == expected_rows
    for call in calls:
        assert call["seq_threshold_3D"] == builder.seq_threshold_3D
        for name in ("softmax_segm_output", "softmax_segm_max", "softmax_segm_expsum"):
            assert call[name] is getattr(builder, name)


@pytest.mark.parametrize(
    "buffer", ["softmax_segm_output", "softmax_segm_max", "softmax_segm_expsum"]
)
def test_c32_partition_checks_each_workspace(builder_factory, monkeypatch, buffer):
    builder = builder_factory(max_seqs=32)
    setattr(builder, buffer, getattr(builder, buffer)[:247])
    metadata = builder.build(0, common([8] * 31 + [1024]))
    assert not metadata.partitions
    calls = run_forward(monkeypatch, metadata)
    assert [call["q"].shape[0] for call in calls] == [1272]


@pytest.mark.parametrize("spec_kind", ["sliding", "full-sliding"])
@pytest.mark.parametrize("q_lens", [[8] * 16, [8] * 32, [8] * 31 + [1024]])
def test_local_attention_retains_inherited_capacity(
    builder_factory, monkeypatch, spec_kind, q_lens
):
    builder = builder_factory(
        max_seqs=32, captures=(1, 128, 256, 512), spec_kind=spec_kind
    )
    assert builder.seq_threshold_3D == 128
    metadata = builder.build(0, common(q_lens))
    assert not metadata.partitions
    for call in run_forward(monkeypatch, metadata):
        assert call["seq_threshold_3D"] == 128
