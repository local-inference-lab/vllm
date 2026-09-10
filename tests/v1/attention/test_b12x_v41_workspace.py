# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-kernel memory and graph-lifetime regressions for V4.1 attention."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Native b12x attention requires CUDA"
)


@pytest.fixture
def native_workspace(monkeypatch):
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("Native b12x attention requires SM12x")
    import vllm.v1.worker.workspace as workspace
    from vllm.models.deepseek_v4_1 import attention

    manager = workspace.WorkspaceManager(
        torch.device("cuda", torch.cuda.current_device())
    )
    monkeypatch.setattr(workspace, "_manager", manager)
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    monkeypatch.setattr(attention, "get_tensor_model_parallel_world_size", lambda: 1)
    with workspace.use_workspace_lane(0):
        yield attention, manager, workspace


def _layer(attention, layer_id=0):
    # Avoid checkpoint/model construction: exercise the real planning and
    # attention methods with the same TP4 head geometry and serving capacity.
    layer = attention.DeepseekV4Attention.__new__(attention.DeepseekV4Attention)
    torch.nn.Module.__init__(layer)
    layer.prefix = f"model.layers.{layer_id}.self_attn"
    layer.layer_id = layer_id
    layer.config = SimpleNamespace(cache_config=SimpleNamespace(block_size=64))
    layer.capacity = 4096
    layer.max_model_len = 1024
    layer.n_local_heads = 16
    layer.swa_width = layer.window_size = 128
    layer.compress_ratio = 1
    layer.is_draft = False
    layer.is_index_source = True
    layer.kv_source_layer_id = layer.index_source_layer_id = layer_id
    layer.candidate_source_layer = 99
    layer.topk_indices_buffer = None
    layer.indexer = SimpleNamespace(
        heads=8, k_cache=SimpleNamespace(prefix=layer.prefix + ".indexer.k_cache")
    )
    layer.swa_cache_layer = SimpleNamespace(prefix=layer.prefix + ".swa_cache")
    layer.compressor = None
    layer._context = {layer.prefix: layer}
    layer._ready = False
    return layer


def test_prepare_memory_is_metadata_not_capacity_activations(native_workspace):
    attention, manager, _ = native_workspace
    device = torch.device("cuda", torch.cuda.current_device())
    first = _layer(attention)
    first._prepare(device)
    manager.lock()
    before = torch.cuda.memory_allocated()
    second = _layer(attention, 1)
    second._prepare(device)
    allocated = torch.cuda.memory_allocated() - before
    c = second.CHUNK
    metadata_bytes = 4 * (
        c * (second.swa_width + second._main_width + second._index_width) + 3 * c + 1
    )
    persistent_topk_bytes = 4 * second.capacity * 512
    # A second layer must not own Q/O/inverse, indexer activations, or another
    # copy of either mode's planned scratch. Allow CUDA allocator rounding.
    assert allocated <= metadata_bytes + persistent_topk_bytes + 1024**2


@pytest.mark.parametrize("is_decode", [True, False])
def test_attention_shared_scratch_graph_replay(
    native_workspace, monkeypatch, is_decode
):
    attention, manager, workspace = native_workspace
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(142)
    layer = _layer(attention)
    layer._prepare(device)
    rows, length = 65, 1024  # Cross the static 64-row scratch boundary.
    positions = torch.arange(length - rows, length, dtype=torch.int64, device=device)
    starts = torch.tensor([0, rows], dtype=torch.int32, device=device)
    reqs = torch.zeros(rows, dtype=torch.int32, device=device)
    visible = (positions + 1).int()

    def metadata(page):
        return SimpleNamespace(
            positions=positions,
            req_id_per_token=reqs,
            block_table=torch.arange(
                1, length // page + 1, dtype=torch.int32, device=device
            )[None],
            query_start_loc=starts,
            request_positions=positions[:1],
            cache_lengths=visible,
            is_decode=is_decode,
        )

    context = SimpleNamespace(
        attn_metadata={
            layer.swa_cache_layer.prefix: metadata(32),
            layer.prefix: metadata(64),
            layer.indexer.k_cache.prefix: metadata(64),
        }
    )
    monkeypatch.setattr(attention, "get_forward_context", lambda: context)
    kv = torch.randn((length, 512), device=device, dtype=torch.bfloat16)
    for kind, page in (("swa", 32), ("indexed", 64)):
        cache = torch.empty(
            (
                length // page + 1,
                attention.mla.page_nbytes(
                    page, cache_kind=kind, cache_format="deepseek_v41"
                ),
            ),
            device=device,
            dtype=torch.uint8,
        )
        attention.mla.write_cache(
            kv,
            cache,
            torch.arange(length, device=device) + page,
            page_size=page,
            cache_kind=kind,
            cache_format="deepseek_v41",
        )
        if kind == "swa":
            layer.swa_cache_layer.kv_cache = cache
        else:
            layer.kv_cache = cache
    layer.indexer.k_cache.kv_cache = torch.empty(
        (length // 64 + 1, attention.dsa_indexer.MXFP4_INDEX_PAGE_BYTES),
        dtype=torch.uint8,
        device=device,
    )
    # Every query sees all three groups. A=e0, B=e0+e1, C=e1:
    # (1,-.25) selects exactly A+B; (-.25,1) selects exactly B+C.
    # The 512 selected scores are positive and all others are zero, so
    # the native selector's unspecified BF16 cutoff tie order is irrelevant.
    index_keys = torch.zeros((length, 128), dtype=torch.bfloat16, device=device)
    index_keys[:512, 0] = 1
    index_keys[256:768, 1] = 1
    attention.dsa_indexer.quantize_write_index_k_mxfp4(
        index_keys,
        index_k_cache=layer.indexer.k_cache.kv_cache,
        slot_mapping=torch.arange(length, device=device) + 64,
    )
    layer.attn_sink = torch.zeros(layer.n_local_heads, device=device)
    q = torch.randn(
        (rows, layer.n_local_heads, 512), dtype=torch.bfloat16, device=device
    )
    iq = torch.zeros((rows, 8, 128), dtype=torch.bfloat16, device=device)
    iq[..., 0], iq[..., 1] = 1, -0.25
    weights = (torch.rand((rows, 8), dtype=torch.bfloat16, device=device) + 1) / 64
    packed = torch.empty((rows, 8, 64), dtype=torch.uint8, device=device)
    scales = torch.empty((rows, 8, 4), dtype=torch.uint8, device=device)
    out = torch.empty_like(q)

    def run():
        attention.dsa_indexer.quantize_q_mxfp4(iq, q_mxfp4=packed, q_scales=scales)
        layer.forward_mqa(
            q, None, positions, out, index_query=(packed, scales, weights)
        )

    run()
    manager.lock()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with workspace.collect_cuda_graph_capture_resources() as resources:
        with torch.cuda.graph(graph, stream=stream):
            run()
    # Changed queries exercise both selection and attention on replay, after
    # all plan storage has been reused by another operation.
    for step, seed in enumerate((143, 144)):
        torch.manual_seed(seed)
        q.normal_()
        iq[..., 0], iq[..., 1] = (1, -0.25) if step == 0 else (-0.25, 1)
        run()
        expected = out.clone()
        expected_topk = torch.arange(
            256 * step, 256 * step + 512, dtype=torch.int32, device=device
        )[None].expand(rows, -1)
        for plan in (*layer._plans.values(), *layer._index_plans.values()):
            for scratch in attention._scratch(plan):
                scratch.fill_(0xA5)
        out.fill_(float("nan"))
        layer.topk_indices_buffer[:rows].fill_(-1)
        graph.replay()
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            layer.topk_indices_buffer[:rows], expected_topk, rtol=0, atol=0
        )
    # Keep the collector alive for all replays, as the production graph owner does.
    del graph, resources


def test_grouped_projection_live_storage_survives_workspace_reuse(native_workspace):
    attention, manager, workspace = native_workspace
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(145)
    rows, groups, width, rank = 3, 2, 512, 64
    layer = SimpleNamespace(
        weight=torch.randn((groups * rank, width), device=device, dtype=torch.bfloat16)
        / 16
    )
    method = attention._GroupedLinearMethod(attention.B12xLinearMethod(), groups)
    method.process_weights_after_loading(layer)
    source = torch.randn((rows, groups, width), dtype=torch.bfloat16, device=device)
    positions = torch.zeros(rows, dtype=torch.int64, device=device)
    # Identity RoPE makes an independent torch matmul oracle straightforward.
    cs = torch.cat(
        (torch.ones((1, 32), device=device), torch.zeros((1, 32), device=device)),
        dim=-1,
    )
    (scratch,) = manager.get_simultaneous((source.shape, source.dtype))
    manager.lock()
    result = torch.empty((rows, groups * rank), device=device, dtype=torch.bfloat16)

    def run():
        inverse = attention._rotated(source, positions, cs, inverse=True)
        # A nested GEMM may reuse every arena byte while inverse is still live.
        scratch.fill_(float("nan"))
        local = method.apply(layer, inverse)
        scratch.zero_()
        result.copy_(local)

    run()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with workspace.collect_cuda_graph_capture_resources() as resources:
        with torch.cuda.graph(graph, stream=stream):
            run()
    for seed in (146, 147):
        torch.manual_seed(seed)
        source.normal_()
        expected = torch.cat(
            [
                (
                    source[:, g].float()
                    @ layer.weight[g * rank : (g + 1) * rank].float().T
                ).to(torch.bfloat16)
                for g in range(groups)
            ],
            dim=-1,
        )
        graph.replay()
        torch.testing.assert_close(result, expected, rtol=0.01, atol=0.01)
    del graph, resources
