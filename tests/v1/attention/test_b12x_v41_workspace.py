# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-kernel memory and graph-lifetime regressions for V4.1 attention."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CUDAGraphMode

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Native b12x attention requires CUDA"
)


@pytest.fixture
def native_workspace(monkeypatch):
    from vllm.platforms import current_platform

    capability = current_platform.get_device_capability()
    if capability is None or capability.major != 12:
        pytest.skip("Native b12x attention requires SM12x")
    import vllm.v1.worker.workspace as workspace
    from vllm.models.deepseek_v4_1 import attention

    manager = workspace.WorkspaceManager(
        torch.device("cuda", torch.accelerator.current_device_index())
    )
    monkeypatch.setattr(workspace, "_manager", manager)
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    monkeypatch.setattr(attention, "get_tensor_model_parallel_world_size", lambda: 1)
    with workspace.use_workspace_lane(0):
        yield attention, manager, workspace


def test_indexer_loads_complete_projections_on_tp_rank(native_workspace, monkeypatch):
    from vllm.distributed import parallel_state

    attention, _, _ = native_workspace
    monkeypatch.setattr(
        parallel_state,
        "get_tp_group",
        lambda: SimpleNamespace(rank_in_group=3, world_size=4),
    )
    monkeypatch.setattr(attention, "get_tensor_model_parallel_world_size", lambda: 4)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                index_n_heads=32,
                q_lora_rank=128,
                hidden_size=256,
            )
        ),
        quant_config=None,
    )
    indexer = attention.DeepseekV4Indexer(
        config,
        "model.layers.2.self_attn.indexer",
        owns_k=False,
        k_cache=None,
        ratio=1,
    )
    assert indexer.heads == 32
    for projection, shape in (
        (indexer.wq_b, (4096, 128)),
        (indexer.weights_proj, (32, 256)),
    ):
        assert isinstance(projection, attention.ReplicatedLinear)
        assert projection.weight.shape == shape
        source = torch.arange(projection.weight.numel(), dtype=torch.float32)
        source = source.reshape(shape).to(projection.weight)
        projection.weight.weight_loader(projection.weight, source)
        torch.testing.assert_close(projection.weight, source, rtol=0, atol=0)


def _layer(attention, layer_id=0):
    # Avoid checkpoint/model construction: exercise the real planning and
    # attention methods with the same TP4 head geometry and serving capacity.
    layer = attention.DeepseekV4Attention.__new__(attention.DeepseekV4Attention)
    torch.nn.Module.__init__(layer)
    layer.prefix = f"model.layers.{layer_id}.self_attn"
    layer.layer_id = layer_id
    layer.config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=64),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        speculative_config=SimpleNamespace(
            num_speculative_tokens=5, parallel_drafting=False
        ),
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=128),
    )
    layer.capacity = 4096
    layer.max_model_len = 1024
    layer.n_local_heads = 16
    layer.swa_width = layer.window_size = 128
    layer.compress_ratio = 1
    layer.is_draft = False
    layer.is_ced_decoder = False
    layer.is_index_source = True
    layer.kv_source_layer_id = layer.index_source_layer_id = layer_id
    layer.candidate_source_layer = 99
    layer.topk_indices_buffer = None
    layer.indexer = SimpleNamespace(
        heads=32, k_cache=SimpleNamespace(prefix=layer.prefix + ".indexer.k_cache")
    )
    layer.swa_cache_layer = SimpleNamespace(prefix=layer.prefix + ".swa_cache")
    layer.compressor = None
    layer._context = {layer.prefix: layer}
    layer._ready = False
    return layer


def test_prepare_memory_is_metadata_not_capacity_activations(native_workspace):
    attention, manager, _ = native_workspace
    device = torch.device("cuda", torch.accelerator.current_device_index())
    first = _layer(attention)
    first._prepare(device)
    manager.lock()
    before = torch.accelerator.memory_allocated(device)
    second = _layer(attention, 1)
    second._prepare(device)
    allocated = torch.accelerator.memory_allocated(device) - before
    c = second.INDEX_CHUNK
    metadata_bytes = 4 * (c * second._index_width + 1)
    persistent_topk_bytes = 4 * second.capacity * 512
    # A second layer must not own Q/O/inverse, indexer activations, or another
    # copy of either mode's planned scratch. Allow CUDA allocator rounding.
    assert allocated <= metadata_bytes + persistent_topk_bytes + 1024**2


def test_mhc_fixed_capacity_buckets_preserve_decode_policy(
    native_workspace, monkeypatch
):
    from vllm.models.deepseek_v4_1 import b12x_layers

    _, _, workspace = native_workspace
    observed = []
    original = b12x_layers._mhc_plan

    def record(device, capacity, hidden):
        plan = original(device, capacity, hidden)
        observed.append((capacity, plan.config.backend))
        return plan

    monkeypatch.setattr(b12x_layers, "_mhc_plan", record)
    device = torch.device("cuda", torch.accelerator.current_device_index())
    hidden = 5120
    fn = torch.randn((24, hidden * 4), device=device) / 64
    scale = torch.ones(3, device=device)
    bias = torch.zeros(24, device=device)
    norm = torch.ones(hidden, device=device, dtype=torch.bfloat16)
    for rows, capacity, backend in (
        (6, 64, "native"),
        (64, 64, "native"),
        (65, 4096, "tf32_tma"),
    ):
        residual = torch.randn((rows, 4, hidden), device=device, dtype=torch.bfloat16)
        pre = torch.full((rows, 4), 0.25, device=device)
        out = torch.empty_like(residual)
        y = torch.empty((rows, hidden), device=device, dtype=torch.bfloat16)
        post = torch.empty_like(pre)
        comb = torch.empty((rows, 4, 4), device=device)
        predicted = torch.empty_like(pre)
        with workspace.use_workspace_lane(0):
            b12x_layers._mhc_pre(
                residual,
                fn,
                scale,
                bias,
                norm,
                pre,
                out,
                y,
                post,
                comb,
                predicted,
                1e-20,
                1e-6,
                20,
                4096,
            )
        assert observed[-1] == (capacity, backend)
        assert torch.equal(out, residual)
        assert bool(torch.isfinite(y).all())


@pytest.mark.parametrize(
    "is_decode,rows,live_rows",
    [
        (True, 6, 6),
        (True, 36, 36),
        (True, 48, 6),
        (False, 65, 65),
        (False, 257, 257),
        (False, 1025, 1025),
    ],
)
def test_attention_shared_scratch_graph_replay(
    native_workspace, monkeypatch, is_decode, rows, live_rows
):
    attention, manager, workspace = native_workspace
    # A TP4 rank must score all replicated heads without obtaining a TP group.
    monkeypatch.setattr(attention, "get_tensor_model_parallel_world_size", lambda: 4)
    device = torch.device("cuda", torch.accelerator.current_device_index())
    torch.manual_seed(142)
    layer = _layer(attention)
    layer.max_model_len = 4096
    if rows == 36:
        layer.config.speculative_config.parallel_drafting = True
        layer.config.compilation_config.max_cudagraph_capture_size = 32
    layer._prepare(device)
    if rows == 36:
        assert layer._plans["decode"].caps.max_q_rows == 4 * (1 + 2 * 5)
    length = 2048
    positions = torch.full((rows,), -1, dtype=torch.int64, device=device)
    positions[:live_rows] = torch.arange(
        length - live_rows, length, dtype=torch.int64, device=device
    )
    starts = torch.tensor([0, live_rows], dtype=torch.int32, device=device)
    reqs = torch.full((rows,), -1, dtype=torch.int32, device=device)
    reqs[:live_rows] = 0
    visible = torch.clamp(positions + 1, min=0).int()

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
            max_seq_len=length,
            decoder=None,
            # Encoder attention must ignore decoder-only replay metadata.
            swa_replay_start=torch.full(
                (1,), length + 1, dtype=torch.int64, device=device
            ),
        )

    context = SimpleNamespace(
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        attn_metadata={
            layer.swa_cache_layer.prefix: metadata(32),
            layer.prefix: metadata(64),
            layer.indexer.k_cache.prefix: metadata(64),
        },
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
    iq = torch.zeros((rows, 32, 128), dtype=torch.bfloat16, device=device)
    iq[..., 0], iq[..., 1] = 1, -0.25
    weights = (torch.rand((rows, 32), dtype=torch.bfloat16, device=device) + 1) / 64
    packed = torch.empty((rows, 32, 64), dtype=torch.uint8, device=device)
    scales = torch.empty((rows, 32, 4), dtype=torch.uint8, device=device)
    out = torch.empty_like(q)

    def run():
        attention.dsa_indexer.quantize_q_mxfp4(iq, q_mxfp4=packed, q_scales=scales)
        layer.forward_mqa(
            q, None, positions, out, index_query=(packed, scales, weights)
        )

    run()
    batched_output = out.clone()
    # Index-score chunking must not change the full attention output. Reuse
    # the plan and operands while exercising multiple scoring batches.
    chunk_attribute = "DECODE_CHUNK" if is_decode else "INDEX_CHUNK"
    setattr(layer, chunk_attribute, 4 if is_decode else 64)
    run()
    torch.testing.assert_close(out, batched_output, rtol=0, atol=0)
    delattr(layer, chunk_attribute)
    manager.lock()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with (
        workspace.collect_cuda_graph_capture_resources() as resources,
        torch.cuda.graph(graph, stream=stream),
    ):
        run()
    # Changed queries exercise both selection and attention on replay, after
    # all plan storage has been reused by another operation.
    for step, seed in enumerate((143, 144)):
        torch.manual_seed(seed)
        q.normal_()
        iq[..., 0], iq[..., 1] = (1, -0.25) if step == 0 else (-0.25, 1)
        run()
        expected = out.clone()
        expected_topk = (
            torch.arange(
                256 * step, 256 * step + 512, dtype=torch.int32, device=device
            )[None]
            .expand(rows, -1)
            .clone()
        )
        expected_topk[live_rows:] = -1
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
    device = torch.device("cuda", torch.accelerator.current_device_index())
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
    with (
        workspace.collect_cuda_graph_capture_resources() as resources,
        torch.cuda.graph(graph, stream=stream),
    ):
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


@pytest.mark.parametrize("width,k", [(128, 512), (1024, 4096)])
def test_grouped_fp8_weight_only_prefill_preserves_bf16_contract(
    native_workspace, monkeypatch, width, k
):
    from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution
    from b12x.gemm import bf16_gemv

    attention, manager, workspace = native_workspace
    monkeypatch.setattr(
        attention,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=4096)
        ),
    )
    torch.manual_seed(41003)
    groups = 2
    layer = torch.nn.Module()
    value = torch.randn((groups * width, k), device="cuda").to(torch.float8_e4m3fn)
    exponent = torch.randint(121, 128, (groups * width // 32, k // 32), device="cuda")
    scale = exponent.byte().view(torch.float8_e8m0fnu)
    layer.weight = torch.nn.Parameter(value, requires_grad=False)
    layer.weight_scale_inv = torch.nn.Parameter(scale, requires_grad=False)
    method = attention._GroupedLinearMethod(
        attention.B12xFP8LinearMethod(SimpleNamespace(weight_block_size=[32, 32])),
        groups,
    )
    method.process_weights_after_loading(layer)
    dense = (
        value.float()
        * torch.exp2(exponent.float() - 127)
        .repeat_interleave(32, 0)
        .repeat_interleave(32, 1)
    ).bfloat16()
    torch.testing.assert_close(layer.weight, dense, rtol=0, atol=0)
    source = torch.randn((1025, groups, k), device="cuda").bfloat16()
    method.apply(layer, source, is_prefill=True)
    manager.lock()
    freeze_kernel_resolution("V4.1 weight-only prefill with BF16 activations")
    try:
        for rows in (1, 6, 64, 257, 1025):
            x = source[:rows]
            expected = torch.cat(
                [
                    x[:, g].float() @ dense[g * width : (g + 1) * width].float().T
                    for g in range(groups)
                ],
                dim=1,
            )
            gemv = torch.cat(
                [
                    bf16_gemv.mm(x[:, g], dense[g * width : (g + 1) * width])
                    for g in range(groups)
                ],
                dim=1,
            )
            actual = method.apply(layer, x, is_prefill=True)
            actual_rmse = (actual.float() - expected).square().mean().sqrt()
            gemv_rmse = (gemv.float() - expected).square().mean().sqrt()
            assert actual_rmse <= gemv_rmse * 1.01 + 1e-7
            torch.testing.assert_close(actual.float(), expected, rtol=0.01, atol=0.1)
            torch.testing.assert_close(method.apply(layer, x), gemv, rtol=0, atol=0)
            graph = torch.cuda.CUDAGraph()
            with (
                workspace.collect_cuda_graph_capture_resources() as resources,
                torch.cuda.graph(graph),
            ):
                result = method.apply(layer, x, is_prefill=True)
            source.normal_()
            expected = method.apply(layer, x, is_prefill=True)
            result.fill_(float("nan"))
            graph.replay()
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
            del graph, resources
    finally:
        unfreeze_kernel_resolution()


@pytest.mark.parametrize("ratio", [1, 2])
def test_index_visibility_crosses_old_capacity_cutoff(native_workspace, ratio):
    """Source and reindex must retain winners beyond the former 64-state cap."""
    attention, _, _ = native_workspace
    device = torch.device("cuda", torch.accelerator.current_device_index())
    layers = [_layer(attention, index) for index in (2, 20, 24)]
    for layer in layers:
        layer.config.cache_config.block_size = 256
        layer.max_model_len = 1 << 20
        layer.compress_ratio = ratio
        layer.candidate_source_layer = 20
        layer._prepare(device)
    page = 256 // ratio
    old_cutoff = (1 << 20) // 256 * 64
    positions = torch.arange(
        old_cutoff - 256, old_cutoff + 256, dtype=torch.int64, device=device
    )
    first_page = (old_cutoff - 256) // page
    page_count = 512 // page
    stride = 227840
    high_pid = 2**31 // stride + 1
    storage = torch.empty(
        (high_pid + page_count, stride), dtype=torch.uint8, device=device
    )
    pool = storage[:, : attention.dsa_indexer.index_mxfp4_page_bytes(page)]
    physical_pages = torch.arange(
        high_pid, high_pid + page_count, dtype=torch.int32, device=device
    )
    table = torch.full((1, (1 << 20) // 256), -1, dtype=torch.int32, device=device)
    table[:, first_page : first_page + page_count] = physical_pages
    slots = (
        physical_pages[positions // page - first_page].long() * page + positions % page
    )
    keys = torch.zeros((512, 128), dtype=torch.bfloat16, device=device)
    keys[:, 0] = 1
    attention.dsa_indexer.quantize_write_index_k_mxfp4(
        keys, index_k_cache=pool, slot_mapping=slots, page_size=page
    )
    heads = layers[0].indexer.heads
    q = torch.zeros((1, heads, 128), dtype=torch.bfloat16, device=device)
    q[..., 0] = 1
    packed = torch.empty((1, heads, 64), dtype=torch.uint8, device=device)
    scales = torch.empty((1, heads, 4), dtype=torch.uint8, device=device)
    attention.dsa_indexer.quantize_q_mxfp4(q, q_mxfp4=packed, q_scales=scales)
    lengths = torch.tensor([old_cutoff + 256], dtype=torch.int32, device=device)
    weights = torch.ones((1, heads), dtype=torch.bfloat16, device=device) / 32
    for mode in ("decode", "prefill"):
        for layer in layers:
            candidate_args = {}
            if layer.layer_id == 20:
                candidate_args = dict(
                    candidate_output=layer._candidates[:1],
                    candidate_output_lengths=layer._candidate_lens[:1],
                )
            elif layer.layer_id == 24:
                candidate_args = dict(
                    candidate_indices=layers[1]._candidates[:1],
                    candidate_lengths=layers[1]._candidate_lens[:1],
                )
            plan = layer._index_plans[mode]
            binding = attention.dsa_indexer.bind(
                plan,
                scratch=attention._scratch(plan),
                q_mxfp4=packed,
                q_scales=scales,
                query_weights=weights,
                index_k_cache=pool,
                page_table=table,
                cache_lengths=lengths,
                active_width=layer._active,
                output_indices=layer.topk_indices_buffer[:1],
                **candidate_args,
            )
            attention.dsa_indexer.run(binding)
            torch.testing.assert_close(
                layer.topk_indices_buffer[:1], positions.int()[None], atol=0, rtol=0
            )


@torch.inference_mode()
def test_ced_global_preparation_preserves_full_row_cache_bytes(
    native_workspace, monkeypatch
):
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear
    from vllm.models.deepseek_v4_1 import b12x_layers
    from vllm.models.deepseek_v4_1.compressor import DeepseekCompressor

    attention, _, _ = native_workspace
    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(b12x_layers, "_capacity", lambda: 256)
    torch.manual_seed(712)
    device = torch.device("cuda")
    layer = _layer(attention, 20)
    layer.capacity = 256
    layer.is_ced_decoder = True
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=256, max_num_seqs=1),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(rms_norm_eps=1e-6)),
    )
    layer.compressor = DeepseekCompressor(config, 1, 128, 512).to(device)
    layer.compressor.fused_wkv_wgate.weight.normal_(0, 0.1)
    wk = linear.ReplicatedLinear(
        512, 128, bias=False, return_bias=False, params_dtype=torch.bfloat16
    ).to(device)
    wk.quant_method = b12x_layers.B12xLinearMethod()
    wk.weight.normal_(0, 0.1)
    layer.indexer.wk = wk
    layer.indexer.k_norm = b12x_layers.B12xRMSNorm(128).to(device)
    angles = torch.randn((256, 32), device=device)
    layer.rotary_emb = SimpleNamespace(
        cos_sin_cache=torch.cat((angles.cos(), angles.sin()), dim=-1)
    )
    layer._prepare(device)
    layer.kv_cache = torch.zeros(
        (
            5,
            attention.mla.page_nbytes(
                64, cache_kind="indexed", cache_format="deepseek_v41"
            ),
        ),
        dtype=torch.uint8,
        device=device,
    )
    layer.indexer.k_cache.kv_cache = torch.zeros(
        (5, attention.dsa_indexer.index_mxfp4_page_bytes(64)),
        dtype=torch.uint8,
        device=device,
    )
    positions = torch.arange(256, device=device, dtype=torch.int64)
    positions[192:] = -1
    slots = torch.arange(256, device=device, dtype=torch.int64) + 64
    slots[192:] = -1
    full = SimpleNamespace(
        query_start_loc=torch.tensor([0, 192], device=device, dtype=torch.int32),
        request_positions=torch.zeros(1, device=device, dtype=torch.int64),
        live_counts=torch.tensor([192, 1], device=device, dtype=torch.int32),
        slot_mapping=slots,
        decoder=None,
    )
    context = SimpleNamespace(
        attn_metadata={layer.prefix: full, layer.indexer.k_cache.prefix: full},
        no_compile_layers={layer.prefix: layer},
    )
    monkeypatch.setattr(attention, "get_forward_context", lambda: context)
    hidden = torch.randn((256, 128), device=device, dtype=torch.bfloat16)
    # Original full-forward computation, with real compressor/projection/writers.
    latent, emitted_slots = layer.compressor(hidden, full)
    key = layer.indexer.k_norm(layer.indexer.wk(latent))
    key = attention._rotated(key, positions, layer.rotary_emb.cos_sin_cache, ratio=1)
    attention.dsa_indexer.quantize_write_index_k_mxfp4(
        key,
        index_k_cache=layer.indexer.k_cache.kv_cache,
        slot_mapping=slots,
        page_size=64,
    )
    latent = attention._rotated(
        latent, positions, layer.rotary_emb.cos_sin_cache, ratio=1
    )
    attention.mla.write_cache(
        latent,
        layer.kv_cache,
        emitted_slots,
        page_size=64,
        cache_kind="indexed",
        cache_format="deepseek_v41",
    )
    expected = (layer.kv_cache.clone(), layer.indexer.k_cache.kv_cache.clone())
    # The decoder view intentionally lacks early rows. Global preparation must
    # neither read this view nor leave the discarded encoder prefix unwritten.
    full.decoder = SimpleNamespace(
        query_start_loc=torch.tensor([0, 128], device=device, dtype=torch.int32),
        request_positions=torch.tensor([64], device=device, dtype=torch.int64),
        live_counts=torch.tensor([128, 1], device=device, dtype=torch.int32),
        slot_mapping=slots[64:192],
    )
    layer.kv_cache.zero_()
    layer.indexer.k_cache.kv_cache.zero_()
    layer.prepare_global_kv(positions, hidden)
    torch.testing.assert_close(layer.kv_cache, expected[0], rtol=0, atol=0)
    torch.testing.assert_close(
        layer.indexer.k_cache.kv_cache, expected[1], rtol=0, atol=0
    )


@torch.inference_mode()
def test_ced_compact_attention_bounded_oracle_and_frozen_replay(
    native_workspace, monkeypatch
):
    from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution
    from b12x.attention._shared.mla.compressed_reference import (
        unpack_deepseek_v41_cache_reference,
    )

    attention, manager, workspace = native_workspace
    torch.manual_seed(713)
    device = torch.device("cuda")
    layer = _layer(attention, 20)
    layer.is_ced_decoder = True
    layer._prepare(device)
    rows, length, boundary = 128, 256, 128
    positions = torch.arange(boundary, length, device=device, dtype=torch.int64)
    reqs = torch.zeros(rows, device=device, dtype=torch.int32)
    starts = torch.tensor([0, rows], device=device, dtype=torch.int32)
    visible = (positions + 1).int()
    replay_start = torch.tensor([boundary], device=device, dtype=torch.int64)

    def metadata(page):
        compact = SimpleNamespace(
            positions=positions,
            req_id_per_token=reqs,
            block_table=torch.arange(
                1, length // page + 1, device=device, dtype=torch.int32
            )[None],
            query_start_loc=starts,
            request_positions=replay_start,
            cache_lengths=visible,
            is_decode=False,
            max_seq_len=length,
            swa_replay_start=replay_start,
            decoder=None,
        )
        # Full input has different query anchors and must not be used here.
        return SimpleNamespace(
            positions=torch.arange(length, device=device, dtype=torch.int64),
            decoder=compact,
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
        cache = torch.zeros(
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
            swa_values = unpack_deepseek_v41_cache_reference(
                cache, page_size=page, cache_kind=kind
            )[page : page + length].clone()
            # NaN FP8 payloads in every old SWA row: reading below the
            # replay boundary cannot accidentally look like a valid zero page.
            cache[1 : 1 + boundary // page].fill_(0x7F)
        else:
            layer.kv_cache = cache
            main_values = unpack_deepseek_v41_cache_reference(
                cache, page_size=page, cache_kind=kind
            )[page : page + length].clone()
    layer.indexer.k_cache.kv_cache = torch.zeros(
        (length // 64 + 1, attention.dsa_indexer.index_mxfp4_page_bytes(64)),
        device=device,
        dtype=torch.uint8,
    )
    keys = torch.zeros((length, 128), device=device, dtype=torch.bfloat16)
    # Future keys score highest. Any missing causal mask is observable.
    keys[:, 0] = torch.arange(1, length + 1, device=device)
    attention.dsa_indexer.quantize_write_index_k_mxfp4(
        keys,
        index_k_cache=layer.indexer.k_cache.kv_cache,
        slot_mapping=torch.arange(length, device=device) + 64,
        page_size=64,
    )
    q = torch.randn((rows, 16, 512), device=device, dtype=torch.bfloat16)
    heads = layer.indexer.heads
    iq = torch.zeros((rows, heads, 128), device=device, dtype=torch.bfloat16)
    iq[..., 0] = 1
    packed = torch.empty((rows, heads, 64), device=device, dtype=torch.uint8)
    scales = torch.empty((rows, heads, 4), device=device, dtype=torch.uint8)
    weights = torch.full((rows, heads), 1 / 64, device=device, dtype=torch.bfloat16)
    layer.attn_sink = torch.zeros(16, device=device)
    out = torch.empty_like(q)

    def run():
        attention.dsa_indexer.quantize_q_mxfp4(iq, q_mxfp4=packed, q_scales=scales)
        layer.forward_mqa(
            q, None, positions, out, index_query=(packed, scales, weights)
        )

    def oracle(live):
        values = torch.cat((swa_values[boundary:], main_values))
        logical = torch.cat(
            (
                torch.arange(boundary, length, device=device),
                torch.arange(length, device=device),
            )
        )
        logits = torch.einsum("rhd,kd->rhk", q[:live].float(), values) * 512**-0.5
        logits.masked_fill_(
            logical[None, None] > positions[:live, None, None], -torch.inf
        )
        logits = torch.cat(
            (logits, layer.attn_sink[None, :, None].expand(live, -1, -1)), -1
        )
        return torch.einsum("rhk,kd->rhd", logits.softmax(-1)[..., :-1], values)

    run()
    manager.lock()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    freeze_kernel_resolution("CED attention must replay using capacity-planned kernels")
    try:
        with (
            workspace.collect_cuda_graph_capture_resources() as resources,
            torch.cuda.graph(graph, stream=stream),
        ):
            run()
        for live in (128, 65, 17):
            positions.fill_(-1)
            positions[:live] = torch.arange(boundary, boundary + live, device=device)
            reqs.fill_(-1)
            reqs[:live] = 0
            visible.copy_((positions + 1).clamp_min(0).int())
            starts[1] = live
            q.normal_()
            expected = oracle(live)
            # Eager under freeze also catches live-count specialization leaks.
            run()
            torch.testing.assert_close(
                out[:live].float(), expected, rtol=0.04, atol=0.025
            )
            out.fill_(float("nan"))
            graph.replay()
            torch.testing.assert_close(
                out[:live].float(), expected, rtol=0.04, atol=0.025
            )
            torch.testing.assert_close(
                out[live:], torch.zeros_like(out[live:]), rtol=0, atol=0
            )
            selected = layer.topk_indices_buffer[:live]
            assert torch.all((selected < 0) | (selected <= positions[:live, None]))
    finally:
        unfreeze_kernel_resolution()
    del graph, resources


@torch.inference_mode()
def test_ced_replay_window_high_page_stride_and_invalid_rows(
    native_workspace, monkeypatch
):
    from b12x.attention._shared.mla.compressed_reference import (
        unpack_deepseek_v41_cache_reference,
    )

    attention, _, _ = native_workspace
    device = torch.device("cuda")
    layer = _layer(attention, 20)
    layer.is_ced_decoder = True
    layer.compress_ratio = 0
    layer.indexer = None
    layer._prepare(device)
    stride = 227840
    high_pid = 2**31 // stride + 1
    storage = torch.empty((high_pid + 1, stride), device=device, dtype=torch.uint8)
    cache = storage[
        :,
        : attention.mla.page_nbytes(32, cache_kind="swa", cache_format="deepseek_v41"),
    ]
    kv = torch.randn((1, 512), device=device, dtype=torch.bfloat16)
    attention.mla.write_cache(
        kv,
        cache,
        torch.tensor([high_pid * 32], device=device),
        page_size=32,
        cache_kind="swa",
        cache_format="deepseek_v41",
    )
    layer.swa_cache_layer.kv_cache = cache
    positions = torch.tensor([128, -1, 128], device=device, dtype=torch.int64)
    metadata = SimpleNamespace(
        positions=positions,
        req_id_per_token=torch.tensor([0, 0, -1], device=device, dtype=torch.int32),
        block_table=torch.tensor(
            [[-1, -1, -1, -1, high_pid]], device=device, dtype=torch.int32
        ),
        query_start_loc=torch.tensor([0, 1], device=device, dtype=torch.int32),
        request_positions=positions[:1],
        cache_lengths=torch.tensor([129, 99, 99], device=device, dtype=torch.int32),
        is_decode=False,
        max_seq_len=129,
        decoder=None,
        swa_replay_start=positions[:1],
    )
    monkeypatch.setattr(
        attention,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={layer.swa_cache_layer.prefix: metadata}),
    )
    q = torch.randn((3, 16, 512), device=device, dtype=torch.bfloat16)
    out = torch.empty_like(q)
    layer.attn_sink = torch.zeros(16, device=device)
    layer.forward_mqa(q, None, positions, out)
    value = unpack_deepseek_v41_cache_reference(
        cache[high_pid : high_pid + 1].contiguous(), page_size=32, cache_kind="swa"
    )[0]
    score = torch.einsum("hd,d->h", q[0].float(), value) * 512**-0.5
    expected = score.sigmoid()[:, None] * value
    torch.testing.assert_close(out[0].float(), expected, rtol=0.04, atol=0.025)
    torch.testing.assert_close(out[1:], torch.zeros_like(out[1:]), rtol=0, atol=0)


@torch.inference_mode()
def test_global_preparation_dependency_survives_functionalization(
    native_workspace, monkeypatch
):
    attention, _, _ = native_workspace
    source = torch.arange(16, dtype=torch.float32, device="cuda").view(4, 4)
    positions = torch.arange(4, device="cuda", dtype=torch.int64)
    backing = torch.zeros((8, 4), device="cuda")

    class CacheConsumer(torch.nn.Module):
        prepare_global_kv = attention.DeepseekV4Attention.prepare_global_kv
        forward = attention.DeepseekV4Attention.forward

        def __init__(self):
            super().__init__()
            self.prefix = "functionalized_ced"
            self.compressor = SimpleNamespace(state_cache=None)
            self.kv_cache = backing[:4]
            self.indexer = SimpleNamespace(
                k_cache=SimpleNamespace(kv_cache=backing[4:])
            )

        def _prepare_global_kv(self, positions, hidden):
            self.kv_cache.copy_(hidden * 2)
            self.indexer.k_cache.kv_cache.copy_(hidden * 3)

        def _forward(self, positions, hidden):
            return self.kv_cache + self.indexer.k_cache.kv_cache

    layer = CacheConsumer()
    context = SimpleNamespace(no_compile_layers={layer.prefix: layer})
    monkeypatch.setattr(attention, "get_forward_context", lambda: context)

    def run(hidden):
        ready = layer.prepare_global_kv(positions, hidden)
        return layer(positions, hidden, global_kv_ready=ready)

    compiled = torch.compile(run, backend="aot_eager", fullgraph=True)
    for factor in (1, -2):
        hidden = source * factor
        output = compiled(hidden)
        torch.testing.assert_close(output, hidden * 5)
        # Preparation must remain visible to later consumers as well; mutable
        # alias-list functionalization used to copy stale clones back here.
        torch.testing.assert_close(backing, torch.cat((hidden * 2, hidden * 3)))


@torch.inference_mode()
def test_metadata_refresh_preserves_padded_graph_domain_and_addresses():
    from vllm.models.deepseek_v4_1.sparse_mla import DeepseekV41B12xMetadataBuilder

    device = torch.device("cuda")
    builder = DeepseekV41B12xMetadataBuilder.__new__(DeepseekV41B12xMetadataBuilder)
    builder.tokens, builder.requests = 4096, 4
    builder.page, builder.ratio, builder.circular = 128, 1, False
    builder.reorder_batch_threshold = 6
    builder.starts = torch.empty(5, dtype=torch.int32, device=device)
    builder.request_positions = torch.empty(4, dtype=torch.int64, device=device)
    builder.counts = torch.empty(2, dtype=torch.int32, device=device)
    for name, dtype in (
        ("positions", torch.int64),
        ("reqs", torch.int32),
        ("slots", torch.int64),
        ("lengths", torch.int32),
    ):
        setattr(builder, name, torch.empty(builder.tokens, dtype=dtype, device=device))
    addresses = [
        getattr(builder, name).data_ptr()
        for name in ("positions", "reqs", "slots", "lengths")
    ]
    base_page = 2**31 // builder.page + 17
    table = torch.zeros((2, 16), dtype=torch.int32, device=device)
    table[0] = torch.arange(base_page, base_page + 16, dtype=torch.int32, device=device)
    for live, padded in ((257, 320), (5, 10), (129, 160), (0, 5)):
        seq = torch.tensor([1024, 0], dtype=torch.int32, device=device)
        slots = torch.full((padded,), -1, dtype=torch.int64, device=device)
        slots[:live] = 1
        common = SimpleNamespace(
            block_table_tensor=table,
            query_start_loc=torch.tensor(
                [0, live, live], dtype=torch.int32, device=device
            ),
            seq_lens=seq,
            slot_mapping=slots,
            num_reqs=2,
            num_actual_tokens=padded,
            max_query_len=live,
            max_seq_len=2048,
        )
        builder.build(0, common)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            metadata = builder.build(0, common)
        seq[0] += 7
        graph.replay()
        torch.cuda.synchronize()
        expected = torch.arange(1031 - live, 1031, dtype=torch.int64, device=device)
        torch.testing.assert_close(metadata.positions[:live], expected, rtol=0, atol=0)
        torch.testing.assert_close(
            metadata.slot_mapping[:live],
            base_page * builder.page + expected,
            rtol=0,
            atol=0,
        )
        assert bool((metadata.req_id_per_token[:live] == 0).all())
        for tensor in (
            metadata.positions,
            metadata.req_id_per_token,
            metadata.slot_mapping,
        ):
            assert bool((tensor[live:padded] == -1).all())
        assert bool((metadata.cache_lengths[live:padded] == 0).all())
        assert metadata.live_counts.tolist() == [padded, 2]
        assert addresses == [
            getattr(builder, name).data_ptr()
            for name in ("positions", "reqs", "slots", "lengths")
        ]
