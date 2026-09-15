# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepared native attention resources and inverse-RoPE WO integration."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch


def test_dcp_index_cache_spec_is_opt_in_sharded():
    from vllm.models.deepseek_v4_1.attention import _Cache

    config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=4),
        cache_config=SimpleNamespace(block_size=256, swa_block_size=128),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    cache = _Cache(
        config,
        "model.layers.2.attn.indexer.k_cache",
        kind="index",
        ratio=2,
        dcp_replicated=False,
    )
    spec = cache.get_kv_cache_spec(config)
    assert spec.dcp_replicated is False
    assert spec.num_states == 128
    assert spec.tokens_per_state == 2


def test_dcp_local_length_matches_striped_ownership():
    from vllm.models.deepseek_v4_1.attention import _dcp_local_length

    for length in (0, 1, 63, 64, 65, 255, 256, 257, 1025):
        expected = [
            sum(position // 64 % 4 == rank for position in range(length))
            for rank in range(4)
        ]
        actual = [_dcp_local_length(length, 4, rank, 64) for rank in range(4)]
        assert actual == expected


def test_dcp_kv_replica_declares_one_shared_output_before_materialization(monkeypatch):
    """Catch wrapper contract errors before loading weights or creating IPC."""
    from b12x.comm import pcie
    from b12x.preparation import FrozenMapping
    from b12x.preparation import device as preparation_device

    from vllm.models.deepseek_v4_1 import dcp

    runtime = pcie.PagedKvReplica.__new__(pcie.PagedKvReplica)
    runtime.device, runtime.rank, runtime.world_size = torch.device("cuda", 0), 0, 4
    runtime.max_requests, runtime.max_tokens, runtime.local_capacity = 4, 1536, 384
    runtime.slab_bytes = 4 * 384 * 288 + 4096
    monkeypatch.setattr(
        pcie.PagedKvReplica, "from_process_group", lambda **kwargs: runtime
    )
    monkeypatch.setattr(dcp.DCPKVReplica, "_instances", {})
    group = SimpleNamespace(unique_name="test-replica", cpu_group=None, ranks=range(4))
    monkeypatch.setattr(dcp, "get_dcp_group", lambda: group)
    monkeypatch.setattr(
        preparation_device, "detect_device",
        lambda device: SimpleNamespace(identity=None),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    attention = SimpleNamespace(
        attn_sink=SimpleNamespace(device=runtime.device),
        max_model_len=3072, _main_page=128, dcp_stripe=128, _helpers=object(),
        config=SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_seqs=4),
            parallel_config=SimpleNamespace(use_ubatching=False),
        ),
    )
    replica = dcp.DCPKVReplica.get(attention)
    assert replica.output is None
    assert replica.plan.invocation["vllm_prefill_shape"] == (49, 128 * 288)
    bad_query = replace(replica.plan.query, call=FrozenMapping({
        "page_size": 256, "stripe": 256, "ratio": 1, "max_tokens": 1536,
    }))
    with pytest.raises(ValueError, match="owner capacity"):
        pcie.plan(bad_query, runtime=runtime)
    memory = replica.plan.memory_requirements()
    assert sum(item.required_nbytes for item in memory.persistent) == (
        runtime.slab_bytes + 49 * 128 * 288
    )
    assert sum(item.resident_nbytes for item in memory.persistent) == runtime.slab_bytes
    output = replica.prepare_output(torch.device("cpu"))
    assert replica.prepare_output(torch.device("cpu")) is output
    other = SimpleNamespace(**{**vars(attention), "_helpers": object()})
    assert dcp.DCPKVReplica.get(other) is replica
    assert replica.preparation_units(other._helpers) == ()
    memory = replica.plan.memory_requirements()
    assert sum(item.resident_nbytes for item in memory.persistent) == sum(
        item.required_nbytes for item in memory.persistent
    )


def test_dcp_kv_replica_binds_fixed_tensors_once_before_live_launch(monkeypatch):
    from vllm.models.deepseek_v4_1 import dcp

    calls = []

    class Runtime:
        def bind(self, *args, **kwargs):
            calls.append(("bind", args, kwargs))
            count = sum(call[0] == "bind" for call in calls)
            return f"binding-{count}"

        def replicate(self, *args, **kwargs):
            calls.append(("replicate", args, kwargs))

    replica = dcp.DCPKVReplica.__new__(dcp.DCPKVReplica)
    replica.runtime = Runtime()
    replica.plan = object()
    replica.output = torch.empty(1)
    replica.bindings = {}
    attention = SimpleNamespace(prefix="layer.1", kv_cache=torch.empty(1))
    metadata = SimpleNamespace(
        block_table=torch.empty(1),
        request_positions=torch.empty(1),
        query_start_loc=torch.empty(1),
        num_reqs=1,
        max_seq_len=7,
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    replica.replicate(attention, metadata)
    metadata.num_reqs, metadata.max_seq_len = 2, 15
    replica.replicate(attention, metadata)
    other = SimpleNamespace(prefix="layer.2", kv_cache=torch.empty(1))
    replica.replicate(other, metadata)

    assert [call[0] for call in calls] == [
        "bind",
        "replicate",
        "replicate",
        "bind",
        "replicate",
    ]
    assert calls[0][1] == (
        attention.kv_cache,
        metadata.block_table,
        metadata.request_positions,
        metadata.query_start_loc,
        replica.output,
    )
    assert calls[1][1] == ("binding-1",)
    assert calls[1][2] == {"plan": replica.plan, "requests": 1, "max_tokens": 4}
    assert calls[2][1] == ("binding-1",)
    assert calls[2][2] == {"plan": replica.plan, "requests": 2, "max_tokens": 8}
    assert calls[3][1][0] is other.kv_cache
    assert calls[4][1] == ("binding-2",)


def test_shared_dcp_channel_exposes_one_unit_in_each_preparation_stage(monkeypatch):
    """Layer reuse must not produce conflicting collective request names."""
    from vllm.models.deepseek_v4_1 import dcp

    monkeypatch.setattr(dcp.DCPExchange, "_instances", {})
    monkeypatch.setattr(
        dcp, "get_dcp_group", lambda: SimpleNamespace(unique_name="test-dcp")
    )
    unit = object()

    def init(self, group, heads, device):
        self.unit = unit
        self._preparation_owner = None

    monkeypatch.setattr(dcp.DCPExchange, "__init__", init)
    first = dcp.DCPExchange.get(16, torch.device("cuda", 0))
    helpers = [object() for _ in range(40)]
    for _stage in ("weights", "state", "weights"):
        collected = []
        for helper in helpers:
            exchange = dcp.DCPExchange.get(16, torch.device("cuda", 0))
            assert exchange is first
            collected.extend(exchange.preparation_units(helper))
        assert collected == [unit]


def test_dcp_helpers_require_loaded_channel_without_allocating_cuda_memory():
    """Declaration cannot initialize an IPC channel behind the planner."""
    if not torch.cuda.is_available():
        pytest.skip("allocation guard requires CUDA")
    from vllm.models.deepseek_v4_1 import attention
    from vllm.utils.b12x import PreparationResourceUnavailableError

    module = SimpleNamespace(
        dcp_active=True,
        rotary_emb=SimpleNamespace(cos_sin_cache=torch.empty(1, device="cuda")),
    )
    allocated = torch.cuda.memory_allocated()
    with pytest.raises(
        PreparationResourceUnavailableError, match="after weight loading"
    ):
        attention._AttentionHelpers(module).get_b12x_preparation_units(module, None)
    assert torch.cuda.memory_allocated() == allocated


@pytest.mark.parametrize("groups,heads_per_group,rank,hidden", [(1, 1, 128, 128), (2, 8, 1024, 5120)])
@pytest.mark.parametrize("dcp_size", [1, 4])
@torch.no_grad()
def test_wo_preparation_exact_rows_owns_output_and_replays(
    monkeypatch, workspace_init, groups, heads_per_group, rank, hidden, dcp_size,
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("native b12x WO projection requires SM12x")
    from b12x.gemm import wo_projection as wo
    from b12x.gemm._shared.wo_mxfp8 import quantize_wo_projection_weights_mxfp8_torch
    from b12x.preparation import PreparationSession
    from vllm.models.deepseek_v4_1 import attention
    from vllm.utils.b12x import B12xWorkload, PreparationResourceUnavailableError
    from vllm.v1.worker.workspace import current_workspace_manager

    torch.manual_seed(411)
    device = torch.device("cuda", torch.cuda.current_device())
    module = attention.DeepseekV4Attention.__new__(attention.DeepseekV4Attention)
    torch.nn.Module.__init__(module)
    module.prefix = "model.layers.0.attn"
    module.n_local_groups, module.n_local_heads = groups, groups * heads_per_group
    module.head_dim, module.rope_head_dim = 512, 64
    module._wo_plans = {}
    angles = torch.randn(32, 32, device=device)
    table = torch.cat((angles.cos(), angles.sin()), dim=-1).bfloat16()
    module.rotary_emb = SimpleNamespace(cos_sin_cache=table)
    group_width = heads_per_group * 512
    module._wo_projection_weights = quantize_wo_projection_weights_mxfp8_torch(
        torch.randn(groups, rank, group_width, device=device, dtype=torch.bfloat16) / group_width**0.5,
        torch.randn(hidden, groups * rank, device=device, dtype=torch.bfloat16) / (groups * rank)**0.5,
    )
    monkeypatch.setattr(attention, "get_tensor_model_parallel_world_size", lambda: 1)
    compacted = groups == 2
    capacity = 4096 if compacted else 24
    draft_counts = {*range(7, 57, 7), *range(9, 73, 9)} if compacted else set()
    declared_counts = tuple(sorted({1, 4, 8, 24, capacity, *draft_counts}))
    counts = tuple(sorted({*declared_counts, *range(128, 1025, 128)})) if compacted else declared_counts
    workload = B12xWorkload(
        stage="weights", token_counts=declared_counts, fixed_token_counts=(1, 4, 8),
        output_dtype=torch.bfloat16, max_tokens=capacity, max_seqs=8, max_model_len=4096,
    )
    module.capacity, module.max_model_len, module.swa_width = capacity, 4096, 128
    module.dcp_size = dcp_size
    module.is_ced_decoder = compacted
    module.compress_ratio = 0
    module.indexer = module.compressor = None
    module.is_index_source = False
    module.topk_indices_buffer = None
    module._helper_plans = {}
    module._ready = False
    module.swa_cache_layer = SimpleNamespace(
        block_size=256, kv_cache=torch.empty(0, dtype=torch.uint8, device=device),
    )
    module.config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=256), speculative_config=None,
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=24),
    )
    units = attention._AttentionHelpers(module).get_b12x_preparation_units(module, workload)
    assert all(unit.stage == "weights" for unit in units)
    assert tuple(key for key in module._wo_plans if key != "prefill") == counts
    source = torch.randn(capacity, module.n_local_heads, 512, device=device, dtype=torch.bfloat16) / 8
    positions = torch.arange(capacity, device=device, dtype=torch.int64).remainder_(table.shape[0])
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare(tuple(request for unit in units for request in unit.requests))
        assert module._ready and module._helper_plan("q").prepared is not None
        assert module.swa_cache_layer.kv_cache.numel() == 0
        for plan in module._wo_plans.values():
            current_workspace_manager().get_simultaneous(
                *((spec.shape, spec.dtype) for spec in plan.scratch_specs())
            )
        session.freeze()
        cases = [(rows, False, module._wo_plans[rows]) for rows in counts]
        remainders = (3575, 3582) if compacted else (13, 23)
        cases.extend((rows, True, module._wo_plans["prefill"]) for rows in remainders)
        for rows, is_prefill, plan in cases:
            scratch = tuple(torch.empty(spec.shape, dtype=spec.dtype, device=device) for spec in plan.scratch_specs())
            binding = wo.bind_inv_rope(
                plan, scratch=scratch, o=source[:rows], positions=positions[:rows],
                cos_sin_cache=table, weights=module._wo_projection_weights,
                heads_per_group=heads_per_group, nope_dim=448, rope_dim=64,
            )
            expected = wo.run_inv_rope(binding=binding, plan=plan).clone()
            actual = module._o_proj(source[:rows], positions[:rows], is_prefill=is_prefill)
            if is_prefill:
                assert rows not in module._wo_plans
            assert torch.isfinite(actual).all() and torch.count_nonzero(actual) > 0
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            for tensor in current_workspace_manager().get_simultaneous(
                *((spec.shape, spec.dtype) for spec in plan.scratch_specs())
            ):
                tensor.zero_()
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            graph = torch.cuda.CUDAGraph()
            try:
                with session.capture(), torch.cuda.graph(graph):
                    replayed = module._o_proj(source[:rows], positions[:rows], is_prefill=is_prefill)
                pointer = replayed.data_ptr()
                source[:rows].neg_()
                positions[:rows].add_(1).remainder_(table.shape[0])
                replayed.fill_(float("nan"))
                allocated = torch.cuda.memory_allocated(device)
                graph.replay()
                torch.cuda.synchronize(device)
                assert torch.cuda.memory_allocated(device) == allocated
                assert replayed.data_ptr() == pointer
                expected = wo.run_inv_rope(binding=binding, plan=plan)
                torch.testing.assert_close(replayed, expected, atol=0, rtol=0)
                assert torch.isfinite(replayed).all() and torch.count_nonzero(replayed) > 0
            finally:
                graph.reset()
        if dcp_size == 1:
            with pytest.raises(RuntimeError, match="frozen"):
                module._o_proj(source[:5], positions[:5])
        else:
            with pytest.raises(PreparationResourceUnavailableError, match="decode.*capacity"):
                module._o_proj(source[:5], positions[:5])
            declared = tuple(module._wo_plans)
            module._o_proj(source[:5], positions[:5], is_prefill=True)
            assert tuple(module._wo_plans) == declared
            with pytest.raises(PreparationResourceUnavailableError, match="capacity"):
                module._wo_plan(capacity + 1)


@pytest.mark.parametrize("compacted", [False, True])
@pytest.mark.parametrize("dcp_size", [1, 4])
@pytest.mark.parametrize("shard_index", [False, True])
def test_indexer_declares_bounded_score_rows_at_model_context_capacity(
    monkeypatch, compacted, dcp_size, shard_index
):
    if not torch.cuda.is_available():
        pytest.skip("native b12x attention declarations require CUDA")
    from b12x.attention import compressed_sparse_mla as mla, dsa_indexer
    from vllm.models.deepseek_v4_1 import attention
    from vllm.utils.b12x import B12xWorkload
    from vllm.v1.worker.workspace import WorkspaceManager

    device = torch.device("cuda", torch.cuda.current_device())
    module = attention.DeepseekV4Attention.__new__(attention.DeepseekV4Attention)
    torch.nn.Module.__init__(module)
    module.prefix = "model.layers.0.attn"
    module.capacity, module.max_model_len, module.swa_width = 4096, 1048576, 128
    module.is_ced_decoder, module.is_index_source = compacted, True
    module.n_local_heads, module.compress_ratio = 16, 1 if compacted else 2
    module.dcp_size, module.dcp_active = dcp_size, dcp_size > 1
    module.dcp_shard_index = shard_index and dcp_size > 1
    module.dcp_prefill_replica = dcp_size == 4 and not compacted
    module.is_kv_source = True
    module.layer_id = 12 if compacted else 0
    module.candidate_source_layer, module.kv_source_layer_id = 12, 0
    module._context = {module.prefix: module}
    module.topk_indices_buffer = None
    module.config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=256), speculative_config=None,
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=64),
    )
    module.rotary_emb = SimpleNamespace(cos_sin_cache=torch.empty(1, 64, device=device))
    module.swa_cache_layer = SimpleNamespace(
        block_size=256,
        kv_cache=torch.empty((1, mla.page_nbytes(256, cache_kind="swa")), dtype=torch.uint8, device=device),
    )
    main_page = 256 // module.compress_ratio
    module.kv_cache = torch.empty(
        (1, mla.page_nbytes(main_page, cache_kind="indexed")),
        dtype=torch.uint8, device=device,
    )
    module.indexer = SimpleNamespace(heads=32, k_cache=SimpleNamespace(
        kv_cache=torch.empty(
            (1, dsa_indexer.index_mxfp4_page_bytes(main_page)),
            dtype=torch.uint8, device=device,
        ),
    ))
    workload = B12xWorkload(
        stage="state", token_counts=(1, 7, 64, 257, 4089, 4096),
        fixed_token_counts=(1, 7, 64), output_dtype=torch.bfloat16,
        max_tokens=4096, max_seqs=8, max_model_len=1048576,
    )
    allocated = torch.cuda.memory_allocated(device)
    (unit,) = module.get_b12x_preparation_units(module, workload)
    assert torch.cuda.memory_allocated(device) == allocated
    assert not hasattr(module, "_dcp_exchange")
    for plan in module._attention_plans.values():
        replica = module.dcp_prefill_replica and plan.query.mode == "extend"
        assert plan.query.num_q_heads == (16 if replica else 16 * dcp_size)
        if dcp_size > 1 and not replica:
            assert plan.query.query_rows <= 256
        if replica:
            assert plan.query.query_rows == module.capacity
            assert plan.query.max_page_table_width == 4096
            assert not plan.query.return_lse
    assert module._main_width == 4096 // dcp_size
    expected_index_width = 4096 // dcp_size if module.dcp_shard_index else 4096
    assert module._index_width == expected_index_width
    assert "_prefill_kv" not in {name for name, _, _ in module._staging_specs}
    manager = WorkspaceManager(device)
    monkeypatch.setattr(attention, "current_workspace_manager", lambda: manager)
    module.reserve_profile_scratch()
    manager.lock()
    allocated = torch.cuda.memory_allocated(device)
    regimes = [("decode", 64), ("prefill", 256)]
    if module._short_index_shape is not None:
        regimes.append(("prefill_short", module._short_index_shape[0]))
    for mode, rows in regimes:
        declaration = module._declare_index_plan(mode, rows)
        manager.get_simultaneous(
            *((spec.shape, spec.dtype) for spec in declaration.scratch_specs())
        )
    assert torch.cuda.memory_allocated(device) == allocated
    assert unit.requests and all(request.collective is None for request in unit.requests)
    counts = module._preparation_token_counts(workload)
    chunks = {"decode": 64, "prefill": 256}
    if not compacted:
        chunks["prefill_short"] = 1024
    for mode, chunk in chunks.items():
        for total in counts:
            for offset in range(0, total, chunk):
                rows = min(total - offset, chunk)
                plan = module._index_plan(mode, rows)
                assert plan.query.max_q_rows == rows
                assert plan.query.max_q_rows <= chunk
                assert rows * plan.query.max_page_table_width * plan.query.page_size < 2**31
    assert ("prefill", 4096) not in module._index_plans
    assert ("decode", 4096) not in module._index_plans
    for mode, chunk in chunks.items():
        assert (mode, 13) not in module._index_plans
        prepared_capacity = module._index_plans[mode, chunk]
        assert module._index_plan(mode, 13) is prepared_capacity
        assert module._index_plan(mode, 29) is prepared_capacity
        assert (mode, 13) not in module._index_plans
        assert (mode, 29) not in module._index_plans


@pytest.mark.parametrize("layer_id", [2, 12, 14])
@torch.no_grad()
def test_indexer_primer_restores_live_cache(layer_id):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("native b12x indexer requires SM12x")
    from b12x.attention import dsa_indexer
    from b12x.preparation import PreparationSession
    from vllm.models.deepseek_v4_1 import attention

    device = torch.device("cuda", torch.cuda.current_device())
    module = attention.DeepseekV4Attention.__new__(attention.DeepseekV4Attention)
    torch.nn.Module.__init__(module)
    module.layer_id, module.candidate_source_layer, module._index_page = layer_id, 12, 128
    cache = torch.randint(0, 256, (8, dsa_indexer.index_mxfp4_page_bytes(128)),
                          dtype=torch.uint8, device=device)
    original = cache.clone()
    module.indexer = SimpleNamespace(heads=32, k_cache=SimpleNamespace(kv_cache=cache))
    plans = []
    requests = []
    for mode, rows, width in (("decode", 1, 4096), ("prefill", 256, 4096),
                              ("prefill_short", 1024, 128)):
        plan = dsa_indexer.plan(dsa_indexer.Caps(
            device=device, num_q_heads=32, max_q_rows=rows,
            max_page_table_width=width, topk=512,
            mode="decode" if mode == "decode" else "prefill",
            cache_format="mxfp4", page_size=128,
            max_candidates=16384 if layer_id > 12 else 0,
            candidate_topk_blocks=2048 if layer_id == 12 else 0,
        ))
        plans.append((plan, mode, rows))
        call = module._index_call(mode, rows)
        requests.append(plan.request(name=f"indexer.{mode}.{rows}", prepare_call=call, benchmark_call=call))
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare(requests)
        torch.testing.assert_close(cache, original, rtol=0, atol=0)
        session.freeze()
        for plan, mode, rows in plans:
            call = module._index_call(mode, rows)(plan.prepared.state)
            try:
                call.produce()
                call.reset()
                call.run()
                torch.cuda.synchronize(device)
                assert call.output.shape == (rows, 512)
                assert ((call.output >= 0) & (call.output < 1024)).all()
                assert torch.sort(call.output, dim=1).values.diff(dim=1).gt(0).all()
            finally:
                call.restore()
            torch.testing.assert_close(cache, original, rtol=0, atol=0)


@pytest.mark.parametrize("dcp_size", [1, 4])
def test_wo_prefill_remainders_reuse_declared_chunk_capacity(dcp_size):
    from vllm.models.deepseek_v4_1 import attention
    from vllm.utils.b12x import B12xWorkload

    module = attention.DeepseekV4Attention.__new__(attention.DeepseekV4Attention)
    torch.nn.Module.__init__(module)
    module.prefix, module.capacity, module.is_ced_decoder = "model.layers.0.attn", 4096, False
    module.dcp_size = dcp_size
    module.n_local_groups, module.n_local_heads = 2, 16
    module.head_dim, module.rope_head_dim = 512, 64
    module._wo_plans = {}
    module._wo_projection_weights = SimpleNamespace(groups=2, group_width=4096, rank=1024, hidden=5120)
    module.rotary_emb = SimpleNamespace(cos_sin_cache=torch.empty(1, 64, dtype=torch.bfloat16))
    workload = B12xWorkload(
        stage="weights", token_counts=(1, 8, 4096), fixed_token_counts=(1, 8),
        output_dtype=torch.bfloat16, max_tokens=4096, max_seqs=8, max_model_len=4096,
    )
    unit = module._wo_preparation_unit(workload)
    declarations = dict(module._wo_plans)
    prefill = module._wo_plan(4096, is_prefill=True)
    assert any(request.plan is prefill for request in unit.requests)
    assert prefill.query.max_tokens == 4096 and prefill.query.dynamic_tokens
    for rows in (1, 127, 128, 129, 3575, 3582, 4096):
        assert module._wo_plan(rows, is_prefill=True) is prefill
    assert module._wo_plans == declarations
    assert not module._wo_plan(1).query.dynamic_tokens
    assert module._wo_plan(8).query.max_tokens == 8
    if dcp_size > 1:
        from vllm.utils.b12x import PreparationResourceUnavailableError

        with pytest.raises(PreparationResourceUnavailableError, match="decode.*capacity"):
            module._wo_plan(7)
        assert module._wo_plans == declarations
