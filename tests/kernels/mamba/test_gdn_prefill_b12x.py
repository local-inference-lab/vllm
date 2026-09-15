# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pooled-state GDN prefill integration and fixed-capacity replay."""

from math import prod
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.mamba.ops.b12x_gdn_prefill import (
    B12xGdnPrefill,
    prefill_capacities,
)


@pytest.mark.parametrize(
    "capacity,expected",
    [
        (1, (1,)),
        (16, (16,)),
        (33, (16, 32, 33)),
        (128, (16, 32, 64, 128)),
    ],
)
def test_prefill_capacity_family_covers_exact_scheduler_limit(capacity, expected):
    assert prefill_capacities(capacity) == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepared_gdn_decode_plan_is_scoped_to_the_bound_recurrent_pool(
    default_vllm_config,
):
    """The decode plan is declared from the bound pool and dropped on rebind."""
    from b12x.preparation import PreparationSession

    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
        RMSNormGated,
    )
    from vllm.utils.b12x import B12xWorkload
    from vllm.v1.worker.workspace import (
        current_workspace_manager,
        init_workspace_manager,
    )

    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("SM12x is required")

    device = torch.device("cuda")
    max_tokens, max_seqs, key_heads, value_heads = 6, 2, 2, 6
    state_columns = 3
    packed_width = (2 * key_heads + value_heads) * 128
    state_shapes = ((packed_width, 5), (value_heads, 128, 128))
    state_dtypes = (torch.bfloat16, torch.float32)

    layer = QwenGatedDeltaNetAttention.__new__(QwenGatedDeltaNetAttention)
    torch.nn.Module.__init__(layer)
    layer.gqa_interleaved_layout = False
    layer.gdn_decode_kernel = "b12x"
    layer.gdn_prefill_backend = "triton"
    layer.num_spec = state_columns - 1
    layer.num_k_heads = key_heads
    layer.num_v_heads = value_heads
    layer.tp_size = 1
    layer.head_k_dim = layer.head_v_dim = 128
    layer.model_config = SimpleNamespace(dtype=torch.bfloat16)
    layer.get_state_shape = lambda: state_shapes
    layer.get_state_dtype = lambda: state_dtypes
    layer.norm = RMSNormGated(
        128,
        eps=1e-6,
        norm_before_gate=True,
        activation="silu",
        device=device,
    )
    layer.norm.weight.data.fill_(1)
    layer.A_log = torch.nn.Parameter(torch.full((value_heads,), -1.0, device=device))
    layer.dt_bias = torch.nn.Parameter(torch.zeros(value_heads, device=device))
    layer._b12x_gdn_api = None
    layer._b12x_prefill_api = None
    layer._b12x_decode_plan = None
    layer._b12x_decode_staging = None
    layer._b12x_prefill_plans = {}
    layer._b12x_prefill_staging = None
    layer._b12x_prefill = None
    layer._b12x_preparation_prefix = "test.gdn-publication"
    layer._initialize_b12x_gdn_decode(
        SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=max_seqs))
    )

    init_workspace_manager(device)
    workspace = current_workspace_manager()
    # This fixture isolates resource publication. The model profile hook owns
    # the equivalent serving reservation in a complete worker startup.
    workspace.reserve_all(((64 << 20,), torch.uint8))

    workload = B12xWorkload(
        stage="state", token_counts=(1, 2, max_tokens), fixed_token_counts=(),
        output_dtype=torch.bfloat16, max_tokens=max_tokens, max_seqs=max_seqs,
        max_model_len=max_tokens, speculative_tokens=state_columns - 1,
    )
    # Before any pool is bound, the layer declares no decode plan: preparation
    # is scoped entirely to the bound pool, not to a registry publication step.
    assert layer._b12x_decode_plan is None
    assert layer.get_b12x_preparation_units(layer, workload) == ()

    page_nbytes = sum(
        prod(shape) * dtype.itemsize
        for shape, dtype in zip(state_shapes, state_dtypes)
    )

    def publish_pool() -> torch.Tensor:
        raw = torch.empty(
            (4, 1, 1, page_nbytes), dtype=torch.uint8, device=device
        )
        layer.bind_kv_cache(raw)
        return raw

    session = PreparationSession(device=device, autotune=False)

    def prepare() -> None:
        units = layer.get_b12x_preparation_units(layer, workload)
        requests = tuple(request for unit in units for request in unit.requests)
        session.prepare(requests, autotune=False)

    first_pool = publish_pool()
    first_recurrent = layer.kv_cache[1]
    assert first_recurrent.untyped_storage().data_ptr() == first_pool.data_ptr()
    first_plan = layer._b12x_decode_plan
    assert first_plan is not None and first_plan.prepared is None
    prepare()
    assert first_plan.prepared is not None

    session.release(first_plan)
    assert first_plan.prepared is None

    second_pool = publish_pool()
    second_recurrent = layer.kv_cache[1]
    assert second_recurrent.untyped_storage().data_ptr() == second_pool.data_ptr()
    assert second_recurrent.data_ptr() != first_recurrent.data_ptr()
    second_plan = layer._b12x_decode_plan
    assert second_plan is not first_plan
    assert second_plan.prepared is None

    prepare()
    assert second_plan.prepared is not None
    assert first_plan.prepared is None
    session.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_group_worklist_copies_update_captured_buffers_without_aliasing():
    from vllm.v1.attention.backends.b12x_gdn_metadata import B12xGdnMixedMetadata

    device = torch.device("cuda")
    groups = [
        B12xGdnMixedMetadata(max_tokens=64, max_seqs=3, state_columns=4, device=device)
        for _ in range(3)
    ]
    outputs = [
        [
            torch.empty_like(work.state_indices),
            torch.empty_like(work.spec_state_indices),
            torch.empty_like(work.checkpoint.state_indices),
            torch.empty_like(work.token_indices),
        ]
        for work in groups
    ]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for work, output in zip(groups, outputs):
            for destination, source in zip(
                output,
                (
                    work.state_indices,
                    work.spec_state_indices,
                    work.checkpoint.state_indices,
                    work.token_indices,
                ),
            ):
                destination.copy_(source)
    for lengths, computed, drafts in (
        ([1, 33, 4], [32, 0, 16], [-1, -1, 3]),
        ([17, 1, 2], [0, 32, 16], [-1, -1, 1]),
        ([4, 4, 4], [32, 32, 32], [3, 3, 3]),
        ([0, 0, 0], [0, 0, 0], [-1, -1, -1]),
    ):
        starts = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
        )
        seq_lens = torch.tensor(lengths, dtype=torch.int32) + torch.tensor(computed)
        common = SimpleNamespace(
            query_start_loc_cpu=starts,
            query_start_loc=starts.to(device),
            num_reqs=3,
            seq_lens=seq_lens.to(device),
            seq_lens_cpu_upper_bound=seq_lens,
            block_table_tensor=torch.arange(
                9, dtype=torch.int32, device=device
            ).reshape(3, 3)
            + 50,
        )
        indices = torch.arange(12, dtype=torch.int32, device=device).reshape(3, 4) + 1
        groups[0].stage(
            common,
            indices,
            torch.ones(3, dtype=torch.int32, device=device),
            torch.tensor(drafts),
            checkpoint_block_size=16,
        )
        expected = []
        for group, work in enumerate(groups):
            work.copy_worklists_from(groups[0])
            work.refresh_state_indices(
                indices + 100 * group, common.block_table_tensor + 100 * group
            )
            expected.append(
                [
                    value.clone()
                    for value in (
                        work.state_indices,
                        work.spec_state_indices,
                        work.checkpoint.state_indices,
                        work.token_indices,
                    )
                ]
            )
        torch.accelerator.synchronize()
        before = torch.accelerator.memory_stats()
        graph.replay()
        torch.accelerator.synchronize()
        after = torch.accelerator.memory_stats()
        assert after["allocation.all.allocated"] == before["allocation.all.allocated"]
        for actual, reference in zip(outputs, expected):
            for value, target in zip(actual, reference):
                torch.testing.assert_close(value, target, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prefill_pooled_state_graph_replays_changed_lengths_and_slots():
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.testing.delta_prefill_cases import (
        PrefillCase,
        assert_close,
        make_inputs,
        oracle,
        prepared_binding,
        run_binding,
    )

    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("SM12x is required")
    device = torch.device("cuda")
    case = PrefillCase("gdn", 2, 6, (33, 16))
    tensors = make_inputs(case, device=device, max_tokens=64, max_seqs=2)
    pool = tensors["recurrent_state"]
    with prepared_binding(
        case,
        tensors,
        max_tokens=64,
        max_seqs=2,
        checkpoint_export=True,
        null_state_index=0,
    ) as binding:
        graph = torch.cuda.CUDAGraph()
        run_binding("gdn", binding)
        with torch.cuda.graph(graph):
            run_binding("gdn", binding)
        guard = kernel_resolution_guard("GDN prefill integration replay")
        guard.__enter__()
        saved = pool.clone()
        try:
            for lengths, state_slots in (((33, 16), (1, 2)), ((16, 31), (2, 1))):
                pool.copy_(saved)
                tensors["cu_seqlens"].copy_(
                    torch.tensor(
                        [0, lengths[0], sum(lengths)],
                        dtype=torch.int32,
                        device=device,
                    )
                )
                tensors["initial_state_indices"].copy_(
                    torch.tensor(state_slots, dtype=torch.int32, device=device)
                )
                tensors["initial_state_indices"][1] = 0
                tensors["final_state_indices"].copy_(
                    torch.tensor(state_slots, dtype=torch.int32, device=device)
                )
                tensors["checkpoint_state_indices"].copy_(
                    torch.tensor((3, 4), dtype=torch.int32, device=device)
                )
                tensors["checkpoint_offsets"].copy_(
                    torch.tensor((16, 0), dtype=torch.int32, device=device)
                )
                tensors["num_seqs"].fill_(2)
                tensors["num_tokens"].fill_(sum(lengths))
                expected, expected_pool = oracle(
                    PrefillCase("gdn", 2, 6, lengths),
                    tensors,
                    null_state_index=0,
                )
                tensors["output"].fill_(float("nan"))
                graph.replay()
                torch.accelerator.synchronize()
                assert_close(
                    "output",
                    tensors["output"][: sum(lengths)],
                    expected[: sum(lengths)],
                    ratio=1e-2,
                )
                for slot in (1, 2, 3):
                    assert_close(
                        f"state[{slot}]",
                        pool[slot],
                        expected_pool[slot],
                        ratio=5e-3,
                    )
                torch.testing.assert_close(pool[0], saved[0], rtol=0, atol=0)
        finally:
            guard.__exit__(None, None, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mixed_gdn_graph_replays_prefill_decode_verification_and_empty_worklists(
    default_vllm_config,
):
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.preparation import PreparationSession
    from b12x.sequence.gdn_decode.reference import decode
    from b12x.sequence.gdn_prefill.reference import prefill_gdn
    from b12x.testing.delta_prefill_cases import assert_close

    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
        RMSNormGated,
        is_conv_state_dim_first,
    )
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_fn,
        causal_conv1d_update,
    )
    from vllm.utils.b12x import B12xWorkload
    from vllm.v1.attention.backends.b12x_gdn_metadata import B12xGdnMixedMetadata
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
    from vllm.v1.worker.workspace import (
        current_workspace_manager,
        init_workspace_manager,
    )

    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("SM12x is required")
    torch.manual_seed(23)
    device = torch.device("cuda")
    rows, seqs, key_heads, value_heads = 64, 3, 2, 6
    width = (2 * key_heads + value_heads) * 128
    layer = QwenGatedDeltaNetAttention.__new__(QwenGatedDeltaNetAttention)
    torch.nn.Module.__init__(layer)
    layer.gqa_interleaved_layout = False
    layer.gdn_decode_kernel = "b12x"
    layer.gdn_prefill_backend = "b12x"
    layer.num_spec = 2
    layer.num_k_heads, layer.num_v_heads, layer.tp_size = key_heads, value_heads, 1
    layer.head_k_dim = layer.head_v_dim = 128
    layer.layer_norm_epsilon = 1e-6
    layer.model_config = SimpleNamespace(dtype=torch.bfloat16)
    layer.get_state_dtype = lambda: (torch.bfloat16, torch.float32)
    layer.norm = RMSNormGated(
        128, eps=1e-6, norm_before_gate=True, activation="silu", device=device
    )
    layer.norm.weight.data.fill_(1)
    layer.A_log = torch.nn.Parameter(torch.full((value_heads,), -1.0, device=device))
    layer.dt_bias = torch.nn.Parameter(torch.zeros(value_heads, device=device))
    layer.activation = "silu"
    layer.conv1d = torch.nn.Conv1d(
        width, width, 4, groups=width, device=device, dtype=torch.bfloat16
    )
    conv = (torch.randn(20, width, 5, device=device) * 0.1).bfloat16()
    pool = torch.randn(20, value_heads, 128, 128, device=device) * 0.1
    layer.kv_cache = (
        conv if is_conv_state_dim_first() else conv.transpose(-1, -2),
        pool,
    )
    layer._b12x_preparation_prefix = "test.mixed-gdn"
    layer._b12x_prefill_max_tokens = rows
    layer._b12x_prefill_max_seqs = seqs
    layer._b12x_prefill_staging = None
    layer._b12x_decode_plan = None
    layer._b12x_prefill_plans = {}
    layer._initialize_b12x_gdn_decode(
        SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=seqs))
    )
    layer._initialize_b12x_gdn_prefill()
    init_workspace_manager(device)

    # Declare the decode and prefill plans against the already-bound pool,
    # exactly as ``QwenGatedDeltaNetAttention.bind_kv_cache`` does for a
    # published recurrent-state pool.
    from vllm.model_executor.layers.mamba.ops.b12x_gdn_prefill import (
        B12xGdnPrefill,
        prefill_capacities,
    )

    layer._b12x_decode_plan = layer._make_b12x_gdn_plan(pool.shape[0])
    layer._b12x_prefill_plans = {
        capacity: layer._b12x_gdn_prefill_declaration(capacity)
        for capacity in prefill_capacities(layer._b12x_prefill_max_tokens)
    }
    prefill_staging = layer._ensure_b12x_gdn_prefill_staging()
    layer._b12x_prefill = B12xGdnPrefill(
        recurrent_state=pool,
        A_log=layer.A_log,
        dt_bias=layer.dt_bias,
        max_tokens=layer._b12x_prefill_max_tokens,
        max_seqs=layer._b12x_prefill_max_seqs,
        key_heads=layer._b12x_local_key_heads,
        value_heads=layer._b12x_local_value_heads,
        checkpoint_export=True,
        plans=layer._b12x_prefill_plans,
        staging=prefill_staging,
    )

    workload = B12xWorkload(
        stage="state", token_counts=(1, 2, 4, 16, 32, rows), fixed_token_counts=(),
        output_dtype=torch.bfloat16, max_tokens=rows, max_seqs=seqs,
        max_model_len=rows, speculative_tokens=2,
    )
    units = layer.get_b12x_preparation_units(layer, workload)
    requests = tuple(request for unit in units for request in unit.requests)
    session = PreparationSession(device=device, autotune=False)
    session.prepare(requests, autotune=False)
    metadata = B12xGdnMixedMetadata(
        max_tokens=rows, max_seqs=seqs, state_columns=3, device=device
    )
    attention = GDNAttentionMetadata(0, 0, 0, 0, 0, 0, rows, b12x_mixed=metadata)
    inputs = dict(
        mixed_qkv=(torch.randn(rows, width, device=device) * 0.25).bfloat16(),
        a=(torch.randn(rows, value_heads, device=device) * 0.25).bfloat16(),
        b=(torch.randn(rows, value_heads, device=device) * 0.25).bfloat16(),
        output_gate=(
            torch.randn(rows, value_heads, 128, device=device) * 0.25
        ).bfloat16(),
        core_attn_out=torch.empty(
            rows, value_heads, 128, dtype=torch.bfloat16, device=device
        ),
        attn_metadata=attention,
    )
    saved_conv, saved_pool = conv.clone(), pool.clone()

    def stage(lengths, computed, drafts, swap):
        starts = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
        )
        lengths_tensor = torch.tensor(lengths, dtype=torch.int32)
        seq_lens = lengths_tensor + torch.tensor(computed, dtype=torch.int32)
        common = SimpleNamespace(
            query_start_loc_cpu=starts,
            query_start_loc=starts.to(device),
            num_reqs=seqs,
            seq_lens=seq_lens.to(device),
            seq_lens_cpu_upper_bound=seq_lens,
            block_table_tensor=torch.tensor(
                [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                dtype=torch.int32,
                device=device,
            ),
        )
        slots = torch.arange(1, 10, dtype=torch.int32, device=device).reshape(3, 3)
        if swap:
            slots = slots[[1, 0, 2]]
        metadata.stage(
            common,
            slots,
            torch.tensor([1, 1, 1 if swap else 2], dtype=torch.int32, device=device),
            torch.tensor(drafts),
            checkpoint_block_size=16,
        )

    def oracle():
        expected_conv, expected_pool = saved_conv.clone(), saved_pool.clone()
        expected = torch.zeros_like(inputs["core_attn_out"])
        packed = inputs["mixed_qkv"].index_select(0, metadata.token_indices)
        weights = layer.conv1d.weight.view(width, 4)
        convolved = causal_conv1d_fn(
            packed.T,
            weights,
            layer.conv1d.bias,
            activation="silu",
            conv_states=expected_conv,
            has_initial_state=metadata.has_initial_state,
            cache_indices=metadata.state_indices,
            query_start_loc=metadata.query_start_loc,
            metadata=metadata.convolution_metadata(rows),
        ).T
        q, k, v = convolved.split(
            (key_heads * 128, key_heads * 128, value_heads * 128), dim=-1
        )
        actual = prefill_gdn(
            q.reshape(rows, key_heads, 128),
            k.reshape(rows, key_heads, 128),
            v.reshape(rows, value_heads, 128),
            inputs["a"][metadata.token_indices],
            inputs["b"][metadata.token_indices],
            layer.A_log,
            layer.dt_bias,
            expected_pool,
            metadata.query_start_loc,
            torch.where(metadata.has_initial_state, metadata.state_indices, 0),
            metadata.state_indices,
            metadata.checkpoint.state_indices,
            metadata.checkpoint.checkpoint_offsets,
            metadata.live_counts[0],
            metadata.live_counts[1],
            null_state_index=0,
        )
        layer._rms_norm_gated_cuda(
            actual, inputs["output_gate"][metadata.token_indices], actual
        )
        count = int(metadata.live_counts[1])
        expected[metadata.token_indices[:count]] = actual[:count]
        indices = metadata.spec_token_indices
        spec_convolved = causal_conv1d_update(
            inputs["mixed_qkv"][indices],
            expected_conv,
            weights,
            layer.conv1d.bias,
            "silu",
            conv_state_indices=metadata.spec_state_indices[:, 0],
            num_accepted_tokens=metadata.spec_accepted,
            query_start_loc=metadata.spec_query_start_loc,
            max_query_len=3,
            validate_data=False,
        )
        actual = decode(
            spec_convolved,
            inputs["a"][indices],
            inputs["b"][indices],
            inputs["output_gate"][indices],
            layer.A_log,
            layer.dt_bias,
            layer.norm.weight,
            expected_pool,
            metadata.spec_query_start_loc,
            metadata.spec_accepted,
            metadata.spec_state_indices,
            metadata.spec_counts[0],
            metadata.spec_counts[1],
            key_heads=key_heads,
            value_heads=value_heads,
            gate_activation="silu",
        )
        count = int(metadata.spec_counts[1])
        expected[indices[:count]] = actual[:count]
        return expected, expected_conv, expected_pool

    with torch.inference_mode():
        stage((1, 33, 3), (9, 0, 9), (-1, -1, 2), False)
        layer._forward_core_b12x_mixed(**inputs)
        graph = torch.cuda.CUDAGraph()
        try:
            with session.capture():
                with torch.cuda.graph(graph):
                    layer._forward_core_b12x_mixed(**inputs)
            guard = kernel_resolution_guard("vLLM mixed GDN replay")
            guard.__enter__()
            try:
                for trial in (
                    ((1, 33, 3), (9, 0, 9), (-1, -1, 2), False),
                    ((17, 1, 2), (0, 9, 9), (-1, -1, 1), True),
                    ((1, 1, 1), (9, 9, 9), (-1, -1, -1), False),
                    ((0, 0, 0), (0, 0, 0), (-1, -1, -1), False),
                ):
                    stage(*trial)
                    expected, expected_conv, expected_pool = oracle()
                    conv.copy_(saved_conv)
                    pool.copy_(saved_pool)
                    inputs["core_attn_out"].fill_(float("nan"))
                    torch.accelerator.synchronize()
                    before = torch.accelerator.memory_stats()
                    graph.replay()
                    torch.accelerator.synchronize()
                    after = torch.accelerator.memory_stats()
                    for key in (
                        "allocation.all.allocated",
                        "allocated_bytes.all.allocated",
                    ):
                        assert after[key] == before[key]
                    assert_close(
                        "mixed output",
                        inputs["core_attn_out"],
                        expected,
                        ratio=1e-2,
                    )
                    assert_close("mixed state", pool, expected_pool, ratio=5e-3)
                    torch.testing.assert_close(conv, expected_conv, rtol=0, atol=0)
            finally:
                guard.__exit__(None, None, None)
        finally:
            del graph
            torch.accelerator.synchronize()
            session.close()
