# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pooled-state GDN prefill integration and fixed-capacity replay."""

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
    from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution
    from b12x.policy.generation.delta_prefill_cases import (
        PrefillCase,
        assert_close,
        make_inputs,
        oracle,
    )

    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("SM12x is required")
    device = torch.device("cuda")
    case = PrefillCase("gdn", 2, 6, (33, 16))
    tensors = make_inputs(case, device=device, max_tokens=64, max_seqs=2)
    pool = tensors["recurrent_state"]
    saved = pool.clone()
    runner = B12xGdnPrefill(
        recurrent_state=pool,
        A_log=tensors["A_log"],
        dt_bias=tensors["dt_bias"],
        max_tokens=64,
        max_seqs=2,
        key_heads=2,
        value_heads=6,
        checkpoint_export=True,
    )
    packed = torch.cat([tensors[key].flatten(1) for key in ("q", "k", "v")], dim=-1)
    slots = torch.tensor([1, 2], dtype=torch.int32, device=device)
    fresh = torch.tensor([True, False], device=device)
    counts = torch.tensor([2, 49], dtype=torch.int32, device=device)
    checkpoint = SimpleNamespace(
        state_indices=torch.tensor([3, 4], dtype=torch.int32, device=device),
        checkpoint_offsets=torch.tensor([16, 0], dtype=torch.int32, device=device),
    )
    args = dict(
        mixed_qkv=packed,
        a=tensors["raw_g"],
        b=tensors["raw_beta"],
        query_start_loc=tensors["cu_seqlens"],
        state_indices=slots,
        has_initial_state=fresh,
        live_counts=counts,
        output=tensors["output"],
        checkpoint=checkpoint,
    )
    runner.run(**args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runner.run(**args)
    freeze_kernel_resolution("GDN prefill integration replay")
    try:
        for lengths, state_slots in (((33, 16), (1, 2)), ((16, 31), (2, 1))):
            pool.copy_(saved)
            slots.copy_(torch.tensor(state_slots, dtype=torch.int32, device=device))
            counts.copy_(
                torch.tensor([2, sum(lengths)], dtype=torch.int32, device=device)
            )
            tensors["cu_seqlens"].copy_(
                torch.tensor(
                    [0, lengths[0], sum(lengths)], dtype=torch.int32, device=device
                )
            )
            tensors["initial_state_indices"].copy_(slots)
            tensors["initial_state_indices"][1] = 0
            tensors["final_state_indices"].copy_(slots)
            tensors["checkpoint_state_indices"].copy_(checkpoint.state_indices)
            tensors["checkpoint_offsets"].copy_(checkpoint.checkpoint_offsets)
            tensors["num_tokens"].copy_(counts[1:])
            expected, expected_pool = oracle(
                PrefillCase("gdn", 2, 6, lengths), tensors, null_state_index=0
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
                    f"state[{slot}]", pool[slot], expected_pool[slot], ratio=5e-3
                )
            torch.testing.assert_close(pool[0], saved[0], rtol=0, atol=0)
    finally:
        unfreeze_kernel_resolution()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mixed_gdn_graph_replays_prefill_decode_verification_and_empty_worklists(
    default_vllm_config,
):
    from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution
    from b12x.policy.generation.delta_prefill_cases import assert_close
    from b12x.sequence.gdn_decode.reference import decode
    from b12x.sequence.gdn_prefill.reference import prefill_gdn

    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
        RMSNormGated,
        is_conv_state_dim_first,
    )
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_fn,
        causal_conv1d_update,
    )
    from vllm.v1.attention.backends.b12x_gdn_metadata import B12xGdnMixedMetadata
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

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
    layer._initialize_b12x_gdn_decode(
        SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=seqs))
    )
    plan = layer._make_b12x_gdn_plan(max_state_slots=pool.shape[0])
    layer._b12x_plan = plan
    spec = plan.scratch_specs()[0]
    layer._b12x_scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
    layer._b12x_prefill = B12xGdnPrefill(
        recurrent_state=pool,
        A_log=layer.A_log,
        dt_bias=layer.dt_bias,
        max_tokens=rows,
        max_seqs=seqs,
        key_heads=key_heads,
        value_heads=value_heads,
        checkpoint_export=True,
    )
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
        with torch.cuda.graph(graph):
            layer._forward_core_b12x_mixed(**inputs)
        freeze_kernel_resolution("vLLM mixed GDN replay")
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
                    "mixed output", inputs["core_attn_out"], expected, ratio=1e-2
                )
                assert_close("mixed state", pool, expected_pool, ratio=5e-3)
                torch.testing.assert_close(conv, expected_conv, rtol=0, atol=0)
        finally:
            unfreeze_kernel_resolution()
