# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CED packing, native metadata and original-coordinate DSpark regressions."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.models.deepseek_v4_1.ced import (
    CEDState,
    ced_decoder_start,
    gather_rows,
    plan_ced,
    scatter_rows,
)
from vllm.models.deepseek_v4_1.sparse_mla import DeepseekV41B12xMetadata

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_boundary_is_full_resolution_source_after_encoder():
    config = SimpleNamespace(
        num_hidden_layers=40,
        compress_ratios=[4] * 20 + [1] * 20,
        kv_source_layer_ids=[0, 10, 20],
    )
    assert ced_decoder_start(config) == 20
    config.kv_source_layer_ids = [0, 10]
    assert ced_decoder_start(config) is None


def test_short_continuation_does_not_replay_old_rows():
    assert plan_ced([1, 128, 17], [False] * 3, (128, 256, 512)) is None
    plan = plan_ced([1, 4096], [False, False], (128, 256, 512, 8192))
    assert plan.capacity == 256
    assert plan.num_tokens == 129


def _state(lengths, seq_lens=None, full=None, prefix=None):
    lengths = np.asarray(lengths, dtype=np.int32)
    nr = len(lengths)
    starts = torch.tensor(np.r_[0, lengths.cumsum()], dtype=torch.int32, device="cuda")
    seq = torch.tensor(
        lengths if seq_lens is None else seq_lens, dtype=torch.int32, device="cuda"
    )
    state = CEDState(4, 8192, (128, 256, 384, 512, 8192), "cuda")
    state.stage(starts, seq, lengths, full or [False] * nr, prefix or [0] * nr)
    return state, starts, seq


@cuda
def test_mixed_metadata_keeps_context_bounds_and_padding():
    state, starts, seq = _state([1, 4096], [900, 8192])
    # Adaptive verification may produce fewer rows than the CPU scheduling bound.
    state.stage(starts, seq, [2, 4096], [False, False], [0, 0])
    indices = state.get_indices()
    assert indices.numel() == 130
    expected = torch.cat(
        (torch.tensor([0], device="cuda"), torch.arange(3969, 4097, device="cuda"))
    )
    torch.testing.assert_close(indices[:129], expected)
    assert torch.all(indices[129:] == -1)
    positions = torch.cat(
        (torch.tensor([899], device="cuda"), torch.arange(4096, 8192, device="cuda"))
    )
    metadata = DeepseekV41B12xMetadata(
        4097,
        2,
        4096,
        128,
        torch.ones((4, 64), dtype=torch.int32, device="cuda"),
        starts,
        torch.tensor([899, 4096, -1, -1], device="cuda"),
        torch.tensor([4097, 2], dtype=torch.int32, device="cuda"),
        positions,
        torch.cat(
            (
                torch.zeros(1, device="cuda", dtype=torch.int32),
                torch.ones(4096, device="cuda", dtype=torch.int32),
            )
        ),
        positions + 128,
        (positions + 1).int(),
        False,
        8192,
    )
    decoder = state.decoder_metadata(metadata)
    torch.testing.assert_close(decoder.positions[:129], positions[expected])
    assert torch.all(decoder.positions[129:] == -1)
    assert torch.all(decoder.req_id_per_token[129:] == -1)
    assert torch.all(decoder.slot_mapping[129:] == -1)
    assert torch.all(decoder.cache_lengths[129:] == 0)
    assert decoder.query_start_loc.tolist() == [0, 1, 129, 129, 129]
    assert decoder.request_positions.tolist() == [899, 8064, -1, -1]
    assert decoder.live_counts.tolist() == [129, 2]
    assert decoder.swa_replay_start.tolist() == [0, 8064, 0, 0]
    assert decoder.max_seq_len == 8192
    # Global projection metadata remains the complete encoder boundary input.
    assert metadata.positions[-1].item() == 8191
    assert metadata.query_start_loc.tolist() == [0, 1, 4097]


@cuda
def test_runtime_starts_override_scheduled_upper_bound():
    state, starts, seq = _state([1, 4096])
    starts.copy_(torch.tensor([0, 1, 4001], dtype=torch.int32, device="cuda"))
    seq.copy_(torch.tensor([1, 4000], dtype=torch.int32, device="cuda"))
    state.stage(starts, seq, [1, 4096], [False, False], [0, 0])
    indices = state.get_indices()
    assert indices[:2].tolist() == [0, 3873]
    assert indices[128].item() == 4000
    assert state.counts.tolist() == [129, 2]


@cuda
def test_prompt_logprobs_zero_keeps_every_prompt_row(monkeypatch):
    from vllm.models.deepseek_v4_1.nvidia.model_state import DeepseekV41ModelState
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    # Isolate the CED request policy from unrelated multimodal/rope bookkeeping.
    monkeypatch.setattr(DefaultModelState, "add_request", lambda *args: None)
    monkeypatch.setattr(DefaultModelState, "remove_request", lambda *args: None)
    state = DeepseekV41ModelState.__new__(DeepseekV41ModelState)
    state.ced_state, starts, seq = _state([4096, 4096])
    state._ced_prompt_logprobs = set()
    state._ced_prefix_start = {}
    batch = SimpleNamespace(
        num_reqs=2,
        req_ids=["logprobs", "ordinary"],
        query_start_loc=starts,
        seq_lens=seq,
        num_scheduled_tokens=np.array([4096, 4096]),
    )
    state.add_request(
        0,
        SimpleNamespace(
            req_id="logprobs",
            num_computed_tokens=0,
            sampling_params=SimpleNamespace(prompt_logprobs=0),
        ),
    )
    state._stage_ced(batch)
    indices = state.get_ced_indices()
    torch.testing.assert_close(indices[:4096], torch.arange(4096, device="cuda"))
    torch.testing.assert_close(
        indices[4096:4224], torch.arange(8064, 8192, device="cuda")
    )
    state.remove_request("logprobs")
    state._stage_ced(batch)
    assert state.get_ced_indices().numel() == 256


@cuda
def test_prefix_hit_short_chunk_has_no_unwritten_decoder_history():
    state, _, _ = _state([128, 17], [4224, 9017], prefix=[4096, 0])
    assert state.get_indices() is None
    assert state.replay_start.tolist() == [4096, 0, 0, 0]


@cuda
@pytest.mark.parametrize("transpose", [False, True])
def test_packing_trailing_dimensions_invalid_rows_and_scatter(transpose):
    source = torch.arange(7 * 3 * 5, device="cuda", dtype=torch.float32).view(7, 3, 5)
    if transpose:
        source = source.transpose(1, 2)
    indices = torch.tensor([6, 1, -1, 3, 99], device="cuda")
    packed = gather_rows(source, indices)
    expected = torch.zeros((5, *source.shape[1:]), device="cuda")
    expected[[0, 1, 3]] = source[[6, 1, 3]]
    torch.testing.assert_close(packed, expected)
    restored = scatter_rows(packed, indices, 7)
    full = torch.zeros_like(source)
    full[[6, 1, 3]] = source[[6, 1, 3]]
    torch.testing.assert_close(restored, full)


@cuda
def test_scaled_row_address_above_signed_int32():
    # High row IDs must be converted BEFORE multiplying the row/page stride.
    if torch.cuda.mem_get_info()[0] < 3 * 1024**3:
        pytest.skip("Needs a 2 GiB address-span allocation")
    source = torch.empty((2**23 + 2, 256), dtype=torch.uint8, device="cuda")
    source[2**23 + 1].fill_(173)
    indices = torch.tensor([2**23 + 1, -1], device="cuda")
    packed = gather_rows(source, indices)
    assert torch.all(packed[0] == 173)
    assert torch.all(packed[1] == 0)


@cuda
def test_frozen_packing_graph_replays_new_live_rows(monkeypatch):
    from vllm.models.deepseek_v4_1 import ced

    source = torch.arange(256 * 4, dtype=torch.float32, device="cuda").view(256, 2, 2)
    indices = torch.full((128,), -1, dtype=torch.int64, device="cuda")
    indices[:3] = torch.tensor([0, 17, 255], device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        packed = gather_rows(source, indices)
        restored = scatter_rows(packed, indices, 256)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        packed = gather_rows(source, indices)
        restored = scatter_rows(packed, indices, 256)
    # Replay cannot consult Python or compile another packing specialization.
    monkeypatch.setattr(ced, "_copy_rows", None)
    indices.fill_(-1)
    indices[:2] = torch.tensor([31, 127], device="cuda")
    source.add_(1000)
    graph.replay()
    expected = torch.zeros_like(source)
    expected[[31, 127]] = source[[31, 127]]
    torch.testing.assert_close(restored, expected)


@cuda
def test_dspark_compact_context_preserves_query_anchor_and_rejections(monkeypatch):
    from vllm.v1.worker.gpu.input_batch import InputBuffers
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import prepare_dflash_inputs

    device = torch.device("cuda")
    state, starts, seq = _state([1, 4096], [11, 4096])
    target_positions = torch.cat(
        (torch.tensor([10], device=device), torch.arange(4096, device=device))
    )
    batch = SimpleNamespace(
        num_reqs=2,
        num_scheduled_tokens=np.array([1, 4096]),
        positions=target_positions,
        query_start_loc=starts,
        idx_mapping=torch.arange(2, device=device),
    )
    buffers = InputBuffers(4, 8192, device)
    slots = torch.empty(8192, dtype=torch.int64, device=device)
    context_positions = torch.empty_like(slots)
    context_slots = torch.empty_like(slots)
    sample_indices = torch.empty(8, dtype=torch.int64, device=device)
    sample_pos = torch.empty_like(sample_indices)
    sample_reqs = torch.empty(8, dtype=torch.int32, device=device)
    temperature = torch.ones(4, device=device)
    seeds = torch.zeros(4, dtype=torch.int64, device=device)
    table = torch.arange(1, 4 * 40 + 1, dtype=torch.int32, device=device).view(4, 40)
    table[1, 31] = 0  # Evicted/null page in retained context must remain unwritable.
    prepare_dflash_inputs(
        buffers,
        slots,
        context_positions,
        context_slots,
        sample_indices,
        sample_pos,
        sample_reqs,
        temperature,
        seeds,
        batch,
        torch.tensor([1, 1], device=device),
        torch.tensor([0, 2], device=device),
        torch.tensor([77, 88, 0, 0], device=device),
        torch.zeros(4, device=device),
        temperature,
        seeds,
        table,
        128,
        0,
        1,
        1,
        999,
        2,
        2,
        4,
        8192,
        8192,
        True,
    )
    indices = state.get_indices()
    packed_positions = gather_rows(context_positions, indices)
    packed_slots = gather_rows(context_slots, indices).masked_fill(indices < 0, -1)
    assert buffers.positions[:4].tolist() == [11, 12, 4094, 4095]
    assert buffers.input_ids[:4].tolist() == [77, 999, 88, 999]
    assert sample_pos[:4].tolist() == [12, 13, 4095, 4096]
    assert packed_positions[:3].tolist() == [10, 3968, 3969]
    assert packed_positions[127:129].tolist() == [0, 0]
    assert packed_slots[0].item() == 138
    assert torch.all(packed_slots[1:] == -1)
    # Only the first request writes; skipped prefix, null pages, rejected rows,
    # and compact padding cannot alias slot zero or resurrect stale draft KV.
    pool = torch.full((1024,), -7.0, device=device)
    writable = packed_slots >= 0
    pool[packed_slots[writable]] = 5
    assert pool[0].item() == -7
    assert torch.count_nonzero(pool != -7).item() == 1

    # Exercise propose itself: this catches packing after combine (too late),
    # packing raw rather than prepared slots, and checkpoint context overwrites.
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash

    table[1, 31] = 72
    context_pool = torch.full((25600, 2), -7.0, device=device)
    aux = torch.arange(4097 * 2, dtype=torch.float32, device=device).view(4097, 2)
    combine_rows = []

    def combine(hidden):
        combine_rows.append(hidden.shape[0])
        return hidden + 3

    def project(hidden, positions, context_slot_mapping):
        valid = context_slot_mapping >= 0
        context_pool[context_slot_mapping[valid]] = (
            hidden[valid] + positions[valid, None]
        )

    proposer = dflash.DFlashSpeculator.__new__(dflash.DFlashSpeculator)
    proposer.model = SimpleNamespace(
        combine_hidden_states=combine, precompute_and_store_context_kv=project
    )
    proposer.model_state = SimpleNamespace(get_ced_indices=state.get_indices)
    proposer.hidden_states = torch.zeros((8192, 2), device=device)
    proposer.context_positions = context_positions
    proposer._context_slot_mappings = context_slots[None]
    proposer._context_preparer = SimpleNamespace(
        can_run=lambda count: pytest.fail(
            "Compact prefill entered decode context graph"
        )
    )
    proposer._layer_group_idx = None
    proposer.draft_kv_cache_group_id = 0
    proposer.draft_kv_cache_group_ids = [0]
    proposer.block_tables = SimpleNamespace(
        get_group_cp_parameters=lambda gid: (0, 1, 1),
        slot_mappings=slots[None],
        input_block_tables=[table],
        kernel_block_sizes=[128],
    )
    proposer.input_buffers = buffers
    proposer.sample_indices = sample_indices
    proposer.sample_pos = sample_pos
    proposer.sample_idx_mapping = sample_reqs
    proposer.temperature = temperature
    proposer.seeds = seeds
    proposer.parallel_drafting_token_id = 999
    proposer.num_query_per_req = proposer.num_speculative_steps = 2
    proposer.max_num_reqs = 4
    proposer.max_num_tokens = proposer.max_model_len = 8192
    proposer.sample_from_anchor = True
    proposer.dp_size = 1
    proposer.dp_rank = 0
    proposer.query_cudagraph_manager = None
    proposer.kv_cache_config = None
    proposer._group_causal = False
    proposer._build_draft_attn_metadata = lambda **kwargs: None
    proposer._prepare_eplb_forward = lambda count: None
    proposer._generate_draft = lambda *args, **kwargs: None
    proposer.draft_tokens = torch.zeros((4, 2), device=device, dtype=torch.int64)
    batch.num_tokens = 4097
    batch.seq_lens_cpu_upper_bound = seq.cpu()
    monkeypatch.setattr(
        dflash,
        "dispatch_cg_and_sync_dp",
        lambda *args, **kwargs: (
            SimpleNamespace(num_reqs=2, num_tokens=4, cg_mode=CUDAGraphMode.NONE),
            None,
        ),
    )
    monkeypatch.setattr(dflash, "build_slot_mappings_by_layer", lambda *args: None)
    arguments = dict(
        input_batch=batch,
        attn_metadata={},
        slot_mappings={},
        last_hidden_states=aux,
        aux_hidden_states=[aux],
        num_sampled=torch.tensor([1, 1], device=device),
        num_rejected=torch.tensor([0, 2], device=device),
        last_sampled=torch.tensor([77, 88, 0, 0], device=device),
        next_prefill_tokens=torch.zeros(4, device=device),
        temperature=temperature,
        seeds=seeds,
    )
    proposer.propose(**arguments)
    assert combine_rows == [129]
    expected_pool = torch.full_like(context_pool, -7)
    expected_pool[138] = aux[0] + 3 + 10
    expected_pool[72 * 128 : 72 * 128 + 126] = (
        aux[3969:4095] + 3 + torch.arange(3968, 4094, device=device)[:, None]
    )
    torch.testing.assert_close(context_pool, expected_pool)
    assert buffers.positions[:4].tolist() == [11, 12, 4094, 4095]
    proposer.propose(**arguments, context_kv_is_restored=True)
    assert combine_rows == [129]
    torch.testing.assert_close(context_pool, expected_pool)


@cuda
def test_ced_policy_staging_does_not_synchronize_gpu():
    state, starts, seq = _state([1, 4096])
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        state.stage(starts, seq, [1, 4096], [False, False], [0, 0])
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    assert state.counts.tolist() == [129, 2]


@cuda
def test_streaming_update_preserves_private_decoder_history(monkeypatch):
    from vllm.models.deepseek_v4_1.nvidia.model_state import DeepseekV41ModelState
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    monkeypatch.setattr(DefaultModelState, "add_request", lambda *args: None)
    monkeypatch.setattr(DefaultModelState, "remove_request", lambda *args: None)
    state = DeepseekV41ModelState.__new__(DeepseekV41ModelState)
    state.ced_state, starts, seq = _state([1], [1025])
    state._ced_prompt_logprobs = set()
    state._ced_prefix_start = {"stream": 256}
    batch = SimpleNamespace(
        num_reqs=1,
        req_ids=["stream"],
        query_start_loc=starts,
        seq_lens=seq,
        num_scheduled_tokens=np.array([1]),
    )
    updated = SimpleNamespace(
        req_id="stream", num_computed_tokens=1024, sampling_params=None
    )
    state.prepare_streaming_update("stream")
    state.remove_request("stream")
    state.add_request(0, updated)
    state._stage_ced(batch)
    assert state.ced_state.replay_start[0].item() == 256
    # A genuinely new incarnation must not retain that older history bound.
    state.remove_request("stream")
    state.add_request(0, updated)
    state._stage_ced(batch)
    assert state.ced_state.replay_start[0].item() == 1024


@cuda
def test_large_full_capture_decodes_small_live_prefix(monkeypatch):
    from vllm.config.compilation import CUDAGraphMode
    from vllm.models.deepseek_v4_1.nvidia.model_state import DeepseekV41ModelState
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    monkeypatch.setattr(DefaultModelState, "prepare_dummy_inputs", lambda *args: {})
    monkeypatch.setattr(DefaultModelState, "prepare_attn", lambda *args: {})
    state = DeepseekV41ModelState.__new__(DeepseekV41ModelState)
    state.ced_state, starts, seq = _state([256])
    state._ced_prompt_logprobs = set()
    state._ced_prefix_start = {}
    state.lookback_token_ids = None
    state.disk_engram_models = ()
    state.device = torch.device("cuda")
    dummy_inputs = state.prepare_dummy_inputs(1, 256)
    batch = SimpleNamespace(
        num_reqs=1,
        req_ids=["request"],
        query_start_loc=starts,
        seq_lens=seq,
        num_scheduled_tokens=np.array([256]),
    )
    state.prepare_attn(batch, CUDAGraphMode.FULL, (), None, [], None, True)
    source = torch.ones((256, 2), device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = source + 1
            indices = dummy_inputs["ced_indices"]
            if indices is not None:
                result = scatter_rows(gather_rows(result, indices), indices, 256)
        starts.copy_(torch.tensor([0, 2], dtype=torch.int32, device="cuda"))
        seq.fill_(2)
        batch.num_scheduled_tokens[:] = 2
        state._stage_ced(batch)
        source.fill_(3)
        graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(result[:2], torch.full_like(result[:2], 4))


@cuda
def test_startup_warmup_covers_live_compaction_without_jit(monkeypatch):
    from collections import defaultdict

    from triton import knobs

    from vllm.models.deepseek_v4_1 import ced

    # Isolate compilation state so earlier tests cannot hide a missing warmup.
    for kernel in (ced._decoder_requests, ced._decoder_indices, ced._copy_rows):
        monkeypatch.setattr(
            kernel, "device_caches", defaultdict(kernel.device_caches.default_factory)
        )
    state = CEDState(4, 8192, (128, 256, 384, 512, 8192), "cuda")
    state.warmup(hidden_size=8, hc_mult=4)

    def forbid_compile(**kwargs):
        raise AssertionError(
            f"CED compiled after warmup: {getattr(kwargs.get('fn'), 'name', 'unknown')}"
        )

    monkeypatch.setattr(knobs.runtime, "jit_post_compile_hook", forbid_compile)
    for lengths in ([129], [4096, 1], [1024, 1024, 1], [1024] * 4):
        starts_cpu = np.r_[0, np.cumsum(lengths)]
        starts = torch.tensor(starts_cpu, dtype=torch.int32, device="cuda")
        seq = torch.tensor(lengths, dtype=torch.int32, device="cuda")
        state.stage(starts, seq, lengths, [False] * len(lengths), [0] * len(lengths))
        indices = state.get_indices()
        expected = torch.tensor(
            [
                row
                for begin, end in zip(starts_cpu[:-1], starts_cpu[1:])
                for row in range(max(begin, end - 128), end)
            ],
            dtype=torch.int64,
            device="cuda",
        )
        torch.testing.assert_close(indices, expected)
        source = torch.ones((sum(lengths), 8), dtype=torch.bfloat16, device="cuda")
        compact = gather_rows(source, indices)
        result = scatter_rows(compact, indices, source.shape[0])
        expected_rows = torch.zeros_like(source)
        expected_rows[expected] = 1
        torch.testing.assert_close(result, expected_rows)
