# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import torch

import vllm.v1.worker.gpu.spec_decode.dflash.speculator as dflash_speculator
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.models.qwen3_dflash import DFlashAttention
from vllm.v1.attention.backend import AttentionType
from vllm.v1.kv_cache_interface import SlidingWindowSpec


def test_dflash_sliding_window_cache_uses_aligned_block_size():
    attention = SimpleNamespace(
        sliding_window=2048,
        attn_type=AttentionType.DECODER,
        num_kv_heads=1,
        head_size=576,
        head_size_v=576,
        kv_cache_torch_dtype=torch.bfloat16,
        kv_cache_dtype="auto",
    )
    padded_page_size = 6 * 1024 * 1024
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=2304,
            skip_page_size_padded=padded_page_size,
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )

    spec = DFlashAttention.get_kv_cache_spec(attention, config)

    assert isinstance(spec, SlidingWindowSpec)
    assert spec.block_size == 2304
    assert spec.sliding_window == 2048
    assert spec.extra_retained_tokens == 2048
    assert spec.page_size_padded == padded_page_size


def test_dflash_sliding_window_cache_is_replicated_under_dcp():
    attention = SimpleNamespace(
        sliding_window=2048,
        attn_type=AttentionType.DECODER,
        num_kv_heads=1,
        head_size=128,
        head_size_v=128,
        kv_cache_torch_dtype=torch.float8_e4m3fn,
        kv_cache_dtype="fp8",
    )
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16),
        parallel_config=SimpleNamespace(decode_context_parallel_size=4),
    )

    spec = DFlashAttention.get_kv_cache_spec(attention, config)

    assert isinstance(spec, SlidingWindowSpec)
    assert spec.sliding_window == 2048
    assert spec.extra_retained_tokens == 2048
    assert spec.dcp_replicated is True


def test_dflash_uses_draft_group_dcp_slot_parameters(monkeypatch):
    """A sharded DSpark-style draft group must retain the real DCP mapping."""
    cp_args = []

    def capture_prepare_args(*args):
        cp_args.append(args[18:21])

    monkeypatch.setattr(
        dflash_speculator, "prepare_dflash_inputs", capture_prepare_args
    )
    monkeypatch.setattr(
        dflash_speculator,
        "dispatch_cg_and_sync_dp",
        lambda *_args, **_kwargs: (
            SimpleNamespace(
                num_reqs=1,
                num_tokens=1,
                cg_mode=CUDAGraphMode.NONE,
            ),
            None,
        ),
    )
    monkeypatch.setattr(
        dflash_speculator,
        "build_slot_mappings_by_layer",
        lambda *_args, **_kwargs: {},
    )

    block_tables = SimpleNamespace(
        slot_mappings=torch.zeros((1, 1), dtype=torch.int64),
        input_block_tables=[torch.zeros((1, 1), dtype=torch.int32)],
        kernel_block_sizes=[16],
        get_group_cp_parameters=lambda _gid: (2, 4, 8),
    )
    model = SimpleNamespace(precompute_and_store_context_kv=lambda *_args: None)
    speculator = SimpleNamespace(
        num_query_per_req=1,
        num_speculative_steps=1,
        max_model_len=128,
        max_num_reqs=1,
        max_num_tokens=1,
        hidden_states=torch.zeros((1, 1)),
        context_positions=torch.zeros(1, dtype=torch.int64),
        sample_indices=torch.zeros(1, dtype=torch.int64),
        sample_pos=torch.zeros(1, dtype=torch.int64),
        sample_idx_mapping=torch.zeros(1, dtype=torch.int32),
        temperature=torch.zeros(1),
        seeds=torch.zeros(1, dtype=torch.int64),
        input_buffers=SimpleNamespace(),
        block_tables=block_tables,
        draft_kv_cache_group_id=0,
        draft_kv_cache_group_ids=[0],
        _context_slot_mappings=torch.zeros((1, 1), dtype=torch.int64),
        _layer_group_idx=None,
        parallel_drafting_token_id=0,
        sample_from_anchor=False,
        model=model,
        query_cudagraph_manager=None,
        dp_size=1,
        dp_rank=0,
        _group_causal=False,
        kv_cache_config=SimpleNamespace(),
        draft_tokens=torch.zeros((1, 1), dtype=torch.int64),
        _build_draft_attn_metadata=lambda **_kwargs: {},
        _prepare_eplb_forward=lambda *_args: None,
        _generate_draft=lambda *_args, **_kwargs: None,
    )
    input_batch = SimpleNamespace(
        num_reqs=1,
        num_tokens=1,
        seq_lens_cpu_upper_bound=torch.tensor([1], dtype=torch.int32),
    )
    one_i32 = torch.zeros(1, dtype=torch.int32)
    one_i64 = torch.zeros(1, dtype=torch.int64)
    one_f32 = torch.zeros(1)

    dflash_speculator.DFlashSpeculator.propose(
        speculator,
        input_batch=input_batch,
        attn_metadata={},
        slot_mappings={},
        last_hidden_states=torch.zeros((1, 1)),
        aux_hidden_states=None,
        num_sampled=one_i32,
        num_rejected=one_i32,
        last_sampled=one_i64,
        next_prefill_tokens=one_i64,
        temperature=one_f32,
        seeds=one_i64,
    )

    assert cp_args == [(2, 4, 8)]


def test_replicated_draft_metadata_uses_full_sequence_lengths(monkeypatch):
    seq_lens = torch.tensor([128, 64], dtype=torch.int32)
    captured = []

    def capture_metadata(_self, *_args, **kwargs):
        captured.append(kwargs["dcp_local_seq_lens"])
        return {}

    monkeypatch.setattr(
        dflash_speculator.DraftModelSpeculator,
        "_build_draft_attn_metadata",
        capture_metadata,
    )
    speculator = object.__new__(dflash_speculator.DFlashSpeculator)
    object.__setattr__(speculator, "draft_attn_layer_names", ["draft"])
    object.__setattr__(speculator, "draft_cp_size", 1)
    object.__setattr__(speculator, "block_tables", SimpleNamespace(cp_size=4))
    object.__setattr__(
        speculator,
        "input_buffers",
        SimpleNamespace(
            seq_lens=seq_lens,
            dcp_local_seq_lens=torch.empty_like(seq_lens),
        ),
    )
    object.__setattr__(speculator, "num_query_per_req", 4)

    result = speculator._build_draft_attn_metadata(
        num_reqs=2,
        num_reqs_padded=2,
        num_tokens_padded=8,
        seq_lens_cpu_upper_bound=seq_lens,
        step=0,
    )

    assert result == {}
    assert len(captured) == 1
    assert captured[0] is seq_lens
