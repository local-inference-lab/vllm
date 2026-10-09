# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.v1.worker.gpu.spec_decode.dflash.speculator as dflash_speculator
from vllm.config import (
    AttentionConfig,
    CacheConfig,
    ParallelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.models.qwen3_dflash import DFlashAttention
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.selector import _cached_get_attn_backend
from vllm.v1.kv_cache_interface import SlidingWindowSpec


def test_dflash_selects_local_triton_without_relaxing_target_dcp():
    """A replicated draft selects local attention; a sharded target cannot."""
    config = VllmConfig(
        attention_config=AttentionConfig(backend=AttentionBackendEnum.TRITON_ATTN),
        cache_config=CacheConfig(block_size=16),
        parallel_config=ParallelConfig(
            tensor_parallel_size=16,
            decode_context_parallel_size=16,
            distributed_executor_backend="mp",
        ),
    )
    _cached_get_attn_backend.cache_clear()
    with set_current_vllm_config(config):
        draft = DFlashAttention(
            2,
            128,
            128**-0.5,
            num_kv_heads=1,
            cache_config=config.cache_config,
            prefix="draft",
        )
        assert draft.get_attn_backend().get_name() == "TRITON_ATTN"
        spec = draft.get_kv_cache_spec(config)
        assert spec is not None
        assert not spec.dcp_sharded
        with pytest.raises(ValueError, match="DCP not supported"):
            Attention(
                2,
                128,
                128**-0.5,
                num_kv_heads=1,
                cache_config=config.cache_config,
                prefix="target",
            )
    assert config.parallel_config.decode_context_parallel_size == 16


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
    assert spec.dcp_sharded is False


@pytest.mark.parametrize("external_context", [False, True])
@pytest.mark.parametrize("context_kv_is_restored", [False, True])
@pytest.mark.parametrize("use_full_graph", [False, True])
def test_dflash_uses_draft_group_dcp_slot_parameters(
    monkeypatch, context_kv_is_restored, use_full_graph, external_context
):
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
                cg_mode=(CUDAGraphMode.FULL if use_full_graph else CUDAGraphMode.NONE),
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
    context_writes = []
    query_runs = []
    model = SimpleNamespace(
        precompute_and_store_context_kv=lambda *args: context_writes.append(args)
    )
    speculator = object.__new__(dflash_speculator.DFlashSpeculator)
    speculator.__dict__.update(
        num_query_per_req=1,
        num_speculative_steps=1,
        max_model_len=128,
        max_num_reqs=1,
        max_num_tokens=1,
        hidden_states=torch.full((1, 1), -99.0),
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
        _external_context_preparation=external_context,
        _context_preparer=None,
        parallel_drafting_token_id=0,
        sample_from_anchor=False,
        model=model,
        query_cudagraph_manager=SimpleNamespace(
            run_fullgraph=lambda *_args: query_runs.append("graph")
        ),
        dp_size=1,
        dp_rank=0,
        pcp_manager=None,
        _group_causal=False,
        kv_cache_config=SimpleNamespace(),
        draft_tokens=torch.zeros((1, 1), dtype=torch.int64),
        _build_uniform_attn_metadata=lambda **_kwargs: {},
        _prepare_eplb_forward=lambda *_args: None,
        _generate_draft=lambda *_args, **_kwargs: query_runs.append("eager"),
    )

    def replay(_desc):
        if not external_context:
            speculator._precompute_context_kv(0, 1)
        query_runs.append("graph")

    assert speculator.query_cudagraph_manager is not None
    monkeypatch.setattr(speculator.query_cudagraph_manager, "run_fullgraph", replay)
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
        last_hidden_states=torch.full((1, 1), 5.0),
        aux_hidden_states=None,
        num_sampled=one_i32,
        num_rejected=one_i32,
        last_sampled=one_i64,
        next_prefill_tokens=one_i64,
        temperature=one_f32,
        seeds=one_i64,
        context_kv_is_restored=context_kv_is_restored,
    )

    assert cp_args == [(2, 4, 8)]
    active_writes = [args for args in context_writes if (args[2] != -1).any()]
    assert len(active_writes) == (0 if context_kv_is_restored else 1)
    if context_kv_is_restored and use_full_graph and not external_context:
        assert len(context_writes) == 1
        assert (context_writes[0][2] == -1).all()
    else:
        assert len(context_writes) == len(active_writes)
    assert speculator.hidden_states.item() == (-99.0 if context_kv_is_restored else 5.0)
    assert query_runs == ["graph" if use_full_graph else "eager"]


def test_replicated_draft_metadata_uses_full_sequence_lengths(monkeypatch):
    seq_lens = torch.tensor([128, 64], dtype=torch.int32)
    captured = []

    def capture_metadata(_self, *_args, **kwargs):
        captured.append(kwargs["dcp_local_seq_lens"])
        return {}

    monkeypatch.setattr(
        dflash_speculator.DraftModelSpeculator,
        "_build_attn_metadata",
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

    result = speculator._build_attn_metadata(
        num_reqs=2,
        batch_desc=SimpleNamespace(num_reqs=2, num_tokens=8),
        query_start_loc_np=np.array([0, 4, 8], dtype=np.int32),
        seq_lens_cpu_upper_bound=seq_lens,
        step=0,
    )

    assert result == {}
    assert len(captured) == 1
    assert captured[0] is seq_lens


@pytest.mark.parametrize("hook", [None, "streaming", "context_graph", "ced"])
def test_context_preparation_owner_is_selected_from_model_capabilities(hook):
    # Generic Qwen drafts expose the batch predicate even without streaming.
    model = SimpleNamespace(is_streamed_context_states=lambda states: False)
    model_state = SimpleNamespace()
    if hook == "streaming":
        model.uses_external_context_preparation = True
    elif hook == "context_graph":
        model.capture_context_preparation = lambda *args: None
    elif hook == "ced":
        model_state.get_ced_indices = lambda: None
    assert dflash_speculator._needs_external_context_preparation(
        model, model_state
    ) is (hook is not None)


@pytest.mark.parametrize("external_context", [False, True])
def test_query_graph_context_owner_is_identical_for_warmup_capture_and_replay(
    monkeypatch, external_context
):
    from vllm.v1.worker.gpu.spec_decode.dflash import cudagraph

    manager = object.__new__(cudagraph.DFlashCudaGraphManager)
    manager.max_num_reqs = 2
    manager.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=1)
    )
    events = []
    desc = SimpleNamespace(num_tokens=4, num_reqs=2, cg_mode=CUDAGraphMode.FULL)
    monkeypatch.setattr(
        cudagraph, "_prepare_dflash_inputs_to_capture", lambda *a, **kw: (None, None)
    )

    def capture(_self, create_forward_fn, progress_bar_desc):
        for warmup in (True, False):
            forward = create_forward_fn(desc, warmup)
            forward(desc.cg_mode)
        forward(desc.cg_mode)

    monkeypatch.setattr(cudagraph.CudaGraphManager, "capture", capture)
    manager.capture(
        lambda *args: events.append("query"),
        None,
        None,
        [],
        None,
        128,
        True,
        None if external_context else lambda num_reqs: events.append("context"),
    )
    assert events == (["query"] if external_context else ["context", "query"]) * 3
