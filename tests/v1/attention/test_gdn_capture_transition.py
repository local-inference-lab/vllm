# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep graph-captured speculative metadata valid when requests are padded."""

from types import SimpleNamespace as NS

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import MambaSpec


def _common(query_lens, padded_tokens=32):
    starts = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(0).to(torch.int32)
    seq_lens = torch.tensor([64 if n else 0 for n in query_lens], dtype=torch.int32)
    return CommonAttentionMetadata(
        query_start_loc=starts,
        query_start_loc_cpu=starts.clone(),
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens.clone(),
        num_reqs=len(query_lens),
        num_actual_tokens=padded_tokens,
        max_query_len=max(query_lens),
        max_seq_len=64,
        block_table_tensor=torch.arange(len(query_lens) * 4, dtype=torch.int32).reshape(
            len(query_lens), 4
        ),
        slot_mapping=torch.arange(padded_tokens, dtype=torch.int64),
        is_prefilling=torch.zeros(len(query_lens), dtype=torch.bool),
        causal=True,
    )


@pytest.mark.parametrize("fastpath", [False, True])
@pytest.mark.parametrize("active_reqs", [1, 5, 8])
@pytest.mark.parametrize("alias_accepted", [False, True])
@pytest.mark.parametrize("consumer", ["build", "update_block_table"])
def test_padded_replay_updates_captured_spec_buffers(
    monkeypatch, fastpath, active_reqs, consumer, alias_accepted
):
    monkeypatch.setenv("VLLM_GDN_SPEC_DECODE_METADATA_FASTPATH", str(int(fastpath)))
    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn._resolve_gdn_prefill_backend",
        lambda config: ("triton", "triton"),
    )
    monkeypatch.setattr(
        "vllm.v1.attention.backends.gdn_attn.async_tensor_h2d",
        lambda data, dtype=None, device="cpu", **kwargs: torch.as_tensor(
            data, dtype=dtype, device=device
        ),
    )
    config = NS(
        compilation_config=NS(
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            max_cudagraph_capture_size=32,
        ),
        speculative_config=NS(num_speculative_tokens=3, parallel_drafting=False),
        scheduler_config=NS(max_num_seqs=8),
        parallel_config=NS(decode_context_parallel_size=1),
        cache_config=NS(mamba_cache_mode="align"),
    )
    builder = GDNAttentionMetadataBuilder(
        MambaSpec(
            block_size=16,
            shapes=((16, 64),),
            dtypes=(torch.float16,),
            mamba_cache_mode="align",
        ),
        ["layer.0"],
        config,
        torch.device("cpu"),
    )
    builder.mamba_aligned_state_indices = torch.arange(32, dtype=torch.int32).reshape(
        8, 4
    )
    builder.mamba_spec_accepted_tokens = torch.ones(8, dtype=torch.int32)
    captured = builder.build_for_cudagraph_capture(_common([4] * 8))
    other = GDNAttentionMetadataBuilder(
        builder.kv_cache_spec, ["layer.1"], config, torch.device("cpu")
    )
    other.mamba_aligned_state_indices = builder.mamba_aligned_state_indices.clone() + 64
    other.mamba_spec_accepted_tokens = builder.mamba_spec_accepted_tokens
    captured_other = other.build_for_cudagraph_capture(_common([4] * 8))
    fields = [
        "spec_state_indices_tensor",
        "spec_query_start_loc",
        "spec_sequence_masks",
        "num_accepted_tokens",
    ]
    captured = captured_other if consumer == "update_block_table" else captured
    owner = other if consumer == "update_block_table" else builder
    pointers = {name: getattr(captured, name).data_ptr() for name in fields}
    for count in (active_reqs, 8, active_reqs):
        for current in (builder, other):
            current.mamba_aligned_state_indices.copy_(
                torch.arange(32, dtype=torch.int32).reshape(8, 4) + 100
            )
            current.mamba_aligned_state_indices[count:].fill_(NULL_BLOCK_ID)
        expected_accepted = torch.tensor([1, 2, 1, 3, 2, 1, 1, 1], dtype=torch.int32)
        expected_accepted[count:] = 1
        accepted = expected_accepted.clone()
        if alias_accepted:
            builder.mamba_spec_accepted_tokens.copy_(accepted)
            accepted = builder.mamba_spec_accepted_tokens
        runtime_common = _common([4] * count + [0] * (8 - count))
        replay = builder.build(
            0,
            runtime_common,
            accepted,
            torch.tensor([3] * count + [-1] * (8 - count), dtype=torch.int32),
        )
        if consumer == "update_block_table":
            replay = other.update_block_table(
                replay, runtime_common.block_table_tensor, None
            )
        for name in fields:
            assert getattr(replay, name).data_ptr() == pointers[name], name
            torch.testing.assert_close(getattr(captured, name), getattr(replay, name))
        torch.testing.assert_close(captured.num_accepted_tokens, expected_accepted)
        torch.testing.assert_close(
            captured.spec_query_start_loc, runtime_common.query_start_loc
        )
        torch.testing.assert_close(
            captured.spec_sequence_masks, torch.arange(8) < count
        )
        torch.testing.assert_close(
            captured.spec_state_indices_tensor, owner.mamba_aligned_state_indices
        )
