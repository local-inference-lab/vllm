# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

import vllm.v1.worker.cp_utils as cp_utils
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
from vllm.v1.worker.cp_utils import should_skip_dcp_context_attention


def test_skip_gate_only_for_zero_context():
    assert should_skip_dcp_context_attention(torch.zeros(3, dtype=torch.int32))
    assert not should_skip_dcp_context_attention(
        torch.tensor([0, 5, 0], dtype=torch.int32)
    )


def test_replicated_draft_attention_executes_as_local_dcp(monkeypatch):
    layer_impl = SimpleNamespace(
        supports_mtp_with_cp_non_trivial_interleave_size=False,
        need_to_return_lse_for_decode=False,
        dcp_world_size=4,
        dcp_rank=2,
        total_cp_world_size=4,
        total_cp_rank=2,
    )
    layer = SimpleNamespace(
        impl=layer_impl,
        get_kv_cache_spec=lambda _config: SimpleNamespace(dcp_replicated=True),
    )
    monkeypatch.setattr(
        cp_utils,
        "get_layers_from_vllm_config",
        lambda *_args, **_kwargs: {"draft": layer},
    )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=4,
            cp_kv_cache_interleave_size=4,
        ),
        speculative_config=SimpleNamespace(method="dflash"),
    )

    cp_utils.check_attention_cp_compatibility(config)

    assert layer_impl.dcp_world_size == 1
    assert layer_impl.dcp_rank == 0
    assert layer_impl.total_cp_world_size == 1
    assert layer_impl.total_cp_rank == 0


@pytest.mark.parametrize(
    "dcp_world_size,interleave_size,context_len",
    [(2, 16, 10), (4, 16, 10), (8, 16, 10), (4, 1, 2)],
)
def test_skip_gate_rank_invariant_with_divergent_local_context(
    dcp_world_size: int, interleave_size: int, context_len: int
):
    """Contexts shorter than a full interleave round land entirely on a
    subset of DCP ranks, so the per-rank local context lengths diverge:
    some ranks hold zero local context while others hold all of it. Ranks
    with zero local context must still take the collective (non-skip) path,
    otherwise the query all-gather in _forward_with_dcp deadlocks across
    ranks. The skip gate must therefore depend only on the rank-invariant
    global context lengths, never on get_dcp_local_seq_lens output.
    """
    context_kv_lens = torch.tensor([context_len], dtype=torch.int32)
    local_maxes = [
        int(
            get_dcp_local_seq_lens(
                context_kv_lens, dcp_world_size, rank, interleave_size
            ).max()
        )
        for rank in range(dcp_world_size)
    ]
    # Precondition: the local view diverges across ranks.
    assert 0 in local_maxes
    assert max(local_maxes) > 0
    # The batch still has context globally, so no rank may skip.
    assert not should_skip_dcp_context_attention(context_kv_lens)
