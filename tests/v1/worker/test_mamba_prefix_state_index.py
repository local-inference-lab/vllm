# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Use recurrent block units when resuming cached hybrid requests."""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState


@pytest.mark.parametrize("recurrent_block_size", [256, 2048, 4096])
@pytest.mark.parametrize("computed_tokens", [0, 256, 257, 4096, 7680, 7936, 8192])
def test_resumed_state_index_uses_recurrent_blocks(
    monkeypatch, recurrent_block_size, computed_tokens
):
    monkeypatch.setattr(DefaultModelState, "add_request", lambda *args: None)
    state = MambaHybridModelState.__new__(MambaHybridModelState)
    state.cache_config = SimpleNamespace(
        block_size=2048, mamba_block_size=recurrent_block_size
    )
    state._align_mode = True
    state.num_accepted_tokens_gpu = torch.full((1,), 4, dtype=torch.int32)
    state._mamba_state_idx_gpu = torch.full((1,), 99, dtype=torch.int32)

    state.add_request(0, SimpleNamespace(num_computed_tokens=computed_tokens))

    assert state.num_accepted_tokens_gpu[0].item() == 1
    expected_column = (computed_tokens - 1) // recurrent_block_size
    assert state._mamba_state_idx_gpu[0].item() == expected_column
