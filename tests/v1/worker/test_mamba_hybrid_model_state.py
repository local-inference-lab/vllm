# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize(("num_sampled", "expected_value"), [(0, 1), (3, 3)])
def test_postprocess_state_scalar_with_int32_mapping(
    num_sampled: int, expected_value: int
) -> None:
    state = object.__new__(MambaHybridModelState)
    state.num_accepted_tokens_gpu = torch.full(
        (4,), 9, dtype=torch.int32, device="cuda"
    )
    state._align_mode = False
    state._mamba_ctx = None
    idx_mapping = torch.tensor([2, -1, 0], dtype=torch.int32, device="cuda")

    state.postprocess_state(idx_mapping, num_sampled)

    expected = torch.tensor(
        [expected_value, 9, expected_value, 9], dtype=torch.int32, device="cuda"
    )
    torch.testing.assert_close(state.num_accepted_tokens_gpu, expected)


@pytest.mark.parametrize(
    ("num_computed_tokens", "expected_state_index"),
    [(0, -1), (110_592, 8), (110_593, 9)],
)
def test_prefix_hit_uses_mamba_checkpoint_cadence(
    num_computed_tokens: int, expected_state_index: int
) -> None:
    """A resumed request indexes recurrent checkpoints, not attention pages."""
    state = object.__new__(MambaHybridModelState)
    state.rope_state = None
    state._align_mode = True
    state.cache_config = SimpleNamespace(block_size=768, mamba_block_size=12_288)
    state._mamba_block_size = 12_288
    state._mamba_state_idx_gpu = torch.zeros(1, dtype=torch.int32)
    state.num_accepted_tokens_gpu = torch.full((1,), 9, dtype=torch.int32)
    request = NewRequestData(
        req_id="prefix-hit",
        prompt_token_ids=[],
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=(),
        num_computed_tokens=num_computed_tokens,
        lora_request=None,
    )

    state.add_request(0, request)

    assert state._mamba_state_idx_gpu.item() == expected_state_index
    assert state.num_accepted_tokens_gpu.item() == 1
