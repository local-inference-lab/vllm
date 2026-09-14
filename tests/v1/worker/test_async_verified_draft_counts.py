# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu.async_utils import AsyncOutput
from vllm.v1.worker.gpu.sample.output import SamplerOutput


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA streams")
@pytest.mark.parametrize(
    "deferred", [True, False], ids=["deferred-delivery", "cuda-copy"]
)
def test_verified_counts_belong_to_output_step(monkeypatch, deferred):
    """A subsequent verifier step may reuse capacity storage before CPU delivery."""
    main_stream = torch.cuda.current_stream()
    copy_stream = torch.cuda.Stream()
    capacities = torch.tensor([3, 1], dtype=torch.int32, device="cuda")
    sampled_ids = torch.tensor([[7, 8, 9, 10], [11, 12, -1, -1]], device="cuda")
    num_sampled = torch.tensor([4, 2], dtype=torch.int32, device="cuda")
    pending_copies = []

    def defer_copy(source):
        # Delay CPU delivery deterministically, without relying on GPU timing.
        # A reference is deliberately retained to test storage ownership, not
        # allocator lifetime; production enqueues these copies on copy_stream.
        destination = torch.empty_like(source, device="cpu").numpy()
        pending_copies.append((source, destination))
        return destination

    if deferred:
        monkeypatch.setattr(
            "vllm.v1.worker.gpu.async_utils.async_copy_to_np", defer_copy
        )
    else:
        torch.accelerator.synchronize()
        with torch.cuda.stream(copy_stream):
            torch.cuda._sleep(100_000_000)
    output = AsyncOutput(
        model_runner_output=ModelRunnerOutput(
            req_ids=["a", "b"],
            req_id_to_index={"a": 0, "b": 1},
            sampled_token_ids=[],
            logprobs=None,
            prompt_logprobs_dict={},
        ),
        sampler_output=SamplerOutput(sampled_ids, None, None, num_sampled),
        num_sampled_tokens=num_sampled,
        main_stream=main_stream,
        copy_stream=copy_stream,
        num_verified_draft_tokens=capacities,
    )
    capacities.copy_(torch.tensor([0, 2], dtype=torch.int32, device="cuda"))
    torch.accelerator.synchronize()
    for source, destination in pending_copies:
        destination[:] = source.cpu().numpy()
    result = output.get_output()
    assert result.sampled_token_ids == [[7, 8, 9, 10], [11, 12]]
    assert result.num_verified_draft_tokens == [3, 1]
