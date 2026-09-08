# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator


def test_dspark_generate_draft_accepts_dflash_capture_contract():
    head_hidden = torch.randn(10, 8)
    speculator = SimpleNamespace(
        num_query_per_req=5,
        capacity_activation_batch_size=1,
        _markov_outside_cudagraph=False,
        _speculative_steps_for_query_len=Mock(return_value=5),
        _run_model=Mock(return_value=head_hidden),
        _sample_sequential=Mock(),
    )

    DSparkSpeculator._generate_draft(
        speculator,
        num_reqs=2,
        num_tokens_padded=10,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        is_profile=True,
        num_query_per_req=5,
        capture_only=True,
    )

    speculator._sample_sequential.assert_called_once_with(
        2,
        head_hidden,
        5,
        5,
        is_profile=True,
        use_capacity=True,
    )
