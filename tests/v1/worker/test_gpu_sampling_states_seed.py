# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample import states


class _HostBackedTensor:
    def __init__(self, size: int, dtype: torch.dtype):
        self.cpu = torch.zeros(size, dtype=dtype)
        self.np = self.cpu.numpy()
        self.gpu = self.cpu

    def copy_to_uva(self, n: int | None = None) -> torch.Tensor:
        return self.gpu[:n] if n is not None else self.gpu


def test_fallback_seeds_do_not_depend_on_global_numpy_rng(monkeypatch) -> None:
    monkeypatch.setattr(states, "UvaBackedTensor", _HostBackedTensor)
    rank0 = states.SamplingStates(4, 128, seed=17)

    np.random.seed(1234)
    np.random.random(1000)
    rank1 = states.SamplingStates(4, 128, seed=17)

    params = SamplingParams(seed=None)
    for req_idx in range(4):
        rank0.add_request(req_idx, params)
        np.random.random(req_idx + 1)
        rank1.add_request(req_idx, params)

    np.testing.assert_array_equal(rank0.seeds.np, rank1.seeds.np)
