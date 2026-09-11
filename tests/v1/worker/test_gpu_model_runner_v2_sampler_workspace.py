# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.v1.sample.ops import topk_topp_triton
from vllm.v1.worker.gpu import model_runner as mrv2


def test_top_k_top_p_workspace_reserves_maximum_program_count(monkeypatch):
    cache = topk_topp_triton._TRITON_BUFFER_CACHE
    saved_cache = dict(cache)
    cache.clear()
    monkeypatch.setattr(topk_topp_triton, "num_compute_units", lambda _index: 188)
    device = torch.device("cpu")

    try:
        reserved = topk_topp_triton.reserve_top_k_top_p_workspace(
            device=device,
            dtype=torch.float32,
            vocab_size=128,
            max_batch_size=256,
        )
        buffer = cache[(device, torch.float32, 128)]
        assert buffer.shape == (188, 128)
        assert reserved == 188 * 128 * 4

        data_ptr = buffer.data_ptr()
        topk_topp_triton.reserve_top_k_top_p_workspace(
            device=device,
            dtype=torch.float32,
            vocab_size=128,
            max_batch_size=64,
        )
        assert cache[(device, torch.float32, 128)].data_ptr() == data_ptr
    finally:
        cache.clear()
        cache.update(saved_cache)


def test_reserve_sampler_workspace_covers_expanded_decode_logits(monkeypatch):
    calls: list[dict[str, object]] = []
    runner = mrv2.GPUModelRunner.__new__(mrv2.GPUModelRunner)
    runner.sampler = SimpleNamespace()
    runner.max_num_reqs = 64
    runner.max_num_tokens = 192
    runner.decode_query_len = 4
    runner.device = torch.device("cuda:0")
    runner.vocab_size = 202048

    def reserve(**kwargs):
        calls.append(kwargs)
        return 123

    monkeypatch.setattr(mrv2, "reserve_top_k_top_p_workspace", reserve)

    reserved = runner.reserve_sampler_workspace()

    assert reserved == 123
    assert calls == [
        {
            "device": torch.device("cuda:0"),
            "vocab_size": 202048,
            "max_batch_size": 192,
        }
    ]


def test_reserve_sampler_workspace_skips_non_generative_runner(monkeypatch):
    runner = mrv2.GPUModelRunner.__new__(mrv2.GPUModelRunner)
    runner.sampler = None
    reserve = SimpleNamespace()
    monkeypatch.setattr(mrv2, "reserve_top_k_top_p_workspace", reserve)

    assert runner.reserve_sampler_workspace() == 0
