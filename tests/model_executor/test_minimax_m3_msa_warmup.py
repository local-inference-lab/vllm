# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import vllm.model_executor.warmup.minimax_m3_msa_warmup as msa_warmup


def test_minimax_m3_triton_msa_warmup_supports_blackwell(monkeypatch) -> None:
    platform = SimpleNamespace(
        is_cuda=lambda: True,
        is_device_capability_family=lambda family: family == 120,
    )

    monkeypatch.setattr(msa_warmup, "current_platform", platform)
    monkeypatch.setattr(msa_warmup.envs, "VLLM_USE_B12X_MINIMAX_M3_MSA", False)

    assert msa_warmup._supports_minimax_m3_msa_warmup()


def test_minimax_m3_b12x_msa_warmup_stays_blackwell_only(monkeypatch) -> None:
    platform = SimpleNamespace(
        is_cuda=lambda: True,
        is_device_capability_family=lambda family: family == 100,
    )

    monkeypatch.setattr(msa_warmup, "current_platform", platform)
    monkeypatch.setattr(msa_warmup.envs, "VLLM_USE_B12X_MINIMAX_M3_MSA", True)

    assert not msa_warmup._supports_minimax_m3_msa_warmup()


def test_minimax_m3_warmup_covers_dense_b12x_attention(monkeypatch) -> None:
    class FakeSparseAttention:
        pass

    class FakeDenseAttention:
        def get_attn_backend(self):
            return SimpleNamespace(get_name=lambda: "B12X_ATTN")

    dense = FakeDenseAttention()
    model = SimpleNamespace(modules=lambda: iter([FakeSparseAttention(), dense]))
    dummy_runs = []

    worker = SimpleNamespace(
        get_model=lambda: model,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=2048, max_num_seqs=4),
        model_config=SimpleNamespace(max_model_len=256000),
        model_runner=SimpleNamespace(
            _dummy_run=lambda **kwargs: dummy_runs.append(kwargs),
        ),
    )

    monkeypatch.setattr(msa_warmup, "Attention", FakeDenseAttention)
    monkeypatch.setattr(msa_warmup, "MiniMaxM3SparseAttention", FakeSparseAttention)
    monkeypatch.setattr(msa_warmup, "_supports_minimax_m3_msa_warmup", lambda: True)
    monkeypatch.setattr(msa_warmup, "_warmup_slot_mapping", lambda *args: None)

    msa_warmup.minimax_m3_msa_warmup(worker)

    assert dummy_runs == [
        {
            "num_tokens": 2048,
            "skip_eplb": True,
            "is_profile": True,
            "force_attention": True,
            "include_mm_inputs": False,
            "single_request_prefill": True,
        },
        {
            "num_tokens": 2048,
            "skip_eplb": True,
            "is_profile": True,
            "force_attention": True,
            "create_mixed_batch": True,
            "include_mm_inputs": False,
        },
        {
            "num_tokens": 4,
            "skip_eplb": True,
            "is_profile": True,
            "force_attention": True,
            "uniform_decode": True,
            "profile_seq_lens": 2048,
            "include_mm_inputs": False,
        },
        {
            "num_tokens": 1,
            "skip_eplb": True,
            "is_profile": True,
            "force_attention": True,
            "uniform_decode": True,
            "profile_seq_lens": 2048,
            "include_mm_inputs": False,
        },
    ]
