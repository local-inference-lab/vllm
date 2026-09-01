# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.v1.worker.gpu.spec_decode.dspark.remote_speculator as remote_speculator
from vllm import envs
from vllm.v1.worker.gpu.spec_decode.dspark.remote_speculator import (
    RemoteK3DSparkSpeculator,
    _anchor_positions_from_context,
    _build_valid_context_plan,
    _contiguous_draft_output,
    _decode_bfloat16_logits_frame,
    _RetainedRequestPrefix,
)


def test_build_valid_context_plan_drops_rejected_tail_rows():
    batch = SimpleNamespace(
        num_reqs=2,
        num_scheduled_tokens=np.array([4, 3], dtype=np.int32),
        num_computed_tokens_np=np.array([10, 20], dtype=np.int32),
    )

    indices, counts = _build_valid_context_plan(batch, [2, 0])

    assert indices == [0, 1, 4, 5, 6]
    assert counts == [2, 3]


def test_remote_k3_environment_registry_preserves_legacy_fallback(monkeypatch):
    monkeypatch.delenv("VLLM_K3_DRAFT_REMOTE_TIMEOUT_MS", raising=False)
    monkeypatch.setenv("VLLM_K3_DSPARK_REMOTE_TIMEOUT_MS", "1234")
    monkeypatch.setenv("VLLM_K3_DRAFT_REMOTE_ADDRESS", "tcp://127.0.0.1:9000")
    monkeypatch.setenv("VLLM_K3_DSPARK_REMOTE_ADDRESS", "tcp://127.0.0.1:9001")
    monkeypatch.setenv("VLLM_K3_DRAFT_TIMING_LOG_INTERVAL", "7")
    envs.disable_envs_cache()
    try:
        assert envs.VLLM_K3_DRAFT_REMOTE_TIMEOUT_MS == 1234
        assert envs.VLLM_K3_DSPARK_REMOTE_TIMEOUT_MS == 1234
        assert envs.VLLM_K3_DRAFT_REMOTE_ADDRESS == "tcp://127.0.0.1:9000"
        assert envs.VLLM_K3_DSPARK_REMOTE_ADDRESS == "tcp://127.0.0.1:9001"
        assert envs.VLLM_K3_DRAFT_TIMING_LOG_INTERVAL == 7
    finally:
        envs.disable_envs_cache()


def test_anchor_positions_follow_actual_valid_context_rows():
    positions = torch.tensor([24, 25, 26, 80, 81], dtype=torch.int64)

    anchors = _anchor_positions_from_context([3, 2], positions)

    assert anchors == [27, 82]


def test_remote_tokens_copy_supports_adaptive_depth():
    proxy = RemoteK3DSparkSpeculator.__new__(RemoteK3DSparkSpeculator)
    proxy.device = torch.device("cpu")
    proxy.vocab_size = 100
    proxy.draft_tokens = torch.full((3, 8), -1, dtype=torch.int64)

    proxy._copy_tokens_from_response(
        {"tokens": [[11, 12], [21, 22]]},
        active_indices=[0, 2],
        num_speculative_tokens=2,
    )

    assert proxy.draft_tokens.tolist() == [
        [11, 12, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1],
        [21, 22, -1, -1, -1, -1, -1, -1],
    ]


@pytest.mark.parametrize("invalid_token", [-1, 100, True, 1.5])
def test_remote_tokens_reject_invalid_vocab_ids(invalid_token):
    proxy = RemoteK3DSparkSpeculator.__new__(RemoteK3DSparkSpeculator)
    proxy.device = torch.device("cpu")
    proxy.vocab_size = 100
    proxy.draft_tokens = torch.full((1, 2), -1, dtype=torch.int64)

    with pytest.raises(ValueError, match="out-of-vocabulary"):
        proxy._copy_tokens_from_response(
            {"tokens": [[1, invalid_token]]},
            active_indices=[0],
            num_speculative_tokens=2,
        )


def test_failed_proposal_discards_all_uncertain_local_state():
    proxy = RemoteK3DSparkSpeculator.__new__(RemoteK3DSparkSpeculator)
    proxy.num_speculative_steps = 2
    proxy.draft_tokens = torch.full((2, 2), 7, dtype=torch.int64)
    proxy._probabilistic = False
    proxy._tp_rank = 0
    proxy._known_requests = {"first", "second", "retained"}
    proxy._disabled_requests = set()
    proxy._active_requests = {"first", "second"}
    proxy._retained_prefixes = {
        request_id: _RetainedRequestPrefix(
            token_ids=torch.arange(2),
            committed_end=2,
            context_start=0,
            serial=index,
        )
        for index, request_id in enumerate(proxy._known_requests)
    }
    broadcasts = []
    proxy._tp_group = SimpleNamespace(
        broadcast=lambda tensor, src: broadcasts.append((tensor.clone(), src))
    )

    def fail(*_args, **_kwargs):
        raise RuntimeError("transport failed")

    proxy._rank0_propose = fail
    batch = SimpleNamespace(num_reqs=2, req_ids=["first", "second"])
    empty = torch.empty(0)

    output = proxy.propose(
        batch,
        {},
        {},
        empty,
        None,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
    )

    assert output.tolist() == [[-1, -1], [-1, -1]]
    assert proxy._disabled_requests == {"first", "second"}
    assert proxy._known_requests == {"retained"}
    assert proxy._active_requests == set()
    assert set(proxy._retained_prefixes) == {"retained"}
    assert len(broadcasts) == 1


def test_remote_speculator_accepts_scheduler_selected_zero_depth():
    proxy = RemoteK3DSparkSpeculator.__new__(RemoteK3DSparkSpeculator)
    proxy.num_speculative_steps = 3
    proxy.draft_tokens = torch.full((4, 3), -1, dtype=torch.int64)
    batch = SimpleNamespace(num_reqs=2)
    empty = torch.empty(0)

    output = proxy.propose(
        batch,
        {},
        {},
        empty,
        None,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        num_speculative_tokens=0,
    )

    assert output.shape == (2, 0)
    assert output.is_contiguous()


def test_adaptive_depth_output_is_contiguous_for_tp_broadcast():
    draft_tokens = torch.arange(24, dtype=torch.int64).view(3, 8)

    output = _contiguous_draft_output(draft_tokens, 2, 3)

    assert output.is_contiguous()
    assert output.tolist() == [[0, 1, 2], [8, 9, 10]]


@pytest.mark.parametrize("rejected", [[5, 0], [-1, 0]])
def test_build_valid_context_plan_rejects_invalid_counts(rejected):
    batch = SimpleNamespace(
        num_reqs=2,
        num_scheduled_tokens=np.array([4, 3], dtype=np.int32),
        num_computed_tokens_np=np.array([0, 0], dtype=np.int32),
    )

    with pytest.raises(ValueError, match="Invalid valid-context length"):
        _build_valid_context_plan(batch, rejected)


def _make_prefix_matcher() -> RemoteK3DSparkSpeculator:
    proxy = RemoteK3DSparkSpeculator.__new__(RemoteK3DSparkSpeculator)
    proxy._known_requests = {"old"}
    proxy._remote_block_size = 16
    proxy._remote_window_size = 32
    proxy._remote_prefix_cache_tokens = 128
    proxy._retained_prefixes = {
        "old": _RetainedRequestPrefix(
            token_ids=torch.arange(96, dtype=torch.int32),
            committed_end=96,
            context_start=0,
            serial=1,
        )
    }
    return proxy


def test_remote_prefix_match_requires_exact_token_identity():
    proxy = _make_prefix_matcher()
    matching = torch.arange(80, dtype=torch.int32)

    assert proxy._find_reconnect_source(matching, 80, {"new"}) == "old"

    mismatched = matching.clone()
    mismatched[40] = -1
    assert proxy._find_reconnect_source(mismatched, 80, {"new"}) is None


def test_remote_prefix_match_rejects_range_evicted_from_projected_cache():
    proxy = _make_prefix_matcher()
    proxy._remote_prefix_cache_tokens = 48
    matching = torch.arange(40, dtype=torch.int32)

    assert proxy._find_reconnect_source(matching, 40, {"new"}) is None


def test_remote_prefix_match_rejects_history_before_cold_bootstrap():
    proxy = _make_prefix_matcher()
    proxy._retained_prefixes["old"].context_start = 64

    assert (
        proxy._find_reconnect_source(torch.arange(80, dtype=torch.int32), 80, {"new"})
        is None
    )
    assert (
        proxy._find_reconnect_source(torch.arange(96, dtype=torch.int32), 96, {"new"})
        == "old"
    )


def test_decode_remote_bfloat16_logits_frame_validates_metadata():
    logits = torch.tensor(
        [[[1.0, -2.0], [3.5, 0.25]]],
        dtype=torch.bfloat16,
    )
    frame = logits.view(torch.uint16).numpy().tobytes()
    response = {
        "logits": {
            "capability": "dflash_logits_bf16_v1",
            "dtype": "bfloat16",
            "shape": [1, 2, 2],
            "nbytes": len(frame),
        },
        "_logits_frame": frame,
    }

    decoded = _decode_bfloat16_logits_frame(response, (1, 2, 2))

    assert torch.equal(decoded, logits)


def test_decode_remote_logits_rejects_truncated_frame():
    response = {
        "logits": {
            "capability": "dflash_logits_bf16_v1",
            "dtype": "bfloat16",
            "shape": [1, 2, 2],
            "nbytes": 8,
        },
        "_logits_frame": b"\0\0",
    }

    with pytest.raises(ValueError, match="byte count mismatch"):
        _decode_bfloat16_logits_frame(response, (1, 2, 2))


def test_remote_probabilistic_sampling_uses_persistent_request_rows():
    proxy = RemoteK3DSparkSpeculator.__new__(RemoteK3DSparkSpeculator)
    proxy.draft_logits = torch.zeros(4, 3, 5, dtype=torch.bfloat16)
    proxy._remote_logits = torch.arange(
        4 * 3 * 5,
        dtype=torch.bfloat16,
    ).reshape(4, 3, 5)
    proxy._remote_sample_positions = torch.tensor(
        [[10, 11, -1], [-1, -1, -1], [20, 21, -1], [-1, -1, -1]],
        dtype=torch.int64,
    )
    proxy.draft_tokens = torch.full((4, 3), -1, dtype=torch.int64)
    recorded = {}

    def sample(**kwargs):
        recorded.update(kwargs)
        return torch.tensor([101, 102, 201, 202], dtype=torch.int64)

    proxy._sample_probabilistic_draft = sample
    batch = SimpleNamespace(
        num_reqs=3,
        idx_mapping=torch.tensor([2, 0, 3], dtype=torch.int32),
    )
    temperature = torch.ones(4, dtype=torch.float32)
    seeds = torch.arange(4, dtype=torch.int64)

    proxy._sample_remote_probabilistic(batch, temperature, seeds, 2)

    assert recorded["idx_mapping"].tolist() == [2, 2, 3, 3]
    assert recorded["positions"].tolist() == [8, 9, 18, 19]
    assert recorded["draft_step"].tolist() == [0, 1, 0, 1]
    assert proxy.draft_tokens.tolist() == [
        [101, 102, -1],
        [-1, -1, -1],
        [201, 202, -1],
        [-1, -1, -1],
    ]


def test_remote_probabilistic_sampler_uses_standard_draft_stream(monkeypatch):
    proxy = RemoteK3DSparkSpeculator.__new__(RemoteK3DSparkSpeculator)
    proxy.use_fp64_gumbel = True
    logits = torch.zeros(2, 5, dtype=torch.bfloat16)
    idx_mapping = torch.tensor([3, 1], dtype=torch.int32)
    temperature = torch.ones(4, dtype=torch.float32)
    seeds = torch.arange(4, dtype=torch.int64)
    positions = torch.tensor([8, 9], dtype=torch.int64)
    draft_step = torch.tensor([0, 1], dtype=torch.int64)
    draft_logits = torch.zeros(4, 3, 5, dtype=torch.bfloat16)
    recorded = {}

    monkeypatch.setattr(
        remote_speculator,
        "draft_gumbel_pos",
        lambda value: value + 1000,
    )

    def sample(*args, **kwargs):
        recorded["args"] = args
        recorded["kwargs"] = kwargs
        return torch.tensor([7, 8], dtype=torch.int64)

    monkeypatch.setattr(remote_speculator, "gumbel_sample", sample)

    sampled = proxy._sample_probabilistic_draft(
        logits=logits,
        positions=positions,
        idx_mapping=idx_mapping,
        temperature=temperature,
        seeds=seeds,
        draft_step=draft_step,
        draft_logits=draft_logits,
    )

    assert sampled.tolist() == [7, 8]
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(
            recorded["args"],
            (logits, idx_mapping, temperature, seeds, positions + 1000),
        )
    )
    assert recorded["kwargs"]["apply_temperature"] is True
    assert recorded["kwargs"]["use_fp64"] is True
    assert recorded["kwargs"]["logits_cache"] is draft_logits
    assert torch.equal(recorded["kwargs"]["logits_cache_col"], draft_step)
