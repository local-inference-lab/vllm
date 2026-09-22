# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU launch-policy checks, independent of mixed-attention metadata fixtures."""

import pytest
import torch

from vllm.v1.attention.ops import triton_unified_attention_diffkv as attention


class LaunchSpy:
    def __init__(self, calls, kind):
        self.calls, self.kind = calls, kind

    def __getitem__(self, grid):
        def launch(**kwargs):
            self.calls.append((self.kind, grid, kwargs))

        return launch


@pytest.fixture
def launches(monkeypatch):
    calls = []
    monkeypatch.setattr(
        attention, "kernel_unified_attention_diffkv", LaunchSpy(calls, "attention")
    )
    monkeypatch.setattr(
        attention, "kernel_reduce_segments_diffkv", LaunchSpy(calls, "reduce")
    )
    monkeypatch.setattr(attention, "is_batch_invariant", False)
    return calls


def inputs(requests, rows=1, window=128, capacity=None):
    tokens = requests * rows
    capacity = tokens if capacity is None else capacity
    cache = torch.empty(2, 16, 2, 320)
    return dict(
        q=torch.empty(tokens, 16, 192),
        k=cache[..., :192],
        v=cache[..., 192:],
        out=torch.empty(tokens, 16, 128),
        cu_seqlens_q=torch.arange(requests + 1, dtype=torch.int32) * rows,
        seqused_k=torch.full((requests,), 2048, dtype=torch.int32),
        softmax_scale=192**-0.5,
        causal=True,
        window_size=(-1, -1) if window is None else (window - 1, 0),
        block_table=torch.zeros(requests, 128, dtype=torch.int32),
        softcap=0,
        max_seqlen_q=rows,
        seq_threshold_3D=32,
        num_par_softmax_segments=16,
        softmax_segm_output=torch.empty(capacity, 16, 16, 128),
        softmax_segm_max=torch.empty(capacity, 16, 16),
        softmax_segm_expsum=torch.empty(capacity, 16, 16),
    )


def assert_dispatch(calls, args, split):
    assert [kind for kind, _, _ in calls] == (
        ["attention", "reduce"] if split else ["attention"]
    )
    _, grid, kwargs = calls[0]
    assert kwargs["IS_3D"] is split
    assert len(grid) == (3 if split else 2)
    assert kwargs["seq_lens_ptr"] is args["seqused_k"]
    assert kwargs["query_start_len_ptr"] is args["cu_seqlens_q"]
    assert kwargs["SLIDING_WINDOW"] == (
        0 if args["window_size"][0] < 0 else args["window_size"][0] + 1
    )
    if split:
        assert kwargs["segm_output_ptr"] is args["softmax_segm_output"]
        assert grid[-1] == 16
        assert calls[1][1] == (args["q"].shape[0], 16)
    else:
        assert kwargs["segm_output_ptr"] is args["out"]
        assert kwargs["segm_max_ptr"] is args["out"]
        assert kwargs["segm_expsum_ptr"] is args["out"]
        assert kwargs["NUM_SEGMENTS_PER_SEQ"] == 1


@pytest.mark.parametrize("requests", [7, 8, 32])
@pytest.mark.parametrize("window", [1, 64, 128, 129, None])
@pytest.mark.parametrize("rows", [1, 4])
def test_dispatch_request_and_window_boundaries(launches, requests, window, rows):
    args = inputs(requests, rows, window)
    attention.unified_attention_diffkv(**args)
    split = rows == 1 and (requests < 8 or window is None or window > 128)
    assert_dispatch(launches, args, split)


@pytest.mark.parametrize(
    "missing",
    [
        "seq_threshold_3D",
        "num_par_softmax_segments",
        "softmax_segm_output",
        "softmax_segm_max",
        "softmax_segm_expsum",
    ],
)
@pytest.mark.parametrize("window,requests", [(128, 7), (128, 8), (None, 8)])
def test_absent_workspace_disables_split(launches, missing, window, requests):
    args = inputs(requests, window=window)
    args[missing] = None
    attention.unified_attention_diffkv(**args)
    assert_dispatch(launches, args, False)


@pytest.mark.parametrize("threshold,split", [(7, False), (8, True)])
def test_existing_sequence_capacity_guard(launches, threshold, split):
    args = inputs(8, window=129)
    args["seq_threshold_3D"] = threshold
    attention.unified_attention_diffkv(**args)
    assert_dispatch(launches, args, split)


@pytest.mark.parametrize("rows,window", [(1, 128), (4, 128), (4, 129)])
@pytest.mark.parametrize("capacity", [0, 1])
def test_2d_does_not_use_undersized_workspace(launches, rows, window, capacity):
    # The upstream dispatcher has no row-capacity guards. Check that paths
    # requiring 2D do not use these buffers, without adding a split-KV feature.
    args = inputs(8, rows=rows, window=window, capacity=capacity)
    attention.unified_attention_diffkv(**args)
    assert_dispatch(launches, args, False)


@pytest.mark.parametrize("window", [128, 129, None])
def test_batch_invariance_preserves_2d(monkeypatch, launches, window):
    monkeypatch.setattr(attention, "is_batch_invariant", True)
    args = inputs(7, window=window)
    attention.unified_attention_diffkv(**args)
    assert_dispatch(launches, args, False)


@pytest.mark.parametrize("window,split", [(128, False), (129, True), (None, True)])
def test_static_dispatch_keeps_live_metadata(monkeypatch, launches, window, split):
    args = inputs(8, window=window)
    starts = args["cu_seqlens_q"]
    lengths = args["seqused_k"]
    pointers = starts.data_ptr(), lengths.data_ptr()

    def forbidden(*args, **kwargs):
        raise AssertionError("dispatch must not read tensor values on the host")

    for context in (1, 32, 4096):
        lengths.fill_(context)
        expected_lengths = lengths.clone()
        expected_starts = starts.clone()
        launches.clear()
        with monkeypatch.context() as guard:
            for name in ("item", "tolist", "cpu", "numpy", "__bool__", "__int__"):
                guard.setattr(torch.Tensor, name, forbidden)
            attention.unified_attention_diffkv(**args)
        assert_dispatch(launches, args, split)
        assert (starts.data_ptr(), lengths.data_ptr()) == pointers
        torch.testing.assert_close(lengths, expected_lengths)
        torch.testing.assert_close(starts, expected_starts)
