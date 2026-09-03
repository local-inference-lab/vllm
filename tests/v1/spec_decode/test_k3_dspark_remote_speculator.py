# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.dspark.remote_speculator import (
    RemoteK3DSparkSpeculator,
    _anchor_positions_from_context,
    _build_valid_context_plan,
    _contiguous_draft_output,
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


def test_anchor_positions_follow_actual_valid_context_rows():
    positions = torch.tensor([24, 25, 26, 80, 81], dtype=torch.int64)

    anchors = _anchor_positions_from_context([3, 2], positions)

    assert anchors == [27, 82]


def test_remote_tokens_copy_supports_adaptive_depth():
    proxy = RemoteK3DSparkSpeculator.__new__(RemoteK3DSparkSpeculator)
    proxy.device = torch.device("cpu")
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


def test_deferred_peer_ingest_does_not_reserve_or_compute_a_reply():
    proxy = RemoteK3DSparkSpeculator.__new__(RemoteK3DSparkSpeculator)
    proxy._probabilistic = True
    proxy._logits_topk = 128
    proxy._peer = SimpleNamespace(reply_slots=2)
    proxy._peer_reply_slot = 1
    header = {}

    reply_slot = proxy._configure_proposal_reply(
        header,
        deferred_ingest=True,
        peer_context=True,
        peer_context_slot=2,
    )

    assert reply_slot == -1
    assert proxy._peer_reply_slot == 1
    assert header == {
        "return_logits": False,
        "logits_topk": 0,
        "p2p_context_slot": 2,
    }


def test_consumed_peer_proposal_reserves_the_named_reply_slot():
    proxy = RemoteK3DSparkSpeculator.__new__(RemoteK3DSparkSpeculator)
    proxy._probabilistic = True
    proxy._logits_topk = 128
    proxy._peer = SimpleNamespace(reply_slots=2)
    proxy._peer_reply_slot = 1
    header = {}

    reply_slot = proxy._configure_proposal_reply(
        header,
        deferred_ingest=False,
        peer_context=True,
        peer_context_slot=2,
    )

    assert reply_slot == 1
    assert proxy._peer_reply_slot == 0
    assert header == {
        "return_logits": True,
        "logits_topk": 128,
        "p2p_context_slot": 2,
        "p2p_reply": True,
        "p2p_reply_slot": 1,
    }


def test_free_remote_request_releases_capture_bookkeeping(monkeypatch):
    from vllm.v1.worker.gpu.spec_decode.dspark import remote_speculator as rs

    proxy = RemoteK3DSparkSpeculator.__new__(RemoteK3DSparkSpeculator)
    proxy._known_requests = {"finished"}
    proxy._retained_prefixes = {"finished": object()}
    proxy._rpc = lambda frames: {"ok": True}
    monkeypatch.setitem(rs._K3_CAPTURE_STATE, "finished", {"records": []})

    proxy._free_remote_requests({"finished"})

    assert "finished" not in proxy._known_requests
    assert "finished" not in proxy._retained_prefixes
    assert "finished" not in rs._K3_CAPTURE_STATE


def test_deferred_ingest_failures_disable_every_affected_request():
    proxy = RemoteK3DSparkSpeculator.__new__(RemoteK3DSparkSpeculator)
    proxy.method = "dflash"
    proxy._disabled_requests = set()
    proxy._async_error_lock = threading.Lock()
    proxy._async_errors = [
        (RuntimeError("first failure"), ["request-a"]),
        (RuntimeError("second failure"), ["request-b", "request-c"]),
    ]

    proxy._apply_async_error()

    assert proxy._disabled_requests == {"request-a", "request-b", "request-c"}
    assert proxy._async_errors == []


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
    proxy.method = "dflash"
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


# --- deferred resolve: the reply is consumed in stream order -----------------


def _requires_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the stream gate")


def _topk_response(tokens, values, indices, positions):
    """A PROPOSE reply in the top-k transport format."""
    from vllm.v1.worker.gpu.spec_decode.dspark import remote_speculator as rs

    shape = tuple(values.shape)
    frame = (
        values.contiguous().view(torch.uint16).numpy().tobytes()
        + indices.contiguous().to(torch.int32).numpy().tobytes()
    )
    return {
        "ok": True,
        "protocol": rs.PROTOCOL_VERSION,
        "tokens": tokens,
        "logits": {
            "capability": rs.TOPK_LOGITS_CAPABILITY,
            "dtype": "bfloat16",
            "shape": list(shape),
            "nbytes_values": values.numel() * 2,
            "nbytes_indices": indices.numel() * 4,
            "sample_positions": positions,
        },
        "_logits_frame": frame,
        "timing_ms": {"total": 1.0},
    }


class _NoBroadcast:
    def broadcast(self, tensor, src=0):
        return tensor


def _make_deferred_proxy(*, max_reqs=3, steps=2, vocab=64, topk=4):
    """A rank-0 proxy with the transport buffers of the probabilistic top-k path."""
    import threading

    from vllm.v1.worker.gpu.spec_decode.dspark import remote_speculator as rs

    device = torch.device("cuda")
    proxy = RemoteK3DSparkSpeculator.__new__(RemoteK3DSparkSpeculator)
    proxy.device = device
    proxy.method = "dflash"
    proxy._tp_rank = 0
    proxy._tp_group = _NoBroadcast()
    proxy._probabilistic = True
    proxy._logits_topk = topk
    proxy.vocab_size = vocab
    proxy.use_fp64_gumbel = False
    proxy.max_num_reqs = max_reqs
    proxy.num_speculative_steps = steps
    proxy._timing_log_interval = 0
    proxy._timing_count = 0
    proxy._timing_totals_ms = {}
    proxy._disabled_requests = set()
    proxy._async_queue = None
    proxy._async_lock = threading.Lock()
    proxy.draft_tokens = torch.full(
        (max_reqs, steps), -1, dtype=torch.int64, device=device
    )
    proxy.draft_logits = torch.zeros(
        (max_reqs, steps, vocab), dtype=torch.bfloat16, device=device
    )
    proxy._remote_logits = torch.zeros_like(proxy.draft_logits)
    proxy._remote_sample_positions = torch.full(
        (max_reqs, steps), -1, dtype=torch.int64, device=device
    )
    proxy._remote_topk_values = torch.full(
        (max_reqs, steps, topk),
        rs.TOPK_LOGITS_FILL,
        dtype=torch.bfloat16,
        device=device,
    )
    proxy._remote_topk_indices = torch.zeros(
        (max_reqs, steps, topk), dtype=torch.int64, device=device
    )
    proxy._topk_values_staging = torch.empty(
        (max_reqs, steps, topk), dtype=torch.bfloat16, pin_memory=True
    )
    proxy._topk_indices_staging = torch.empty(
        (max_reqs, steps, topk), dtype=torch.int32, pin_memory=True
    )
    proxy._tokens_staging = [
        torch.full((max_reqs, steps), -1, dtype=torch.int64, pin_memory=True)
        for _ in range(2)
    ]
    proxy._sample_positions_staging = [
        torch.full((max_reqs, steps), -1, dtype=torch.int64, pin_memory=True)
        for _ in range(2)
    ]
    proxy._staging_slot = 0
    proxy._gate = rs._StreamGate()
    proxy._deferred_resolve = True
    proxy.deferred_resolve_allowed = True
    proxy._pending = None
    proxy._pending_failure = None
    proxy._pending_epoch = 0
    proxy._last_resolved = None
    proxy._peer = None
    proxy._peer_stale = False
    return proxy


def _canned_reply(active, steps, topk, vocab, seed=0):
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, vocab, (active, steps), generator=g).tolist()
    values = (torch.randn(active, steps, topk, generator=g) * 2).to(torch.bfloat16)
    indices = torch.stack(
        [torch.randperm(vocab, generator=g)[:topk] for _ in range(active * steps)]
    ).view(active, steps, topk)
    positions = [
        [
            5 + r * 10 + s if not (r == 0 and s == steps - 1) else -1
            for s in range(steps)
        ]
        for r in range(active)
    ]
    return tokens, values, indices, positions


def test_stream_gate_parks_the_stream_until_released():
    """Work enqueued behind the gate runs only after another thread releases it."""
    _requires_cuda()
    import threading
    import time

    from vllm.v1.worker.gpu.spec_decode.dspark.remote_speculator import _StreamGate

    gate = _StreamGate()
    stream = torch.cuda.Stream()
    x = torch.zeros(1024, device="cuda")
    with torch.cuda.stream(stream):
        gate.arm(stream)
        x.fill_(7.0)
        done = torch.cuda.Event()
        done.record(stream)
    time.sleep(0.05)
    assert not done.query()
    threading.Thread(target=gate.release, daemon=True).start()
    done.synchronize()
    assert torch.all(x == 7.0)


def test_deferred_resolve_reproduces_the_immediate_reply_path():
    """The gate-ordered copies, broadcast and masked sampling yield the same
    draft tokens and cached draft logits as consuming the reply immediately."""
    _requires_cuda()
    import time

    steps, topk, vocab, max_reqs = 2, 4, 64, 3
    active = [0, 2]
    reply = _canned_reply(len(active), steps, topk, vocab)
    response = _topk_response(*reply)
    idx_mapping = torch.tensor([1, 4, 2], dtype=torch.int64, device="cuda")
    temperature = torch.full((8,), 0.7, device="cuda")
    seeds = torch.arange(8, device="cuda", dtype=torch.int64) * 977
    batch = SimpleNamespace(num_reqs=max_reqs, idx_mapping=idx_mapping)

    immediate = _make_deferred_proxy(
        max_reqs=max_reqs, steps=steps, vocab=vocab, topk=topk
    )
    immediate._copy_tokens_from_response(response, active, steps)
    immediate._copy_logits_from_response(response, active, steps)
    immediate._broadcast_remote_logits(steps)
    immediate._sample_remote_probabilistic(batch, temperature, seeds, steps)
    torch.cuda.synchronize()

    deferred = _make_deferred_proxy(
        max_reqs=max_reqs, steps=steps, vocab=vocab, topk=topk
    )
    # Compile the sampling kernels before the timed run: a kernel compiled
    # behind the gate would only delay the host, not the check below.
    deferred._sample_remote_probabilistic_masked(
        idx_mapping, max_reqs, temperature, seeds, steps
    )
    deferred.draft_tokens.fill_(-1)
    deferred.draft_logits.zero_()
    torch.cuda.synchronize()

    def slow_rpc(frames):
        time.sleep(0.3)
        return response

    deferred._rpc = slow_rpc
    deferred._start_reply_thread(
        [b"header"], active, ["a", "c"], steps, batch, temperature, seeds
    )
    assert deferred.has_pending_proposal()
    output = deferred.resolve_pending()
    marker = torch.cuda.Event()
    marker.record()
    assert not marker.query()  # the stream is parked until the reply is staged
    torch.cuda.synchronize()
    assert not deferred.has_pending_proposal()
    assert torch.equal(output, immediate.draft_tokens[:max_reqs, :steps])
    assert torch.equal(deferred.draft_tokens, immediate.draft_tokens)
    assert torch.equal(deferred.draft_logits, immediate.draft_logits)
    assert torch.equal(
        deferred._remote_sample_positions, immediate._remote_sample_positions
    )
    deferred._join_reply_thread()


def test_deferred_resolve_failure_yields_no_draft_and_disables_requests():
    """A reply thread that fails still releases the gate with a no-draft reply,
    and the failed requests are disabled at the next proposal."""
    _requires_cuda()
    steps, topk, vocab, max_reqs = 2, 4, 64, 2
    proxy = _make_deferred_proxy(max_reqs=max_reqs, steps=steps, vocab=vocab, topk=topk)
    proxy.draft_tokens.fill_(3)

    def failing_rpc(frames):
        raise RuntimeError("draft server unreachable")

    proxy._rpc = failing_rpc
    batch = SimpleNamespace(
        num_reqs=max_reqs,
        idx_mapping=torch.tensor([0, 1], dtype=torch.int64, device="cuda"),
    )
    proxy._start_reply_thread(
        [b"header"],
        [0, 1],
        ["a", "b"],
        steps,
        batch,
        torch.ones(4, device="cuda"),
        torch.zeros(4, dtype=torch.int64, device="cuda"),
    )
    output = proxy.resolve_pending()
    torch.cuda.synchronize()
    assert output.tolist() == [[-1, -1], [-1, -1]]
    proxy._join_reply_thread()
    proxy._apply_pending_failure()
    assert proxy._disabled_requests == {"a", "b"}


class _FakePeer:
    """Stands in for the mapped draft buffers: the reply slot content is a
    tensor here, the pull is a device copy on the caller's stream."""

    def __init__(self, values: torch.Tensor, indices: torch.Tensor):
        self._values = values  # [slots, requests, K, topk]
        self._indices = indices
        self.closed = False
        self.pulled: list[tuple[int, int, int]] = []

    def pull_reply(self, slot, rows, steps, values_out, indices_out, stream):
        self.pulled.append((slot, rows, steps))
        values_out.copy_(self._values[slot, :rows, :steps])
        indices_out.copy_(self._indices[slot, :rows, :steps])

    def close(self):
        self.closed = True


def test_deferred_resolve_pulls_a_peer_reply_like_the_frame_reply():
    """A reply that names a peer slot instead of carrying a logits frame
    yields the same draft tokens and cached logits as the frame transport."""
    _requires_cuda()
    from vllm.v1.worker.gpu.spec_decode.dspark import remote_speculator as rs

    steps, topk, vocab, max_reqs = 2, 4, 64, 3
    active = [0, 2]
    tokens, values, indices, positions = _canned_reply(len(active), steps, topk, vocab)
    frame_response = _topk_response(tokens, values, indices, positions)
    idx_mapping = torch.tensor([1, 4, 2], dtype=torch.int64, device="cuda")
    temperature = torch.full((8,), 0.7, device="cuda")
    seeds = torch.arange(8, device="cuda", dtype=torch.int64) * 977
    batch = SimpleNamespace(num_reqs=max_reqs, idx_mapping=idx_mapping)

    via_frames = _make_deferred_proxy(
        max_reqs=max_reqs, steps=steps, vocab=vocab, topk=topk
    )
    via_frames._rpc = lambda frames: frame_response
    via_frames._start_reply_thread(
        [b"h"], active, ["a", "c"], steps, batch, temperature, seeds
    )
    via_frames.resolve_pending()
    torch.cuda.synchronize()
    via_frames._join_reply_thread()

    via_peer = _make_deferred_proxy(
        max_reqs=max_reqs, steps=steps, vocab=vocab, topk=topk
    )
    slot_values = torch.zeros((2, max_reqs, steps, topk), dtype=torch.bfloat16)
    slot_indices = torch.zeros((2, max_reqs, steps, topk), dtype=torch.int32)
    slot_values[1, : len(active)] = values
    slot_indices[1, : len(active)] = indices
    via_peer._peer = _FakePeer(slot_values.cuda(), slot_indices.cuda())
    via_peer._topk_values_peer = torch.empty(
        max_reqs * steps * topk, dtype=torch.bfloat16, device="cuda"
    )
    via_peer._topk_indices_peer = torch.empty(
        max_reqs * steps * topk, dtype=torch.int32, device="cuda"
    )
    peer_response = {
        "ok": True,
        "protocol": rs.PROTOCOL_VERSION,
        "tokens": tokens,
        "logits": {
            "capability": rs.TOPK_LOGITS_P2P_CAPABILITY,
            "dtype": "bfloat16",
            "shape": [len(active), steps, topk],
            "topk": topk,
            "p2p_reply_slot": 1,
            "sample_positions": positions,
        },
    }
    via_peer._rpc = lambda frames: peer_response
    via_peer._start_reply_thread(
        [b"h"],
        active,
        ["a", "c"],
        steps,
        batch,
        temperature,
        seeds,
        p2p_reply=(1, len(active), steps),
    )
    output = via_peer.resolve_pending()
    torch.cuda.synchronize()
    via_peer._join_reply_thread()
    assert via_peer._peer.pulled == [(1, len(active), steps)]
    assert torch.equal(output, via_frames.draft_tokens[:max_reqs, :steps])
    assert torch.equal(via_peer.draft_tokens, via_frames.draft_tokens)
    assert torch.equal(via_peer.draft_logits, via_frames.draft_logits)
    assert torch.equal(via_peer._remote_topk_values, via_frames._remote_topk_values)
    assert torch.equal(via_peer._remote_topk_indices, via_frames._remote_topk_indices)
