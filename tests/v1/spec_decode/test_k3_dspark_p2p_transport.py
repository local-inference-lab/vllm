# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The peer-memory draft transport moves the bulk payloads between two
processes with device copies: the verifier pushes context rows into the
draft's ring slot and pulls the draft's top-k reply slot, both through CUDA
IPC mappings opened from the exported handles."""

import multiprocessing as mp
import os
from unittest import mock

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.dspark import p2p_transport as p2p


def _draft_process(conn, seed: int) -> None:
    """The exporting side: owns the ring and the reply slots."""
    torch.cuda.set_device(0)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    context = torch.zeros((3, 8, 16), dtype=torch.bfloat16, device="cuda")
    values = torch.randn((2, 4, 3, 5), generator=generator, device="cuda").to(
        torch.bfloat16
    )
    indices = torch.randint(
        0, 1000, (2, 4, 3, 5), generator=generator, device="cuda", dtype=torch.int32
    )
    torch.cuda.synchronize()
    conn.send(
        {
            "context": p2p.export_tensor(context),
            "values": p2p.export_tensor(values),
            "indices": p2p.export_tensor(indices),
            "values_cpu": values.cpu(),
            "indices_cpu": indices.cpu(),
        }
    )
    # The verifier pushed rows into slot 1; report what landed there.
    assert conn.recv() == "pushed"
    torch.cuda.synchronize()
    conn.send(context.cpu())
    assert conn.recv() == "done"


def test_peer_buffers_push_context_and_pull_reply():
    """Rows pushed into a ring slot land there, and a reply slot pulled into
    device tensors matches the exporter's contents, including a depth
    narrower than the slot."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    proc = ctx.Process(target=_draft_process, args=(child, 7), daemon=True)
    # IPC memory handles exist for cudaMalloc allocations only; the spawned
    # exporter reads the allocator configuration when it imports torch.
    with mock.patch.dict(
        os.environ, {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False"}
    ):
        proc.start()
    try:
        assert parent.poll(180), "exporter did not report its handles"
        info = parent.recv()
        torch.cuda.set_device(0)
        peer = p2p.DraftPeerBuffers(info)
        assert (peer.context_slots, peer.context_rows, peer.context_width) == (3, 8, 16)
        assert (peer.reply_slots, peer.max_requests, peer.num_steps, peer.topk) == (
            2,
            4,
            3,
            5,
        )
        stream = torch.cuda.current_stream()
        rows = torch.arange(6 * 16, device="cuda", dtype=torch.float32).view(6, 16)
        rows = (rows / 7).to(torch.bfloat16)
        peer.push_context(1, rows, stream.cuda_stream)
        stream.synchronize()
        parent.send("pushed")
        assert parent.poll(60), "exporter did not report the ring contents"
        landed = parent.recv()
        assert torch.equal(landed[1, :6], rows.cpu())
        assert torch.all(landed[0] == 0) and torch.all(landed[2] == 0)

        values = torch.empty((3, 2, 5), dtype=torch.bfloat16, device="cuda")
        indices = torch.empty((3, 2, 5), dtype=torch.int32, device="cuda")
        peer.pull_reply(1, 3, 2, values, indices, stream.cuda_stream)
        stream.synchronize()
        assert torch.equal(values.cpu(), info["values_cpu"][1, :3, :2])
        assert torch.equal(indices.cpu(), info["indices_cpu"][1, :3, :2])

        full_values = torch.empty((4, 3, 5), dtype=torch.bfloat16, device="cuda")
        full_indices = torch.empty((4, 3, 5), dtype=torch.int32, device="cuda")
        peer.pull_reply(0, 4, 3, full_values, full_indices, stream.cuda_stream)
        stream.synchronize()
        assert torch.equal(full_values.cpu(), info["values_cpu"][0])
        assert torch.equal(full_indices.cpu(), info["indices_cpu"][0])
        with pytest.raises(ValueError):
            peer.push_context(3, rows, stream.cuda_stream)
        peer.close()
        parent.send("done")
    finally:
        proc.join(timeout=30)
        if proc.is_alive():
            proc.kill()
    assert proc.exitcode == 0
