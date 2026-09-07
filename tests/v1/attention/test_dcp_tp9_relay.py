# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Five/four switch routing and the native DMA publisher's compact KV layout."""

import os

import pytest
import torch

from vllm.v1.attention.ops import dcp_utils


def test_tp9_relay_delivers_every_source_with_twelve_cross_switch_copies(monkeypatch):
    monkeypatch.setenv("VLLM_K3_DCP_GATHER_CLUSTERS", "0,1,2,3,8;4,5,6,7")
    larger = {0, 1, 2, 3, 8}
    delivered: set[tuple[int, int]] = set()
    cross_switch_copies = 0
    for rank in range(9):
        relay = dcp_utils.dcp_gather_relay_layout(9, rank)
        if relay is None:
            destinations = set(range(9))
        else:
            partner, mask = relay
            mates = {peer for peer in range(9) if mask & (1 << peer)}
            destinations = {rank, partner} | mates
            delivered.update((partner, peer) for peer in mates)
        delivered.update((rank, peer) for peer in destinations)
        cross_switch_copies += sum(
            (rank in larger) != (peer in larger) for peer in destinations
        )
    assert delivered == {(source, peer) for source in range(9) for peer in range(9)}
    assert dcp_utils.dcp_gather_relay_layout(9, 8) is None
    assert cross_switch_copies == 12


@pytest.mark.skipif(
    os.getenv("VLLM_K3_TEST_TP9_DMA") != "1",
    reason="set VLLM_K3_TEST_TP9_DMA=1 for the single-device DMA protocol test",
)
def test_tp9_mixed_relay_and_direct_publication_preserves_all_kv_bytes(monkeypatch):
    from vllm import _custom_ops  # noqa: F401

    torch.ops.load_library(os.environ["VLLM_K3_DCP_GATHER_ROTATE_LIB"])
    print("DMA extension loaded", flush=True)
    op = torch.ops._C_k3ext.direct_dcp_kv_gather_dma
    # Load each CUDA entry before enqueuing mutually dependent logical ranks.
    # Seed an already-arrived peer so warmup completes without another stream.
    warm_kv = [
        torch.zeros((3, 2, 576), dtype=torch.bfloat16, device="cuda") for _ in range(2)
    ]
    warm_signal = [
        torch.zeros((3, 2), dtype=torch.int32, device="cuda") for _ in range(2)
    ]
    warm_signal[0][:, 1].fill_(1)
    warm_epoch = torch.zeros(1, dtype=torch.int64, device="cuda")
    warm_runs = torch.tensor([[0, 0, 1]], dtype=torch.int64)
    op(
        torch.ones((1, 512), dtype=torch.bfloat16, device="cuda"),
        torch.ones((1, 64), dtype=torch.bfloat16, device="cuda"),
        warm_runs,
        torch.tensor([value.data_ptr() for value in warm_kv], dtype=torch.int64),
        torch.tensor(
            [value.data_ptr() for value in warm_signal],
            dtype=torch.int64,
            device="cuda",
        ),
        warm_kv[0],
        warm_signal[0],
        warm_epoch,
        1,
        512,
        0,
        2,
        0,
        2,
        -1,
        0,
        warm_runs,
    )
    torch.accelerator.synchronize()
    monkeypatch.setenv("VLLM_K3_DCP_GATHER_CLUSTERS", "0,1,2,3,8;4,5,6,7")
    world, slots, capacity, split, dim = 9, 3, 9216, 512, 576
    received = [
        torch.full(
            (slots, capacity, dim), float("nan"), dtype=torch.bfloat16, device="cuda"
        )
        for _ in range(world)
    ]
    signals = [
        torch.zeros((slots, world), dtype=torch.int32, device="cuda")
        for _ in range(world)
    ]
    epochs = [torch.zeros(1, dtype=torch.int64, device="cuda") for _ in range(world)]
    streams = [torch.Stream(device="cuda") for _ in range(world)]
    peer_kv = torch.tensor([value.data_ptr() for value in received], dtype=torch.int64)
    peer_signal = torch.tensor(
        [value.data_ptr() for value in signals], dtype=torch.int64, device="cuda"
    )
    padded = [512, 512]
    lengths = [
        [511 - rank * 19 for rank in range(world)],
        [320, 0, 500, 256, 64, 300, 111, 0, 512],
    ]
    starts = [0, 0]
    for window in range(4):
        sources = [
            (torch.randn((1024, dim), device="cuda") + rank + window).to(torch.bfloat16)
            for rank in range(world)
        ]
        pieces = [
            sources[rank][request * 512 : request * 512 + lengths[request][rank]]
            for request in range(2)
            for rank in range(world)
        ]
        expected = torch.cat(pieces, dim=0)
        rows = expected.shape[0]
        planes = [
            (source[:, :split].contiguous(), source[:, split:].contiguous())
            for source in sources
        ]
        runs = [
            dcp_utils.build_dcp_kv_final_layout_runs(padded, lengths, starts, rank)
            for rank in range(world)
        ]
        torch.accelerator.synchronize()
        print(f"DMA window {window}: inputs ready", flush=True)
        slot = window % slots
        for rank in range(world):
            relay = dcp_utils.dcp_gather_relay_layout(world, rank)
            partner, mask = (-1, 0) if relay is None else relay
            partner_runs = runs[rank] if relay is None else runs[partner]
            with torch.cuda.stream(streams[rank]):
                op(
                    *planes[rank],
                    runs[rank],
                    peer_kv,
                    peer_signal,
                    received[rank],
                    signals[rank],
                    epochs[rank],
                    rows,
                    split,
                    slot,
                    world,
                    rank,
                    capacity,
                    partner,
                    mask,
                    partner_runs,
                )
            print(f"DMA window {window}: rank {rank} submitted", flush=True)
        torch.accelerator.synchronize()
        for rank in range(world):
            flat = received[rank][slot].flatten()
            actual_c = flat[: capacity * split].view(capacity, split)[:rows]
            actual_pe = flat[capacity * split :].view(capacity, dim - split)[:rows]
            assert torch.equal(actual_c, expected[:, :split])
            assert torch.equal(actual_pe, expected[:, split:])
            assert torch.all(signals[rank][slot] == window + 1)
