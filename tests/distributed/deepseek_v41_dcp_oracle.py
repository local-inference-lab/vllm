# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Four-rank native DS4.1 attention oracle, run with torch.distributed.run."""

import os
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist
from b12x.attention import compressed_sparse_mla as mla
from b12x.attention.compressed_sparse_mla.reference import (
    compressed_sparse_mla_reference,
    pack_deepseek_v41_cache_reference,
)
from b12x.preparation import PreparationSession, PreparedCall

from vllm.models.deepseek_v4_1 import dcp


class _OracleDcpGroup:
    """Minimal CUDA coordinator used by the standalone index-shard oracle."""

    def __init__(self, world_size, rank, device_group):
        self.unique_name = "oracle-index-shard"
        self.world_size = world_size
        self.rank_in_group = rank
        self.ranks = list(range(world_size))
        self.cpu_group = dist.group.WORLD
        self.device_group = device_group

    def all_gather(self, value, dim=-1):
        gathered = [torch.empty_like(value) for _ in range(self.world_size)]
        dist.all_gather(gathered, value, group=self.device_group)
        return torch.cat(gathered, dim=dim)


def _global_index(local, rank, world, stripe):
    return (
        local // stripe * (world * stripe)
        + rank * stripe
        + local % stripe
    )


def _index_shard_case(device, rank, world, topk, stripe):
    """Build rank-local candidates and their exact global top-k reference."""
    rows, global_width = 4, 32768
    local_width = global_width // world
    local = torch.arange(local_width, device=device, dtype=torch.int64)
    global_ids = _global_index(local, rank, world, stripe)
    lengths = torch.tensor(
        [global_width, global_width - 137, topk + 79, topk],
        device=device,
        dtype=torch.int64,
    )
    selected_ids = torch.full(
        (rows, topk), -1, device=device, dtype=torch.int32
    )
    selected_scores = torch.full(
        (rows, topk), -float("inf"), device=device, dtype=torch.float32
    )
    expected = []
    all_ids = torch.arange(global_width, device=device, dtype=torch.int64)
    for row in range(rows):
        local_scores = (
            (global_ids * 8191 + row * 131071) % 16777213
        ).to(torch.float32)
        local_scores.masked_fill_(global_ids >= lengths[row], -float("inf"))
        values, positions = local_scores.topk(topk)
        valid = torch.isfinite(values)
        selected_ids[row, valid] = positions[valid].to(torch.int32)
        selected_scores[row, valid] = values[valid]

        global_scores = (
            (all_ids * 8191 + row * 131071) % 16777213
        ).to(torch.float32)
        global_scores.masked_fill_(all_ids >= lengths[row], -float("inf"))
        expected.append(global_scores.topk(topk).indices.to(torch.int32))
    return selected_ids, selected_scores, torch.stack(expected)


def check_index_shards(device, group):
    """Exact global selection, candidate ownership, and graph replay."""
    from vllm.v1.attention.backends.mla import b12x_indexer

    rank, world = group.rank_in_group, group.world_size
    b12x_indexer.get_dcp_group = lambda: group
    merged_by_shape = {}
    for topk, stripe in ((512, 64), (2048, 8)):
        indices, scores, expected = _index_shard_case(
            device, rank, world, topk, stripe
        )
        b12x_indexer._merge_dcp_topk(
            indices, scores, rank, world, stripe
        )
        torch.cuda.synchronize()
        for row in range(indices.shape[0]):
            assert set(indices[row].cpu().tolist()) == set(
                expected[row].cpu().tolist()
            )
        merged_by_shape[(topk, stripe)] = indices.clone()

    candidates = torch.full(
        (4, 16384), -1, device=device, dtype=torch.int32
    )
    candidates[:, :512].copy_(merged_by_shape[(512, 64)])
    candidate_lengths = torch.full(
        (4,), 512, device=device, dtype=torch.int32
    )
    local_candidates = torch.empty_like(candidates)
    local_lengths = torch.empty_like(candidate_lengths)
    tile_offsets = torch.empty((4, 32), device=device, dtype=torch.int32)
    dcp.localize_index_candidates(
        candidates,
        candidate_lengths,
        local_candidates,
        local_lengths,
        tile_offsets,
        world_size=world,
        rank=rank,
        stripe=64,
    )
    torch.cuda.synchronize()
    for row in range(candidates.shape[0]):
        source = candidates[row, : candidate_lengths[row]].cpu().tolist()
        expected_local = [
            value // (world * 64) * 64 + value % 64
            for value in source
            if (value // 64) % world == rank
        ]
        length = int(local_lengths[row].item())
        assert local_candidates[row, :length].cpu().tolist() == expected_local

    source_indices, source_scores, expected = _index_shard_case(
        device, rank, world, 512, 64
    )
    graph_indices = torch.empty_like(source_indices)
    graph_scores = torch.empty_like(source_scores)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_indices.copy_(source_indices)
        graph_scores.copy_(source_scores)
        b12x_indexer._merge_dcp_topk(
            graph_indices, graph_scores, rank, world, 64
        )
    allocated = torch.cuda.memory_allocated(device)
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated(device) == allocated
        for row in range(graph_indices.shape[0]):
            assert set(graph_indices[row].cpu().tolist()) == set(
                expected[row].cpu().tolist()
            )
    graph.reset()
    print(
        f"rank={rank} index-shard exact-topk/ownership/graphs PASS",
        flush=True,
    )


def check_tp_plain_push(device, session, coordinator):
    """Unchanged BF16/FP32 rounding against native pull and an independent sum."""
    from b12x.comm import pcie
    from b12x.comm.pcie._oneshot_preparation import _prepare_plain_call
    from b12x.preparation import CollectiveRequirement

    rank = dist.get_rank()
    cases, runtimes, requests = [], [], []
    shapes = ((1, 5120), (6, 5120), (8, 1280), (12, 1280))
    for push in (False, True):
        os.environ["B12X_PCIE_TP4_REMOTE_PUSH"] = "1" if push else "0"
        runtime = pcie.OneshotAllReduce.from_process_group(
            process_group=dist.group.WORLD, device=device,
            max_input_bytes=84 * 1024, rank_data_bytes=64 * 1024,
        )
        runtimes.append(runtime)
        for rows, hidden in shapes:
            torch.manual_seed(927 + rank)
            inp = (torch.randn(rows, hidden, device=device) * 0.125).bfloat16()
            inp.mul_((1e-3, 1.0, 1e3, 1e-6)[rank])
            out = torch.empty_like(inp)
            query = pcie.query_from_runtime(
                runtime, surface="OneshotAllReduce.all_reduce", call={"inp": inp},
            )
            expected_route = "tp4_remote_push" if push and rows <= 8 else "pull"
            assert query.call["transport"] == expected_route
            plan = pcie.plan(query, runtime=runtime)
            requests.append(plan.request(
                name=f"oracle.tp4.{push}.{rows}.{hidden}",
                prepare_call=lambda state, inp=inp, out=out: _prepare_plain_call(
                    state, inp=inp, out=out,
                ),
                collective=CollectiveRequirement(
                    key=f"oracle.tp4.{push}.{rows}.{hidden}", ranks=tuple(range(4)),
                ),
            ))
            cases.append((runtime, plan, inp, out))
    session.prepare(requests, coordinator=coordinator)

    def check_pair(index):
        pull, push = cases[index], cases[index + len(shapes)]
        torch.testing.assert_close(push[3], pull[3], rtol=0, atol=0)
        peers = [None] * 4
        dist.all_gather_object(peers, push[2].cpu())
        expected = peers[rank].float()
        for peer in range(1, 4):
            expected.add_(peers[(rank + peer) % 4].float())
        torch.testing.assert_close(push[3].cpu(), expected.bfloat16(), rtol=0, atol=0)
        assert push[3].abs().sum() > 0

    with session.capture():
        for index in range(len(shapes)):
            for runtime, plan, inp, out in (cases[index], cases[index + len(shapes)]):
                runtime.all_reduce(inp, out=out, plan=plan)
            torch.cuda.synchronize()
            check_pair(index)
        graph = torch.cuda.CUDAGraph()
        # The real TP channel mixes small push and larger pull declarations.
        # They must safely alternate the same IPC slots and graph epoch.
        graph_cases = [cases[index] for index in (1, 5, 3, 7)]
        with torch.cuda.graph(graph):
            for runtime, plan, inp, out in graph_cases:
                runtime.all_reduce(inp, out=out, plan=plan)
        allocated = torch.cuda.memory_allocated(device)
        for _ in range(3):
            for _, _, inp, out in graph_cases:
                inp.add_(0.125)
                out.zero_()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated(device) == allocated
            check_pair(1)
            check_pair(3)
        graph.reset()
    for _, plan, _, _ in cases:
        session.release(plan)
    for runtime in runtimes:
        runtime.close()
    print(f"rank={rank} TP4 plain push native-pull/FP32-rounding/fresh-graphs PASS", flush=True)


def check_kv_replica(device, session, coordinator):
    """Packed bytes, recycled high pages, changing grids and serial graph reuse."""
    from b12x.comm import pcie
    from b12x.comm.pcie._owner_preparation import prepared_call
    from b12x.preparation import CollectiveRequirement

    rank = dist.get_rank()
    capacity, requests, page, stripe = 1536, 2, 128, 64
    runtime = pcie.PagedKvReplica.from_process_group(
        process_group=dist.group.WORLD, device=device,
        max_requests=requests, max_tokens=capacity, stripe_alignment=stripe,
    )
    query = pcie.query_from_runtime(
        runtime, surface="PagedKvReplica.replicate",
        call=dict(page_size=page, stripe=stripe, ratio=2, max_tokens=capacity),
    )
    plan = pcie.plan(query, runtime=runtime)
    width = (capacity + page - 1) // page
    out = torch.empty((requests * width + 1, page * 288), device=device, dtype=torch.uint8)
    local_width = (runtime.local_capacity + page - 1) // page
    page_stride = page * 288 + 256
    base = 2**31 // page_stride + 3
    storage = torch.empty((base + requests * local_width, page_stride),
                          device=device, dtype=torch.uint8)
    cache = storage[:, :page * 288]
    table = torch.arange(base, base + requests * local_width, device=device,
                         dtype=torch.int32).view(requests, local_width)
    positions = torch.zeros(requests, device=device, dtype=torch.int64)
    starts = torch.tensor([0, 2048, 3072], device=device, dtype=torch.int32)
    torch.manual_seed(947)
    expected = torch.randint(0, 256, (requests, capacity, 288),
                             device=device, dtype=torch.uint8)
    tokens = torch.arange(capacity, device=device)
    owned = tokens // stripe % 4 == rank
    local_records = cache[base:].view(requests, local_width, page, 288)
    local_records.copy_(expected[:, owned].view(requests, local_width, page, 288))
    actual = dict(cache=cache, table=table, positions=positions, starts=starts,
                  out=out, requests=requests, max_tokens=1024)
    session.prepare((plan.request(
        name="oracle.prefill_kv_replica",
        prepare_call=lambda state: prepared_call(state, **actual),
        collective=CollectiveRequirement(
            key="oracle.prefill_kv_replica", ranks=tuple(range(4)),
        ),
    ),), coordinator=coordinator)

    def run(max_tokens):
        runtime.replicate(**{**actual, "max_tokens": max_tokens}, plan=plan)

    def check(lengths=(1024, 512)):
        records = out.view(requests * width + 1, page, 288)
        for req, length in enumerate(lengths):
            live = records[1 + req * width:1 + (req + 1) * width].flatten(0, 1)
            torch.testing.assert_close(live[:length], expected[req, :length], rtol=0, atol=0)

    with session.capture():
        starts.copy_(torch.tensor([0, 126, 192], device=device, dtype=torch.int32))
        run(64)  # 16 CTAs, then return to the 64-CTA live domain.
        torch.cuda.synchronize()
        check((63, 33))
        positions.copy_(torch.tensor([2046, 0], device=device, dtype=torch.int64))
        starts.copy_(torch.tensor([0, 2, 1026], device=device, dtype=torch.int32))
        for max_tokens in (1024, 1280, 1536):
            expected.bitwise_xor_(19)
            local_records.copy_(expected[:, owned].view(requests, local_width, page, 288))
            run(max_tokens)
            torch.cuda.synchronize()
            check()
        # vLLM preparation and model profiling use different logical streams.
        # Handoffs must order previous cache producers and replica consumers.
        default = torch.cuda.current_stream(device)
        other = torch.cuda.Stream(device=device)
        other.wait_stream(default)
        with torch.cuda.stream(other):
            expected.bitwise_xor_(23)
            local_records.copy_(expected[:, owned].view(requests, local_width, page, 288))
            run(1024)
        run(1024)  # Native handoff back waits for the other stream's tail.
        torch.cuda.synchronize()
        check()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(1536)
        allocated = torch.cuda.memory_allocated(device)
        for _ in range(3):
            expected.bitwise_xor_(71)
            local_records.copy_(expected[:, owned].view(requests, local_width, page, 288))
            graph.replay()
            torch.cuda.synchronize()
            check()
            assert allocated == torch.cuda.memory_allocated(device)
        graph.reset()
        # The same physical high pages are recycled with a different logical
        # mapping. Graphs must consume the updated table, not a prior chat's IDs.
        table.copy_(table.flip(1))
        cache[base:].view(requests, local_width, page, 288).copy_(
            expected[:, owned].view(requests, local_width, page, 288).flip(1)
        )
        run(1024)
        torch.cuda.synchronize()
        check()
    session.release(plan)
    runtime.close()
    print(f"rank={rank} KV replica byte-exact/high-PID/live-grids/graphs PASS", flush=True)


def main():
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo")
    index_shard_only = "--index-shard-only" in sys.argv
    if index_shard_only:
        device_group = dist.new_group(ranks=list(range(world)), backend="nccl")
        index_group = _OracleDcpGroup(world, rank, device_group)
        check_index_shards(device, index_group)
        dist.destroy_process_group()
        return
    group = SimpleNamespace(
        unique_name="oracle",
        world_size=world,
        cpu_group=dist.group.WORLD,
        ranks=list(range(world)),
    )
    dcp.get_dcp_group = lambda: group
    replica_only = "--kv-replica-only" in sys.argv
    tp_plain_only = "--tp-plain-push-only" in sys.argv
    exchange = (
        None if replica_only or tp_plain_only
        else dcp.DCPExchange.get(64 // world, device)
    )

    def coordinator(progress):
        keys = [requirement.key for requirement in progress.ready_collectives]
        peers = [None] * world
        dist.all_gather_object(peers, keys)
        common = set.intersection(*(set(peer) for peer in peers))
        return min(common) if common else None

    with PreparationSession(
        device=device, autotune=False, compile_workers=2
    ) as session:
        if tp_plain_only:
            check_tp_plain_push(device, session, coordinator)
            dist.destroy_process_group()
            return
        if replica_only:
            check_kv_replica(device, session, coordinator)
            dist.destroy_process_group()
            return
        assert exchange is not None
        session.prepare(exchange.unit.requests, coordinator=coordinator)
        torch.manual_seed(414)
        rows, heads = 32, 64
        source = (
            torch.randn(rows, heads // world, 512, device=device) * 0.25
        ).bfloat16()
        source.add_(rank / 8)
        gathered = torch.empty(rows, heads, 512, device=device, dtype=torch.bfloat16)
        if "--head-gather-only" in sys.argv:
            # Changed wire contract only: no model weights or multi-GiB KV pool.
            assert exchange.plans["all_gather_heads"].query.call["peer_write"] == (
                os.getenv("B12X_PCIE_DCP_HEAD_GATHER_PUSH", "0") == "1"
            )
            with session.capture():
                for live in (1, 6, rows):
                    source.add_(0.125)
                    exchange.gather(source[:live], gathered[:live])
                    torch.cuda.synchronize()
                    peers = [None] * world
                    dist.all_gather_object(peers, source[:live].cpu())
                    torch.testing.assert_close(
                        gathered[:live].cpu(), torch.cat(peers, dim=1), rtol=0, atol=0
                    )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    exchange.gather(source[:6], gathered[:6])
                allocated = torch.cuda.memory_allocated(device)
                output_ptr = gathered.data_ptr()
                for _ in range(3):
                    source.add_(0.25)
                    gathered.zero_()
                    graph.replay()
                    torch.cuda.synchronize()
                    assert torch.cuda.memory_allocated(device) == allocated
                    assert gathered.data_ptr() == output_ptr
                    peers = [None] * world
                    dist.all_gather_object(peers, source[:6].cpu())
                    torch.testing.assert_close(
                        gathered[:6].cpu(), torch.cat(peers, dim=1), rtol=0, atol=0
                    )
                graph.reset()
            for plan in exchange.plans.values():
                session.release(plan)
            exchange.runtime.close()
            dist.destroy_process_group()
            print(f"rank={rank} head gather live-rows/mutation/frozen-graphs PASS", flush=True)
            return
        exchange.gather(source, gathered)
        peers = [None] * world
        dist.all_gather_object(peers, source.cpu())
        torch.testing.assert_close(gathered.cpu(), torch.cat(peers, dim=1))
        main_kv = (torch.randn(1024, 512, device=device) * 0.2 + 0.5).bfloat16()
        swa_kv = (torch.randn(128, 512, device=device) * 0.2 + 0.5).bfloat16()
        full = pack_deepseek_v41_cache_reference(
            main_kv, page_size=128, cache_kind="indexed"
        )
        swa = pack_deepseek_v41_cache_reference(swa_kv, page_size=128, cache_kind="swa")
        owned = torch.arange(1024, device=device) // 64 % world == rank
        packed = pack_deepseek_v41_cache_reference(
            main_kv[owned], page_size=128, cache_kind="indexed"
        )
        # The high physical page forces a >2 GiB native address span.
        base = 2**31 // packed.shape[1] + 3
        cache = torch.empty(
            (base + packed.shape[0], packed.shape[1]), device=device, dtype=torch.uint8
        )
        cache[base:].copy_(packed)
        table = torch.arange(
            base, base + packed.shape[0], device=device, dtype=torch.int32
        ).repeat(rows, 1)
        selected = torch.arange(512, device=device, dtype=torch.int32).repeat(rows, 1)
        # One query has context only on rank zero; one has no indexed context.
        selected[0, 64:] = -1
        selected[1] = -1
        local = torch.empty_like(selected)
        global_lengths = torch.full((rows,), 512, device=device, dtype=torch.int32)
        lengths = torch.full((rows,), 512, device=device, dtype=torch.int32)
        dcp.local_indices(selected, local, lengths,
                          world_size=world, rank=rank, stripe=64)
        swa_ids = torch.arange(128, device=device, dtype=torch.int32).repeat(rows, 1)
        swa_lens = torch.full(
            (rows,), 128 if rank == 0 else 0, device=device, dtype=torch.int32
        )
        sink = torch.full((heads,), 5.0 if rank == 0 else -float("inf"), device=device)
        partial = torch.empty_like(gathered)
        result = torch.empty_like(source)
        for mode in ("decode", "extend"):
            caps = mla.Caps(
                device=device,
                num_q_heads=heads,
                max_q_rows=rows,
                max_width=640,
                swa_width=128,
                indexed_width=512,
                swa_page_size=128,
                indexed_page_size=128,
                max_page_table_width=table.shape[1],
                mode=mode,
                cache_format="deepseek_v41",
                use_cuda_graph=True,
            )
            plan = mla.plan(
                caps,
                invocation=mla.invocation_from_tensors(
                    q=gathered,
                    swa_k_cache=swa,
                    indexed_k_cache=cache,
                    attn_sink=sink,
                    return_lse=True,
                    lse_scale="natural",
                    out=partial,
                ),
            )
            scratch = [
                torch.empty(spec.shape, dtype=spec.dtype, device=device)
                for spec in plan.scratch_specs()
            ]
            kwargs = dict(
                swa_k_cache=swa,
                indexed_k_cache=cache,
                swa_page_size=128,
                indexed_page_size=128,
                sm_scale=512**-0.5,
                attn_sink=sink,
                out=partial,
                cache_format="deepseek_v41",
                return_lse=True,
                lse_scale="natural",
            )
            bind_args = dict(
                scratch=scratch,
                q=gathered,
                swa_indices=swa_ids,
                swa_lengths=swa_lens,
                indexed_indices=local,
                indexed_lengths=lengths,
                indexed_page_table=table,
            )

            def prepare(state, bind_args=bind_args, kwargs=kwargs):
                binding = state.bind_for_preparation(**bind_args)
                return PreparedCall(run=lambda: state.run(binding, **kwargs))

            session.prepare(
                (plan.request(name=f"oracle.{mode}", prepare_call=prepare),)
            )
            binding = mla.bind(plan, **bind_args)

            def run(binding=binding, kwargs=kwargs):
                exchange.gather(source, gathered)
                output, lse = mla.run(binding=binding, **kwargs)
                exchange.reduce(output, lse, result)

            def check(label, mode=mode):
                expected = compressed_sparse_mla_reference(
                    gathered,
                    swa,
                    swa_ids,
                    torch.full_like(swa_lens, 128),
                    sm_scale=512**-0.5,
                    attn_sink=torch.full_like(sink, 5.0),
                    extra_k_cache=full,
                    extra_indices=selected,
                    extra_topk_lengths=global_lengths,
                    swa_page_size=128,
                    extra_page_size=128,
                    cache_format="deepseek_v41",
                )
                expected = expected[
                    :, rank * (heads // world) : (rank + 1) * (heads // world)
                ]
                torch.testing.assert_close(
                    result.float(), expected.float(), rtol=0.03, atol=0.01
                )
                assert torch.isfinite(result).all()
                max_abs = (result.float() - expected.float()).abs().max().item()
                print(
                    f"rank={rank} {mode} {label} PASS max_abs={max_abs:.6f}",
                    flush=True,
                )

            run()
            torch.cuda.synchronize()
            check("eager")
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                run()
            allocated = torch.cuda.memory_allocated(device)
            for replay in range(3):
                source.add_(0.125)
                graph.replay()
                torch.cuda.synchronize()
                assert allocated == torch.cuda.memory_allocated(device)
                check(f"replay{replay}")
            graph.reset()
            session.release(plan)
    dist.barrier()
    exchange.runtime.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
