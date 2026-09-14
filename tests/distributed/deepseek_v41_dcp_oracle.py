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
    group = SimpleNamespace(
        unique_name="oracle",
        world_size=world,
        cpu_group=dist.group.WORLD,
        ranks=list(range(world)),
    )
    dcp.get_dcp_group = lambda: group
    replica_only = "--kv-replica-only" in sys.argv
    exchange = None if replica_only else dcp.DCPExchange.get(64 // world, device)

    def coordinator(progress):
        keys = [requirement.key for requirement in progress.ready_collectives]
        peers = [None] * world
        dist.all_gather_object(peers, keys)
        common = set.intersection(*(set(peer) for peer in peers))
        return min(common) if common else None

    with PreparationSession(
        device=device, autotune=False, compile_workers=2
    ) as session:
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

            def prepare(state):
                binding = state.bind_for_preparation(**bind_args)
                return PreparedCall(run=lambda: state.run(binding, **kwargs))

            session.prepare(
                (plan.request(name=f"oracle.{mode}", prepare_call=prepare),)
            )
            binding = mla.bind(plan, **bind_args)

            def run():
                exchange.gather(source, gathered)
                output, lse = mla.run(binding=binding, **kwargs)
                exchange.reduce(output, lse, result)

            def check(label):
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
