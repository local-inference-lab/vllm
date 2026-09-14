# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native head exchange for owner-sharded DS4.1 compressed KV."""

from dataclasses import replace

import torch

from vllm.distributed import get_dcp_group
from vllm.triton_utils import tl, triton
from vllm.utils.b12x import B12xPreparationUnit


@triton.jit
def _local_indices(
    Indices,
    Out,
    Lengths,
    DCP: tl.constexpr,
    RANK: tl.constexpr,
    STRIPE: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, B)
    length = tl.load(Lengths + row)
    pos = tl.load(Indices + row * 512 + col, col < 512, other=-1)
    owned = (col < length) & (pos >= 0) & ((pos // STRIPE) % DCP == RANK)
    # Compact in the global selection's order, not local numerical-ID order.
    order = tl.sort(tl.where(owned, col, 512 + col), descending=False)
    pos = tl.load(Indices + row * 512 + tl.minimum(order, 511), order < 512, other=-1)
    local = pos // (STRIPE * DCP) * STRIPE + pos % STRIPE
    tl.store(Out + row * 512 + col, tl.where(order < 512, local, -1), col < 512)
    tl.store(Lengths + row, tl.sum(owned.to(tl.int32), axis=0))


def local_indices(indices, out, lengths, *, world_size, rank, stripe):
    """Compact owned selections and replace global lengths with local counts."""
    _local_indices[(indices.shape[0],)](
        indices,
        out,
        lengths,
        world_size,
        rank,
        stripe,
        512,
    )


class DCPExchange:
    """One serial native channel shared by all layers in a DCP group."""

    _instances = {}
    CHUNK = 256

    @classmethod
    def get(cls, local_heads, device):
        group = get_dcp_group()
        key = (group.unique_name, local_heads, device)
        if key not in cls._instances:
            cls._instances[key] = cls(group, local_heads, device)
        return cls._instances[key]

    def preparation_units(self, owner):
        """The first layer helper exposes the shared unit in every stage."""
        if self._preparation_owner is None:
            self._preparation_owner = owner
        return (self.unit,) if self._preparation_owner is owner else ()

    def __init__(self, group, local_heads, device):
        from b12x.comm import pcie
        from b12x.comm.pcie._dcp_preparation import prepare_call
        from b12x.preparation import CollectiveRequirement

        self.runtime = pcie.DcpAllToAll.from_process_group(
            process_group=group.cpu_group,
            device=device,
            max_batch_size=self.CHUNK,
            total_heads=local_heads * group.world_size,
            head_dim=512,
            stream_affine=False,
        )
        self.plans = {}
        self._preparation_owner = None
        requests = []
        # The channel owns maximum-capacity storage; priming runs one row.
        for operation in ("all_gather_heads", "lse_reduce_scatter"):
            heads = (
                local_heads
                if operation == "all_gather_heads"
                else self.runtime.total_heads
            )
            source = torch.empty((1, heads, 512), dtype=torch.bfloat16, device="meta")
            out_heads = (
                self.runtime.total_heads
                if operation == "all_gather_heads"
                else local_heads
            )
            out = torch.empty((1, out_heads, 512), dtype=torch.bfloat16, device="meta")
            if operation == "all_gather_heads":
                call = dict(local_input=source, out=out)
            else:
                call = dict(
                    partial_output=source,
                    partial_lse=torch.empty((1, heads), device="meta"),
                    out=out,
                    is_lse_base_on_e=True,
                )
            query = pcie.query_from_runtime(
                self.runtime,
                surface=f"DcpAllToAll.{operation}",
                call=call,
            )
            plan = pcie.plan(query, runtime=self.runtime)
            self.plans[operation] = plan

            def prepare(state, operation=operation, heads=heads, out_heads=out_heads):
                source = torch.ones(
                    (1, heads, 512), dtype=torch.bfloat16, device=device
                )
                out = torch.empty(
                    (1, out_heads, 512), dtype=torch.bfloat16, device=device
                )
                if operation == "all_gather_heads":
                    call = dict(local_input=source, out=out)
                else:
                    call = dict(
                        partial_output=source,
                        partial_lse=torch.zeros((1, heads), device=device),
                        out=out,
                        is_lse_base_on_e=True,
                    )
                return prepare_call(state, **call)

            requests.append(
                plan.request(
                    name=f"ds41.dcp.{operation}",
                    prepare_call=prepare,
                    collective=CollectiveRequirement(
                        key=f"ds41-dcp:{group.unique_name}:{operation}",
                        ranks=tuple(group.ranks),
                    ),
                )
            )
        self.unit = B12xPreparationUnit(
            name="V41DCP",
            key=(group.unique_name, local_heads),
            requests=tuple(requests),
            stage="weights",
            autotune=False,
        )

    def gather(self, query, out):
        return self.runtime.all_gather_heads(
            query,
            out=out,
            plan=self.plans["all_gather_heads"],
        )

    def reduce(self, partial, lse, out):
        return self.runtime.lse_reduce_scatter(
            partial,
            lse,
            out=out,
            plan=self.plans["lse_reduce_scatter"],
            is_lse_base_on_e=True,
        )


class DCPKVReplica:
    """One packed-KV transport for the full-row encoder's shared cache sources."""

    _instances = {}

    @classmethod
    def get(cls, attention):
        group = get_dcp_group()
        device = attention.attn_sink.device
        key = (group.unique_name, device)
        if key not in cls._instances:
            cls._instances[key] = cls(group, attention, device)
        return cls._instances[key]

    def __init__(self, group, attention, device):
        from b12x.comm import pcie
        from b12x.comm.pcie._owner_preparation import prepared_call
        from b12x.preparation import (
            CollectiveRequirement,
            FrozenMapping,
            MemoryRequirements,
            PersistentMemory,
        )

        if attention.config.parallel_config.use_ubatching:
            raise ValueError("DCP KV replica does not support overlapping microbatches")
        self.max_tokens = (attention.max_model_len + 1) // 2
        self.page_size = attention._main_page
        # Encoder KV-source intervals are serial and do not overlap. Refresh
        # after every source write, rather than retaining one replica per source.
        pages = (self.max_tokens + self.page_size - 1) // self.page_size
        self.output_shape = (
            attention.config.scheduler_config.max_num_seqs * pages + 1,
            self.page_size * 288,
        )
        self.output = None
        self.runtime = pcie.PagedKvReplica.from_process_group(
            process_group=group.cpu_group,
            device=device,
            max_requests=attention.config.scheduler_config.max_num_seqs,
            max_tokens=self.max_tokens,
            stripe_alignment=attention.dcp_stripe // 2,
        )
        query = pcie.query_from_runtime(
            self.runtime,
            surface="PagedKvReplica.replicate",
            call=dict(
                page_size=self.page_size,
                stripe=attention.dcp_stripe // 2,
                ratio=2,
                max_tokens=self.max_tokens,
            ),
        )
        declaration = pcie.plan(query, runtime=self.runtime)
        native_memory = declaration._memory_requirements

        def memory(config, detected):
            requirement = native_memory(config, detected)
            nbytes = self.output_shape[0] * self.output_shape[1]
            return MemoryRequirements(
                scratch=requirement.scratch,
                persistent=(
                    *requirement.persistent,
                    PersistentMemory(
                        key=(self, "prefill_kv"),
                        required_nbytes=nbytes,
                        resident_nbytes=nbytes if self.output is not None else 0,
                    ),
                ),
            )

        self.plan = replace(
            declaration,
            _memory_requirements=memory,
            invocation=FrozenMapping({"vllm_prefill_shape": self.output_shape}),
        )
        self._preparation_owner = attention._helpers

        def prepare(state):
            attention._prepare(device)
            cache = torch.ones(
                (2, self.page_size * 288), dtype=torch.uint8, device=device
            )
            width = (self.runtime.local_capacity + self.page_size - 1) // self.page_size
            table = torch.ones((1, width), dtype=torch.int32, device=device)
            positions = torch.zeros(1, dtype=torch.int64, device=device)
            starts = torch.tensor([0, 2], dtype=torch.int32, device=device)
            return prepared_call(
                state,
                cache=cache,
                table=table,
                positions=positions,
                starts=starts,
                out=self.output,
                requests=1,
                max_tokens=1,
            )

        self.unit = B12xPreparationUnit(
            name="DS41DCPKVReplica",
            key=(group.unique_name, self.max_tokens),
            requests=(
                self.plan.request(
                    name="ds41.dcp.prefill_kv_replica",
                    prepare_call=prepare,
                    collective=CollectiveRequirement(
                        key="ds41.dcp.prefill_kv_replica",
                        ranks=tuple(group.ranks),
                    ),
                ),
            ),
            stage="weights",
            autotune=False,
        )

    def preparation_units(self, owner):
        return (self.unit,) if owner is self._preparation_owner else ()

    def prepare_output(self, device):
        """Allocate the single admitted replica before any graph capture."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("DCP KV replica must be prepared before capture")
        if self.output is None:
            self.output = torch.empty(
                self.output_shape, dtype=torch.uint8, device=device
            )
        return self.output

    def replicate(self, attention, metadata):
        self.runtime.replicate(
            attention.kv_cache,
            metadata.block_table,
            metadata.request_positions,
            metadata.query_start_loc,
            self.output,
            plan=self.plan,
            requests=metadata.num_reqs,
            max_tokens=max(1, (metadata.max_seq_len + 1) // 2),
        )
