# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native head exchange for owner-sharded DS4.1 compressed KV."""

import torch

from vllm.distributed import get_dcp_group
from vllm.triton_utils import tl, triton
from vllm.utils.b12x import B12xPreparationUnit


@triton.jit
def _local_indices(
    Indices,
    Out,
    DCP: tl.constexpr,
    RANK: tl.constexpr,
    STRIPE: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, B)
    pos = tl.load(Indices + row * 512 + col, col < 512, other=-1)
    owned = (pos >= 0) & ((pos // STRIPE) % DCP == RANK)
    local = pos // (STRIPE * DCP) * STRIPE + pos % STRIPE
    tl.store(Out + row * 512 + col, tl.where(owned, local, -1), col < 512)


def local_indices(indices, out, *, world_size, rank, stripe):
    """Preserve selected order, masking positions owned by other ranks."""
    _local_indices[(indices.shape[0],)](
        indices,
        out,
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
            source = torch.empty((1, heads, 512), dtype=torch.bfloat16, device=device)
            out_heads = (
                self.runtime.total_heads
                if operation == "all_gather_heads"
                else local_heads
            )
            out = torch.empty((1, out_heads, 512), dtype=torch.bfloat16, device=device)
            if operation == "all_gather_heads":
                call = dict(local_input=source, out=out)
            else:
                call = dict(
                    partial_output=source,
                    partial_lse=torch.zeros((1, heads), device=device),
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

            def prepare(state, call=call, source=source):
                source.fill_(1)
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
