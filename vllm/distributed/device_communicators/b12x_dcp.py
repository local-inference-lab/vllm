# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepared PCIe query exchange for serialized MLA execution.

The process-group owner retains separate eager and graph IPC channels. Model
layers share transport geometry, but every invocation supplies its own output.
Concurrent microbatches are unsupported: the caller must serialize replay and
eager work on each channel, as the single-microbatch model runner does.
"""

from functools import partial

import torch

from vllm.utils.b12x import (
    B12xPreparationUnit,
    register_b12x_unit_provider,
)


class B12xDCPTransport:
    def __init__(
        self,
        group,
        device,
        max_tokens,
        num_heads,
        query_dim,
        output_dim,
        query_dtype,
        output_dtype,
        operations=("all_gather_heads", "lse_reduce_scatter"),
    ):
        from b12x.comm.pcie import DcpAllToAll

        self.group = group
        self.device = torch.device(device)
        self.max_tokens = max_tokens
        self.num_heads = num_heads
        self.query_dim = query_dim
        self.output_dim = output_dim
        self.query_dtype = query_dtype
        self.output_dtype = output_dtype
        self.operations = operations
        self.total_heads = num_heads * group.world_size
        self.identity = (
            f"dcp:{','.join(map(str, group.ranks))}:"
            f"{max_tokens}:{num_heads}:{query_dim}:{output_dim}:"
            f"{query_dtype}:{output_dtype}:{','.join(operations)}"
        )
        self.channels = {}
        self.plans = {}
        try:
            for name in ("eager", "graph"):
                self.channels[name] = DcpAllToAll.from_process_group(
                    process_group=group.cpu_group,
                    device=self.device,
                    max_batch_size=max_tokens,
                    total_heads=self.total_heads,
                    head_dim=output_dim,
                    query_head_dim=query_dim,
                    stream_affine=False,
                )
        except Exception:
            self.close()
            raise
        register_b12x_unit_provider(self)

    def _example_call(self, operation):
        if operation == "all_gather_heads":
            return {
                "local_input": torch.zeros(
                    (self.max_tokens, self.num_heads, self.query_dim),
                    dtype=self.query_dtype,
                    device=self.device,
                ),
                "out": torch.empty(
                    (self.max_tokens, self.total_heads, self.query_dim),
                    dtype=self.query_dtype,
                    device=self.device,
                ),
            }
        return {
            "partial_output": torch.ones(
                (self.max_tokens, self.total_heads, self.output_dim),
                dtype=self.output_dtype,
                device=self.device,
            ),
            "partial_lse": torch.zeros(
                (self.max_tokens, self.total_heads),
                dtype=torch.float32,
                device=self.device,
            ),
            "out": torch.empty(
                (self.max_tokens, self.num_heads, self.output_dim),
                dtype=self.output_dtype,
                device=self.device,
            ),
        }

    def _prepare_call(self, operation, state):
        from b12x.comm.pcie._dcp_preparation import prepare_call

        return prepare_call(state, **self._example_call(operation))

    def get_b12x_preparation_units(self, owner, workload):
        if owner is not self:
            raise ValueError("DCP preparation owner mismatch")
        if workload.stage != "weights" or workload.eager_only:
            return ()
        from b12x.comm.pcie import plan, query_from_runtime
        from b12x.preparation import CollectiveRequirement

        requests = []
        for channel_name, channel in self.channels.items():
            for operation in self.operations:
                key = (channel_name, operation)
                if key not in self.plans:
                    query = query_from_runtime(
                        channel,
                        surface=f"DcpAllToAll.{operation}",
                        call=self._example_call(operation),
                    )
                    self.plans[key] = plan(query, runtime=channel)
                name = f"{self.identity}:{channel_name}:{operation}"
                requests.append(
                    self.plans[key].request(
                        name=name,
                        prepare_call=partial(self._prepare_call, operation),
                        collective=CollectiveRequirement(
                            key=name,
                            ranks=tuple(sorted(self.group.ranks)),
                        ),
                    )
                )
        return (
            B12xPreparationUnit(
                name="PCIE_DCP",
                key=self.identity,
                requests=tuple(requests),
                stage="weights",
                autotune=False,
            ),
        )

    def _channel(self, operation):
        name = "graph" if torch.cuda.is_current_stream_capturing() else "eager"
        return self.channels[name], self.plans[name, operation]

    def gather(self, query):
        if query.dtype != self.query_dtype:
            raise ValueError("DCP query dtype differs from its prepared plan")
        channel, plan = self._channel("all_gather_heads")
        output = torch.empty(
            (query.shape[0], self.total_heads, self.query_dim),
            dtype=query.dtype,
            device=query.device,
        )
        return channel.all_gather_heads(query, output, plan=plan)

    def combine(self, output, lse, *, is_lse_base_on_e):
        if output.dtype != self.output_dtype:
            raise ValueError("DCP output dtype differs from its prepared plan")
        channel, plan = self._channel("lse_reduce_scatter")
        result = torch.empty(
            (output.shape[0], self.num_heads, self.output_dim),
            dtype=output.dtype,
            device=output.device,
        )
        return channel.lse_reduce_scatter(
            output,
            lse.contiguous(),
            result,
            plan=plan,
            is_lse_base_on_e=is_lse_base_on_e,
        )

    def close(self):
        for channel in self.channels.values():
            channel.close()
        self.channels.clear()
        self.plans.clear()


def get_b12x_dcp_transport(
    group,
    device,
    max_tokens,
    num_heads,
    query_dim,
    output_dim,
    query_dtype,
    output_dtype,
):
    """Return group-owned IPC state for one fixed collective geometry."""
    from b12x.comm.pcie import is_supported

    if group.world_size not in (2, 4, 8, 9, 10, 12, 16) or not is_supported(device):
        return None
    if query_dtype not in (torch.float16, torch.bfloat16, torch.float8_e4m3fn):
        return None
    if output_dtype not in (torch.float16, torch.bfloat16):
        return None
    communicator = group.device_communicator
    owners = getattr(communicator, "b12x_dcp_transports", None)
    if owners is None:
        owners = {}
        communicator.b12x_dcp_transports = owners
    key = (max_tokens, num_heads, query_dim, output_dim, query_dtype, output_dtype)
    if key not in owners:
        owners[key] = B12xDCPTransport(
            group,
            device,
            *key,
        )
    return owners[key]


class B12xKimiProjectionTransport(B12xDCPTransport):
    """Gather Kimi latent/router shards and select experts with retained programs."""

    returns_logits = False

    def __init__(self, group, device):
        super().__init__(
            group,
            device,
            8,
            1,
            (3584 * 2 + 896 * 4) // group.world_size,
            8,
            torch.bfloat16,
            torch.bfloat16,
            operations=("all_gather_pair", "all_gather_pair_kimi_topk", "kimi_topk16"),
        )

    def _example_call(self, operation):
        rows = 1 if operation == "all_gather_pair_kimi_topk" else self.max_tokens
        down = torch.zeros(
            rows,
            3584 // self.group.world_size,
            device=self.device,
            dtype=torch.bfloat16,
        )
        router = torch.zeros(
            rows, 896 // self.group.world_size, device=self.device, dtype=torch.float32
        )
        output = torch.empty(rows, 3584, device=self.device, dtype=torch.bfloat16)
        bias = torch.zeros(896, device=self.device, dtype=torch.float32)
        weights = torch.empty(rows, 16, device=self.device, dtype=torch.float32)
        ids = torch.empty(rows, 16, device=self.device, dtype=torch.int32)
        if operation == "all_gather_pair":
            return dict(
                local_first=down,
                local_second=router,
                out_first=output,
                out_second=torch.empty(
                    rows, 896, device=self.device, dtype=torch.float32
                ),
                threads=512,
            )
        if operation == "all_gather_pair_kimi_topk":
            return dict(
                local_down=down,
                local_router=router,
                correction_bias=bias,
                out_down=output,
                topk_weights=weights,
                topk_ids=ids,
            )
        return dict(
            router_logits=torch.zeros(
                rows, 896, device=self.device, dtype=torch.float32
            ),
            correction_bias=bias,
            output_weights=weights,
            output_ids=ids,
        )

    def gather_projections(self, local_down, local_router, bias):
        """Return latent rows and float32 [weights; int32-id bits] routing rows."""
        rows = local_down.shape[0]
        down = local_down.new_empty(rows, 3584)
        payload = local_router.new_empty(2 * rows, 16)
        weights, ids = payload[:rows], payload[rows:].view(torch.int32)
        if rows == 1:
            channel, plan = self._channel("all_gather_pair_kimi_topk")
            channel.all_gather_pair_kimi_topk(
                local_down,
                local_router,
                bias,
                down,
                weights,
                ids,
                plan=plan,
            )
        else:
            logits = local_router.new_empty(rows, 896)
            channel, plan = self._channel("all_gather_pair")
            channel.all_gather_pair(
                local_down,
                local_router,
                down,
                logits,
                plan=plan,
                threads=512,
            )
            channel, plan = self._channel("kimi_topk16")
            channel.kimi_topk16(logits, bias, weights, ids, plan=plan)
        return down, payload


class B12xKimiPaddedProjectionTransport(B12xDCPTransport):
    """Exchange uneven TP shards without changing BF16/FP32 projection bits."""

    returns_logits = True

    def __init__(self, group, device):
        self.down_width = (3584 + group.world_size - 1) // group.world_size
        self.router_width = (896 + group.world_size - 1) // group.world_size
        # FP32 logits occupy two opaque BF16 words; transport never casts them.
        words = self.down_width + 2 * self.router_width
        self.packed_words = (words + 7) // 8 * 8
        super().__init__(
            group,
            device,
            8,
            1,
            self.packed_words,
            8,
            torch.bfloat16,
            torch.bfloat16,
            operations=("all_gather_heads",),
        )

    def gather_projections(self, local_down, local_router, bias):
        """Return contiguous latent rows and unmodified FP32 routing logits."""
        rows = local_down.shape[0]
        if (
            local_down.shape != (rows, self.down_width)
            or local_router.shape != (rows, self.router_width)
            or local_down.dtype != torch.bfloat16
            or local_router.dtype != torch.float32
            or not local_down.is_contiguous()
            or not local_router.is_contiguous()
        ):
            raise ValueError(
                "Kimi padded projection shards differ from prepared geometry"
            )
        packed = local_down.new_zeros(rows, 1, self.packed_words)
        packed[:, 0, : self.down_width].copy_(local_down)
        packed[:, 0, self.down_width : self.down_width + 2 * self.router_width].copy_(
            local_router.view(torch.bfloat16)
        )
        gathered = self.gather(packed)
        down = gathered[:, :, : self.down_width].reshape(
            rows, self.group.world_size * self.down_width
        )
        router_bits = gathered[
            :, :, self.down_width : self.down_width + 2 * self.router_width
        ].contiguous()
        router = router_bits.view(torch.float32).reshape(
            rows, self.group.world_size * self.router_width
        )
        return down[:, :3584].contiguous(), router[:, :896].contiguous()


def get_b12x_kimi_projection_transport(group, device, *, latent_width=None):
    """Create projection IPC state during model construction, never in forward."""
    from b12x.comm.pcie import is_supported

    if group.world_size not in (2, 4, 8, 9, 10, 12, 16) or not is_supported(device):
        return None
    padded = group.world_size in (9, 10, 12)
    if padded and latent_width is not None:
        return None
    if latent_width is not None and latent_width % 8:
        return None
    communicator = group.device_communicator
    owners = getattr(communicator, "b12x_dcp_transports", None)
    if owners is None:
        owners = {}
        communicator.b12x_dcp_transports = owners
    key = ("kimi_projection", latent_width)
    projection_type = (
        B12xKimiPaddedProjectionTransport if padded else B12xKimiProjectionTransport
    )
    if key not in owners:
        owners[key] = (
            projection_type(group, device)
            if latent_width is None
            else B12xDCPTransport(
                group,
                device,
                8,
                1,
                latent_width,
                8,
                torch.bfloat16,
                torch.bfloat16,
                operations=("all_gather_heads",),
            )
        )
    return owners[key]
