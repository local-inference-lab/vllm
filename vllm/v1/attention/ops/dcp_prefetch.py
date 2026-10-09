# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded stream-ordered prefetch for MLA context-parallel KV chunks."""

import functools

import torch

from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["rank_stride", "rank_elements", "elements"])
def _unpack_fp8_packets(
    packets, output, rank_stride, rank_elements, elements, BLOCK: tl.constexpr
):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    rank = index // rank_elements
    offset = index % rank_elements
    base = rank * rank_stride
    raw = tl.load(packets + base + 16 + offset, index < elements, 0)
    scale = tl.load(
        (packets + base).to(tl.pointer_type(tl.float32)), index < elements, 0
    )
    value = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32) * scale
    tl.store(output + index, value, index < elements)


class DCPContextPrefetch:
    """Two bounded KV buffers with explicit producer/consumer stream ordering.

    The metadata builder owns storage and events. Every rank enqueues gathers
    in context-chunk order; a buffer is reusable only after its consumer has
    finished. The transfer stream joins the caller before returning.
    """

    def __init__(self, manager, workspace: torch.Tensor, *, fp8_transport=False):
        gather = manager._kv_gather
        if not (
            isinstance(gather, functools.partial)
            and gather.func is torch.distributed.all_gather_into_tensor
        ):
            raise ValueError("MLA context prefetch requires standard NCCL KV gather")
        self.communicator = manager.group.device_communicator.pynccl_comm
        if self.communicator is None or self.communicator.disabled:
            raise ValueError(
                "MLA context prefetch requires an enabled PyNCCL communicator"
            )
        self.world_size = manager.group.world_size
        self.local_capacity = workspace.shape[0] // (self.world_size + 1)
        self.fp8_transport = fp8_transport
        self.workspace = workspace
        self.width = workspace.shape[-1]
        if fp8_transport:
            if workspace.dtype != torch.bfloat16 or self.width % 16:
                raise ValueError("FP8 KV transport requires aligned BF16 destination")
            self.packet_capacity = self.local_capacity * self.width + 16
            self.buffers = tuple(
                torch.empty(
                    self.packet_capacity * (self.world_size + 1),
                    device=workspace.device,
                    dtype=torch.uint8,
                )
                for _ in range(2)
            )
            for buffer in self.buffers:
                buffer[:16].zero_()
            # Compile the shape-independent unpack helper before graph capture.
            self.buffers[0][: self.width + 16].zero_()
            _unpack_fp8_packets[(triton.cdiv(self.width, 1024),)](
                self.buffers[0],
                workspace,
                self.width + 16,
                self.width,
                self.width,
                BLOCK=1024,
            )
        else:
            self.buffers = (workspace, torch.empty_like(workspace))
        self.stream = torch.cuda.Stream(device=workspace.device)
        self.ready = [torch.cuda.Event(), torch.cuda.Event()]
        self.free = [torch.cuda.Event(), torch.cuda.Event()]

    def gather_chunks(self, chunks, extract_local, scale=None):
        """Yield gathered chunks; extraction writes only the supplied local view."""
        if self.fp8_transport and (
            scale is None
            or scale.dtype != torch.float32
            or scale.numel() != 1
            or scale.device != self.workspace.device
        ):
            raise ValueError("FP8 KV transport requires one device FP32 scale per rank")
        consumer = torch.cuda.current_stream(self.buffers[0].device)
        self.stream.wait_stream(consumer)

        def views(slot, count):
            if self.fp8_transport:
                size = count * self.width + 16
                packet = self.buffers[slot][:size]
                local = packet[16:].view(count, self.width)
                gathered = self.buffers[slot][
                    self.packet_capacity : self.packet_capacity + size * self.world_size
                ]
                return local, packet, gathered
            local = self.buffers[slot][:count]
            gathered = self.buffers[slot][
                self.local_capacity : self.local_capacity + count * self.world_size
            ]
            return local, local, gathered

        def enqueue(index):
            slot = index % 2
            chunk = chunks[index]
            count = chunk.num_local_context_tokens
            if count > self.local_capacity:
                raise ValueError("MLA context chunk exceeds prepared prefetch capacity")
            local, packet, gathered = views(slot, count)
            with torch.cuda.stream(self.stream):
                if index >= 2:
                    self.stream.wait_event(self.free[slot])
                extract_local(chunk, local)
                if self.fp8_transport:
                    packet[:4].copy_(scale.reshape(1).view(torch.uint8))
                self.communicator.all_gather(gathered, packet, stream=self.stream)
                self.ready[slot].record(self.stream)

        if not chunks:
            return
        enqueue(0)
        try:
            for index, chunk in enumerate(chunks):
                slot = index % 2
                consumer.wait_event(self.ready[slot])
                if index + 1 < len(chunks):
                    enqueue(index + 1)
                count = chunk.num_local_context_tokens
                _, _, gathered = views(slot, count)
                if self.fp8_transport:
                    converted = self.workspace[: count * self.world_size]
                    elements = converted.numel()
                    _unpack_fp8_packets[(triton.cdiv(elements, 1024),)](
                        gathered,
                        converted,
                        count * self.width + 16,
                        count * self.width,
                        elements,
                        BLOCK=1024,
                    )
                    gathered = converted
                yield chunk, gathered
                self.free[slot].record(consumer)
        finally:
            consumer.wait_stream(self.stream)
