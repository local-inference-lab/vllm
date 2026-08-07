# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stream-isolated adapter for FlashInfer's PCIe IPC all-reduce."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup


@lru_cache(maxsize=1)
def _load_workspace_class() -> type:
    from flashinfer.comm import PcieIpcAllReduceWorkspace

    return PcieIpcAllReduceWorkspace


class _FlashInferPcieChannel:
    def __init__(
        self,
        owner: FlashInferPcieIpcAllReducePool,
        channel_id: str,
    ) -> None:
        self._owner = owner
        self._channel_id = channel_id

    def should_allreduce(self, inp: torch.Tensor) -> bool:
        return self._owner._workspace(self._channel_id).supports(inp)

    def register_graph_buffers(self) -> None:
        # FlashInfer owns fixed IPC slabs; graph replay does not register input
        # pointers with the legacy vLLM custom-allreduce runtime.
        return None


class FlashInferPcieIpcAllReducePool:
    """Give every eager/CUDA-graph owner a distinct FlashInfer workspace.

    FlashInfer's protocol state is stream-affine. vLLM may own an eager stream
    plus independent target and draft CUDA-graph streams, so sharing one
    workspace would permit their epochs to interleave. Semantic ``channel_id``
    values provide a rank-stable key and one IPC workspace is allocated for
    each live channel.
    """

    _EAGER_CHANNEL_ID = "vllm:eager:allreduce"

    def __init__(
        self,
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        max_size: int,
        single_channel: bool = False,
    ) -> None:
        self.group = exchange_group
        self.device = torch.device(device)
        self.rank = dist.get_rank(group=exchange_group)
        self.world_size = dist.get_world_size(group=exchange_group)
        self.max_size = max_size
        self.max_numel = max_size // torch.bfloat16.itemsize
        self.single_channel = single_channel
        self._workspaces: dict[str, Any] = {}
        self._active_channel_id: str | None = None
        self._closed = False

        if max_size <= 0 or max_size % 16:
            raise ValueError(
                "FlashInfer PCIe IPC max_size must be a positive multiple "
                f"of 16 bytes, got {max_size}"
            )
        # Workspace creation is collective. Keep construction allocation-free so
        # the integration layer can prepare semantic channels at a rank-stable
        # lifecycle point; standalone callers may still create them on first use.

    @classmethod
    def from_exchange_group(
        cls,
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        eager_buffer_bytes: int,
        max_size: int,
        single_channel: bool = False,
        max_concurrent_channels: int | None = None,
        **_: Any,
    ) -> FlashInferPcieIpcAllReducePool:
        del eager_buffer_bytes, max_concurrent_channels
        return cls(
            exchange_group=exchange_group,
            device=device,
            max_size=max_size,
            single_channel=single_channel,
        )

    def _canonical_channel_id(self, channel_id: str | None) -> str:
        if self.single_channel or channel_id is None:
            return self._EAGER_CHANNEL_ID
        return channel_id

    def _verify_channel_id(self, channel_id: str) -> None:
        gathered: list[str | None] = [None] * self.world_size
        dist.all_gather_object(gathered, channel_id, group=self.group)
        if any(peer != channel_id for peer in gathered):
            raise RuntimeError(
                "FlashInfer PCIe IPC channel creation must be rank-stable; "
                f"received {gathered}"
            )

    def _workspace(self, channel_id: str | None) -> Any:
        if self._closed:
            raise RuntimeError("FlashInfer PCIe IPC pool is closed")
        key = self._canonical_channel_id(channel_id)
        workspace = self._workspaces.get(key)
        if workspace is None:
            self._verify_channel_id(key)
            workspace = _load_workspace_class()(
                group=self.group,
                max_numel=self.max_numel,
                dtype=torch.bfloat16,
            )
            self._workspaces[key] = workspace
        return workspace

    def prepare_channels(self, channel_ids: Sequence[str]) -> None:
        for channel_id in channel_ids:
            self._workspace(channel_id)

    def for_stream(
        self,
        stream: torch.cuda.Stream | None = None,
        *,
        channel_id: str | None = None,
    ) -> _FlashInferPcieChannel:
        del stream
        key = self._canonical_channel_id(channel_id)
        self._workspace(key)
        return _FlashInferPcieChannel(self, key)

    @contextmanager
    def capture(
        self,
        stream: torch.cuda.Stream | None = None,
        *,
        channel_id: str | None = None,
    ) -> Iterator[FlashInferPcieIpcAllReducePool]:
        del stream
        key = self._canonical_channel_id(channel_id)
        self._workspace(key)
        previous = self._active_channel_id
        self._active_channel_id = key
        try:
            yield self
        finally:
            self._active_channel_id = previous

    def prepare_graph_all_reduce(
        self,
        inp: torch.Tensor,
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        del stream
        channel_id = self._active_channel_id or self._EAGER_CHANNEL_ID
        if not self._workspace(channel_id).supports(inp):
            raise ValueError(
                "FlashInfer PCIe IPC graph warmup received an unsupported "
                f"shape {tuple(inp.shape)}"
            )

    def all_reduce(
        self,
        inp: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
        stream: torch.cuda.Stream | None = None,
        channel_id: str | None = None,
    ) -> torch.Tensor:
        workspace = self._workspace(channel_id)
        if stream is None or torch.cuda.current_stream(self.device) == stream:
            return workspace.all_reduce(inp, out=out)
        with torch.cuda.stream(stream):
            return workspace.all_reduce(inp, out=out)

    def checkpoint_channels(self) -> tuple[str, ...]:
        return tuple(self._workspaces)

    def rollback_channels(self, checkpoint: tuple[str, ...]) -> None:
        retained = set(checkpoint)
        for channel_id in reversed(tuple(self._workspaces)):
            if channel_id in retained:
                continue
            self._workspaces.pop(channel_id).destroy()

    def close(self) -> None:
        if self._closed:
            return
        for workspace in reversed(tuple(self._workspaces.values())):
            workspace.destroy()
        self._workspaces.clear()
        self._closed = True
