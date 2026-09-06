# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scoped transport for checkpoint copies into model parameter views."""

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from typing import Any, TypeVar

import torch

WeightWriter = Callable[[torch.Tensor, torch.Tensor], bool]
_writer: ContextVar[WeightWriter | None] = ContextVar("weight_writer", default=None)
_allocator: ContextVar[Callable[[], AbstractContextManager] | None] = ContextVar(
    "weight_allocator", default=None
)
_T = TypeVar("_T")


@contextmanager
def weight_transfer(
    writer: WeightWriter,
    *,
    allocator: Callable[[], AbstractContextManager] | None = None,
) -> Iterator[None]:
    """Install a synchronous writer during exclusive model weight loading.

    Return True after accepting a copy, retaining its owners until completion.
    Call flush_weight_transfers before consuming queued parameter values.
    Return False to use Torch's usual copy semantics.
    """
    token = _writer.set(writer)
    allocation_token = _allocator.set(allocator)
    try:
        yield
    finally:
        _allocator.reset(allocation_token)
        _writer.reset(token)


def allocate_weights(factory: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
    """Create checkpoint destinations using the active loader's allocation policy.

    The factory must only allocate weights. Runtime state, outputs and scratch
    use ordinary allocation, including during model construction and preparation.
    """
    allocator = _allocator.get()
    with allocator() if allocator is not None else nullcontext():
        return factory(*args, **kwargs)


def copy_weight(destination: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    writer = _writer.get()
    if destination.is_meta or writer is None or not writer(destination, source):
        destination.copy_(source)
    return destination


def flush_weight_transfers() -> None:
    """Complete queued checkpoint reads before numerical weight preparation."""
    writer = _writer.get()
    flush = getattr(writer, "flush", None)
    if flush is not None:
        flush()


def materialize_weight(source: torch.Tensor) -> torch.Tensor:
    """Own checkpoint values needed by a numerical loading transform."""
    writer = _writer.get()
    materialize = getattr(writer, "materialize", None)
    return materialize(source) if materialize is not None else source.clone()
