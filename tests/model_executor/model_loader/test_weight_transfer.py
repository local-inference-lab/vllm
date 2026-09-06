# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight transport preserves routing and ordinary Torch copy semantics."""

import inspect
from contextlib import contextmanager

import pytest
import torch

from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.weight_transfer import (
    allocate_weights,
    copy_weight,
    weight_transfer,
)


def test_writer_receives_final_parameter_slice_and_scope_restores_on_error():
    parameter = torch.zeros(4, 2)
    source = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    calls = []

    def writer(destination, loaded):
        calls.append(destination.data_ptr())
        destination.copy_(loaded + 1)
        return True

    with (
        pytest.raises(RuntimeError, match="abort loading"),
        weight_transfer(writer),
    ):
        default_weight_loader(parameter[2:], source)
        raise RuntimeError("abort loading")
    torch.testing.assert_close(parameter[2:], source + 1)
    assert calls == [parameter[2:].data_ptr()]
    default_weight_loader(parameter[:2], source)
    torch.testing.assert_close(parameter[:2], source)
    assert len(calls) == 1


def test_declined_transfer_preserves_torch_cast_and_broadcast():
    destination = torch.empty((2, 3), dtype=torch.float64)
    with weight_transfer(lambda destination, source: False):
        copy_weight(destination, torch.tensor([1, 2, 3]))
    torch.testing.assert_close(
        destination, torch.tensor([[1, 2, 3], [1, 2, 3]], dtype=torch.float64)
    )


def test_weight_factory_restores_loader_policy_after_failure():
    active = []

    @contextmanager
    def allocator():
        active.append(True)
        try:
            yield
        finally:
            active.pop()

    def factory():
        assert active
        raise RuntimeError("allocation failed")

    with weight_transfer(lambda *_: False, allocator=allocator):
        assert not active
        with pytest.raises(RuntimeError, match="allocation failed"):
            allocate_weights(factory)
        assert not active
    assert allocate_weights(lambda: len(active)) == 0


def test_layerwise_meta_probe_counts_elements_without_io():
    from vllm.model_executor.model_loader.reload.meta import get_numel_loaded

    parameter = torch.empty((2, 3), device="meta")
    source = torch.empty_like(parameter)
    arguments = inspect.signature(default_weight_loader).bind(parameter, source)

    def unexpected_io(destination, source):
        pytest.fail("metadata-only load probes must not submit I/O")

    with weight_transfer(unexpected_io):
        count, _ = get_numel_loaded(default_weight_loader, arguments)
    assert count == 6


def test_composed_loader_waits_for_queued_reads_before_transforming():
    from vllm.model_executor.model_loader.weight_utils import composed_weight_loader

    class DeferredWriter:
        def __call__(self, destination, source):
            self.pending = (destination, source)
            return True

        def flush(self):
            destination, source = self.pending
            destination.copy_(source)

    parameter = torch.zeros(4)
    loader = composed_weight_loader(default_weight_loader, lambda value: value * 2)
    with weight_transfer(DeferredWriter()):
        loader(parameter, torch.arange(4, dtype=torch.float32))
    torch.testing.assert_close(parameter, torch.tensor([0.0, 2.0, 4.0, 6.0]))
