# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest
import torch

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce


def _make_custom_allreduce(
    *, allreduce_max_size: int
) -> tuple[CustomAllreduce, MagicMock]:
    runtime = MagicMock()
    runtime.for_stream.return_value.should_allreduce.return_value = True

    custom_allreduce = object.__new__(CustomAllreduce)
    custom_allreduce.disabled = False
    custom_allreduce._pcie_runtime = runtime
    custom_allreduce._pcie_dma = None
    custom_allreduce._pcie_capture_stream = None
    custom_allreduce._pcie_allreduce_max_size = allreduce_max_size
    custom_allreduce._pcie_logged_first_allreduce = False
    custom_allreduce._IS_CAPTURING = True
    custom_allreduce._ptr = 0
    custom_allreduce.max_size = allreduce_max_size
    return custom_allreduce, runtime


def _mock_capture_warmup(
    monkeypatch: pytest.MonkeyPatch,
    custom_allreduce: CustomAllreduce,
) -> object:
    capture_stream = object()
    monkeypatch.setattr(
        custom_allreduce,
        "_pcie_runtime_stream",
        lambda: capture_stream,
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        "vllm.distributed.device_communicators.custom_all_reduce."
        "_is_piecewise_cudagraph_runtime",
        lambda: False,
    )
    return capture_stream


def test_capture_warmup_prepares_plain_graph_allreduce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_allreduce, runtime = _make_custom_allreduce(allreduce_max_size=64)
    capture_stream = _mock_capture_warmup(monkeypatch, custom_allreduce)
    inp = torch.randn(2, 4)

    output = custom_allreduce.custom_all_reduce(inp)

    assert output is not None
    assert output.shape == inp.shape
    runtime.prepare_graph_all_reduce.assert_called_once_with(
        inp,
        stream=capture_stream,
    )
    runtime.all_reduce.assert_not_called()


def test_capture_warmup_does_not_prepare_dma_only_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_allreduce, runtime = _make_custom_allreduce(allreduce_max_size=16)
    custom_allreduce._pcie_dma = MagicMock()
    custom_allreduce._pcie_dma.should_allreduce.return_value = True
    _mock_capture_warmup(monkeypatch, custom_allreduce)
    inp = torch.randn(4, 4)

    output = custom_allreduce.custom_all_reduce(inp)

    assert output is not None
    runtime.prepare_graph_all_reduce.assert_not_called()
