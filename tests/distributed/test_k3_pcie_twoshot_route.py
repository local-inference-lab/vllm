# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Size routing of the Kimi-K3 PCIe all-reduce wrapper with the lossless bf16
two-shot: one-shot below its ceiling, two-shot up to its limit, the DMA ring
above, and no custom path when nothing accepts the tensor. Device-free: the
runtimes are mocks and the tensors live on the CPU."""

from unittest.mock import MagicMock

import pytest
import torch

from vllm.distributed.device_communicators import custom_all_reduce as car

ONESHOT_MAX = 112 << 10
TWOSHOT_MAX = 768 << 10


def _wrapper(*, twoshot: bool = True, dma: bool = True) -> car.CustomAllreduce:
    w = car.CustomAllreduce.__new__(car.CustomAllreduce)
    w.disabled = False
    w.world_size = 8
    w._IS_CAPTURING = False
    w._pcie_capture_stream = None
    w._pcie_capture_channel_id = None
    w._pcie_logged_first_allreduce = True
    w._pcie_allreduce_max_size = ONESHOT_MAX
    runtime = MagicMock()
    runtime.for_stream.return_value.should_allreduce.side_effect = lambda t: (
        t.nbytes <= ONESHOT_MAX
    )
    runtime.all_reduce.side_effect = lambda inp, **kw: inp * 8
    w._pcie_runtime = runtime
    if twoshot:
        ts = MagicMock()
        ts.accepts.side_effect = lambda t: t.dtype == torch.bfloat16
        ts.all_reduce.side_effect = lambda inp, out=None: inp * 8
        w._pcie_twoshot = ts
        w._pcie_twoshot_max_bytes = TWOSHOT_MAX
    else:
        w._pcie_twoshot = None
        w._pcie_twoshot_max_bytes = 0
    if dma:
        d = MagicMock()
        d.should_allreduce.side_effect = lambda t: t.nbytes >= (6 << 20)
        d.all_reduce.side_effect = lambda inp, out=None: inp * 8
        w._pcie_dma = d
    else:
        w._pcie_dma = None
    return w


def _payload(tokens: int) -> torch.Tensor:
    return torch.ones(tokens, 7168, dtype=torch.bfloat16)


def test_twoshot_limit_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE", raising=False)
    assert car._b12x_pcie_twoshot_max_bytes() == 768 << 10
    monkeypatch.setenv("VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE", "2MB")
    assert car._b12x_pcie_twoshot_max_bytes() == 2 << 20
    for disabled in ("0", "off", "none", ""):
        monkeypatch.setenv("VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE", disabled)
        assert car._b12x_pcie_twoshot_max_bytes() == 0
    monkeypatch.delenv("VLLM_PCIE_TWOSHOT_ROW_ELEMS", raising=False)
    assert car._b12x_pcie_twoshot_row_elems() == 896


@pytest.mark.parametrize(
    ("tokens", "route"),
    [
        (4, "oneshot"),
        (8, "oneshot"),  # 114,688 bytes: exactly the one-shot ceiling
        (9, "twoshot"),
        (16, "twoshot"),
        (48, "twoshot"),  # 688 KB
        (56, "nccl"),  # 802 KB: above the two-shot limit, below the DMA floor
        (512, "dma"),
    ],
)
def test_size_routing(tokens: int, route: str) -> None:
    w = _wrapper()
    inp = _payload(tokens)
    assert w.should_custom_ar(inp) == (route != "nccl")
    if route == "nccl":
        return
    out = w.all_reduce(inp)
    assert torch.equal(out, inp * 8)
    assert w._pcie_runtime.all_reduce.called == (route == "oneshot")
    assert w._pcie_twoshot.all_reduce.called == (route == "twoshot")
    assert w._pcie_dma.all_reduce.called == (route == "dma")


def test_twoshot_respects_runtime_acceptance() -> None:
    w = _wrapper()
    inp = torch.ones(9, 7168, dtype=torch.float16)  # the two-shot serves bf16 only
    assert not w.should_custom_ar(inp)


def test_without_twoshot_midsize_stays_on_nccl() -> None:
    w = _wrapper(twoshot=False)
    assert not w.should_custom_ar(_payload(9))
    assert w.should_custom_ar(_payload(8))
