# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist

from vllm.compilation.passes.fusion import allreduce_rms_fusion
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.device_communicators import b12x_pcie_all_reduce
from vllm.distributed.device_communicators.b12x_pcie_all_reduce import (
    B12xPcieAllReduce,
    _allreduce_max_bytes,
    _dma_min_bytes,
    _oneshot_limits,
    _parse_byte_size,
    get_b12x_pcie_allreduce,
)
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_tp_group,
    graph_capture,
)
from vllm.platforms import current_platform

from ..utils import (
    get_open_port,
    init_test_distributed_environment,
    multi_gpu_test,
)


def _make_communicator(
    *, allreduce_max_bytes: int = 64, fused_max_bytes: int = 64
) -> tuple[B12xPcieAllReduce, MagicMock]:
    runtime = MagicMock()
    runtime.for_stream.return_value.should_allreduce.return_value = True

    communicator = object.__new__(B12xPcieAllReduce)
    communicator.disabled = False
    communicator._runtime = runtime
    communicator._dma = None
    communicator._is_capturing = False
    communicator._capture_stream = None
    communicator.allreduce_max_bytes = allreduce_max_bytes
    communicator.fused_max_bytes = fused_max_bytes
    return communicator, runtime


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("17", 17),
        ("84KB", 84 << 10),
        ("6 MiB", 6 << 20),
        ("2g", 2 << 30),
    ],
)
def test_parse_byte_size(value: str, expected: int) -> None:
    assert _parse_byte_size(value) == expected


def test_oneshot_limits_use_b12x_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE", raising=False)
    recommender = MagicMock(return_value=160 << 10)
    monkeypatch.setattr(
        b12x_pcie_all_reduce,
        "_load_b12x_recommended_max_bytes",
        lambda: recommender,
    )
    monkeypatch.setattr(
        b12x_pcie_all_reduce.envs,
        "VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE",
        "84KB",
    )
    monkeypatch.setattr(
        b12x_pcie_all_reduce.envs,
        "VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE",
        "96KB",
    )

    assert _allreduce_max_bytes(16) == 160 << 10
    assert _oneshot_limits(16) == (160 << 10, 96 << 10, 160 << 10)
    recommender.assert_called_with(16, default=84 << 10)


def test_explicit_oneshot_limit_overrides_b12x_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE", "64KB")
    recommender = MagicMock(return_value=160 << 10)
    monkeypatch.setattr(
        b12x_pcie_all_reduce,
        "_load_b12x_recommended_max_bytes",
        lambda: recommender,
    )

    assert _allreduce_max_bytes(16) == 64 << 10
    recommender.assert_not_called()


def test_dma_crossover_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(b12x_pcie_all_reduce.envs, "VLLM_PCIE_DMA_MIN_BYTES", "24MB")
    assert _dma_min_bytes() == 24 << 20

    for disabled in ("off", " DISABLED ", "NoNe"):
        monkeypatch.setattr(
            b12x_pcie_all_reduce.envs, "VLLM_PCIE_DMA_MIN_BYTES", disabled
        )
        assert _dma_min_bytes() is None


def test_eager_allreduce_dispatches_oneshot() -> None:
    communicator, runtime = _make_communicator()
    inp = torch.randn(2, 4)
    expected = torch.empty_like(inp)
    runtime.all_reduce.return_value = expected

    assert communicator.custom_all_reduce(inp) is expected
    runtime.all_reduce.assert_called_once_with(inp, stream=None)


def test_large_allreduce_dispatches_dma() -> None:
    communicator, runtime = _make_communicator(allreduce_max_bytes=16)
    dma = MagicMock()
    dma.should_allreduce.return_value = True
    expected = torch.empty(16)
    dma.all_reduce.return_value = expected
    communicator._dma = dma
    inp = torch.randn(16)

    assert communicator.custom_all_reduce(inp) is expected
    runtime.all_reduce.assert_not_called()
    dma.all_reduce.assert_called_once_with(inp)


def test_graph_warmup_prepares_oneshot_without_communication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    communicator, runtime = _make_communicator()
    communicator._is_capturing = True
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        b12x_pcie_all_reduce, "_is_piecewise_cudagraph_runtime", lambda: False
    )
    inp = torch.randn(2, 4)

    output = communicator.custom_all_reduce(inp)

    assert output is not None and output.shape == inp.shape
    runtime.prepare_graph_all_reduce.assert_called_once_with(inp, stream=None)
    runtime.all_reduce.assert_not_called()


def test_fused_allreduce_has_an_independent_cutoff() -> None:
    communicator, runtime = _make_communicator(
        allreduce_max_bytes=16, fused_max_bytes=64
    )
    inp = torch.randn(2, 4)
    residual = torch.randn_like(inp)
    weight = torch.randn(4)

    assert not communicator.should_custom_ar(inp)
    assert communicator.try_fused_add_rms_norm(inp, residual, weight, 1e-6)
    runtime.all_reduce_fused_add_rms_norm.assert_called_once_with(
        inp,
        residual,
        weight,
        1e-6,
        out=inp,
        residual_out=residual,
        stream=None,
    )


def test_capture_forwards_the_vllm_stream() -> None:
    communicator, runtime = _make_communicator()
    stream = object()

    @contextmanager
    def capture(*, stream):
        assert stream is not None
        yield

    runtime.capture = capture
    with communicator.capture(stream=stream):
        assert communicator._capture_stream is stream
        assert communicator._is_capturing
    assert communicator._capture_stream is None
    assert not communicator._is_capturing


def test_fused_custom_op_falls_back_atomically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    communicator = MagicMock()
    communicator.try_fused_add_rms_norm.return_value = False
    group = MagicMock()
    reduced = torch.randn(2, 4)
    group._all_reduce_out_place.return_value = reduced
    rms_norm = MagicMock()
    monkeypatch.setattr(
        allreduce_rms_fusion, "get_b12x_pcie_allreduce", lambda: communicator
    )
    monkeypatch.setattr(allreduce_rms_fusion, "get_tp_group", lambda: group)
    monkeypatch.setattr(allreduce_rms_fusion.ops, "fused_add_rms_norm", rms_norm)
    inp = torch.randn_like(reduced)
    residual = torch.randn_like(reduced)
    weight = torch.randn(4)

    allreduce_rms_fusion.call_b12x_fused_allreduce_add_rms_norm(
        inp, residual, weight, 1e-6
    )

    torch.testing.assert_close(inp, reduced)
    rms_norm.assert_called_once_with(inp, residual, weight, 1e-6)


def _reference_fused_add_rms_norm(
    inp: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    group: dist.ProcessGroup,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    reduced = inp.clone()
    dist.all_reduce(reduced, group=group)
    residual_out = (reduced.float() + residual.float()).to(inp.dtype)
    variance = residual_out.float().square().mean(dim=-1, keepdim=True)
    out = residual_out.float() * torch.rsqrt(variance + epsilon)
    return (out * weight.float()).to(inp.dtype), residual_out


def _run_b12x_fused_allreduce_gpu(rank: int, port: int) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    config = VllmConfig()
    config.model_config = MagicMock()
    config.model_config.dtype = torch.bfloat16
    config.model_config.get_hidden_size.return_value = 6144
    with set_current_vllm_config(config):
        init_test_distributed_environment(2, 1, rank, str(port), local_rank=rank)
    tp_group = get_tp_group()
    communicator = get_b12x_pcie_allreduce()
    assert communicator is not None

    epsilon = 1e-6
    weight = torch.linspace(0.5, 1.5, 6144, dtype=torch.bfloat16, device=device)
    inp = torch.full((4, 6144), rank + 1, dtype=torch.bfloat16, device=device)
    residual = torch.linspace(
        -0.5, 0.5, inp.numel(), dtype=torch.bfloat16, device=device
    ).view_as(inp)
    expected, expected_residual = _reference_fused_add_rms_norm(
        inp, residual, weight, tp_group.device_group, epsilon
    )
    original_inp = inp.clone()
    original_residual = residual.clone()

    torch.ops.vllm.b12x_fused_allreduce_add_rms_norm.default(
        inp, residual, weight, epsilon
    )
    torch.testing.assert_close(inp, expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(residual, expected_residual)

    inp.copy_(original_inp)
    residual.copy_(original_residual)
    with graph_capture(device=device) as capture_context:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_context.stream):
            torch.ops.vllm.b12x_fused_allreduce_add_rms_norm.default(
                inp, residual, weight, epsilon
            )
    inp.copy_(original_inp)
    residual.copy_(original_residual)
    graph.replay()
    torch.accelerator.synchronize()
    torch.testing.assert_close(inp, expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(residual, expected_residual)

    dma = communicator._dma
    assert dma is not None
    dma_inp = torch.full((16, 4096), rank + 1, dtype=torch.bfloat16, device=device)
    expected_dma = torch.full_like(dma_inp, 3)
    dma_out = tp_group.device_communicator.all_reduce(dma_inp)
    torch.testing.assert_close(dma_out, expected_dma)

    with graph_capture(device=device) as capture_context:
        dma_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(dma_graph, stream=capture_context.stream):
            dma_out = tp_group.device_communicator.all_reduce(dma_inp)
    dma_graph.replay()
    torch.accelerator.synchronize()
    torch.testing.assert_close(dma_out, expected_dma)

    destroy_model_parallel()
    destroy_distributed_environment()


@multi_gpu_test(num_gpus=2)
@pytest.mark.skipif(
    not current_platform.has_device_capability(120),
    reason="B12X PCIe all-reduce requires SM120",
)
def test_b12x_fused_allreduce_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("b12x.comm.pcie")
    monkeypatch.setenv("VLLM_ENABLE_PCIE_ALLREDUCE", "1")
    monkeypatch.setenv("VLLM_PCIE_ALLREDUCE_BACKEND", "b12x")
    monkeypatch.setenv("VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE", "16KB")
    monkeypatch.setenv("VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE", "72KB")
    monkeypatch.setenv("VLLM_PCIE_DMA_MIN_BYTES", "64KB")
    torch.multiprocessing.spawn(
        _run_b12x_fused_allreduce_gpu,
        args=(get_open_port(),),
        nprocs=2,
        join=True,
    )
