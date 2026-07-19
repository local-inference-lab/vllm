# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist

import vllm.envs as envs
import vllm.ir.ops
from tests.compile.backend import TestBackend
from vllm.compilation.passes.fusion import allreduce_rms_fusion
from vllm.compilation.passes.fusion.allreduce_rms_fusion import AllReduceFusionPass
from vllm.compilation.passes.utility.fix_functionalization import (
    FixFunctionalizationPass,
)
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    DeviceConfig,
    PassConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.distributed.device_communicators.custom_all_reduce import (
    CustomAllreduce,
    _b12x_pcie_oneshot_limits,
    get_b12x_pcie_allreduce,
)
from vllm.distributed.parallel_state import get_tp_group, graph_capture
from vllm.platforms import current_platform

from ..utils import (
    get_open_port,
    init_test_distributed_environment,
    multi_gpu_test,
)


class B12XFusedAllReduceModel(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))

    def forward(
        self,
        inp: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reduced = tensor_model_parallel_all_reduce(torch.relu(inp))
        return vllm.ir.ops.fused_add_rms_norm(
            reduced,
            residual,
            self.weight,
            1e-6,
        )


def make_b12x_custom_allreduce(
    *,
    allreduce_max_size: int,
    fused_max_size: int,
) -> tuple[CustomAllreduce, MagicMock]:
    runtime = MagicMock()
    runtime.for_stream.return_value.should_allreduce.return_value = True

    custom_allreduce = object.__new__(CustomAllreduce)
    custom_allreduce.disabled = False
    custom_allreduce._pcie_runtime = runtime
    custom_allreduce._pcie_dma = None
    custom_allreduce._pcie_capture_stream = None
    custom_allreduce._pcie_allreduce_max_size = allreduce_max_size
    custom_allreduce._pcie_fused_add_rms_norm_max_size = fused_max_size
    custom_allreduce._pcie_logged_first_allreduce = False
    custom_allreduce._IS_CAPTURING = False
    custom_allreduce._ptr = 0
    custom_allreduce.max_size = max(allreduce_max_size, fused_max_size)
    return custom_allreduce, runtime


def test_b12x_fused_allreduce_uses_independent_cutoff() -> None:
    custom_allreduce, runtime = make_b12x_custom_allreduce(
        allreduce_max_size=16,
        fused_max_size=32,
    )
    inp = torch.randn(2, 4)
    residual = torch.randn_like(inp)
    weight = torch.randn(4)

    assert not custom_allreduce.should_custom_ar(inp)
    assert custom_allreduce.try_fused_add_rms_norm(
        inp,
        residual,
        weight,
        1e-6,
    )
    runtime.all_reduce_fused_add_rms_norm.assert_called_once_with(
        inp,
        residual,
        weight,
        1e-6,
        out=inp,
        residual_out=residual,
        stream=None,
    )


def test_b12x_fused_allreduce_falls_back_above_its_cutoff() -> None:
    custom_allreduce, runtime = make_b12x_custom_allreduce(
        allreduce_max_size=64,
        fused_max_size=16,
    )
    inp = torch.randn(2, 4)

    assert not custom_allreduce.try_fused_add_rms_norm(
        inp,
        torch.randn_like(inp),
        torch.randn(4),
        1e-6,
    )
    runtime.all_reduce_fused_add_rms_norm.assert_not_called()


def test_b12x_fused_allreduce_zero_cutoff_disables_support() -> None:
    custom_allreduce, _ = make_b12x_custom_allreduce(
        allreduce_max_size=64,
        fused_max_size=0,
    )

    assert not custom_allreduce.supports_fused_add_rms_norm()


def test_b12x_oneshot_defaults_to_stream_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLLM_PCIE_ONESHOT_SINGLE_CHANNEL", raising=False)

    assert not envs.environment_variables["VLLM_PCIE_ONESHOT_SINGLE_CHANNEL"]()


def test_b12x_oneshot_buffer_tracks_dispatch_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE", "64KB")
    monkeypatch.setenv(
        "VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE",
        "84KB",
    )

    assert _b12x_pcie_oneshot_limits() == (
        64 * 1024,
        84 * 1024,
        84 * 1024,
    )


def test_b12x_fused_custom_op_dispatch(monkeypatch) -> None:
    custom_allreduce = MagicMock()
    custom_allreduce.try_fused_add_rms_norm.return_value = True
    group = MagicMock()
    monkeypatch.setattr(
        allreduce_rms_fusion,
        "get_b12x_pcie_allreduce",
        lambda: custom_allreduce,
    )
    monkeypatch.setattr(allreduce_rms_fusion, "get_tp_group", lambda: group)

    inp = torch.randn(2, 4)
    residual = torch.randn_like(inp)
    weight = torch.randn(4)
    allreduce_rms_fusion.call_b12x_fused_allreduce_add_rms_norm(
        inp,
        residual,
        weight,
        1e-6,
    )

    custom_allreduce.try_fused_add_rms_norm.assert_called_once_with(
        inp,
        residual,
        weight,
        1e-6,
    )
    group._all_reduce_out_place.assert_not_called()


def test_b12x_fused_custom_op_fallback(monkeypatch) -> None:
    custom_allreduce = MagicMock()
    custom_allreduce.try_fused_add_rms_norm.return_value = False
    group = MagicMock()
    monkeypatch.setattr(
        allreduce_rms_fusion,
        "get_b12x_pcie_allreduce",
        lambda: custom_allreduce,
    )
    monkeypatch.setattr(allreduce_rms_fusion, "get_tp_group", lambda: group)

    inp = torch.randn(2, 4)
    residual = torch.randn_like(inp)
    weight = torch.randn(4)
    reduced = torch.randn_like(inp)
    group._all_reduce_out_place.return_value = reduced
    rms_norm = MagicMock()
    monkeypatch.setattr(allreduce_rms_fusion.ops, "fused_add_rms_norm", rms_norm)

    allreduce_rms_fusion.call_b12x_fused_allreduce_add_rms_norm(
        inp,
        residual,
        weight,
        1e-6,
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
    init_test_distributed_environment(2, 1, rank, str(port), local_rank=rank)
    tp_group = get_tp_group()
    custom_allreduce = get_b12x_pcie_allreduce()
    assert custom_allreduce is not None

    epsilon = 1e-6
    weight = torch.linspace(0.5, 1.5, 6144, dtype=torch.bfloat16, device=device)

    def make_inputs() -> tuple[torch.Tensor, torch.Tensor]:
        inp = torch.full(
            (4, 6144),
            rank + 1,
            dtype=torch.bfloat16,
            device=device,
        )
        residual = torch.linspace(
            -0.5,
            0.5,
            inp.numel(),
            dtype=torch.bfloat16,
            device=device,
        ).view_as(inp)
        return inp, residual

    inp, residual = make_inputs()
    expected, expected_residual = _reference_fused_add_rms_norm(
        inp,
        residual,
        weight,
        tp_group.device_group,
        epsilon,
    )
    torch.ops.vllm.b12x_fused_allreduce_add_rms_norm.default(
        inp,
        residual,
        weight,
        epsilon,
    )
    torch.testing.assert_close(inp, expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(residual, expected_residual)

    config = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.VLLM_COMPILE)
    )
    config.compilation_config.pass_config = PassConfig(fuse_allreduce_rms=True)
    config.device_config = DeviceConfig(device=device)
    config.model_config = MagicMock()
    config.model_config.dtype = torch.bfloat16
    config.model_config.get_hidden_size.return_value = 16
    with set_current_vllm_config(config):
        fusion_pass = AllReduceFusionPass(config)
        backend = TestBackend(
            fusion_pass,
            FixFunctionalizationPass(config),
            PostCleanupPass(config),
        )
        model = B12XFusedAllReduceModel(16).to(
            device=device,
            dtype=torch.bfloat16,
        )
        model_inp = torch.randn(2, 16, dtype=torch.bfloat16, device=device)
        model_residual = torch.randn_like(model_inp)
        expected_model_out = model(model_inp, model_residual)
        compiled_model = torch.compile(model, backend=backend)
        actual_model_out = compiled_model(model_inp, model_residual)
        torch.testing.assert_close(
            actual_model_out,
            expected_model_out,
            atol=2e-2,
            rtol=2e-2,
        )
        assert fusion_pass.matched_count == 1
        backend.check_after_ops(
            [torch.ops.vllm.b12x_fused_allreduce_add_rms_norm.default]
        )

    inp, residual = make_inputs()
    original_inp = inp.clone()
    original_residual = residual.clone()
    expected, expected_residual = _reference_fused_add_rms_norm(
        inp,
        residual,
        weight,
        tp_group.device_group,
        epsilon,
    )
    inp.copy_(original_inp)
    residual.copy_(original_residual)
    with graph_capture(device=device) as capture_context:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_context.stream):
            torch.ops.vllm.b12x_fused_allreduce_add_rms_norm.default(
                inp,
                residual,
                weight,
                epsilon,
            )
    inp.copy_(original_inp)
    residual.copy_(original_residual)
    graph.replay()
    torch.accelerator.synchronize()
    torch.testing.assert_close(inp, expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(residual, expected_residual)


@multi_gpu_test(num_gpus=2)
@pytest.mark.skipif(
    not current_platform.has_device_capability(120),
    reason="B12X fused PCIe all-reduce test requires SM120",
)
def test_b12x_fused_allreduce_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("sparkinfer.comm.pcie")
    monkeypatch.setenv("VLLM_ENABLE_PCIE_ALLREDUCE", "1")
    monkeypatch.setenv("VLLM_PCIE_ALLREDUCE_BACKEND", "b12x")
    monkeypatch.setenv("VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE", "16KB")
    monkeypatch.setenv(
        "VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE",
        "72KB",
    )
    monkeypatch.setenv("VLLM_SKIP_P2P_CHECK", "1")
    torch.multiprocessing.spawn(
        _run_b12x_fused_allreduce_gpu,
        args=(get_open_port(),),
        nprocs=2,
        join=True,
    )
