# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-rank startup oracle for the b12x preparation lifecycle.

Runs on two GPUs without a checkpoint. One compiled module chains a b12x MXFP8
linear reached through the layer-name op, the splitting attention op, and the
fused b12x all-reduce plus add-RMSNorm op. Both ranks run the driver's weights
stage (sharded timing with winner exchange), freeze the session, size and lock
the workspace, capture piecewise graphs under the no-compilation guard, and
replay them with changed inputs. Invariants are gathered across ranks.

The parent process runs a watchdog over per-rank heartbeats: a stalled or dead
rank has both ranks dumped with py-spy and terminated, so a rank-divergent hang
fails within the timeout instead of blocking. Injected faults on rank 1 prove
that: workspace growth under capture, a Triton JIT under capture, and a skipped
collective must each surface as a bounded failure.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from vllm.platforms import current_platform

from ..utils import get_open_port, multi_gpu_test

HIDDEN = 256
# Captured sizes, the batch maximum above them, and eager-only counts that no
# layer declares: a request between capture sizes or above the largest one
# runs eagerly at its exact count and must resolve to prepared state.
SIZES = (1, 2, 4, 8)
MAX_TOKENS = 16
UNDECLARED_SIZES = (11, 3)
EPSILON = 1e-6
STALL_SECONDS = 60.0
CAPTURE_STEP_SECONDS = 10.0


def _heartbeat(beats, rank):
    beats[rank] = time.monotonic()


def _check_linear(kernel, layer, x, weight_fp8):
    """The block-scaled GEMM against an fp32 reference, at the kernel's own tolerance.

    Returns the kernel's output so the downstream stages are checked against
    what they actually consumed; a split-K default accumulates in BF16 and
    its rounding must not be charged to the collective or the norm.
    """
    local = kernel.apply_weights(layer, x)
    expected = x.float() @ weight_fp8.float().T
    error = torch.linalg.vector_norm(local.float() - expected)
    assert float(error / torch.linalg.vector_norm(expected).clamp_min(1e-12)) < 0.004
    torch.testing.assert_close(
        local.float(), expected, atol=float(expected.abs().max()) * 0.008 + 1e-6, rtol=0.008,
    )
    return local


def _reference(local, residual, norm_weight, group):
    attention = (local + local + local)
    reduced = attention.float().clone()
    dist.all_reduce(reduced, group=group)
    residual_out = (reduced + residual.float()).bfloat16()
    variance = residual_out.float().square().mean(dim=-1, keepdim=True)
    normed = (residual_out.float() * torch.rsqrt(variance + EPSILON) * norm_weight.float()).bfloat16()
    return normed, residual_out


def _build_model(vllm_config, device, rank):
    from vllm.compilation.decorators import support_torch_compile
    from vllm.model_executor.kernels.linear.mxfp8.b12x import B12xMxfp8LinearKernel
    from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import Mxfp8LinearLayerConfig
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        MXFP8_SCALE_DTYPE,
        MXFP8_VALUE_DTYPE,
    )

    generator = torch.Generator(device=device).manual_seed(1234)
    values = (torch.randn(HIDDEN, HIDDEN, generator=generator, device=device) * 0.05).to(MXFP8_VALUE_DTYPE)
    scales = torch.full((HIDDEN, HIDDEN // 32), 127, dtype=torch.uint8, device=device).view(MXFP8_SCALE_DTYPE)
    layer = nn.Module()
    layer.prefix = f"oracle.layers.{rank}.mxfp8"
    layer.weight = nn.Parameter(values, requires_grad=False)
    layer.weight_scale = nn.Parameter(scales, requires_grad=False)
    kernel = B12xMxfp8LinearKernel(Mxfp8LinearLayerConfig())
    kernel.process_weights_after_loading(layer)
    norm_weight = torch.linspace(0.5, 1.5, HIDDEN, dtype=torch.bfloat16, device=device)

    @support_torch_compile(dynamic_arg_dims={"x": 0, "residual": 0})
    class OracleModel(nn.Module):
        def __init__(self, *, vllm_config, prefix="", **kwargs):
            super().__init__()
            self.layer = layer
            self.norm_weight = norm_weight

        def forward(self, x: torch.Tensor, residual: torch.Tensor):
            hidden = kernel.apply_weights(self.layer, x)
            attention = torch.empty_like(hidden)
            torch.ops.silly.attention(hidden, hidden, hidden, attention)
            out = attention.clone()
            residual_out = residual.clone()
            torch.ops.vllm.b12x_fused_allreduce_add_rms_norm.default(
                out, residual_out, self.norm_weight, EPSILON
            )
            return out, residual_out

    from vllm.config import set_current_vllm_config

    with set_current_vllm_config(vllm_config):
        model = OracleModel(vllm_config=vllm_config, prefix="")
    return model, layer, kernel, values, norm_weight


def _vllm_config():
    from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode, KernelConfig, VllmConfig
    from vllm.config.scheduler import SchedulerConfig

    config = VllmConfig(
        # B12X_ORACLE_AUTOTUNE=0 prepares every family with its default
        # configuration and times nothing, the way a disabled autotune
        # setting does in a server start.
        kernel_config=KernelConfig(
            enable_b12x_autotune=os.environ.get("B12X_ORACLE_AUTOTUNE", "1") != "0",
        ),
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=MAX_TOKENS,
            max_num_seqs=2,
            max_model_len=MAX_TOKENS,
            is_encoder_decoder=False,
        ),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            backend="inductor",
            custom_ops=["all"],
            splitting_ops=["silly::attention"],
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
            cudagraph_capture_sizes=list(SIZES),
            cudagraph_num_of_warmups=1,
            compile_ranges_endpoints=[MAX_TOKENS],
        )
    )
    # VllmConfig plans capture sizes only with a model config present; the
    # oracle has none at construction, so the sizes are finalized here the way
    # the worker sees them.
    config.compilation_config.cudagraph_capture_sizes = list(SIZES)
    config.compilation_config.max_cudagraph_capture_size = max(SIZES)
    config.compilation_config.post_init_cudagraph_sizes()
    config.model_config = MagicMock()
    config.model_config.dtype = torch.bfloat16
    config.model_config.max_model_len = MAX_TOKENS
    config.model_config.model = "b12x-tp2-oracle"
    config.model_config.get_hidden_size.return_value = HIDDEN
    return config


def _run_rank(rank: int, port: int, beats, fault: str | None, scratch: str) -> None:
    from b12x._lib.compile_plan import observe_programs
    from b12x.preparation._measurement import no_compilation
    from vllm.config import set_current_vllm_config
    from vllm.distributed.device_communicators import b12x_pcie_all_reduce
    from vllm.distributed.device_communicators.b12x_pcie_all_reduce import get_b12x_pcie_allreduce
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_tp_group,
        graph_capture,
    )
    from vllm.forward_context import BatchDescriptor, set_forward_context
    from vllm.config import CUDAGraphMode
    from vllm.model_executor.warmup.b12x_prepare import begin_b12x_preparation
    from vllm.v1.worker.workspace import (
        current_workspace_manager,
        init_workspace_manager,
        lock_workspace,
        unlock_workspace,
    )

    from ..compile import silly_attention  # noqa: F401  registers silly::attention
    from ..utils import init_test_distributed_environment

    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    _heartbeat(beats, rank)
    vllm_config = _vllm_config()
    with set_current_vllm_config(vllm_config):
        init_test_distributed_environment(2, 1, rank, str(port), local_rank=rank)
    _run_rank_body(rank, beats, fault, scratch, device, vllm_config)


def _run_rank_body(rank, beats, fault, scratch, device, vllm_config) -> None:
    from b12x._lib.compile_plan import observe_programs
    from b12x.preparation._measurement import no_compilation
    from vllm.config import set_current_vllm_config
    from vllm.distributed.device_communicators import b12x_pcie_all_reduce
    from vllm.distributed.device_communicators.b12x_pcie_all_reduce import get_b12x_pcie_allreduce
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_tp_group,
        graph_capture,
    )
    from vllm.forward_context import BatchDescriptor, set_forward_context
    from vllm.config import CUDAGraphMode
    from vllm.model_executor.warmup.b12x_prepare import begin_b12x_preparation
    from vllm.v1.worker.workspace import (
        current_workspace_manager,
        init_workspace_manager,
        lock_workspace,
        unlock_workspace,
    )

    try:
        tp_group = get_tp_group()
        communicator = get_b12x_pcie_allreduce()
        assert communicator is not None
        init_workspace_manager(device, 1, 1)
        model, layer, kernel, values, norm_weight = _build_model(vllm_config, device, rank)
        communicator.register_describer(
            model,
            lambda workload: tuple(
                b12x_pcie_all_reduce.B12xPcieInvocation(
                    name=f"oracle.fused.{tokens}x{HIDDEN}",
                    operation="all_reduce_fused_add_rms_norm",
                    shape=(tokens, HIDDEN),
                    dtype=torch.bfloat16,
                    norm_weight=norm_weight,
                    epsilon=EPSILON,
                )
                for tokens in workload.token_counts
            ),
        )
        worker = SimpleNamespace(
            get_model=lambda: model,
            get_draft_model=lambda: None,
            model_runner=SimpleNamespace(mm_registry=None, _draft_workspace_lane=0, cudagraph_manager=None),
            vllm_config=vllm_config,
            scheduler_config=vllm_config.scheduler_config,
            model_config=vllm_config.model_config,
            rank=rank,
            device=device,
            _b12x_session=None,
        )
        _heartbeat(beats, rank)

        # Weights stage: sharded timing, winner exchange, collective priming.
        with set_current_vllm_config(vllm_config):
            coordinator = begin_b12x_preparation(worker, stage="weights")
            outcome = coordinator.status()
            while not outcome["done"]:
                outcome = coordinator.advance()
                _heartbeat(beats, rank)
            if outcome["error"] is not None:
                raise RuntimeError(f"preparation failed: {outcome['error']}")
        session = worker._b12x_session
        assert session is not None
        plan = layer.b12x_linear.plan
        assert plan is not None and plan.prepared is not None
        winners = {
            count: child.selection.config.to_dict() for count, child in plan.variants.items()
        }
        gathered = [None, None]
        dist.all_gather_object(gathered, winners, group=tp_group.cpu_group)
        assert gathered[0] == gathered[1], f"ranks selected different winners: {gathered}"
        if not vllm_config.kernel_config.enable_b12x_autotune:
            assert all(child.selection.source == "default" for child in plan.variants.values())
        # State stage: the worker drives it after the pools exist. This model
        # has no pool-dependent family, so every request it re-collects is
        # already prepared, and the stage must still complete on both ranks.
        with set_current_vllm_config(vllm_config):
            coordinator = begin_b12x_preparation(worker, stage="state")
            outcome = coordinator.status()
            while not outcome["done"]:
                outcome = coordinator.advance()
                _heartbeat(beats, rank)
            if outcome["error"] is not None:
                raise RuntimeError(f"state preparation failed: {outcome['error']}")
        _heartbeat(beats, rank)

        # Eager execution at every size compiles the model and checks numerics.
        # The largest size runs first, as the runner's profile pass does, so the
        # dynamic token dimension is never specialized on a size of one. The
        # undeclared counts run last, after the session is frozen, so a layer
        # that needs a plan for the exact count fails here rather than on the
        # first request.
        inputs = {}
        for tokens in (MAX_TOKENS, *reversed(SIZES), *UNDECLARED_SIZES):
            x = torch.randn(tokens, HIDDEN, device=device, dtype=torch.bfloat16) * (rank + 1)
            residual = torch.linspace(-0.5, 0.5, tokens * HIDDEN, device=device, dtype=torch.bfloat16).view(tokens, HIDDEN)
            inputs[tokens] = (x, residual)
            with set_forward_context(None, vllm_config=vllm_config):
                out, residual_out = model(x, residual)
            local = _check_linear(kernel, layer, x, values)
            expected, expected_residual = _reference(local, residual, norm_weight, tp_group.device_group)
            torch.testing.assert_close(out, expected, atol=5e-2, rtol=5e-2)
            torch.testing.assert_close(residual_out, expected_residual, atol=2e-2, rtol=2e-2)
            _heartbeat(beats, rank)

        manager = current_workspace_manager()
        manager.reserve_all()
        lock_workspace()
        if fault == "growth" and rank == 1:
            # An unlocked workspace and a linear whose every call draws more
            # than the last: the warmup pass may grow, the capture pass must be
            # refused by the capture guard rather than grown.
            unlock_workspace()
            holder = layer.b12x_linear
            original_run = holder.run
            draws = []

            def escalating_run(source, bias):
                draws.append(len(draws) + 1)
                manager.get_simultaneous(((draws[-1] << 20,), torch.uint8))
                return original_run(source, bias)

            holder.run = escalating_run
        slot_sizes_before = tuple(manager._workspace_size_bytes(slot) for slot in manager._current_workspaces)

        # Capture and replay under the guards that make lazy work an error.
        program_keys = set()
        replay_outputs = {}
        with no_compilation(), observe_programs() as observed, session.capture(), graph_capture(device=device):
            for tokens in reversed(SIZES):
                x, residual = inputs[tokens]
                started = time.perf_counter()
                for pass_index in range(2):
                    if fault == "skip_collective" and rank == 1 and tokens == SIZES[0] and pass_index == 1:
                        # Rank 1 stops participating in the last collective.
                        _heartbeat(beats, rank)
                        time.sleep(STALL_SECONDS * 3)
                    with set_forward_context(
                        None,
                        vllm_config=vllm_config,
                        cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
                        batch_descriptor=BatchDescriptor(num_tokens=tokens),
                    ):
                        if fault == "jit" and rank == 1 and pass_index == 0:
                            _fresh_triton_kernel(x)
                        model(x, residual)
                    torch.cuda.synchronize(device)
                    _heartbeat(beats, rank)
                elapsed = time.perf_counter() - started
                assert elapsed < CAPTURE_STEP_SECONDS, f"capture of {tokens} tokens took {elapsed:.1f} s"
                # Replay with changed inputs must track a fresh eager reference.
                # The first replay may allocate its output tensors; a second
                # replay must leave live memory unchanged.
                x.mul_(0.5).add_(0.25)
                out = residual_out = None
                for replay in range(2):
                    allocated = torch.cuda.memory_allocated(device)
                    with set_forward_context(
                        None,
                        vllm_config=vllm_config,
                        cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
                        batch_descriptor=BatchDescriptor(num_tokens=tokens),
                    ):
                        out, residual_out = model(x, residual)
                    torch.cuda.synchronize(device)
                    if replay:
                        assert torch.cuda.memory_allocated(device) == allocated, (
                            tokens, allocated, torch.cuda.memory_allocated(device))
                local = _check_linear(kernel, layer, x, values)
                expected, expected_residual = _reference(local, residual, norm_weight, tp_group.device_group)
                torch.testing.assert_close(out, expected, atol=5e-2, rtol=5e-2)
                torch.testing.assert_close(residual_out, expected_residual, atol=2e-2, rtol=2e-2)
                replay_outputs[tokens] = (out.clone(), residual_out.clone())
                _heartbeat(beats, rank)
        program_keys = sorted(f"{program.dialect}:{program.name}" for program in observed)
        slot_sizes_after = tuple(manager._workspace_size_bytes(slot) for slot in manager._current_workspaces)
        assert slot_sizes_after == slot_sizes_before, (slot_sizes_before, slot_sizes_after)

        invariants = {"programs": program_keys, "slots": slot_sizes_after}
        gathered = [None, None]
        dist.all_gather_object(gathered, invariants, group=tp_group.cpu_group)
        assert gathered[0]["programs"] == gathered[1]["programs"], gathered
        assert gathered[0]["slots"] == gathered[1]["slots"], gathered
        torch.cuda.synchronize(device)
    except BaseException:
        # Recorded before teardown, which can block while the peer waits in a
        # collective; the parent reads it when it terminates a stalled world.
        import traceback

        with open(os.path.join(scratch, f"rank{rank}.error"), "w") as handle:
            handle.write(traceback.format_exc())
        raise
    finally:
        session = getattr(worker, "_b12x_session", None) if "worker" in locals() else None
        if session is not None:
            session.close()
        destroy_model_parallel()
        destroy_distributed_environment()


def _fresh_triton_kernel(x: torch.Tensor) -> None:
    import triton
    import triton.language as tl

    @triton.jit
    def _oracle_fault_kernel(pointer, count, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < count
        tl.store(pointer + offsets, tl.load(pointer + offsets, mask=mask) + 1, mask=mask)

    flat = x.view(-1)
    _oracle_fault_kernel[(triton.cdiv(flat.numel(), 64),)](flat, flat.numel(), BLOCK=64)


def _spawn_with_watchdog(fault: str | None, scratch: str) -> tuple[str | None, list[str]]:
    """Run both ranks; return (error, dump paths). Never blocks past a stall."""
    context = mp.get_context("spawn")
    beats = context.Array("d", [time.monotonic(), time.monotonic()])
    procs = mp.spawn(
        _run_rank,
        args=(get_open_port(), beats, fault, scratch),
        nprocs=2,
        join=False,
    )
    dumps: list[str] = []
    error: str | None = None
    deadline_total = time.monotonic() + 15 * 60
    while True:
        try:
            if procs.join(timeout=1.0):
                break
        except Exception as failure:  # a rank raised: terminate the peer
            error = str(failure)
            break
        now = time.monotonic()
        stale = [rank for rank, process in enumerate(procs.processes)
                 if process.is_alive() and now - beats[rank] > STALL_SECONDS]
        if stale or now > deadline_total:
            recorded = sorted(
                path for path in os.listdir(scratch) if path.endswith(".error")
            )
            py_spy = shutil.which("py-spy") or os.path.expanduser("~/.local/bin/py-spy")
            for rank, process in enumerate(procs.processes):
                if process.is_alive() and os.path.exists(py_spy):
                    path = os.path.join(scratch, f"rank{rank}.pyspy")
                    with open(path, "w") as handle:
                        subprocess.run([py_spy, "dump", "--pid", str(process.pid), "--nonblocking", "--native"],
                                       stdout=handle, stderr=subprocess.STDOUT, timeout=60)
                    dumps.append(path)
            error = f"stall on ranks {stale}" if stale else "total timeout"
            if recorded:
                # A rank that raised and then blocked in teardown reports its
                # own failure rather than the stall it caused.
                error = "\n".join(
                    open(os.path.join(scratch, path)).read() for path in recorded
                ) + f"\n{error}"
            break
    for process in procs.processes:
        if process.is_alive():
            process.terminate()
    for process in procs.processes:
        process.join(timeout=30)
    return error, dumps


def _oracle_env(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_ENABLE_PCIE_ALLREDUCE", "1")
    monkeypatch.setenv("VLLM_PCIE_ALLREDUCE_BACKEND", "b12x")
    monkeypatch.setenv("VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE", "16KB")
    monkeypatch.setenv("VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE", "72KB")
    monkeypatch.setenv("VLLM_PCIE_DMA_MIN_BYTES", "64KB")
    monkeypatch.setenv("B12X_COMPILE_CACHE_DIR", str(tmp_path / "compile"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "vllm"))


@multi_gpu_test(num_gpus=2)
@pytest.mark.skipif(
    not current_platform.has_device_capability(120),
    reason="b12x startup oracle requires SM120",
)
@pytest.mark.parametrize(
    "autotune",
    [
        True,
        pytest.param(
            False,
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "the block-scaled a16 default (tile 128x64, split-K 4) accumulates "
                    "its slices in BF16 and exceeds the kernel's 0.4% relative-norm "
                    "tolerance at K=256; the timed selection does not pick it"
                ),
            ),
        ),
    ],
    ids=["autotune", "defaults"],
)
def test_b12x_tp2_startup_oracle(monkeypatch: pytest.MonkeyPatch, tmp_path, autotune) -> None:
    pytest.importorskip("b12x.comm.pcie")
    _oracle_env(monkeypatch, tmp_path)
    monkeypatch.setenv("B12X_ORACLE_AUTOTUNE", "1" if autotune else "0")
    error, dumps = _spawn_with_watchdog(None, str(tmp_path))
    assert error is None, f"{error}\ndumps: {dumps}"


@multi_gpu_test(num_gpus=2)
@pytest.mark.skipif(
    not current_platform.has_device_capability(120),
    reason="b12x startup oracle requires SM120",
)
@pytest.mark.parametrize(
    ("fault", "expected"),
    [
        ("growth", "during CUDA graph capture"),
        ("jit", "compil"),
        ("skip_collective", "stall on ranks"),
    ],
)
def test_b12x_tp2_startup_oracle_faults_fail_within_the_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path, fault, expected
) -> None:
    pytest.importorskip("b12x.comm.pcie")
    _oracle_env(monkeypatch, tmp_path)
    started = time.monotonic()
    error, dumps = _spawn_with_watchdog(fault, str(tmp_path))
    elapsed = time.monotonic() - started
    assert error is not None, f"fault {fault!r} did not surface"
    assert expected in error, error
    assert elapsed < STALL_SECONDS * 4 + 120, f"fault {fault!r} took {elapsed:.0f} s to surface"
    if fault == "skip_collective":
        assert dumps, "the watchdog must dump both ranks on a stall"
