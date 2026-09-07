# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.config import (
    CompilationConfig,
    CUDAGraphMode,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.distributed.device_communicators import pynccl_allocator
from vllm.v1.worker.gpu import cudagraph_utils as gpu_cudagraph_utils
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _reset_graph_pool_id():
    pynccl_allocator._graph_pool_id = None
    yield
    pynccl_allocator._graph_pool_id = None


def _create_vllm_config() -> MagicMock:
    compilation_config = CompilationConfig(
        cudagraph_mode="FULL",
        cudagraph_capture_sizes=[4],
    )
    compilation_config.max_cudagraph_capture_size = 4
    compilation_config.post_init_cudagraph_sizes()

    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.compilation_config = compilation_config
    vllm_config.scheduler_config = SchedulerConfig.default_factory(max_num_seqs=4)
    vllm_config.parallel_config = ParallelConfig()
    vllm_config.speculative_config = None
    vllm_config.num_speculative_tokens = 0
    return vllm_config


def test_full_capture_sets_graph_pool_id_before_cuda_graph(monkeypatch):
    """FULL capture must set graph_pool_id before entering torch.cuda.graph().

    NCCL symmetric memory checks this global during graph capture; without
    it, capture fails with:
    AssertionError: graph_pool_id is not set under graph capture
    """
    graph_pool = object()
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.current_platform,
        "get_global_graph_pool",
        lambda: graph_pool,
    )

    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=_create_vllm_config(),
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL,
        decode_query_len=1,
    )

    desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=4,
        num_reqs=4,
        uniform_token_count=1,
    )
    manager._capture_descs[CUDAGraphMode.FULL] = [desc]

    def create_forward_fn(desc, warmup):
        return lambda _mode: None

    @contextmanager
    def fake_graph_capture(*args, **kwargs):
        yield SimpleNamespace(stream=MagicMock())

    fake_offloader = MagicMock()

    def cuda_graph_enter(*args, **kwargs):
        assert pynccl_allocator._graph_pool_id is graph_pool

    mock_cuda_graph_ctx = MagicMock()
    mock_cuda_graph_ctx.__enter__ = cuda_graph_enter
    mock_cuda_graph_ctx.__exit__ = MagicMock(return_value=False)

    with (
        patch.object(gpu_cudagraph_utils, "graph_capture", fake_graph_capture),
        patch.object(gpu_cudagraph_utils, "get_offloader", lambda: fake_offloader),
        patch.object(gpu_cudagraph_utils.torch.cuda, "CUDAGraph"),
        patch.object(
            gpu_cudagraph_utils.torch.cuda,
            "graph",
            return_value=mock_cuda_graph_ctx,
        ) as mock_cuda_graph,
    ):
        manager.capture(create_forward_fn, channel_id="vllm:target:test")

    mock_cuda_graph.assert_called_once()


def _manager_with_prefill_and_decode_graphs():
    manager = gpu_cudagraph_utils.CudaGraphManager.__new__(
        gpu_cudagraph_utils.CudaGraphManager
    )
    manager._graphs_captured = True
    manager._lora_dispatch_map = {}
    manager._max_lora_case = 0
    full = BatchExecutionDescriptor(CUDAGraphMode.FULL, 4, 1, 4)
    piecewise = BatchExecutionDescriptor(CUDAGraphMode.PIECEWISE, 4, None)
    manager._exact_uniform_candidates = {(4, 0): [full]}
    manager._candidates = {(4, 0): [full, piecewise]}
    return manager, full, piecewise


@pytest.mark.parametrize("has_prefill", [False, True])
def test_four_token_prefill_does_not_replay_verifier_graph(has_prefill):
    """A four-token prompt tail is not a four-token speculative decode."""
    manager, full, piecewise = _manager_with_prefill_and_decode_graphs()
    desc = manager.dispatch(1, 4, 4, 0, has_prefill=has_prefill)
    assert desc is (piecewise if has_prefill else full)


def test_short_prefill_excludes_full_graph_without_uniform_descriptor():
    """Varlen FULL candidates also capture decode-only attention."""
    manager, _, piecewise = _manager_with_prefill_and_decode_graphs()
    manager._candidates[(4, 0)].insert(
        0, BatchExecutionDescriptor(CUDAGraphMode.FULL, 4, 4, None, 4)
    )
    assert manager.dispatch(2, 4, None, 0, has_prefill=True) is piecewise


def test_prefill_retains_generic_full_graph_captured_for_mixed_batches():
    manager, _, _ = _manager_with_prefill_and_decode_graphs()
    mixed = BatchExecutionDescriptor(CUDAGraphMode.FULL, 4, 4)
    manager._candidates[(4, 0)] = [mixed]
    assert manager.dispatch(1, 4, 4, 0, has_prefill=True) is mixed


def test_short_prefill_falls_back_to_eager_when_piecewise_is_absent():
    manager, full, _ = _manager_with_prefill_and_decode_graphs()
    manager._candidates[(4, 0)] = [full]
    desc = manager.dispatch(1, 4, 4, 0, has_prefill=True)
    assert desc.cg_mode == CUDAGraphMode.NONE
    assert (desc.num_tokens, desc.num_reqs) == (4, 1)


def test_prefill_guard_survives_data_parallel_graph_agreement(monkeypatch):
    """A peer prefill must prevent FULL promotion during the second dispatch."""
    from vllm.v1.worker.gpu import dp_utils

    manager, _, piecewise = _manager_with_prefill_and_decode_graphs()
    monkeypatch.setattr(
        dp_utils, "get_dp_group", lambda: SimpleNamespace(cpu_group=object())
    )

    def gather_peer_state(tensor, group):
        tensor[:, 1] = torch.tensor([4, CUDAGraphMode.PIECEWISE.value, 4, 4, 1])

    monkeypatch.setattr(dp_utils.dist, "all_reduce", gather_peer_state)
    desc, _ = dp_utils.dispatch_cg_and_sync_dp(
        manager, 1, 4, 4, 4, 2, 0, has_prefill=False
    )
    assert desc is piecewise


def test_prefill_guard_reaches_single_rank_dispatch():
    from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp

    manager, _, piecewise = _manager_with_prefill_and_decode_graphs()
    desc, across = dispatch_cg_and_sync_dp(manager, 1, 4, 4, 4, 1, 0, has_prefill=True)
    assert desc is piecewise
    assert across is None


@pytest.mark.parametrize(
    "computed,prefill,scheduled,dummy,expected",
    [
        ([4608], [4612], {"r0": 4}, False, True),
        ([115200], [115204], {"r0": 4}, False, True),
        ([4612], [4612], {"r0": 4}, False, False),
        ([4612, 4608], [4612, 4612], {"r1": 4, "r0": 4}, False, True),
        ([4608, 4612], [4612, 4612], {"r1": 4}, False, False),
        ([0], [4612], {"r0": 4}, True, False),
    ],
)
def test_v2_runner_passes_scheduled_prefill_state_to_dispatch(
    monkeypatch, computed, prefill, scheduled, dummy, expected
):
    """The real runner must use request state, not shape or inactive slots."""
    from vllm.v1.worker.gpu import model_runner

    class DispatchObserved(Exception):
        pass

    runner = model_runner.GPUModelRunner.__new__(model_runner.GPUModelRunner)
    runner._resolve_pending_draft = lambda: None
    runner.update_pp_decode_requests = lambda: None
    for method in ("finish_requests", "free_states", "add_requests", "update_requests"):
        setattr(runner, method, lambda _output: None)
    runner.block_tables = SimpleNamespace(apply_staged_writes=lambda: None)
    runner.req_states = SimpleNamespace(
        req_id_to_index={f"r{i}": i for i in range(len(computed))},
        num_computed_prefill_tokens=computed,
        prefill_len=SimpleNamespace(np=prefill),
    )
    runner.verification_capacity_manager = None
    runner.lora_config = None
    runner.is_encoder_decoder = False
    runner.cudagraph_manager = object()
    runner.dp_size, runner.dp_rank = 1, 0
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens=scheduled,
        total_num_scheduled_tokens=sum(scheduled.values()),
        scheduled_spec_decode_tokens={},
    )

    def observe_dispatch(*args, **kwargs):
        assert kwargs["has_prefill"] is expected
        raise DispatchObserved

    monkeypatch.setattr(model_runner, "dispatch_cg_and_sync_dp", observe_dispatch)
    with pytest.raises(DispatchObserved):
        runner.execute_model(scheduler_output, dummy_run=dummy)
