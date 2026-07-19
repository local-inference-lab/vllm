# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for DCP A2A communication backend (no GPU required).

Tests cover:
1. DCP A2A config validation (--dcp-comm-backend)
2. KVP group function exists
3. LSE-weighted combination correctness
"""

import importlib.util
import math
from contextlib import contextmanager
from typing import Any

import multiprocess as mp
import pytest
import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.config.parallel import ParallelConfig
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables

mp.set_start_method("spawn", force=True)


class _FakeCPGroup:
    def __init__(
        self,
        world_size: int,
        device_group: dist.ProcessGroup,
        cpu_group: dist.ProcessGroup | None = None,
    ):
        self.world_size = world_size
        self.device_group = device_group
        self.cpu_group = cpu_group


def _dtype_from_name(dtype_name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def _packed_a2a_reference(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    world_size: int,
    h_per_rank: int,
    is_lse_base_on_e: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.v1.attention.ops.dcp_alltoall import _lse_weighted_combine

    B, _H, D = cp_attn_out.shape
    outputs = (
        cp_attn_out.view(B, world_size, h_per_rank, D)
        .permute(1, 0, 2, 3)
        .contiguous()
        .float()
    )
    lses = cp_attn_lse.view(B, world_size, h_per_rank).permute(1, 0, 2).contiguous()
    return _lse_weighted_combine(
        outputs,
        lses,
        return_lse=True,
        is_lse_base_on_e=is_lse_base_on_e,
    )


def _assert_packed_a2a_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    if dtype == torch.float32:
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    else:
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=3e-2, atol=3e-2
        )


def _distributed_run(fn, world_size: int, extra_env: dict[str, str]) -> None:
    port = str(get_open_port())
    processes: list[mp.Process] = []
    for rank in range(world_size):
        env = {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": port,
            **extra_env,
        }
        process = mp.Process(target=fn, args=(env,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join(timeout=120)

    for process in processes:
        if process.is_alive():
            process.kill()
            process.join()
        assert process.exitcode == 0


class TestDCPCommBackendConfig:
    """Test --dcp-comm-backend config validation."""

    def test_default_is_ag_rs(self):
        """Default comm backend is ag_rs."""
        config = ParallelConfig()
        assert config.dcp_comm_backend == "ag_rs"

    def test_a2a_is_ignored_without_dcp(self):
        """The DCP backend is inert when decode context parallelism is off."""
        config = ParallelConfig(
            dcp_comm_backend="a2a",
            decode_context_parallel_size=1,
        )
        assert config.dcp_comm_backend == "ag_rs"

    def test_a2a_with_dcp_valid(self):
        """A2A backend is valid when DCP > 1."""
        config = ParallelConfig(
            dcp_comm_backend="a2a",
            tensor_parallel_size=4,
            decode_context_parallel_size=4,
        )
        assert config.dcp_comm_backend == "a2a"

    def test_invalid_backend_rejected(self):
        """Invalid backend values are rejected."""
        with pytest.raises(ValueError, match="must be one of|Input should be"):
            ParallelConfig(
                dcp_comm_backend="invalid",
            )

    def test_ag_rs_with_dcp_1_valid(self):
        """ag_rs backend is valid with DCP=1 (no DCP)."""
        config = ParallelConfig(
            dcp_comm_backend="ag_rs",
            decode_context_parallel_size=1,
        )
        assert config.dcp_comm_backend == "ag_rs"


class TestLSEWeightedCombine:
    """Test LSE-weighted combination logic (CPU only, no GPU).

    The _lse_weighted_combine function is the reference implementation
    that verifies the Triton kernel's correctness. It computes:

        result[b,h,d] = sum_n(w_n * output_n[b,h,d])

    where w_n = softmax(lse_n) = exp(lse_n) / sum_k(exp(lse_k))
    """

    def test_importable(self):
        """Verify _lse_weighted_combine is importable."""
        from vllm.v1.attention.ops.dcp_alltoall import _lse_weighted_combine

        assert callable(_lse_weighted_combine)

    def test_single_rank(self):
        """Single rank: output unchanged."""
        from vllm.v1.attention.ops.dcp_alltoall import _lse_weighted_combine

        # N=1, B=2, H=4, D=8
        outputs = torch.randn(1, 2, 4, 8)
        lses = torch.randn(1, 2, 4)

        result = _lse_weighted_combine(outputs, lses)

        assert result.shape == (2, 4, 8)
        torch.testing.assert_close(result, outputs.squeeze(0), rtol=1e-5, atol=1e-5)

    def test_equal_lse(self):
        """Equal LSE values: outputs averaged equally."""
        from vllm.v1.attention.ops.dcp_alltoall import _lse_weighted_combine

        _N, B, H, D = 2, 1, 1, 4
        outputs = torch.tensor(
            [
                [[[1.0, 2.0, 3.0, 4.0]]],  # Rank 0
                [[[5.0, 6.0, 7.0, 8.0]]],  # Rank 1
            ]
        )
        lses = torch.tensor(
            [
                [[0.0]],  # Rank 0
                [[0.0]],  # Rank 1
            ]
        )

        result = _lse_weighted_combine(outputs, lses)

        expected = (outputs[0] + outputs[1]) / 2
        assert result.shape == (B, H, D)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_dominant_rank(self):
        """Different LSE values: larger LSE gets more weight."""
        from vllm.v1.attention.ops.dcp_alltoall import _lse_weighted_combine

        B, H, D = 1, 1, 2
        outputs = torch.tensor(
            [
                [[[0.0, 0.0]]],  # Rank 0
                [[[1.0, 1.0]]],  # Rank 1
            ]
        )
        lses = torch.tensor(
            [
                [[-100.0]],  # Rank 0: negligible contribution
                [[0.0]],  # Rank 1: dominant
            ]
        )

        result = _lse_weighted_combine(outputs, lses)

        assert result.shape == (B, H, D)
        torch.testing.assert_close(result, outputs[1], atol=1e-5, rtol=1e-5)

    def test_mathematically_correct(self):
        """Verify mathematical correctness of LSE combination."""
        from vllm.v1.attention.ops.dcp_alltoall import _lse_weighted_combine

        outputs = torch.tensor(
            [
                [[[2.0, 4.0]]],
                [[[6.0, 8.0]]],
            ]
        )
        lses = torch.tensor(
            [
                [[1.0]],  # exp(1) ≈ 2.718
                [[2.0]],  # exp(2) ≈ 7.389
            ]
        )

        result = _lse_weighted_combine(outputs, lses)

        w0 = math.exp(1) / (math.exp(1) + math.exp(2))
        w1 = math.exp(2) / (math.exp(1) + math.exp(2))
        expected = torch.tensor([[[w0 * 2.0 + w1 * 6.0, w0 * 4.0 + w1 * 8.0]]])

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    def test_return_lse(self):
        """return_lse=True returns global LSE (logsumexp of inputs)."""
        from vllm.v1.attention.ops.dcp_alltoall import _lse_weighted_combine

        B, H, D = 1, 1, 2
        outputs = torch.tensor(
            [
                [[[1.0, 2.0]]],
                [[[3.0, 4.0]]],
            ]
        )
        lses = torch.tensor(
            [
                [[1.0]],
                [[2.0]],
            ]
        )

        result, global_lse = _lse_weighted_combine(outputs, lses, return_lse=True)

        expected_global_lse = math.log(math.exp(1) + math.exp(2))

        assert result.shape == (B, H, D)
        assert global_lse.shape == (B, H)
        assert abs(global_lse.item() - expected_global_lse) < 1e-5

    def test_base2_return_lse(self):
        """Base-2 LSE mode returns log2-sum-exp2 global LSE."""
        from vllm.v1.attention.ops.dcp_alltoall import _lse_weighted_combine

        outputs = torch.tensor(
            [
                [[[1.0, 2.0]]],
                [[[3.0, 4.0]]],
            ]
        )
        lses = torch.tensor(
            [
                [[1.0]],
                [[2.0]],
            ]
        )

        result, global_lse = _lse_weighted_combine(
            outputs,
            lses,
            return_lse=True,
            is_lse_base_on_e=False,
        )

        expected_global_lse = math.log2(2**1 + 2**2)
        w0 = 2**1 / (2**1 + 2**2)
        w1 = 2**2 / (2**1 + 2**2)
        expected = torch.tensor([[[w0 * 1.0 + w1 * 3.0, w0 * 2.0 + w1 * 4.0]]])

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(
            global_lse,
            torch.tensor([[expected_global_lse]]),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_lse_pack_dim(self):
        """Packed A2A stores one fp32 LSE in output-dtype lanes."""
        from vllm.v1.attention.ops.dcp_alltoall import _dcp_a2a_lse_pack_dim

        assert _dcp_a2a_lse_pack_dim(torch.bfloat16) == 2
        assert _dcp_a2a_lse_pack_dim(torch.float16) == 2
        assert _dcp_a2a_lse_pack_dim(torch.float32) == 1


def test_b12x_dispatch_bypasses_packed_nccl(monkeypatch: pytest.MonkeyPatch):
    from vllm.v1.attention.ops import dcp_alltoall

    monkeypatch.setenv("VLLM_USE_B12X_DCP_A2A", "1")
    partial_output = torch.zeros(1, 16, 64, dtype=torch.bfloat16)
    partial_lse = torch.zeros(1, 16, dtype=torch.float32)
    expected = torch.ones(1, 8, 64, dtype=torch.bfloat16)
    captured: dict[str, Any] = {}

    def fake_b12x(
        cp_attn_out,
        cp_attn_lse,
        cp_group,
        *,
        return_lse,
        is_lse_base_on_e,
        max_batch_size,
        query_head_dim,
    ):
        captured.update(
            output=cp_attn_out,
            lse=cp_attn_lse,
            group=cp_group,
            return_lse=return_lse,
            is_lse_base_on_e=is_lse_base_on_e,
            max_batch_size=max_batch_size,
            query_head_dim=query_head_dim,
        )
        return expected

    monkeypatch.setattr(dcp_alltoall, "_try_b12x_dcp_lse_reduce", fake_b12x)
    group = _FakeCPGroup(2, None)  # type: ignore[arg-type]
    actual = dcp_alltoall.dcp_a2a_lse_reduce(
        partial_output,
        partial_lse,
        group,  # type: ignore[arg-type]
        use_b12x=True,
        b12x_max_batch_size=8192,
    )

    assert actual is expected
    assert captured == {
        "output": partial_output,
        "lse": partial_lse,
        "group": group,
        "return_lse": False,
        "is_lse_base_on_e": True,
        "max_batch_size": 8192,
        "query_head_dim": None,
    }


def test_packed_a2a_capture_buffers_stay_live_per_shape(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.v1.attention.ops import dcp_alltoall

    created: list[object] = []

    def fake_empty(*args, **kwargs):
        value = object()
        created.append(value)
        return value

    dcp_alltoall._DCP_A2A_GRAPH_BUFFERS.clear()
    monkeypatch.setattr(torch, "empty", fake_empty)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    device = torch.device("cuda:0")

    first = dcp_alltoall._dcp_a2a_send_recv_buffers(
        (3, 4, 11, 514), device, torch.bfloat16
    )
    same_shape = dcp_alltoall._dcp_a2a_send_recv_buffers(
        (3, 4, 11, 514), device, torch.bfloat16
    )
    larger = dcp_alltoall._dcp_a2a_send_recv_buffers(
        (3, 8, 11, 514), device, torch.bfloat16
    )

    assert same_shape is first
    assert larger is not first
    assert len(created) == 4
    assert len(dcp_alltoall._DCP_A2A_GRAPH_BUFFERS) == 2
    dcp_alltoall._DCP_A2A_GRAPH_BUFFERS.clear()


def test_packed_a2a_prewarm_buffers_are_retained_before_cuda_capture(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.v1.attention.ops import dcp_alltoall

    created: list[object] = []

    def fake_empty(*args, **kwargs):
        value = object()
        created.append(value)
        return value

    dcp_alltoall._DCP_A2A_GRAPH_BUFFERS.clear()
    monkeypatch.setattr(torch, "empty", fake_empty)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(dcp_alltoall, "is_vllm_cudagraph_capture_active", lambda: True)
    device = torch.device("cuda:0")

    prewarm = dcp_alltoall._dcp_a2a_send_recv_buffers(
        (3, 16, 11, 514), device, torch.bfloat16
    )
    capture = dcp_alltoall._dcp_a2a_send_recv_buffers(
        (3, 16, 11, 514), device, torch.bfloat16
    )

    assert capture is prewarm
    assert len(created) == 2
    assert len(dcp_alltoall._DCP_A2A_GRAPH_BUFFERS) == 1
    dcp_alltoall._DCP_A2A_GRAPH_BUFFERS.clear()


def test_packed_a2a_eager_buffers_are_not_retained(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.v1.attention.ops import dcp_alltoall

    created: list[object] = []

    def fake_empty(*args, **kwargs):
        value = object()
        created.append(value)
        return value

    dcp_alltoall._DCP_A2A_GRAPH_BUFFERS.clear()
    monkeypatch.setattr(torch, "empty", fake_empty)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(dcp_alltoall, "is_vllm_cudagraph_capture_active", lambda: False)
    device = torch.device("cuda:0")

    first = dcp_alltoall._dcp_a2a_send_recv_buffers(
        (3, 4, 11, 514), device, torch.bfloat16
    )
    second = dcp_alltoall._dcp_a2a_send_recv_buffers(
        (3, 4, 11, 514), device, torch.bfloat16
    )

    assert first is not second
    assert len(created) == 4
    assert not dcp_alltoall._DCP_A2A_GRAPH_BUFFERS


def test_b12x_query_gather_dispatch_bypasses_group(monkeypatch: pytest.MonkeyPatch):
    from vllm.v1.attention.ops import dcp_alltoall

    monkeypatch.setenv("VLLM_USE_B12X_DCP_A2A", "1")
    local_query = torch.zeros(2, 8, 64, dtype=torch.bfloat16)
    expected = torch.ones(2, 16, 64, dtype=torch.bfloat16)
    captured: dict[str, Any] = {}

    def fake_b12x(local_input, cp_group, *, max_batch_size, output_head_dim):
        captured.update(
            local_input=local_input,
            group=cp_group,
            max_batch_size=max_batch_size,
            output_head_dim=output_head_dim,
        )
        return expected

    monkeypatch.setattr(
        dcp_alltoall,
        "_try_b12x_dcp_all_gather_heads",
        fake_b12x,
    )
    group = _FakeCPGroup(2, None)  # type: ignore[arg-type]
    actual = dcp_alltoall.dcp_b12x_all_gather_heads(
        local_query,
        group,  # type: ignore[arg-type]
        max_batch_size=8192,
    )

    assert actual is expected
    assert captured == {
        "local_input": local_query,
        "group": group,
        "max_batch_size": 8192,
        "output_head_dim": None,
    }


def test_b12x_pool_init_consensus_uses_exchange_group(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.v1.attention.ops import dcp_alltoall

    device_group = object()
    cpu_group = object()
    group = _FakeCPGroup(4, device_group, cpu_group)  # type: ignore[arg-type]
    device = torch.device("cuda:0")
    captured: dict[str, Any] = {}
    original_tensor = torch.tensor

    def fake_tensor(data, *, dtype, device):
        captured["tensor_device"] = device
        return original_tensor(data, dtype=dtype)

    def fake_all_reduce(tensor, *, op, group):
        captured.update(tensor=tensor, op=op, group=group)

    monkeypatch.setattr(dcp_alltoall.torch, "tensor", fake_tensor)
    monkeypatch.setattr(dcp_alltoall.dist, "all_reduce", fake_all_reduce)

    assert not dcp_alltoall._b12x_dcp_init_failed(group, device, None)
    assert captured["tensor_device"] == device
    assert captured["group"] is device_group
    assert captured["op"] == dist.ReduceOp.MAX


def test_b12x_pool_uses_independent_stream_channels(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.v1.attention.ops import dcp_alltoall

    captured: dict[str, Any] = {}

    class _FakePool:
        @classmethod
        def from_exchange_group(cls, **kwargs):
            captured.update(kwargs)
            return cls()

        def for_stream(self):
            captured["warmed"] = True

    group = _FakeCPGroup(2, object())  # type: ignore[arg-type]
    monkeypatch.setattr(dcp_alltoall, "_B12X_DCP_A2A_POOLS", {})
    monkeypatch.setattr(dcp_alltoall, "_B12X_DCP_A2A_DISABLED", set())
    monkeypatch.setattr(dcp_alltoall, "_load_b12x_dcp_a2a_pool", lambda: _FakePool)
    monkeypatch.setattr(dcp_alltoall, "_b12x_dcp_init_failed", lambda *args: False)
    monkeypatch.setattr(
        dcp_alltoall.torch.cuda,
        "is_current_stream_capturing",
        lambda: False,
    )

    pool = dcp_alltoall._get_b12x_dcp_a2a_pool(
        group,  # type: ignore[arg-type]
        device=torch.device("cuda:0"),
        total_heads=64,
        head_dim=512,
        query_head_dim=576,
        max_batch_size=64,
    )

    assert pool is not None
    assert captured["single_channel"] is False
    assert captured["warmed"] is True


def test_b12x_dcp_capture_selects_only_current_group_pools(monkeypatch):
    from vllm.v1.attention.ops import dcp_alltoall

    events = []

    class _FakePool:
        def __init__(self, name):
            self.name = name

        @contextmanager
        def capture(self, *, stream):
            events.append(("enter", self.name, stream))
            try:
                yield
            finally:
                events.append(("exit", self.name, stream))

    device_group = object()
    group = _FakeCPGroup(2, device_group)  # type: ignore[arg-type]
    stream = object()
    pools = {
        (id(device_group), 0, 64, 512, 576, 64): _FakePool("output"),
        (id(device_group), 0, 64, 576, 576, 64): _FakePool("query"),
        (id(object()), 0, 64, 512, 576, 64): _FakePool("foreign"),
    }
    monkeypatch.setattr(dcp_alltoall, "_B12X_DCP_A2A_POOLS", pools)

    with dcp_alltoall.capture_b12x_dcp_a2a(group, stream):  # type: ignore[arg-type]
        events.append(("body", None, stream))

    assert events == [
        ("enter", "output", stream),
        ("enter", "query", stream),
        ("body", None, stream),
        ("exit", "query", stream),
        ("exit", "output", stream),
    ]


def test_global_graph_capture_enters_b12x_dcp_pool(monkeypatch):
    from vllm.distributed import parallel_state
    from vllm.v1.attention.ops import dcp_alltoall

    events = []

    class _FakeGroup:
        world_size = 2

        @contextmanager
        def graph_capture(self, context):
            yield context

    tp_group = _FakeGroup()
    pp_group = _FakeGroup()
    dcp_group = _FakeGroup()
    stream = object()
    context = parallel_state.GraphCaptureContext(stream)  # type: ignore[arg-type]

    @contextmanager
    def fake_b12x_capture(group, selected_stream):
        events.append((group, selected_stream))
        yield

    monkeypatch.setattr(parallel_state, "_DCP", dcp_group)
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(parallel_state, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(parallel_state, "get_dcp_group", lambda: dcp_group)
    monkeypatch.setattr(dcp_alltoall, "capture_b12x_dcp_a2a", fake_b12x_capture)

    with parallel_state.graph_capture(torch.device("cpu"), context) as actual:
        assert actual is context

    assert events == [(dcp_group, stream)]


@pytest.mark.skipif(torch.accelerator.device_count() < 1, reason="CUDA is required.")
def test_b12x_lse_reduce_honors_token_cap(monkeypatch: pytest.MonkeyPatch):
    from vllm.v1.attention.ops import dcp_alltoall

    monkeypatch.setenv("VLLM_USE_B12X_DCP_A2A", "1")
    monkeypatch.setenv("VLLM_DCP_A2A_MAX_TOKENS", "4")
    created: dict[str, Any] = {}
    sentinel = torch.zeros(1)

    class _FakePool:
        def lse_reduce_scatter(
            self, partial, lse, out=None, *, is_lse_base_on_e
        ):
            return sentinel

    def fake_get_pool(
        cp_group, *, device, total_heads, head_dim, query_head_dim, max_batch_size
    ):
        created["max_batch_size"] = max_batch_size
        return _FakePool()

    monkeypatch.setattr(dcp_alltoall, "_get_b12x_dcp_a2a_pool", fake_get_pool)
    group = _FakeCPGroup(4, None)  # type: ignore[arg-type]

    out = torch.zeros(4, 16, 64, dtype=torch.bfloat16, device="cuda")
    lse = torch.zeros(4, 16, dtype=torch.float32, device="cuda")
    result = dcp_alltoall._try_b12x_dcp_lse_reduce(
        out,
        lse,
        group,  # type: ignore[arg-type]
        return_lse=False,
        is_lse_base_on_e=True,
        max_batch_size=8192,
        query_head_dim=64,
    )
    # Batch within the cap uses B12X, with the staging pool capped too.
    assert result is sentinel
    assert created["max_batch_size"] == 4

    out_large = torch.zeros(8, 16, 64, dtype=torch.bfloat16, device="cuda")
    lse_large = torch.zeros(8, 16, dtype=torch.float32, device="cuda")
    result = dcp_alltoall._try_b12x_dcp_lse_reduce(
        out_large,
        lse_large,
        group,  # type: ignore[arg-type]
        return_lse=False,
        is_lse_base_on_e=True,
        max_batch_size=8192,
        query_head_dim=64,
    )
    # Batch above the cap declines B12X so the caller picks an NCCL path.
    assert result is None


@pytest.mark.skipif(torch.accelerator.device_count() < 1, reason="CUDA is required.")
def test_b12x_query_gather_honors_token_cap(monkeypatch: pytest.MonkeyPatch):
    from vllm.v1.attention.ops import dcp_alltoall

    monkeypatch.setenv("VLLM_USE_B12X_DCP_A2A", "1")
    monkeypatch.setenv("VLLM_DCP_A2A_MAX_TOKENS", "4")
    created: dict[str, Any] = {}
    sentinel = torch.zeros(1)

    class _FakePool:
        def all_gather_heads(self, local_input):
            return sentinel

    def fake_get_pool(
        cp_group, *, device, total_heads, head_dim, query_head_dim, max_batch_size
    ):
        created["max_batch_size"] = max_batch_size
        return _FakePool()

    monkeypatch.setattr(dcp_alltoall, "_get_b12x_dcp_a2a_pool", fake_get_pool)
    group = _FakeCPGroup(4, None)  # type: ignore[arg-type]

    small = torch.zeros(4, 8, 64, dtype=torch.bfloat16, device="cuda")
    result = dcp_alltoall._try_b12x_dcp_all_gather_heads(
        small,
        group,  # type: ignore[arg-type]
        max_batch_size=8192,
        output_head_dim=64,
    )
    assert result is sentinel
    assert created["max_batch_size"] == 4

    large = torch.zeros(8, 8, 64, dtype=torch.bfloat16, device="cuda")
    result = dcp_alltoall._try_b12x_dcp_all_gather_heads(
        large,
        group,  # type: ignore[arg-type]
        max_batch_size=8192,
        output_head_dim=64,
    )
    assert result is None


@pytest.mark.skipif(torch.accelerator.device_count() < 1, reason="CUDA is required.")
def test_b12x_lse_reduce_preserves_supported_layouts(monkeypatch: pytest.MonkeyPatch):
    """Preserve head-major input while materializing legacy head slices."""
    from vllm.v1.attention.ops import dcp_alltoall

    monkeypatch.setenv("VLLM_USE_B12X_DCP_A2A", "1")
    received: dict[str, Any] = {}
    sentinel = torch.zeros(1)

    class _FakePool:
        def lse_reduce_scatter(
            self, partial, lse, out=None, *, is_lse_base_on_e
        ):
            received.update(partial=partial, lse=lse, out=out)
            return sentinel

    monkeypatch.setattr(
        dcp_alltoall,
        "_get_b12x_dcp_a2a_pool",
        lambda *a, **k: _FakePool(),
    )
    group = _FakeCPGroup(4, None)  # type: ignore[arg-type]

    # Simulate the GLM TP6 head66 pattern: kernel-padded buffers sliced back
    # in the head dim produce non-contiguous views.
    out_padded = torch.zeros(4, 24, 64, dtype=torch.bfloat16, device="cuda")
    lse_padded = torch.zeros(4, 24, dtype=torch.float32, device="cuda")
    out_view = out_padded[:, :16]
    lse_view = lse_padded[:, :16]
    assert not out_view.is_contiguous() and not lse_view.is_contiguous()

    result = dcp_alltoall._try_b12x_dcp_lse_reduce(
        out_view,
        lse_view,
        group,  # type: ignore[arg-type]
        return_lse=False,
        is_lse_base_on_e=True,
        max_batch_size=8192,
        query_head_dim=64,
    )
    assert result is sentinel
    assert received["partial"].is_contiguous()
    assert received["lse"].is_contiguous()
    assert received["out"].movedim(0, 1).is_contiguous()

    head_major_storage = torch.zeros(
        16, 8, 64, dtype=torch.bfloat16, device="cuda"
    )
    head_major = head_major_storage.transpose(0, 1)[:4]
    result = dcp_alltoall._try_b12x_dcp_lse_reduce(
        head_major,
        torch.zeros(4, 16, dtype=torch.float32, device="cuda"),
        group,  # type: ignore[arg-type]
        return_lse=False,
        is_lse_base_on_e=True,
        max_batch_size=8192,
        query_head_dim=64,
    )
    assert result is sentinel
    assert received["partial"] is head_major
    assert received["out"].stride() == (64, 4 * 64, 1)


def test_b12x_query_gather_requires_env(monkeypatch: pytest.MonkeyPatch):
    from vllm.v1.attention.ops import dcp_alltoall

    monkeypatch.delenv("VLLM_USE_B12X_DCP_A2A", raising=False)
    local_query = torch.zeros(2, 8, 64, dtype=torch.bfloat16)
    expected = torch.ones(2, 16, 64, dtype=torch.bfloat16)
    group = _FakeCPGroup(2, None)  # type: ignore[arg-type]
    group.all_gather = lambda value, dim: expected  # type: ignore[attr-defined]
    monkeypatch.setattr(
        dcp_alltoall,
        "_try_b12x_dcp_all_gather_heads",
        lambda *args, **kwargs: pytest.fail("B12X path must remain disabled"),
    )

    actual = dcp_alltoall.dcp_b12x_all_gather_heads(
        local_query,
        group,  # type: ignore[arg-type]
        max_batch_size=8192,
    )

    assert actual is expected


def test_warmup_skips_unsupported_world_size(monkeypatch: pytest.MonkeyPatch):
    from vllm.v1.attention.ops import dcp_alltoall

    monkeypatch.setenv("VLLM_USE_B12X_DCP_A2A", "1")
    monkeypatch.setattr(
        dcp_alltoall,
        "_try_b12x_dcp_all_gather_heads",
        lambda *args, **kwargs: pytest.fail(
            "warmup must not touch the B12X channel for world size 6"
        ),
    )
    group = _FakeCPGroup(6, None)  # type: ignore[arg-type]

    # Must log-and-return instead of raising: the runtime dispatchers fall
    # back to NCCL for DCP world sizes without a B12X channel (e.g. TP6).
    dcp_alltoall.warmup_b12x_dcp_a2a(
        group,  # type: ignore[arg-type]
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_batch_size=8192,
        total_heads=66,
        head_dim=512,
        query_head_dim=576,
    )


class TestPackedA2AKernels:
    @pytest.mark.skipif(
        torch.accelerator.device_count() < 1, reason="CUDA is required."
    )
    @pytest.mark.parametrize("dtype_name", ["float16", "bfloat16", "float32"])
    @pytest.mark.parametrize("return_lse", [False, True])
    @pytest.mark.parametrize("is_lse_base_on_e", [False, True])
    def test_pack_unpack_combine_matches_reference(
        self,
        dtype_name: str,
        return_lse: bool,
        is_lse_base_on_e: bool,
    ):
        from vllm.v1.attention.ops.dcp_alltoall import (
            _dcp_a2a_lse_pack_dim,
            _dcp_a2a_pack_send,
            _dcp_a2a_unpack_combine,
        )

        torch.manual_seed(0)
        dtype = _dtype_from_name(dtype_name)
        device = torch.device("cuda")
        world_size, B, h_per_rank, D = 4, 7, 2, 32
        H = world_size * h_per_rank
        cp_attn_out = torch.randn(B, H, D, device=device, dtype=dtype)
        cp_attn_lse = torch.randn(B, H, device=device, dtype=torch.float32)
        lse_pack_dim = _dcp_a2a_lse_pack_dim(dtype)
        send_buffer = torch.empty(
            (world_size, B, h_per_rank, D + lse_pack_dim),
            device=device,
            dtype=dtype,
        )

        _dcp_a2a_pack_send(
            cp_attn_out,
            cp_attn_lse,
            send_buffer,
            world_size,
            h_per_rank,
            D,
            lse_pack_dim,
        )
        actual = _dcp_a2a_unpack_combine(
            send_buffer, D, lse_pack_dim, return_lse, is_lse_base_on_e
        )
        expected_out, expected_lse = _packed_a2a_reference(
            cp_attn_out, cp_attn_lse, world_size, h_per_rank, is_lse_base_on_e
        )

        if return_lse:
            actual_out, actual_lse = actual
            _assert_packed_a2a_close(actual_out, expected_out, dtype)
            torch.testing.assert_close(actual_lse, expected_lse, rtol=1e-4, atol=1e-4)
        else:
            actual_out = actual
            _assert_packed_a2a_close(actual, expected_out, dtype)
        assert actual_out.movedim(0, 1).is_contiguous()
        assert not actual_out.is_contiguous()


def test_cuda_reduce_scatter_can_preserve_head_major_output(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.distributed.device_communicators import cuda_communicator

    monkeypatch.setattr(
        cuda_communicator,
        "should_nccl_symm_mem_ag_rs",
        lambda: False,
    )

    class FakePyNccl:
        disabled = False

        def reduce_scatter(self, output, input_):
            output.copy_(input_[: output.shape[0]])

    class FakeCommunicator:
        world_size = 2
        pynccl_comm = FakePyNccl()

    input_storage = torch.arange(8 * 3 * 16, dtype=torch.bfloat16).view(8, 3, 16)
    input_ = input_storage.movedim(0, 1)
    actual = cuda_communicator.CudaCommunicator.reduce_scatter_head_major(
        FakeCommunicator(), input_, dim=1
    )

    expected = input_[:, :4]
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (3, 4, 16)
    assert actual.stride() == (16, 3 * 16, 1)
    assert actual.movedim(0, 1).is_contiguous()


def _distributed_packed_a2a_worker(env: dict[str, str]) -> None:
    update_environment_variables(env)
    local_rank = int(env["LOCAL_RANK"])
    torch.accelerator.set_device_index(local_rank)
    if envs.VLLM_DISTRIBUTED_USE_SPLIT_GROUP:
        dist.init_process_group(
            backend="cpu:gloo,cuda:nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
        )
    else:
        dist.init_process_group(backend="nccl")
    use_workspace = env.get("USE_WORKSPACE") == "1"
    if use_workspace:
        from vllm.v1.worker.workspace import init_workspace_manager

        init_workspace_manager(torch.device(f"cuda:{local_rank}"))
    try:
        from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce

        dtype = _dtype_from_name(env["TEST_DTYPE"])
        return_lse = env["RETURN_LSE"] == "1"
        is_lse_base_on_e = env["LSE_BASE_E"] == "1"
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        B, h_per_rank, D = 5, 2, 32
        H = world_size * h_per_rank

        generator = torch.Generator(device=f"cuda:{local_rank}")
        generator.manual_seed(1234 + rank)
        cp_attn_out = torch.randn(
            B,
            H,
            D,
            device=f"cuda:{local_rank}",
            dtype=dtype,
            generator=generator,
        )
        cp_attn_lse = torch.randn(
            B,
            H,
            device=f"cuda:{local_rank}",
            dtype=torch.float32,
            generator=generator,
        )
        actual = dcp_a2a_lse_reduce(
            cp_attn_out,
            cp_attn_lse,
            _FakeCPGroup(world_size, dist.group.WORLD),
            return_lse=return_lse,
            is_lse_base_on_e=is_lse_base_on_e,
        )

        gathered_out = [torch.empty_like(cp_attn_out) for _ in range(world_size)]
        gathered_lse = [torch.empty_like(cp_attn_lse) for _ in range(world_size)]
        dist.all_gather(gathered_out, cp_attn_out)
        dist.all_gather(gathered_lse, cp_attn_lse)
        outputs = torch.stack(
            [
                t[:, rank * h_per_rank : (rank + 1) * h_per_rank, :]
                for t in gathered_out
            ],
            dim=0,
        ).float()
        lses = torch.stack(
            [t[:, rank * h_per_rank : (rank + 1) * h_per_rank] for t in gathered_lse],
            dim=0,
        )
        from vllm.v1.attention.ops.dcp_alltoall import _lse_weighted_combine

        expected_out, expected_lse = _lse_weighted_combine(
            outputs,
            lses,
            return_lse=True,
            is_lse_base_on_e=is_lse_base_on_e,
        )

        if return_lse:
            actual_out, actual_lse = actual
            _assert_packed_a2a_close(actual_out, expected_out, dtype)
            torch.testing.assert_close(actual_lse, expected_lse, rtol=1e-4, atol=1e-4)
        else:
            _assert_packed_a2a_close(actual, expected_out, dtype)
    finally:
        if use_workspace:
            from vllm.v1.worker.workspace import reset_workspace_manager

            reset_workspace_manager()
        dist.destroy_process_group()


def _distributed_b12x_a2a_worker(env: dict[str, str]) -> None:
    update_environment_variables(env)
    local_rank = int(env["LOCAL_RANK"])
    torch.accelerator.set_device_index(local_rank)
    dist.init_process_group(backend="nccl")
    try:
        from vllm.v1.attention.ops import dcp_alltoall

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        batch, h_per_rank, head_dim, query_head_dim = 3, 8, 512, 576
        total_heads = world_size * h_per_rank
        group = _FakeCPGroup(world_size, dist.group.WORLD)

        def make_inputs(step: int):
            generator = torch.Generator(device=f"cuda:{local_rank}")
            generator.manual_seed(1000 * step + rank)
            output = torch.randn(
                batch,
                total_heads,
                head_dim,
                device=f"cuda:{local_rank}",
                dtype=torch.bfloat16,
                generator=generator,
            )
            lse = torch.randn(
                batch,
                total_heads,
                device=f"cuda:{local_rank}",
                dtype=torch.float32,
                generator=generator,
            )
            return output, lse

        def make_query(step: int) -> torch.Tensor:
            generator = torch.Generator(device=f"cuda:{local_rank}")
            generator.manual_seed(10000 * step + rank)
            return torch.randn(
                batch,
                h_per_rank,
                query_head_dim,
                device=f"cuda:{local_rank}",
                dtype=torch.bfloat16,
                generator=generator,
            )

        def expected_query(query: torch.Tensor) -> torch.Tensor:
            gathered = [torch.empty_like(query) for _ in range(world_size)]
            dist.all_gather(gathered, query)
            return torch.cat(gathered, dim=1)

        def expected(output: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
            gathered_output = [torch.empty_like(output) for _ in range(world_size)]
            gathered_lse = [torch.empty_like(lse) for _ in range(world_size)]
            dist.all_gather(gathered_output, output)
            dist.all_gather(gathered_lse, lse)
            outputs = torch.stack(
                [
                    value[:, rank * h_per_rank : (rank + 1) * h_per_rank]
                    for value in gathered_output
                ]
            ).float()
            lses = torch.stack(
                [
                    value[:, rank * h_per_rank : (rank + 1) * h_per_rank]
                    for value in gathered_lse
                ]
            )
            return dcp_alltoall._lse_weighted_combine(outputs, lses)

        dcp_alltoall.warmup_b12x_dcp_a2a(
            group,  # type: ignore[arg-type]
            device=torch.device(f"cuda:{local_rank}"),
            dtype=torch.bfloat16,
            max_batch_size=4,
            total_heads=total_heads,
            head_dim=head_dim,
            query_head_dim=query_head_dim,
        )
        assert dcp_alltoall._B12X_DCP_A2A_POOLS

        query = make_query(0)
        gathered_query = dcp_alltoall.dcp_b12x_all_gather_heads(
            query,
            group,  # type: ignore[arg-type]
            max_batch_size=4,
            output_head_dim=head_dim,
        )
        torch.accelerator.synchronize()
        torch.testing.assert_close(
            gathered_query,
            expected_query(query),
            rtol=0,
            atol=0,
        )

        partial_output, partial_lse = make_inputs(0)
        actual = dcp_alltoall.dcp_a2a_lse_reduce(
            partial_output,
            partial_lse,
            group,  # type: ignore[arg-type]
            use_b12x=True,
            b12x_max_batch_size=4,
            b12x_query_head_dim=query_head_dim,
        )
        torch.accelerator.synchronize()
        torch.testing.assert_close(
            actual.float(),
            expected(partial_output, partial_lse),
            rtol=3e-2,
            atol=3e-2,
        )

        static_output = torch.empty_like(partial_output)
        static_lse = torch.empty_like(partial_lse)
        static_query = torch.empty_like(query)

        def fail_packed_nccl(*args, **kwargs):
            raise AssertionError("captured path fell back to packed NCCL A2A")

        def fail_query_nccl(*args, **kwargs):
            raise AssertionError("captured path fell back to NCCL all-gather")

        dcp_alltoall._dcp_a2a_send_recv_buffers = fail_packed_nccl
        group.all_gather = fail_query_nccl  # type: ignore[attr-defined]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_query = dcp_alltoall.dcp_b12x_all_gather_heads(
                static_query,
                group,  # type: ignore[arg-type]
                max_batch_size=4,
                output_head_dim=head_dim,
            )
            graph_output = dcp_alltoall.dcp_a2a_lse_reduce(
                static_output,
                static_lse,
                group,  # type: ignore[arg-type]
                use_b12x=True,
                b12x_max_batch_size=4,
                b12x_query_head_dim=query_head_dim,
            )

        for step in range(1, 4):
            query = make_query(step)
            output, lse = make_inputs(step)
            static_query.copy_(query)
            static_output.copy_(output)
            static_lse.copy_(lse)
            graph.replay()
            torch.accelerator.synchronize()
            torch.testing.assert_close(
                graph_query,
                expected_query(static_query),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                graph_output.float(),
                expected(static_output, static_lse),
                rtol=3e-2,
                atol=3e-2,
            )
    finally:
        from vllm.v1.attention.ops import dcp_alltoall

        for pool in dcp_alltoall._B12X_DCP_A2A_POOLS.values():
            pool.close()
        dcp_alltoall._B12X_DCP_A2A_POOLS.clear()
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.accelerator.device_count() < 4, reason="Need at least 4 GPUs."
)
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16", "float32"])
def test_distributed_packed_a2a_matches_reference(dtype_name: str):
    _distributed_run(
        _distributed_packed_a2a_worker,
        world_size=4,
        extra_env={
            "TEST_DTYPE": dtype_name,
            "RETURN_LSE": "1",
            "LSE_BASE_E": "1",
        },
    )


@pytest.mark.skipif(
    torch.accelerator.device_count() < 4, reason="Need at least 4 GPUs."
)
def test_distributed_packed_a2a_with_workspace_matches_reference():
    _distributed_run(
        _distributed_packed_a2a_worker,
        world_size=4,
        extra_env={
            "TEST_DTYPE": "bfloat16",
            "RETURN_LSE": "1",
            "LSE_BASE_E": "1",
            "USE_WORKSPACE": "1",
        },
    )


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2
    or importlib.util.find_spec("sparkinfer") is None,
    reason="Need two GPUs and sparkinfer.",
)
def test_distributed_b12x_a2a_eager_and_graph_matches_reference():
    from sparkinfer.comm.pcie.pcie_dcp_a2a import _load_extension

    _load_extension()
    _distributed_run(
        _distributed_b12x_a2a_worker,
        world_size=2,
        extra_env={"VLLM_USE_B12X_DCP_A2A": "1"},
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
