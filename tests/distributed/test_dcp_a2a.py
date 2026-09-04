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
from contextlib import ExitStack, contextmanager, nullcontext
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

    def test_empty_shard_ignores_undefined_output(self):
        from vllm.v1.attention.ops.dcp_alltoall import _lse_weighted_combine

        outputs = torch.tensor([[[[float("nan")]]], [[[3.0]]]])
        lses = torch.tensor([[[-float("inf")]], [[0.0]]])

        result = _lse_weighted_combine(outputs, lses)

        torch.testing.assert_close(result, outputs[1])

    def test_ag_rs_masks_empty_shard_and_padded_lse(self, monkeypatch):
        import vllm.v1.attention.ops.common as common

        class FakeGroup:
            world_size = 2
            rank_in_group = 0

            def all_gather(self, tensor, dim):
                assert dim == 0
                return torch.cat((tensor, tensor), dim=dim)

        monkeypatch.setattr(
            common,
            "correct_attn_out",
            lambda output, lses, *args, **kwargs: (output, lses[0]),
        )
        output = torch.ones(7, 1, 1)
        lse = torch.ones(7, 1)
        seq_lens = torch.tensor([0, 2], dtype=torch.int32)
        query_start_loc = torch.tensor([0, 1, 5], dtype=torch.int32)

        _, masked_lse = common._cp_lse_common(
            output,
            lse,
            FakeGroup(),
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
        )

        assert torch.isneginf(masked_lse[:1]).all()
        torch.testing.assert_close(masked_lse[1:5], torch.ones_like(masked_lse[1:5]))
        assert torch.isneginf(masked_lse[5:]).all()

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


def test_a2a_converts_activation_dtype_lse_before_packing(monkeypatch):
    from vllm.v1.attention.ops import dcp_alltoall

    output = torch.zeros(2, 4, 8, dtype=torch.bfloat16)
    lse = torch.zeros(2, 4, dtype=torch.bfloat16)
    send = torch.empty(2, 2, 2, 10, dtype=torch.bfloat16)
    recv = torch.empty_like(send)
    result = torch.empty(2, 2, 8, dtype=torch.bfloat16)
    received: dict[str, torch.dtype] = {}

    monkeypatch.setattr(
        dcp_alltoall,
        "_dcp_a2a_send_recv_buffers",
        lambda *args, **kwargs: (send, recv),
    )

    def record_pack(_cp_attn_out, cp_attn_lse, *_args, **_kwargs):
        received["dtype"] = cp_attn_lse.dtype

    class _Work:
        def wait(self):
            return None

    monkeypatch.setattr(dcp_alltoall, "_dcp_a2a_pack_send", record_pack)
    monkeypatch.setattr(
        dcp_alltoall.dist,
        "all_to_all_single",
        lambda *args, **kwargs: _Work(),
    )
    monkeypatch.setattr(
        dcp_alltoall,
        "_dcp_a2a_unpack_combine",
        lambda *args, **kwargs: result,
    )
    group = _FakeCPGroup(2, object())  # type: ignore[arg-type]

    actual = dcp_alltoall.dcp_a2a_lse_reduce(
        output,
        lse,
        group,  # type: ignore[arg-type]
    )

    assert actual is result
    assert received == {"dtype": torch.float32}


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


def test_b12x_large_batch_uses_configured_ag_rs(monkeypatch: pytest.MonkeyPatch):
    from vllm.v1.attention.ops import common, dcp_alltoall

    monkeypatch.setenv("VLLM_USE_B12X_DCP_A2A", "1")
    monkeypatch.setenv("VLLM_DCP_A2A_MAX_TOKENS", "4")
    monkeypatch.setenv("VLLM_DCP_A2A_LARGE_BACKEND", "ag_rs")
    partial_output = torch.zeros(5, 4, 8, dtype=torch.bfloat16)
    partial_lse = torch.zeros(5, 4, dtype=torch.float32)
    expected = torch.ones(5, 2, 8, dtype=torch.bfloat16)
    captured: dict[str, Any] = {}
    group = _FakeCPGroup(2, object())  # type: ignore[arg-type]

    monkeypatch.setattr(
        dcp_alltoall,
        "_try_b12x_dcp_lse_reduce",
        lambda *args, **kwargs: None,
    )

    def fake_ag_rs(output, lse, cp_group, **kwargs):
        captured.update(output=output, lse=lse, group=cp_group, **kwargs)
        return expected

    monkeypatch.setattr(common, "cp_lse_ag_out_rs", fake_ag_rs)
    monkeypatch.setattr(
        dcp_alltoall.dist,
        "all_to_all_single",
        lambda *args, **kwargs: pytest.fail("packed NCCL A2A must not run"),
    )

    actual = dcp_alltoall.dcp_a2a_lse_reduce(
        partial_output,
        partial_lse,
        group,  # type: ignore[arg-type]
        use_b12x=True,
        b12x_max_batch_size=4,
    )

    assert actual is expected
    assert captured == {
        "output": partial_output,
        "lse": partial_lse,
        "group": group,
        "ctx": None,
        "return_lse": False,
        "is_lse_base_on_e": True,
        "head_major_output": True,
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

        def prepare_channels(self, channel_ids):
            captured["prepared"] = tuple(channel_ids)

        def for_stream(self, *, channel_id):
            captured["warmed"] = channel_id

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
    assert captured["max_concurrent_channels"] == 2
    assert captured["prepared"] == ("vllm:eager:dcp",)
    assert captured["warmed"] == "vllm:eager:dcp"


def test_b12x_dcp_capture_selects_only_current_group_pools(monkeypatch):
    from vllm.v1.attention.ops import dcp_alltoall

    events: list[Any] = []

    class _FakePool:
        def __init__(self, name):
            self.name = name

        @contextmanager
        def capture(self, *, stream, channel_id):
            events.append(("enter", self.name, stream, channel_id))
            try:
                yield
            finally:
                events.append(("exit", self.name, stream, channel_id))

    device_group = object()
    group = _FakeCPGroup(2, device_group)  # type: ignore[arg-type]
    stream = object()
    pools = {
        (id(device_group), 0, 64, 512, 576, 64): _FakePool("output"),
        (id(device_group), 0, 64, 576, 576, 64): _FakePool("query"),
        (id(object()), 0, 64, 512, 576, 64): _FakePool("foreign"),
    }
    monkeypatch.setattr(dcp_alltoall, "_B12X_DCP_A2A_POOLS", pools)

    with dcp_alltoall.capture_b12x_dcp_a2a(  # type: ignore[arg-type]
        group,
        stream,
        channel_id="vllm:target:profile",
    ):
        events.append(("body", None, stream, "vllm:target:profile"))

    assert events == [
        ("enter", "output", stream, "vllm:target:profile"),
        ("enter", "query", stream, "vllm:target:profile"),
        ("body", None, stream, "vllm:target:profile"),
        ("exit", "query", stream, "vllm:target:profile"),
        ("exit", "output", stream, "vllm:target:profile"),
    ]


def test_b12x_dcp_capture_reuses_nested_shared_group_scope(monkeypatch):
    from vllm.v1.attention.ops import dcp_alltoall

    events: list[str] = []

    class _FakePool:
        @contextmanager
        def capture(self, *, stream, channel_id):
            events.append("enter")
            try:
                yield
            finally:
                events.append("exit")

    device_group = object()
    tp_group = _FakeCPGroup(16, device_group)  # type: ignore[arg-type]
    dcp_group = _FakeCPGroup(16, device_group)  # type: ignore[arg-type]
    stream = object()
    key = (id(device_group), 0, 64, 512, 576, 64)
    monkeypatch.setattr(dcp_alltoall, "_B12X_DCP_A2A_POOLS", {key: _FakePool()})
    monkeypatch.setattr(dcp_alltoall, "_B12X_DCP_ACTIVE_CAPTURE", {})

    with (
        dcp_alltoall.capture_b12x_dcp_a2a(
            tp_group,  # type: ignore[arg-type]
            stream,
            channel_id="vllm:target:decode",
        ),
        dcp_alltoall.capture_b12x_dcp_a2a(
            dcp_group,  # type: ignore[arg-type]
            stream,
            channel_id="vllm:target:decode",
        ),
    ):
        events.append("body")

    assert events == ["enter", "body", "exit"]


def test_b12x_dcp_capture_rejects_missing_semantic_id(monkeypatch):
    from vllm.v1.attention.ops import dcp_alltoall

    device_group = object()
    group = _FakeCPGroup(2, device_group)  # type: ignore[arg-type]
    key = (id(device_group), 0, 64, 512, 576, 64)
    monkeypatch.setattr(
        dcp_alltoall,
        "_B12X_DCP_A2A_POOLS",
        {key: object()},
    )

    with (
        pytest.raises(RuntimeError, match="semantic channel_id"),
        dcp_alltoall.capture_b12x_dcp_a2a(group),
    ):
        pass


def test_b12x_pool_initialized_inside_graph_context_joins_graph_channel(monkeypatch):
    from vllm.v1.attention.ops import dcp_alltoall

    events: list[Any] = []

    class _FakePool:
        @classmethod
        def from_exchange_group(cls, **kwargs):
            events.append(("create", kwargs["exchange_group"]))
            return cls()

        def prepare_channels(self, channel_ids):
            events.append(("prepare", tuple(channel_ids)))

        def for_stream(self, *, channel_id):
            events.append(("eager", channel_id))

        @contextmanager
        def capture(self, *, stream, channel_id):
            events.append(("enter", stream, channel_id))
            try:
                yield
            finally:
                events.append(("exit", stream, channel_id))

    device_group = object()
    group = _FakeCPGroup(2, device_group)  # type: ignore[arg-type]
    stream = object()
    monkeypatch.setattr(dcp_alltoall, "_B12X_DCP_A2A_POOLS", {})
    monkeypatch.setattr(dcp_alltoall, "_B12X_DCP_A2A_DISABLED", set())
    monkeypatch.setattr(dcp_alltoall, "_B12X_DCP_ACTIVE_CAPTURE", {})
    monkeypatch.setattr(dcp_alltoall, "_load_b12x_dcp_a2a_pool", lambda: _FakePool)
    monkeypatch.setattr(dcp_alltoall, "_b12x_dcp_init_failed", lambda *args: False)
    monkeypatch.setattr(
        dcp_alltoall.torch.cuda,
        "is_current_stream_capturing",
        lambda: False,
    )

    with dcp_alltoall.capture_b12x_dcp_a2a(
        group,  # type: ignore[arg-type]
        stream,
        channel_id="vllm:target:decode",
    ):
        pool = dcp_alltoall._get_b12x_dcp_a2a_pool(
            group,  # type: ignore[arg-type]
            device=torch.device("cuda:0"),
            total_heads=64,
            head_dim=512,
            query_head_dim=576,
            max_batch_size=64,
        )
        assert pool is not None
        assert (
            dcp_alltoall._b12x_dcp_channel_id(group)  # type: ignore[arg-type]
            == "vllm:target:decode"
        )

    assert dcp_alltoall._b12x_dcp_channel_id(group) == "vllm:eager:dcp"  # type: ignore[arg-type]
    assert events == [
        ("create", device_group),
        ("prepare", ("vllm:eager:dcp", "vllm:target:decode")),
        ("enter", stream, "vllm:target:decode"),
        ("exit", stream, "vllm:target:decode"),
    ]


def test_b12x_dcp_channel_rollback_restores_existing_and_closes_new_pools(
    monkeypatch,
):
    from vllm.v1.attention.ops import dcp_alltoall

    events: list[Any] = []

    class _FakePool:
        def __init__(self, name):
            self.name = name

        def checkpoint_channels(self):
            events.append(("checkpoint", self.name))
            return f"{self.name}-checkpoint"

        def rollback_channels(self, checkpoint):
            events.append(("rollback", self.name, checkpoint))

        def close(self):
            events.append(("close", self.name))

    device_group = object()
    group = _FakeCPGroup(2, device_group)  # type: ignore[arg-type]
    existing_key = (id(device_group), 0, 64, 512, 576, 64)
    new_key = (id(device_group), 0, 64, 576, 576, 64)
    foreign_key = (id(object()), 0, 64, 512, 576, 64)
    existing = _FakePool("existing")
    foreign = _FakePool("foreign")
    pools = {existing_key: existing, foreign_key: foreign}
    monkeypatch.setattr(dcp_alltoall, "_B12X_DCP_A2A_POOLS", pools)

    checkpoint = dcp_alltoall.checkpoint_b12x_dcp_a2a_channels(group)
    transient = _FakePool("transient")
    pools[new_key] = transient
    dcp_alltoall.rollback_b12x_dcp_a2a_channels(checkpoint)

    assert pools == {existing_key: existing, foreign_key: foreign}
    assert events == [
        ("checkpoint", "existing"),
        ("rollback", "existing", "existing-checkpoint"),
        ("close", "transient"),
    ]


def test_profile_channel_checkpoint_rolls_back_all_b12x_transports(monkeypatch):
    from vllm.distributed import parallel_state
    from vllm.v1.attention.ops import dcp_alltoall

    events: list[Any] = []

    class _FakeCommunicator:
        def checkpoint_pcie_channels(self):
            events.append("checkpoint-tp")
            return "tp-checkpoint"

        def rollback_pcie_channels(self, checkpoint):
            events.append(("rollback-tp", checkpoint))

    class _FakeGroup:
        def __init__(self, name, *, world_size, communicator=None):
            self.name = name
            self.world_size = world_size
            self.device_group = object()
            self.device_communicator = type(
                "DeviceCommunicator", (), {"ca_comm": communicator}
            )()

    communicator = _FakeCommunicator()
    tp_group = _FakeGroup("tp", world_size=8, communicator=communicator)
    pp_group = _FakeGroup("pp", world_size=1, communicator=communicator)
    dcp_group = _FakeGroup("dcp", world_size=2)
    monkeypatch.setattr(parallel_state, "_TP", tp_group)
    monkeypatch.setattr(parallel_state, "_PP", pp_group)
    monkeypatch.setattr(parallel_state, "_DCP", dcp_group)

    def checkpoint_dcp(group):
        events.append(("checkpoint-pool", group.name))
        return f"{group.name}-pool-checkpoint"

    monkeypatch.setattr(
        dcp_alltoall,
        "checkpoint_b12x_dcp_a2a_channels",
        checkpoint_dcp,
    )
    monkeypatch.setattr(
        dcp_alltoall,
        "rollback_b12x_dcp_a2a_channels",
        lambda checkpoint: events.append(("rollback-dcp", checkpoint)),
    )

    checkpoint = parallel_state.checkpoint_b12x_graph_channels()
    parallel_state.rollback_b12x_graph_channels(checkpoint)

    assert events == [
        "checkpoint-tp",
        ("checkpoint-pool", "tp"),
        ("checkpoint-pool", "dcp"),
        ("rollback-dcp", "dcp-pool-checkpoint"),
        ("rollback-dcp", "tp-pool-checkpoint"),
        ("rollback-tp", "tp-checkpoint"),
    ]


def test_profile_channel_checkpoint_deduplicates_shared_b12x_pool(monkeypatch):
    from vllm.distributed import parallel_state
    from vllm.v1.attention.ops import dcp_alltoall

    class _FakeGroup:
        def __init__(self, name, *, world_size, device_group):
            self.name = name
            self.world_size = world_size
            self.device_group = device_group
            self.device_communicator = type(
                "DeviceCommunicator", (), {"ca_comm": None}
            )()

    events: list[str] = []
    shared_device_group = object()
    tp_group = _FakeGroup("tp", world_size=16, device_group=shared_device_group)
    dcp_group = _FakeGroup("dcp", world_size=16, device_group=shared_device_group)
    monkeypatch.setattr(parallel_state, "_TP", tp_group)
    monkeypatch.setattr(parallel_state, "_PP", None)
    monkeypatch.setattr(parallel_state, "_DCP", dcp_group)

    def checkpoint(group: _FakeGroup) -> str:
        events.append(group.name)
        return group.name

    monkeypatch.setattr(
        dcp_alltoall,
        "checkpoint_b12x_dcp_a2a_channels",
        checkpoint,
    )

    parallel_state.checkpoint_b12x_graph_channels()

    assert events == ["tp"]


def test_global_graph_capture_enters_b12x_tp_and_dcp_pools(monkeypatch):
    from vllm.distributed import parallel_state
    from vllm.v1.attention.ops import dcp_alltoall

    events: list[Any] = []

    class _FakeGroup:
        world_size = 2

        def __init__(self, name):
            self.name = name
            self.device_group = object()

        @contextmanager
        def graph_capture(self, context):
            events.append(("enter-group", self.name, context.channel_id))
            try:
                yield context
            finally:
                events.append(("exit-group", self.name, context.channel_id))

    tp_group = _FakeGroup("tp")
    pp_group = _FakeGroup("pp")
    dcp_group = _FakeGroup("dcp")
    stream = object()
    context = parallel_state.GraphCaptureContext(  # type: ignore[arg-type]
        stream,
        channel_id="vllm:target:profile",
    )

    @contextmanager
    def fake_b12x_capture(group, selected_stream, *, channel_id):
        events.append(("enter-b12x", group, selected_stream, channel_id))
        try:
            yield
        finally:
            events.append(("exit-b12x", group, selected_stream, channel_id))

    monkeypatch.setattr(parallel_state, "_DCP", dcp_group)
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(parallel_state, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(parallel_state, "get_dcp_group", lambda: dcp_group)
    monkeypatch.setattr(dcp_alltoall, "capture_b12x_dcp_a2a", fake_b12x_capture)

    with parallel_state.graph_capture(torch.device("cpu"), context) as actual:
        assert actual is context

    assert events == [
        ("enter-group", "tp", "vllm:target:profile"),
        ("enter-group", "pp", "vllm:target:profile"),
        ("enter-group", "dcp", "vllm:target:profile"),
        (
            "enter-b12x",
            tp_group,
            stream,
            "vllm:target:profile",
        ),
        (
            "enter-b12x",
            dcp_group,
            stream,
            "vllm:target:profile",
        ),
        (
            "exit-b12x",
            dcp_group,
            stream,
            "vllm:target:profile",
        ),
        (
            "exit-b12x",
            tp_group,
            stream,
            "vllm:target:profile",
        ),
        ("exit-group", "dcp", "vllm:target:profile"),
        ("exit-group", "pp", "vllm:target:profile"),
        ("exit-group", "tp", "vllm:target:profile"),
    ]


def test_global_graph_capture_keeps_distinct_b12x_coordinator_scopes(monkeypatch):
    from vllm.distributed import parallel_state
    from vllm.v1.attention.ops import dcp_alltoall

    class _FakeGroup:
        world_size = 16

        def __init__(self, name, device_group):
            self.name = name
            self.device_group = device_group

        @contextmanager
        def graph_capture(self, context):
            yield context

    shared_device_group = object()
    tp_group = _FakeGroup("tp", shared_device_group)
    dcp_group = _FakeGroup("dcp", shared_device_group)
    pp_group = _FakeGroup("pp", object())
    entered_pools: list[str] = []

    @contextmanager
    def fake_b12x_capture(group, selected_stream, *, channel_id):
        entered_pools.append(group.name)
        yield

    monkeypatch.setattr(parallel_state, "_DCP", dcp_group)
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(parallel_state, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(parallel_state, "get_dcp_group", lambda: dcp_group)
    monkeypatch.setattr(dcp_alltoall, "capture_b12x_dcp_a2a", fake_b12x_capture)
    context = parallel_state.GraphCaptureContext(  # type: ignore[arg-type]
        object(),
        channel_id="vllm:target:decode",
    )

    with parallel_state.graph_capture(torch.device("cpu"), context):
        pass

    assert entered_pools == ["tp", "dcp"]


def test_group_capture_forwards_semantic_id_to_custom_allreduce(monkeypatch):
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.distributed import parallel_state
    from vllm.distributed.device_communicators.cuda_communicator import (
        CudaCommunicator,
    )

    events: list[Any] = []

    class FakeCustomAllreduce:
        @contextmanager
        def capture(self, *, stream, channel_id):
            events.append(("enter", stream, channel_id))
            try:
                yield
            finally:
                events.append(("exit", stream, channel_id))

    communicator = object.__new__(CudaCommunicator)
    communicator.ca_comm = FakeCustomAllreduce()
    group = object.__new__(parallel_state.GroupCoordinator)
    group.device_communicator = communicator
    stream = object()
    context = parallel_state.GraphCaptureContext(  # type: ignore[arg-type]
        stream,
        channel_id="vllm:draft:decode:production",
    )

    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: False)
    monkeypatch.setattr(
        parallel_state.torch.cuda,
        "current_stream",
        lambda: stream,
    )
    monkeypatch.setattr(
        parallel_state.torch.cuda,
        "stream",
        lambda selected: nullcontext(),
    )

    with parallel_state.GroupCoordinator.graph_capture(group, context) as actual:
        assert actual is context

    assert events == [
        ("enter", stream, "vllm:draft:decode:production"),
        ("exit", stream, "vllm:draft:decode:production"),
    ]


@pytest.mark.skipif(torch.accelerator.device_count() < 1, reason="CUDA is required.")
@pytest.mark.parametrize("world_size", [4, 16])
def test_b12x_lse_reduce_honors_token_cap(
    monkeypatch: pytest.MonkeyPatch, world_size: int
):
    from vllm.v1.attention.ops import dcp_alltoall

    monkeypatch.setenv("VLLM_USE_B12X_DCP_A2A", "1")
    monkeypatch.setenv("VLLM_DCP_A2A_MAX_TOKENS", "4")
    created: dict[str, Any] = {}
    sentinel = torch.zeros(1)

    class _FakePool:
        def lse_reduce_scatter(
            self,
            partial,
            lse,
            out=None,
            *,
            is_lse_base_on_e,
            channel_id,
        ):
            created["channel_id"] = channel_id
            return sentinel

    def fake_get_pool(
        cp_group, *, device, total_heads, head_dim, query_head_dim, max_batch_size
    ):
        created["max_batch_size"] = max_batch_size
        return _FakePool()

    monkeypatch.setattr(dcp_alltoall, "_get_b12x_dcp_a2a_pool", fake_get_pool)
    group = _FakeCPGroup(world_size, None)  # type: ignore[arg-type]

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
    assert created["channel_id"] == "vllm:eager:dcp"

    monkeypatch.setattr(
        dcp_alltoall,
        "_B12X_DCP_ACTIVE_CAPTURE",
        {id(group.device_group): ("vllm:target:decode", None, ExitStack())},
    )
    result = dcp_alltoall._try_b12x_dcp_lse_reduce(
        out,
        lse,
        group,  # type: ignore[arg-type]
        return_lse=False,
        is_lse_base_on_e=True,
        max_batch_size=8192,
        query_head_dim=64,
    )
    assert result is sentinel
    assert created["channel_id"] == "vllm:target:decode"

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
@pytest.mark.parametrize("world_size", [4, 16])
def test_b12x_query_gather_honors_token_cap(
    monkeypatch: pytest.MonkeyPatch, world_size: int
):
    from vllm.v1.attention.ops import dcp_alltoall

    monkeypatch.setenv("VLLM_USE_B12X_DCP_A2A", "1")
    monkeypatch.setenv("VLLM_DCP_A2A_MAX_TOKENS", "4")
    created: dict[str, Any] = {}
    sentinel = torch.zeros(1)

    class _FakePool:
        def all_gather_heads(self, local_input, *, channel_id):
            created["channel_id"] = channel_id
            return sentinel

    def fake_get_pool(
        cp_group, *, device, total_heads, head_dim, query_head_dim, max_batch_size
    ):
        created["max_batch_size"] = max_batch_size
        return _FakePool()

    monkeypatch.setattr(dcp_alltoall, "_get_b12x_dcp_a2a_pool", fake_get_pool)
    group = _FakeCPGroup(world_size, None)  # type: ignore[arg-type]

    small = torch.zeros(4, 8, 64, dtype=torch.bfloat16, device="cuda")
    result = dcp_alltoall._try_b12x_dcp_all_gather_heads(
        small,
        group,  # type: ignore[arg-type]
        max_batch_size=8192,
        output_head_dim=64,
    )
    assert result is sentinel
    assert created["max_batch_size"] == 4
    assert created["channel_id"] == "vllm:eager:dcp"

    monkeypatch.setattr(
        dcp_alltoall,
        "_B12X_DCP_ACTIVE_CAPTURE",
        {id(group.device_group): ("vllm:target:decode", None, ExitStack())},
    )
    result = dcp_alltoall._try_b12x_dcp_all_gather_heads(
        small,
        group,  # type: ignore[arg-type]
        max_batch_size=8192,
        output_head_dim=64,
    )
    assert result is sentinel
    assert created["channel_id"] == "vllm:target:decode"

    large = torch.zeros(8, 8, 64, dtype=torch.bfloat16, device="cuda")
    result = dcp_alltoall._try_b12x_dcp_all_gather_heads(
        large,
        group,  # type: ignore[arg-type]
        max_batch_size=8192,
        output_head_dim=64,
    )
    assert result is None


@pytest.mark.skipif(torch.accelerator.device_count() < 1, reason="CUDA is required.")
def test_b12x_query_gather_supports_fp8_caller_output(monkeypatch):
    from vllm.v1.attention.ops import dcp_alltoall

    received: dict[str, Any] = {}

    class _FakePool:
        def all_gather_heads(self, local_input, *, out, channel_id):
            received.update(input=local_input, out=out, channel_id=channel_id)
            return out

    def fake_get_pool(
        cp_group, *, device, total_heads, head_dim, query_head_dim, max_batch_size
    ):
        received.update(
            total_heads=total_heads,
            head_dim=head_dim,
            query_head_dim=query_head_dim,
            max_batch_size=max_batch_size,
        )
        return _FakePool()

    monkeypatch.setattr(dcp_alltoall, "_get_b12x_dcp_a2a_pool", fake_get_pool)
    group = _FakeCPGroup(4, object())  # type: ignore[arg-type]
    local_input = torch.zeros(2, 2, 64, dtype=torch.float8_e4m3fn, device="cuda")
    out = torch.empty(2, 8, 64, dtype=local_input.dtype, device=local_input.device)

    actual = dcp_alltoall._try_b12x_dcp_all_gather_heads(
        local_input,
        group,  # type: ignore[arg-type]
        max_batch_size=8,
        output_head_dim=512,
        out=out,
    )

    assert actual is out
    assert received.pop("input") is local_input
    assert received.pop("out") is out
    assert received == {
        "channel_id": "vllm:eager:dcp",
        "total_heads": 8,
        "head_dim": 512,
        "query_head_dim": 64,
        "max_batch_size": 8,
    }

    unaligned = torch.zeros(
        2, 2, 24, dtype=torch.float8_e4m3fn, device=local_input.device
    )
    assert (
        dcp_alltoall._try_b12x_dcp_all_gather_heads(
            unaligned,
            group,  # type: ignore[arg-type]
            max_batch_size=8,
            output_head_dim=512,
        )
        is None
    )


def test_query_gather_fallback_copies_into_caller_output(monkeypatch):
    from vllm.v1.attention.ops import dcp_alltoall

    monkeypatch.delenv("VLLM_USE_B12X_DCP_A2A", raising=False)
    local_input = torch.zeros(2, 2, 8, dtype=torch.bfloat16)
    gathered = torch.arange(64, dtype=torch.bfloat16).reshape(2, 4, 8)
    out = torch.empty_like(gathered)
    group = _FakeCPGroup(2, object())  # type: ignore[arg-type]
    group.all_gather = lambda _value, dim: gathered  # type: ignore[attr-defined]

    actual = dcp_alltoall.dcp_b12x_all_gather_heads(
        local_input,
        group,  # type: ignore[arg-type]
        out=out,
    )

    assert actual is out
    torch.testing.assert_close(actual, gathered)


@pytest.mark.skipif(torch.accelerator.device_count() < 1, reason="CUDA is required.")
def test_b12x_lse_reduce_preserves_supported_layouts(monkeypatch: pytest.MonkeyPatch):
    """Preserve head-major input while materializing legacy head slices."""
    from vllm.v1.attention.ops import dcp_alltoall

    monkeypatch.setenv("VLLM_USE_B12X_DCP_A2A", "1")
    received: dict[str, Any] = {}
    sentinel = torch.zeros(1)

    class _FakePool:
        def lse_reduce_scatter(
            self,
            partial,
            lse,
            out=None,
            *,
            is_lse_base_on_e,
            channel_id,
        ):
            received.update(
                partial=partial,
                lse=lse,
                out=out,
                channel_id=channel_id,
            )
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
    assert received["channel_id"] == "vllm:eager:dcp"

    head_major_storage = torch.zeros(16, 8, 64, dtype=torch.bfloat16, device="cuda")
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


@pytest.mark.skipif(torch.accelerator.device_count() < 1, reason="CUDA is required.")
def test_b12x_pair_gather_uses_one_pool_operation_at_dcp16(monkeypatch):
    from vllm.v1.attention.ops import dcp_alltoall

    monkeypatch.setenv("VLLM_USE_B12X_DCP_A2A", "1")
    received: dict[str, Any] = {}
    local_first = torch.zeros(2, 224, dtype=torch.bfloat16, device="cuda")
    local_second = torch.zeros(2, 56, dtype=torch.float32, device="cuda")
    expected = (
        torch.empty(2, 224 * 16, dtype=local_first.dtype, device="cuda"),
        torch.empty(2, 56 * 16, dtype=local_second.dtype, device="cuda"),
    )

    class _FakePool:
        def all_gather_pair(self, first, second, *, channel_id):
            received.update(first=first, second=second, channel_id=channel_id)
            return expected

    def fake_get_pool(cp_group, **kwargs):
        received.update(kwargs)
        return _FakePool()

    monkeypatch.setattr(dcp_alltoall, "_get_b12x_dcp_a2a_pool", fake_get_pool)
    monkeypatch.setattr(
        dcp_alltoall,
        "_b12x_dcp_channel_id",
        lambda _group: "vllm:target:decode:graph-1",
    )
    group = _FakeCPGroup(16, object())  # type: ignore[arg-type]

    actual = dcp_alltoall.dcp_b12x_all_gather_pair(
        local_first,
        local_second,
        group,  # type: ignore[arg-type]
        max_batch_size=8,
    )

    assert actual is expected
    assert received.pop("first") is local_first
    assert received.pop("second") is local_second
    assert received.pop("device") == local_first.device
    assert received == {
        "channel_id": "vllm:target:decode:graph-1",
        "total_heads": 16,
        "head_dim": 672,
        "query_head_dim": 672,
        "max_batch_size": 8,
    }
    unaligned_first = local_first[:, :223].contiguous()
    assert (
        dcp_alltoall._try_b12x_dcp_all_gather_pair(
            unaligned_first,
            local_second,
            group,  # type: ignore[arg-type]
            max_batch_size=8,
        )
        is None
    )


def test_pair_gather_fallback_preserves_tensor_order(monkeypatch):
    from vllm.v1.attention.ops import dcp_alltoall

    monkeypatch.delenv("VLLM_USE_B12X_DCP_A2A", raising=False)
    local_first = torch.zeros(2, 8)
    local_second = torch.ones(2, 4)
    gathered_first = torch.full((2, 16), 2.0)
    gathered_second = torch.full((2, 8), 3.0)
    calls: list[tuple[torch.Tensor, int]] = []

    def gather(value, dim):
        calls.append((value, dim))
        return gathered_first if value is local_first else gathered_second

    group = _FakeCPGroup(2, object())  # type: ignore[arg-type]
    group.all_gather = gather  # type: ignore[attr-defined]

    actual = dcp_alltoall.dcp_b12x_all_gather_pair(
        local_first,
        local_second,
        group,  # type: ignore[arg-type]
    )

    assert actual[0] is gathered_first
    assert actual[1] is gathered_second
    assert len(calls) == 2
    assert calls[0][0] is local_first and calls[0][1] == -1
    assert calls[1][0] is local_second and calls[1][1] == -1


@pytest.mark.parametrize("world_size", [2, 4, 8, 16])
@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.skipif(torch.accelerator.device_count() < 1, reason="CUDA is required.")
def test_b12x_kimi_pair_topk_supports_projection_world_sizes(
    monkeypatch: pytest.MonkeyPatch,
    world_size: int,
    batch: int,
):
    from vllm.v1.attention.ops import dcp_alltoall

    monkeypatch.setenv("VLLM_USE_B12X_DCP_A2A", "1")
    local_down = torch.zeros(
        batch, 3584 // world_size, dtype=torch.bfloat16, device="cuda"
    )
    local_router = torch.zeros(
        batch, 896 // world_size, dtype=torch.float32, device="cuda"
    )
    correction_bias = torch.zeros(896, dtype=torch.float32, device="cuda")
    received: dict[str, Any] = {}

    class _FakePool:
        def all_gather_pair_kimi_topk(
            self,
            down,
            router,
            bias,
            out_down,
            topk_weights,
            topk_ids,
            *,
            channel_id,
        ):
            received.update(
                operation="combined",
                down=down,
                router=router,
                bias=bias,
                out_down=out_down,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                channel_id=channel_id,
            )

        def all_gather_pair(self, down, router, *, channel_id):
            gathered_down = torch.empty(
                batch, 3584, dtype=torch.bfloat16, device="cuda"
            )
            gathered_router = torch.empty(
                batch, 896, dtype=torch.float32, device="cuda"
            )
            received.update(
                operation="batched",
                down=down,
                router=router,
                gathered_down=gathered_down,
                gathered_router=gathered_router,
                channel_id=channel_id,
            )
            return gathered_down, gathered_router

        def kimi_topk16(
            self,
            router,
            bias,
            topk_weights,
            topk_ids,
            *,
            channel_id,
        ):
            received.update(
                topk_router=router,
                bias=bias,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                topk_channel_id=channel_id,
            )

    def fake_get_pool(cp_group, **kwargs):
        received.update(kwargs)
        return _FakePool()

    monkeypatch.setattr(dcp_alltoall, "_get_b12x_dcp_a2a_pool", fake_get_pool)
    monkeypatch.setattr(
        dcp_alltoall,
        "_b12x_dcp_channel_id",
        lambda _group: "vllm:draft:decode:graph-2",
    )
    group = _FakeCPGroup(world_size, object())  # type: ignore[arg-type]

    actual = dcp_alltoall.try_dcp_b12x_all_gather_pair_kimi_topk(
        local_down,
        local_router,
        correction_bias,
        group,  # type: ignore[arg-type]
    )

    assert actual is not None
    gathered_down, routing_payload = actual
    assert gathered_down.shape == (batch, 3584)
    assert gathered_down.dtype == torch.bfloat16
    assert routing_payload.shape == (batch * 2, 16)
    assert routing_payload.dtype == torch.float32
    assert received.pop("down") is local_down
    assert received.pop("router") is local_router
    assert received.pop("bias") is correction_bias
    assert received.pop("topk_weights").data_ptr() == routing_payload.data_ptr()
    assert received.pop("topk_ids").data_ptr() == routing_payload[batch].data_ptr()
    if batch == 1:
        assert received.pop("operation") == "combined"
        assert received.pop("out_down") is gathered_down
    else:
        assert received.pop("operation") == "batched"
        assert received.pop("gathered_down") is gathered_down
        assert received.pop("topk_router") is received.pop("gathered_router")
        assert received.pop("topk_channel_id") == "vllm:draft:decode:graph-2"
    combined_row_bytes = 7168 // world_size + 3584 // world_size
    assert received == {
        "channel_id": "vllm:draft:decode:graph-2",
        "device": local_down.device,
        "total_heads": world_size,
        "head_dim": combined_row_bytes,
        "query_head_dim": combined_row_bytes,
        "max_batch_size": 1 if batch == 1 else 8,
    }


@pytest.mark.skipif(torch.accelerator.device_count() < 1, reason="CUDA is required.")
def test_b12x_kimi_pair_topk_rejects_non_contract_inputs(monkeypatch):
    from vllm.v1.attention.ops import dcp_alltoall

    monkeypatch.setenv("VLLM_USE_B12X_DCP_A2A", "1")
    monkeypatch.setattr(
        dcp_alltoall,
        "_get_b12x_dcp_a2a_pool",
        lambda *args, **kwargs: pytest.fail("invalid inputs must not create a pool"),
    )
    group = _FakeCPGroup(8, object())  # type: ignore[arg-type]
    down = torch.zeros(1, 448, dtype=torch.bfloat16, device="cuda")
    router = torch.zeros(1, 112, dtype=torch.float32, device="cuda")
    bias = torch.zeros(896, dtype=torch.float32, device="cuda")

    invalid_inputs = (
        (down.expand(9, -1), router.expand(9, -1), bias),
        (down.expand(2, -1), router, bias),
        (down[:, :-1].contiguous(), router, bias),
        (down, router[:, :-1].contiguous(), bias),
        (down, router, bias[:-1].contiguous()),
        (down.float(), router, bias),
        (down, router.bfloat16(), bias),
    )
    for invalid_down, invalid_router, invalid_bias in invalid_inputs:
        assert (
            dcp_alltoall.try_dcp_b12x_all_gather_pair_kimi_topk(
                invalid_down,
                invalid_router,
                invalid_bias,
                group,  # type: ignore[arg-type]
            )
            is None
        )


@pytest.mark.skipif(torch.accelerator.device_count() < 1, reason="CUDA is required.")
def test_b12x_kimi_batched_topk_requires_batched_router_binding(monkeypatch):
    from vllm.v1.attention.ops import dcp_alltoall

    monkeypatch.setenv("VLLM_USE_B12X_DCP_A2A", "1")

    class _OneTokenPool:
        def all_gather_pair_kimi_topk(self, *args, **kwargs):
            pytest.fail("an eight-token input must not use the one-token operation")

    monkeypatch.setattr(
        dcp_alltoall,
        "_get_b12x_dcp_a2a_pool",
        lambda *args, **kwargs: _OneTokenPool(),
    )
    group = _FakeCPGroup(8, object())  # type: ignore[arg-type]

    result = dcp_alltoall.try_dcp_b12x_all_gather_pair_kimi_topk(
        torch.zeros(8, 448, dtype=torch.bfloat16, device="cuda"),
        torch.zeros(8, 112, dtype=torch.float32, device="cuda"),
        torch.zeros(896, dtype=torch.float32, device="cuda"),
        group,  # type: ignore[arg-type]
    )

    assert result is None


@pytest.mark.skipif(torch.accelerator.device_count() < 1, reason="CUDA is required.")
def test_b12x_kimi_router_topk_is_stateless(monkeypatch):
    from vllm.v1.attention.ops import dcp_alltoall

    monkeypatch.setenv("VLLM_USE_B12X_DCP_A2A", "1")
    monkeypatch.setenv("VLLM_DCP_A2A_MAX_TOKENS", "1")
    router_logits = torch.zeros(8, 896, dtype=torch.float32, device="cuda")
    correction_bias = torch.zeros(896, dtype=torch.float32, device="cuda")
    received: dict[str, Any] = {}

    def kimi_topk16(router, bias, weights, ids):
        received.update(
            router=router,
            bias=bias,
            weights=weights,
            ids=ids,
        )

    monkeypatch.setattr(
        dcp_alltoall,
        "_load_b12x_kimi_topk16",
        lambda: kimi_topk16,
    )

    actual = dcp_alltoall.try_b12x_kimi_topk16(
        router_logits,
        correction_bias,
    )

    assert actual is not None
    weights, ids = actual
    assert weights.shape == ids.shape == (8, 16)
    assert weights.dtype == torch.float32
    assert ids.dtype == torch.int32
    assert received.pop("router") is router_logits
    assert received.pop("bias") is correction_bias
    assert received.pop("weights") is weights
    assert received.pop("ids") is ids
    assert received == {}


def test_b12x_kimi_router_topk_requires_b12x_transport(monkeypatch):
    from vllm.v1.attention.ops import dcp_alltoall

    monkeypatch.delenv("VLLM_USE_B12X_DCP_A2A", raising=False)
    monkeypatch.setattr(
        dcp_alltoall,
        "_load_b12x_kimi_topk16",
        lambda: pytest.fail("disabled routing must not load B12X"),
    )

    result = dcp_alltoall.try_b12x_kimi_topk16(
        torch.zeros(8, 896),
        torch.zeros(896),
    )

    assert result is None


def test_kimi_projection_warmup_respects_transport_token_cap(monkeypatch):
    from vllm.v1.attention.ops import dcp_alltoall

    monkeypatch.setenv("VLLM_USE_B12X_DCP_A2A", "1")
    monkeypatch.setenv("VLLM_DCP_A2A_MAX_TOKENS", "4")
    group = _FakeCPGroup(8, object())  # type: ignore[arg-type]
    calls: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    def pair(local_down, local_router, projection_group, *, max_batch_size):
        assert projection_group is group
        assert max_batch_size == 8
        calls.append(("pair", tuple(local_down.shape), tuple(local_router.shape)))
        return local_down, local_router

    def topk(local_down, local_router, correction_bias, projection_group):
        assert projection_group is group
        assert correction_bias.shape == (896,)
        calls.append(("topk", tuple(local_down.shape), tuple(local_router.shape)))
        return local_down, torch.empty(local_down.shape[0] * 2, 16)

    monkeypatch.setattr(dcp_alltoall, "_try_b12x_dcp_all_gather_pair", pair)
    monkeypatch.setattr(
        dcp_alltoall,
        "try_dcp_b12x_all_gather_pair_kimi_topk",
        topk,
    )

    warmed = dcp_alltoall.warmup_b12x_kimi_projection_gathers(
        group,  # type: ignore[arg-type]
        device=torch.device("cpu"),
    )

    assert warmed == 3
    assert calls == [
        ("pair", (4, 448), (4, 112)),
        ("topk", (1, 448), (1, 112)),
        ("topk", (4, 448), (4, 112)),
    ]


def test_kimi_router_topk_warmup_is_independent_of_dcp_world_size(monkeypatch):
    from vllm.v1.attention.ops import dcp_alltoall

    monkeypatch.setenv("VLLM_USE_B12X_DCP_A2A", "1")
    group = _FakeCPGroup(12, object())  # type: ignore[arg-type]
    calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def topk(router_logits, correction_bias):
        calls.append((tuple(router_logits.shape), tuple(correction_bias.shape)))
        return torch.empty(8, 16), torch.empty(8, 16, dtype=torch.int32)

    monkeypatch.setattr(dcp_alltoall, "try_b12x_kimi_topk16", topk)
    monkeypatch.setattr(
        dcp_alltoall,
        "_try_b12x_dcp_all_gather_pair",
        lambda *args, **kwargs: pytest.fail("TP12 has no DCP gather runtime"),
    )

    warmed = dcp_alltoall.warmup_b12x_kimi_projection_gathers(
        group,  # type: ignore[arg-type]
        device=torch.device("cpu"),
    )

    assert warmed == 1
    assert calls == [((8, 896), (896,))]


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


def test_warmup_accepts_dcp16_geometry(monkeypatch: pytest.MonkeyPatch):
    from vllm.v1.attention.ops import dcp_alltoall

    monkeypatch.setenv("VLLM_USE_B12X_DCP_A2A", "1")
    calls: list[tuple[str, tuple[int, ...]]] = []

    def gather(local_input, *args, **kwargs):
        calls.append(("gather", tuple(local_input.shape)))
        return torch.empty(1)

    def reduce(partial_output, *args, **kwargs):
        calls.append(("reduce", tuple(partial_output.shape)))
        return torch.empty(1)

    monkeypatch.setattr(dcp_alltoall, "_try_b12x_dcp_all_gather_heads", gather)
    monkeypatch.setattr(dcp_alltoall, "_try_b12x_dcp_lse_reduce", reduce)
    group = _FakeCPGroup(16, None)  # type: ignore[arg-type]

    dcp_alltoall.warmup_b12x_dcp_a2a(
        group,  # type: ignore[arg-type]
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_batch_size=8192,
        total_heads=96,
        head_dim=512,
        query_head_dim=576,
    )

    assert calls == [
        ("gather", (1, 6, 576)),
        ("reduce", (1, 96, 512)),
    ]


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

    @pytest.mark.skipif(
        torch.accelerator.device_count() < 1, reason="CUDA is required."
    )
    def test_empty_seq_lens_ignore_undefined_output(self):
        from vllm.v1.attention.ops.dcp_alltoall import (
            _dcp_a2a_lse_pack_dim,
            _dcp_a2a_pack_send,
            _dcp_a2a_unpack_combine,
        )

        device = torch.device("cuda")
        world_size, num_tokens, h_per_rank, head_dim = 2, 5, 1, 32
        num_heads = world_size * h_per_rank
        output = torch.randn(
            num_tokens,
            num_heads,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        output[:1] = float("nan")
        lse = torch.randn(num_tokens, num_heads, device=device, dtype=output.dtype)
        seq_lens = torch.tensor([0, 2], device=device, dtype=torch.int32)
        query_start_loc = torch.tensor([0, 1, 5], device=device, dtype=torch.int32)
        lse_pack_dim = _dcp_a2a_lse_pack_dim(output.dtype)
        send_buffer = torch.empty(
            (
                world_size,
                num_tokens,
                h_per_rank,
                head_dim + lse_pack_dim,
            ),
            device=device,
            dtype=output.dtype,
        )

        _dcp_a2a_pack_send(
            output,
            lse,
            send_buffer,
            world_size,
            h_per_rank,
            head_dim,
            lse_pack_dim,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
        )
        actual_output, actual_lse = _dcp_a2a_unpack_combine(
            send_buffer,
            head_dim,
            lse_pack_dim,
            return_lse=True,
            is_lse_base_on_e=True,
        )

        torch.testing.assert_close(
            actual_output[:1], torch.zeros_like(actual_output[:1])
        )
        assert torch.isneginf(actual_lse[:1]).all()
        assert torch.isfinite(actual_output[1:]).all()


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
            dtype=dtype,
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
        ).float()
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
        capture_stream = torch.cuda.Stream(device=f"cuda:{local_rank}")
        with (
            dcp_alltoall.capture_b12x_dcp_a2a(
                group,  # type: ignore[arg-type]
                capture_stream,
                channel_id="test:dcp:graph",
            ),
            torch.cuda.graph(graph, stream=capture_stream),
        ):
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


def _distributed_b12x_packed_query_gather_worker(env: dict[str, str]) -> None:
    """Gather-only warmup of the E4M3 656-byte query record signature, then an
    eager and a graph-captured byte gather through the caller-owned output."""
    update_environment_variables(env)
    local_rank = int(env["LOCAL_RANK"])
    torch.accelerator.set_device_index(local_rank)
    dist.init_process_group(backend="nccl")
    try:
        from vllm.v1.attention.ops import dcp_alltoall

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        batch, h_per_rank, head_dim, record_bytes = 3, 8, 512, 656
        total_heads = world_size * h_per_rank
        group = _FakeCPGroup(world_size, dist.group.WORLD)
        device = torch.device(f"cuda:{local_rank}")

        dcp_alltoall.warmup_b12x_dcp_query_gather(
            group,  # type: ignore[arg-type]
            device=device,
            dtype=torch.float8_e4m3fn,
            max_batch_size=4,
            total_heads=total_heads,
            head_dim=head_dim,
            query_head_dim=record_bytes,
        )
        assert dcp_alltoall._B12X_DCP_A2A_POOLS

        def make_records(step: int) -> torch.Tensor:
            generator = torch.Generator(device=device)
            generator.manual_seed(20000 * step + rank)
            return torch.randint(
                0, 256, (batch, h_per_rank, record_bytes), device=device,
                dtype=torch.uint8, generator=generator,
            )

        def expected(records: torch.Tensor) -> torch.Tensor:
            gathered = [torch.empty_like(records) for _ in range(world_size)]
            dist.all_gather(gathered, records)
            return torch.cat(gathered, dim=1)

        out = torch.empty(
            (batch, total_heads, record_bytes), device=device, dtype=torch.uint8
        )

        def gather(records: torch.Tensor) -> torch.Tensor:
            return dcp_alltoall.dcp_b12x_all_gather_heads(
                records.view(torch.float8_e4m3fn),
                group,  # type: ignore[arg-type]
                max_batch_size=4,
                output_head_dim=head_dim,
                out=out.view(torch.float8_e4m3fn),
            ).view(torch.uint8)

        records = make_records(0)
        gathered = gather(records)
        torch.accelerator.synchronize()
        assert gathered.data_ptr() == out.data_ptr()
        assert torch.equal(gathered, expected(records))

        def fail_query_nccl(*args, **kwargs):
            raise AssertionError("captured path fell back to NCCL all-gather")

        group.all_gather = fail_query_nccl  # type: ignore[attr-defined]
        static_records = make_records(1)
        graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream(device=device)
        with (
            dcp_alltoall.capture_b12x_dcp_a2a(
                group,  # type: ignore[arg-type]
                capture_stream,
                channel_id="test:dcp:graph",
            ),
            torch.cuda.graph(graph, stream=capture_stream),
        ):
            graph_out = gather(static_records)
        for step in range(2, 4):
            static_records.copy_(make_records(step))
            graph.replay()
            torch.accelerator.synchronize()
            assert torch.equal(graph_out, expected(static_records))
    finally:
        from vllm.v1.attention.ops import dcp_alltoall

        for pool in dcp_alltoall._B12X_DCP_A2A_POOLS.values():
            pool.close()
        dcp_alltoall._B12X_DCP_A2A_POOLS.clear()
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2 or importlib.util.find_spec("b12x") is None,
    reason="Need two GPUs and b12x.",
)
def test_distributed_b12x_packed_query_gather_warmup_eager_and_graph():
    _distributed_run(
        _distributed_b12x_packed_query_gather_worker,
        world_size=2,
        extra_env={"VLLM_USE_B12X_DCP_A2A": "1"},
    )


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2 or importlib.util.find_spec("b12x") is None,
    reason="Need two GPUs and b12x.",
)
def test_distributed_b12x_a2a_eager_and_graph_matches_reference():
    _distributed_run(
        _distributed_b12x_a2a_worker,
        world_size=2,
        extra_env={"VLLM_USE_B12X_DCP_A2A": "1"},
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
