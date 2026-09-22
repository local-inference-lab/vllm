# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
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
    _dma_capacity_plan,
    _dma_min_bytes,
    _oneshot_limits,
    _parse_byte_size,
    _twoshot_max_bytes,
    get_b12x_pcie_allreduce,
)
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_tp_group,
    get_world_group,
    graph_capture,
)
from vllm.platforms import current_platform
from vllm.utils.b12x import B12xWorkload, PreparationResourceUnavailableError

from ..utils import (
    get_open_port,
    init_test_distributed_environment,
    multi_gpu_test,
)


def _make_communicator(
    *, allreduce_max_bytes: int = 64, fused_max_bytes: int = 64
) -> tuple[B12xPcieAllReduce, MagicMock]:
    runtime = MagicMock()
    runtime.algorithm = "oneshot"
    runtime.for_stream.return_value.should_allreduce.return_value = True

    communicator = object.__new__(B12xPcieAllReduce)
    communicator.disabled = False
    communicator._runtime = runtime
    communicator._dma = None
    communicator._is_capturing = False
    communicator._capture_stream = None
    communicator.allreduce_max_bytes = allreduce_max_bytes
    communicator.fused_max_bytes = fused_max_bytes
    communicator._twoshot = None
    communicator.twoshot_max_bytes = 0
    plan = object()
    communicator._plans = {"prepared": plan}
    communicator._plan_index = {}
    communicator._routes = {}
    communicator._invocations = {}
    communicator._plan_for = MagicMock(return_value=plan)
    communicator._lookup_plan = MagicMock(return_value=plan)
    return communicator, runtime


def _attach_twoshot(
    communicator: B12xPcieAllReduce, *, max_bytes: int, accepts: bool = True
) -> MagicMock:
    twoshot = MagicMock()
    twoshot.accepts.return_value = accepts
    communicator._twoshot = twoshot
    communicator.twoshot_max_bytes = max_bytes
    return twoshot


def test_hierarchical_communicator_preserves_native_owner_plan():
    communicator, runtime = _make_communicator()
    runtime.algorithm = "hierarchical"
    runtime.all_reduce.side_effect = lambda inp, *, out, **kwargs: out
    inp = torch.zeros(16, dtype=torch.bfloat16)
    actual = communicator.custom_all_reduce(inp)
    assert actual is not inp and actual.shape == inp.shape
    assert runtime.all_reduce.call_args.kwargs["out"] is actual
    assert not communicator.supports_fused_add_rms_norm()
    fused = b12x_pcie_all_reduce.B12xPcieInvocation(
        name="norm",
        operation="all_reduce_fused_add_rms_norm",
        shape=(16,),
        dtype=torch.bfloat16,
        norm_weight=torch.ones(16, dtype=torch.bfloat16),
        epsilon=1e-6,
    )
    assert communicator._route_invocation(fused) is None


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


def test_descriptor_registration_rejects_non_native_communicator() -> None:
    from vllm.distributed.parallel_state import register_b12x_collective_describer

    native_like = MagicMock()
    group = SimpleNamespace(
        device_communicator=SimpleNamespace(b12x_ar_comm=native_like)
    )

    assert not register_b12x_collective_describer(object(), lambda _: (), group=group)
    native_like.register_describer.assert_not_called()


@pytest.mark.parametrize("share_target", [False, True])
def test_embedding_collectives_follow_live_module_ownership(
    monkeypatch: pytest.MonkeyPatch, share_target: bool
) -> None:
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        _register_b12x_embedding_collective,
    )

    communicator, _ = _make_communicator()
    communicator._describers = []
    communicator.global_ranks = (0, 1)
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.register_b12x_collective_describer",
        communicator.register_describer,
    )
    target, draft = torch.nn.Module(), torch.nn.Module()
    target.embed_tokens = torch.nn.Module()
    draft.embed_tokens = torch.nn.Module()
    discarded = weakref.ref(draft.embed_tokens)
    for module in (target, draft):
        _register_b12x_embedding_collective(
            module.embed_tokens, "model.embed_tokens", 128, 2
        )
    workload = SimpleNamespace(
        stage="weights", token_counts=(1, 4), output_dtype=torch.bfloat16
    )
    if not share_target:
        with pytest.raises(ValueError, match="duplicate names"):
            communicator.get_b12x_preparation_units(communicator, workload)
        return

    draft.embed_tokens = target.embed_tokens
    assert discarded() is None
    _register_b12x_embedding_collective(
        target.embed_tokens, "model.embed_tokens", 128, 2
    )
    observed = []
    monkeypatch.setattr(
        communicator, "_route_invocation", lambda item: observed.append(item)
    )
    assert communicator.get_b12x_preparation_units(communicator, workload) == ()
    assert [item.name for item in observed] == [
        "model.embed_tokens.embedding_all_reduce.m1",
        "model.embed_tokens.embedding_all_reduce.m4",
    ]


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


@pytest.fixture
def dma_config(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16, get_hidden_size=lambda: 5120
        ),
        speculative_config=None,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4096),
    )
    monkeypatch.setattr("vllm.config.get_current_vllm_config_or_none", lambda: config)
    return config


@pytest.mark.parametrize("min_bytes", [24 << 20, 48 << 20])
def test_dma_capacity_includes_fp32_without_inflating_bf16(
    dma_config: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, min_bytes: int
) -> None:
    assert _dma_capacity_plan() == {
        torch.bfloat16: 20_971_520,
        torch.float32: 20_971_520,
    }
    monkeypatch.setattr(
        b12x_pcie_all_reduce.envs, "VLLM_PCIE_DMA_MIN_BYTES", str(min_bytes)
    )
    monkeypatch.setattr(b12x_pcie_all_reduce.envs, "VLLM_PCIE_DMA_FP8", False)
    communicator, _ = _make_communicator()
    communicator.device_group = object()
    communicator.device = torch.device("cpu")
    communicator._all_ranks_succeeded = lambda error: error is None
    dma_cls = MagicMock()
    dma_cls.return_value.wire_mode = "bf16"

    communicator._initialize_dma(dma_cls)

    assert dma_cls.call_args.kwargs["max_bytes"] == 80 << 20
    assert communicator._dma is dma_cls.return_value


@pytest.mark.parametrize(
    ("draft_dtype", "draft_hidden", "expected"),
    [
        (
            torch.float16,
            8192,
            {
                torch.bfloat16: 20_971_520,
                torch.float16: 33_554_432,
                torch.float32: 33_554_432,
            },
        ),
        (
            torch.float32,
            2048,
            {torch.bfloat16: 20_971_520, torch.float32: 20_971_520},
        ),
        (
            torch.bfloat16,
            8192,
            {torch.bfloat16: 33_554_432, torch.float32: 33_554_432},
        ),
    ],
)
def test_dma_capacity_merges_target_and_draft(
    dma_config: SimpleNamespace,
    draft_dtype: torch.dtype,
    draft_hidden: int,
    expected: dict[torch.dtype, int],
) -> None:
    dma_config.speculative_config = SimpleNamespace(
        draft_model_config=SimpleNamespace(
            dtype=draft_dtype, get_hidden_size=lambda: draft_hidden
        )
    )

    assert _dma_capacity_plan() == expected


@pytest.mark.parametrize("config", [None, SimpleNamespace(model_config=None)])
def test_dma_without_model_config_stays_optional(
    monkeypatch: pytest.MonkeyPatch, config: SimpleNamespace | None
) -> None:
    monkeypatch.setattr("vllm.config.get_current_vllm_config_or_none", lambda: config)
    monkeypatch.setattr(b12x_pcie_all_reduce.envs, "VLLM_PCIE_DMA_MIN_BYTES", "24MB")
    assert _dma_capacity_plan() is None
    communicator, _ = _make_communicator()
    dma_cls = MagicMock()

    communicator._initialize_dma(dma_cls)

    assert communicator._dma is None
    dma_cls.assert_not_called()


def test_eager_allreduce_dispatches_oneshot() -> None:
    communicator, runtime = _make_communicator()
    inp = torch.randn(2, 4)
    expected = torch.empty_like(inp)
    runtime.all_reduce.return_value = expected

    assert communicator.custom_all_reduce(inp) is expected
    runtime.all_reduce.assert_called_once_with(
        inp, stream=None, plan=communicator._plan_for.return_value
    )


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
    dma.all_reduce.assert_called_once_with(
        inp, plan=communicator._plan_for.return_value
    )


def test_dispatch_rejects_missing_exact_preparation() -> None:
    communicator, _ = _make_communicator()
    communicator._plans = {}
    communicator._invocations = {}
    communicator._lookup_plan.return_value = None

    with pytest.raises(
        PreparationResourceUnavailableError,
        match="no declared plan for",
    ):
        B12xPcieAllReduce._plan_for(communicator, torch.randn(2, 4))


@pytest.mark.parametrize("declared_strides", [None, (8, 1)])
@pytest.mark.parametrize(
    ("shape", "strides", "accepted"),
    [
        ((1, 8), (10, 1), True),
        ((2, 8), (8, 1), True),
        ((2, 8), (10, 1), False),
        ((1, 8), (8, 2), False),
    ],
)
def test_plan_matches_address_layout_not_singleton_stride(
    declared_strides, shape, strides, accepted
) -> None:
    """A trimmed TP output with one row retains its padded row stride."""
    communicator = _indexed_communicator(
        [
            b12x_pcie_all_reduce.B12xPcieInvocation(
                name="prepared",
                operation="all_reduce",
                shape=shape,
                dtype=torch.bfloat16,
                strides=declared_strides,
            )
        ]
    )
    inp = torch.empty_strided(shape, strides, dtype=torch.bfloat16)
    assert communicator._has_plan_for(inp) is accepted
    if accepted:
        assert communicator._plan_for(inp) is communicator._plans["prepared"]


def _indexed_communicator(invocations):
    communicator = object.__new__(B12xPcieAllReduce)
    communicator._invocations = {item.name: item for item in invocations}
    communicator._plans = {item.name: object() for item in invocations}
    communicator._index_declared_plans()
    return communicator


def test_declared_plan_index_preserves_complete_fused_identity():
    operation = "all_reduce_fused_add_rms_norm"
    weight_a = torch.ones(4)
    weight_b = weight_a.clone()
    source = torch.empty((2, 4), dtype=torch.bfloat16)
    invocations = [
        b12x_pcie_all_reduce.B12xPcieInvocation(
            name=name,
            operation=operation,
            shape=(2, 4),
            dtype=source.dtype,
            norm_weight=weight,
            epsilon=epsilon,
        )
        for name, weight, epsilon in (
            ("a", weight_a, 1e-6),
            ("b", weight_b, 1e-6),
            ("epsilon", weight_a, 1e-5),
        )
    ]
    communicator = _indexed_communicator(invocations)
    for invocation in invocations:
        assert (
            communicator._plan_for(
                source,
                operation=operation,
                weight=invocation.norm_weight,
                epsilon=invocation.epsilon,
            )
            is communicator._plans[invocation.name]
        )
    assert not communicator._has_plan_for(source)
    assert not communicator._has_plan_for(
        source, operation=operation, weight=weight_a.clone(), epsilon=1e-6
    )
    assert not communicator._has_plan_for(
        source, operation=operation, weight=weight_a, epsilon=1e-4
    )


def test_declared_plan_index_matches_shape_dtype_stride_and_first_declaration():
    source = torch.empty((2, 4), dtype=torch.bfloat16)
    invocation = b12x_pcie_all_reduce.B12xPcieInvocation
    communicator = _indexed_communicator(
        [
            invocation("first", "all_reduce", (2, 4), source.dtype),
            invocation(
                "equivalent", "all_reduce", (2, 4), source.dtype, strides=(4, 1)
            ),
            invocation("strided", "all_reduce", (2, 4), source.dtype, strides=(8, 1)),
        ]
    )
    assert communicator._plan_for(source) is communicator._plans["first"]
    strided = torch.empty((2, 8), dtype=source.dtype)[:, :4]
    assert communicator._plan_for(strided) is communicator._plans["strided"]
    assert not communicator._has_plan_for(source.float())
    assert not communicator._has_plan_for(source.flatten())
    communicator._invocations = {}
    assert communicator._plan_for(source) is communicator._plans["first"]
    communicator._index_declared_plans()
    assert not communicator._has_plan_for(source)


def test_undeclared_plan_probe_neither_raises_nor_grows_the_index():
    communicator = _indexed_communicator([])
    for rows in range(1, 65):
        source = torch.empty((rows, 4), dtype=torch.bfloat16)
        assert not communicator._has_plan_for(source)
        with pytest.raises(
            PreparationResourceUnavailableError, match="no declared plan"
        ):
            communicator._plan_for(source)
    assert communicator._plan_index == {}


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
        plan=communicator._plan_for.return_value,
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


@pytest.mark.parametrize("raise_inside", [False, True])
def test_capture_passes_a_declared_twoshot_plan_and_restores_state(raise_inside):
    communicator, runtime = _make_communicator()
    twoshot = _attach_twoshot(communicator, max_bytes=1 << 20)
    selected = object()
    communicator._plans.update({"two": selected, "two_other": object()})
    communicator._routes = {
        "prepared": "oneshot",
        "two": "twoshot",
        "two_other": "twoshot",
    }
    entered = []

    @contextmanager
    def capture(*, plan):
        assert plan is selected
        entered.append("enter")
        try:
            yield
        finally:
            entered.append("exit")

    twoshot.capture = capture
    stream = object()
    try:
        with communicator.capture(stream=stream):
            assert entered == ["enter"]
            assert communicator._capture_stream is stream
            assert communicator._is_capturing
            if raise_inside:
                raise ValueError("capture body failed")
    except ValueError as error:
        assert raise_inside and str(error) == "capture body failed"
    assert entered == ["enter", "exit"]
    assert communicator._capture_stream is None
    assert not communicator._is_capturing
    runtime.capture.assert_called_once_with(stream=stream)


def test_capture_does_not_enter_twoshot_without_a_declared_route():
    communicator, _ = _make_communicator()
    twoshot = _attach_twoshot(communicator, max_bytes=1 << 20)
    communicator._routes = {"prepared": "oneshot"}
    with communicator.capture():
        assert communicator._is_capturing
    twoshot.capture.assert_not_called()


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
    from b12x.preparation import PreparationSession

    from vllm.model_executor.warmup.b12x_prepare import b12x_batches
    from vllm.v1.worker.b12x_startup import B12xPreparationCoordinator

    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    config = VllmConfig()
    config.model_config = MagicMock()
    config.model_config.dtype = torch.bfloat16
    config.model_config.get_hidden_size.return_value = 6144
    session = PreparationSession(device=device, autotune=True)
    with ExitStack() as owned, set_current_vllm_config(config):
        init_test_distributed_environment(2, 1, rank, str(port), local_rank=rank)
        owned.callback(destroy_distributed_environment)
        owned.callback(destroy_model_parallel)
        owned.callback(session.close)
        tp_group = get_tp_group()
        communicator = get_b12x_pcie_allreduce()
        assert communicator is not None
        from vllm.model_executor.layers.fused_moe.b12x import (
            _register_b12x_moe_output_collective,
        )

        # Real RoutedExperts expose layer_name, not prefix. Distinct owners
        # remain distinct, while refreshing one owner's describer is idempotent.
        moe_owners = []
        for index, hidden_size in enumerate((2048, 4096)):
            owner = torch.nn.Module()
            owner.layer_name = f"model.layers.{index}.mlp.routed_experts"
            moe_owners.append(owner)
            _register_b12x_moe_output_collective(owner, hidden_size=hidden_size)
            _register_b12x_moe_output_collective(owner, hidden_size=hidden_size)

        epsilon = 1e-6
        weight = torch.linspace(0.5, 1.5, 6144, dtype=torch.bfloat16, device=device)
        alternate_weight = weight.flip(0).contiguous()
        inp = torch.full((4, 6144), rank + 1, dtype=torch.bfloat16, device=device)
        residual = torch.linspace(
            -0.5, 0.5, inp.numel(), dtype=torch.bfloat16, device=device
        ).view_as(inp)
        communicator.register_describer(
            communicator,
            lambda requirements: tuple(
                invocation
                for index in range(40)
                for invocation in (
                    b12x_pcie_all_reduce.B12xPcieInvocation(
                        name=f"test.oneshot.1x6144.layer{index}",
                        operation="all_reduce",
                        shape=(1, 6144),
                        dtype=torch.bfloat16,
                    ),
                    b12x_pcie_all_reduce.B12xPcieInvocation(
                        name=f"test.fused.4x6144.layer{index}",
                        operation="all_reduce_fused_add_rms_norm",
                        shape=(4, 6144),
                        dtype=torch.bfloat16,
                        norm_weight=weight if index % 2 == 0 else alternate_weight,
                        epsilon=epsilon,
                    ),
                    b12x_pcie_all_reduce.B12xPcieInvocation(
                        name=f"test.dma.16x4096.layer{index}",
                        operation="all_reduce",
                        shape=(16, 4096),
                        dtype=torch.bfloat16,
                    ),
                )
            ),
        )
        workload = B12xWorkload(
            stage="weights",
            token_counts=(1, 4, 16),
            fixed_token_counts=(),
            output_dtype=torch.bfloat16,
            max_tokens=16,
            max_seqs=2,
            max_model_len=16,
        )
        units = list(communicator.get_b12x_preparation_units(communicator, workload))
        for prefix in ("test.oneshot.", "test.fused.", "test.dma."):
            plans = [
                plan
                for name, plan in communicator._plans.items()
                if name.startswith(prefix)
            ]
            assert len(plans) == 40
            assert all(plan is plans[0] for plan in plans)
        batches = b12x_batches(units)

        coordinator = B12xPreparationCoordinator(
            session,
            batches,
            global_rank=rank,
            world_group=get_world_group(),
        )
        while True:
            outcome = coordinator.advance(cancel_tuning=rank == 1)
            if outcome["done"]:
                if outcome["error"] is not None:
                    raise RuntimeError(outcome["error"])
                break
        assert session._stop.is_set()
        with session.capture():
            for index, hidden_size in enumerate((2048, 4096)):
                value = torch.full(
                    (16, hidden_size),
                    rank + index + 1,
                    dtype=torch.bfloat16,
                    device=device,
                )
                reduced = tp_group.device_communicator.all_reduce(value)
                torch.testing.assert_close(
                    reduced, torch.full_like(value, 3 + 2 * index)
                )

        oneshot_inp = torch.full(
            (1, 6144), rank + 1, dtype=torch.bfloat16, device=device
        )
        oneshot_eager = tp_group.device_communicator.all_reduce(oneshot_inp)
        torch.testing.assert_close(oneshot_eager, torch.full_like(oneshot_eager, 3))
        with session.capture(), graph_capture(device=device) as capture_context:
            oneshot_graph = torch.cuda.CUDAGraph()
            owned.callback(oneshot_graph.reset)
            with torch.cuda.graph(oneshot_graph, stream=capture_context.stream):
                oneshot_out = tp_group.device_communicator.all_reduce(oneshot_inp)
        oneshot_inp.fill_(rank + 2)
        oneshot_graph.replay()
        torch.accelerator.synchronize()
        torch.testing.assert_close(oneshot_out, torch.full_like(oneshot_out, 5))
        oneshot_inp.fill_(rank + 3)
        oneshot_graph.replay()
        torch.accelerator.synchronize()
        torch.testing.assert_close(oneshot_out, torch.full_like(oneshot_out, 7))

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
        expected_alternate, expected_residual_alternate = _reference_fused_add_rms_norm(
            inp, residual, alternate_weight, tp_group.device_group, epsilon
        )
        assert communicator.try_fused_add_rms_norm(
            inp, residual, alternate_weight, epsilon
        )
        torch.testing.assert_close(inp, expected_alternate, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(residual, expected_residual_alternate)

        inp.copy_(original_inp)
        residual.copy_(original_residual)
        with session.capture(), graph_capture(device=device) as capture_context:
            graph = torch.cuda.CUDAGraph()
            owned.callback(graph.reset)
            with torch.cuda.graph(graph, stream=capture_context.stream):
                torch.ops.vllm.b12x_fused_allreduce_add_rms_norm.default(
                    inp, residual, weight, epsilon
                )
        inp.fill_(rank + 2)
        residual.copy_(original_residual)
        expected, expected_residual = _reference_fused_add_rms_norm(
            inp, residual, weight, tp_group.device_group, epsilon
        )
        graph.replay()
        torch.accelerator.synchronize()
        torch.testing.assert_close(inp, expected, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(residual, expected_residual)
        # This size exceeds the configured plain-oneshot limit and stays below
        # DMA's threshold; the existing PYNCCL fallback is deliberately retained.
        plain_inp = torch.full((4, 6144), rank + 1, dtype=torch.bfloat16, device=device)
        expected_plain = plain_inp.clone()
        dist.all_reduce(expected_plain, group=tp_group.device_group)
        plain_out = tp_group.device_communicator.all_reduce(plain_inp)
        torch.testing.assert_close(plain_out, expected_plain)

        with session.capture(), graph_capture(device=device) as capture_context:
            plain_graph = torch.cuda.CUDAGraph()
            owned.callback(plain_graph.reset)
            with torch.cuda.graph(plain_graph, stream=capture_context.stream):
                plain_out = tp_group.device_communicator.all_reduce(plain_inp)
        plain_inp.fill_(rank + 2)
        expected_plain = plain_inp.clone()
        dist.all_reduce(expected_plain, group=tp_group.device_group)
        plain_graph.replay()
        torch.accelerator.synchronize()
        torch.testing.assert_close(plain_out, expected_plain)

        dma = communicator._dma
        assert dma is not None
        dma_inp = torch.full((16, 4096), rank + 1, dtype=torch.bfloat16, device=device)
        expected_dma = torch.full_like(dma_inp, 3)
        dma_out = tp_group.device_communicator.all_reduce(dma_inp)
        torch.testing.assert_close(dma_out, expected_dma)

        with session.capture(), graph_capture(device=device) as capture_context:
            dma_graph = torch.cuda.CUDAGraph()
            owned.callback(dma_graph.reset)
            with torch.cuda.graph(dma_graph, stream=capture_context.stream):
                dma_out = tp_group.device_communicator.all_reduce(dma_inp)
        dma_inp.fill_(rank + 2)
        expected_dma = torch.full_like(dma_inp, 5)
        dma_graph.replay()
        torch.accelerator.synchronize()
        torch.testing.assert_close(dma_out, expected_dma)

        # ExitStack resets every graph before execution leases and IPC teardown.
        torch.accelerator.synchronize()


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


def _run_b12x_hierarchical_allreduce_gpu(rank: int, port: int) -> None:
    from b12x.preparation import PreparationSession

    from vllm.model_executor.warmup.b12x_prepare import b12x_batches
    from vllm.v1.worker.b12x_startup import B12xPreparationCoordinator

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    config = VllmConfig()
    config.model_config = MagicMock()
    config.model_config.dtype = torch.bfloat16
    config.model_config.get_hidden_size.return_value = 7168
    session = PreparationSession(device=device, autotune=False, compile_workers=1)
    with ExitStack() as owned, set_current_vllm_config(config):
        init_test_distributed_environment(16, 1, rank, str(port), local_rank=rank)
        owned.callback(destroy_distributed_environment)
        owned.callback(destroy_model_parallel)
        owned.callback(session.close)
        communicator = get_tp_group().device_communicator.b12x_ar_comm
        assert isinstance(communicator, B12xPcieAllReduce)
        assert communicator._runtime.algorithm == "hierarchical"
        counts = (1, 2, 4, 16, 257)
        communicator.register_describer(
            communicator,
            lambda workload: tuple(
                b12x_pcie_all_reduce.B12xPcieInvocation(
                    name=f"test.hierarchical.rows{rows}",
                    operation="all_reduce",
                    shape=(rows, 7168),
                    dtype=torch.bfloat16,
                )
                for rows in counts
            ),
        )
        workload = B12xWorkload(
            stage="weights",
            token_counts=counts,
            fixed_token_counts=(),
            output_dtype=torch.bfloat16,
            max_tokens=257,
            max_seqs=4,
            max_model_len=32768,
        )
        units = communicator.get_b12x_preparation_units(communicator, workload)
        coordinator = B12xPreparationCoordinator(
            session,
            b12x_batches(units),
            global_rank=rank,
            world_group=get_world_group(),
        )
        while True:
            outcome = coordinator.advance()
            if outcome["done"]:
                assert outcome["error"] is None, outcome["error"]
                break
        for rows in counts:
            inp = torch.full(
                (rows, 7168), rank + 1, dtype=torch.bfloat16, device=device
            )
            eager = communicator.custom_all_reduce(inp)
            assert eager is not None
            torch.testing.assert_close(eager, torch.full_like(inp, 136), rtol=0, atol=0)
            with session.capture(), graph_capture(device=device) as capture_context:
                graph = torch.cuda.CUDAGraph()
                owned.callback(graph.reset)
                with torch.cuda.graph(graph, stream=capture_context.stream):
                    out = communicator.custom_all_reduce(inp)
            assert out is not None
            for step in (1, 2, -1):
                inp.fill_(rank + 1 + step)
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    out, torch.full_like(inp, 136 + 16 * step), rtol=0, atol=0
                )


@multi_gpu_test(num_gpus=16)
def test_b12x_hierarchical_allreduce_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("b12x.comm.pcie")
    monkeypatch.setenv("VLLM_ENABLE_PCIE_ALLREDUCE", "1")
    monkeypatch.setenv("VLLM_PCIE_ALLREDUCE_BACKEND", "b12x")
    monkeypatch.setenv("VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE", "6MiB")
    monkeypatch.setenv("B12X_PCIE_ALLREDUCE_ALGORITHM", "island_rs")
    monkeypatch.setenv("B12X_PCIE_HIERARCHICAL_DEFERRED_CONSUMPTION", "1")
    monkeypatch.setenv("B12X_PCIE_HIERARCHICAL_BF16X2", "1")
    monkeypatch.setenv("B12X_PCIE_HIERARCHICAL_BF16X2_MAX_ELEMENTS", "7168")
    torch.multiprocessing.spawn(
        _run_b12x_hierarchical_allreduce_gpu,
        args=(get_open_port(),),
        nprocs=16,
        join=True,
    )


def _run_b12x_dcp_gpu(rank: int, port: int, world_size: int) -> None:
    from b12x.preparation import PreparationSession

    from vllm.distributed.device_communicators.b12x_dcp import (
        get_b12x_dcp_transport,
        get_b12x_kimi_projection_transport,
    )
    from vllm.model_executor.warmup.b12x_prepare import b12x_batches
    from vllm.v1.worker.b12x_startup import B12xPreparationCoordinator

    device = torch.device(f"cuda:{rank}")
    heads_per_rank = (96 + world_size - 1) // world_size
    total_heads = heads_per_rank * world_size
    torch.cuda.set_device(device)
    config = VllmConfig()
    config.model_config = MagicMock()
    config.model_config.dtype = torch.bfloat16
    config.model_config.get_hidden_size.return_value = 7168
    session = PreparationSession(device=device, autotune=False, compile_workers=1)
    with ExitStack() as owned, set_current_vllm_config(config):
        init_test_distributed_environment(
            world_size, 1, rank, str(port), local_rank=rank
        )
        owned.callback(destroy_distributed_environment)
        owned.callback(destroy_model_parallel)
        owned.callback(session.close)
        transports = [
            get_b12x_dcp_transport(
                get_tp_group(),
                device,
                8,
                heads_per_rank,
                576,
                512,
                dtype,
                torch.bfloat16,
            )
            for dtype in (torch.bfloat16, torch.float8_e4m3fn)
        ]
        assert all(transport is not None for transport in transports)
        projection = get_b12x_kimi_projection_transport(get_tp_group(), device)
        latent = get_b12x_kimi_projection_transport(
            get_tp_group(), device, latent_width=136
        )
        if world_size in (9, 10, 12):
            assert projection is not None and projection.returns_logits
            assert latent is None
        else:
            assert projection is not None and latent is not None
        workload = B12xWorkload(
            stage="weights",
            token_counts=(1, 2, 4, 8),
            fixed_token_counts=(),
            output_dtype=torch.bfloat16,
            max_tokens=8,
            max_seqs=8,
            max_model_len=32768,
        )
        units = tuple(
            unit
            for transport in [*transports, projection, latent]
            if transport is not None
            for unit in transport.get_b12x_preparation_units(transport, workload)
        )
        coordinator = B12xPreparationCoordinator(
            session,
            b12x_batches(units),
            global_rank=rank,
            world_group=get_world_group(),
        )
        while True:
            outcome = coordinator.advance()
            if outcome["done"]:
                assert outcome["error"] is None, outcome["error"]
                break
        session.freeze()
        records = []
        for transport in transports:
            for rows in (1, 2, 4, 8):
                base_e = rows % 3 != 2
                query = torch.empty(
                    rows,
                    transport.num_heads,
                    576,
                    dtype=transport.query_dtype,
                    device=device,
                )
                # MLA may supply head-major output with a fixed token capacity.
                partial = torch.empty(
                    total_heads,
                    8,
                    512,
                    dtype=torch.bfloat16,
                    device=device,
                ).transpose(0, 1)[:rows]
                lse = torch.empty(rows, total_heads, device=device)
                with session.capture(), graph_capture(device=device) as capture_context:
                    graph = torch.cuda.CUDAGraph()
                    owned.callback(graph.reset)
                    with torch.cuda.graph(graph, stream=capture_context.stream):
                        gathered = transport.gather(query)
                        combined = transport.combine(
                            partial, lse, is_lse_base_on_e=base_e
                        )
                records.append(
                    (
                        transport,
                        rows,
                        base_e,
                        query,
                        partial,
                        lse,
                        graph,
                        gathered,
                        combined,
                    )
                )
        for step in (1, 2, -1, 3):
            # Alternate capture sizes and eager execution using changed inputs.
            for (
                transport,
                rows,
                base_e,
                query,
                partial,
                lse,
                graph,
                gathered,
                combined,
            ) in reversed(records):
                gen = torch.Generator().manual_seed(1931 + rows + step)
                all_queries = torch.randn(
                    rows,
                    total_heads,
                    576,
                    generator=gen,
                ).to(device=device, dtype=query.dtype)
                all_outputs = torch.randn(
                    world_size,
                    rows,
                    total_heads,
                    512,
                    generator=gen,
                ).to(device=device, dtype=torch.bfloat16)
                all_lses = torch.randn(world_size, rows, total_heads, generator=gen).to(
                    device
                )
                all_lses[:, 0, 0] = -torch.inf
                all_lses[0, :, 1] = -torch.inf
                query.copy_(all_queries.chunk(world_size, dim=1)[rank])
                partial.copy_(all_outputs[rank])
                lse.copy_(all_lses[rank])
                natural_lses = all_lses if base_e else all_lses * 0.6931471805599453
                weights = torch.softmax(natural_lses, dim=0).nan_to_num()
                expected = (all_outputs.float() * weights[..., None]).sum(0)
                expected = expected.chunk(world_size, dim=1)[rank].to(torch.bfloat16)
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    gathered.float(), all_queries.float(), rtol=0, atol=0
                )
                torch.testing.assert_close(combined, expected, rtol=0.01, atol=0.01)
                assert torch.isfinite(combined).all() and combined.abs().sum() > 0
                eager_query = transport.gather(query)
                eager_output = transport.combine(partial, lse, is_lse_base_on_e=base_e)
                torch.testing.assert_close(
                    eager_query.float(), all_queries.float(), rtol=0, atol=0
                )
                torch.testing.assert_close(eager_output, expected, rtol=0.01, atol=0.01)

        if projection.returns_logits:
            for rows in (1, 3, 8):
                down = torch.zeros(
                    rows, projection.down_width, device=device, dtype=torch.bfloat16
                )
                logits = torch.zeros(rows, projection.router_width, device=device)
                with session.capture(), graph_capture(device=device) as capture_context:
                    graph = torch.cuda.CUDAGraph()
                    owned.callback(graph.reset)
                    with torch.cuda.graph(graph, stream=capture_context.stream):
                        output, payload = projection.gather_projections(
                            down, logits, None
                        )
                for step in range(5):
                    gen = torch.Generator().manual_seed(4811 + rows + step)
                    all_down = torch.randn(
                        rows, world_size * projection.down_width, generator=gen
                    ).to(down)
                    all_logits = torch.randn(
                        rows, world_size * projection.router_width, generator=gen
                    ).to(logits)
                    down.copy_(all_down.chunk(world_size, dim=-1)[rank])
                    logits.copy_(all_logits.chunk(world_size, dim=-1)[rank])
                    output.fill_(torch.nan)
                    payload.fill_(torch.nan)
                    allocated = torch.cuda.memory_allocated()
                    graph.replay()
                    torch.cuda.synchronize()
                    assert torch.cuda.memory_allocated() == allocated
                    eager = projection.gather_projections(down, logits, None)
                    for actual_down, actual_logits in ((output, payload), eager):
                        torch.testing.assert_close(
                            actual_down, all_down[:, :3584], rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            actual_logits, all_logits[:, :896], rtol=0, atol=0
                        )
                        assert actual_down.is_contiguous()
                        assert actual_logits.is_contiguous()
            return

        projection_graphs = []
        for rows in (1, 2, 8):
            down = torch.empty(
                rows, 3584 // world_size, device=device, dtype=torch.bfloat16
            )
            logits = torch.empty(rows, 896 // world_size, device=device)
            bias = torch.zeros(896, device=device)
            latent_input = torch.empty(
                rows, 1, 136, device=device, dtype=torch.bfloat16
            )
            with session.capture(), graph_capture(device=device) as capture_context:
                graph = torch.cuda.CUDAGraph()
                owned.callback(graph.reset)
                with torch.cuda.graph(graph, stream=capture_context.stream):
                    output, payload = projection.gather_projections(down, logits, bias)
                    gathered_latent = latent.gather(latent_input)
            projection_graphs.append(
                (
                    rows,
                    graph,
                    down,
                    logits,
                    bias,
                    latent_input,
                    output,
                    payload,
                    gathered_latent,
                )
            )
        for step in range(5):
            for (
                rows,
                graph,
                down,
                logits,
                bias,
                latent_input,
                output,
                payload,
                gathered_latent,
            ) in reversed(projection_graphs):
                gen = torch.Generator().manual_seed(4811 + rows + step)
                all_down = torch.randn(rows, 3584, generator=gen).to(down)
                all_logits = torch.randn(rows, 896, generator=gen).to(logits)
                correction = torch.randn(896, generator=gen).to(bias) * 0.1
                all_latents = torch.randn(rows, world_size, 136, generator=gen).to(down)
                down.copy_(all_down.chunk(world_size, dim=-1)[rank])
                logits.copy_(all_logits.chunk(world_size, dim=-1)[rank])
                bias.copy_(correction)
                latent_input.copy_(all_latents[:, rank : rank + 1])
                scores = all_logits.sigmoid()
                ids = (scores + correction).topk(16, dim=-1).indices
                weights = scores.gather(-1, ids)
                weights /= weights.sum(-1, keepdim=True)
                graph.replay()
                torch.cuda.synchronize()
                eager_output, eager_payload = projection.gather_projections(
                    down, logits, bias
                )
                for actual_down, actual_payload in (
                    (output, payload),
                    (eager_output, eager_payload),
                ):
                    torch.testing.assert_close(actual_down, all_down, rtol=0, atol=0)
                    torch.testing.assert_close(
                        actual_payload[rows:].view(torch.int32),
                        ids.int(),
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        actual_payload[:rows], weights, rtol=2e-6, atol=1e-7
                    )
                    assert torch.isfinite(actual_payload[:rows]).all()
                    assert (actual_payload[:rows] > 0).all()
                torch.testing.assert_close(gathered_latent, all_latents, rtol=0, atol=0)


@pytest.mark.parametrize("world_size", [2, 9, 10, 12, 16])
def test_b12x_dcp_prepared_eager_graph_interleaving(monkeypatch, world_size):
    """Fixed plans preserve rank heads and LSE weighting across live row counts."""
    pytest.importorskip("b12x.comm.pcie")
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} GPUs")
    monkeypatch.setenv("VLLM_ENABLE_PCIE_ALLREDUCE", "0")
    torch.multiprocessing.spawn(
        _run_b12x_dcp_gpu,
        args=(get_open_port(), world_size),
        nprocs=world_size,
        join=True,
    )


def test_twoshot_limit_defaults_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE", raising=False)
    assert _twoshot_max_bytes() == 0
    monkeypatch.setattr(
        b12x_pcie_all_reduce.envs,
        "VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE",
        "768KB",
    )
    assert _twoshot_max_bytes() == 768 << 10
    for disabled in ("0", "off", " NONE ", "", "disabled"):
        monkeypatch.setattr(
            b12x_pcie_all_reduce.envs,
            "VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE",
            disabled,
        )
        assert _twoshot_max_bytes() == 0


def test_midsize_allreduce_dispatches_twoshot() -> None:
    communicator, runtime = _make_communicator(allreduce_max_bytes=16)
    twoshot = _attach_twoshot(communicator, max_bytes=1 << 20)
    twoshot.all_reduce.side_effect = lambda _, *, out, plan: out
    inp = torch.randn(64)  # 256 bytes: above the one-shot ceiling

    assert communicator.should_custom_ar(inp)
    out = communicator.custom_all_reduce(inp)
    assert out is not None and out.shape == inp.shape and out is not inp
    twoshot.all_reduce.assert_called_once_with(
        inp, out=out, plan=communicator._plan_for.return_value
    )
    runtime.all_reduce.assert_not_called()


def test_oneshot_keeps_priority_below_its_ceiling() -> None:
    communicator, runtime = _make_communicator(allreduce_max_bytes=1024)
    twoshot = _attach_twoshot(communicator, max_bytes=1 << 20)
    expected = torch.empty(64)
    runtime.all_reduce.return_value = expected
    inp = torch.randn(64)

    assert communicator.custom_all_reduce(inp) is expected
    runtime.all_reduce.assert_called_once_with(
        inp, stream=None, plan=communicator._plan_for.return_value
    )
    twoshot.all_reduce.assert_not_called()


def test_twoshot_window_ends_at_its_limit() -> None:
    communicator, runtime = _make_communicator(allreduce_max_bytes=16)
    twoshot = _attach_twoshot(communicator, max_bytes=128)
    dma = MagicMock()
    dma.should_allreduce.return_value = True
    expected = torch.empty(64)
    dma.all_reduce.return_value = expected
    communicator._dma = dma
    inp = torch.randn(64)  # 256 bytes: above the two-shot window

    assert communicator.custom_all_reduce(inp) is expected
    twoshot.all_reduce.assert_not_called()
    dma.all_reduce.assert_called_once_with(
        inp, plan=communicator._plan_for.return_value
    )


def test_twoshot_respects_runtime_acceptance() -> None:
    communicator, _ = _make_communicator(allreduce_max_bytes=16)
    twoshot = _attach_twoshot(communicator, max_bytes=1 << 20, accepts=False)
    inp = torch.randn(64)

    assert not communicator.should_custom_ar(inp)
    assert communicator.custom_all_reduce(inp) is None
    twoshot.all_reduce.assert_not_called()


@pytest.mark.parametrize(("tokens", "expected"), [(12, None), (16, "twoshot")])
def test_twoshot_metadata_requires_complete_rank_shards(tokens, expected):
    communicator, _ = _make_communicator(allreduce_max_bytes=96 << 10)
    communicator.world_size = 4
    twoshot = _attach_twoshot(communicator, max_bytes=768 << 10)
    twoshot.row_elems = 4096
    invocation = b12x_pcie_all_reduce.B12xPcieInvocation(
        name="embedding.all_reduce",
        operation="all_reduce",
        shape=(tokens, 5120),
        dtype=torch.bfloat16,
    )
    assert communicator._route_invocation(invocation) == expected


def test_graph_capture_supplies_caller_owned_twoshot_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    communicator, runtime = _make_communicator(allreduce_max_bytes=16)
    twoshot = _attach_twoshot(communicator, max_bytes=1 << 20)
    communicator._is_capturing = True
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    inp = torch.randn(64)
    twoshot.all_reduce.side_effect = lambda _, *, out, plan: out

    out = communicator.custom_all_reduce(inp)

    assert out is not None and out.shape == inp.shape and out is not inp
    twoshot.all_reduce.assert_called_once_with(
        inp, out=out, plan=communicator._plan_for.return_value
    )
    runtime.all_reduce.assert_not_called()


def _run_b12x_twoshot_gpu(
    rank: int, port: int, device_indices: tuple[int, ...]
) -> None:
    from b12x.preparation import PreparationSession

    from vllm.model_executor.warmup.b12x_prepare import b12x_batches
    from vllm.v1.worker.b12x_startup import B12xPreparationCoordinator

    device_index = device_indices[rank]
    device = torch.device(f"cuda:{device_index}")
    torch.accelerator.set_device_index(device)
    config = VllmConfig()
    config.model_config = MagicMock()
    config.model_config.dtype = torch.bfloat16
    config.model_config.get_hidden_size.return_value = 4096
    with ExitStack() as owned, set_current_vllm_config(config):
        init_test_distributed_environment(
            4, 1, rank, str(port), local_rank=device_index
        )
        owned.callback(destroy_distributed_environment)
        owned.callback(destroy_model_parallel)
        session = PreparationSession(device=device, autotune=False)
        owned.callback(session.close)
        tp_group = get_tp_group()
        communicator = tp_group.device_communicator.b12x_ar_comm
        assert communicator is not None and communicator._twoshot is not None
        communicator.register_describer(
            communicator,
            lambda requirements: tuple(
                b12x_pcie_all_reduce.B12xPcieInvocation(
                    name=f"test.twoshot.64x4096.layer{index}",
                    operation="all_reduce",
                    shape=(64, 4096),
                    dtype=torch.bfloat16,
                )
                for index in range(40)
            ),
        )
        workload = B12xWorkload(
            stage="weights",
            token_counts=(64,),
            fixed_token_counts=(),
            output_dtype=torch.bfloat16,
            max_tokens=64,
            max_seqs=2,
            max_model_len=64,
        )
        units = communicator.get_b12x_preparation_units(communicator, workload)
        plans = tuple(communicator._plans.values())
        assert len(plans) == 40 and all(plan is plans[0] for plan in plans)
        coordinator = B12xPreparationCoordinator(
            session,
            b12x_batches(units),
            global_rank=rank,
            world_group=get_world_group(),
        )
        while True:
            outcome = coordinator.advance()
            if outcome["done"]:
                if outcome["error"] is not None:
                    raise RuntimeError(outcome["error"])
                break

        inp = torch.full((64, 4096), rank + 1, dtype=torch.bfloat16, device=device)
        assert communicator.should_custom_ar(inp)
        eager_out = tp_group.device_communicator.all_reduce(inp)
        torch.testing.assert_close(eager_out, torch.full_like(inp, 10))

        with session.capture(), graph_capture(device=device) as capture_context:
            graph = torch.cuda.CUDAGraph()
            owned.callback(graph.reset)
            with torch.cuda.graph(graph, stream=capture_context.stream):
                graph_out = tp_group.device_communicator.all_reduce(inp)
        graph_out_ptr = graph_out.data_ptr()
        inp.fill_(rank + 2)
        graph.replay()
        torch.accelerator.synchronize()
        assert graph_out.data_ptr() == graph_out_ptr
        torch.testing.assert_close(graph_out, torch.full_like(inp, 14))
        inp.fill_(rank + 3)
        graph.replay()
        torch.accelerator.synchronize()
        torch.testing.assert_close(graph_out, torch.full_like(inp, 18))


@multi_gpu_test(num_gpus=4)
@pytest.mark.skipif(
    not current_platform.has_device_capability(120),
    reason="B12X PCIe all-reduce requires SM120",
)
def test_b12x_twoshot_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("b12x.comm.pcie")
    monkeypatch.setenv("VLLM_ENABLE_PCIE_ALLREDUCE", "1")
    monkeypatch.setenv("VLLM_PCIE_ALLREDUCE_BACKEND", "b12x")
    monkeypatch.setenv("VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE", "16KB")
    monkeypatch.setenv("VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE", "768KB")
    monkeypatch.setenv("VLLM_PCIE_DMA_MIN_BYTES", "off")
    torch.multiprocessing.spawn(
        _run_b12x_twoshot_gpu,
        args=(get_open_port(), tuple(range(4))),
        nprocs=4,
        join=True,
    )
