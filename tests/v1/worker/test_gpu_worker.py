# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import patch
from weakref import ref

import pytest
from torch import nn

from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.worker import gpu_worker, startup_plan
from vllm.v1.worker.gpu_worker import maybe_rocm_profiling_fallback
from vllm.v1.worker.startup_plan import (
    maybe_apply_startup_plan,
    maybe_save_startup_plan,
)


def test_mark_b12x_eager_shapes_covers_encoder_and_connector_profile_shapes(
    monkeypatch,
) -> None:
    import vllm.multimodal.encoder_budget as encoder_budget
    from vllm.model_executor.warmup.b12x_prepare import mark_b12x_eager_shapes

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = nn.Module()
            self.visual.block = nn.Module()
            self.visual.merger = nn.Sequential(nn.Module())
            self.visual.deepstack_merger_list = nn.ModuleList([nn.Module()])

        def get_mm_lora_token_counts(self, *, modality, mm_kwargs, num_mm_embeds):
            assert modality == "image"
            assert mm_kwargs is None
            return num_mm_embeds * 4, num_mm_embeds

        def get_mm_mapping(self):
            return SimpleNamespace(
                connector=("visual.merger", "visual.deepstack_merger_list")
            )

    class _FakeBudget:
        def __init__(self, vllm_config, mm_registry, enable_cache):
            del vllm_config, mm_registry, enable_cache

        def get_encoder_budget(self):
            return 16_384

    monkeypatch.setattr(encoder_budget, "MultiModalBudget", _FakeBudget)
    model = _Model()
    worker = SimpleNamespace(
        get_model=lambda: model,
        vllm_config=SimpleNamespace(),
        model_runner=SimpleNamespace(mm_registry=object()),
    )

    mark_b12x_eager_shapes(worker)

    assert model.visual.block.b12x_eager_token_counts == (65_536,)
    assert model.visual.block.b12x_eager_only is True
    for connector in (model.visual.merger, model.visual.deepstack_merger_list):
        assert connector.b12x_eager_token_counts == (16_384,)
        assert all(module.b12x_eager_only for module in connector.modules())


def test_b12x_workload_covers_target_and_draft_profile_shapes() -> None:
    from vllm.model_executor.warmup.b12x_prepare import b12x_workload

    compilation = SimpleNamespace(
        cudagraph_capture_sizes=(1, 2, 4, 8),
        compile_sizes=(),
        get_compile_ranges=lambda: (SimpleNamespace(end=128),),
    )
    worker = SimpleNamespace(
        get_model=lambda: nn.Module(),
        model_runner=SimpleNamespace(mm_registry=None),
        vllm_config=SimpleNamespace(
            compilation_config=compilation,
            speculative_config=SimpleNamespace(num_speculative_tokens=3),
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=128,
            max_num_seqs=1,
        ),
        model_config=SimpleNamespace(dtype="bf16", max_model_len=4096),
    )

    workload = b12x_workload(worker, stage="weights")

    # b12x_preparation_token_counts also reserves the post-speculative decode
    # regime (max_tokens - speculative_tokens = 128 - 3 = 125).
    assert workload.token_counts == (1, 2, 4, 8, 125, 128)
    assert workload.fixed_token_counts == (1, 2, 4, 8)
    assert workload.max_tokens == 128
    assert workload.speculative_tokens == 3


# Startup-plan persistence (vllm/v1/worker/startup_plan.py), applied and
# saved by Worker.determine_available_memory / compile_or_warm_up_model.


def _plan_worker(config_hash="abc123", free_memory=78 * GiB_bytes, kv_bytes=None):
    """The minimal Worker surface the startup-plan entry points touch."""
    return SimpleNamespace(
        vllm_config=SimpleNamespace(compute_hash=lambda: config_hash),
        rank=0,
        parallel_config=SimpleNamespace(world_size=1),
        init_snapshot=SimpleNamespace(free_memory=free_memory),
        cache_config=SimpleNamespace(kv_cache_memory_bytes=kv_bytes),
    )


def _plan_platform(name="NVIDIA H100 PCIe"):
    return SimpleNamespace(
        get_device_name=lambda device_id=0: name,
        get_device_total_memory=lambda device_id=0: 80 * GiB_bytes,
        get_device_capability=lambda device_id=0: (9, 0),
    )


@pytest.fixture
def plan_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Enable the startup plan, isolated under a tmp cache root."""
    monkeypatch.setenv("VLLM_ENABLE_STARTUP_PLAN", "1")
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    with patch.object(startup_plan, "current_platform", _plan_platform()):
        yield


def test_startup_plan_fingerprint_sensitivity(plan_env):
    """The fingerprint is the OOM-safety key: stable for identical inputs,
    different for anything the profiled value depends on."""
    fp = startup_plan.compute_plan_fingerprint
    base = fp(_plan_worker().vllm_config, 0, 1)
    assert base == fp(_plan_worker().vllm_config, 0, 1)
    assert base != fp(_plan_worker("other").vllm_config, 0, 1)
    assert base != fp(_plan_worker().vllm_config, 1, 2)
    with patch.object(startup_plan, "current_platform", _plan_platform("NVIDIA A100")):
        assert base != fp(_plan_worker().vllm_config, 0, 1)
    with patch("vllm.__version__", "0.0.0+plan-test"):
        assert base != fp(_plan_worker().vllm_config, 0, 1)


def test_startup_plan_apply_gate(plan_env):
    """Only a fingerprint-matching, memory-safe plan is ever applied."""
    maybe_save_startup_plan(_plan_worker(), 50 * GiB_bytes)

    applied = _plan_worker()
    maybe_apply_startup_plan(applied)
    assert applied.cache_config.kv_cache_memory_bytes == 50 * GiB_bytes

    less_memory = _plan_worker(free_memory=60 * GiB_bytes)
    other_config = _plan_worker(config_hash="zzz999")
    for refused in (less_memory, other_config):
        maybe_apply_startup_plan(refused)
        assert refused.cache_config.kv_cache_memory_bytes is None

    # An explicit --kv-cache-memory is never overridden.
    explicit = _plan_worker(kv_bytes=7 * GiB_bytes)
    maybe_apply_startup_plan(explicit)
    assert explicit.cache_config.kv_cache_memory_bytes == 7 * GiB_bytes


# Memory accounting of the profiling run (Worker.determine_available_memory).


@pytest.mark.parametrize("failure", [False, True])
def test_profile_release_collects_cycles_before_flushing_allocator(
    monkeypatch, failure
):
    """The allocator flush must follow plan release and Python cycle collection."""

    class _Batch:
        def release(self):
            self.resource = None
            if failure:
                raise RuntimeError("release failed")

    class _Temporary:
        def __init__(self):
            self.cycle = self

    batch = _Batch()
    batch.resource = _Temporary()
    resource_ref = ref(batch.resource)
    worker = SimpleNamespace(_b12x_profile_batch=batch)
    del batch
    events = []

    def empty_cache():
        assert resource_ref() is None
        assert worker._b12x_profile_batch is None
        events.append("empty_cache")

    monkeypatch.setattr(
        gpu_worker.torch.accelerator, "synchronize", lambda: events.append("sync")
    )
    monkeypatch.setattr(gpu_worker.torch.accelerator, "empty_cache", empty_cache)
    if failure:
        with pytest.raises(RuntimeError, match="release failed"):
            gpu_worker.Worker._release_b12x_profile_state(worker)
    else:
        gpu_worker.Worker._release_b12x_profile_state(worker)
    assert events == ["sync", "empty_cache"]


def test_serving_kv_allocation_collects_temporary_cycles(monkeypatch):
    """No profiling garbage or freed allocator blocks survive into KV allocation."""

    class _Temporary:
        def __init__(self):
            self.cycle = self

    temporary = _Temporary()
    temporary_ref = ref(temporary)
    del temporary
    events: list[str] = []

    def allocate(*args, **kwargs):
        assert temporary_ref() is None
        assert events == ["sync", "empty_cache"]
        events.append("allocate")

    worker = SimpleNamespace(
        cache_config=SimpleNamespace(),
        vllm_config=object(),
        model_config=SimpleNamespace(enable_return_routed_experts=False),
        model_runner=SimpleNamespace(initialize_kv_cache=allocate),
        _maybe_get_memory_pool_context=lambda **kw: nullcontext(),
    )
    monkeypatch.setattr(gpu_worker, "set_current_vllm_config", lambda _: nullcontext())
    monkeypatch.setattr(
        gpu_worker, "ensure_kv_transfer_initialized", lambda *args: None
    )
    monkeypatch.setattr(
        gpu_worker.torch.accelerator, "synchronize", lambda: events.append("sync")
    )
    monkeypatch.setattr(
        gpu_worker.torch.accelerator,
        "empty_cache",
        lambda: events.append("empty_cache"),
    )

    gpu_worker.Worker.initialize_from_config(
        worker,
        SimpleNamespace(
            num_blocks=32, kv_cache_layout=None, needs_kv_cache_zeroing=False
        ),
    )
    assert events[-1] == "allocate"


# The fallback reads only the sign of the measured drop and this process's torch
# reservation; free memory is only logged, so no amount here is a device size.
ANY_FREE_MEMORY = 8 * GiB_bytes
MEASURED_DROP = 4 * GiB_bytes
TORCH_RESERVED = 3 * GiB_bytes
RELEASED_BY_OTHERS = 2 * GiB_bytes


def _snapshot(free_memory, torch_memory=0):
    return SimpleNamespace(free_memory=free_memory, torch_memory=torch_memory)


def _profile_result(consumed, reserved_before=0, reserved_after=0):
    """A result whose free-memory readings agree with `consumed`, which
    `memory_profiling` derives as the drop in free memory, negative when it grew."""
    return SimpleNamespace(
        total_consumed=consumed,
        transient_peak_headroom=0,
        before_create=_snapshot(ANY_FREE_MEMORY, reserved_before),
        after_profile=_snapshot(ANY_FREE_MEMORY - consumed, reserved_after),
    )


@pytest.fixture
def rocm(request):
    with patch.object(
        gpu_worker, "current_platform", SimpleNamespace(is_rocm=lambda: request.param)
    ):
        yield request.param


@pytest.mark.parametrize("rocm", [True, False], indirect=True)
def test_profiling_fallback_declines_when_free_memory_dropped(rocm):
    """The profiling measurement is kept as-is whenever free memory dropped."""
    result = _profile_result(consumed=MEASURED_DROP)

    assert maybe_rocm_profiling_fallback(result) is None


@pytest.mark.parametrize("rocm", [True], indirect=True)
def test_profiling_fallback_replaces_a_released_measurement(rocm):
    """A negative measurement describes the rest of the device, so it is replaced
    by this process's reservation, which the rest of the device cannot move."""
    result = _profile_result(
        consumed=-RELEASED_BY_OTHERS,
        reserved_after=TORCH_RESERVED,
    )

    assert maybe_rocm_profiling_fallback(result) == TORCH_RESERVED


@pytest.mark.parametrize("rocm", [True], indirect=True)
def test_profiling_fallback_never_returns_a_negative_amount(rocm):
    """A reservation that shrank across the run cannot become negative usage."""
    result = _profile_result(
        consumed=-RELEASED_BY_OTHERS,
        reserved_before=TORCH_RESERVED,
        reserved_after=0,
    )

    assert maybe_rocm_profiling_fallback(result) == 0


@pytest.mark.parametrize("rocm", [False], indirect=True)
def test_profiling_fallback_declines_off_rocm(rocm):
    """Platforms that account frees eagerly keep reporting the error, so the
    caller's assertion stays reachable there."""
    result = _profile_result(consumed=-RELEASED_BY_OTHERS)

    assert maybe_rocm_profiling_fallback(result) is None


class _OrderedHandle:
    """Send handle that logs when it is waited."""

    def __init__(self, log: list[str], name: str):
        self.log = log
        self.name = name

    def is_completed(self) -> bool:
        return True

    def wait(self) -> None:
        self.log.append(f"wait:{self.name}")


def test_execute_model_waits_previous_pp_send_before_forward(
    monkeypatch: pytest.MonkeyPatch,
):
    """Previous device handles are waited before the forward pass; the
    metadata handle is left to the GroupCoordinator's reaper."""
    import torch

    from vllm.sequence import IntermediateTensors

    log: list[str] = []
    previous_tensor_send = _OrderedHandle(log, "prev-tensor")
    metadata_handle = _OrderedHandle(log, "meta")
    tensor_handle = _OrderedHandle(log, "tensor")

    def isend_tensor_dict(tensors, all_gather_group=None, all_gather_tensors=None):
        log.append("isend")
        return [metadata_handle, tensor_handle]

    pp_group = SimpleNamespace(
        is_first_rank=True,
        is_last_rank=False,
        isend_tensor_dict=isend_tensor_dict,
    )
    monkeypatch.setattr(gpu_worker, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(gpu_worker, "get_tp_group", lambda: SimpleNamespace())

    def run_model(scheduler_output, intermediate_tensors):
        log.append("forward")
        return IntermediateTensors({"hidden_states": torch.zeros(1)})

    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(
                pass_config=SimpleNamespace(enable_sp=False)
            ),
            parallel_config=SimpleNamespace(
                pipeline_parallel_size=2, distributed_executor_backend="mp"
            ),
        ),
        use_v2_model_runner=False,
        model_runner=SimpleNamespace(execute_model=run_model),
        annotate_profile=lambda scheduler_output: nullcontext(),
        _pp_send_work=[previous_tensor_send],
    )
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=4, num_scheduled_tokens={"r0": 4}
    )

    assert gpu_worker.Worker.execute_model(worker, scheduler_output) is None

    assert log == ["wait:prev-tensor", "forward", "isend"]
    assert worker._pp_send_work == [tensor_handle]


@pytest.mark.parametrize(
    "final_free_memory,expected_available_memory",
    [(90, 75), (85, 70)],
)
@pytest.mark.parametrize("graph_estimate", [0, 4])
@pytest.mark.parametrize("estimate_graphs", [False, True])
@pytest.mark.parametrize(
    "native_profile,final_native,overlap",
    [
        (None, 0, 0),  # Runners without native capture measurements retain the budget.
        ((100, 103, 4), 103, 3),  # Persistent capture initialization.
        ((100, 103, 4), 101, 1),  # Prepared-plan release frees part of the growth.
        ((100, 103, 4), 100, 0),  # All native capture storage was temporary.
        ((103, 103, 4), 107, 0),  # Bootstrap and cleanup growth are not capture cost.
        ((100, 103, 1), 107, 1),  # The measured capture delta bounds the overlap.
        ((100, 103, 4), 107, 3),  # Cleanup-only growth cannot increase the discount.
    ],
)
def test_cudagraph_memory_profile_prepares_and_releases_b12x_state(
    monkeypatch,
    final_free_memory,
    expected_available_memory,
    graph_estimate,
    estimate_graphs,
    native_profile,
    final_native,
    overlap,
):
    """KV admission counts retained allocations, not released profiling blocks."""
    events: list[object] = []

    def profile_cudagraph_memory(prepare_profile_state):
        prepare_profile_state()
        events.append("profile_cudagraph_memory")
        return graph_estimate

    def profile_run(prepare_profile_state):
        prepare_profile_state()
        events.append("profile_run")

    def profile_glm_dcp_attention(prepare_profile_state):
        prepare_profile_state()
        events.append("profile_glm_dcp_attention")

    model_runner = SimpleNamespace(
        cudagraph_native_memory_profile=native_profile,
        model_memory_usage=0,
        profile_run=profile_run,
        profile_glm_dcp_attention=profile_glm_dcp_attention,
        profile_cudagraph_memory=profile_cudagraph_memory,
    )
    profile_result = SimpleNamespace(
        total_consumed=10,
        transient_peak_headroom=5,
        after_profile=SimpleNamespace(free_memory=90),
        non_kv_cache_memory=15,
    )

    @contextmanager
    def fake_memory_profiling(*args, **kwargs):
        yield profile_result

    worker = SimpleNamespace(
        cache_config=SimpleNamespace(
            kv_cache_memory_bytes=None,
            gpu_memory_utilization=0.9,
        ),
        model_runner=model_runner,
        init_snapshot=SimpleNamespace(free_memory=100, total_memory=100),
        requested_memory=90,
        device="cuda:0",
        model_config=SimpleNamespace(multimodal_config=None),
        parallel_config=SimpleNamespace(),
        _prepare_b12x_profile_state=lambda: events.append("prepare_b12x_profile_state"),
        _release_b12x_profile_state=lambda: events.append("release_b12x_profile_state"),
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(
                cudagraph_mode=gpu_worker.CUDAGraphMode.PIECEWISE,
                cudagraph_capture_sizes=[8, 4],
            )
        ),
    )

    monkeypatch.setattr(gpu_worker, "maybe_apply_startup_plan", lambda worker: None)
    monkeypatch.setenv(
        "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", str(int(estimate_graphs))
    )
    monkeypatch.setattr(gpu_worker, "memory_profiling", fake_memory_profiling)

    def final_snapshot(**_kwargs):
        assert events[-3:] == ["collect", "synchronize", "empty_cache"]
        events.append("final_snapshot")
        return SimpleNamespace(
            free_memory=final_free_memory, non_torch_memory=final_native
        )

    monkeypatch.setattr(gpu_worker.gc, "collect", lambda: events.append("collect"))
    monkeypatch.setattr(
        gpu_worker.torch.accelerator,
        "synchronize",
        lambda: events.append("synchronize"),
    )
    monkeypatch.setattr(
        gpu_worker.torch.accelerator,
        "empty_cache",
        lambda: events.append("empty_cache"),
    )
    monkeypatch.setattr(
        gpu_worker,
        "MemorySnapshot",
        final_snapshot,
    )
    monkeypatch.setattr(
        gpu_worker,
        "current_platform",
        SimpleNamespace(is_cuda_alike=lambda: True),
    )
    monkeypatch.setattr(
        gpu_worker,
        "reserve_mm_ipc_gpu_memory",
        lambda requested, *args: requested,
    )

    available = gpu_worker.Worker.determine_available_memory(worker)

    assert events == [
        "prepare_b12x_profile_state",
        "profile_run",
        "release_b12x_profile_state",
        "prepare_b12x_profile_state",
        "profile_glm_dcp_attention",
        "release_b12x_profile_state",
        "prepare_b12x_profile_state",
        "profile_cudagraph_memory",
        "release_b12x_profile_state",
        "collect",
        "synchronize",
        "empty_cache",
        "final_snapshot",
    ]
    graph_estimate -= min(overlap, graph_estimate, 90 - final_free_memory)
    applied_graph_estimate = graph_estimate if estimate_graphs else 0
    assert available == expected_available_memory - applied_graph_estimate
    assert worker.peak_activation_memory == 5
    assert worker.total_consumed == 10 + (90 - final_free_memory)
    assert worker.cudagraph_memory_estimate == graph_estimate


@pytest.mark.parametrize("estimated_gib", [0, 4])
@pytest.mark.parametrize("measured_gib", [3, 7])
def test_post_capture_recommendation_counts_measured_graph_memory_once(
    monkeypatch, estimated_gib, measured_gib
):
    """The saved KV budget uses measured graph storage, not its estimate."""
    compilation = SimpleNamespace(
        mode=gpu_worker.CompilationMode.NONE,
        compilation_time=0.0,
        encoder_compilation_time=0.0,
    )
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(compilation_config=compilation),
        compilation_config=compilation,
        model_runner=SimpleNamespace(
            lora_config=None,
            maybe_remove_all_loras=lambda config: None,
            capture_model=lambda: measured_gib * GiB_bytes,
        ),
        model_config=SimpleNamespace(enforce_eager=False, seed=0),
        cache_config=SimpleNamespace(
            kv_cache_memory_bytes=None, gpu_memory_utilization=0.9
        ),
        init_snapshot=SimpleNamespace(
            free_memory=100 * GiB_bytes, total_memory=100 * GiB_bytes
        ),
        requested_memory=90 * GiB_bytes,
        total_consumed=10 * GiB_bytes,
        peak_activation_memory=5 * GiB_bytes,
        cudagraph_memory_estimate=estimated_gib * GiB_bytes,
        available_kv_cache_memory_bytes=(75 - estimated_gib) * GiB_bytes,
        use_v2_model_runner=False,
        observability_config=SimpleNamespace(
            jit_monitor_mode="off", jit_monitor_verbose=False
        ),
    )
    saved = []
    monkeypatch.setattr(
        gpu_worker, "maybe_save_startup_plan", lambda w, budget: saved.append(budget)
    )
    monkeypatch.setattr(
        gpu_worker, "get_pp_group", lambda: SimpleNamespace(is_last_rank=False)
    )
    for name in (
        "kernel_warmup",
        "set_random_seed",
        "freeze_gc_heap",
        "maybe_attach_gc_debug_callback",
        "enable_gpu_sync_check",
        "set_torch_threads_for_runtime",
    ):
        monkeypatch.setattr(gpu_worker, name, lambda *args: None)
    monkeypatch.setattr("vllm.utils.jit_monitor.activate", lambda **kwargs: None)

    worker._compile_or_warm_up_model_after_preparation = lambda: (
        gpu_worker.Worker._compile_or_warm_up_model_after_preparation(worker)
    )
    worker._b12x_session = None
    worker._get_cudagraph_capture_context = nullcontext
    gpu_worker.Worker.compile_or_warm_up_model(worker)

    assert saved == [(90 - 10 - 5 - measured_gib) * GiB_bytes - 150 * (1 << 20)]
