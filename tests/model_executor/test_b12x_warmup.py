# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from types import SimpleNamespace

import pytest
import torch

import vllm.utils.b12x as b12x_utils
from vllm.model_executor.kernels.linear import (
    B12xFp8BlockScaledMMKernel,
    B12xMxFp4LinearKernel,
    B12xMxfp8LinearKernel,
    B12xNvFp4LinearKernel,
    B12xTensorFP8ScaledMMLinearKernel,
)
from vllm.model_executor.warmup.b12x_warmup import b12x_warmup
from vllm.utils.b12x import B12xWarmupUnit, b12x_warmup_token_counts


def test_b12x_scratch_uses_shared_workspace(monkeypatch) -> None:
    import vllm.v1.worker.workspace as workspace

    shared = torch.empty(32, dtype=torch.uint8)
    requests: list[tuple[object, ...]] = []

    def get_simultaneous(*specs: object) -> list[torch.Tensor]:
        requests.append(specs)
        return [shared]

    manager = SimpleNamespace(get_simultaneous=get_simultaneous)
    plan = SimpleNamespace(
        scratch_specs=lambda: (
            SimpleNamespace(shape=(32,), dtype=torch.uint8, device=torch.device("cpu")),
        )
    )
    monkeypatch.setattr(workspace, "is_workspace_manager_initialized", lambda: True)
    monkeypatch.setattr(workspace, "current_workspace_manager", lambda: manager)

    (scratch,) = b12x_utils.get_b12x_scratch_buffers(plan)
    assert scratch is shared
    assert requests == [(((32,), torch.uint8),)]


def test_b12x_scratch_allocates_without_workspace_manager(monkeypatch) -> None:
    import vllm.v1.worker.workspace as workspace

    plan = SimpleNamespace(
        scratch_specs=lambda: (
            SimpleNamespace(shape=(17,), dtype=torch.uint8, device=torch.device("cpu")),
        )
    )
    monkeypatch.setattr(workspace, "is_workspace_manager_initialized", lambda: False)

    (scratch,) = b12x_utils.get_b12x_scratch_buffers(plan)
    assert scratch.shape == (17,)
    assert scratch.dtype == torch.uint8
    assert scratch.device == torch.device("cpu")


@pytest.mark.parametrize(
    ("getter_name", "module_name"),
    [
        ("get_b12x_qsa", "b12x.attention.qsa"),
        ("get_b12x_hyperconnection", "b12x.norm.hyperconnection"),
        ("get_b12x_gdn_decode", "b12x.sequence.gdn_decode"),
        ("get_b12x_mtp_feedback", "b12x.sequence.mtp_feedback"),
        ("get_b12x_ple", "b12x.sequence.ple"),
        ("get_b12x_ple_embedding", "b12x.sequence.ple_embedding"),
        ("get_b12x_ple_hash", "b12x.sequence.ple_hash"),
    ],
)
def test_b12x_model_component_accessors(
    monkeypatch, getter_name: str, module_name: str
) -> None:
    sentinel = object()
    monkeypatch.setitem(b12x_utils._B12X_SUBMODULES, module_name, sentinel)

    assert getattr(b12x_utils, getter_name)() is sentinel


def test_b12x_warmup_token_counts_cover_serving_regimes() -> None:
    assert b12x_warmup_token_counts(
        max_tokens=2048,
        cudagraph_capture_sizes=[1, 2, 8, 128],
    ) == (1, 2, 8, 128, 2048)


@pytest.mark.parametrize(
    ("kernel_cls", "module_name", "call_name", "layer", "name"),
    [
        (
            B12xMxFp4LinearKernel,
            "vllm.model_executor.kernels.linear.mxfp4.b12x",
            "_apply_b12x_mxfp4_linear",
            SimpleNamespace(
                weight=torch.empty((48, 64), dtype=torch.uint8),
                weight_scale=torch.empty((128, 4), dtype=torch.uint8),
            ),
            "MXFP4",
        ),
        (
            B12xNvFp4LinearKernel,
            "vllm.model_executor.kernels.linear.nvfp4.b12x",
            "_apply_b12x_nvfp4_linear",
            SimpleNamespace(
                weight=torch.empty((48, 64), dtype=torch.uint8),
                weight_scale=torch.empty((128, 8), dtype=torch.float8_e4m3fn),
                input_global_scale_inv=torch.tensor(2.0),
                alpha=torch.tensor(0.25),
            ),
            "NVFP4",
        ),
        (
            B12xFp8BlockScaledMMKernel,
            "vllm.model_executor.kernels.linear.scaled_mm.b12x",
            "_run_b12x_fp8_block_scaled_mm",
            SimpleNamespace(
                weight=torch.empty((256, 128), dtype=torch.float8_e4m3fn),
                weight_scale_inv=torch.empty((2, 1), dtype=torch.float32),
            ),
            "block-FP8",
        ),
    ],
)
def test_b12x_warmup_units_cover_token_counts(
    monkeypatch,
    kernel_cls,
    module_name: str,
    call_name: str,
    layer,
    name: str,
) -> None:
    calls = []
    monkeypatch.setattr(
        importlib.import_module(module_name),
        call_name,
        lambda *args: calls.append(args),
    )
    kernel = object.__new__(kernel_cls)

    unit = kernel.get_b12x_warmup_unit(layer, (1, 8), torch.bfloat16)
    unit.compile()

    assert unit.name == name
    assert [args[0].shape[0] for args in calls] == [1, 8]
    assert unit.key[-1] == torch.bfloat16


def test_b12x_mxfp8_warmup_unit(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear.mxfp8.b12x as b12x_mod

    calls = []
    monkeypatch.setattr(
        b12x_mod,
        "_import_b12x_mxfp8",
        lambda: SimpleNamespace(
            mm=lambda *args, **kwargs: calls.append((args, kwargs))
        ),
    )
    monkeypatch.setattr(
        b12x_mod,
        "current_stream",
        lambda: SimpleNamespace(cuda_stream=object()),
    )
    packed_weight = SimpleNamespace(
        in_features=128,
        padded_in_features=128,
        out_features=256,
        weight=SimpleNamespace(values=torch.empty(1)),
    )
    layer = SimpleNamespace(b12x_mxfp8_packed_weight=packed_weight)
    kernel = object.__new__(B12xMxfp8LinearKernel)

    unit = kernel.get_b12x_warmup_unit(layer, (1, 8), torch.float16)
    unit.compile()

    assert [args[0].shape for args, _ in calls] == [(1, 128), (8, 128)]
    assert [kwargs["expected_m"] for _, kwargs in calls] == [1, 8]


def test_b12x_tensor_fp8_warmup_unit(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear.scaled_mm.b12x as b12x_mod

    calls = []
    monkeypatch.setattr(
        b12x_mod,
        "_import_b12x_tensor_fp8",
        lambda: SimpleNamespace(
            prewarm=lambda *args, **kwargs: calls.append((args, kwargs))
        ),
    )
    monkeypatch.setattr(
        b12x_mod,
        "current_stream",
        lambda: SimpleNamespace(cuda_stream=object()),
    )
    packed_weight = SimpleNamespace(
        in_features=128,
        padded_in_features=128,
        out_features=256,
        values=torch.empty(1),
    )
    layer = SimpleNamespace(b12x_tensor_fp8_packed_weight=packed_weight)
    kernel = object.__new__(B12xTensorFP8ScaledMMLinearKernel)

    unit = kernel.get_b12x_warmup_unit(layer, (1, 8), torch.bfloat16)
    unit.compile()

    assert calls[0][0] == (packed_weight, (1, 8))
    assert calls[0][1]["out_dtype"] == torch.bfloat16


def test_b12x_warmup_deduplicates_registered_signatures(monkeypatch) -> None:
    import vllm.model_executor.warmup.b12x_warmup as warmup_mod

    calls: list[tuple[str, tuple[int, ...], torch.dtype]] = []

    class Provider:
        def get_b12x_warmup_unit(self, layer, token_counts, output_dtype):
            return B12xWarmupUnit(
                name="fake",
                key=(type(self), layer.shape, output_dtype),
                compile=lambda: calls.append((layer.name, token_counts, output_dtype)),
            )

    provider = Provider()
    layers = [
        SimpleNamespace(name="first", shape=(128, 256), b12x_warmup_provider=provider),
        SimpleNamespace(
            name="duplicate", shape=(128, 256), b12x_warmup_provider=provider
        ),
        SimpleNamespace(name="second", shape=(256, 256), b12x_warmup_provider=provider),
        SimpleNamespace(),
    ]
    scans = 0

    def modules():
        nonlocal scans
        scans += 1
        return iter(layers)

    worker = SimpleNamespace(
        get_model=lambda: SimpleNamespace(modules=modules),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=8,
            max_num_scheduled_tokens=32,
        ),
        model_config=SimpleNamespace(dtype=torch.float32),
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(compile_sizes=[4, 16])
        ),
    )
    platform = SimpleNamespace(
        is_cuda=lambda: True,
        is_device_capability_family=lambda family: family == 120,
    )
    synchronized = []
    monkeypatch.setattr(warmup_mod, "current_platform", platform)
    monkeypatch.setattr(
        warmup_mod.torch.accelerator,
        "synchronize",
        lambda: synchronized.append(True),
    )

    b12x_warmup(worker, [1, 2])

    assert scans == 1
    assert calls == [
        ("first", (1, 2, 4, 8, 16, 32), torch.bfloat16),
        ("second", (1, 2, 4, 8, 16, 32), torch.bfloat16),
    ]
    assert synchronized == [True]


def test_b12x_dsa_indexer_warmup_unit_compiles_before_the_index_cache(
    monkeypatch,
) -> None:
    import vllm.v1.attention.backends.mla.b12x_indexer as indexer_mod

    calls = []
    monkeypatch.setattr(
        indexer_mod, "_run_paged_topk", lambda **kwargs: calls.append(kwargs)
    )
    device = torch.device("cpu")
    plan = SimpleNamespace(
        caps=SimpleNamespace(
            max_q_rows=4,
            mode="decode",
            num_q_heads=8,
            device=device,
            max_page_table_width=16,
        ),
        layout=SimpleNamespace(route="paged_fused"),
    )
    indexer = SimpleNamespace(
        k_cache=SimpleNamespace(kv_cache=torch.empty(0, dtype=torch.uint8)),
        _decode_plans={4: plan},
        _prefill_plans={},
        max_model_len=4096,
        topk_tokens=512,
        dcp_world_size=1,
        _module=object(),
        active_width_cap=torch.zeros(1, dtype=torch.int32),
        topk_indices_buffer=torch.zeros((4, 512), dtype=torch.int32),
        output_physical_slots=False,
    )
    unit = indexer_mod.B12xSparseIndexer.get_b12x_warmup_unit(
        indexer, None, (1,), torch.bfloat16
    )

    # Memory profiling runs warmup before the index cache exists: the unit
    # compiles against a placeholder cache with the production page layout.
    unit.compile()
    assert len(calls) == 1
    assert calls[0]["plan"] is plan
    assert tuple(calls[0]["kv_cache"].shape) == (2, 64, 132)
    assert calls[0]["kv_cache"].dtype == torch.uint8
    assert tuple(calls[0]["q"].shape) == (4, 8, 128)

    indexer.k_cache.kv_cache = torch.zeros((3, 64, 132), dtype=torch.uint8)
    unit.compile()
    assert len(calls) == 2
    assert calls[1]["kv_cache"] is indexer.k_cache.kv_cache
