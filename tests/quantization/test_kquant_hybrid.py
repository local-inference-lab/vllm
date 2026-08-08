# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization.kquant_hybrid import (
    KQuantHybridConfig,
    KQuantHybridMoEMethod,
    _b12x_tiles_for_geometry,
    _HybridSharedRuntime,
    _is_dense_layer_ignored,
    _read_hybrid_keys,
    _require_rank_local_kept_kernel,
    _stack_exl3_intermediate_rotations,
)


def _base_config(**updates):
    config = {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "hybrid_bit_map": {"1": [4, 3]},
        "kept_format": "mxfp4_e8m0k32",
    }
    config.update(updates)
    return config


def _qsrt_descriptor(**updates):
    descriptor = {
        "schema": "kquant_kimi_k3_qsrt_atoms_v1",
        "storage_format": "qsrt_atoms_v1",
        "encoding": "qsrt_sqg_e4m3",
        "codebook": "sqg_xor_cheb_t12",
        "artifact_manifest": "qsrt-manifest.json",
    }
    descriptor.update(updates)
    return descriptor


def _install_fake_b12x(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fused_moe: object,
    trellis_w4a8: object,
) -> None:
    modules = {
        name: ModuleType(name)
        for name in (
            "b12x",
            "b12x.moe",
            "b12x.moe._shared",
            "b12x.moe._shared.kernels",
            "b12x.moe._shared.kernels.w4a16",
            "b12x.moe._shared.kernels.w4a16.host",
            "b12x.moe._shared.kernels.trellis_w4a8",
        )
    }
    for name in (
        "b12x",
        "b12x.moe",
        "b12x.moe._shared",
        "b12x.moe._shared.kernels",
        "b12x.moe._shared.kernels.w4a16",
    ):
        modules[name].__path__ = []
    modules["b12x"].moe = modules["b12x.moe"]
    modules["b12x.moe"].fused_moe = fused_moe
    modules["b12x.moe"]._shared = modules["b12x.moe._shared"]
    modules["b12x.moe._shared"].kernels = modules["b12x.moe._shared.kernels"]
    modules["b12x.moe._shared.kernels"].w4a16 = modules[
        "b12x.moe._shared.kernels.w4a16"
    ]
    modules["b12x.moe._shared.kernels"].trellis_w4a8 = trellis_w4a8
    modules["b12x.moe._shared.kernels.w4a16"].host = modules[
        "b12x.moe._shared.kernels.w4a16.host"
    ]
    modules["b12x.moe._shared.kernels.w4a16.host"].make_w4a16_packed_buffers = (
        lambda *args, **kwargs: None
    )
    modules["b12x.moe._shared.kernels.w4a16.host"].max_packed_route_slots = (
        lambda *args, **kwargs: 1
    )
    modules["b12x.moe._shared.kernels.trellis_w4a8"] = trellis_w4a8
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def _w4a8_method(runtime: _HybridSharedRuntime) -> KQuantHybridMoEMethod:
    method = object.__new__(KQuantHybridMoEMethod)
    method.quant_config = SimpleNamespace(
        qsrt_runtime="w4a8",
        shared_runtime=runtime,
    )
    method.moe = SimpleNamespace(
        activation=SimpleNamespace(value="silu"),
        in_dtype=torch.float16,
        max_num_tokens=32,
        num_experts=256,
    )
    return method


def _prepared_part(marker: int) -> SimpleNamespace:
    prepared = SimpleNamespace(
        marker=marker,
        gate_suh=torch.empty((256, 1024)),
        up_suh=torch.empty((256, 1024)),
        w13=torch.empty(0),
    )
    return SimpleNamespace(
        plan=SimpleNamespace(intermediate_size=256),
        representation=SimpleNamespace(value=prepared),
    )


def _w4a8_state(parts: tuple[SimpleNamespace, ...]) -> SimpleNamespace:
    return SimpleNamespace(
        emap_secondary=torch.arange(256, dtype=torch.int32),
        hidden_size=1024,
        num_kept=0,
        num_secondary=256,
        prep_kept=None,
        runtime_ready=True,
        trellis_plan=None,
        trellis_weights=parts,
    )


@pytest.mark.parametrize(
    "raw",
    [
        {
            "hybrid_bit_map": {"0": [4, 3]},
            "kept_format": "mxfp4_e8m0k32",
        },
        {
            "quantization": {
                "hybrid_bit_map": {"0": [4, 3]},
                "kept_format": "mxfp4_e8m0k32",
            }
        },
    ],
)
def test_reads_and_detects_hybrid_checkpoint(raw) -> None:
    bit_map, kept_format = _read_hybrid_keys(raw)
    assert bit_map == {"0": [4, 3]}
    assert kept_format == "mxfp4_e8m0k32"
    assert KQuantHybridConfig.override_quantization_method(raw, None) == (
        "kquant_hybrid"
    )
    assert KQuantHybridConfig.override_quantization_method(raw, "fp8") is None


def test_config_registration_and_generic_exl3_default() -> None:
    assert get_quantization_config("kquant_hybrid") is KQuantHybridConfig
    config = KQuantHybridConfig.from_config(_base_config())
    assert config.hybrid_bit_map == {"1": [4, 3]}
    assert config.kept_format == "mxfp4_e8m0k32"
    assert config.demoted_format == "exl3_3"
    assert config.kept_storage == "inline-mxfp4"


@pytest.mark.parametrize(
    "schema",
    [
        "kquant_kimi_k3_qsrt_atoms_v1",
        "kquant_fruit_qsrt_atoms_v1",
    ],
)
def test_config_accepts_tp_independent_qsrt(schema: str) -> None:
    descriptor = _qsrt_descriptor(schema=schema)
    config = KQuantHybridConfig.from_config(
        _base_config(demoted_format="qsrt_sqg_e4m3", qsrt=descriptor)
    )
    assert config.demoted_format == "qsrt_sqg_e4m3"
    assert config.kept_storage == "x4t"
    assert config.qsrt == descriptor
    assert config.trellis_codebook == "sqg_xor_cheb_t12"


def test_config_accepts_fruit_w4a8_runtime() -> None:
    descriptor = _qsrt_descriptor(
        schema="kquant_fruit_qsrt_atoms_v1",
        runtime="W4A8",
    )
    config = KQuantHybridConfig.from_config(
        _base_config(demoted_format="qsrt_sqg_e4m3", qsrt=descriptor)
    )
    assert config.qsrt_runtime == "w4a8"
    assert config.qsrt == descriptor


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"demoted_format": "obsolete_private"}, "unsupported demoted_format"),
        (
            {"demoted_format": "qsrt_sqg_e4m3"},
            "requires a qsrt format descriptor",
        ),
        (
            {
                "demoted_format": "qsrt_sqg_e4m3",
                "qsrt": _qsrt_descriptor(storage_format="legacy-v1"),
            },
            "storage_format",
        ),
    ],
)
def test_config_rejects_obsolete_or_noncanonical_secondary_formats(
    updates, message
) -> None:
    with pytest.raises(ValueError, match=message):
        KQuantHybridConfig.from_config(_base_config(**updates))


def test_exl3_rotation_bundle_follows_b12x_projection_order() -> None:
    w13 = torch.arange(2 * 2 * 4, dtype=torch.float16).reshape(2, 2, 4)
    w2 = (100 + torch.arange(2 * 4, dtype=torch.float16)).reshape(2, 4)
    result = _stack_exl3_intermediate_rotations(w13, w2)
    expected = torch.cat((w13[:, 0], w13[:, 1], w2), dim=1)
    torch.testing.assert_close(result, expected)


def test_hybrid_kept_kernel_must_return_rank_local_partial() -> None:
    _require_rank_local_kept_kernel(SimpleNamespace(output_is_reduced=lambda: False))
    with pytest.raises(RuntimeError, match="unreduced rank-local partial"):
        _require_rank_local_kept_kernel(SimpleNamespace(output_is_reduced=lambda: True))


def test_b12x_tile_selection_is_geometry_driven() -> None:
    assert _b12x_tiles_for_geometry(3584, 3072) == (64, 256, 64, 256)
    assert _b12x_tiles_for_geometry(4096, 1536) == (64, 256, 64, 256)
    assert _b12x_tiles_for_geometry(1024, 256) == (64, 256, 64, 256)
    with pytest.raises(ValueError, match="no fixed b12x tile"):
        _b12x_tiles_for_geometry(3585, 3072)


@pytest.mark.parametrize(
    ("prefix", "ignored", "expected"),
    [
        ("model.layers.1.self_attn.q_proj", ["q_proj"], True),
        ("model.layers.1.self_attn.q_b_proj", ["b_proj"], False),
        ("model.layers.1.self_attn.q_proj", ["model.layers.1.self_attn.q_proj"], True),
    ],
)
def test_dense_short_exclusions_match_path_components(
    prefix, ignored, expected
) -> None:
    assert _is_dense_layer_ignored(prefix, ignored, {}) is expected


def test_w4a8_prepares_per_expert_scratch_and_dispatches_m_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLLM_KQUANT_CAPTURE_DIR", raising=False)
    backings = []
    make_calls = []
    view_calls = []
    run_calls = []
    kernel_output = torch.empty((32, 1024), dtype=torch.float32)

    def make_scratch(**kwargs):
        make_calls.append(kwargs)
        rows = kwargs["m"] if kwargs["shared_suh"] else kwargs["m"] * kwargs["topk"]
        backing = SimpleNamespace(
            gate=torch.empty(rows, dtype=torch.float8_e4m3fn),
            up=torch.empty(rows, dtype=torch.float8_e4m3fn),
        )
        backings.append(backing)
        return backing

    def view_scratch(scratch, **kwargs):
        rows = kwargs["m"] if kwargs["shared_suh"] else kwargs["m"] * kwargs["topk"]
        view = SimpleNamespace(
            backing=scratch,
            gate=scratch.gate[:rows],
            up=scratch.up[:rows],
            **kwargs,
        )
        view_calls.append(view)
        return view

    def run_w4a8(a, prepared, weights, ids, scratch, **kwargs):
        run_calls.append((prepared, scratch, kwargs))
        kernel_output.fill_(prepared.marker)
        return kernel_output

    trellis_w4a8 = SimpleNamespace(
        make_trellis_w4a8_moe_scratch=make_scratch,
        run_trellis_w4a8_moe=run_w4a8,
        view_trellis_w4a8_moe_scratch=view_scratch,
    )
    plan = SimpleNamespace(
        scratch_specs=lambda: [SimpleNamespace(shape=(1,), dtype=torch.uint8)]
    )
    fused_moe = SimpleNamespace(
        Caps=lambda **kwargs: SimpleNamespace(**kwargs),
        plan=lambda caps: plan,
    )
    _install_fake_b12x(
        monkeypatch,
        fused_moe=fused_moe,
        trellis_w4a8=trellis_w4a8,
    )

    runtime = _HybridSharedRuntime()
    runtime.trellis_scratch = torch.empty(1, dtype=torch.uint8)
    method = _w4a8_method(runtime)
    parts = (_prepared_part(0), _prepared_part(1))
    state = _w4a8_state(parts)
    state.runtime_ready = False
    layer = SimpleNamespace(hybrid_state=state)

    method._ensure_runtime(layer, m=2, topk=2)
    output_accum = runtime.trellis_output_accum
    assert output_accum is not None
    assert output_accum.shape == (32, 1024)
    assert output_accum.dtype == torch.float32

    assert make_calls == [
        {
            "m": 16,
            "topk": 2,
            "hidden_size": 1024,
            "intermediate_size": 256,
            "device": torch.device("cpu"),
            "shared_suh": False,
        }
    ]
    assert [view.m for view in view_calls] == list(range(1, 17))
    assert len(backings) == 1
    backing = backings[0]
    assert backing.gate.numel() == backing.up.numel() == 32
    assert all(view.backing is backing for view in view_calls)
    assert view_calls[1].gate.numel() == view_calls[1].up.numel() == 4
    assert view_calls[1].gate._base is backing.gate
    assert view_calls[1].up._base is backing.up
    assert all(view.topk == 2 and view.shared_suh is False for view in view_calls)
    assert len(run_calls) == 1
    assert run_calls[0][0].marker == 0
    assert run_calls[0][1] is runtime.w4a8_scratch[1]

    run_calls.clear()
    x = torch.zeros((2, 1024), dtype=torch.float16)
    output = method._apply_once(
        layer,
        x,
        torch.ones((2, 2), dtype=torch.float32),
        torch.zeros((2, 2), dtype=torch.int32),
        None,
        None,
    )
    assert runtime.trellis_output_accum is output_accum
    torch.testing.assert_close(output, torch.ones_like(output))
    assert [call[0].marker for call in run_calls] == [0, 1]
    assert all(call[1] is runtime.w4a8_scratch[2] for call in run_calls)


def test_w4a8_tuple_parts_accumulate_in_fp32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {0: 2048.0, 1: 1.0, 2: -2048.0}
    calls = []
    kernel_output = torch.empty((32, 1024), dtype=torch.float32)

    def run_w4a8(a, prepared, weights, ids, scratch, **kwargs):
        calls.append(prepared.marker)
        kernel_output.fill_(values[prepared.marker])
        return kernel_output

    _install_fake_b12x(
        monkeypatch,
        fused_moe=SimpleNamespace(),
        trellis_w4a8=SimpleNamespace(run_trellis_w4a8_moe=run_w4a8),
    )
    runtime = _HybridSharedRuntime()
    runtime.max_m = 32
    runtime.topk = 2
    runtime.w4a8_scratch[2] = object()
    runtime.trellis_output_accum = torch.empty((32, 1024), dtype=torch.float32)
    method = _w4a8_method(runtime)
    state = _w4a8_state(
        (_prepared_part(0), _prepared_part(1), _prepared_part(2))
    )

    output = method._apply_once(
        SimpleNamespace(hybrid_state=state),
        torch.zeros((2, 1024), dtype=torch.float16),
        torch.ones((2, 2), dtype=torch.float16),
        torch.zeros((2, 2), dtype=torch.int64),
        None,
        None,
    )

    assert calls == [0, 1, 2]
    torch.testing.assert_close(output, torch.ones_like(output))


def test_w4a8_above_m_limit_falls_back_to_w4a16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {0: 2048.0, 1: 1.0, 2: -2048.0}
    bindings = []
    kernel_output = torch.empty((32, 1024), dtype=torch.float32)

    def bind(plan, **kwargs):
        kwargs["plan"] = plan
        bindings.append(kwargs)
        return kwargs

    def run(*, binding):
        marker = binding["experts"].representation.value.marker
        kernel_output.fill_(values[marker])
        return kernel_output

    def fail_w4a8(*args, **kwargs):
        pytest.fail("M above the W4A8 limit must not dispatch W4A8")

    _install_fake_b12x(
        monkeypatch,
        fused_moe=SimpleNamespace(bind=bind, run=run),
        trellis_w4a8=SimpleNamespace(run_trellis_w4a8_moe=fail_w4a8),
    )
    runtime = _HybridSharedRuntime()
    runtime.max_m = 32
    runtime.topk = 2
    runtime.trellis_scratch = object()
    runtime.trellis_output_accum = torch.empty((32, 1024), dtype=torch.float32)
    method = _w4a8_method(runtime)
    state = _w4a8_state(
        (_prepared_part(0), _prepared_part(1), _prepared_part(2))
    )
    state.trellis_plan = object()

    output = method._apply_once(
        SimpleNamespace(hybrid_state=state),
        torch.zeros((17, 1024), dtype=torch.float16),
        torch.ones((17, 2), dtype=torch.float16),
        torch.zeros((17, 2), dtype=torch.int64),
        None,
        None,
    )

    assert len(bindings) == 3
    assert all(binding["scratch"] is runtime.trellis_scratch for binding in bindings)
    assert all(binding["topk_weights"].dtype == torch.float32 for binding in bindings)
    assert all(binding["topk_ids"].dtype == torch.int32 for binding in bindings)
    torch.testing.assert_close(output, torch.ones_like(output))


def test_w4a8_rejects_kept_tier_during_runtime_initialization() -> None:
    runtime = _HybridSharedRuntime()
    method = _w4a8_method(runtime)
    layer = SimpleNamespace(hybrid_state=SimpleNamespace(num_kept=1))

    with pytest.raises(RuntimeError, match="does not support a hybrid kept tier"):
        method._ensure_runtime(layer, m=1, topk=2)
    assert runtime.max_m is None
