# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization.kquant_hybrid import (
    KQuantHybridConfig,
    KQuantHybridMoEMethod,
    _b12x_tiles_for_geometry,
    _HybridSharedRuntime,
    _is_decode_only_forward,
    _is_dense_layer_ignored,
    _qsrt_backend_module,
    _read_hybrid_keys,
    _require_rank_local_kept_kernel,
    _stack_exl3_intermediate_rotations,
    _uses_fruit_w4a8,
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
    if descriptor["schema"] == "kquant_fruit_qsrt_atoms_v1":
        descriptor.setdefault("profile_id", 1)
        descriptor.setdefault("producer_fingerprint", "1" * 64)
        descriptor.setdefault("encoder_fingerprint", "2" * 64)
        descriptor.setdefault("source_kind", "safetensors_manifest")
        descriptor.setdefault("source_sha256", "3" * 64)
    return descriptor


def _canonical_fruit_config() -> dict[str, object]:
    return {
        "hidden_size": 1024,
        "moe_intermediate_size": 512,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 13,
        "quantization_config": _base_config(
            hybrid_bit_map={str(layer): [3] * 256 for layer in range(3, 14)},
            demoted_format="qsrt_sqg_e4m3",
            qsrt=_qsrt_descriptor(
                schema="kquant_fruit_qsrt_atoms_v1",
                runtime="w4a8",
            ),
        ),
    }


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
    modules["b12x.moe.fused_moe"] = fused_moe
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


def _prepared_part(
    marker: int,
    *,
    gate_rows: int = 256,
    up_rows: int | None = None,
    hidden_size: int = 1024,
    intermediate_size: int = 256,
    num_experts: int = 256,
    shared_suh: bool | None = None,
    gate_suh: torch.Tensor | None = None,
    up_suh: torch.Tensor | None = None,
) -> SimpleNamespace:
    up_rows = gate_rows if up_rows is None else up_rows
    if shared_suh is None:
        shared_suh = gate_rows == 1
    prepared = SimpleNamespace(
        marker=marker,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        gate_suh=(
            torch.empty((gate_rows, hidden_size)) if gate_suh is None else gate_suh
        ),
        up_suh=(torch.empty((up_rows, hidden_size)) if up_suh is None else up_suh),
        shared_suh=shared_suh,
        w13=torch.empty(0),
    )
    return SimpleNamespace(
        plan=SimpleNamespace(intermediate_size=intermediate_size),
        representation=SimpleNamespace(value=prepared),
    )


def _w4a8_state(parts: tuple[SimpleNamespace, ...]) -> SimpleNamespace:
    prepared = parts[0].representation.value
    shared_suh = prepared.shared_suh
    return SimpleNamespace(
        emap_secondary=torch.arange(256, dtype=torch.int32),
        hidden_size=int(prepared.gate_suh.shape[1]),
        num_kept=0,
        num_secondary=256,
        prep_kept=None,
        runtime_ready=True,
        trellis_plan=None,
        trellis_weights=parts,
        w4a8_scratch_geometry=(
            int(prepared.gate_suh.shape[1]),
            int(parts[0].plan.intermediate_size),
            2,
            shared_suh,
        ),
        trellis_output_accum_key=(
            32,
            int(prepared.gate_suh.shape[1]),
            torch.device("cpu"),
        ),
    )


def test_qsrt_backend_falls_back_to_sparkinfer(monkeypatch) -> None:
    expected = ModuleType("sparkinfer.moe.fused_moe")
    calls: list[str] = []

    def fake_import(name: str):
        calls.append(name)
        if name.startswith("b12x."):
            error = ModuleNotFoundError(name)
            error.name = name
            raise error
        return expected

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.kquant_hybrid.importlib.import_module",
        fake_import,
    )

    assert _qsrt_backend_module("moe.fused_moe") is expected
    assert calls == ["b12x.moe.fused_moe", "sparkinfer.moe.fused_moe"]


@pytest.mark.parametrize(
    ("num_decode", "num_prefill", "m", "expected"),
    ((2, 0, 2, True), (0, 2, 2, False), (1, 1, 2, False), (0, 0, 2, False)),
)
def test_qsrt_w4a8_phase_gate(
    monkeypatch: pytest.MonkeyPatch,
    num_decode: int,
    num_prefill: int,
    m: int,
    expected: bool,
) -> None:
    context = SimpleNamespace(
        additional_kwargs={},
        attn_metadata={
            "attention": SimpleNamespace(
                num_decode_tokens=num_decode,
                num_prefill_tokens=num_prefill,
            )
        },
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.kquant_hybrid."
        "is_forward_context_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.kquant_hybrid.get_forward_context",
        lambda: context,
    )

    assert _is_decode_only_forward(m) is expected


@pytest.mark.parametrize("metadata_shape", ("eager", "graph"))
def test_qsrt_w4a8_phase_gate_normalizes_b12x_spec_decode_metadata(
    monkeypatch: pytest.MonkeyPatch,
    metadata_shape: str,
) -> None:
    # MTP verifies one expert token while the sparse-attention metadata
    # represents the target plus draft rows as a two-token generic extend.
    metadata = SimpleNamespace(
        num_decode_tokens=0,
        num_prefill_tokens=2,
        is_spec_decode=True,
    )
    attn_metadata = (
        {"attention": metadata}
        if metadata_shape == "eager"
        else [{"attention": metadata}, {}]
    )
    context = SimpleNamespace(additional_kwargs={}, attn_metadata=attn_metadata)
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.kquant_hybrid."
        "is_forward_context_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.kquant_hybrid.get_forward_context",
        lambda: context,
    )

    assert _is_decode_only_forward(1)


@pytest.mark.parametrize("metadata_shape", ("eager", "graph"))
def test_qsrt_w4a8_phase_gate_rejects_mixed_spec_decode_and_prompt(
    monkeypatch: pytest.MonkeyPatch,
    metadata_shape: str,
) -> None:
    spec_decode = SimpleNamespace(
        num_decode_tokens=0,
        num_prefill_tokens=2,
        is_spec_decode=True,
    )
    prompt_prefill = SimpleNamespace(
        num_decode_tokens=0,
        num_prefill_tokens=1,
    )
    attn_metadata = (
        {"attention": spec_decode, "prompt_attention": prompt_prefill}
        if metadata_shape == "eager"
        else [{"attention": spec_decode}, {"prompt_attention": prompt_prefill}]
    )
    context = SimpleNamespace(additional_kwargs={}, attn_metadata=attn_metadata)
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.kquant_hybrid."
        "is_forward_context_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.kquant_hybrid.get_forward_context",
        lambda: context,
    )

    assert not _is_decode_only_forward(1)


def test_qsrt_w4a8_phase_gate_fails_closed_without_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.kquant_hybrid."
        "is_forward_context_available",
        lambda: False,
    )

    assert not _is_decode_only_forward(1)


def test_qsrt_w4a8_phase_gate_honors_explicit_draft_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = SimpleNamespace(
        additional_kwargs={"speculative_draft_decode_only": True},
        attn_metadata={
            "attention": SimpleNamespace(
                num_decode_tokens=0,
                num_prefill_tokens=1,
                is_spec_decode=False,
            )
        },
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.kquant_hybrid."
        "is_forward_context_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.kquant_hybrid.get_forward_context",
        lambda: context,
    )

    assert _is_decode_only_forward(1)
    context.additional_kwargs["speculative_draft_decode_only"] = False
    context.attn_metadata["attention"] = SimpleNamespace(
        num_decode_tokens=1,
        num_prefill_tokens=0,
        is_spec_decode=False,
    )
    assert not _is_decode_only_forward(1)


@pytest.mark.parametrize(
    ("runtime", "m", "decode_only", "expected"),
    (
        ("w4a8", 1, True, True),
        ("w4a8", 16, True, True),
        ("w4a8", 17, True, False),
        ("w4a8", 1, False, False),
        ("w4a16", 1, True, False),
    ),
)
def test_qsrt_w4a8_dispatch_and_audit_predicate(
    runtime: str,
    m: int,
    decode_only: bool,
    expected: bool,
) -> None:
    assert _uses_fruit_w4a8(runtime, m, decode_only) is expected


@pytest.mark.parametrize(
    ("metadata", "expected_mode", "expected_decode"),
    (
        (
            SimpleNamespace(
                num_decode_tokens=0,
                num_prefill_tokens=2,
                is_spec_decode=True,
            ),
            "w4a8",
            True,
        ),
        (
            SimpleNamespace(
                num_decode_tokens=0,
                num_prefill_tokens=1,
                is_spec_decode=False,
            ),
            "w4a16",
            False,
        ),
    ),
)
def test_qsrt_mtp_dispatch_and_evidence_follow_normalized_phase(
    monkeypatch: pytest.MonkeyPatch,
    metadata: SimpleNamespace,
    expected_mode: str,
    expected_decode: bool,
) -> None:
    context = SimpleNamespace(
        additional_kwargs={},
        attn_metadata={"attention": metadata},
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.kquant_hybrid."
        "is_forward_context_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.kquant_hybrid.get_forward_context",
        lambda: context,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.kquant_hybrid."
        "runtime_evidence_enabled",
        True,
    )
    monkeypatch.delenv("B12X_MOE_REPEAT_CHECK", raising=False)
    monkeypatch.delenv("VLLM_B12X_MOE_REPEAT_CHECK", raising=False)

    dispatch_phases: list[bool] = []
    evidence: list[tuple[object, str, bool, int]] = []

    def apply_once(*args, decode_only: bool):
        dispatch_phases.append(decode_only)
        return args[1]

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.kquant_hybrid.record_layer_execution",
        lambda *args: evidence.append(args),
    )
    method = _w4a8_method(_HybridSharedRuntime())
    monkeypatch.setattr(method, "_apply_once", apply_once)
    evidence_layer = object()
    layer = SimpleNamespace(
        hybrid_state=SimpleNamespace(
            trellis_weights=(object(),),
            uses_qsrt_atoms=True,
            runtime_evidence_layer=evidence_layer,
        )
    )
    x = torch.zeros((1, 1024), dtype=torch.float16)

    assert (
        method.apply(
            layer,
            x,
            torch.ones((1, 2), dtype=torch.float32),
            torch.zeros((1, 2), dtype=torch.int32),
            None,
            None,
        )
        is x
    )
    assert dispatch_phases == [expected_decode]
    assert evidence == [(evidence_layer, expected_mode, expected_decode, 1)]


def test_qsrt_backend_reports_both_missing_namespaces(monkeypatch) -> None:
    calls: list[str] = []

    def fake_import(name: str):
        calls.append(name)
        error = ModuleNotFoundError(f"No module named {name!r}")
        error.name = name
        raise error

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.kquant_hybrid.importlib.import_module",
        fake_import,
    )

    with pytest.raises(ModuleNotFoundError) as captured:
        _qsrt_backend_module("moe.fused_moe")
    message = str(captured.value)
    assert "b12x.moe.fused_moe" in message
    assert "sparkinfer.moe.fused_moe" in message
    assert calls == ["b12x.moe.fused_moe", "sparkinfer.moe.fused_moe"]


def test_qsrt_backend_preserves_broken_primary_import(monkeypatch) -> None:
    broken = ModuleNotFoundError("No module named 'b12x.internal.helper'")
    broken.name = "b12x.internal.helper"

    def fake_import(name: str):
        if name == "b12x.moe.fused_moe":
            raise broken
        pytest.fail("a broken primary backend must not fall back")

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.kquant_hybrid.importlib.import_module",
        fake_import,
    )

    with pytest.raises(ModuleNotFoundError) as captured:
        _qsrt_backend_module("moe.fused_moe")
    assert captured.value is broken


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


def test_legacy_siq_rank0_trellis_loads_k4_and_k3_tiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = KQuantHybridConfig.from_config(
        _base_config(
            hybrid_bit_map={"3": [4, 3]},
            kept_format="exl3_4",
            trellis={
                "codebook": "mcg",
                "mcg_mult": 3417055213,
                "shared_su": False,
            },
        )
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.kquant_hybrid."
        "get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.kquant_hybrid."
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: "cpu")
    method = object.__new__(KQuantHybridMoEMethod)
    method.quant_config = config
    layer = torch.nn.Module()
    layer.activation = MoEActivation.SILU
    layer.layer_name = "model.layers.3.mlp.experts"

    method.create_weights(
        layer,
        num_experts=2,
        hidden_size=128,
        intermediate_size_per_partition=64,
        params_dtype=torch.float16,
    )

    assert layer.hybrid_state.kept_exl4 is True
    assert tuple(layer.w13_exl4_trellis.shape) == (1, 2, 8, 4, 64)
    assert tuple(layer.w13_exl3_trellis.shape) == (1, 2, 8, 4, 48)
    loader = layer.w13_rank0.trellis.weight_loader
    k4 = torch.arange(8 * 4 * 64, dtype=torch.int16).reshape(8, 4, 64)
    k3 = torch.arange(8 * 4 * 48, dtype=torch.int16).reshape(8, 4, 48)
    assert loader(
        layer.w13_rank0.trellis,
        k4,
        name_mapped="w13_rank0.trellis",
        shard_id="w1",
        expert_id=0,
    )
    assert loader(
        layer.w13_rank0.trellis,
        k3,
        name_mapped="w13_rank0.trellis",
        shard_id="w3",
        expert_id=1,
    )
    torch.testing.assert_close(layer.w13_exl4_trellis[0, 0], k4)
    torch.testing.assert_close(layer.w13_exl3_trellis[0, 1], k3)


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


def test_fruit_publication_anchor_comes_only_from_required_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw_config = _canonical_fruit_config()
    method = object.__new__(KQuantHybridMoEMethod)
    method.quant_config = KQuantHybridConfig.from_config(
        raw_config["quantization_config"]
    )
    model_config = SimpleNamespace(model=str(tmp_path))
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config",
        lambda: SimpleNamespace(model_config=model_config),
    )
    monkeypatch.setenv("FRUIT_QSRT_AUTHENTICATED_MODEL_ROOT", str(tmp_path))
    monkeypatch.delenv("FRUIT_QSRT_EXPECTED_COMPLETE_SHA256", raising=False)
    monkeypatch.delenv("FRUIT_QSRT_EXPECTED_CANDIDATE_SHA256", raising=False)

    with pytest.raises(ValueError, match="FRUIT_QSRT_EXPECTED_COMPLETE_SHA256"):
        method._ensure_fruit_publication()

    monkeypatch.setenv("FRUIT_QSRT_EXPECTED_COMPLETE_SHA256", "A" * 64)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        method._ensure_fruit_publication()

    trusted = "a" * 64
    publication = SimpleNamespace(
        root=tmp_path.resolve(),
        config=raw_config,
    )
    received: list[tuple[Path, dict[str, object]]] = []
    monkeypatch.setenv("FRUIT_QSRT_EXPECTED_COMPLETE_SHA256", trusted)
    runtime_environment = {"MODE": "fixed"}
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.kquant_hybrid."
        "active_runtime_environment_from_env",
        lambda: runtime_environment,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.kquant_hybrid.verify_qsrt_publication",
        lambda root, **trust: received.append((Path(root), trust)) or publication,
    )

    assert method._ensure_fruit_publication() is publication
    assert received == [
        (
            tmp_path,
            {
                "expected_complete_sha256": trusted,
                "candidate_mode": False,
                "expected_candidate_sha256": None,
                "active_runtime_environment": runtime_environment,
            },
        )
    ]


def test_fruit_candidate_publication_requires_exclusive_explicit_trust(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw_config = _canonical_fruit_config()
    method = object.__new__(KQuantHybridMoEMethod)
    method.quant_config = KQuantHybridConfig.from_config(
        raw_config["quantization_config"]
    )
    model_config = SimpleNamespace(model=str(tmp_path))
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config",
        lambda: SimpleNamespace(model_config=model_config),
    )
    monkeypatch.setenv("FRUIT_QSRT_AUTHENTICATED_MODEL_ROOT", str(tmp_path))
    monkeypatch.setenv("FRUIT_QSRT_EXPECTED_CANDIDATE_SHA256", "A" * 64)
    monkeypatch.delenv("FRUIT_QSRT_EXPECTED_COMPLETE_SHA256", raising=False)

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        method._ensure_fruit_publication()

    trusted = "b" * 64
    monkeypatch.setenv("FRUIT_QSRT_EXPECTED_CANDIDATE_SHA256", trusted)
    monkeypatch.setenv("FRUIT_QSRT_EXPECTED_COMPLETE_SHA256", "a" * 64)
    with pytest.raises(ValueError, match="exactly one"):
        method._ensure_fruit_publication()

    monkeypatch.delenv("FRUIT_QSRT_EXPECTED_COMPLETE_SHA256")
    publication = SimpleNamespace(
        root=tmp_path.resolve(),
        config=raw_config,
    )
    received: list[tuple[Path, dict[str, object]]] = []
    runtime_environment = {"MODE": "fixed"}
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.kquant_hybrid."
        "active_runtime_environment_from_env",
        lambda: runtime_environment,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.kquant_hybrid.verify_qsrt_publication",
        lambda root, **trust: received.append((Path(root), trust)) or publication,
    )

    assert method._ensure_fruit_publication() is publication
    assert received == [
        (
            tmp_path,
            {
                "expected_complete_sha256": None,
                "candidate_mode": True,
                "expected_candidate_sha256": trusted,
                "active_runtime_environment": runtime_environment,
            },
        )
    ]


def test_authenticated_config_controls_fruit_dispatch() -> None:
    raw_config = _canonical_fruit_config()
    method = object.__new__(KQuantHybridMoEMethod)
    method.quant_config = KQuantHybridConfig.from_config(
        raw_config["quantization_config"]
    )
    method._reconcile_authenticated_dispatch(raw_config)

    mutated = {
        **raw_config,
        "quantization_config": {
            **raw_config["quantization_config"],
            "hybrid_bit_map": {
                **raw_config["quantization_config"]["hybrid_bit_map"],
                "3": [4] + [3] * 255,
            },
        },
    }
    with pytest.raises(RuntimeError, match="authenticated config bytes"):
        method._reconcile_authenticated_dispatch(mutated)


def test_config_rejects_kimi_w4a8_runtime() -> None:
    descriptor = _qsrt_descriptor(runtime="w4a8")

    with pytest.raises(ValueError, match="runtime 'w4a8' requires schema"):
        KQuantHybridConfig.from_config(
            _base_config(demoted_format="qsrt_sqg_e4m3", qsrt=descriptor)
        )


def test_fruit_w4a8_rejects_kept_experts_before_parameter_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = _qsrt_descriptor(
        schema="kquant_fruit_qsrt_atoms_v1",
        runtime="W4A8",
    )
    config = KQuantHybridConfig.from_config(
        _base_config(demoted_format="qsrt_sqg_e4m3", qsrt=descriptor)
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.kquant_hybrid."
        "get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.kquant_hybrid."
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    method = object.__new__(KQuantHybridMoEMethod)
    method.quant_config = config
    monkeypatch.setattr(method, "_ensure_fruit_publication", lambda: object())
    layer = SimpleNamespace(
        activation=MoEActivation.SILU,
        layer_name="model.layers.1.mlp.experts",
    )

    with pytest.raises(ValueError, match="does not support a hybrid kept tier"):
        method.create_weights(
            layer,
            num_experts=2,
            hidden_size=256,
            intermediate_size_per_partition=256,
            params_dtype=torch.float16,
        )

    assert not hasattr(layer, "qsrt_atom_placeholder")


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


@pytest.mark.parametrize(
    ("rotation_rows", "shared_suh"),
    ((256, False), (1, True)),
)
def test_w4a8_prepares_geometry_scratch_and_dispatches_m_view(
    monkeypatch: pytest.MonkeyPatch,
    rotation_rows: int,
    shared_suh: bool,
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

    def run_w4a8_parts(a, prepared_parts, weights, ids, scratch, **kwargs):
        run_calls.append((prepared_parts, scratch, kwargs))
        target = kwargs.get("output_accum")
        if target is None:
            target = kernel_output[: int(a.shape[0])]
        target.fill_(sum(part.marker for part in prepared_parts))
        return target

    trellis_w4a8 = SimpleNamespace(
        make_trellis_w4a8_moe_scratch=make_scratch,
        run_trellis_w4a8_moe_parts=run_w4a8_parts,
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
    gate_suh = torch.empty((rotation_rows, 1024))
    up_suh = torch.empty((rotation_rows, 1024))
    parts = (
        _prepared_part(
            0,
            gate_rows=rotation_rows,
            gate_suh=gate_suh,
            up_suh=up_suh,
        ),
        _prepared_part(
            1,
            gate_rows=rotation_rows,
            gate_suh=gate_suh,
            up_suh=up_suh,
        ),
    )
    state = _w4a8_state(parts)
    state.runtime_ready = False
    layer = SimpleNamespace(hybrid_state=state)

    method._ensure_runtime(layer, m=2, topk=2)
    output_accum = runtime.trellis_output_accums[state.trellis_output_accum_key]
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
            "shared_suh": shared_suh,
        }
    ]
    assert [view.m for view in view_calls] == list(range(1, 17))
    assert len(backings) == 1
    backing = backings[0]
    expected_backing_rows = 16 if shared_suh else 32
    assert backing.gate.numel() == backing.up.numel() == expected_backing_rows
    assert all(view.backing is backing for view in view_calls)
    expected_view_rows = 2 if shared_suh else 4
    assert view_calls[1].gate.numel() == expected_view_rows
    assert view_calls[1].up.numel() == expected_view_rows
    assert view_calls[1].gate.data_ptr() == backing.gate.data_ptr()
    assert view_calls[1].up.data_ptr() == backing.up.data_ptr()
    assert all(view.topk == 2 and view.shared_suh is shared_suh for view in view_calls)
    assert len(run_calls) == 1
    assert [part.marker for part in run_calls[0][0]] == [0, 1]
    geometry = (1024, 256, 2, shared_suh)
    assert run_calls[0][1] is runtime.w4a8_scratch[(*geometry, 1)]

    run_calls.clear()
    x = torch.zeros((2, 1024), dtype=torch.float16)
    output = method._apply_once(
        layer,
        x,
        torch.ones((2, 2), dtype=torch.float32),
        torch.zeros((2, 2), dtype=torch.int32),
        None,
        None,
        decode_only=True,
    )
    assert runtime.trellis_output_accums[state.trellis_output_accum_key] is output_accum
    torch.testing.assert_close(output, torch.ones_like(output))
    assert len(run_calls) == 1
    assert [part.marker for part in run_calls[0][0]] == [0, 1]
    assert run_calls[0][1] is runtime.w4a8_scratch[(*geometry, 2)]


@pytest.mark.parametrize(
    ("row_shapes", "message"),
    (
        (((256, 256), (1, 1)), "disagree on shared_suh"),
        (((2, 2),), "disagree with shared_suh"),
        (((1, 256),), "disagree with shared_suh"),
    ),
)
def test_w4a8_rejects_inconsistent_rotation_rows(
    monkeypatch: pytest.MonkeyPatch,
    row_shapes: tuple[tuple[int, int], ...],
    message: str,
) -> None:
    plan = SimpleNamespace(
        scratch_specs=lambda: [SimpleNamespace(shape=(1,), dtype=torch.uint8)]
    )
    _install_fake_b12x(
        monkeypatch,
        fused_moe=SimpleNamespace(
            Caps=lambda **kwargs: SimpleNamespace(**kwargs),
            plan=lambda caps: plan,
        ),
        trellis_w4a8=SimpleNamespace(
            make_trellis_w4a8_moe_scratch=lambda **kwargs: pytest.fail(
                "invalid rotations must fail before scratch allocation"
            ),
            run_trellis_w4a8_moe_parts=lambda *args, **kwargs: None,
            view_trellis_w4a8_moe_scratch=lambda *args, **kwargs: None,
        ),
    )
    runtime = _HybridSharedRuntime()
    runtime.trellis_scratch = torch.empty(1, dtype=torch.uint8)
    method = _w4a8_method(runtime)
    parts = tuple(
        _prepared_part(index, gate_rows=gate_rows, up_rows=up_rows)
        for index, (gate_rows, up_rows) in enumerate(row_shapes)
    )
    state = _w4a8_state(parts)
    state.runtime_ready = False

    with pytest.raises(RuntimeError, match=message):
        method._ensure_runtime(SimpleNamespace(hybrid_state=state), m=2, topk=2)


def test_w4a8_rejects_multipart_suh_identity_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = SimpleNamespace(
        scratch_specs=lambda: [SimpleNamespace(shape=(1,), dtype=torch.uint8)]
    )
    _install_fake_b12x(
        monkeypatch,
        fused_moe=SimpleNamespace(
            Caps=lambda **kwargs: SimpleNamespace(**kwargs),
            plan=lambda caps: plan,
        ),
        trellis_w4a8=SimpleNamespace(
            make_trellis_w4a8_moe_scratch=lambda **kwargs: pytest.fail(
                "incompatible parts must fail before scratch allocation"
            ),
            run_trellis_w4a8_moe_parts=lambda *args, **kwargs: None,
            view_trellis_w4a8_moe_scratch=lambda *args, **kwargs: None,
        ),
    )
    runtime = _HybridSharedRuntime()
    runtime.trellis_scratch = torch.empty(1, dtype=torch.uint8)
    method = _w4a8_method(runtime)
    parts = (_prepared_part(0), _prepared_part(1))
    state = _w4a8_state(parts)
    state.runtime_ready = False

    with pytest.raises(RuntimeError, match="does not share gate_suh identity"):
        method._ensure_runtime(SimpleNamespace(hybrid_state=state), m=2, topk=2)


def test_w4a8_scratch_cache_separates_layer_geometries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    make_calls = []

    def make_scratch(**kwargs):
        make_calls.append(kwargs)
        return SimpleNamespace(**kwargs)

    plan = SimpleNamespace(
        scratch_specs=lambda: [SimpleNamespace(shape=(1,), dtype=torch.uint8)]
    )
    _install_fake_b12x(
        monkeypatch,
        fused_moe=SimpleNamespace(
            Caps=lambda **kwargs: SimpleNamespace(**kwargs),
            plan=lambda caps: plan,
        ),
        trellis_w4a8=SimpleNamespace(
            make_trellis_w4a8_moe_scratch=make_scratch,
            run_trellis_w4a8_moe_parts=lambda *args, **kwargs: torch.empty(0),
            view_trellis_w4a8_moe_scratch=lambda scratch, **kwargs: scratch,
        ),
    )
    runtime = _HybridSharedRuntime()
    runtime.trellis_scratch = torch.empty(1, dtype=torch.uint8)
    method = _w4a8_method(runtime)
    geometries = ((1024, 256), (2048, 256))

    for marker, (hidden_size, intermediate_size) in enumerate(geometries):
        gate_suh = torch.empty((256, hidden_size))
        up_suh = torch.empty((256, hidden_size))
        parts = tuple(
            _prepared_part(
                marker * 2 + part_index,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                gate_suh=gate_suh,
                up_suh=up_suh,
            )
            for part_index in range(2)
        )
        state = _w4a8_state(parts)
        state.runtime_ready = False
        method._ensure_runtime(
            SimpleNamespace(hybrid_state=state),
            m=2,
            topk=2,
        )

    assert [
        (call["hidden_size"], call["intermediate_size"]) for call in make_calls
    ] == list(geometries)
    for hidden_size, intermediate_size in geometries:
        assert (hidden_size, intermediate_size, 2, False, 2) in runtime.w4a8_scratch
    expected_accum_keys = {
        (32, hidden_size, torch.device("cpu"))
        for hidden_size, _intermediate_size in geometries
    }
    assert set(runtime.trellis_output_accums) == expected_accum_keys


def test_w4a8_tuple_parts_accumulate_in_fp32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {0: 2048.0, 1: 1.0, 2: -2048.0}
    calls = []

    def run_w4a8_parts(a, prepared_parts, weights, ids, scratch, **kwargs):
        calls.extend(part.marker for part in prepared_parts)
        output_accum = kwargs["output_accum"]
        output_accum.fill_(sum(values[part.marker] for part in prepared_parts))
        return output_accum

    _install_fake_b12x(
        monkeypatch,
        fused_moe=SimpleNamespace(),
        trellis_w4a8=SimpleNamespace(run_trellis_w4a8_moe_parts=run_w4a8_parts),
    )
    runtime = _HybridSharedRuntime()
    runtime.max_m = 32
    runtime.topk = 2
    runtime.w4a8_scratch[(1024, 256, 2, False, 2)] = object()
    accum_key = (32, 1024, torch.device("cpu"))
    runtime.trellis_output_accums[accum_key] = torch.empty(
        (32, 1024), dtype=torch.float32
    )
    method = _w4a8_method(runtime)
    state = _w4a8_state((_prepared_part(0), _prepared_part(1), _prepared_part(2)))

    output = method._apply_once(
        SimpleNamespace(hybrid_state=state),
        torch.zeros((2, 1024), dtype=torch.float32),
        torch.ones((2, 2), dtype=torch.float16),
        torch.zeros((2, 2), dtype=torch.int64),
        None,
        None,
        decode_only=True,
    )

    assert calls == [0, 1, 2]
    torch.testing.assert_close(output, torch.ones_like(output))
    assert output.data_ptr() != runtime.trellis_output_accums[accum_key].data_ptr()


def test_w4a8_single_part_returns_owned_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel_output = torch.ones((32, 1024), dtype=torch.float32)
    _install_fake_b12x(
        monkeypatch,
        fused_moe=SimpleNamespace(),
        trellis_w4a8=SimpleNamespace(
            run_trellis_w4a8_moe_parts=lambda a, *args, **kwargs: kernel_output[
                : int(a.shape[0])
            ]
        ),
    )
    runtime = _HybridSharedRuntime()
    runtime.max_m = 32
    runtime.topk = 2
    runtime.w4a8_scratch[(1024, 256, 2, False, 2)] = object()
    method = _w4a8_method(runtime)
    state = _w4a8_state((_prepared_part(0),))

    output = method._apply_once(
        SimpleNamespace(hybrid_state=state),
        torch.zeros((2, 1024), dtype=torch.float32),
        torch.ones((2, 2), dtype=torch.float32),
        torch.zeros((2, 2), dtype=torch.int32),
        None,
        None,
        decode_only=True,
    )

    torch.testing.assert_close(output, torch.ones_like(output))
    assert output.data_ptr() != kernel_output.data_ptr()


def test_w4a8_capture_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_KQUANT_CAPTURE_DIR", "/tmp/capture")
    _install_fake_b12x(
        monkeypatch,
        fused_moe=SimpleNamespace(),
        trellis_w4a8=SimpleNamespace(
            run_trellis_w4a8_moe_parts=lambda *args, **kwargs: pytest.fail(
                "capture must fail before W4A8 launch"
            )
        ),
    )
    runtime = _HybridSharedRuntime()
    runtime.max_m = 32
    runtime.topk = 2
    runtime.w4a8_scratch[(1024, 256, 2, False, 1)] = object()
    method = _w4a8_method(runtime)
    state = _w4a8_state((_prepared_part(0),))

    with pytest.raises(RuntimeError, match="capture is unavailable for Fruit W4A8"):
        method._apply_once(
            SimpleNamespace(hybrid_state=state),
            torch.zeros((1, 1024), dtype=torch.float32),
            torch.ones((1, 2), dtype=torch.float32),
            torch.zeros((1, 2), dtype=torch.int32),
            None,
            None,
            decode_only=True,
        )


@pytest.mark.parametrize(("m", "decode_only"), ((17, True), (2, False)))
def test_w4a8_falls_back_outside_decode_contract(
    monkeypatch: pytest.MonkeyPatch,
    m: int,
    decode_only: bool,
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
        pytest.fail("non-decode or oversized batches must not dispatch W4A8")

    _install_fake_b12x(
        monkeypatch,
        fused_moe=SimpleNamespace(bind=bind, run=run),
        trellis_w4a8=SimpleNamespace(run_trellis_w4a8_moe_parts=fail_w4a8),
    )
    runtime = _HybridSharedRuntime()
    runtime.max_m = 32
    runtime.topk = 2
    runtime.trellis_scratch = object()
    runtime.trellis_output_accums[(32, 1024, torch.device("cpu"))] = torch.empty(
        (32, 1024), dtype=torch.float32
    )
    method = _w4a8_method(runtime)
    state = _w4a8_state((_prepared_part(0), _prepared_part(1), _prepared_part(2)))
    state.trellis_plan = object()

    output = method._apply_once(
        SimpleNamespace(hybrid_state=state),
        torch.zeros((m, 1024), dtype=torch.float16),
        torch.ones((m, 2), dtype=torch.float16),
        torch.zeros((m, 2), dtype=torch.int64),
        None,
        None,
        decode_only=decode_only,
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
