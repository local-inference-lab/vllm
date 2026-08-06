# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.models.kimi_linear as kimi_linear
from vllm.model_executor.models.kimi_linear import (
    _EXPERT_TENSOR_RE,
    KimiLinearModel,
    _canonical_grouped_topk,
    _KimiCorrectnessTrace,
    _parse_int_ranges,
    _resolve_expert_parameter_name,
)


def test_parse_int_ranges() -> None:
    assert _parse_int_ranges("0,2-4, 7", name="TEST") == frozenset({0, 2, 3, 4, 7})
    with pytest.raises(ValueError, match="ascending"):
        _parse_int_ranges("4-2", name="TEST")


def test_trace_environment_requires_explicit_layers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("KIMI_CORRECTNESS_TRACE_DIR", str(tmp_path))
    monkeypatch.delenv("KIMI_CORRECTNESS_TRACE_LAYERS", raising=False)
    with pytest.raises(ValueError, match="TRACE_LAYERS"):
        _KimiCorrectnessTrace.from_env()


def test_trace_is_bounded_and_records_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(kimi_linear, "get_tensor_model_parallel_rank", lambda: 3)
    monkeypatch.setattr(kimi_linear, "get_tensor_model_parallel_world_size", lambda: 12)
    trace = _KimiCorrectnessTrace(
        tmp_path,
        layers=frozenset({1}),
        ranks=frozenset({3}),
        start_call=1,
        max_calls=1,
        max_tokens=2,
        token_window="head",
    )
    tensor = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    trace.capture(1, "router_logits", tensor)
    trace.capture(1, "router_logits", tensor)
    trace.capture(1, "router_logits", tensor)

    files = list((tmp_path / "tp-rank-003").glob("*.pt"))
    assert [path.name for path in files] == ["layer-001.call-000001.router_logits.pt"]
    payload = torch.load(files[0])
    torch.testing.assert_close(payload["tensor"], tensor[:2])
    assert payload["metadata"]["original_shape"] == [3, 4]
    assert payload["metadata"]["saved_shape"] == [2, 4]
    assert payload["metadata"]["tp_world_size"] == 12
    assert len(payload["metadata"]["sha256"]) == 64

    manifest = json.loads((tmp_path / "tp-rank-003" / "manifest.json").read_text())
    assert manifest["tp_rank"] == 3
    assert manifest["start_call"] == 1
    assert manifest["max_calls"] == 1
    assert manifest["token_window"] == "head"


def test_trace_defaults_to_tail_window(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(kimi_linear, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(kimi_linear, "get_tensor_model_parallel_world_size", lambda: 1)
    trace = _KimiCorrectnessTrace(
        tmp_path,
        layers=frozenset({0}),
        max_tokens=2,
    )
    tensor = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    trace.capture(0, "layer_input", tensor)

    path = tmp_path / "tp-rank-000" / "layer-000.call-000000.layer_input.pt"
    payload = torch.load(path)
    torch.testing.assert_close(payload["tensor"], tensor[-2:])
    assert payload["metadata"]["token_window"] == "tail"


def test_trace_requires_eager(tmp_path) -> None:
    trace = _KimiCorrectnessTrace(tmp_path, layers=frozenset({1}))
    with pytest.raises(ValueError, match="--enforce-eager"):
        trace.require_eager(1, enforce_eager=False)
    trace.require_eager(1, enforce_eager=True)
    trace.require_eager(2, enforce_eager=False)


def test_canonical_grouped_topk_uses_bias_only_for_selection() -> None:
    logits = torch.tensor([[0.0, 0.1, 0.2, 0.3]])
    bias = torch.tensor([[0.0, 2.0, 0.0, 0.0]]).squeeze(0)
    weights, ids = _canonical_grouped_topk(
        logits,
        top_k=1,
        num_expert_group=2,
        topk_group=1,
        scoring_func="sigmoid",
        renormalize=False,
        routed_scaling_factor=2.0,
        correction_bias=bias,
    )
    assert ids.tolist() == [[1]]
    expected_weight = logits.sigmoid()[0, 1] * 2.0
    torch.testing.assert_close(weights[0, 0], expected_weight)


@pytest.mark.parametrize(
    ("name", "groups"),
    [
        (
            "model.layers.1.block_sparse_moe.experts.17.w1.weight_packed",
            (
                "model.layers.1.block_sparse_moe.experts",
                "17",
                "w1",
                "weight_packed",
            ),
        ),
        (
            "model.layers.91.block_sparse_moe.experts.895.w3.nf3_scale",
            (
                "model.layers.91.block_sparse_moe.experts",
                "895",
                "w3",
                "nf3_scale",
            ),
        ),
    ],
)
def test_expert_tensor_fast_path_regex(
    name: str,
    groups: tuple[str, str, str, str],
) -> None:
    match = _EXPERT_TENSOR_RE.match(name)
    assert match is not None
    assert match.groups() == groups


@pytest.mark.parametrize(
    ("parameter_name", "expected"),
    [
        (
            "model.layers.1.block_sparse_moe.experts.routed_experts.w13_weight_packed",
            "model.layers.1.block_sparse_moe.experts.routed_experts.w13_weight_packed",
        ),
        (
            "model.layers.1.block_sparse_moe.experts.w13_weight_packed",
            "model.layers.1.block_sparse_moe.experts.w13_weight_packed",
        ),
    ],
)
def test_resolve_expert_parameter_name_supports_routed_and_legacy_layouts(
    parameter_name: str,
    expected: str,
) -> None:
    parameter = torch.nn.Parameter(torch.empty(0), requires_grad=False)
    params = {parameter_name: parameter}
    assert (
        _resolve_expert_parameter_name(
            params,
            expert_prefix="model.layers.1.block_sparse_moe.experts",
            weight_key="w1",
            suffix="weight_packed",
        )
        == expected
    )


def test_kda_qkv_checkpoint_shards_load_into_merged_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeKimiModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
            self.model.layers[0].self_attn = torch.nn.Module()
            self.model.layers[0].self_attn.qkv_proj = torch.nn.Module()
            weight = torch.nn.Parameter(torch.empty(6, 4), requires_grad=False)
            calls: list[int] = []

            def weight_loader(param, loaded_weight, shard_id):
                calls.append(shard_id)
                param.data.narrow(0, shard_id * 2, 2).copy_(loaded_weight)

            weight.weight_loader = weight_loader
            self.model.layers[0].self_attn.qkv_proj.register_parameter("weight", weight)
            self.calls = calls
            self.config = SimpleNamespace(
                q_lora_rank=None,
                is_moe=False,
                is_linear_attn=True,
            )

    monkeypatch.setattr(
        kimi_linear,
        "get_spec_layer_idx_from_weight_name",
        lambda config, name: None,
    )
    model = FakeKimiModel()
    shards = [
        (
            f"model.layers.0.self_attn.{name}_proj.weight",
            torch.full((2, 4), value),
        )
        for name, value in (("q", 1.0), ("k", 2.0), ("v", 3.0))
    ]

    loaded = KimiLinearModel.load_weights(model, shards)

    assert model.calls == [0, 1, 2]
    assert loaded == {"model.layers.0.self_attn.qkv_proj.weight"}
    expected = torch.cat([shard for _, shard in shards])
    torch.testing.assert_close(
        model.model.layers[0].self_attn.qkv_proj.weight,
        expected,
    )


def test_mla_separate_q_projection_bypasses_kda_qkv_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeKimiModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
            self.model.layers[0].self_attn = torch.nn.Module()
            self.model.layers[0].self_attn.q_proj = torch.nn.Module()
            weight = torch.nn.Parameter(torch.empty(2, 4), requires_grad=False)

            def weight_loader(param, loaded_weight):
                param.data.copy_(loaded_weight)

            weight.weight_loader = weight_loader
            self.model.layers[0].self_attn.q_proj.register_parameter("weight", weight)
            self.config = SimpleNamespace(
                q_lora_rank=None,
                is_moe=False,
                is_linear_attn=True,
            )

    monkeypatch.setattr(
        kimi_linear,
        "get_spec_layer_idx_from_weight_name",
        lambda config, name: None,
    )
    model = FakeKimiModel()
    source = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    loaded = KimiLinearModel.load_weights(
        model,
        [("model.layers.0.self_attn.q_proj.weight", source)],
    )

    assert loaded == {"model.layers.0.self_attn.q_proj.weight"}
    torch.testing.assert_close(
        model.model.layers[0].self_attn.q_proj.weight,
        source,
    )
