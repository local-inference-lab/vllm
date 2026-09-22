# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from vllm.models.kimi_k3.nvidia import l2_prefetch


def test_decode_prefetch_bounds_rows_and_retains_weights(monkeypatch):
    """Cache hints must not execute for prefill or outlive their weight owner."""
    hints = l2_prefetch.cache_hints
    stream = Mock()
    factory = Mock(return_value=stream)
    monkeypatch.setattr(hints, "L2Prefetcher", factory)
    segment_calls = []

    def segments(module, prefix, **kwargs):
        segment_calls.append(prefix)
        return [(prefix, id(module), 1024)]

    monkeypatch.setattr(hints, "segments_of", segments)
    monkeypatch.setattr(
        hints, "tensor_segment", lambda name, value: (name, id(value), 1024)
    )
    plan = Mock(
        side_effect=lambda ranges, budget, device: (SimpleNamespace(ranges=ranges), [])
    )
    monkeypatch.setattr(hints, "make_plan", plan)
    layers = []
    for _ in range(2):
        layer = torch.nn.Module()
        layer.register_parameter("weight", torch.nn.Parameter(torch.ones(4)))
        layer.self_attn = SimpleNamespace(
            o_proj=SimpleNamespace(), in_proj_qkv=SimpleNamespace()
        )
        layer.mlp = SimpleNamespace(
            _paired_decode_weight=torch.ones(4),
            shared_experts=SimpleNamespace(),
            experts=SimpleNamespace(runner=SimpleNamespace()),
        )
        layers.append(layer)
    owner = l2_prefetch.KimiDecodePrefetch(layers, torch.device("cpu"))
    factory.assert_called_once_with(torch.device("cpu"), persisting_l2_request="0")
    assert [call.args[1] for call in plan.call_args_list] == [
        24 * 1024**2,
        48 * 1024**2,
        24 * 1024**2,
    ]
    assert len(owner.plans) == len(owner.hooks) == 3
    assert any(weight is layers[0].weight for weight in owner.weights)
    assert any(
        weight is layers[0].mlp._paired_decode_weight for weight in owner.weights
    )
    for target, callback in owner.hooks:
        assert target._l2_prefetch_pre_reduce_hook is callback
        callback(1)
        callback(8)
        callback(9)
        callback(4096)
    assert stream.issue.call_count == 6
    assert "in_proj_qkv." in segment_calls and "shared." in segment_calls
    targets = [target for target, _ in owner.hooks]
    owner.close()
    stream.join.assert_called_once()
    assert not owner.hooks
    assert all(
        not hasattr(target, "_l2_prefetch_pre_reduce_hook") for target in targets
    )


def test_disabled_prefetch_releases_previous_hooks(monkeypatch):
    """Reload invalidates the address plans even when cache hints are disabled."""
    from vllm import envs

    previous = Mock()
    model = SimpleNamespace(_decode_prefetch=previous)
    monkeypatch.setattr(envs, "VLLM_KIMI_L2_PREFETCH", False)
    l2_prefetch.prepare_decode_prefetch(model)
    previous.close.assert_called_once()
    assert model._decode_prefetch is None
