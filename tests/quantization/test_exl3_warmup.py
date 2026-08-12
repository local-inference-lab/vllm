from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.layers.quantization.exl3 import (
    warmup_exl3_mixed_trellis_route_pack,
)


def _model_with_mixed_trellis(mixed: dict) -> torch.nn.Module:
    model = torch.nn.Module()
    holder = torch.nn.Module()
    holder.routed_experts = SimpleNamespace(
        exl3_mixed_trellis=mixed,
        layer_name="model.layers.3.mlp.experts",
    )
    model.add_module("holder", holder)
    return model


def test_mixed_trellis_warmup_delegates_each_runtime_state() -> None:
    warmup = Mock(side_effect=(6, 12))
    api = SimpleNamespace(warmup_mixed_trellis_route_pack=warmup)
    decode = {"launch": object(), "buffers": object()}
    prefill = {"launch": object(), "buffers": object()}
    expert_map = object()
    model = _model_with_mixed_trellis(
        {
            "runtime": {
                "mixed_api": api,
                "decode": decode,
                "prefill": prefill,
            },
            "global_to_combined": expert_map,
        }
    )

    assert warmup_exl3_mixed_trellis_route_pack(model) == 18
    assert warmup.call_args_list == [
        ((decode["launch"], decode["buffers"]), {"expert_map": expert_map}),
        ((prefill["launch"], prefill["buffers"]), {"expert_map": expert_map}),
    ]


def test_mixed_trellis_warmup_fails_closed_without_profiled_runtime() -> None:
    model = _model_with_mixed_trellis({"global_to_combined": object()})

    with pytest.raises(RuntimeError, match="eager profile pass must plan"):
        warmup_exl3_mixed_trellis_route_pack(model)


def test_mixed_trellis_warmup_ignores_unrelated_models() -> None:
    assert warmup_exl3_mixed_trellis_route_pack(torch.nn.Linear(4, 4)) == 0
