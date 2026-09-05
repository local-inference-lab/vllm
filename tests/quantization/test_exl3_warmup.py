# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.layers.quantization.exl3 import (
    Exl3MoEMethod,
)


def _layer_with_mixed_trellis(mixed: dict) -> SimpleNamespace:
    return SimpleNamespace(
        exl3_mixed_trellis=mixed,
        exl3_mixed_bitrate=True,
        layer_name="model.layers.3.mlp.experts",
    )


def test_mixed_trellis_warmup_unit_delegates_each_runtime_state() -> None:
    warmup = Mock(side_effect=(6, 12))
    api = SimpleNamespace(warmup_mixed_trellis_route_pack=warmup)
    decode = {"launch": object(), "buffers": object()}
    prefill = {"launch": object(), "buffers": object()}
    expert_map = object()
    layer = _layer_with_mixed_trellis(
        {
            "runtime": {
                "mixed_api": api,
                "decode": decode,
                "prefill": prefill,
            },
            "global_to_combined": expert_map,
        }
    )
    method = object.__new__(Exl3MoEMethod)

    unit = method.get_b12x_warmup_unit(layer, (1, 32), torch.bfloat16)
    unit.compile()

    assert unit.name == "EXL3 mixed-Trellis route pack"
    assert warmup.call_args_list == [
        ((decode["launch"], decode["buffers"]), {"expert_map": expert_map}),
        ((prefill["launch"], prefill["buffers"]), {"expert_map": expert_map}),
    ]


def test_mixed_trellis_warmup_unit_fails_closed_without_profiled_runtime() -> None:
    layer = _layer_with_mixed_trellis({"global_to_combined": object()})
    method = object.__new__(Exl3MoEMethod)

    unit = method.get_b12x_warmup_unit(layer, (1,), torch.bfloat16)

    with pytest.raises(RuntimeError, match="eager profile pass must plan"):
        unit.compile()
