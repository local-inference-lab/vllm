# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import SimpleNamespace

from vllm.model_executor.warmup.kimi_k3_triton_warmup import _get_kda_layer


def test_kda_warmup_uses_only_loaded_model_modules(monkeypatch):
    module_name = "vllm.models.kimi_k3.nvidia.kda"
    layer = SimpleNamespace()
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context={"kda": layer})
        )
    )
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    assert _get_kda_layer(worker) is None
    assert module_name not in sys.modules

    monkeypatch.setitem(
        sys.modules, module_name, SimpleNamespace(KimiK3DeltaAttention=SimpleNamespace)
    )
    assert _get_kda_layer(worker) is layer
