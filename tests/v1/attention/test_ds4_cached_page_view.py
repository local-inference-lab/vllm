# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.models.deepseek_v4.nvidia.b12x import DeepseekV4B12xAttention


def test_cached_page_view_returns_tensor_on_miss_and_hit():
    layer = SimpleNamespace(_b12x_cache_page_views={})
    storage = torch.empty((2, 600), dtype=torch.uint8)
    first = DeepseekV4B12xAttention._get_cache_page_view(layer, storage, 1, "SWA")
    second = DeepseekV4B12xAttention._get_cache_page_view(layer, storage, 1, "SWA")

    assert isinstance(first, torch.Tensor)
    assert second is first
    assert first.shape == (2, 584)
    assert first.stride() == (600, 1)
    assert first.data_ptr() == storage.data_ptr()
