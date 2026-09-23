# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch import nn

from vllm.models.kimi_k3.nvidia import dspark_mla


class _Rotary(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scaling_factor = 2.0
        self.mscale = 1.25
        self.head_size = 4
        self.is_neox_style = False
        self.register_buffer(
            "cos_sin_cache",
            torch.arange(64, dtype=torch.float32).view(16, 4),
        )

    def _compute_inv_freq(self, scaling_factor: float) -> torch.Tensor:
        assert scaling_factor == self.scaling_factor
        return torch.tensor([0.5, 0.25], dtype=torch.float32)


class _Attention(nn.Module):
    def __init__(self, rotary: _Rotary) -> None:
        super().__init__()
        self.rotary_emb = rotary


class _Layer(nn.Module):
    def __init__(self, rotary: _Rotary) -> None:
        super().__init__()
        self.self_attn = _Attention(rotary)


def _draft_model(rotary: _Rotary) -> dspark_mla.K3DSparkModel:
    model = object.__new__(dspark_mla.K3DSparkModel)
    nn.Module.__init__(model)
    model.layers = nn.ModuleList([_Layer(rotary), _Layer(rotary)])
    model._max_num_context_tokens = 8
    return model


def test_compact_rope_materializes_requested_global_positions() -> None:
    positions = torch.tensor([2, 7], dtype=torch.int64)
    inv_freq = torch.tensor([0.5, 0.25], dtype=torch.float32)
    freqs = torch.empty((4, 2), dtype=torch.float32)
    cache = torch.empty((4, 4), dtype=torch.float32)

    result = dspark_mla._fill_compact_rope_cache(
        positions,
        inv_freq,
        freqs,
        cache,
        mscale=1.25,
    )

    expected_freqs = positions[:, None] * inv_freq[None, :]
    expected = torch.cat((expected_freqs.cos(), expected_freqs.sin()), dim=-1) * 1.25
    torch.testing.assert_close(result, expected)
    assert result.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()


def test_compact_rope_rejects_more_positions_than_workspace() -> None:
    with pytest.raises(ValueError, match="workspace is too small"):
        dspark_mla._fill_compact_rope_cache(
            torch.arange(3),
            torch.ones(2),
            torch.empty((2, 2)),
            torch.empty((2, 4)),
            mscale=1.0,
        )


def test_compact_rope_releases_unshared_full_position_table() -> None:
    rotary = _Rotary()
    original_bytes = rotary.cos_sin_cache.nbytes
    model = _draft_model(rotary)

    model._init_compact_rope()
    model._compact_rope_enabled = True
    remapped_positions, compact_cache = model._get_rope_inputs(
        torch.tensor([2, 7], dtype=torch.int64)
    )

    assert original_bytes == 256
    assert rotary.cos_sin_cache.shape == (0, 4)
    torch.testing.assert_close(remapped_positions, torch.tensor([0, 1]))
    expected_freqs = torch.tensor([[1.0, 0.5], [3.5, 1.75]])
    expected_cache = (
        torch.cat((expected_freqs.cos(), expected_freqs.sin()), dim=-1) * 1.25
    )
    torch.testing.assert_close(compact_cache, expected_cache)


def test_compact_rope_retains_target_owned_position_table() -> None:
    rotary = _Rotary()
    target = nn.Module()
    target.rotary = rotary
    model = _draft_model(rotary)

    with dspark_mla.protect_k3_compact_rope_sources(target):
        model._init_compact_rope()

    assert rotary.cos_sin_cache.shape == (16, 4)
    assert not dspark_mla._COMPACT_ROPE_PROTECTED_IDS
