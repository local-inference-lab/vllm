# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import MethodType, SimpleNamespace

import pytest
import torch

from vllm.model_executor.warmup.deepseek_v4_mhc_warmup import _warmup_layer_mhc
from vllm.models.deepseek_v4.nvidia import dspark as dspark_module
from vllm.models.deepseek_v4.nvidia.model import DeepseekV4DecoderLayer


def _make_b12x_layer() -> DeepseekV4DecoderLayer:
    return object.__new__(DeepseekV4DecoderLayer)


def test_b12x_mhc_requires_fused_norm_weight() -> None:
    layer = _make_b12x_layer()

    with pytest.raises(RuntimeError, match="requires fused RMSNorm"):
        layer._require_b12x_mhc_norm_weight(None)

    norm_weight = torch.ones(4)

    assert layer._require_b12x_mhc_norm_weight(norm_weight) is norm_weight


def test_b12x_forward_broadcasts_initial_residual() -> None:
    layer = _make_b12x_layer()
    layer._use_b12x_mhc = True
    layer.hc_mult = 4
    layer.attn_norm = SimpleNamespace(
        weight=SimpleNamespace(data=torch.ones(4)),
        variance_epsilon=1e-6,
    )
    layer.ffn_norm = SimpleNamespace(
        weight=SimpleNamespace(data=torch.ones(4)),
        variance_epsilon=1e-6,
    )
    layer.hc_attn_fn = torch.ones(24, 16)
    layer.hc_attn_fn_broadcast = torch.ones(24, 4)
    layer.hc_attn_scale = torch.ones(3)
    layer.hc_attn_base = torch.zeros(24)
    layer.hc_ffn_fn = torch.ones(24, 16)
    layer.hc_ffn_scale = torch.ones(3)
    layer.hc_ffn_base = torch.zeros(24)
    layer.hc_ffn_fn_bf16 = torch.ones(24, 16, dtype=torch.bfloat16)
    calls: list[tuple[str, torch.Tensor | None]] = []

    def hc_pre(
        self: DeepseekV4DecoderLayer,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        calls.append(("pre", None))
        assert x.shape == (2, 4)
        assert hc_fn is self.hc_attn_fn_broadcast
        residual = x.unsqueeze(1).expand(-1, 4, -1).clone()
        post_mix = torch.zeros(x.shape[0], 4)
        res_mix = torch.zeros(x.shape[0], 4, 4)
        return residual, post_mix, res_mix, x

    def hc_post_pre(
        self: DeepseekV4DecoderLayer,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
        hc_fn_bf16: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        calls.append(("post_pre", hc_fn_bf16))
        return residual, post, comb, x

    layer.hc_pre = MethodType(hc_pre, layer)
    layer.hc_post_pre = MethodType(hc_post_pre, layer)
    layer.attn = lambda positions, x, kv_cache: x
    layer.ffn = lambda x, input_ids: x

    x = torch.ones(2, 4)
    out = DeepseekV4DecoderLayer.forward(layer, x, torch.arange(2), None)

    assert [tuple(t.shape) for t in out] == [
        (2, 4),
        (2, 4, 4),
        (2, 4),
        (2, 4, 4),
    ]
    assert calls == [("pre", None), ("post_pre", layer.hc_ffn_fn_bf16)]


def test_b12x_mhc_warmup_uses_broadcast_pre_contract() -> None:
    hidden_size = 4
    hc_mult = 4
    full_fn = torch.ones(24, hc_mult * hidden_size)
    broadcast_fn = torch.ones(24, hidden_size)
    norm = SimpleNamespace(
        weight=SimpleNamespace(data=torch.ones(hidden_size)),
        variance_epsilon=1e-6,
    )
    pre_sizes: list[int] = []
    post_sizes: list[int] = []

    def hc_pre(
        x: torch.Tensor,
        fn: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
        *,
        norm_weight: torch.Tensor,
        norm_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del scale, base, norm_weight, norm_eps
        assert fn is broadcast_fn
        assert x.shape == (x.shape[0], hidden_size)
        pre_sizes.append(x.shape[0])
        residual = x.unsqueeze(1).expand(-1, hc_mult, -1).clone()
        post = torch.zeros(x.shape[0], hc_mult)
        comb = torch.zeros(x.shape[0], hc_mult, hc_mult)
        return residual, post, comb, x

    def hc_post_pre(
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        fn: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
        *,
        norm_weight: torch.Tensor,
        norm_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del fn, scale, base, norm_weight, norm_eps
        assert residual.shape == (x.shape[0], hc_mult, hidden_size)
        post_sizes.append(x.shape[0])
        return residual, post, comb, x

    layer = SimpleNamespace(
        _use_b12x_mhc=True,
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        hc_attn_fn=full_fn,
        hc_attn_fn_broadcast=broadcast_fn,
        hc_attn_scale=torch.ones(3),
        hc_attn_base=torch.zeros(24),
        hc_ffn_fn=full_fn.clone(),
        hc_ffn_scale=torch.ones(3),
        hc_ffn_base=torch.zeros(24),
        attn_norm=norm,
        ffn_norm=norm,
        hc_pre=hc_pre,
        hc_post_pre=hc_post_pre,
    )

    _warmup_layer_mhc(layer, [1, 3])

    assert pre_sizes == [1, 3]
    assert post_sizes == [1, 1, 3, 3]


def test_b12x_dspark_keeps_initial_embeddings_rank_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs_embeds = torch.ones(2, 4)

    class FakeB12xLayer(torch.nn.Module):
        _use_b12x_mhc = True

        def __init__(self) -> None:
            super().__init__()
            self.hidden_size = 4
            self.hc_mult = 4
            self.hc_attn_fn = torch.arange(24 * 16).reshape(24, 16)
            self.hc_attn_fn_broadcast = None
            self.refreshed = False

        def refresh_b12x_mhc_bf16_weights(self) -> None:
            self.refreshed = True

        def forward(
            self,
            hidden_states: torch.Tensor,
            positions: torch.Tensor,
            input_ids: torch.Tensor,
            post_mix: torch.Tensor | None,
            res_mix: torch.Tensor | None,
            residual: torch.Tensor | None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            del positions, input_ids
            assert hidden_states is inputs_embeds
            assert post_mix is None
            assert res_mix is None
            assert residual is None
            residual = hidden_states.unsqueeze(1).expand(-1, 4, -1).clone()
            post_mix = torch.zeros(hidden_states.shape[0], 4)
            res_mix = torch.zeros(hidden_states.shape[0], 4, 4)
            return hidden_states, residual, post_mix, res_mix

    model = object.__new__(dspark_module.DSparkDeepseekV4Model)
    torch.nn.Module.__init__(model)
    model.hc_mult = 4
    model.hc_eps = 1e-6
    model.rms_norm_eps = 1e-6
    model.layers = torch.nn.ModuleList([FakeB12xLayer()])
    model.hc_head_fn = torch.ones(4, 16)
    model.hc_head_scale = torch.ones(1)
    model.hc_head_base = torch.zeros(4)
    model.finalize_mhc_weights()

    def mhc_post(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        res_mix: torch.Tensor,
    ) -> torch.Tensor:
        del hidden_states, post_mix, res_mix
        return residual

    def hc_head(hidden_states: torch.Tensor, *args: object) -> torch.Tensor:
        del args
        return hidden_states.mean(dim=1)

    monkeypatch.setattr(dspark_module, "mhc_post_tilelang", mhc_post)
    monkeypatch.setattr(dspark_module, "hc_head_fused_kernel_tilelang", hc_head)

    output = model(torch.arange(2), torch.arange(2), inputs_embeds)

    assert output.shape == inputs_embeds.shape
    layer = model.layers[0]
    assert isinstance(layer, FakeB12xLayer)
    assert layer.refreshed
    expected_broadcast = layer.hc_attn_fn.view(24, 4, 4).sum(dim=1)
    torch.testing.assert_close(layer.hc_attn_fn_broadcast, expected_broadcast)
