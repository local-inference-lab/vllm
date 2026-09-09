# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 DSpark draft pieces the lightseek checkpoint needs: per-tap
``fc_norm`` before the context projection and the confidence head.

CPU only. The confidence-head test loads the head's two tensors from the
checkpoint when it is mounted at ``KIMI_K3_DSPARK_CHECKPOINT`` (default
``/mnt/models/Kimi-K3-DSpark-lightseek``) and is skipped otherwise.
"""

import json
import os
import struct
from types import SimpleNamespace

import pytest
import torch
from torch import nn

CHECKPOINT = os.environ.get(
    "KIMI_K3_DSPARK_CHECKPOINT", "/mnt/models/Kimi-K3-DSpark-lightseek"
)


def _rms_norm_reference(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (y * weight.float()).to(x.dtype)


class _RMSNorm(nn.Module):
    """Plain-torch stand-in for vLLM's RMSNorm CustomOp (no engine config)."""

    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _rms_norm_reference(x, self.weight, self.eps)


class _Linear(nn.Module):
    """Plain-torch stand-in for ReplicatedLinear (weight ``[out, in]``)."""

    def __init__(
        self,
        input_size,
        output_size,
        bias=False,
        return_bias=True,
        params_dtype=None,
        prefix="",
        **_,
    ):
        super().__init__()
        dtype = params_dtype or torch.float32
        self.weight = nn.Parameter(torch.zeros(output_size, input_size, dtype=dtype))
        self.bias = (
            nn.Parameter(torch.zeros(output_size, dtype=dtype)) if bias else None
        )
        self.return_bias = return_bias

    def forward(self, x: torch.Tensor):
        out = torch.nn.functional.linear(
            x.to(self.weight.dtype), self.weight, self.bias
        )
        return (out, None) if self.return_bias else out


def test_normalize_taps_matches_per_tap_rmsnorm():
    """``normalize_taps`` applies norm ``i`` to tap ``i`` of the concatenated
    state and leaves the tap order unchanged."""
    from vllm.models.kimi_k3.nvidia.dspark_mla import K3DSparkModel

    taps, width, tokens, eps = 5, 16, 7, 1e-6
    norms = torch.nn.ModuleList([_RMSNorm(width, eps=eps) for _ in range(taps)])
    for i, norm in enumerate(norms):
        norm.weight.data = (
            0.5 + 0.1 * i + 0.01 * torch.arange(width, dtype=torch.float32)
        )
    model = SimpleNamespace(
        fc_norm=norms,
        config=SimpleNamespace(target_hidden_size=width, num_target_layers=taps),
    )
    x = torch.randn(tokens, taps * width, dtype=torch.bfloat16)
    out = K3DSparkModel.normalize_taps(model, x)
    assert out.shape == x.shape
    for i in range(taps):
        ref = _rms_norm_reference(
            x[:, i * width : (i + 1) * width], norms[i].weight, eps
        )
        torch.testing.assert_close(
            out[:, i * width : (i + 1) * width], ref, atol=1e-2, rtol=1e-2
        )
    # Streamed path: one tap at a time, same numbers.
    for i in range(taps):
        tap = K3DSparkModel.normalize_tap(
            model, i, x[:, i * width : (i + 1) * width].contiguous()
        )
        torch.testing.assert_close(tap, out[:, i * width : (i + 1) * width])
    # Without fc_norm the state passes through untouched.
    plain = SimpleNamespace(fc_norm=None, config=model.config)
    assert K3DSparkModel.normalize_taps(plain, x) is x
    with pytest.raises(ValueError):
        K3DSparkModel.normalize_taps(model, x[:, : (taps - 1) * width])


def _load_tensors(path: str, names: list[str]) -> dict[str, torch.Tensor]:
    dtypes = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        out = {}
        for name in names:
            meta = header[name]
            start, end = meta["data_offsets"]
            f.seek(8 + header_len + start)
            raw = f.read(end - start)
            out[name] = torch.frombuffer(
                bytearray(raw), dtype=dtypes[meta["dtype"]]
            ).reshape(meta["shape"])
    return out


@pytest.mark.skipif(
    not os.path.exists(os.path.join(CHECKPOINT, "model.safetensors")),
    reason="lightseek Kimi-K3 DSpark checkpoint not mounted",
)
def test_confidence_head_matches_checkpoint_math(monkeypatch):
    """The head with the checkpoint's weights computes w^T [h; e] + b in fp32
    for hidden width 7,168 and Markov rank 256 (input 7,424)."""
    with open(os.path.join(CHECKPOINT, "config.json")) as cfg_file:
        cfg = json.load(cfg_file)
    assert cfg["enable_confidence_head"] and cfg["confidence_head_with_markov"]
    hidden, rank = cfg["hidden_size"], cfg["markov_rank"]
    tensors = _load_tensors(
        os.path.join(CHECKPOINT, "model.safetensors"),
        ["confidence_head.proj.weight", "confidence_head.proj.bias"],
    )
    weight, bias = (
        tensors["confidence_head.proj.weight"],
        tensors["confidence_head.proj.bias"],
    )
    assert tuple(weight.shape) == (1, hidden + rank) and tuple(bias.shape) == (1,)
    from vllm.model_executor.models import qwen3_dspark

    monkeypatch.setattr(qwen3_dspark, "ReplicatedLinear", _Linear)
    head = qwen3_dspark.DSparkConfidenceHead(
        hidden + rank, prefix="confidence_head", bias=True, include_markov=True
    )
    head.proj.weight.data.copy_(weight.float())
    head.proj.bias.data.copy_(bias.float())
    torch.manual_seed(0)
    h = torch.randn(4, hidden, dtype=torch.bfloat16)
    e = torch.randn(4, rank, dtype=torch.bfloat16)
    got = head(h, e)
    ref = torch.cat([h, e], dim=-1).float() @ weight.float().t() + bias.float()
    assert got.shape == (4,)
    torch.testing.assert_close(got, ref.squeeze(-1), atol=1e-3, rtol=1e-4)
    # The wrapper's compute_confidence delegates to the head and reports None
    # without one.
    from vllm.models.kimi_k3.nvidia.dspark_mla import K3DSparkForCausalLM

    with_head = SimpleNamespace(model=SimpleNamespace(confidence_head=head))
    torch.testing.assert_close(
        K3DSparkForCausalLM.compute_confidence(with_head, h, e), got
    )
    without = SimpleNamespace(model=SimpleNamespace(confidence_head=None))
    assert K3DSparkForCausalLM.compute_confidence(without, h, e) is None


def _projection_model(taps: int, width: int, hidden: int, eps: float, plain: bool):
    """DSpark draft stand-in with per-tap norms, a plain context projection
    and the context norm; ``plain=False`` marks the projection weight as
    quantized so the concatenated fallback path must be taken."""
    torch.manual_seed(0)
    norms = torch.nn.ModuleList([_RMSNorm(width, eps=eps) for _ in range(taps)])
    for i, norm in enumerate(norms):
        norm.weight.data = (
            0.5 + 0.1 * i + 0.01 * torch.arange(width, dtype=torch.float32)
        )
    proj = _Linear(taps * width, hidden, return_bias=False)
    proj.weight.data = torch.randn(hidden, taps * width) * 0.05
    if not plain:
        proj.quant_method = object()
    context_norm = _RMSNorm(hidden, eps=eps)
    context_norm.weight.data = 1.0 + 0.01 * torch.arange(hidden, dtype=torch.float32)
    return _DraftProjection(
        fc_norm=norms,
        context_proj=proj,
        context_norm=context_norm,
        context_proj_sharded=False,
        config=SimpleNamespace(target_hidden_size=width, num_target_layers=taps),
    )


class _DraftProjection(SimpleNamespace):
    """Attribute bag carrying the draft model's projection methods."""

    def __getattr__(self, name):
        from vllm.models.kimi_k3.nvidia.dspark_mla import K3DSparkModel

        method = getattr(K3DSparkModel, name, None)
        if method is None or not callable(method):
            raise AttributeError(name)
        return method.__get__(self, type(self))


@pytest.mark.parametrize("plain", [True, False])
def test_combine_tap_states_matches_concatenated_projection(plain):
    """``combine_tap_states`` (per-tap accumulation, no concatenation) equals
    ``combine_hidden_states`` on the concatenated taps up to accumulation
    order, and takes the concatenated path verbatim for quantized weights."""
    from vllm.models.kimi_k3.nvidia.dspark_mla import K3DSparkModel

    taps, width, hidden, tokens, eps = 5, 16, 24, 7, 1e-6
    model = _projection_model(taps, width, hidden, eps, plain)
    tap_states = [torch.randn(tokens, width) * (1.0 + i) for i in range(taps)]
    expected = K3DSparkModel.combine_hidden_states(model, torch.cat(tap_states, dim=-1))
    actual = K3DSparkModel.combine_tap_states(model, tap_states)
    assert actual.shape == expected.shape == (tokens, hidden)
    if plain:
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    else:
        assert torch.equal(actual, expected)


def test_combine_tap_states_rejects_wrong_tap_count():
    from vllm.models.kimi_k3.nvidia.dspark_mla import K3DSparkModel

    model = _projection_model(5, 16, 24, 1e-6, True)
    with pytest.raises(ValueError, match="expects 5 target taps"):
        K3DSparkModel.combine_tap_states(model, [torch.randn(3, 16)] * 4)
    with pytest.raises(ValueError, match="shape"):
        K3DSparkModel.combine_tap_states(model, [torch.randn(3, 8)] * 5)
