# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MLA prefill backend fused-quant-output support.

Covers two things:
  * `MLAPrefillBackend.supports_quant_output`, the capability gate that decides
    whether the prefill kernel writes quantized output directly (FA4 native
    fused FP8, see flash-attention#135) instead of the post-quant path.
  * The numerical equivalence of that fused FP8 write versus the bf16-attention
    + standalone static-FP8-quant path it replaces (GPU-only, SM100/SM110).
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
)
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla.prefill.b12x import (
    B12xContextPrefillBackend,
    B12xPrefillBackend,
)
from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend
from vllm.v1.attention.backends.mla.prefill.flash_attn import (
    FlashAttnPrefillBackend,
)

_FA_MODULE = "vllm.v1.attention.backends.mla.prefill.flash_attn"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "heads,capacity", [(6, 257), (11, 257), (12, 257), (11, 76032)]
)
@pytest.mark.parametrize("dcp_head_axis", [False, True])
@torch.inference_mode()
def test_b12x_context_projection_reuses_scratch_and_preserves_attention(
    heads, capacity, dcp_head_axis, monkeypatch
):
    """Ragged live chunks and graph replay retain exact BF16 K/V and attention."""
    from b12x.preparation import PreparationSession

    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonMetadataBuilder,
    )
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.v1.worker import workspace

    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("requires SM12x")
    device = torch.device("cuda", torch.accelerator.current_device_index())
    manager = workspace.WorkspaceManager(device)
    monkeypatch.setattr(workspace, "_manager", manager)
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    query_rows = 16
    monkeypatch.setattr(
        MLACommonMetadataBuilder,
        "determine_chunked_prefill_workspace_size",
        staticmethod(lambda _: capacity),
    )
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=query_rows, max_num_seqs=1
        ),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
    )
    backend = B12xPrefillBackend(heads, 192**-0.5, 512, 128, 64, 128, config)
    projection = torch.nn.Linear(
        512, heads * 256, bias=False, device=device, dtype=torch.bfloat16
    )
    projection.quant_method = UnquantizedLinearMethod()
    owner = SimpleNamespace(kv_b_proj=projection, layer_name="test.context_projection")
    units = backend.get_b12x_preparation_units(owner, SimpleNamespace(stage="weights"))
    latent_storage = (
        torch.randn(capacity, 576, device=device, dtype=torch.bfloat16) * 0.1
    )
    q = torch.randn(query_rows, heads, 192, device=device, dtype=torch.bfloat16) * 0.1
    cu_q = torch.tensor([0, query_rows], device=device, dtype=torch.int32)
    cu_k = torch.tensor([0, capacity], device=device, dtype=torch.int32)

    with PreparationSession(device=device, autotune=False) as session:
        session.prepare(tuple(request for unit in units for request in unit.requests))
        manager.lock()
        address = manager._current_workspaces[0].data_ptr()
        for rows in (1, 65, capacity, 7):
            latent = latent_storage[:rows, :512]
            if dcp_head_axis:
                latent = latent.unsqueeze(1)
            rope = latent_storage[:rows, 512:].unsqueeze(1)
            cu_k[1] = rows

            def reference(latent=latent, rows=rows, rope=rope):
                kv = projection(latent).view(rows, heads, 256)
                k = torch.cat((kv[..., :128], rope.expand(-1, heads, -1)), dim=-1)
                return backend._run(
                    q, k, kv[..., 128:], cu_q, cu_k, query_rows, rows, False, None
                )

            def bounded(latent=latent, rows=rows, rope=rope):
                k, v = backend.project_context_kv(projection, latent, rope)
                return backend._run(q, k, v, cu_q, cu_k, query_rows, rows, False, None)

            kv = projection(latent).view(rows, heads, 256)
            k, v = backend.project_context_kv(projection, latent, rope)
            torch.testing.assert_close(k[..., :128], kv[..., :128], atol=0, rtol=0)
            torch.testing.assert_close(
                k[..., 128:], rope.expand(-1, heads, -1), atol=0, rtol=0
            )
            torch.testing.assert_close(v, kv[..., 128:], atol=0, rtol=0)
            pointers = (k.data_ptr(), v.data_ptr())
            reference()
            bounded()
            graph = torch.cuda.CUDAGraph()
            try:
                with session.capture(), torch.cuda.graph(graph):
                    actual = bounded()
                for _ in range(3):
                    latent_storage.normal_(std=0.1)
                    expected = reference()
                    graph.replay()
                    torch.accelerator.synchronize()
                    for got, want in zip(actual, expected):
                        assert torch.isfinite(got).all() and torch.count_nonzero(got)
                        torch.testing.assert_close(got, want, atol=0, rtol=0)
                    k, v = backend.project_context_kv(projection, latent, rope)
                    assert (k.data_ptr(), v.data_ptr()) == pointers
                    assert manager._current_workspaces[0].data_ptr() == address
            finally:
                graph.reset()
        with pytest.raises(ValueError, match="prepared capacity"):
            backend.project_context_kv(
                projection,
                torch.empty(capacity + 1, 512, device=device, dtype=torch.bfloat16),
                torch.empty(capacity + 1, 1, 64, device=device, dtype=torch.bfloat16),
            )


def test_b12x_context_projection_respects_linear_dispatch(monkeypatch):
    """Caller scratch must not bypass quantization, hooks or alternate GEMMs."""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    layer = torch.nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
    layer.quant_method = UnquantizedLinearMethod()
    assert B12xPrefillBackend._can_project_context(layer)
    handle = layer.register_forward_hook(lambda *_: None)
    assert not B12xPrefillBackend._can_project_context(layer)
    handle.remove()
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    assert not B12xPrefillBackend._can_project_context(layer)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    layer.quant_method._gemm_impl = lambda *_: None
    assert not B12xPrefillBackend._can_project_context(layer)


@pytest.mark.parametrize("provide_out", [False, True])
def test_b12x_context_preserves_causal_values_and_compact_output(provide_out):
    """FA2's padded result must preserve values and honor caller-owned output."""
    backend = object.__new__(B12xContextPrefillBackend)
    backend.scale = 0.125
    backend.v_head_dim = 128
    padded = torch.arange(2 * 3 * 192, dtype=torch.float32).view(2, 3, 192)
    lse = torch.ones(3, 2)

    def causal(**kwargs):
        assert kwargs["causal"] and kwargs["return_softmax_lse"]
        assert "out" not in kwargs
        assert kwargs["softmax_scale"] == backend.scale
        return padded, lse

    backend._causal_backend = SimpleNamespace(_flash_attn_varlen_diff_headdims=causal)
    out = torch.full((2, 3, 128), float("nan")) if provide_out else None
    value, actual_lse = backend._run(None, None, None, None, None, 2, 2, True, out)
    torch.testing.assert_close(value, padded[..., :128], rtol=0, atol=0)
    assert value.is_contiguous() and actual_lse is lse
    if provide_out:
        assert value is out
    with patch.object(B12xPrefillBackend, "_run", return_value=(value, lse)) as context:
        assert backend._run(None, None, None, None, None, 2, 2, False, out)[0] is value
        context.assert_called_once()


class _DummyPrefillBackend(MLAPrefillBackend):
    """Concrete backend that does NOT override supports_quant_output."""

    @staticmethod
    def get_name() -> str:
        return "DUMMY"

    def run_prefill_new_tokens(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError

    def run_prefill_context_chunk(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


@pytest.mark.parametrize(
    "quant_key", [kFp8StaticTensorSym, kFp8Dynamic128Sym, kNvfp4Dynamic, None]
)
def test_base_backend_never_supports_quant_output(quant_key):
    """The base default opts every backend out unless it overrides."""
    backend = object.__new__(_DummyPrefillBackend)
    assert backend.supports_quant_output(quant_key) is False


def _make_fa_backend(version: int | None, is_vllm_fa: bool):
    """Build a FlashAttnPrefillBackend without running its heavy __init__."""
    backend = object.__new__(FlashAttnPrefillBackend)
    backend.vllm_flash_attn_version = version
    backend._is_vllm_fa = is_vllm_fa
    return backend


@pytest.mark.parametrize(
    ("version", "is_vllm_fa", "dc_major", "quant_key", "expected"),
    [
        # FA4 + vLLM-FA + Blackwell SM100/SM110 + static FP8 -> fused.
        (4, True, 10, kFp8StaticTensorSym, True),
        (4, True, 11, kFp8StaticTensorSym, True),
        # Wrong compute capability (SM90 / SM120) -> not supported (#135).
        (4, True, 9, kFp8StaticTensorSym, False),
        (4, True, 12, kFp8StaticTensorSym, False),
        # Not FA4.
        (3, True, 10, kFp8StaticTensorSym, False),
        (2, True, 10, kFp8StaticTensorSym, False),
        (None, True, 10, kFp8StaticTensorSym, False),
        # Upstream (ROCm) flash-attn, not vLLM-FA.
        (4, False, 10, kFp8StaticTensorSym, False),
        # Quant keys not wired through FA4 yet.
        (4, True, 10, kFp8Dynamic128Sym, False),
        (4, True, 10, kNvfp4Dynamic, False),
    ],
)
def test_flash_attn_supports_quant_output(
    version, is_vllm_fa, dc_major, quant_key, expected
):
    backend = _make_fa_backend(version, is_vllm_fa)
    with patch(f"{_FA_MODULE}.current_platform") as plat:
        plat.get_device_capability.return_value = DeviceCapability(
            major=dc_major, minor=0
        )
        assert backend.supports_quant_output(quant_key) is expected


def test_flash_attn_supports_quant_output_unknown_device():
    """A None device capability (e.g. capability probe failed) is safe."""
    backend = _make_fa_backend(version=4, is_vllm_fa=True)
    with patch(f"{_FA_MODULE}.current_platform") as plat:
        plat.get_device_capability.return_value = None
        assert backend.supports_quant_output(kFp8StaticTensorSym) is False


@pytest.mark.parametrize(
    ("version", "enable_jit_warmup", "expected_calls"),
    [(4, True, 1), (4, False, 0), (3, True, 0)],
)
def test_flash_attn_registers_warmup_only_for_fa4(
    version: int,
    enable_jit_warmup: bool,
    expected_calls: int,
):
    vllm_config = SimpleNamespace(
        kernel_config=SimpleNamespace(enable_jit_warmup=enable_jit_warmup)
    )
    with (
        patch(f"{_FA_MODULE}.flash_attn_varlen_func"),
        patch(f"{_FA_MODULE}.get_flash_attn_version", return_value=version),
        patch(f"{_FA_MODULE}._FA4_MLA_PREFILL_KERNEL.register_warmup") as register,
        patch(f"{_FA_MODULE}.current_platform") as platform,
    ):
        platform.get_device_capability.return_value = DeviceCapability(10, 0)
        FlashAttnPrefillBackend(
            num_heads=16,
            scale=1.0,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            vllm_config=vllm_config,
        )

    assert register.call_count == expected_calls


def test_flash_attn_prefill_backend_signature_accepts_fused_kwargs():
    """run_prefill_new_tokens must accept out/output_scale so the direct
    (non-**kwargs) call in forward_mha type- and runtime-checks."""
    import inspect

    params = inspect.signature(
        FlashAttnPrefillBackend.run_prefill_new_tokens
    ).parameters
    assert "out" in params
    assert "output_scale" in params
    # The base contract must expose them too (Liskov / direct call site).
    base_params = inspect.signature(MLAPrefillBackend.run_prefill_new_tokens).parameters
    assert "out" in base_params
    assert "output_scale" in base_params


def test_mla_impl_forward_mha_accepts_output_scale():
    """The abstract MLA impl forward_mha must carry output_scale so every
    override (and the unconditional forward_impl call) stays compatible."""
    import inspect

    from vllm.v1.attention.backend import MLAAttentionImpl

    params = inspect.signature(MLAAttentionImpl.forward_mha).parameters
    assert "output_scale" in params
    assert params["output_scale"].default is None


def _fused_fp8_skip_reason() -> str | None:
    """FA4 fused FP8 output needs a real Blackwell SM100/SM110 GPU."""
    if not torch.cuda.is_available():
        return "requires CUDA"
    major = torch.cuda.get_device_capability()[0]
    if major not in (10, 11):
        return f"FA4 fused FP8 output requires SM100/SM110, got SM{major}x"
    return None


_FUSED_FP8_SKIP = _fused_fp8_skip_reason()


@pytest.mark.skipif(_FUSED_FP8_SKIP is not None, reason=_FUSED_FP8_SKIP or "")
def test_fa4_fused_fp8_output_matches_post_quant(default_vllm_config):
    """FA4's fused FP8 write (output_scale, flash-attention#135) must match the
    bf16-attention + standalone static-FP8-quant path it replaces, since
    production uses the same output_scale for both."""
    from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
    from vllm.platforms import current_platform
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    torch.manual_seed(0)
    device = torch.device("cuda")
    fp8_dtype = current_platform.fp8_dtype()

    # MLA prefill head dims (post kv_b_proj): q/k = qk_nope(128)+qk_rope(64),
    # v = v_head_dim(128); DeepSeek-V2-Lite has 16 query heads.
    num_heads, qk_head_dim, v_head_dim, seqlen = 16, 192, 128, 512
    cu_seqlens = torch.tensor([0, seqlen], dtype=torch.int32, device=device)
    q = torch.randn(seqlen, num_heads, qk_head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(seqlen, num_heads, qk_head_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn(seqlen, num_heads, v_head_dim, dtype=torch.bfloat16, device=device)

    fa_kwargs = dict(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=seqlen,
        max_seqlen_k=seqlen,
        causal=True,
        fa_version=4,
    )

    # Reference: bf16 attention, then standalone static per-tensor FP8 quant.
    out_bf16 = flash_attn_varlen_func(q=q, k=k, v=v, **fa_kwargs)
    out_2d = out_bf16.reshape(seqlen, num_heads * v_head_dim)
    # Scale the amax near e4m3 max so the check uses the representable range.
    finfo = torch.finfo(fp8_dtype)
    scale = (out_2d.abs().max() / finfo.max).to(torch.float32).reshape(1)
    quant_op = QuantFP8(static=True, group_shape=GroupShape.PER_TENSOR)
    ref_fp8, _ = quant_op(out_2d, scale)

    # Feature: FA4 writes e4m3 into the (tokens, heads*dim) buffer directly.
    fused_fp8 = torch.empty(
        seqlen, num_heads * v_head_dim, dtype=fp8_dtype, device=device
    )
    flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        out=fused_fp8.view(seqlen, num_heads, v_head_dim),
        output_scale=scale,
        **fa_kwargs,
    )

    # Non-degenerate (catches a no-op / all-zero write).
    assert torch.isfinite(fused_fp8.float()).all()
    assert fused_fp8.float().abs().any()

    # e4m3 has 3 mantissa bits, so allow ~1 mantissa step of rounding slack.
    ref = ref_fp8.float() * scale
    got = fused_fp8.float() * scale
    torch.testing.assert_close(got, ref, rtol=0.125, atol=float(scale) * 2)

    # ...and most elements land in the exact same fp8 bucket.
    exact = (fused_fp8.view(torch.uint8) == ref_fp8.view(torch.uint8)).float().mean()
    assert exact > 0.9, f"only {exact:.1%} of fused FP8 outputs matched the baseline"
