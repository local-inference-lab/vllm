# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn


def _make_config(
    requested: str | None,
    head_k_dim: int = 128,
    *,
    model_type: str = "qwen3_next",
) -> Any:
    return SimpleNamespace(
        additional_config={"gdn_prefill_backend": requested},
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_text_config=SimpleNamespace(
                model_type=model_type,
                linear_key_head_dim=head_k_dim,
                linear_value_head_dim=128,
                linear_num_key_heads=2,
                linear_num_value_heads=6,
            ),
        ),
    )


@pytest.mark.parametrize("model_type", ["qwen3_next", "qwen3_8_flash_next_text"])
@pytest.mark.parametrize(
    "prefill,decode,env_decode,expected_prefill,expected_decode,explicit_decode",
    [
        (None, None, None, None, None, False),
        ("auto", None, None, None, None, False),
        ("b12x", None, None, "b12x", "b12x", False),
        (None, "b12x", None, "b12x", "b12x", True),
        ("b12x", "b12x", None, "b12x", "b12x", True),
        (None, None, "b12x", "b12x", "b12x", True),
        ("flashinfer", None, None, "flashinfer", "cuda", False),
        ("triton", None, None, "triton", "cuda", False),
        ("cutedsl", None, None, "triton", "cuda", False),
        (None, "cuda", None, "flashinfer", "cuda", True),
        (None, None, "triton", "flashinfer", "triton", True),
        (" B12X ", " B12X ", "cuda", "b12x", "b12x", True),
        ("triton", "triton", "b12x", "triton", "triton", True),
    ],
)
def test_gdn_b12x_selection_couples_execution_and_graph_metadata(
    monkeypatch,
    model_type,
    prefill,
    decode,
    env_decode,
    expected_prefill,
    expected_decode,
    explicit_decode,
):
    """Explicit choices beat defaults; CLI/additional config beats the env."""
    from vllm.v1.attention.backend import AttentionCGSupport
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder

    config = _make_config(prefill, model_type=model_type)
    config.additional_config["gdn_decode_kernel"] = decode
    before = config.additional_config.copy()
    if env_decode is None:
        monkeypatch.delenv("VLLM_GDN_DECODE_KERNEL", raising=False)
    else:
        monkeypatch.setenv("VLLM_GDN_DECODE_KERNEL", env_decode)
    platform = MagicMock()
    platform.is_cuda.return_value = True
    platform.is_device_capability.return_value = False
    platform.is_device_capability_family.side_effect = lambda cap: cap == 120
    platform.get_cuda_runtime_major.return_value = 13
    monkeypatch.setattr(qwen_gdn_linear_attn, "current_platform", platform)
    if expected_prefill is None:
        if model_type == "qwen3_8_flash_next_text":
            expected_prefill = expected_decode = "b12x"
        else:
            expected_prefill, expected_decode = "flashinfer", "cuda"

    assert (
        qwen_gdn_linear_attn._resolve_gdn_prefill_backend(config)[1] == expected_prefill
    )
    assert qwen_gdn_linear_attn._resolve_gdn_decode_kernel(config) == (
        expected_decode,
        explicit_decode,
    )
    expected_support = (
        AttentionCGSupport.ALWAYS
        if model_type == "qwen3_8_flash_next_text" and expected_prefill == "b12x"
        else AttentionCGSupport.UNIFORM_BATCH
    )
    assert (
        GDNAttentionMetadataBuilder.get_cudagraph_support(config, None)
        == expected_support
    )
    assert config.additional_config == before


@pytest.mark.parametrize(
    "prefill,decode,env_decode",
    [
        ("b12x", "cuda", None),
        ("b12x", "triton", None),
        ("flashinfer", "b12x", None),
        ("triton", "b12x", None),
        ("cutedsl", "b12x", None),
        ("b12x", None, "triton"),
        ("triton", None, "b12x"),
    ],
)
def test_gdn_explicit_mixed_b12x_backends_fail_closed(
    monkeypatch, prefill, decode, env_decode
):
    config = _make_config(prefill)
    config.additional_config["gdn_decode_kernel"] = decode
    if env_decode is None:
        monkeypatch.delenv("VLLM_GDN_DECODE_KERNEL", raising=False)
    else:
        monkeypatch.setenv("VLLM_GDN_DECODE_KERNEL", env_decode)
    for resolve in (
        qwen_gdn_linear_attn._resolve_gdn_prefill_backend,
        qwen_gdn_linear_attn._resolve_gdn_decode_kernel,
    ):
        with pytest.raises(
            ValueError, match="prefill and decode must be selected together"
        ):
            resolve(config)


def test_gdn_decode_implied_b12x_prefill_rejects_unsupported_geometry(monkeypatch):
    monkeypatch.delenv("VLLM_GDN_DECODE_KERNEL", raising=False)
    config = _make_config("auto", head_k_dim=64)
    config.additional_config["gdn_decode_kernel"] = "b12x"
    platform = MagicMock()
    platform.is_cuda.return_value = True
    platform.is_device_capability_family.side_effect = lambda cap: cap == 120
    monkeypatch.setattr(qwen_gdn_linear_attn, "current_platform", platform)
    with pytest.raises(ValueError, match="b12x GDN prefill requires"):
        qwen_gdn_linear_attn._resolve_gdn_prefill_backend(config)


@pytest.mark.parametrize(
    "sm100,sm120,requested,cuda_runtime,head_k_dim,expected",
    [
        (False, True, "auto", 13, 128, "flashinfer"),
        (False, True, "flashinfer", 13, 128, "flashinfer"),
        (False, True, "cutedsl", 13, 128, "triton"),
        (True, False, "cutedsl", 13, 128, "cutedsl"),
        (False, True, "auto", 12, 128, "triton"),
        (False, True, "auto", 13, 64, "triton"),
    ],
)
def test_resolve_gdn_prefill_backend(
    sm100: bool,
    sm120: bool,
    requested: str,
    cuda_runtime: int,
    head_k_dim: int,
    expected: str,
) -> None:
    platform = MagicMock()
    platform.is_cuda.return_value = True
    platform.is_device_capability.return_value = False
    platform.is_device_capability_family.side_effect = {100: sm100, 120: sm120}.get
    platform.get_cuda_runtime_major.return_value = cuda_runtime

    with patch.object(qwen_gdn_linear_attn, "current_platform", platform):
        _, active_backend = qwen_gdn_linear_attn._resolve_gdn_prefill_backend(
            _make_config(requested, head_k_dim)
        )

    assert active_backend == expected


@pytest.mark.parametrize(
    "sm120,input_dtype,expected_dtype,preserves_storage",
    [
        (True, torch.int32, torch.int64, False),
        (True, torch.int64, torch.int64, True),
        (False, torch.int32, torch.int32, True),
    ],
)
def test_prepare_flashinfer_cu_seqlens(
    sm120: bool,
    input_dtype: torch.dtype,
    expected_dtype: torch.dtype,
    preserves_storage: bool,
) -> None:
    platform = MagicMock()
    platform.is_device_capability_family.return_value = sm120
    cu_seqlens = torch.tensor([0, 64], dtype=input_dtype)

    with patch.object(qwen_gdn_linear_attn, "current_platform", platform):
        result = qwen_gdn_linear_attn._prepare_flashinfer_cu_seqlens(cu_seqlens)

    assert result is not None
    assert result.dtype == expected_dtype
    assert (result is cu_seqlens) == preserves_storage
