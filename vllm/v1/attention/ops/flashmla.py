# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# adapted from: https://github.com/deepseek-ai/FlashMLA/blob/main/flash_mla/flash_mla_interface.py

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

if current_platform.is_cuda():
    try:
        import vllm._flashmla_C  # noqa: F401

        _flashmla_C_AVAILABLE = True
    except ImportError:
        _flashmla_C_AVAILABLE = False
else:
    _flashmla_C_AVAILABLE = False

if current_platform.is_cuda():
    try:
        import vllm._flashmla_extension_C  # noqa: F401

        _flashmla_extension_C_AVAILABLE = True
    except ImportError:
        _flashmla_extension_C_AVAILABLE = False
else:
    _flashmla_extension_C_AVAILABLE = False

# Out-of-tree FlashMLA backend for SM120 (vendored kernels target SM90/SM100 only).
_flashmla_sm120_AVAILABLE = False
if current_platform.is_cuda() and current_platform.is_device_capability_family(120):
    try:
        import flash_mla_sm120  # noqa: F401

        _flashmla_sm120_AVAILABLE = True
    except ImportError:
        _flashmla_sm120_AVAILABLE = False

# Gates DSv4 _forward_prefill between paged FP8 direct (sm120) and the
# bf16 dequant+gather workspace fallback (vendored FlashMLA).
flash_mla_sparse_fwd_supports_fp8_kv = _flashmla_sm120_AVAILABLE


def _is_flashmla_available() -> tuple[bool, str | None]:
    if not _flashmla_C_AVAILABLE:
        return (
            False,
            "vllm._flashmla_C is not available, likely was not "
            "compiled due to insufficient nvcc version or a supported arch "
            "was not in the list of target arches to compile for.",
        )
    if not _flashmla_extension_C_AVAILABLE:
        return (
            False,
            "vllm._flashmla_extension_C is not available, likely "
            "was not compiled due to a build error.",
        )

    return True, None


def is_flashmla_dense_supported() -> tuple[bool, str | None]:
    """
    Return: is_supported_flag, unsupported_reason (optional).
    """
    is_available, maybe_reason = _is_flashmla_available()
    if not is_available:
        return False, maybe_reason
    if not current_platform.is_device_capability_family(90):
        return False, "FlashMLA Dense is only supported on Hopper devices."
    return True, None


def is_flashmla_sparse_supported() -> tuple[bool, str | None]:
    """
    Return: is_supported_flag, unsupported_reason (optional).
    """
    if _flashmla_sm120_AVAILABLE:
        return True, None
    is_available, maybe_reason = _is_flashmla_available()
    if not is_available:
        return False, maybe_reason
    if not (
        current_platform.is_device_capability_family(90)
        or current_platform.is_device_capability_family(100)
    ):
        return (
            False,
            "FlashMLA Sparse is only supported on Hopper and Blackwell devices.",
        )
    return True, None


def _raise_flashmla_unavailable(*_args, **_kwargs):
    _, reason = _is_flashmla_available()
    raise RuntimeError(reason or "FlashMLA is not available")


if _flashmla_sm120_AVAILABLE:
    # BatchSparseMLAPagedAttentionWrapper migrated to flashinfer (PR #3374).
    # Other sm120 primitives stay on flash_mla_sm120 — they have no
    # flashinfer equivalent today (FlashMLASchedMeta, get_mla_metadata,
    # flash_mla_sparse_fwd, flash_mla_with_kvcache).
    from flash_mla_sm120 import (  # noqa: F401
        FlashMLASchedMeta,
        flash_mla_sparse_fwd,
        flash_mla_with_kvcache,
        get_mla_metadata,
    )
    from flashinfer import (  # noqa: F401
        BatchSparseMLAPagedAttentionWrapper,
    )

    flash_attn_varlen_func = _raise_flashmla_unavailable  # type: ignore[assignment]
    flash_attn_varlen_kvpacked_func = _raise_flashmla_unavailable  # type: ignore[assignment]
    flash_attn_varlen_qkvpacked_func = _raise_flashmla_unavailable  # type: ignore[assignment]
elif _is_flashmla_available()[0]:
    from vllm.third_party.flashmla.flash_mla_interface import (  # noqa: F401
        FlashMLASchedMeta,
        flash_attn_varlen_func,
        flash_attn_varlen_kvpacked_func,
        flash_attn_varlen_qkvpacked_func,
        flash_mla_sparse_fwd,
        flash_mla_with_kvcache,
        get_mla_metadata,
    )

    # The vendored FlashMLA backend does not currently ship a plan/run
    # wrapper; DSv4 callers only construct one when on sm120, so this
    # stub just exists so ``from ... import BatchSparseMLAPagedAttentionWrapper``
    # is well-defined in the non-sm120 branch.
    class BatchSparseMLAPagedAttentionWrapper:  # type: ignore[no-redef]
        def __init__(self, *_args, **_kwargs) -> None:
            raise RuntimeError(
                "BatchSparseMLAPagedAttentionWrapper requires flash_mla_sm120 "
                "(sm120 sparse-MLA backend); not available on this device."
            )
else:

    class FlashMLASchedMeta:  # type: ignore[no-redef]
        pass

    class BatchSparseMLAPagedAttentionWrapper:  # type: ignore[no-redef]
        def __init__(self, *_args, **_kwargs) -> None:
            _raise_flashmla_unavailable()

    flash_attn_varlen_func = _raise_flashmla_unavailable  # type: ignore[assignment]
    flash_attn_varlen_kvpacked_func = _raise_flashmla_unavailable  # type: ignore[assignment]
    flash_attn_varlen_qkvpacked_func = _raise_flashmla_unavailable  # type: ignore[assignment]
    flash_mla_sparse_fwd = _raise_flashmla_unavailable  # type: ignore[assignment]
    flash_mla_with_kvcache = _raise_flashmla_unavailable  # type: ignore[assignment]
    get_mla_metadata = _raise_flashmla_unavailable  # type: ignore[assignment]


def get_mla_metadata_dense_fp8(
    cache_seqlens: torch.Tensor,
    num_q_tokens_per_head_k: int,
    num_heads_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _is_flashmla_available()[0]:
        _raise_flashmla_unavailable()
    return torch.ops._flashmla_extension_C.get_mla_decoding_metadata_dense_fp8(
        cache_seqlens,
        num_q_tokens_per_head_k,
        num_heads_k,
    )


def flash_mla_with_kvcache_fp8(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
    descale_q: torch.Tensor | None = None,
    descale_k: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _is_flashmla_available()[0]:
        _raise_flashmla_unavailable()
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    out, softmax_lse = torch.ops._flashmla_extension_C.fwd_kvcache_mla_fp8(
        q,
        k_cache,
        head_dim_v,
        cache_seqlens,
        block_table,
        softmax_scale,
        causal,
        tile_scheduler_metadata,
        num_splits,
        descale_q,
        descale_k,
    )
    return out, softmax_lse
