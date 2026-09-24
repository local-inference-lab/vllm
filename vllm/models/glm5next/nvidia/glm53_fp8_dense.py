# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash TP3 FP8 weight-only decode for the large BF16 projections.

``modelopt_mixed`` keeps the attention and KDA projections, the dense FFN and
the LM head in BF16. At decode batch sizes those GEMMs stream their weights,
so halving the bytes nearly halves their time. Measured per TP3 rank at M=4
(RTX PRO 6000, weights from HBM): KDA ``in_proj`` 44.4 -> 26.9 us, DSA
``o_proj`` 31.8 -> 19.3 us, KDA ``o_proj`` 16.5 -> 10.7 us, LM head
261 -> 137 us. Small projections are slower under Marlin and stay BF16, as
do the shared-expert projections and the DSA ``fused_qkv_a_proj``.

After loading, each selected weight is quantized to FP8 E4M3 with one scale per
output channel and packed for the Marlin W8A16 kernel (activations stay BF16).
Batches of at most ``VLLM_GLM53_FP8_DENSE_MAX_M`` rows (default 32, every
decode and verify step up to 8 sequences x 4 positions) use it. Larger batches
use the original BF16 GEMM, or with ``VLLM_GLM53_FP8_PREFILL=w8a8`` CUTLASS FP8
on a row-major copy of the same FP8 weights with per-token activation scales
(the BF16 weight is then released). TP padding rows and columns are zero, so
they quantize to exact zeros.

The decoder projections expose the packed FP8 tensor as ``layer.weight`` and
keep the BF16 original in a private attribute, so the GLM-5.3 L2 prefetcher
(which skips private attributes) streams the bytes decode reads. The LM head
keeps its BF16 ``weight`` (the MTP draft head copies it) and holds the FP8
copy in its quant method. Its wrapper is not an unquantized method, so
``LogitsProcessor`` calls ``apply()`` instead of the B12X vocabulary
projection; batches above the threshold use the BF16 cuBLAS projection. A
runtime-quantized LM head (``VLLM_MXFP8_LM_HEAD`` / ``VLLM_MTP_NVFP4_LM_HEAD``,
the only heads ``VLLM_LM_HEAD_A16`` affects) is left untouched.

The shape tables are per-rank TP3 shapes, and several of them name different
projections at TP2/TP4 (for example (4096, 4096) is the dense FFN down_proj at
TP3 but the DSA o_proj at TP4 and the KDA o_proj at TP2), so the path only
engages at tensor_parallel_size == 3 and is a no-op elsewhere.

Flags (all read at model construction):

- ``VLLM_GLM53_FP8_DENSE=1`` enables the path (default off).
- ``VLLM_GLM53_FP8_LM_HEAD``, ``VLLM_GLM53_FP8_FFN`` (the three dense FFN
  layers) and ``VLLM_GLM53_FP8_MTP`` (the MTP draft's DSA projections) default
  on once enabled; set them to 0 to leave those BF16.
- ``VLLM_GLM53_FP8_PREFILL=bf16|w8a8`` (default ``bf16``).
- ``VLLM_GLM53_FP8_DENSE_MAX_M`` (default 32).

This changes decode numerics of layers the checkpoint left in BF16 and needs
quality qualification before production use.
"""

from __future__ import annotations

import os

import torch
from torch import nn

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)

logger = init_logger(__name__)

# The per-rank shapes below are TP3 physical shapes (see the module docstring).
SUPPORTED_TP_SIZE = 3

# (N, K) per TP3 rank of the BF16 weights to convert.
GLM53_TP3_FP8_SHAPES: dict[tuple[int, int], str] = {
    (8598, 4096): "KDA in_proj_qkvgfab",
    (4096, 2816): "KDA o_proj",
    (4096, 6144): "DSA o_proj",
    (6144, 1536): "DSA q_b_proj",
}
# The three dense FFN layers (first_k_dense_replace=3, intermediate 12288/3).
GLM53_TP3_FP8_FFN_SHAPES: dict[tuple[int, int], str] = {
    (8192, 4096): "dense FFN gate_up",
    (4096, 4096): "dense FFN down",
}
# Padded vocabulary 154944 / 3 by hidden size 4096.
GLM53_TP3_LM_HEAD_SHAPE = (51648, 4096)


def _flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip() not in ("", "0")


def enabled() -> bool:
    return _flag("VLLM_GLM53_FP8_DENSE", "0")


def prefill_mode() -> str:
    """How batches above the decode threshold run: ``bf16`` or ``w8a8``."""
    mode = os.getenv("VLLM_GLM53_FP8_PREFILL", "bf16").strip().lower()
    if mode not in ("bf16", "w8a8"):
        raise ValueError(f"VLLM_GLM53_FP8_PREFILL must be bf16 or w8a8; got {mode!r}")
    return mode


def max_m() -> int:
    value = int(os.getenv("VLLM_GLM53_FP8_DENSE_MAX_M", "32"))
    if value <= 0:
        raise ValueError("VLLM_GLM53_FP8_DENSE_MAX_M must be positive")
    return value


def _tp_supported(what: str) -> bool:
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size == SUPPORTED_TP_SIZE:
        return True
    logger.warning_once(
        "GLM-5.3 FP8 decode (%s) is measured and keyed on TP%d shapes; "
        "tensor_parallel_size=%d keeps BF16.",
        what,
        SUPPORTED_TP_SIZE,
        tp_size,
    )
    return False


def fp8_dense_shapes() -> dict[tuple[int, int], str]:
    shapes = dict(GLM53_TP3_FP8_SHAPES)
    if _flag("VLLM_GLM53_FP8_FFN", "1"):
        shapes.update(GLM53_TP3_FP8_FFN_SHAPES)
    return shapes


def _pack_marlin_fp8(weight: torch.Tensor, keep_rowmajor: bool = False) -> nn.Module:
    """Quantize an (N, K) BF16 weight per output channel and pack it for Marlin.

    With ``keep_rowmajor`` the same FP8 values are also kept row-major with an
    (N, 1) scale for the CUTLASS prefill path, so decode and prefill read
    identical quantized weights.
    """
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        prepare_fp8_layer_for_marlin,
    )

    n, k = weight.shape
    qweight, scale = ops.scaled_fp8_quant(
        weight.contiguous(), use_per_token_if_dynamic=True
    )
    packed = nn.Module()
    packed.rowmajor = qweight if keep_rowmajor else None
    packed.scale_col = scale.reshape(n, 1).to(torch.float32) if keep_rowmajor else None
    packed.weight = nn.Parameter(qweight.t().contiguous(), requires_grad=False)
    packed.weight_scale = nn.Parameter(
        scale.reshape(1, n).to(torch.float32), requires_grad=False
    )
    packed.input_size_per_partition = k
    packed.output_size_per_partition = n
    packed.orig_dtype = weight.dtype
    prepare_fp8_layer_for_marlin(packed, size_k_first=True)
    return packed


def _decode_eligible(x: torch.Tensor, bias: torch.Tensor | None, limit: int) -> bool:
    return (
        bias is None
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and 0 < x.shape[0] <= limit
    )


class GLM53Fp8DecodeLinearMethod(UnquantizedLinearMethod):
    """Marlin FP8 W8A16 for decode-sized batches, the original method otherwise."""

    # The packed weight is derived from the BF16 checkpoint after loading.
    supports_pre_processed_weights = False

    def __init__(self, inner: UnquantizedLinearMethod, label: str) -> None:
        super().__init__()
        self._inner = inner
        self._label = label
        self._max_m = max_m()
        self._prefill = prefill_mode()

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        self._inner.process_weights_after_loading(layer)
        bf16 = layer.weight.data
        n, k = bf16.shape
        w8a8 = self._prefill == "w8a8"
        packed = _pack_marlin_fp8(bf16, keep_rowmajor=w8a8)
        if w8a8:
            # Prefill reads the row-major FP8 copy; the BF16 weight is released.
            layer._glm53_bf16_weight = None
            layer._glm53_fp8_rowmajor = packed.rowmajor
            layer._glm53_fp8_scale_col = packed.scale_col
        else:
            layer._glm53_bf16_weight = bf16
        del bf16
        layer._glm53_fp8_scale = packed.weight_scale
        layer._glm53_fp8_workspace = packed.workspace
        layer._glm53_fp8_nk = (n, k)
        # Decode reads the packed tensor; expose it as the public weight the
        # L2 prefetcher discovers.
        layer.weight = packed.weight

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            apply_fp8_marlin_linear,
        )

        if _decode_eligible(x, bias, self._max_m):
            n, k = layer._glm53_fp8_nk
            return apply_fp8_marlin_linear(
                x,
                layer.weight,
                layer._glm53_fp8_scale,
                layer._glm53_fp8_workspace,
                n,
                k,
                None,
            )
        if self._prefill == "w8a8":
            from vllm import _custom_ops as ops

            shape = x.shape
            x2 = x.reshape(-1, shape[-1])
            xq, xs = ops.scaled_fp8_quant(x2, use_per_token_if_dynamic=True)
            out = ops.cutlass_scaled_mm(
                xq,
                layer._glm53_fp8_rowmajor.t(),
                scale_a=xs,
                scale_b=layer._glm53_fp8_scale_col,
                out_dtype=x.dtype,
                bias=bias,
            )
            return out.reshape(*shape[:-1], out.shape[-1])
        return self._inner._gemm_impl(layer, x, layer._glm53_bf16_weight, bias)


class GLM53Fp8DecodeLMHeadMethod(QuantizeMethodBase):
    """Wrap the LM head's method: keep its BF16 weight and add an FP8 copy.

    It must be a ``QuantizeMethodBase`` so the loader calls
    ``process_weights_after_loading``, and must not be an
    ``UnquantizedEmbeddingMethod``, or ``LogitsProcessor`` projects the raw
    BF16 weight itself and never calls ``apply()``.
    """

    supports_pre_processed_weights = False

    def __init__(self, inner: UnquantizedEmbeddingMethod) -> None:
        super().__init__()
        self._inner = inner
        self._max_m = max_m()
        self._packed: nn.Module | None = None

    def __getattr__(self, name: str):
        if name.startswith("__") or name in ("_inner", "_max_m", "_packed"):
            raise AttributeError(name)
        return getattr(self._inner, name)

    def create_weights(self, *args, **kwargs):
        return self._inner.create_weights(*args, **kwargs)

    def embedding(self, layer: nn.Module, *args, **kwargs) -> torch.Tensor:
        return self._inner.embedding(layer, *args, **kwargs)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        self._inner.process_weights_after_loading(layer)
        weight = layer.weight.data
        if (
            tuple(weight.shape) != GLM53_TP3_LM_HEAD_SHAPE
            or weight.dtype != torch.bfloat16
        ):
            logger.warning(
                "GLM-5.3 FP8 decode: LM head is %s %s, not converting",
                tuple(weight.shape),
                weight.dtype,
            )
            return
        self._packed = _pack_marlin_fp8(weight)
        logger.info_once("GLM-5.3 FP8 decode: LM head packed to Marlin FP8")

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            apply_fp8_marlin_linear,
        )

        packed = self._packed
        if packed is not None and _decode_eligible(x, bias, self._max_m):
            return apply_fp8_marlin_linear(
                x,
                packed.weight,
                packed.weight_scale,
                packed.workspace,
                packed.output_size_per_partition,
                packed.input_size_per_partition,
                None,
            )
        return self._inner.apply(layer, x, bias)


def enable_glm53_fp8_dense(module: nn.Module, *, draft: bool = False) -> int:
    """Wrap the selected BF16 projections; conversion happens after loading.

    ``draft=True`` is the MTP draft module (its DSA projections share the
    target's shapes); ``VLLM_GLM53_FP8_MTP=0`` leaves it BF16. Returns the
    number of wrapped projections.
    """
    if not enabled() or (draft and not _flag("VLLM_GLM53_FP8_MTP", "1")):
        return 0
    if not _tp_supported("MTP draft" if draft else "decoder"):
        return 0
    shapes = fp8_dense_shapes()
    mode = prefill_mode()
    counts: dict[str, int] = {}
    for child in module.modules():
        if not isinstance(child, LinearBase) or not isinstance(
            child.quant_method, UnquantizedLinearMethod
        ):
            continue
        if isinstance(child.quant_method, GLM53Fp8DecodeLinearMethod):
            continue
        weight = getattr(child, "weight", None)
        if weight is None or weight.dim() != 2 or weight.dtype != torch.bfloat16:
            continue
        label = shapes.get((int(weight.shape[0]), int(weight.shape[1])))
        if label is None:
            continue
        child.quant_method = GLM53Fp8DecodeLinearMethod(child.quant_method, label)
        counts[label] = counts.get(label, 0) + 1
    wrapped = sum(counts.values())
    summary = ", ".join(f"{k} x{v}" for k, v in sorted(counts.items()))
    logger.info_once(
        "GLM-5.3 FP8 decode%s: %d BF16 projections use Marlin FP8 W8A16 for "
        "batches of at most %d rows, prefill %s [%s]",
        " (MTP draft)" if draft else "",
        wrapped,
        max_m(),
        mode,
        summary,
    )
    return wrapped


def enable_glm53_fp8_lm_head(lm_head: nn.Module) -> bool:
    """Give the target LM head an FP8 decode copy. Returns whether it did."""
    if not enabled() or not _flag("VLLM_GLM53_FP8_LM_HEAD", "1"):
        return False
    if not _tp_supported("LM head"):
        return False
    weight = getattr(lm_head, "weight", None)
    if (
        getattr(lm_head, "runtime_lm_head_quantization", None) is not None
        or type(getattr(lm_head, "quant_method", None))
        is not UnquantizedEmbeddingMethod
        or weight is None
        or tuple(weight.shape) != GLM53_TP3_LM_HEAD_SHAPE
        or weight.dtype != torch.bfloat16
    ):
        logger.warning(
            "GLM-5.3 FP8 decode: LM head is not an unquantized BF16 %s shard; "
            "keeping it as is.",
            GLM53_TP3_LM_HEAD_SHAPE,
        )
        return False
    lm_head.quant_method = GLM53Fp8DecodeLMHeadMethod(lm_head.quant_method)
    logger.info_once("GLM-5.3 FP8 decode: LM head uses Marlin FP8 W8A16 for decode")
    return True
