# SPDX-License-Identifier: Apache-2.0
"""Tensor-parallel head padding for GLM-5.3-Flash (TP3, TP6, ...).

GLM-5.3-Flash has 64 MLA heads, 64 KDA heads and a 2048-wide MoE intermediate.
Tensor-parallel sizes that do not divide 64 (TP3) are made to work by padding
the head axes with inert zero heads at load time:

* MLA: ``num_attention_heads`` 64 -> ``mla_heads`` (72 = 24 per rank on TP3;
  the B12X sparse-MLA decode kernel runs its native H8 grid when the local
  head count is a multiple of 8).  ``q_b_proj`` / ``kv_b_proj`` rows and
  ``o_proj`` columns are zero for the padded heads, so they attend uniformly
  over zero values and contribute nothing.
* KDA: ``linear_num_heads`` 64 -> ``kda_heads`` (66 = 22 per rank on TP3).
  ``q/k/v_proj``, ``b_proj``, ``f_b_proj``, ``g_b_proj``, the three conv1d
  banks, ``A_log`` and ``dt_bias`` are zero-padded and ``o_proj`` columns are
  zero; with q = k = v = 0 the recurrent state of a padded head stays zero.
* The routed experts are not split (run with ``--enable-expert-parallel``:
  288 experts / 3 ranks).  The shared expert (12288) and dense MLPs divide.
* Vocabulary: ``padding_size`` becomes lcm(64, TP) so the padded vocab
  divides across ranks (154,880 -> 155,136 on TP3).

Enable with ``VLLM_GLM53_TP_HEAD_PAD=<mla_heads>,<kda_heads>`` (``72,66`` on
TP3) or ``VLLM_GLM53_TP_HEAD_PAD=auto`` together with the launcher's ``TP``.
Only BF16/FP16/FP32 attention tensors are padded; quantized attention
checkpoints (Spark MXFP8) raise.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Iterable
from typing import Any

import torch

_ENV = "VLLM_GLM53_TP_HEAD_PAD"
# AutoWeightsLoader hands submodules prefix-relative names ("layers.0.self_attn..."), so no leading dot.
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_MTP_SELF_ATTN = ".self_attn."


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def requested_head_counts(ckpt_mla_heads: int, ckpt_kda_heads: int) -> tuple[int, int] | None:
    """Return (mla_heads, kda_heads) after padding, or None when disabled."""
    spec = os.getenv(_ENV, "").strip()
    if not spec:
        return None
    if spec.lower() == "auto":
        tp = int(os.getenv("TP", "0") or 0)
        if tp <= 1:
            return None
        # MLA: local head count a multiple of 8 (B12X H8 decode grid).
        mla = _round_up(ckpt_mla_heads, 8 * tp)
        kda = _round_up(ckpt_kda_heads, tp)
        return mla, kda
    parts = [int(p) for p in spec.split(",")]
    if len(parts) != 2:
        raise ValueError(f"{_ENV} must be 'auto' or '<mla_heads>,<kda_heads>', got {spec!r}")
    mla, kda = parts
    if mla < ckpt_mla_heads or kda < ckpt_kda_heads:
        raise ValueError(f"{_ENV}={spec} pads below the checkpoint head counts ({ckpt_mla_heads}, {ckpt_kda_heads})")
    return mla, kda


def apply_config_padding(cfg: Any) -> None:
    """Pad the head counts on a Glm5Next text config in place (idempotent)."""
    ckpt_mla = int(getattr(cfg, "num_attention_heads_ckpt", None) or cfg.num_attention_heads)
    ckpt_kda = int(getattr(cfg, "linear_num_heads_ckpt", None) or cfg.linear_num_heads)
    counts = requested_head_counts(ckpt_mla, ckpt_kda)
    if counts is None:
        return
    mla, kda = counts
    cfg.num_attention_heads_ckpt = ckpt_mla
    cfg.linear_num_heads_ckpt = ckpt_kda
    cfg.num_attention_heads = mla
    cfg.num_key_value_heads = mla
    cfg.linear_num_heads = kda
    lac = getattr(cfg, "linear_attn_config", None)
    if isinstance(lac, dict):
        cfg.linear_attn_config = {**lac, "num_heads": kda}


def shared_expert_intermediate(config: Any, tp_size: int | None = None) -> int:
    """Shared-expert MLP width padded to a multiple of 64 x TP (2048 -> 2112 on TP3).

    The padded channels have zero gate/up rows and zero down columns, so
    silu(0) * 0 contributes nothing; the math is exact.  No-op for TP 1/2/4/8.
    """
    if tp_size is None:
        from vllm.distributed import get_tensor_model_parallel_world_size

        tp_size = get_tensor_model_parallel_world_size()
    base = int(config.moe_intermediate_size) * int(config.n_shared_experts or 1)
    return _round_up(base, 64 * int(tp_size))


def routed_expert_intermediate(config: Any, tp_size: int | None = None, *, mtp: bool = False) -> int:
    """Routed-expert intermediate padded so it shards across TP.

    Multiple of 128 x TP (2048 -> 2304 on TP3 = 768 per rank, 6 x 128 tiles).
    No-op for TP 1/2/4/8.  Padded gate/up rows and down
    columns are zero (zero codes and zero block scales), so the math is exact.
    """
    if tp_size is None:
        from vllm.distributed import get_tensor_model_parallel_world_size

        tp_size = get_tensor_model_parallel_world_size()
    # 128 x TP for every layer: the 16-aligned 688 shard is rounded to 704 by the runtime and
    # benches 20 % slower at C4+ than 768 (both on heuristic configs); the profile sweep targets 768.
    del mtp
    align = 128 * int(tp_size)
    return _round_up(int(config.moe_intermediate_size), align)


def _pad_expert_tensor(t: torch.Tensor, leaf: str, extra: int, base_intermediate: int) -> torch.Tensor:
    """Zero-pad one routed-expert tensor along its intermediate axis.

    ``extra`` counts logical (unpacked) intermediate elements.  Layouts in the
    GLM-5.3-Flash checkpoints:
      NVFP4 (target): gate/up ``weight`` u8 [I, H/2], ``weight_scale`` fp8 [I, H/16];
                      down ``weight`` u8 [H, I/2], ``weight_scale`` fp8 [H, I/16]
      MXFP8 (MTP):    gate/up ``weight`` fp8 [I, H], ``weight_scale`` u8 [I, H/32];
                      down ``weight`` fp8 [H, I], ``weight_scale`` u8 [H, I/32]
    Scalars (``input_scale``, ``weight_scale_2``) pass through.  Zero codes and
    zero block scales make the padded channels contribute exactly nothing.
    """
    proj, _, kind = leaf.partition(".")
    if kind not in ("weight", "weight_scale") or t.dim() != 2:
        return t
    if proj in ("gate_proj", "up_proj"):
        return _pad_any(t, 0, extra)
    if proj != "down_proj":
        return t
    if kind == "weight":
        packed = 2 if t.dtype == torch.uint8 else 1  # fp4 pairs per byte vs fp8/bf16
        return _pad_any(t, 1, extra // packed)
    group = base_intermediate // int(t.shape[1])  # 16 (NVFP4) or 32 (MXFP8)
    return _pad_any(t, 1, extra // group)


def _pad_any(t: torch.Tensor, dim: int, extra: int) -> torch.Tensor:
    if extra <= 0:
        return t
    shape = list(t.shape)
    shape[dim] = extra
    return torch.cat([t, torch.zeros(shape, dtype=t.dtype, device=t.device)], dim=dim)


def vocab_padding_size(default: int = 64) -> int:
    """Vocab padding that keeps the padded vocab divisible by the TP size."""
    from vllm.distributed import get_tensor_model_parallel_world_size

    tp = get_tensor_model_parallel_world_size()
    return math.lcm(default, tp)


def _pad_dim(t: torch.Tensor, dim: int, extra: int) -> torch.Tensor:
    if extra <= 0:
        return t
    if t.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise NotImplementedError(
            f"TP head padding supports BF16/FP16/FP32 attention tensors only, got {t.dtype} "
            "(quantized attention checkpoints are not supported)"
        )
    shape = list(t.shape)
    shape[dim] = extra
    pad = torch.zeros(shape, dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=dim)


def pad_head_weights(weights: Iterable[tuple], config: Any) -> Iterable[tuple]:
    """Wrap a checkpoint weight iterator, padding head-sharded tensors."""
    ckpt_mla = getattr(config, "num_attention_heads_ckpt", None)
    ckpt_kda = getattr(config, "linear_num_heads_ckpt", None)
    if ckpt_mla is None and ckpt_kda is None:
        yield from weights
        return
    mla_extra = int(config.num_attention_heads) - int(ckpt_mla or config.num_attention_heads)
    kda_extra = int(config.linear_num_heads) - int(ckpt_kda or config.linear_num_heads)
    qk_dim = int(config.qk_nope_head_dim) + int(config.qk_rope_head_dim)
    kv_b_dim = int(config.qk_nope_head_dim) + int(config.v_head_dim)
    v_dim = int(config.v_head_dim)
    kda_hd = int(config.linear_head_dim)

    def is_kda(layer_idx: int) -> bool:
        return bool(config.is_kda_layer(layer_idx))

    shared_extra = 0
    if os.getenv(_ENV, "").strip():
        from vllm.distributed import get_tensor_model_parallel_world_size

        base = int(config.moe_intermediate_size) * int(config.n_shared_experts or 1)
        shared_extra = shared_expert_intermediate(config, get_tensor_model_parallel_world_size()) - base

    routed_extra = routed_extra_mtp = 0
    if os.getenv(_ENV, "").strip():
        from vllm.distributed import get_tensor_model_parallel_world_size

        tp = get_tensor_model_parallel_world_size()
        base_inter = int(config.moe_intermediate_size)
        routed_extra = routed_expert_intermediate(config, tp) - base_inter
        routed_extra_mtp = routed_expert_intermediate(config, tp, mtp=True) - base_inter
    num_target_layers = int(config.num_hidden_layers)

    for item in weights:
        name, w = item[0], item[1]
        if (routed_extra or routed_extra_mtp) and ".mlp.experts." in name and ".shared_experts." not in name:
            leaf = name.rsplit(".mlp.experts.", 1)[1].split(".", 1)[1]  # "<proj>.<kind>"
            lm = _LAYER_RE.search(name)
            is_mtp = lm is not None and int(lm.group(1)) >= num_target_layers
            extra = routed_extra_mtp if is_mtp else routed_extra
            w = _pad_expert_tensor(w, leaf, extra, int(config.moe_intermediate_size))
            yield (name, w, *item[2:])
            continue
        if shared_extra and ".mlp.shared_experts." in name:
            leaf = name.rsplit(".mlp.shared_experts.", 1)[1]
            if leaf in ("gate_proj.weight", "up_proj.weight"):
                w = _pad_dim(w, 0, shared_extra)
            elif leaf == "down_proj.weight":
                w = _pad_dim(w, 1, shared_extra)
            elif "weight_scale" in leaf:
                raise NotImplementedError("TP padding does not support quantized shared experts")
            yield (name, w, *item[2:])
            continue
        m = _LAYER_RE.search(name)
        if m is None or _MTP_SELF_ATTN not in name:
            yield item
            continue
        layer_idx = int(m.group(1))
        suffix = name.split(_MTP_SELF_ATTN, 1)[1]
        suffix = suffix.replace("forget_gate.", "")
        if is_kda(layer_idx):
            if kda_extra:
                if suffix in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "f_b_proj.weight", "g_b_proj.weight", "dt_bias", "q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight"):
                    w = _pad_dim(w, 0, kda_extra * kda_hd)
                elif suffix in ("b_proj.weight", "A_log"):
                    if w.dim() == 4:  # legacy (1, 1, H, 1) A_log
                        w = w.reshape(w.shape[2])
                    w = _pad_dim(w, 0, kda_extra)
                elif suffix == "conv1d.weight":  # fused q/k/v banks
                    rows = w.shape[0] // 3
                    w = torch.cat([_pad_dim(w.narrow(0, i * rows, rows), 0, kda_extra * kda_hd) for i in range(3)], dim=0)
                elif suffix == "o_proj.weight":
                    w = _pad_dim(w, 1, kda_extra * kda_hd)
        elif mla_extra:
            if suffix == "q_b_proj.weight":
                w = _pad_dim(w, 0, mla_extra * qk_dim)
            elif suffix == "kv_b_proj.weight":
                w = _pad_dim(w, 0, mla_extra * kv_b_dim)
            elif suffix == "o_proj.weight":
                w = _pad_dim(w, 1, mla_extra * v_dim)
            elif suffix in ("q_b_proj.weight_scale", "q_b_proj.weight_scale_inv", "o_proj.weight_scale", "o_proj.weight_scale_inv", "kv_b_proj.weight_scale", "kv_b_proj.weight_scale_inv"):
                raise NotImplementedError("TP head padding does not support quantized MLA projections")
        yield (name, w, *item[2:])
