# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up Kimi-K3 Triton kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.models.kimi_k3.nvidia.kda import KimiK3DeltaAttention
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def _warm_vision_position_interpolation(model: torch.nn.Module) -> int:
    """Compile Kimi vision interpolation before KV-cache allocation.

    Args:
        model: Loaded model whose module tree may contain Kimi vision position
            embeddings.

    Returns:
        Number of Kimi vision interpolation modules invoked.
    """
    from vllm.model_executor.models.kimi_k25_vit import (
        Learnable2DInterpPosEmbDivided_fixed,
        get_rope_shape,
    )

    warmed = 0
    for module in model.modules():
        if not isinstance(module, Learnable2DInterpPosEmbDivided_fixed):
            continue
        get_rope_shape(
            module.weight,
            interpolation_mode=module.interpolation_mode,
            shape=(64, 64),
        )
        warmed += 1
    return warmed


def _get_kda_layer(worker: Worker) -> KimiK3DeltaAttention | None:
    from vllm.models.kimi_k3.nvidia.kda import KimiK3DeltaAttention

    # The target model and speculative draft are separate model-runner
    # objects, but both register attention layers in the shared compilation
    # context. Traverse the target model first so a cacheless draft layer can
    # never become the warmup template before KV pages are bound.
    get_model = getattr(worker, "get_model", None)
    if callable(get_model):
        target_model = get_model()
        modules = getattr(target_model, "modules", None)
        if callable(modules):
            target_layer = next(
                (
                    layer
                    for layer in modules()
                    if isinstance(layer, KimiK3DeltaAttention)
                ),
                None,
            )
            if target_layer is not None:
                return target_layer

    compilation_config = getattr(
        worker.model_runner,
        "compilation_config",
        None,
    )
    static_context = getattr(compilation_config, "static_forward_context", None)
    if not isinstance(static_context, dict):
        return None
    candidates = [
        layer
        for layer in static_context.values()
        if isinstance(layer, KimiK3DeltaAttention)
    ]
    return next(
        (layer for layer in candidates if hasattr(layer, "kv_cache")),
        candidates[0] if candidates else None,
    )


def _warm_attn_res(worker: Worker) -> None:
    from vllm.models.kimi_k3.nvidia.ops.attn_res import (
        attn_res,
        get_attn_res_triton_warmup_profiles,
    )

    config = worker.model_config.hf_text_config
    block_size = getattr(config, "attn_res_block_size", None)
    if block_size is None:
        return

    hidden_size = int(config.hidden_size)
    max_blocks = (int(config.num_hidden_layers) + block_size - 1) // block_size
    if max_blocks < 2:
        return

    dtype = worker.model_config.dtype
    device = torch.device("cuda")
    eps = float(config.rms_norm_eps)
    prefix = torch.zeros((1, hidden_size), dtype=dtype, device=device)
    delta = torch.zeros_like(prefix)
    blocks = torch.zeros(
        (1, max_blocks, hidden_size),
        dtype=dtype,
        device=device,
    )
    norm_weight = torch.zeros(hidden_size, dtype=dtype, device=device)
    qk_weight = torch.zeros_like(norm_weight)
    output_norm_weight = torch.zeros_like(norm_weight)

    for (
        num_blocks,
        has_delta,
        block_write_idx,
        apply_output_norm,
    ) in get_attn_res_triton_warmup_profiles(max_blocks):
        attn_res(
            prefix,
            delta if has_delta else None,
            blocks,
            norm_weight,
            qk_weight,
            output_norm_weight if apply_output_norm else None,
            num_blocks=num_blocks,
            block_write_idx=block_write_idx,
            eps=eps,
            output_norm_eps=eps if apply_output_norm else 0.0,
        )


def _warm_recurrent_kda(
    layer: KimiK3DeltaAttention,
    input_dtype: torch.dtype,
) -> None:
    from vllm.models.kimi_k3.nvidia.ops.third_party.kda.fused_recurrent import (
        fused_recurrent_kda,
        get_fused_recurrent_kda_fwd_warmup_profiles,
    )

    num_speculative_tokens = int(layer.num_spec)
    # fused_recurrent_kda_fwd_kernel is only used by speculative decode.
    if num_speculative_tokens <= 0:
        return

    kv_cache = getattr(layer, "kv_cache", None)
    state = (
        kv_cache[1]
        if isinstance(kv_cache, (list, tuple))
        and len(kv_cache) >= 2
        and isinstance(kv_cache[1], torch.Tensor)
        and kv_cache[1].numel()
        else None
    )
    if state is None:
        # Kernel warmup runs before production KV-cache binding. One temporary
        # state page preserves the production layout and dtype for compilation.
        state_shape = layer.get_state_shape()[1]
        state_dtype = layer.get_state_dtype()[1]
        state = torch.empty(
            (1, *state_shape),
            dtype=state_dtype,
            device=layer.A_log.device,
        )

    logger.info("Warming up Kimi-K3 speculative KDA kernels.")
    h = int(layer.local_num_heads)
    d = int(layer.head_dim)
    tokens_per_sequence = num_speculative_tokens + 1
    for num_sequences in get_fused_recurrent_kda_fwd_warmup_profiles(h):
        num_tokens = num_sequences * tokens_per_sequence
        packed_qkv = torch.empty(
            (num_tokens, 3 * h * d),
            dtype=input_dtype,
            device=state.device,
        )
        q, k, v = (
            tensor.view(1, num_tokens, h, d)
            for tensor in packed_qkv.split(h * d, dim=-1)
        )
        fused_recurrent_kda(
            q=q,
            k=k,
            v=v,
            raw_g=torch.empty(
                (1, num_tokens, h, d),
                dtype=input_dtype,
                device=state.device,
            ),
            raw_beta=torch.empty(
                (1, num_tokens, h),
                dtype=input_dtype,
                device=state.device,
            ),
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            lower_bound=layer.gate_lower_bound,
            initial_state=state[:1],
            cu_seqlens=torch.arange(
                0,
                num_tokens + 1,
                tokens_per_sequence,
                dtype=torch.int32,
                device=state.device,
            ),
            ssm_state_indices=torch.zeros(
                (num_sequences, tokens_per_sequence),
                dtype=torch.int32,
                device=state.device,
            ),
            num_accepted_tokens=torch.ones(
                num_sequences,
                dtype=torch.int32,
                device=state.device,
            ),
            out=torch.empty(
                (1, num_tokens, h, d),
                dtype=input_dtype,
                device=state.device,
            ),
        )


def _warm_chunk_kda_prefill(
    layer: KimiK3DeltaAttention,
    input_dtype: torch.dtype,
) -> None:
    """Compile the Triton KDA prefill path before KV-cache allocation."""
    if layer.kda_prefill_backend != "triton":
        return

    from vllm.models.kimi_k3.nvidia.ops.third_party.kda import (
        chunk_kda_with_fused_gate,
    )
    from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE

    device = layer.A_log.device
    num_heads = int(layer.local_num_heads)
    head_dim = int(layer.head_dim)
    num_tokens = FLA_CHUNK_SIZE
    state_shape = layer.get_state_shape()[1]
    state_dtype = layer.get_state_dtype()[1]

    packed_qkv = torch.empty(
        (3, 1, num_tokens, num_heads, head_dim),
        dtype=input_dtype,
        device=device,
    )
    raw_g = torch.zeros(
        (1, num_tokens, num_heads, head_dim),
        dtype=input_dtype,
        device=device,
    )
    raw_beta = torch.zeros(
        (1, num_tokens, num_heads),
        dtype=input_dtype,
        device=device,
    )
    initial_state = torch.zeros(
        (1, *state_shape),
        dtype=state_dtype,
        device=device,
    )
    cu_seqlens = torch.tensor(
        [0, num_tokens],
        dtype=torch.int32,
        device=device,
    )

    logger.info("Warming up Kimi-K3 Triton KDA prefill kernels.")
    chunk_kda_with_fused_gate(
        q=packed_qkv[0],
        k=packed_qkv[1],
        v=packed_qkv[2],
        raw_g=raw_g,
        raw_beta=raw_beta,
        A_log=layer.A_log,
        g_bias=layer.dt_bias,
        lower_bound=layer.gate_lower_bound,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
    )


@torch.inference_mode()
def kimi_k3_triton_warmup(worker: Worker) -> None:
    """Warm Kimi-K3 Triton kernels reachable by this server."""
    if not current_platform.is_cuda():
        return

    warmed_vision_interpolators = _warm_vision_position_interpolation(
        worker.get_model()
    )
    if warmed_vision_interpolators:
        logger.info(
            "Warmed up %d Kimi vision position interpolator(s).",
            warmed_vision_interpolators,
        )

    layer = _get_kda_layer(worker)
    if layer is None:
        return

    _warm_attn_res(worker)
    _warm_chunk_kda_prefill(layer, worker.model_config.dtype)
    _warm_recurrent_kda(layer, worker.model_config.dtype)
