# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark Triton MLA decode at Kimi-K3 per-rank geometries."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch

from vllm.triton_utils import triton
from vllm.v1.attention.ops.triton_decode_attention import (
    _decode_softmax_reducev_fwd,
    _fwd_grouped_kernel_stage1,
    _page_stride,
)


@dataclass(frozen=True)
class KernelConfig:
    block_n: int
    block_h: int
    num_warps: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--page-size", type=int, default=720)
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[32, 512, 4096, 16000, 65136],
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    return parser.parse_args()


def num_kv_splits(seq_len: int) -> int:
    ideal = triton.next_power_of_2(max(1, seq_len // 512))
    return min(ideal, 8)


def benchmark_us(fn, *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    cache_dtype = torch.float8_e4m3fn
    batch = 1
    q_heads = args.heads
    qk_dim = 576
    value_dim = 512
    kv_heads = 1
    max_seq_len = max(args.seq_lens)
    num_pages = math.ceil(max_seq_len / args.page_size)

    # This is the production block-major, cross-layer cache layout. Selecting a
    # layer leaves a gap of `layers * page_size * qk_dim` between pages.
    cross_layer_cache = torch.zeros(
        num_pages,
        args.layers,
        args.page_size,
        kv_heads,
        qk_dim,
        dtype=cache_dtype,
        device=device,
    )
    cache = cross_layer_cache[:, 0]
    cache.copy_(torch.randn(cache.shape, dtype=dtype, device=device).to(cache_dtype))
    value_cache = cache[..., :value_dim]

    q = torch.randn(batch, q_heads, qk_dim, dtype=dtype, device=device)
    req_to_pages = torch.arange(num_pages, dtype=torch.int32, device=device)[None]
    seq_lens = torch.empty(batch, dtype=torch.int32, device=device)
    k_scale = torch.ones((), dtype=torch.float32, device=device)
    v_scale = torch.ones((), dtype=torch.float32, device=device)
    output = torch.empty(batch, q_heads, value_dim, dtype=dtype, device=device)
    lse = torch.empty(batch, q_heads, dtype=dtype, device=device)

    configs = [
        KernelConfig(block_n, block_h, num_warps)
        for block_n in (16, 32)
        for block_h in (1, 2, 4, 8, 16)
        for num_warps in (2, 4, 8)
    ]
    baseline = KernelConfig(block_n=32, block_h=16, num_warps=4)

    print(
        "heads,seq_len,splits,block_n,block_h,warps,stage1_us,total_us,rel_l2,max_abs"
    )
    for seq_len in args.seq_lens:
        splits = num_kv_splits(seq_len)
        seq_lens.fill_(seq_len)
        attn_logits = torch.empty(
            batch,
            q_heads,
            splits,
            value_dim + 1,
            dtype=torch.float32,
            device=device,
        )

        def run(
            config: KernelConfig,
            *,
            splits: int = splits,
            attn_logits: torch.Tensor = attn_logits,
        ) -> None:
            grid = (
                batch,
                triton.cdiv(q_heads, min(config.block_h, q_heads)),
                splits,
            )
            _fwd_grouped_kernel_stage1[grid](
                q,
                cache,
                value_cache,
                qk_dim**-0.5,
                req_to_pages,
                seq_lens,
                attn_logits,
                req_to_pages.stride(0),
                q.stride(0),
                q.stride(1),
                _page_stride(cache, args.page_size),
                cache.stride(-3),
                cache.stride(-2),
                _page_stride(value_cache, args.page_size),
                value_cache.stride(-3),
                value_cache.stride(-2),
                attn_logits.stride(0),
                attn_logits.stride(1),
                attn_logits.stride(2),
                k_scale,
                v_scale,
                kv_group_num=q_heads,
                q_head_num=q_heads,
                BLOCK_DMODEL=value_dim,
                BLOCK_DPE=qk_dim - value_dim,
                BLOCK_DV=value_dim,
                BLOCK_N=config.block_n,
                BLOCK_H=config.block_h,
                NUM_KV_SPLITS=splits,
                PAGE_SIZE=args.page_size,
                logit_cap=0.0,
                num_warps=config.num_warps,
                num_stages=1,
                Lk=qk_dim,
                Lv=value_dim,
                IS_MLA=True,
            )

        def reduce(
            *,
            splits: int = splits,
            attn_logits: torch.Tensor = attn_logits,
        ) -> None:
            _decode_softmax_reducev_fwd(
                attn_logits,
                q,
                output,
                lse,
                value_cache,
                seq_lens,
                splits,
            )

        def run_total(config: KernelConfig) -> None:
            run(config)
            reduce()

        run_total(baseline)
        torch.accelerator.synchronize()
        reference = output.float().clone()

        for config in configs:
            run_total(config)
            torch.accelerator.synchronize()
            actual = output.float()
            rel_l2 = (actual - reference).norm() / reference.norm().clamp_min(1e-12)
            max_abs = (actual - reference).abs().max()
            stage1_us = benchmark_us(
                lambda config=config: run(config),
                warmup=args.warmup,
                iterations=args.iterations,
            )
            total_us = benchmark_us(
                lambda config=config: run_total(config),
                warmup=args.warmup,
                iterations=args.iterations,
            )
            print(
                f"{q_heads},{seq_len},{splits},{config.block_n},"
                f"{config.block_h},{config.num_warps},{stage1_us:.3f},"
                f"{total_us:.3f},{rel_l2.item():.8f},{max_abs.item():.6f}"
            )


if __name__ == "__main__":
    main()
