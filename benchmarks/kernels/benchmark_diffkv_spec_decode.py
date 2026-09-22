# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare automatic DiffKV dispatch with its 2D path for speculative queries.

Run inside an installed vLLM environment. E4M3 additionally requires DiffKV FP8
support. Results include exact shapes and numerical agreement, not a speed gate.
"""

import argparse
import inspect
import json
from pathlib import Path

import torch
import triton

from vllm.v1.attention.ops.triton_unified_attention_diffkv import (
    unified_attention_diffkv,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=["bfloat16", "fp8_e4m3"], default="bfloat16")
    parser.add_argument("--contexts", type=int, nargs="+", default=[8192, 32768, 65536])
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--source-module", type=Path)
    args = parser.parse_args()
    run = unified_attention_diffkv
    if args.source_module:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "diffkv_candidate", args.source_module
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        run = module.unified_attention_diffkv
    if (
        args.dtype == "fp8_e4m3"
        and "k_descale" not in inspect.signature(run).parameters
    ):
        parser.error(
            "This vLLM source lacks FP8 DiffKV support; "
            "apply the FP8 reader change first"
        )
    rows = []
    torch.manual_seed(19)
    for batch in args.concurrency:
        for context in args.contexts:
            q_len, heads, kv_heads, hq, hv, block = 8, 16, 1, 192, 128, 16
            blocks = (context + block - 1) // block
            table = torch.arange(
                batch * blocks, device="cuda", dtype=torch.int32
            ).reshape(batch, blocks)
            packed = (
                torch.randn(batch * blocks, block, kv_heads, hq + hv, device="cuda")
                * 0.25
            ).bfloat16()
            if args.dtype == "fp8_e4m3":
                packed = packed.to(torch.float8_e4m3fn)
            q = torch.randn(
                batch * q_len, heads, hq, device="cuda", dtype=torch.bfloat16
            )
            out = torch.empty(
                batch * q_len, heads, hv, device="cuda", dtype=torch.bfloat16
            )
            partial = torch.full((128, heads, 16, hv), float("nan"), device="cuda")
            maximum = torch.full((128, heads, 16), float("nan"), device="cuda")
            expsum = torch.full_like(maximum, float("nan"))
            kw = dict(
                q=q,
                k=packed[..., :hq],
                v=packed[..., hq:],
                out=out,
                cu_seqlens_q=torch.arange(batch + 1, device="cuda", dtype=torch.int32)
                * q_len,
                seqused_k=torch.full(
                    (batch,), context, device="cuda", dtype=torch.int32
                ),
                softmax_scale=hq**-0.5,
                causal=True,
                window_size=(-1, -1),
                block_table=table,
                softcap=0,
                max_seqlen_q=q_len,
                seq_threshold_3D=128,
                num_par_softmax_segments=16,
                softmax_segm_output=partial,
                softmax_segm_max=maximum,
                softmax_segm_expsum=expsum,
            )
            if args.dtype == "fp8_e4m3":
                kw.update(
                    k_descale=torch.ones(1, device="cuda"),
                    v_descale=torch.ones(1, device="cuda"),
                )
            run(**dict(kw, seq_threshold_3D=0))
            expected = out.clone()
            run(**kw)
            torch.testing.assert_close(out, expected, atol=0.02, rtol=0.02)
            relative_error = (
                (out.float() - expected.float()).norm() / expected.float().norm()
            ).item()
            assert relative_error < 0.01, relative_error
            split_used = bool(torch.isfinite(expsum[: batch * q_len, :, 0]).all())
            two_d = triton.testing.do_bench(
                lambda kw=kw: run(**dict(kw, seq_threshold_3D=0)), warmup=25, rep=100
            )
            auto = triton.testing.do_bench(lambda kw=kw: run(**kw), warmup=25, rep=100)
            row = dict(
                concurrency=batch,
                query_length=q_len,
                context=context,
                dtype=args.dtype,
                heads=heads,
                kv_heads=kv_heads,
                automatic_ms=auto,
                two_d_ms=two_d,
                speedup=two_d / auto,
                split_used=split_used,
                correct=True,
                relative_error=relative_error,
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
    args.output.write_text(
        json.dumps(dict(rows=rows, passed=all(r["correct"] for r in rows)), indent=2)
        + "\n"
    )


if __name__ == "__main__":
    with torch.inference_mode():
        main()
