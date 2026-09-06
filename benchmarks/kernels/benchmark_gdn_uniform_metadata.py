# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Proxy timing for uniform GDN metadata writes, not end-to-end serving."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from vllm.v1.attention.backends.gdn_attn import _fill_uniform_spec_metadata


def benchmark_case(rows: int) -> dict:
    device = torch.device("cuda")
    window = 4
    source = torch.arange(rows * 7, device=device, dtype=torch.int32).view(rows, 7)
    counts = torch.ones(rows, dtype=torch.int32, device=device)
    state = torch.empty(rows, window, device=device, dtype=torch.int32)
    accepted = torch.empty_like(counts)
    masks = torch.empty(rows, device=device, dtype=torch.bool)
    tokens = torch.empty(rows * window, device=device, dtype=torch.int32)
    starts = torch.empty(rows + 1, device=device, dtype=torch.int32)
    token_source = torch.arange(rows * window, device=device, dtype=torch.int32)
    start_source = torch.arange(rows + 1, device=device, dtype=torch.int32) * window

    def copies():
        state.copy_(source[:, :window], non_blocking=True)
        masks.fill_(True)
        tokens.copy_(token_source, non_blocking=True)
        starts.copy_(start_source, non_blocking=True)
        accepted.copy_(counts, non_blocking=True)

    def fused():
        _fill_uniform_spec_metadata[((rows * window + 127) // 128,)](
            source,
            counts,
            state,
            accepted,
            masks,
            tokens,
            starts,
            rows,
            source.stride(0),
            counts.stride(0),
            WINDOW=window,
            BLOCK=128,
        )

    outputs = (state, accepted, masks, tokens, starts)
    copies()
    reference = [tensor.clone() for tensor in outputs]
    for tensor in outputs:
        tensor.zero_()
    fused()
    for actual, expected in zip(outputs, reference):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    samples = {"copies": [], "fused": []}
    for _ in range(5):
        for label, launch in (
            ("copies", copies),
            ("fused", fused),
            ("fused", fused),
            ("copies", copies),
        ):
            torch.accelerator.synchronize()
            started = time.perf_counter()
            for _ in range(1000):
                launch()
            torch.accelerator.synchronize()
            samples[label].append((time.perf_counter() - started) * 1000)
    return {
        "rows": rows,
        "window": window,
        "correctness": "bit-exact",
        "scope": "host enqueue plus completion, microseconds per fill",
        "samples_us": samples,
        "median_us": {
            label: statistics.median(values) for label, values in samples.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "status": "research-only",
        "scope": "isolated metadata-fill proxy",
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cases": [],
    }
    report["cases"] = [benchmark_case(rows) for rows in (1, 4, 16)]
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
