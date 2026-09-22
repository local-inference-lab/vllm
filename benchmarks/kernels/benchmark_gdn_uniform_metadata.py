# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolated uniform and cross-group GDN metadata timing, not serving latency."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from vllm.v1.attention.backends.b12x_gdn_metadata import B12xGdnMixedMetadata
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


def benchmark_group_refresh(non_spec: int, spec: int, groups: int) -> dict:
    from flashinfer.testing import bench_gpu_time_with_cupti

    device = torch.device("cuda")

    def allocate():
        return B12xGdnMixedMetadata(
            max_tokens=1024, max_seqs=2, state_columns=4, device=device
        )

    source = allocate()
    source._num_non_spec = non_spec
    source._num_spec = spec
    source.request_rows.copy_(torch.arange(2, device=device))
    source.spec_request_rows.copy_(torch.arange(2, device=device))
    source.checkpoint_columns.fill_(1)
    for index, values in enumerate(source._worklists):
        if any(
            values is rows
            for rows in (
                source.request_rows,
                source.spec_request_rows,
                source.checkpoint_columns,
            )
        ):
            continue
        values.fill_(index + 1)
    states = [
        torch.arange(16, dtype=torch.int32, device=device).view(2, 8) + group * 100
        for group in range(groups)
    ]
    tables = [
        torch.arange(12, dtype=torch.int32, device=device).view(2, 6) + group * 1000
        for group in range(groups)
    ]
    reference = [allocate() for _ in range(groups)]
    candidate = [allocate() for _ in range(groups)]

    def copies():
        for dest, state, table in zip(reference, states, tables):
            dest.copy_worklists_from(source)
            dest.refresh_state_indices(state, table)

    def fused():
        for dest, state, table in zip(candidate, states, tables):
            dest.copy_and_refresh_from(source, state, table)

    def tensors(metadata):
        return (
            *metadata._worklists,
            metadata.state_indices,
            metadata.spec_state_indices,
            metadata.checkpoint.state_indices,
        )

    copies()
    fused()
    for actual, expected in zip(candidate, reference):
        for a, b in zip(tensors(actual), tensors(expected)):
            torch.testing.assert_close(a, b, atol=0, rtol=0)

    samples = {"gpu_us": {}, "wall_us": {}}
    for label, launch in (
        ("copies", copies),
        ("fused", fused),
        ("fused", fused),
        ("copies", copies),
    ):
        gpu_ms = bench_gpu_time_with_cupti(
            launch,
            use_cuda_graph=True,
            cold_l2_cache=True,
            dry_run_iters=5,
            repeat_iters=30,
        )
        samples["gpu_us"].setdefault(label, []).extend(v * 1000 for v in gpu_ms)
        for _ in range(3):
            torch.accelerator.synchronize()
            started = time.perf_counter()
            for _ in range(100):
                launch()
            torch.accelerator.synchronize()
            elapsed_us = (time.perf_counter() - started) * 1e6 / 100
            samples["wall_us"].setdefault(label, []).append(elapsed_us)
    return {
        "non_spec": non_spec,
        "spec": spec,
        "groups": groups,
        "correctness": "bit-exact",
        **samples,
        "median_us": {
            scope: {label: statistics.median(v) for label, v in values.items()}
            for scope, values in samples.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-refresh", action="store_true")
    parser.add_argument("--groups", type=int, default=35)
    args = parser.parse_args()
    report = {
        "status": "research-only",
        "scope": "isolated metadata-fill proxy",
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cases": [],
    }
    if args.group_refresh:
        if args.groups <= 0:
            parser.error("--groups must be positive")
        report["scope"] = "cross-group metadata refresh; graph cold-L2 and host timing"
        report["cases"] = [
            benchmark_group_refresh(non_spec, spec, args.groups)
            for non_spec, spec in ((0, 1), (1, 0), (1, 1))
        ]
    else:
        report["cases"] = [benchmark_case(rows) for rows in (1, 4, 16)]
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
