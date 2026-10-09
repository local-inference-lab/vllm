# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure CPU-only lookup cost for declared B12X PCIe collective shapes.

No GPU operation is executed. Use --module-file to compare pinned source files
in the same Python/runtime environment; these timings are not serving throughput.
"""

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
import timeit
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module-file", type=Path, required=True)
    parser.add_argument("--declarations", type=int, nargs="+", default=[64, 512, 4096])
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if min(*args.declarations, args.iterations, args.repeats) < 1:
        parser.error("counts must be positive")
    spec = importlib.util.spec_from_file_location("pcie_lookup_probe", args.module_file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    report = {
        "source_sha256": hashlib.sha256(args.module_file.read_bytes()).hexdigest(),
        "iterations": args.iterations,
        "repeats": args.repeats,
        "scope": "CPU metadata lookup; no collective or model execution",
        "results": [],
    }
    for count in args.declarations:
        communicator = object.__new__(module.B12xPcieAllReduce)
        invocations = [
            module.B12xPcieInvocation(
                name=str(rows),
                operation="all_reduce",
                shape=(rows, 16),
                dtype=torch.bfloat16,
            )
            for rows in range(1, count + 1)
        ]
        communicator._invocations = {item.name: item for item in invocations}
        communicator._plans = {item.name: object() for item in invocations}
        if hasattr(communicator, "_index_declared_plans"):
            communicator._index_declared_plans()
        for case, rows, expected in (("hit", count, True), ("miss", count + 1, False)):
            source = torch.empty((rows, 16), dtype=torch.bfloat16)

            def lookup(communicator=communicator, source=source):
                return communicator._has_plan_for(source)

            assert lookup() is expected
            for _ in range(100):
                lookup()
            samples = timeit.repeat(lookup, number=args.iterations, repeat=args.repeats)
            report["results"].append(
                {
                    "declarations": count,
                    "case": case,
                    "microseconds_per_call": [
                        s * 1e6 / args.iterations for s in samples
                    ],
                    "median_us": statistics.median(samples) * 1e6 / args.iterations,
                }
            )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
