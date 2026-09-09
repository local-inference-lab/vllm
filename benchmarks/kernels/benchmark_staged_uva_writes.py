# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare staged token-table updates against a supplied reference module.

Both modules use the same CUDA runtime and pinned-host target geometry. Timings
include CPU staging, transfer, the write kernel, and completion synchronization;
they exclude stage_write() and do not measure model throughput. Run on an idle
GPU and retain independent service-activity observations for shared systems.
"""

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import torch

from vllm.v1.worker.gpu import buffer_utils


def load_reference(path: Path):
    spec = importlib.util.spec_from_file_location("staged_write_reference", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def stage(state, rows: int, values: list[int]) -> None:
    for row in range(rows):
        state.stage_write(row, 0, values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    reference = load_reference(args.reference)
    device = torch.device("cuda:0")
    torch.accelerator.set_device_index(device.index)
    max_rows = 4
    max_length = 1048576
    variants = {
        "gpu_contents": reference.StagedWriteTensor(
            (max_rows, max_length),
            torch.int32,
            device,
            max_concurrency=2,
            uva_instead_of_gpu=True,
        ),
        "uva_contents": buffer_utils.StagedWriteTensor(
            (max_rows, max_length),
            torch.int32,
            device,
            max_concurrency=2,
            uva_instead_of_gpu=True,
        ),
    }
    result = {
        "status": "research-only timing until service activity is checked",
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(
            Path(buffer_utils.__file__).read_bytes()
        ).hexdigest(),
        "target_shape": [max_rows, max_length],
        "dtype": "int32",
        "iterations": args.iterations,
        "order": "alternating reference/candidate each iteration",
        "boundary": "apply_write through device completion; stage_write excluded",
        "cases": [],
    }

    for rows, length in (
        (1, 4),
        (1, 4608),
        (4, 4608),
        (1, 65536),
        (1, 131072),
        (1, 1048576),
    ):
        values = list(range(length))

        # Both rotating slots and the allocator/kernel caches are warmed.
        for _ in range(4):
            for state in variants.values():
                stage(state, rows, values)
                state.apply_write()
                torch.accelerator.synchronize()

        samples = {name: [] for name in variants}
        for iteration in range(args.iterations):
            order = list(variants)
            if iteration % 2:
                order.reverse()
            for name in order:
                state = variants[name]
                stage(state, rows, values)
                torch.accelerator.synchronize()
                start = time.perf_counter_ns()
                state.apply_write()
                torch.accelerator.synchronize()
                samples[name].append((time.perf_counter_ns() - start) / 1000)

        expected = torch.arange(length, dtype=torch.int32, device="cpu")
        hashes = {}
        for name, state in variants.items():
            assert state.cpu is not None
            for row in range(rows):
                torch.testing.assert_close(
                    state.cpu[row, :length], expected, rtol=0, atol=0
                )
            hashes[name] = hashlib.sha256(
                state.cpu[:rows, :length].contiguous().numpy().tobytes()
            ).hexdigest()
        assert len(set(hashes.values())) == 1
        medians = {name: statistics.median(s) for name, s in samples.items()}
        record = {
            "rows": rows,
            "elements_per_row": length,
            "output_sha256": hashes,
            "completion_samples_us": samples,
            "median_us": medians,
            "reference_over_candidate": (
                medians["gpu_contents"] / medians["uva_contents"]
            ),
        }
        result["cases"].append(record)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                {k: v for k, v in record.items() if k != "completion_samples_us"}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
