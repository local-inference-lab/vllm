# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time the actual padded Kimi up-projection under both input-packing modes.

Run on an idle GPU with the same vLLM build as serving. Both modes call the
normal unquantized GEMM dispatcher with the same K=400 weight. Timing is
interleaved CUDA graph replay, after a bitwise output comparison. This is a
kernel diagnostic, not an end-to-end serving-performance verdict.
"""

import argparse
import hashlib
import json
import os
import statistics

import torch
from torch import nn

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.models.kimi_k3.nvidia import model


def digest(tensor):
    return hashlib.sha256(
        tensor.contiguous().view(torch.uint8).cpu().numpy()
    ).hexdigest()


def make_projection(rank, weight):
    projection = model.KimiPaddedRowParallelLinear.__new__(
        model.KimiPaddedRowParallelLinear
    )
    nn.Module.__init__(projection)
    projection.input_is_parallel = False
    projection.tp_size = 9
    projection.tp_rank = rank
    projection.logical_input_size = 3584
    projection.input_pad = 16
    projection.output_size = 7168
    projection.reduce_results = False
    projection.return_bias = True
    projection.skip_bias_add = False
    projection.quant_method = UnquantizedLinearMethod()
    projection.register_parameter("bias", None)
    projection.weight = nn.Parameter(weight, requires_grad=False)
    return projection


def capture(projection, x, enabled):
    model.kimi_decode_shard_pack_enabled = lambda: enabled
    for _ in range(3):
        projection(x)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, _ = projection(x)
    graph.replay()
    torch.accelerator.synchronize()
    return graph, out


def time_graph(graph, iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=400)
    args = parser.parse_args()
    props = torch.cuda.get_device_properties(0)
    report = {
        "schema": "kimi.decode_shard_pack.v1",
        "source_revision": os.getenv("K3_SOURCE_REVISION"),
        "device": str(props),
        "physical_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "precision": "bf16, no arithmetic change, identical padded K=400",
        "ratio_direction": "candidate_over_control, lower_is_better",
        "records": [],
    }
    torch.manual_seed(20260907)
    for rank in range(9):
        weight = torch.randn(7168, 400, device="cuda", dtype=torch.bfloat16)
        if rank == 8:
            weight[:, 384:] = 0
        projection = make_projection(rank, weight)
        for rows in (1, 2, 4, 8, 9, 16):
            x = torch.randn(rows, 3584, device="cuda", dtype=torch.bfloat16)
            control, before = capture(projection, x, False)
            candidate, after = capture(projection, x, True)
            before_digest, after_digest = digest(before), digest(after)
            assert before_digest == after_digest, (rank, rows, "output mismatch")
            ptr = after.data_ptr()
            for step in range(3):
                x.normal_()
                control.replay()
                candidate.replay()
                assert torch.equal(before.view(torch.uint8), after.view(torch.uint8))
                assert after.data_ptr() == ptr
            samples = {"control": [], "candidate": []}
            for trial in range(6):
                arms = [("control", control), ("candidate", candidate)]
                if trial % 2:
                    arms.reverse()
                for name, graph in arms:
                    samples[name].append(time_graph(graph, args.iterations))
            before_us = statistics.median(samples["control"])
            after_us = statistics.median(samples["candidate"])
            row = {
                "rank": rank,
                "rows": rows,
                "control_sha256": before_digest,
                "candidate_sha256": after_digest,
                "exact": True,
                "replay_mutations": 3,
                "control_us": before_us,
                "candidate_us": after_us,
                "ratio": after_us / before_us,
                "samples_us": samples,
            }
            report["records"].append(row)
            print(json.dumps(row), flush=True)
            del control, candidate
    report["passed"] = True
    with open(args.output, "w") as stream:
        json.dump(report, stream, indent=2)
    print("projection_shard_pack: PASS", flush=True)


if __name__ == "__main__":
    main()
