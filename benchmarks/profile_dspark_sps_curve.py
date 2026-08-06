# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profile the engine step-rate curve for the DSpark prefix scheduler.

Times the captured FULL cudagraph replays of the target verification step at
every captured batch token count and emits ``dspark_sps_curve`` breakpoints,
one per capture size. The scheduler linearly interpolates between
breakpoints, which amortizes cudagraph padding smoothly instead of
concentrating it into thresholds at capture-size boundaries.

Example:
    python benchmarks/profile_dspark_sps_curve.py <target-model> \\
        --speculative-config '{"method": "dspark", "model": "...", ...}' \\
        --engine-args '{"tensor_parallel_size": 4, "max_num_seqs": 32}'

Paste the printed ``dspark_sps_curve`` entry into --speculative-config.

Caveats: replays run on whatever (dummy) buffer state capture left behind, so
data-dependent kernels (e.g. MoE routing) may be timed on unrepresentative
inputs, and per-step CPU/draft overhead is modeled only through the constant
``--overhead-ms``. Only the curve's shape matters to the scheduler.
"""

import argparse
import json


def _time_fullgraph_replays(worker, iters: int, warmup: int) -> dict[int, float]:
    """Worker-side: time FULL graph replay per batch token count (ms/step).

    Runs on every TP rank via collective_rpc so the collectives captured in
    the graphs stay matched; every rank replays the same descs in the same
    sorted order. Before timing each descriptor the input buffers are
    refreshed into the same coherent dummy state capture used, so replays
    never read stale metadata.
    """
    import torch

    from vllm.v1.worker.gpu.cudagraph_utils import prepare_inputs_to_capture

    runner = worker.model_runner
    mgr = runner.cudagraph_manager
    assert mgr is not None and mgr.graphs, (
        "No FULL cudagraphs captured; run with a cudagraph_mode that captures "
        "FULL decode graphs."
    )
    # Prefer varlen spec-decode descs; fall back to all captured graphs.
    descs = [d for d in mgr.graphs if d.max_req_tokens is not None]
    if not descs:
        descs = list(mgr.graphs.keys())
    # One desc per token count: the largest request count is the most
    # representative shape under load.
    by_tokens: dict[int, object] = {}
    for d in descs:
        cur = by_tokens.get(d.num_tokens)
        if cur is None or (d.num_reqs or 0) > (cur.num_reqs or 0):
            by_tokens[d.num_tokens] = d

    results: dict[int, float] = {}
    for num_tokens in sorted(by_tokens):
        desc = by_tokens[num_tokens]
        num_reqs = desc.num_reqs or min(num_tokens, mgr.max_num_reqs)
        prepare_inputs_to_capture(
            num_reqs,
            num_tokens,
            runner.model_state,
            runner.input_buffers,
            runner.block_tables,
            runner.attn_groups,
            runner.kv_cache_config,
            max_req_tokens=desc.max_req_tokens,
        )
        graph = mgr.graphs[desc]
        for _ in range(warmup):
            graph.replay()
        torch.accelerator.synchronize()
        start = torch.Event(enable_timing=True)
        end = torch.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        torch.accelerator.synchronize()
        results[num_tokens] = start.elapsed_time(end) / iters
    return results


def curve_breakpoints(
    ms_per_step: dict[int, float], overhead_ms: float
) -> list[list[float]]:
    """Convert per-capture-size step times into ``dspark_sps_curve``
    breakpoints, one per capture size. The scheduler's table linearly
    interpolates between them (and clamps at the ends)."""
    return [
        [size, round(1000.0 / (ms_per_step[size] + overhead_ms), 3)]
        for size in sorted(ms_per_step)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Target model (path or HF id)")
    parser.add_argument(
        "--speculative-config",
        required=True,
        help="JSON speculative config (same value you pass to vllm serve)",
    )
    parser.add_argument(
        "--engine-args",
        default="{}",
        help="JSON dict of extra vllm.LLM kwargs "
        '(e.g. \'{"tensor_parallel_size": 4, "max_num_seqs": 32}\')',
    )
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--overhead-ms",
        type=float,
        default=0.0,
        help="Constant per-step overhead (draft forward, sampling, CPU gap) "
        "added to every measured step time before converting to a rate.",
    )
    parser.add_argument("--output", help="Write the curve JSON to this file")
    args = parser.parse_args()
    if args.iters <= 0:
        parser.error("--iters must be greater than zero")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.overhead_ms < 0:
        parser.error("--overhead-ms must be non-negative")

    # The timing callable is shipped to the workers via collective_rpc, which
    # requires the pickle fallback. Local profiling tool, trusted input.
    import os

    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    from vllm import LLM

    llm = LLM(
        model=args.model,
        speculative_config=json.loads(args.speculative_config),
        **json.loads(args.engine_args),
    )
    per_rank = llm.collective_rpc(
        _time_fullgraph_replays, kwargs={"iters": args.iters, "warmup": args.warmup}
    )
    ms_per_step = per_rank[0]

    print("\nMeasured FULL-graph step times (rank 0):")
    for size in sorted(ms_per_step):
        print(f"  B={size:5d} tokens: {ms_per_step[size]:8.3f} ms/step")

    curve = curve_breakpoints(ms_per_step, args.overhead_ms)
    entry = {"dspark_sps_curve": curve}
    print("\nAdd to --speculative-config:")
    print(json.dumps(entry))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(entry, f, indent=2)
        print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
