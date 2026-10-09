# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint-backed DSV4.1 MoE/shared-expert stream-overlap diagnostic.

Runs one TP shard with fixed serving configurations and identical routes. Timing
includes both expert branches and their addition, but excludes router logits,
top-k selection and TP reduction. Requires the b12x repository benchmark helpers.
"""

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

import torch
from safetensors import safe_open

from vllm import _custom_ops  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b12x-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--retune-overlapped", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.b12x_root))

    from b12x._lib.compiler import b12x_package_fingerprint
    from b12x.gemm import block_fp8_linear as bfl
    from b12x.moe import fused_moe
    from b12x.moe.fused_moe.workloads import (
        TUNING_WORKLOAD_VERSION,
        make_tuning_routes,
    )
    from b12x.preparation import PreparationSession, PreparedCall, require_prepared
    from tests.gemm.test_fp8_quant_deepgemm_parity import _per_token_cast_to_fp8

    from benchmarks import benchmark_moe as harness
    from benchmarks.common import make_l2_flush_fn, nvidia_smi_gpu_mode_snapshot
    from benchmarks.dense_autotune_common import bench_events
    from benchmarks.moe_preparation import request_for_capacity

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda", 0)
    profile = harness.MODEL_PROFILES["deepseek-v4.1-flash"]
    spec = harness.build_model_spec(
        args.checkpoint,
        profile,
        tp_size_override=args.tp,
        tp_rank=args.rank,
        layer_idx=args.layer,
    )
    weights = harness.load_expert_weights(
        args.checkpoint,
        spec,
        layer_idx=args.layer,
        checkpoint_family=profile.checkpoint_family,
    )
    params = harness.get_quant_mode_params(weights, "per-expert", "w4a8_mx")
    activation = harness.ActivationParams(swiglu_limit=10.0)
    selections = json.loads(args.selections.read_text())
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    tensors, weight_hashes = {}, {}
    for role in ("w1", "w3", "w2"):
        for suffix in ("weight", "scale"):
            name = f"layers.{args.layer}.ffn.shared_experts.{role}.{suffix}"
            with safe_open(
                args.checkpoint / index[name], framework="pt", device="cpu"
            ) as f:
                value = f.get_tensor(name).clone()
            weight_hashes[name] = hashlib.sha256(
                value.view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            dim = 1 if role == "w2" else 0
            shard = value.shape[dim] // args.tp
            assert value.shape[dim] % args.tp == 0
            tensors[role, suffix] = value.narrow(dim, args.rank * shard, shard).cuda()
    raw_shared = {
        "gate": tuple(
            torch.cat([tensors[r, s].view(torch.uint8) for r in ("w1", "w3")]).view(
                tensors["w1", s].dtype
            )
            for s in ("weight", "scale")
        ),
        "down": (tensors["w2", "weight"], tensors["w2", "scale"]),
    }
    decoded = {
        role: w.float() * s.float().repeat_interleave(32, 0).repeat_interleave(32, 1)
        for role, (w, s) in raw_shared.items()
    }
    packed = {
        role: bfl.pack_weight(w, s, block_size=(32, 32))
        for role, (w, s) in raw_shared.items()
    }

    def quantized_input(x):
        values, scales = _per_token_cast_to_fp8(x, 32)
        return values.float() * scales.repeat_interleave(32, 1)

    def shared_reference(x):
        gate_up = (quantized_input(x) @ decoded["gate"].T).to(torch.bfloat16)
        gate, up = gate_up.float().chunk(2, dim=-1)
        activated = (
            torch.nn.functional.silu(gate.clamp(max=10)) * up.clamp(-10, 10)
        ).to(torch.bfloat16)
        return (quantized_input(activated) @ decoded["down"].T).to(torch.bfloat16)

    cases = {}
    for m in args.rows:
        torch.manual_seed(42)
        x = torch.empty(
            m, spec.hidden_size, device=device, dtype=torch.bfloat16
        ).normal_(0, 0.125)
        routes = make_tuning_routes(m, spec.top_k, spec.num_experts, device=device)
        route_weights = (
            torch.softmax(
                torch.arange(spec.top_k, device=device, dtype=torch.float32) * 0.125,
                dim=-1,
            )
            .expand(m, -1)
            .contiguous()
        )
        routed_refs = [
            harness.make_oracle_reference(
                "w4a8_mx",
                "w4a8_mx",
                x,
                weights,
                params,
                ids,
                route_weights,
                activation="silu",
                activation_params=activation,
            ).cpu()
            for ids in routes
        ]
        cases[m] = (x, routes, route_weights, routed_refs, shared_reference(x).cpu())
        print("ORACLES", m, flush=True)
    experts, _ = harness.prepare_b12x_benchmark_weights(
        weights,
        params,
        quant_mode="w4a8_mx",
        activation="silu",
        activation_params=activation,
    )
    flush = make_l2_flush_fn(True)
    with args.output.open("x") as log:

        def record(data):
            log.write(json.dumps(data) + "\n")
            log.flush()

        record(
            dict(
                kind="manifest",
                command=[sys.executable, *sys.argv],
                geometry=asdict(spec),
                corpus=TUNING_WORKLOAD_VERSION,
                checkpoint=str(args.checkpoint),
                layer=args.layer,
                source_fingerprint=b12x_package_fingerprint(),
                vllm_revision=subprocess.check_output(
                    [
                        "git",
                        "-c",
                        f"safe.directory={Path(__file__).resolve().parents[2]}",
                        "-C",
                        str(Path(__file__).resolve().parents[2]),
                        "rev-parse",
                        "HEAD",
                    ],
                    text=True,
                ).strip(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                selections=selections,
                weight_hashes=weight_hashes,
                device=nvidia_smi_gpu_mode_snapshot(),
                timing=(
                    "CUDA events around full graph; read-based L2 eviction outside "
                    "timing; no router logits/top-k or TP collective"
                ),
            )
        )

        def run_case(m, case):
            x, routes, route_weights, routed_refs, shared_ref = case
            choice = selections[str(m)]
            moe = fused_moe.plan_execution(
                experts=experts,
                capacity=fused_moe.ExecutionCapacity(
                    max_tokens=m,
                    top_k=spec.top_k,
                    warmup_token_counts=(m,),
                    route_num_experts=0,
                ),
                override=fused_moe.MoeDecodeConfig(**choice["moe"]),
            )
            linears = {
                role: bfl.plan(
                    bfl.Caps(
                        device=device,
                        max_tokens=m,
                        in_features=w.shape[1],
                        out_features=w.shape[0],
                        block_size=(32, 32),
                    ),
                    override=bfl.DenseGemmConfig(**choice[role]),
                )
                for role, (w, _) in raw_shared.items()
            }

            def moe_call(state):
                ids = routes[0].clone()
                output = torch.empty_like(x)
                scratch = tuple(
                    torch.empty(s.shape, dtype=s.dtype, device=s.device)
                    for s in state.scratch.scratch_specs()
                )
                binding = state.bind(
                    scratch=scratch,
                    a=x,
                    experts=experts,
                    topk_ids=ids,
                    topk_weights=route_weights,
                    output=output,
                    input_scales_static=True,
                )

                def reset():
                    output.zero_()
                    for t in scratch:
                        t.zero_()

                return PreparedCall(
                    run=binding.run,
                    output=output,
                    reset=reset,
                    produce=lambda: ids.copy_(routes[0]),
                    benchmark_producers=tuple(partial(ids.copy_, r) for r in routes),
                    owners=(binding, scratch, ids),
                )

            def linear_call(state, role, source=None):
                n, k = raw_shared[role][0].shape
                if source is None:
                    source = (
                        x
                        if role == "gate"
                        else torch.zeros(m, k, device=device, dtype=torch.bfloat16)
                    )
                scratch = tuple(
                    torch.empty(s.shape, dtype=s.dtype, device=s.device)
                    for s in state.scratch.scratch_specs()
                )
                output = torch.empty(m, n, 1, device=device, dtype=torch.bfloat16)
                bound = state.bind(
                    scratch=scratch,
                    source=source,
                    packed_weight=packed[role],
                    output=output,
                )
                return PreparedCall(
                    run=lambda: state.run_binding(bound),
                    output=output,
                    owners=(bound, scratch, source),
                )

            requests = [request_for_capacity(moe, name="routed", calls={m: moe_call})]
            requests.extend(
                plan.request(name=role, prepare_call=partial(linear_call, role=role))
                for role, plan in linears.items()
            )
            with PreparationSession(
                device=device, autotune=True, compile_workers=2
            ) as session:
                session.prepare(requests)
                exact_moe = getattr(moe, "variants", {}).get(m, moe)
                routed = moe_call(require_prepared(exact_moe, exact_moe.component_id))
                gate = linear_call(
                    require_prepared(linears["gate"], "gemm.block_fp8_linear"), "gate"
                )
                activated = torch.empty(
                    m, spec.intermediate_size, device=device, dtype=torch.bfloat16
                )
                down = linear_call(
                    require_prepared(linears["down"], "gemm.block_fp8_linear"),
                    "down",
                    activated,
                )
                combined = torch.empty_like(x)
                aux = torch.cuda.Stream()
                ready, done = torch.cuda.Event(), torch.cuda.Event()

                def shared():
                    gate.run()
                    torch.ops._C.silu_and_mul_with_clamp(
                        activated, gate.output[..., 0], 10.0, 1.0, 0.0
                    )
                    down.run()

                def overlapped(call):
                    main_stream = torch.cuda.current_stream()
                    ready.record(main_stream)
                    with torch.cuda.stream(aux):
                        ready.wait(aux)
                        shared()
                        done.record(aux)
                    call.run()
                    done.wait(main_stream)
                    torch.add(call.output, down.output[..., 0], out=combined)

                candidate = None
                routed_calls = {"overlap": routed, "serial": routed}
                if args.retune_overlapped:
                    candidate = fused_moe.plan_execution(
                        experts=experts,
                        capacity=fused_moe.ExecutionCapacity(
                            max_tokens=m,
                            top_k=spec.top_k,
                            warmup_token_counts=(m,),
                            route_num_experts=0,
                        ),
                        invocation={
                            "tuning_route_pattern": TUNING_WORKLOAD_VERSION,
                            "benchmark_corpus": "dsv41_shared_fp8_overlap_std0125_v1",
                        },
                    )

                    def overlap_trial(state):
                        call = moe_call(state)
                        return PreparedCall(
                            run=partial(overlapped, call),
                            output=combined,
                            reset=call.reset,
                            produce=call.produce,
                            restore=lambda: (call.reset(), call.produce()),
                            benchmark_producers=call.benchmark_producers,
                            capture_safe=False,
                        )

                    session.prepare(
                        [
                            *requests[1:],
                            replace(
                                request_for_capacity(
                                    candidate,
                                    name="routed_with_shared",
                                    calls={m: moe_call},
                                    benchmark_calls={m: overlap_trial},
                                ),
                                dependencies=("gate", "down"),
                            ),
                        ]
                    )
                    candidate = getattr(candidate, "variants", {}).get(m, candidate)
                    routed_calls["retuned_overlap"] = moe_call(
                        require_prepared(candidate, candidate.component_id)
                    )
                session.freeze()

                def run(arm):
                    if arm in ("overlap", "retuned_overlap"):
                        overlapped(routed_calls[arm])
                        return
                    elif arm == "serial":
                        routed.run()
                        shared()
                    elif arm == "routed_only":
                        routed.run()
                        return
                    elif arm == "shared_only":
                        shared()
                        return
                    torch.add(routed.output, down.output[..., 0], out=combined)

                graphs = {}
                for call in routed_calls.values():
                    call.reset()
                    call.produce()
                for arm in (*routed_calls, "routed_only", "shared_only"):
                    run(arm)
                    torch.accelerator.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with session.capture(), torch.cuda.graph(graph):
                        run(arm)
                    graphs[arm] = graph

                def check(output, reference, label, tolerance=0.02):
                    actual, expected = output.float(), reference.to(device).float()
                    assert torch.isfinite(actual).all() and torch.count_nonzero(
                        actual
                    ), label
                    rel = float((actual - expected).norm() / expected.norm())
                    cosine = float(
                        torch.nn.functional.cosine_similarity(
                            actual.flatten(), expected.flatten(), dim=0
                        )
                    )
                    assert rel < tolerance and cosine > 0.998, (label, rel, cosine)
                    return dict(relative_l2=rel, cosine=cosine)

                patterns = []
                pointers = [
                    t.data_ptr()
                    for t in (
                        x,
                        routed.output,
                        gate.output,
                        activated,
                        down.output,
                        combined,
                    )
                ]
                for p, reference in enumerate(routed_refs):
                    for call in routed_calls.values():
                        call.reset()
                        call.benchmark_producers[p]()
                    errors = {}
                    totals = []
                    for arm, call in routed_calls.items():
                        combined.fill_(float("nan"))
                        torch.accelerator.synchronize()
                        count = torch.accelerator.memory_stats()[
                            "allocation.all.allocated"
                        ]
                        graphs[arm].replay()
                        torch.accelerator.synchronize()
                        assert (
                            count
                            == torch.accelerator.memory_stats()[
                                "allocation.all.allocated"
                            ]
                        )
                        errors[arm] = {
                            "routed": check(
                                call.output, reference, arm + ".routed", 0.04
                            ),
                            "shared": check(
                                down.output[..., 0], shared_ref, arm + ".shared"
                            ),
                            "combined": check(
                                combined,
                                reference + shared_ref,
                                arm + ".combined",
                                0.04,
                            ),
                        }
                        totals.append(combined.clone())
                    for actual in totals[1:]:
                        check(totals[0], actual, "matching-input arms", 0.004)
                    samples = {arm: [] for arm in graphs}
                    for order in (tuple(graphs), tuple(reversed(graphs))):
                        for arm in order:
                            samples[arm].extend(
                                t * 1000
                                for t in bench_events(
                                    graphs[arm].replay,
                                    warmup=10,
                                    iters=args.iters,
                                    l2_flush=flush,
                                )
                            )
                    assert pointers == [
                        t.data_ptr()
                        for t in (
                            x,
                            routed.output,
                            gate.output,
                            activated,
                            down.output,
                            combined,
                        )
                    ]
                    for arm in routed_calls:
                        graphs[arm].replay()
                        torch.accelerator.synchronize()
                        check(combined, reference + shared_ref, arm + ".after", 0.04)
                    patterns.append(
                        dict(
                            sharing_percent=p * 20,
                            routes=routes[p].cpu().tolist(),
                            unique_experts=int(routes[p].unique().numel()),
                            oracle=errors,
                            samples_us=samples,
                            medians_us={
                                arm: statistics.median(s) for arm, s in samples.items()
                            },
                        )
                    )
                averages = {
                    arm: statistics.mean(p["medians_us"][arm] for p in patterns)
                    for arm in graphs
                }
                record(
                    dict(
                        kind="case",
                        rows=m,
                        patterns=patterns,
                        mean_pattern_median_us=averages,
                        changed_route_replay=True,
                        allocation_delta=0,
                        stable_pointers=True,
                        selections={
                            "moe": asdict(exact_moe.selection.config),
                            "retuned_moe": asdict(candidate.selection.config)
                            if candidate
                            else None,
                            "retuned_source": candidate.selection.source
                            if candidate
                            else None,
                            **{
                                role: asdict(p.selection.config)
                                for role, p in linears.items()
                            },
                        },
                        device=nvidia_smi_gpu_mode_snapshot(),
                    )
                )
                print("RESULT", m, averages, flush=True)
                if args.profile:
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]
                    ) as prof:
                        for arm in routed_calls:
                            for p in (0, 2, 4):
                                routed_calls[arm].benchmark_producers[p]()
                                flush()
                                torch.accelerator.synchronize()
                                with torch.profiler.record_function(
                                    f"{arm}.m{m}.sharing{p * 20}"
                                ):
                                    graphs[arm].replay()
                                    torch.accelerator.synchronize()
                    prof.export_chrome_trace(
                        str(args.profile.with_name(f"{args.profile.stem}-m{m}.json"))
                    )
                for graph in graphs.values():
                    graph.reset()

        for m, case in cases.items():
            run_case(m, case)
        record(dict(kind="result", correctness="passed"))


if __name__ == "__main__":
    main()
