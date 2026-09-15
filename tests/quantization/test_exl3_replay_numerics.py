# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A replayed Trellis graph must agree with eager, not merely complete.

Capturing and replaying once proves the region is legal to record. It does not
prove the replay reads the current inputs: a graph that quietly reuses the
values present at capture time replays happily and returns stale results. These
tests drive the same layer through several rounds of changed activations and
changed routing, compare each replay against an eager call on the same inputs,
and separately assert the outputs actually move between rounds -- without that
second check, the comparison would pass trivially on a frozen graph.

The expert-parallel case is here for a different reason. Under EP the routing
space is global while the rank owns a slice of it, so a batch can select only
experts this rank does not hold. The correct result is no contribution at all,
and the failure mode is silent: whatever the output buffer happened to contain.
The buffer is poisoned before the call so that reuse shows up as a failure
rather than as plausible numbers.
"""
import importlib

import pytest
import torch

TOPK = 2
HIDDEN = 512
INTERMEDIATE = 256
TILE_CONFIG = (64, 128, 64, 128)


def _b12x():
    try:
        return (
            importlib.import_module("b12x.moe._shared.kernels.w4a16.prepare"),
            importlib.import_module("b12x.moe._shared.kernels.w4a16.kernel"),
            importlib.import_module("b12x.moe._shared.kernels.w4a16.host"),
        )
    except Exception:  # noqa: BLE001
        pytest.skip("B12X w4a16 Trellis kernels are unavailable")


def _build(*, experts, bits, tokens, block_m, rotation_dtype, expert_map=None,
           global_experts=None):
    prepare, kernel, host = _b12x()
    device = torch.device("cuda")
    gen = torch.Generator(device="cpu").manual_seed(1234)

    w13 = torch.randint(
        -32768, 32767,
        (2, experts, HIDDEN // 16, INTERMEDIATE // 16, 16 * bits),
        generator=gen, dtype=torch.int16,
    ).to(device)
    w2 = torch.randint(
        -32768, 32767,
        (experts, INTERMEDIATE // 16, HIDDEN // 16, 16 * bits),
        generator=gen, dtype=torch.int16,
    ).to(device)
    suh = torch.ones((2, experts, HIDDEN), dtype=torch.float16, device=device)
    svh = torch.ones((2, experts, INTERMEDIATE), dtype=torch.float16, device=device)
    down_suh = torch.ones((experts, INTERMEDIATE), dtype=torch.float16, device=device)
    down_svh = torch.ones((experts, HIDDEN), dtype=torch.float16, device=device)
    rotations = torch.cat((svh[0], svh[1], down_suh), dim=1).contiguous()

    props = torch.cuda.get_device_properties(device)
    sms = int(props.multi_processor_count)
    workspace = torch.zeros(sms * 4 + 2, dtype=torch.int32, device=device)

    prepared = prepare.prepare_trellis256_moe_weights(
        w13=w13, w2=w2, hidden_size=HIDDEN, intermediate_size=INTERMEDIATE,
        num_experts=experts, activation="silu",
        fc1_tile_n=TILE_CONFIG[1], fc2_tile_n=TILE_CONFIG[3],
        params_dtype=torch.float16, w13_layout="trellis_t256_proj",
        trellis_bits=bits, codebook="mcg",
        gate_suh=suh[0], up_suh=suh[1], intermediate_rotations=rotations,
        down_svh=down_svh, tile_config=TILE_CONFIG, workspace=workspace,
    )

    direct_routes = expert_map is not None
    if direct_routes:
        # The direct form indexes one block per routed row; the packed form
        # derives its count from the route pack. The kernel checks the planned
        # count against exactly the one implied by the routing form in use.
        max_m_blocks = tokens * TOPK
    else:
        slots = host.max_packed_route_slots(tokens * TOPK, block_m, experts)
        max_m_blocks = (slots + block_m - 1) // block_m

    launch = kernel.compile_w4a16_fused_moe(
        size_m=tokens, hidden_size=HIDDEN, intermediate_size=INTERMEDIATE,
        num_experts=experts, top_k=TOPK, activation="silu",
        apply_router_weight_on_input=False, zero_fc2_output=False,
        moe_block_size=block_m, max_m_blocks=max_m_blocks, sms=sms,
        max_shared_mem=int(props.shared_memory_per_block_optin),
        element_dtype="fp16", weight_layout=prepared.weight_layout,
        scale_format=prepared.scale_format, w13_layout=prepared.w13_layout,
        trellis_bits=bits, trellis_codebook="mcg", force_tile_config=TILE_CONFIG,
        full_rotation=True, intermediate_rotation=True,
        rotation_input_dtype=rotation_dtype,
        direct_topk_routes=direct_routes, use_expert_map=direct_routes,
    )
    buffers = host.make_w4a16_packed_buffers(
        prepared, m=tokens, topk=TOPK, dtype=torch.float16, device=device,
        route_num_experts=(int(global_experts) if direct_routes else None),
        full_rotation=True, block_size_m=block_m,
    )
    return kernel, prepared, launch, buffers, device


def _runner(kernel, prepared, launch, buffers, block_m, x, topk_weights, topk_ids,
            expert_map=None):
    trellis = prepared.trellis

    def run():
        return kernel.run_w4a16_moe(
            x, prepared, topk_weights, topk_ids, activation="silu",
            intermediate_cache13=buffers.intermediate_cache13,
            intermediate_cache2=buffers.intermediate_cache2,
            output=buffers.output[: x.shape[0]],
            fc1_c_tmp=buffers.fc1_c_tmp, fc2_c_tmp=buffers.fc2_c_tmp,
            packed_route_indices=buffers.packed_route_indices,
            block_expert_ids=buffers.block_expert_ids,
            packed_route_count=buffers.packed_route_count,
            expert_offsets=buffers.expert_offsets,
            expert_counts=buffers.expert_counts,
            expert_map=expert_map, output_expert_map=expert_map,
            fused_launch=launch, route_block_size_m=block_m,
            intermediate_rotation_scales=trellis.intermediate_rotations,
            suh_gate_table=trellis.gate_suh, suh_up_table=trellis.up_suh,
            svh_table=trellis.down_svh,
            rotation_a_gate=buffers.rotation_a_gate,
            rotation_a_up=buffers.rotation_a_up,
            full_rotation=True,
        )

    return run


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "bits,activation_dtype,rotation_dtype",
    [(3, torch.float16, "fp16"), (4, torch.float16, "fp16"),
     (4, torch.bfloat16, "bf16")],
)
def test_replay_matches_eager_across_changing_routes(bits, activation_dtype,
                                                     rotation_dtype):
    tokens, block_m, experts = 8, 8, 8
    kernel, prepared, launch, buffers, device = _build(
        experts=experts, bits=bits, tokens=tokens, block_m=block_m,
        rotation_dtype=rotation_dtype,
    )
    x = torch.randn((tokens, HIDDEN), dtype=activation_dtype, device=device)
    topk_weights = torch.rand((tokens, TOPK), dtype=torch.float32, device=device)
    topk_ids = torch.randint(0, experts, (tokens, TOPK), dtype=torch.int32,
                             device=device)
    run = _runner(kernel, prepared, launch, buffers, block_m, x, topk_weights,
                  topk_ids)

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    torch.cuda.synchronize()

    for _ in range(6):
        # Write through the captured storage; a graph only ever sees these.
        x.copy_(torch.randn((tokens, HIDDEN), dtype=activation_dtype, device=device))
        topk_weights.copy_(
            torch.rand((tokens, TOPK), dtype=torch.float32, device=device)
        )
        topk_ids.copy_(
            torch.randint(0, experts, (tokens, TOPK), dtype=torch.int32,
                          device=device)
        )
        graph.replay()
        torch.cuda.synchronize()
        replayed = buffers.output[:tokens].clone()
        run()
        torch.cuda.synchronize()
        eager = buffers.output[:tokens].clone()
        torch.testing.assert_close(
            replayed.float(), eager.float(), rtol=0, atol=0, equal_nan=True
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_replay_observes_new_inputs():
    """Guards the test above: equality is vacuous if the output never moves."""
    tokens, block_m, experts, bits = 8, 8, 8, 4
    kernel, prepared, launch, buffers, device = _build(
        experts=experts, bits=bits, tokens=tokens, block_m=block_m,
        rotation_dtype="fp16",
    )
    x = torch.randn((tokens, HIDDEN), dtype=torch.float16, device=device)
    topk_weights = torch.rand((tokens, TOPK), dtype=torch.float32, device=device)
    topk_ids = torch.randint(0, experts, (tokens, TOPK), dtype=torch.int32,
                             device=device)
    run = _runner(kernel, prepared, launch, buffers, block_m, x, topk_weights,
                  topk_ids)
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    seen = []
    for _ in range(4):
        x.copy_(torch.randn((tokens, HIDDEN), dtype=torch.float16, device=device))
        topk_ids.copy_(
            torch.randint(0, experts, (tokens, TOPK), dtype=torch.int32,
                          device=device)
        )
        graph.replay()
        torch.cuda.synchronize()
        seen.append(buffers.output[:tokens].clone())

    distinct = sum(1 for other in seen[1:] if not torch.equal(seen[0], other))
    assert distinct >= 2, "replayed output did not track its inputs"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_expert_parallel_contributes_nothing_when_no_route_is_local():
    """A batch selecting only non-resident experts must produce exactly zero."""
    local_experts, global_experts = 4, 8
    tokens, block_m, bits = 8, 8, 4
    device = torch.device("cuda")

    # This rank owns globals 0..local_experts-1; the rest map to -1.
    expert_map = torch.full((global_experts,), -1, dtype=torch.int32, device=device)
    for local, global_id in enumerate(range(local_experts)):
        expert_map[global_id] = local

    kernel, prepared, launch, buffers, device = _build(
        experts=local_experts, bits=bits, tokens=tokens, block_m=block_m,
        rotation_dtype="fp16", expert_map=expert_map,
        global_experts=global_experts,
    )
    x = torch.randn((tokens, HIDDEN), dtype=torch.float16, device=device)
    topk_weights = torch.rand((tokens, TOPK), dtype=torch.float32, device=device)
    # Every selection is an expert this rank does not hold.
    topk_ids = torch.randint(local_experts, global_experts, (tokens, TOPK),
                             dtype=torch.int32, device=device)

    # Poison the destination: a stale-reuse bug then shows up as NaN, not as
    # numbers that happen to look reasonable.
    buffers.output.fill_(float("nan"))

    run = _runner(kernel, prepared, launch, buffers, block_m, x, topk_weights,
                  topk_ids, expert_map=expert_map)
    out = run()
    torch.cuda.synchronize()

    assert bool((out == 0).all()), "non-resident routes left stale output behind"
