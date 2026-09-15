# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Uniform Trellis routed experts must be CUDA-graph capturable.

The EXL3 backend exempts routed-expert checkpoints from the eager requirement on
the grounds that everything they need is planned before capture: weights are
prepared, the launch is compiled, and the route-pack/staging buffers are
allocated during the profile pass, so the replayed region allocates nothing.

That is a property of the B12X launch contract, and B12X 1.3.0 moved it. These
tests pin it down at every supported bitrate on a synthetic layer small enough
to run anywhere, so a future contract change fails here rather than as a fault
part-way through capturing a real model.

The weight payloads are synthetic. This exercises the contract -- preparation,
compilation, buffer planning, and a captured replay -- not numerical accuracy,
which belongs to a checkpoint-level test.
"""

import importlib

import pytest
import torch

BITRATES = (3, 4, 5, 6)

# Small enough to be cheap, large enough to satisfy the tile geometry.
NUM_EXPERTS = 8
HIDDEN = 512
INTERMEDIATE = 256
TOPK = 2
TOKENS = 8
BLOCK_M = 8
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bits", BITRATES)
def test_uniform_trellis_experts_capture_a_cuda_graph(bits):
    prepare, kernel, host = _b12x()
    device = torch.device("cuda")

    w13 = torch.zeros(
        (2, NUM_EXPERTS, HIDDEN // 16, INTERMEDIATE // 16, 16 * bits),
        dtype=torch.int16,
        device=device,
    )
    w2 = torch.zeros(
        (NUM_EXPERTS, INTERMEDIATE // 16, HIDDEN // 16, 16 * bits),
        dtype=torch.int16,
        device=device,
    )
    suh13 = torch.ones((2, NUM_EXPERTS, HIDDEN), dtype=torch.float16, device=device)
    svh13 = torch.ones(
        (2, NUM_EXPERTS, INTERMEDIATE), dtype=torch.float16, device=device
    )
    down_suh = torch.ones(
        (NUM_EXPERTS, INTERMEDIATE), dtype=torch.float16, device=device
    )
    down_svh = torch.ones((NUM_EXPERTS, HIDDEN), dtype=torch.float16, device=device)
    intermediate_rotations = torch.cat(
        (svh13[0], svh13[1], down_suh), dim=1
    ).contiguous()

    props = torch.cuda.get_device_properties(device)
    sms = int(props.multi_processor_count)
    # The fused FC1+FC2 launch keeps per-SM state here and rejects anything
    # smaller, so this is a contract, not a scratch guess.
    workspace = torch.zeros(sms * 4 + 2, dtype=torch.int32, device=device)

    prepared = prepare.prepare_trellis256_moe_weights(
        w13=w13,
        w2=w2,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_experts=NUM_EXPERTS,
        activation="silu",
        fc1_tile_n=TILE_CONFIG[1],
        fc2_tile_n=TILE_CONFIG[3],
        params_dtype=torch.float16,
        w13_layout="trellis_t256_proj",
        trellis_bits=bits,
        codebook="mcg",
        gate_suh=suh13[0],
        up_suh=suh13[1],
        intermediate_rotations=intermediate_rotations,
        down_svh=down_svh,
        tile_config=TILE_CONFIG,
        workspace=workspace,
    )

    route_slots = host.max_packed_route_slots(TOKENS * TOPK, BLOCK_M, NUM_EXPERTS)
    launch = kernel.compile_w4a16_fused_moe(
        size_m=TOKENS,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_experts=NUM_EXPERTS,
        top_k=TOPK,
        activation="silu",
        apply_router_weight_on_input=False,
        zero_fc2_output=False,
        moe_block_size=BLOCK_M,
        max_m_blocks=(route_slots + BLOCK_M - 1) // BLOCK_M,
        sms=sms,
        max_shared_mem=int(props.shared_memory_per_block_optin),
        # full_rotation fixes the kernel boundary at fp16 regardless of the
        # model's activation dtype.
        element_dtype="fp16",
        weight_layout=prepared.weight_layout,
        scale_format=prepared.scale_format,
        w13_layout=prepared.w13_layout,
        trellis_bits=bits,
        trellis_codebook="mcg",
        force_tile_config=TILE_CONFIG,
        full_rotation=True,
        intermediate_rotation=True,
        rotation_input_dtype="fp16",
    )
    buffers = host.make_w4a16_packed_buffers(
        prepared,
        m=TOKENS,
        topk=TOPK,
        dtype=torch.float16,
        device=device,
        # Left unset so the planner sizes the route pack from the prepared
        # expert count, which is what the kernel validates against.
        route_num_experts=None,
        full_rotation=True,
        block_size_m=BLOCK_M,
    )

    x = torch.randn((TOKENS, HIDDEN), dtype=torch.float16, device=device)
    topk_weights = torch.full((TOKENS, TOPK), 0.5, dtype=torch.float32, device=device)
    topk_ids = torch.randint(
        0, NUM_EXPERTS, (TOKENS, TOPK), dtype=torch.int32, device=device
    )
    trellis = prepared.trellis

    def run():
        return kernel.run_w4a16_moe(
            x,
            prepared,
            topk_weights,
            topk_ids,
            activation="silu",
            intermediate_cache13=buffers.intermediate_cache13,
            intermediate_cache2=buffers.intermediate_cache2,
            output=buffers.output[:TOKENS],
            fc1_c_tmp=buffers.fc1_c_tmp,
            fc2_c_tmp=buffers.fc2_c_tmp,
            packed_route_indices=buffers.packed_route_indices,
            block_expert_ids=buffers.block_expert_ids,
            packed_route_count=buffers.packed_route_count,
            expert_offsets=buffers.expert_offsets,
            expert_counts=buffers.expert_counts,
            fused_launch=launch,
            route_block_size_m=BLOCK_M,
            intermediate_rotation_scales=trellis.intermediate_rotations,
            suh_gate_table=trellis.gate_suh,
            suh_up_table=trellis.up_suh,
            svh_table=trellis.down_svh,
            rotation_a_gate=buffers.rotation_a_gate,
            rotation_a_up=buffers.rotation_a_up,
            full_rotation=True,
        )

    eager = run()
    torch.cuda.synchronize()
    assert eager.shape == (TOKENS, HIDDEN)
    # Numerics are deliberately not asserted: the payloads are synthetic, so the
    # decoded weights are meaningless and may be non-finite. What this test pins
    # is the launch contract and capture-safety, which is what the eager
    # exemption actually rests on.

    # Warm once outside capture, then capture and replay. A failure here means
    # the launch is allocating or autotuning at run time, which is exactly what
    # the eager exemption promises it does not do.
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize()
