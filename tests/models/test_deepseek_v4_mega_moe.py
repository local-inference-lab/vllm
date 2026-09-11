# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    bind_routed_experts_capturer,
)
from vllm.model_executor.models.utils import WeightsMapper
from vllm.models.deepseek_v4.nvidia.dspark import DSparkDeepseekV4ForCausalLM
from vllm.models.deepseek_v4.nvidia.model import (
    DeepseekV4ForCausalLM,
    DeepseekV4MegaMoEExperts,
    DeepseekV4MoE,
    make_deepseek_v4_expert_params_mapping,
)
from vllm.models.deepseek_v4.nvidia.mtp import DeepSeekV4MTP
from vllm.models.deepseek_v4.nvidia.ops.prepare_megamoe import prepare_megamoe_inputs
from vllm.models.deepseek_v4.nvidia.vl_model import (
    DeepseekV4ForConditionalGeneration,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="DeepSeek V4 MegaMoE requires CUDA",
)


def test_deepseek_v4_mega_moe_expert_mapping():
    mapping = make_deepseek_v4_expert_params_mapping(2)

    assert mapping == [
        ("experts.w13_", "experts.0.w1.", 0, "w1"),
        ("experts.w2_", "experts.0.w2.", 0, "w2"),
        ("experts.w13_", "experts.0.w3.", 0, "w3"),
        ("experts.w13_", "experts.1.w1.", 1, "w1"),
        ("experts.w2_", "experts.1.w2.", 1, "w2"),
        ("experts.w13_", "experts.1.w3.", 1, "w3"),
    ]


def test_deepseek_v4_mega_moe_ue8m0_uint8_to_float():
    raw = torch.tensor([0, 126, 127, 128], dtype=torch.uint8)

    decoded = DeepseekV4MegaMoEExperts._ue8m0_uint8_to_float(raw)

    assert torch.equal(decoded.view(torch.int32), raw.to(torch.int32) << 23)
    assert decoded[0].item() == 0.0
    assert decoded[1].item() == 0.5
    assert decoded[2].item() == 1.0
    assert decoded[3].item() == 2.0


@pytest.mark.parametrize("use_kimi", [False, True])
def test_deep_gemm_mega_moe_capture_precedes_eplb(monkeypatch, use_kimi):
    experts_cls = DeepseekV4MegaMoEExperts
    if use_kimi:
        from vllm.models.kimi_k3.nvidia.model import KimiK3MegaMoEExperts

        experts_cls = KimiK3MegaMoEExperts

    experts = experts_cls.__new__(experts_cls)
    torch.nn.Module.__init__(experts)
    if use_kimi:
        experts.synchronize_first_launch = lambda: None
    experts.prefix = "model.layers.3.ffn.experts"
    experts.max_num_tokens = 4
    experts.capture_fn = None
    experts.get_symm_buffer = lambda: object()
    experts.eplb_state = SimpleNamespace(
        logical_to_physical_map=torch.empty(1),
        expert_load_view=torch.empty(1),
        logical_replica_count=torch.empty(1),
        should_record_tensor=torch.empty(1),
        num_unpadded_tokens_tensors=None,
    )

    topk_ids = torch.tensor([[1, 2], [3, 4]])
    captured: list[tuple[int, torch.Tensor]] = []
    bind_routed_experts_capturer(
        SimpleNamespace(modules=lambda: [experts]),
        SimpleNamespace(capture=lambda layer_id, ids: captured.append((layer_id, ids))),
    )

    class MappingReached(Exception):
        pass

    def map_ids(**kwargs):
        assert captured == [(3, topk_ids)]
        raise MappingReached

    monkeypatch.setattr(
        f"{experts_cls.__module__}.eplb_map_to_physical_and_record",
        map_ids,
    )
    monkeypatch.setattr(
        "vllm.utils.deep_gemm._import_deep_gemm", lambda: SimpleNamespace()
    )

    with pytest.raises(MappingReached):
        experts(
            torch.empty(2, 8),
            torch.empty(2, 2),
            topk_ids,
            activation_clamp=None,
        )


def test_deepseek_v4_mega_moe_weight_loader_uses_ep_expert_ownership():
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    experts = DeepseekV4MegaMoEExperts(
        vllm_config,
        num_experts=4,
        num_local_experts=2,
        experts_start_idx=2,
        top_k=2,
        hidden_size=128,
        intermediate_size=128,
    )

    nonlocal_weight = torch.ones(128, 64, dtype=torch.uint8)
    assert (
        experts.weight_loader(
            experts.w13_weight,
            nonlocal_weight,
            "experts.w13_weight",
            shard_id="w1",
            expert_id=1,
            return_success=True,
        )
        is False
    )

    w1 = torch.full((128, 64), 3, dtype=torch.uint8)
    w3 = torch.full((128, 64), 7, dtype=torch.uint8)
    w2 = torch.full((128, 64), 11, dtype=torch.uint8)

    assert experts.weight_loader(
        experts.w13_weight,
        w1,
        "experts.w13_weight",
        shard_id="w1",
        expert_id=2,
        return_success=True,
    )
    assert experts.weight_loader(
        experts.w13_weight,
        w3,
        "experts.w13_weight",
        shard_id="w3",
        expert_id=2,
        return_success=True,
    )
    assert experts.weight_loader(
        experts.w2_weight,
        w2,
        "experts.w2_weight",
        shard_id="w2",
        expert_id=2,
        return_success=True,
    )

    assert torch.equal(experts.w13_weight[0, :128], w1)
    assert torch.equal(experts.w13_weight[0, 128:], w3)
    assert torch.equal(experts.w2_weight[0], w2)
    assert torch.count_nonzero(experts.w13_weight[1]) == 0


def test_v41_loaded_experts_preserve_gate_up_math_and_replay(monkeypatch):
    """Checkpoint w1 is the gate; b12x's W13 layout names use a different order."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("native b12x experts require SM12x")
    from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution

    from vllm.models.deepseek_v4_1.nvidia import b12x_moe
    from vllm.v1.worker import workspace

    device = torch.device("cuda")
    monkeypatch.setattr(b12x_moe, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(b12x_moe, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    monkeypatch.setattr(workspace, "_manager", workspace.WorkspaceManager(device))
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                hidden_size=256,
                moe_intermediate_size=128,
                swiglu_limit=10.0,
            )
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=3),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    with torch.device(device):
        experts = b12x_moe.B12xV41Experts(
            config,
            num_experts=1,
            top_k=1,
            prefix="model.layers.0.ffn.experts",
        )
    # FP4 nibbles 2 and C decode to +1 and -2 respectively. These
    # constant projections make the published K32 contraction exact.
    for shard, code, exponent, shape in (
        ("w1", 0x22, 123, (128, 128)),
        ("w3", 0xCC, 123, (128, 128)),
        ("w2", 0x22, 120, (256, 64)),
    ):
        name = "w2_weight" if shard == "w2" else "w13_weight"
        for suffix, value, payload_shape in (
            ("", code, shape),
            ("_scale", exponent, (shape[0], shape[1] // 16)),
        ):
            experts.weight_loader(
                getattr(experts, name + suffix),
                torch.full(payload_shape, value, dtype=torch.uint8, device=device),
                f"experts.{name}{suffix}",
                shard_id=shard,
                expert_id=0,
            )
    experts.finalize_weights()
    x = (
        torch.tensor([1 / 16, -1 / 16, 1 / 32], device=device, dtype=torch.bfloat16)[
            :, None
        ]
        .expand(-1, 256)
        .contiguous()
    )
    weights = torch.tensor([[0.5], [0.25], [1.0]], device=device)
    ids = torch.zeros((3, 1), dtype=torch.int32, device=device)
    output = torch.empty(x.shape, dtype=torch.float32, device=device)

    def oracle():
        gate = (x.float().sum(-1, keepdim=True) / 16).bfloat16().float()
        up = -2 * gate
        mid = (torch.nn.functional.silu(gate) * up * weights).bfloat16().float()
        scale = torch.exp2(torch.ceil(torch.log2(mid.abs().clamp_min(1e-4) / 448)))
        quantized = (mid / scale).to(torch.float8_e4m3fn).float() * scale
        # W2 has 128 copies of 1/128, so its contraction is the identity.
        return quantized.bfloat16().float().expand_as(output)

    experts.run_native(x, weights, ids, output)
    torch.testing.assert_close(output, oracle(), rtol=0, atol=0)
    freeze_kernel_resolution("V4.1 loaded expert gate/up replay")
    try:
        graph = torch.cuda.CUDAGraph()
        with workspace.collect_cuda_graph_capture_resources() as retained:
            with torch.cuda.graph(graph):
                experts.run_native(x, weights, ids, output)
        x.neg_()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, oracle(), rtol=0, atol=0)
        del retained
    finally:
        unfreeze_kernel_resolution()


def test_deepseek_v4_mega_moe_finalizes_native_shared_expert_weights(monkeypatch):
    class FakeDeepGemm:
        transformed_dims: list[tuple[int, int]] = []
        scale_inputs: list[tuple[int, ...]] = []

        @staticmethod
        def get_symm_buffer_for_mega_moe(*args, num_shared_experts=0, **kwargs):
            return None

        @staticmethod
        def get_block_m_for_mega_moe(*args, **kwargs):
            return 128

        @staticmethod
        def fp8_fp4_mega_moe(
            y,
            l1_weights,
            l2_weights,
            sym_buffer,
            shared_l1_weights=None,
            shared_l2_weights=None,
            **kwargs,
        ):
            return None

        @classmethod
        def transform_sf_into_required_layout(cls, sf, mn, k, *args, **kwargs):
            cls.scale_inputs.append(tuple(sf.shape))
            return torch.empty((sf.shape[0], mn, k // 128), dtype=torch.int32)

        @classmethod
        def transform_weights_for_mega_moe(cls, l1_weights, l2_weights):
            cls.transformed_dims.append((l1_weights[0].dim(), l2_weights[0].dim()))
            if l1_weights[0].dim() == 2:
                return (l1_weights[0].clone(), l1_weights[1]), l2_weights
            return l1_weights, l2_weights

    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    experts = DeepseekV4MegaMoEExperts(
        vllm_config,
        num_experts=2,
        num_local_experts=1,
        experts_start_idx=0,
        top_k=1,
        hidden_size=128,
        intermediate_size=128,
        num_shared_experts=1,
    )
    experts._check_runtime_supported = lambda: None

    def fp8_parameter(*shape):
        return torch.nn.Parameter(
            torch.empty(*shape, dtype=torch.float8_e4m3fn), requires_grad=False
        )

    def scale_parameter(*shape, dtype=torch.int32):
        return torch.nn.Parameter(torch.ones(*shape, dtype=dtype), requires_grad=False)

    shared_experts = SimpleNamespace(
        gate_up_proj=SimpleNamespace(
            weight=fp8_parameter(256, 128),
            weight_block_size=(128, 128),
            weight_scale_inv=scale_parameter(2, 1, dtype=torch.float8_e8m0fnu),
        ),
        down_proj=SimpleNamespace(
            weight=fp8_parameter(128, 128),
            weight_block_size=(128, 128),
            weight_scale_inv=scale_parameter(1, 1, dtype=torch.float8_e8m0fnu),
        ),
    )
    monkeypatch.setattr("vllm.utils.deep_gemm._import_deep_gemm", lambda: FakeDeepGemm)

    original_gate_up_ptr = shared_experts.gate_up_proj.weight.data_ptr()
    experts.finalize_weights(shared_experts)

    assert FakeDeepGemm.transformed_dims == [(3, 3), (2, 2)]
    assert FakeDeepGemm.scale_inputs[-2:] == [(1, 256, 4), (1, 128, 4)]
    assert experts.has_fused_shared_experts
    assert shared_experts.gate_up_proj.weight.data_ptr() != original_gate_up_ptr
    assert (
        experts._transformed_shared_l1_weights[0].data_ptr()
        == shared_experts.gate_up_proj.weight.data_ptr()
    )
    assert (
        experts._transformed_shared_l2_weights[0].data_ptr()
        == shared_experts.down_proj.weight.data_ptr()
    )


@pytest.mark.parametrize("fused", [False, True])
def test_deepseek_v4_mega_moe_does_not_double_add_fused_shared_expert(
    monkeypatch, fused
):
    class FakeGate(torch.nn.Module):
        tid2eid = None
        e_score_correction_bias = None

        def forward(self, hidden_states):
            return torch.empty(hidden_states.shape[0], 2), None

    class FakeExperts(torch.nn.Module):
        has_fused_shared_experts = fused

        def forward(self, hidden_states, *args, **kwargs):
            return torch.ones_like(hidden_states)

    class FakeSharedExperts(torch.nn.Module):
        calls = 0

        def forward(self, hidden_states):
            self.calls += 1
            return torch.full_like(hidden_states, 2)

    moe = DeepseekV4MoE.__new__(DeepseekV4MoE)
    torch.nn.Module.__init__(moe)
    moe.use_mega_moe = True
    moe.gate = FakeGate()
    moe.experts = FakeExperts()
    moe.shared_experts = FakeSharedExperts()
    moe.scoring_func = "sqrtsoftplus"
    moe.n_activated_experts = 1
    moe.renormalize = True
    moe.hash_indices_dtype = torch.int64
    moe.routed_scaling_factor = 1.0
    moe.swiglu_limit = 10.0
    monkeypatch.setattr(
        "vllm.models.deepseek_v4.nvidia.model.fused_topk_bias",
        lambda **kwargs: (
            torch.ones(kwargs["hidden_states"].shape[0], 1),
            torch.zeros(kwargs["hidden_states"].shape[0], 1, dtype=torch.int64),
        ),
    )

    output = moe(torch.zeros(2, 128))

    expected = 1 if fused else 3
    assert torch.all(output == expected)
    assert moe.shared_experts.calls == (0 if fused else 1)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="DeepSeek V4 MegaMoE fused input staging requires CUDA.",
)
def test_deepseek_v4_mega_moe_fused_input_staging_is_bitwise_exact():
    from vllm.third_party.deep_gemm.utils import per_token_cast_to_fp8

    device = torch.device("cuda")
    num_tokens = 7
    hidden_size = 256
    top_k = 8

    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    hidden_states = (
        torch.randn(
            num_tokens,
            hidden_size,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        * 17.0
    ).to(torch.bfloat16)
    hidden_states[0, :32] = 0
    hidden_states[1, 32:64] = 1.0e-6
    hidden_states[2, 64:96] = -1.0e-6

    topk_ids = torch.randint(
        0,
        256,
        (num_tokens, top_k),
        device=device,
        dtype=torch.int32,
        generator=generator,
    )
    topk_weights = torch.randn(
        num_tokens,
        top_k,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )

    ref_x, ref_x_sf = per_token_cast_to_fp8(
        hidden_states,
        use_ue8m0=True,
        gran_k=32,
        use_packed_ue8m0=True,
    )
    ref_topk_idx = topk_ids.to(torch.int64)
    ref_topk_weights = topk_weights.clone()

    fused_x = torch.empty_like(ref_x)
    fused_x_sf = torch.empty_like(ref_x_sf)
    fused_topk_idx = torch.empty_like(ref_topk_idx)
    fused_topk_weights = torch.empty_like(ref_topk_weights)

    prepare_megamoe_inputs(
        hidden_states,
        topk_weights,
        topk_ids,
        fused_x,
        fused_x_sf,
        fused_topk_idx,
        fused_topk_weights,
    )
    torch.accelerator.synchronize()

    assert torch.equal(fused_x.view(torch.uint8), ref_x.view(torch.uint8))
    assert torch.equal(fused_x_sf, ref_x_sf)
    assert torch.equal(fused_topk_idx, ref_topk_idx)
    assert torch.equal(
        fused_topk_weights.view(torch.uint8),
        ref_topk_weights.view(torch.uint8),
    )


@pytest.mark.parametrize("shared_block_m", [8, 32, 96, 128, 192])
def test_deepseek_v4_mega_moe_stages_shared_scale_tma_layout(shared_block_m):
    from vllm.third_party.deep_gemm.utils import per_token_cast_to_fp8

    device = torch.device("cuda")
    num_tokens = shared_block_m + 7
    hidden_size = 256
    top_k = 8
    generator = torch.Generator(device=device)
    generator.manual_seed(shared_block_m)
    hidden_states = torch.randn(
        num_tokens,
        hidden_size,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    topk_ids = torch.randint(
        0,
        256,
        (num_tokens, top_k),
        device=device,
        dtype=torch.int32,
        generator=generator,
    )
    topk_weights = torch.randn(
        num_tokens,
        top_k,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )

    ref_x, ref_x_sf = per_token_cast_to_fp8(
        hidden_states,
        use_ue8m0=True,
        gran_k=32,
        use_packed_ue8m0=True,
    )
    aligned_block_m = ((shared_block_m + 127) // 128) * 128
    num_shared_rows = ((num_tokens + shared_block_m - 1) // shared_block_m) * (
        aligned_block_m
    )
    ref_shared_x_sf = torch.zeros(
        num_shared_rows,
        hidden_size // 128,
        dtype=torch.int32,
        device=device,
    )
    for token_id in range(num_tokens):
        m_in_block = token_id % shared_block_m
        transposed_m = (
            (m_in_block // 128) * 128 + (m_in_block % 32) * 4 + (m_in_block % 128) // 32
        )
        shared_row = token_id // shared_block_m * aligned_block_m + transposed_m
        ref_shared_x_sf[shared_row].copy_(ref_x_sf[token_id])

    fused_x = torch.empty_like(ref_x)
    fused_x_sf = torch.empty_like(ref_x_sf)
    fused_shared_storage = torch.full(
        (hidden_size // 128, num_shared_rows),
        -1,
        dtype=torch.int32,
        device=device,
    )
    fused_shared_x_sf = fused_shared_storage.t()
    fused_topk_idx = torch.empty_like(topk_ids, dtype=torch.int64)
    fused_topk_weights = torch.empty_like(topk_weights)

    prepare_megamoe_inputs(
        hidden_states,
        topk_weights,
        topk_ids,
        fused_x,
        fused_x_sf,
        fused_topk_idx,
        fused_topk_weights,
        shared_x_sf=fused_shared_x_sf,
        shared_block_m=shared_block_m,
    )
    torch.accelerator.synchronize()

    populated = ref_shared_x_sf != 0
    assert torch.equal(fused_x.view(torch.uint8), ref_x.view(torch.uint8))
    assert torch.equal(fused_x_sf, ref_x_sf)
    assert torch.equal(fused_shared_x_sf[populated], ref_shared_x_sf[populated])


def test_deepseek_v4_pwal_hook_finalizes_mega_moe_and_mhc_broadcast():
    """The loader invokes the model-level PWAL hook for every load format,
    so it must finalize derived MegaMoE, mHC, and B12x weights."""
    calls = []
    stub = SimpleNamespace(
        model=SimpleNamespace(
            finalize_mega_moe_weights=lambda: calls.append("mega_moe"),
            finalize_mhc_broadcast_weights=lambda: calls.append("mhc"),
            process_b12x_weights_after_loading=lambda: calls.append("b12x"),
        )
    )

    DeepseekV4ForCausalLM.process_weights_after_loading(stub)

    assert calls == ["mega_moe", "mhc", "b12x"]


def test_deepseek_v4_vision_loads_interleaved_weights_before_finalizing():
    finalized = []

    class LanguageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Parameter(torch.zeros(1))
            self.last = torch.nn.Parameter(torch.zeros(1))

        def load_weights(self, weights):
            raise AssertionError(
                "Language-model finalization must wait for all weights"
            )

        def process_weights_after_loading(self):
            assert self.first.item() == 1 and self.last.item() == 3
            finalized.append(True)

    model = DeepseekV4ForConditionalGeneration.__new__(
        DeepseekV4ForConditionalGeneration
    )
    torch.nn.Module.__init__(model)
    model.language_model = LanguageModel()
    model.image_start = torch.nn.Parameter(torch.zeros(1))
    model.hf_to_vllm_mapper = WeightsMapper()

    def weights():
        yield "language_model.first", torch.tensor([1.0])
        assert model.language_model.first.item() == 1
        yield "image_start", torch.tensor([2.0])
        assert model.image_start.item() == 2
        assert not finalized
        yield "language_model.last", torch.tensor([3.0])

    with torch.no_grad():
        loaded = model.load_weights(weights())
        model.process_weights_after_loading()

    assert loaded == {"language_model.first", "image_start", "language_model.last"}
    assert finalized == [True]


def test_deepseek_v4_drafter_pwal_hooks_finalize_mega_moe():
    """MTP and DSpark top-level loaders finalize derived backend weights."""
    calls = []
    mtp_block = SimpleNamespace(
        process_b12x_weights_after_loading=lambda: calls.append("mtp_b12x")
    )
    mtp = SimpleNamespace(
        finalize_mega_moe_weights=lambda: calls.append("mtp"),
        model=SimpleNamespace(layers={0: SimpleNamespace(mtp_block=mtp_block)}),
    )
    DeepSeekV4MTP.process_weights_after_loading(mtp)

    dspark = SimpleNamespace(
        _finalize_moe=lambda: calls.append("dspark"),
        model=SimpleNamespace(
            finalize_mhc_broadcast_weights=lambda: calls.append("dspark_mhc"),
            layers=[
                SimpleNamespace(
                    process_b12x_weights_after_loading=lambda: calls.append(
                        "dspark_b12x"
                    )
                )
            ],
        ),
    )
    DSparkDeepseekV4ForCausalLM.process_weights_after_loading(dspark)

    assert calls == ["mtp", "mtp_b12x", "dspark", "dspark_mhc", "dspark_b12x"]


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="DeepSeek V4 MegaMoE fused input staging requires CUDA.",
)
def test_deepseek_v4_mega_moe_fused_input_staging_masks_padding():
    from vllm.third_party.deep_gemm.utils import per_token_cast_to_fp8

    device = torch.device("cuda")
    num_tokens = 7
    hidden_size = 256
    top_k = 8

    generator = torch.Generator(device=device)
    generator.manual_seed(1)
    hidden_states = torch.randn(
        num_tokens,
        hidden_size,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    topk_ids = torch.randint(
        0,
        256,
        (num_tokens, top_k),
        device=device,
        dtype=torch.int32,
        generator=generator,
    )
    topk_weights = torch.randn(
        num_tokens,
        top_k,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    is_padding = torch.tensor(
        [False, True, False, False, True, False, True],
        device=device,
    )

    ref_x, ref_x_sf = per_token_cast_to_fp8(
        hidden_states,
        use_ue8m0=True,
        gran_k=32,
        use_packed_ue8m0=True,
    )
    ref_topk_idx = topk_ids.to(torch.int64)
    ref_topk_idx[is_padding] = -1
    ref_topk_weights = topk_weights.clone()
    ref_topk_weights[is_padding] = 0.0

    fused_x = torch.empty_like(ref_x)
    fused_x_sf = torch.empty_like(ref_x_sf)
    fused_topk_idx = torch.empty_like(ref_topk_idx)
    fused_topk_weights = torch.empty_like(ref_topk_weights)

    prepare_megamoe_inputs(
        hidden_states,
        topk_weights,
        topk_ids,
        fused_x,
        fused_x_sf,
        fused_topk_idx,
        fused_topk_weights,
        is_padding=is_padding,
    )
    torch.accelerator.synchronize()

    assert torch.equal(fused_x.view(torch.uint8), ref_x.view(torch.uint8))
    assert torch.equal(fused_x_sf, ref_x_sf)
    assert torch.equal(fused_topk_idx, ref_topk_idx)
    assert torch.equal(
        fused_topk_weights.view(torch.uint8),
        ref_topk_weights.view(torch.uint8),
    )


def _v41_composition_rank(rank, world_size, rendezvous):
    """Two native routed contributions straddle a BF16 rounding boundary."""
    from types import SimpleNamespace

    import torch.distributed as dist

    from vllm.models.deepseek_v4_1.nvidia import b12x_moe
    from vllm.v1.worker import workspace

    device = torch.device("cuda", rank)
    with torch.cuda.device(device), pytest.MonkeyPatch.context() as patch:
        if world_size > 1:
            dist.init_process_group(
                "nccl", init_method=rendezvous, rank=rank, world_size=world_size
            )
        patch.setattr(
            b12x_moe, "get_tensor_model_parallel_world_size", lambda: world_size
        )
        patch.setattr(b12x_moe, "get_tensor_model_parallel_rank", lambda: rank)
        patch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
        patch.setattr(workspace, "_manager", workspace.WorkspaceManager(device))
        count = max(2, world_size)
        config = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    hidden_size=256, moe_intermediate_size=128, swiglu_limit=10.0
                )
            ),
            scheduler_config=SimpleNamespace(max_num_batched_tokens=1),
            compilation_config=SimpleNamespace(static_forward_context={}),
        )
        with torch.device(device):
            experts = b12x_moe.B12xV41Experts(
                config, num_experts=count, top_k=2, prefix="composition.experts"
            )
        for expert_id in range(experts.experts_start_idx, experts.experts_end_idx):
            for shard, code, exponent, shape in (
                ("w1", 0x22, 123, (128, 128)),
                ("w3", 0xCC, 123, (128, 128)),
                ("w2", 0x22, 130 if expert_id == 0 else 120, (256, 64)),
            ):
                name = "w2_weight" if shard == "w2" else "w13_weight"
                for suffix, value, payload_shape in (
                    ("", code, shape),
                    ("_scale", exponent, (shape[0], shape[1] // 16)),
                ):
                    experts.weight_loader(
                        getattr(experts, name + suffix),
                        torch.full(
                            payload_shape, value, dtype=torch.uint8, device=device
                        ),
                        f"experts.{name}{suffix}",
                        shard_id=shard,
                        expert_id=expert_id,
                    )
        experts.finalize_weights()
        patch.setattr(
            b12x_moe,
            "get_forward_context",
            lambda: SimpleNamespace(no_compile_layers={experts.prefix: experts}),
        )

        def reduce_routed(tensor):
            if world_size > 1:
                dist.all_reduce(tensor)
            return tensor

        patch.setattr(b12x_moe, "tensor_model_parallel_all_reduce", reduce_routed)

        class Gate(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.e_score_correction_bias = torch.zeros(count, device=device)
                self.logits = torch.full((1, count), -100.0, device=device)
                self.logits[:, 0] = self.logits[:, -1] = 10.0

            def forward(self, x):
                return self.logits, None

        class Shared(torch.nn.Module):
            def forward(self, x):
                return torch.full_like(x, 768)

        model = b12x_moe.DeepseekV4MoE.__new__(b12x_moe.DeepseekV4MoE)
        torch.nn.Module.__init__(model)
        model.gate, model.experts, model.shared_experts = Gate(), experts, Shared()
        model.n_activated_experts = 2
        model.routed_scaling_factor = 1.0
        model.is_draft = True
        x = torch.full((1, 256), 1 / 16, dtype=torch.bfloat16, device=device)
        with torch.no_grad():
            result = model(
                x, input_ids=torch.zeros(1, dtype=torch.int64, device=device)
            )
        # Routed values are -768 and -0.75. Rounding their sum to BF16 before
        # adding the shared +768 yields zero rather than the retained -0.75.
        torch.testing.assert_close(result, torch.full_like(x, -0.75), rtol=0, atol=0)
        if world_size > 1:
            dist.destroy_process_group()


def test_v41_moe_composition_preserves_small_routed_contribution():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("native b12x experts require SM12x")
    _v41_composition_rank(0, 1, None)


@pytest.mark.distributed(num_gpus=2)
def test_v41_moe_composition_preserves_small_tp_contribution(tmp_path):
    if torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices required")
    torch.multiprocessing.spawn(
        _v41_composition_rank,
        args=(2, f"file://{tmp_path / 'composition-rendezvous'}"),
        nprocs=2,
        join=True,
    )
