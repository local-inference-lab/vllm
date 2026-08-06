# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for draft capacity tests", allow_module_level=True)

from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.models import qwen3_dspark
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla import indexer as indexer_module
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadataBuilder,
    _needs_varlen_decode,
    _uses_varlen_dspark_capacity,
)
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.spec_decode.capacity import (
    CapacityBasedVerificationManager,
    DSparkDynamicDraftDepthController,
    MaskedCapacityBasedVerificationManager,
    VarlenCapacityBasedVerificationManager,
)
from vllm.v1.worker.gpu.spec_decode.dspark.capacity import (
    compute_draft_token_capacity_from_confidence,
)
from vllm.v1.worker.gpu.spec_decode.dspark.online_sts import DSparkOnlineSTS
from vllm.v1.worker.gpu.states import RequestState


def test_varlen_indexer_is_limited_to_active_dspark_capacity():
    def config(**overrides):
        values = {
            "use_dspark": lambda: True,
            "dspark_capacity_verification_mode": "varlen",
            "dspark_confidence_threshold": 0.0,
            "dspark_budget_frac": 1.0,
            "dspark_sps_curve": None,
        }
        values.update(overrides)
        return SimpleNamespace(
            speculative_config=SimpleNamespace(**values),
        )

    assert not _uses_varlen_dspark_capacity(config())
    assert _uses_varlen_dspark_capacity(config(dspark_confidence_threshold=0.5))
    assert _uses_varlen_dspark_capacity(config(dspark_budget_frac=0.5))
    assert _uses_varlen_dspark_capacity(config(dspark_sps_curve="auto"))
    assert not _uses_varlen_dspark_capacity(
        config(dspark_capacity_verification_mode="mask")
    )
    assert not _uses_varlen_dspark_capacity(config(use_dspark=lambda: False))


def test_varlen_indexer_cudagraph_support_requires_capacity_opt_in(monkeypatch):
    monkeypatch.setattr(
        indexer_module, "_supports_varlen_paged_mqa_logits", lambda: True
    )

    def config(*, threshold: float):
        return SimpleNamespace(
            speculative_config=SimpleNamespace(
                use_dspark=lambda: True,
                dspark_capacity_verification_mode="varlen",
                dspark_confidence_threshold=threshold,
                dspark_budget_frac=1.0,
                dspark_sps_curve=None,
            )
        )

    assert (
        DeepseekV32IndexerMetadataBuilder.get_cudagraph_support(
            config(threshold=0.0), None
        )
        is AttentionCGSupport.UNIFORM_BATCH
    )
    assert (
        DeepseekV32IndexerMetadataBuilder.get_cudagraph_support(
            config(threshold=0.5), None
        )
        is AttentionCGSupport.ALWAYS
    )


def test_varlen_indexer_skips_uniform_full_width_batches():
    assert not _needs_varlen_decode(True, True, 6, 6)
    assert _needs_varlen_decode(True, False, 4, 4)
    assert _needs_varlen_decode(True, False, 6, 6)
    assert not _needs_varlen_decode(False, False, 6, 6)


def test_dynamic_draft_depth_tracks_capacity_and_probes_upward():
    controller = DSparkDynamicDraftDepthController(max_depth=5, observation_window=2)
    attempted_5 = np.array([5, 5], dtype=np.int32)
    capacities_2_3 = np.array([2, 3], dtype=np.int32)

    assert controller.observe(capacities_2_3, attempted_5) == 5
    assert controller.observe(capacities_2_3, attempted_5) == 3

    attempted_3 = np.array([3, 3], dtype=np.int32)
    capacities_3 = np.array([3, 3], dtype=np.int32)
    for _ in range(controller._PROBE_AFTER_WINDOWS * 2 - 1):
        assert controller.observe(capacities_3, attempted_3) == 3
    assert controller.observe(capacities_3, attempted_3) == 4

    # A substantial load decrease resets the next proposal to the maximum.
    assert (
        controller.observe(
            np.array([3, 0], dtype=np.int32),
            np.array([3, 0], dtype=np.int32),
        )
        == 5
    )


def test_dynamic_draft_depth_preserves_longest_useful_request():
    controller = DSparkDynamicDraftDepthController(max_depth=5, observation_window=2)
    attempted = np.array([5, 5, 5], dtype=np.int32)
    capacities = np.array([2, 3, 5], dtype=np.int32)

    assert controller.observe(capacities, attempted) == 5
    assert controller.observe(capacities, attempted) == 5


def test_dynamic_draft_depth_applies_profiled_load_budget():
    controller = DSparkDynamicDraftDepthController(max_depth=5, observation_window=2)
    controller.set_draft_token_budget(160)
    attempted = np.full(64, 5, dtype=np.int32)
    capacities = np.full(64, 5, dtype=np.int32)

    assert controller.observe(capacities, attempted) == 5
    assert controller.observe(capacities, attempted) == 3

    # At half the load, the same profiled budget exposes the full K5 again.
    attempted[32:] = 0
    capacities[32:] = 0
    assert controller.observe(capacities, attempted) == 5


def test_capacity_kernel_supports_shorter_logical_width_than_storage_stride():
    device = torch.device("cuda")
    confidence_probs = torch.full((2, 5), 0.01, device=device)
    confidence_probs[:, :2] = torch.tensor([[0.9, 0.9], [0.8, 0.8]], device=device)
    confidence_logits = torch.logit(confidence_probs)
    capacities = torch.full((2,), -1, dtype=torch.int32, device=device)
    survival = torch.empty_like(confidence_logits)

    compute_draft_token_capacity_from_confidence(
        confidence_logits,
        capacities,
        min_survival_probability=0.7,
        num_reqs=2,
        num_speculative_steps=2,
        runtime_num_reqs=torch.tensor([2], dtype=torch.int32, device=device),
        survival_probs=survival,
    )

    torch.accelerator.synchronize()
    assert capacities.cpu().tolist() == [2, 1]


def test_compute_draft_token_capacity_from_confidence_uses_global_prefix_order():
    device = torch.device("cuda")
    confidence_probs = torch.tensor(
        [
            [0.90, 0.90, 0.90],
            [0.95, 0.10, 0.99],
            [0.70, 0.70, 0.70],
        ],
        dtype=torch.float32,
        device=device,
    )
    confidence_logits = torch.logit(confidence_probs)
    draft_token_capacity = torch.full((3,), -1, dtype=torch.int32, device=device)
    survival_probs = torch.empty_like(confidence_logits)

    compute_draft_token_capacity_from_confidence(
        confidence_logits,
        draft_token_capacity,
        min_survival_probability=0.75,
        num_reqs=3,
        num_speculative_steps=3,
        runtime_num_reqs=torch.tensor([3], dtype=torch.int32, device=device),
        survival_probs=survival_probs,
    )

    torch.accelerator.synchronize()
    assert draft_token_capacity.cpu().tolist() == [2, 1, 0]


def test_compute_draft_token_capacity_uses_budgeted_global_prefix_order():
    device = torch.device("cuda")
    confidence_probs = torch.tensor(
        [
            [0.90, 0.80],
            [0.80, 0.80],
        ],
        dtype=torch.float32,
        device=device,
    )
    confidence_logits = torch.logit(confidence_probs)
    draft_token_capacity = torch.full((2,), -1, dtype=torch.int32, device=device)
    survival_probs = torch.empty_like(confidence_logits)

    compute_draft_token_capacity_from_confidence(
        confidence_logits,
        draft_token_capacity,
        min_survival_probability=0.0,
        num_reqs=2,
        num_speculative_steps=2,
        runtime_num_reqs=torch.tensor([2], dtype=torch.int32, device=device),
        survival_probs=survival_probs,
        budget_frac=0.5,
    )

    torch.accelerator.synchronize()
    # ceil(4 * 0.5) admits exactly two globally ranked prefixes.
    assert draft_token_capacity.cpu().tolist() == [1, 1]


def test_compute_draft_token_capacity_keeps_threshold_ties():
    device = torch.device("cuda")
    confidence_probs = torch.tensor(
        [
            [0.90, 0.80],
            [0.90, 0.80],
        ],
        dtype=torch.float32,
        device=device,
    )
    confidence_logits = torch.logit(confidence_probs)
    draft_token_capacity = torch.full((2,), -1, dtype=torch.int32, device=device)
    survival_probs = torch.empty_like(confidence_logits)

    compute_draft_token_capacity_from_confidence(
        confidence_logits,
        draft_token_capacity,
        min_survival_probability=0.0,
        num_reqs=2,
        num_speculative_steps=2,
        runtime_num_reqs=torch.tensor([2], dtype=torch.int32, device=device),
        survival_probs=survival_probs,
        budget_frac=0.25,
    )

    torch.accelerator.synchronize()
    assert draft_token_capacity.cpu().tolist() == [1, 0]


def test_compute_draft_token_capacity_budget_is_hard_cap_under_ties():
    """Saturated (tied) survival scores must not escape the budget.

    With every confidence saturated to 1.0, a kth-score-threshold recount
    would admit all tokens; the budget must remain a hard cap on the total.
    """
    device = torch.device("cuda")
    confidence_logits = torch.full((4, 7), 40.0, dtype=torch.float32, device=device)
    draft_token_capacity = torch.full((4,), -1, dtype=torch.int32, device=device)
    survival_probs = torch.empty_like(confidence_logits)

    compute_draft_token_capacity_from_confidence(
        confidence_logits,
        draft_token_capacity,
        min_survival_probability=0.0,
        num_reqs=4,
        num_speculative_steps=7,
        runtime_num_reqs=torch.tensor([4], dtype=torch.int32, device=device),
        survival_probs=survival_probs,
        budget_frac=0.5,
    )

    torch.accelerator.synchronize()
    capacities = draft_token_capacity.cpu()
    assert int(capacities.sum()) == int(4 * 7 * 0.5)
    assert int(capacities.max()) <= 7


def test_compute_draft_token_capacity_never_admits_zero_survival():
    """Zero-survival tokens are not candidates (DSpark Alg. 1: a_{r,j} > 0),
    so leftover budget must not be spent past the first dead position."""
    device = torch.device("cuda")
    # Positions 0-1 confident, position 2 dead (sigmoid underflows to an
    # exact fp32 zero) -> survival is exactly 0 from there on.
    confidence_logits = torch.full((2, 5), 40.0, dtype=torch.float32, device=device)
    confidence_logits[:, 2] = -100.0
    draft_token_capacity = torch.full((2,), -1, dtype=torch.int32, device=device)
    survival_probs = torch.empty_like(confidence_logits)

    compute_draft_token_capacity_from_confidence(
        confidence_logits,
        draft_token_capacity,
        min_survival_probability=0.0,
        num_reqs=2,
        num_speculative_steps=5,
        runtime_num_reqs=torch.tensor([2], dtype=torch.int32, device=device),
        survival_probs=survival_probs,
        budget_frac=0.9,
    )

    torch.accelerator.synchronize()
    assert draft_token_capacity.cpu().tolist() == [2, 2]


def test_compute_draft_token_capacity_sps_curve_argmax():
    """With an SPS curve, verification lengths maximize tau * SPS(B)
    (DSpark Alg. 1) instead of spending the whole budget."""
    device = torch.device("cuda")
    confidence_probs = torch.tensor(
        [
            [0.90, 0.80],
            [0.60, 0.50],
        ],
        dtype=torch.float32,
        device=device,
    )
    confidence_logits = torch.logit(confidence_probs)
    draft_token_capacity = torch.full((2,), -1, dtype=torch.int32, device=device)
    survival_probs = torch.empty_like(confidence_logits)
    # Survival: r0 [0.9, 0.72], r1 [0.6, 0.3]; admission order
    # 0.9, 0.72, 0.6, 0.3 with B = 2 + k.
    # SPS drops sharply after B=4 so theta peaks at k=2:
    #   k=0: 2.00*1.00, k=1: 2.90*0.95=2.755, k=2: 3.62*0.90=3.258,
    #   k=3: 4.22*0.20=0.844, k=4: 4.52*0.10=0.452.
    sps_table = torch.tensor(
        [1.0, 1.0, 1.0, 0.95, 0.90, 0.20, 0.10], dtype=torch.float32, device=device
    )

    from vllm.v1.worker.gpu.spec_decode.dspark.capacity import (
        compute_draft_token_capacity_from_confidence as compute,
    )

    compute(
        confidence_logits,
        draft_token_capacity,
        min_survival_probability=0.0,
        num_reqs=2,
        num_speculative_steps=2,
        runtime_num_reqs=torch.tensor([2], dtype=torch.int32, device=device),
        survival_probs=survival_probs,
        budget_frac=1.0,
        sps_table=sps_table,
    )

    torch.accelerator.synchronize()
    assert draft_token_capacity.cpu().tolist() == [2, 0]


def test_compute_draft_token_capacity_temperature_desaturates_zeros():
    """A confidence temperature > 1 keeps saturated-negative positions in the
    candidate set (no exact-zero survival), so the budget is spent instead of
    being truncated by miscalibrated zeros."""
    device = torch.device("cuda")
    confidence_logits = torch.full((2, 5), 40.0, dtype=torch.float32, device=device)
    confidence_logits[:, 2] = -100.0
    survival_probs = torch.empty_like(confidence_logits)

    kwargs = dict(
        min_survival_probability=0.0,
        num_reqs=2,
        num_speculative_steps=5,
        runtime_num_reqs=torch.tensor([2], dtype=torch.int32, device=device),
        survival_probs=survival_probs,
        budget_frac=0.9,
    )
    capacity_t1 = torch.full((2,), -1, dtype=torch.int32, device=device)
    compute_draft_token_capacity_from_confidence(
        confidence_logits, capacity_t1, **kwargs
    )
    capacity_t10 = torch.full((2,), -1, dtype=torch.int32, device=device)
    compute_draft_token_capacity_from_confidence(
        confidence_logits, capacity_t10, confidence_temperature=10.0, **kwargs
    )

    torch.accelerator.synchronize()
    # T=1: exact-zero survival past position 2 truncates both requests.
    assert capacity_t1.cpu().tolist() == [2, 2]
    # T=10: sigmoid(-10) > 0, so the ceil(10*0.9) = 9 admission
    # budget is fully spent.
    assert capacity_t10.cpu().tolist() == [5, 4]


def test_online_sts_fits_order_preserving_temperatures():
    """Online STS fits per-position temperatures from rejection-sampler
    outcomes: identity before data, softens over-confident positions,
    sharpens under-confident ones, and never reorders candidates."""
    device = torch.device("cuda")
    sts = DSparkOnlineSTS(max_num_reqs=4, num_steps=3, device=device)

    # Cold start: identity calibration.
    probe = torch.tensor([[2.0, 1.0, -1.0]], dtype=torch.float32, device=device)
    assert torch.equal(sts.calibrate(probe), probe)

    slots = torch.tensor([0, 1], dtype=torch.int32, device=device)
    # Head claims p~0.88 everywhere (logit 2.0).
    logits = torch.full((2, 3), 2.0, dtype=torch.float32, device=device)
    # Alternate outcomes so pos0 accepts 50% (head over-confident there)
    # while pos1/pos2 always accept once reached (head under-confident).
    acc_hi = torch.tensor([3, 3], device=device)
    acc_lo = torch.tensor([0, 0], device=device)
    ver = torch.tensor([3, 3], device=device)
    for _ in range(1000):
        sts.stage_proposal(slots, logits)
        sts.record(slots, acc_hi, ver)
        sts.stage_proposal(slots, logits)
        sts.record(slots, acc_lo, ver)

    torch.accelerator.synchronize()
    temps = sts.temperatures.cpu()
    calibrated = torch.sigmoid(sts.calibrate(logits)[0]).cpu()
    # pos0 empirical 0.5 vs raw 0.88: temperature must soften (T >> 1,
    # pushed toward the grid edge since sigmoid(2/T) -> 0.5+).
    assert temps[0] > 2.0
    assert calibrated[0] < 0.65
    # pos1/pos2 empirical 1.0 (conditioned on the prefix surviving):
    # temperature sharpens (T < 1).
    assert temps[1] < 1.0 and temps[2] < 1.0
    assert calibrated[1] > 0.9

    # Order preservation within every position, regardless of fit.
    lo = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32, device=device)
    hi = torch.tensor([[3.0, 3.0, 3.0]], dtype=torch.float32, device=device)
    assert (sts.calibrate(hi) > sts.calibrate(lo)).all()


def test_online_sts_ignores_invalid_or_consumed_proposals():
    device = torch.device("cuda")
    sts = DSparkOnlineSTS(max_num_reqs=2, num_steps=2, device=device)
    slots = torch.tensor([0, 1], dtype=torch.int32, device=device)
    logits = torch.ones((2, 2), dtype=torch.float32, device=device)
    accepted = torch.tensor([2, 1], dtype=torch.int32, device=device)
    verified = torch.tensor([2, 2], dtype=torch.int32, device=device)

    pristine_trials = sts.bin_trials.clone()
    pristine_hits = sts.bin_hits.clone()
    sts.stage_proposal(slots, logits)
    sts.invalidate_all()
    sts.record(slots, accepted, verified)
    sts.stage_proposal(slots, logits, valid=False)
    sts.record(slots, accepted, verified)
    torch.accelerator.synchronize()
    assert torch.equal(sts.bin_trials, pristine_trials)
    assert torch.equal(sts.bin_hits, pristine_hits)

    sts.stage_proposal(slots, logits)
    sts.record(slots, accepted, verified)
    recorded_trials = sts.bin_trials.clone()
    recorded_hits = sts.bin_hits.clone()
    sts.record(slots, accepted, verified)
    torch.accelerator.synchronize()
    assert torch.equal(sts.bin_trials, recorded_trials)
    assert torch.equal(sts.bin_hits, recorded_hits)


def test_qwen_dspark_confidence_head_honors_markov_mode(monkeypatch):
    class CapturingLinear(torch.nn.Module):
        def __init__(self, input_size: int, output_size: int, **_kwargs):
            super().__init__()
            self.input_size = input_size
            self.output_size = output_size
            self.last_input: torch.Tensor | None = None

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            assert value.shape[-1] == self.input_size
            self.last_input = value
            return value[..., : self.output_size]

    monkeypatch.setattr(qwen3_dspark, "ReplicatedLinear", CapturingLinear)
    hidden = torch.randn(2, 4)
    markov = torch.randn(2, 3)

    plain = qwen3_dspark.DSparkConfidenceHead(4, prefix="plain", include_markov=False)
    plain(hidden, markov)
    assert isinstance(plain.proj, CapturingLinear)
    assert plain.proj.last_input is not None
    assert plain.proj.last_input.shape == (2, 4)
    assert torch.equal(plain.proj.last_input, hidden.float())

    with_markov = qwen3_dspark.DSparkConfidenceHead(
        7, prefix="markov", include_markov=True
    )
    with_markov(hidden, markov)
    assert isinstance(with_markov.proj, CapturingLinear)
    assert with_markov.proj.last_input is not None
    assert with_markov.proj.last_input.shape == (2, 7)
    assert torch.equal(
        with_markov.proj.last_input, torch.cat([hidden, markov], dim=-1).float()
    )


def test_capacity_based_verification_manager_updates_cpu_capacities():
    device = torch.device("cuda")
    req_states = RequestState(
        max_num_reqs=4,
        max_model_len=4,
        max_num_batched_tokens=16,
        num_speculative_steps=3,
        vocab_size=32,
        device=device,
    )
    req_states.req_id_to_index = {"req0": 2, "req1": 0}
    handler = VarlenCapacityBasedVerificationManager(
        max_num_tokens=16,
        req_states=req_states,
        device=device,
    )
    handler.add_request(2)
    handler.add_request(0)
    input_batch: Any = SimpleNamespace(
        req_ids=["req0", "req1"],
        idx_mapping_np=np.array([2, 0], dtype=np.int32),
        num_tokens=0,
        num_tokens_after_padding=0,
        num_draft_tokens=0,
        num_draft_tokens_per_req=None,
        input_ids=torch.empty(0, dtype=torch.int32, device=device),
        positions=torch.empty(0, dtype=torch.int64, device=device),
        is_padding=torch.empty(0, dtype=torch.bool, device=device),
    )
    draft_token_capacity = torch.tensor([1, 2], dtype=torch.int32, device=device)

    handler.trim_batch(input_batch)
    handler.update_capacities(draft_token_capacity)
    assert handler.copy_event_pending

    torch.accelerator.synchronize()
    handler.trim_batch(input_batch)
    assert handler.draft_token_capacity_np.tolist() == [2, 3, 1, 3]

    handler.update_capacities(draft_token_capacity)
    torch.accelerator.synchronize()
    del req_states.req_id_to_index["req0"]
    handler.draft_token_capacity_np.fill(3)
    handler.trim_batch(input_batch)
    assert handler.draft_token_capacity_np.tolist() == [2, 3, 3, 3]


def test_capacity_update_canonicalizes_across_tp_before_staging(monkeypatch):
    events: list[object] = []
    manager: Any = SimpleNamespace(
        capacity_bypassed=False,
        idx_mapping_np=np.array([0, 1], dtype=np.int32),
        _flush_draft_token_capacity_copy=lambda: events.append("flush"),
        _stage_draft_token_capacity_copy=lambda tensor: events.append(
            ("stage", tensor.tolist())
        ),
    )

    class FakeTPGroup:
        world_size = 2

        @staticmethod
        def broadcast(tensor: torch.Tensor, src: int = 0) -> None:
            events.append(("broadcast", src))
            tensor.copy_(torch.tensor([3, 1], dtype=tensor.dtype))

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tp_group", lambda: FakeTPGroup()
    )

    CapacityBasedVerificationManager.update_capacities(
        manager, torch.tensor([1, 2], dtype=torch.int32)
    )

    assert events == ["flush", ("broadcast", 0), ("stage", [3, 1])]


def test_capacity_manager_bypasses_readback_below_profiled_knee():
    device = torch.device("cuda")
    req_states = RequestState(
        max_num_reqs=4,
        max_model_len=4,
        max_num_batched_tokens=16,
        num_speculative_steps=3,
        vocab_size=32,
        device=device,
    )
    handler = VarlenCapacityBasedVerificationManager(
        max_num_tokens=16,
        req_states=req_states,
        device=device,
    )
    handler.capacity_activation_batch_size = 2
    handler.draft_token_capacity_np.fill(0)

    assert not handler.should_apply_capacity(1)
    assert handler.capacity_bypassed
    assert handler.draft_token_capacity_np.tolist() == [3, 3, 3, 3]

    handler.idx_mapping_np = np.array([0], dtype=np.int32)
    handler.update_capacities(torch.tensor([1], dtype=torch.int32, device=device))
    assert not handler.copy_event_pending

    assert handler.should_apply_capacity(2)
    assert not handler.capacity_bypassed


def test_varlen_capacity_manager_warmup_compacts_inputs_in_place():
    device = torch.device("cuda")
    req_states = RequestState(
        max_num_reqs=4,
        max_model_len=8,
        max_num_batched_tokens=16,
        num_speculative_steps=3,
        vocab_size=32,
        device=device,
    )
    handler = VarlenCapacityBasedVerificationManager(
        max_num_tokens=16,
        req_states=req_states,
        device=device,
    )

    handler.warmup(InputBuffers(4, 16, device))
    torch.accelerator.synchronize()


def test_varlen_capacity_manager_compacts_verifier_batch():
    device = torch.device("cuda")
    req_states = RequestState(
        max_num_reqs=4,
        max_model_len=8,
        max_num_batched_tokens=16,
        num_speculative_steps=3,
        vocab_size=32,
        device=device,
    )
    req_states.last_sampled_tokens[:2] = torch.tensor(
        [[101], [201]], dtype=torch.int64, device=device
    )
    req_states.draft_tokens[:2] = torch.tensor(
        [[11, 12, 13], [21, 22, 23]], dtype=torch.int64, device=device
    )
    handler = VarlenCapacityBasedVerificationManager(
        max_num_tokens=16,
        req_states=req_states,
        device=device,
    )
    handler.add_request(0)
    handler.add_request(1)
    handler.draft_token_capacity_np[:2] = np.array([1, 2], dtype=np.int32)

    input_ids = torch.tensor(
        [101, 11, 12, 13, 201, 21, 22, 23, 0, 0],
        dtype=torch.int32,
        device=device,
    )
    positions = torch.tensor(
        [0, 1, 2, 3, 0, 1, 2, 3, 0, 0],
        dtype=torch.int64,
        device=device,
    )
    is_padding = torch.zeros(10, dtype=torch.bool, device=device)
    input_batch = InputBatch(
        req_ids=["req0", "req1"],
        num_reqs=2,
        num_reqs_after_padding=2,
        idx_mapping=torch.tensor([0, 1], dtype=torch.int32, device=device),
        idx_mapping_np=np.array([0, 1], dtype=np.int32),
        expanded_idx_mapping=torch.tensor(
            [0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32, device=device
        ),
        expanded_local_pos=torch.tensor(
            [0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.int32, device=device
        ),
        num_scheduled_tokens=np.array([4, 4], dtype=np.int32),
        max_query_len=4,
        num_tokens=8,
        num_tokens_after_padding=5,
        num_draft_tokens=6,
        num_draft_tokens_per_req=np.array([3, 3], dtype=np.int32),
        query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32, device=device),
        query_start_loc_np=np.array([0, 4, 8], dtype=np.int32),
        seq_lens=torch.zeros(2, dtype=torch.int32, device=device),
        seq_lens_cpu_upper_bound=torch.tensor([4, 4], dtype=torch.int32),
        max_seq_len_upper_bound=4,
        dcp_local_seq_lens=None,
        num_computed_tokens_np=np.array([0, 0], dtype=np.int32),
        prefill_len_np=np.array([0, 0], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([0, 0], dtype=np.int32),
        is_prefilling_np=np.array([False, False], dtype=np.bool_),
        max_seq_len_np=None,
        input_ids=input_ids[:5],
        positions=positions[:5],
        is_padding=is_padding[:5],
        logits_indices=torch.arange(8, dtype=torch.int64, device=device),
        cu_num_logits=torch.tensor([0, 4, 8], dtype=torch.int32, device=device),
        cu_num_logits_np=np.array([0, 4, 8], dtype=np.int32),
        has_structured_output_reqs=False,
        prompt_lens=None,
    )

    handler.trim_batch(input_batch)

    torch.accelerator.synchronize()
    assert input_batch.num_scheduled_tokens.tolist() == [2, 3]
    assert input_batch.num_draft_tokens_per_req.tolist() == [1, 2]
    assert input_batch.num_tokens == 5
    assert input_batch.num_draft_tokens == 3
    assert input_batch.cu_num_logits_np.tolist() == [0, 2, 5]
    assert input_batch.query_start_loc_np.tolist() == [0, 2, 5]
    assert input_batch.input_ids.shape[0] == input_batch.num_tokens_after_padding
    assert input_batch.input_ids[: input_batch.num_tokens].cpu().tolist() == [
        101,
        11,
        201,
        21,
        22,
    ]
    assert input_batch.positions[: input_batch.num_tokens].cpu().tolist() == [
        0,
        1,
        0,
        1,
        2,
    ]
    assert input_batch.seq_lens.cpu().tolist() == [2, 3]
    assert input_batch.logits_indices.cpu().tolist() == [0, 1, 2, 3, 4]
    assert (
        input_batch.is_padding[: input_batch.num_tokens].cpu().tolist() == [False] * 5
    )


def test_masked_capacity_manager_marks_pruned_tokens_for_forward_and_sampler():
    device = torch.device("cuda")
    req_states = RequestState(
        max_num_reqs=4,
        max_model_len=8,
        max_num_batched_tokens=16,
        num_speculative_steps=3,
        vocab_size=32,
        device=device,
    )
    handler = MaskedCapacityBasedVerificationManager(
        max_num_tokens=16,
        req_states=req_states,
        device=device,
    )
    handler.add_request(0)
    handler.add_request(1)
    handler.draft_token_capacity_np[:2] = np.array([1, 2], dtype=np.int32)

    input_ids = torch.arange(16, dtype=torch.int32, device=device)
    input_batch = InputBatch(
        req_ids=["req0", "req1"],
        num_reqs=2,
        num_reqs_after_padding=2,
        idx_mapping=torch.tensor([0, 1], dtype=torch.int32, device=device),
        idx_mapping_np=np.array([0, 1], dtype=np.int32),
        expanded_idx_mapping=torch.tensor(
            [0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32, device=device
        ),
        expanded_local_pos=torch.tensor(
            [0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.int32, device=device
        ),
        num_scheduled_tokens=np.array([4, 4], dtype=np.int32),
        max_query_len=4,
        num_tokens=8,
        num_tokens_after_padding=10,
        num_draft_tokens=6,
        num_draft_tokens_per_req=np.array([3, 3], dtype=np.int32),
        query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32, device=device),
        query_start_loc_np=np.array([0, 4, 8], dtype=np.int32),
        seq_lens=torch.tensor([4, 4], dtype=torch.int32, device=device),
        seq_lens_cpu_upper_bound=torch.tensor([4, 4], dtype=torch.int32),
        max_seq_len_upper_bound=4,
        dcp_local_seq_lens=None,
        num_computed_tokens_np=np.array([0, 0], dtype=np.int32),
        prefill_len_np=np.array([0, 0], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([0, 0], dtype=np.int32),
        is_prefilling_np=np.array([False, False], dtype=np.bool_),
        max_seq_len_np=None,
        input_ids=input_ids,
        positions=torch.arange(16, dtype=torch.int64, device=device),
        is_padding=torch.zeros(10, dtype=torch.bool, device=device),
        logits_indices=torch.arange(8, dtype=torch.int64, device=device),
        cu_num_logits=torch.tensor([0, 4, 8], dtype=torch.int32, device=device),
        cu_num_logits_np=np.array([0, 4, 8], dtype=np.int32),
        has_structured_output_reqs=False,
        prompt_lens=None,
    )

    handler.trim_batch(input_batch)
    slot_mappings = torch.arange(20, dtype=torch.int64, device=device).view(2, 10)
    slot_mappings.masked_fill_(
        input_batch.is_padding[: slot_mappings.shape[1]].unsqueeze(0),
        PAD_SLOT_ID,
    )
    draft_sampled = input_batch.input_ids[input_batch.logits_indices]
    draft_sampled.masked_fill_(input_batch.is_padding[input_batch.logits_indices], -1)

    torch.accelerator.synchronize()
    assert input_batch.num_scheduled_tokens.tolist() == [4, 4]
    assert input_batch.num_draft_tokens_per_req.tolist() == [3, 3]
    assert input_batch.is_padding.cpu().tolist() == [
        False,
        False,
        True,
        True,
        False,
        False,
        False,
        True,
        False,
        False,
    ]
    assert slot_mappings.cpu().tolist() == [
        [0, 1, -1, -1, 4, 5, 6, -1, 8, 9],
        [10, 11, -1, -1, 14, 15, 16, -1, 18, 19],
    ]
    assert draft_sampled.cpu().tolist() == [0, 1, -1, -1, 4, 5, 6, -1]


def test_capacity_cudagraph_dispatch_filters_by_max_query_len():
    manager = object.__new__(CudaGraphManager)
    manager._graphs_captured = True
    manager._resolve_effective_loras = lambda num_loras: num_loras
    regular_desc = BatchExecutionDescriptor(
        CUDAGraphMode.FULL,
        num_tokens=12,
        num_reqs=12,
        uniform_token_count=6,
    )
    full_capacity_desc = BatchExecutionDescriptor(
        CUDAGraphMode.FULL,
        num_tokens=15,
        num_reqs=4,
        max_req_tokens=6,
    )
    piecewise_desc = BatchExecutionDescriptor(
        CUDAGraphMode.PIECEWISE,
        num_tokens=16,
        num_reqs=None,
    )
    manager._candidates = {
        (11, 0): [
            regular_desc,
            full_capacity_desc,
            piecewise_desc,
        ]
    }

    desc = CudaGraphManager.dispatch(
        manager,
        num_reqs=4,
        num_tokens=11,
        uniform_token_count=None,
        num_active_loras=0,
        max_req_tokens=6,
    )

    assert desc is full_capacity_desc

    desc = CudaGraphManager.dispatch(
        manager,
        num_reqs=4,
        num_tokens=11,
        uniform_token_count=None,
        num_active_loras=0,
        max_req_tokens=7,
    )

    assert desc is piecewise_desc

    manager._candidates[(15, 0)] = [
        regular_desc,
        full_capacity_desc,
        piecewise_desc,
    ]
    desc = CudaGraphManager.dispatch(
        manager,
        num_reqs=4,
        num_tokens=15,
        uniform_token_count=None,
        num_active_loras=0,
        max_req_tokens=6,
    )

    assert desc is full_capacity_desc

    desc = CudaGraphManager.dispatch(
        manager,
        num_reqs=4,
        num_tokens=11,
        uniform_token_count=6,
        num_active_loras=0,
    )

    assert desc is regular_desc


def test_varlen_cudagraph_capture_adds_full_desc():
    manager = object.__new__(CudaGraphManager)
    manager.vllm_config = SimpleNamespace(speculative_config=None)
    manager.compilation_config = SimpleNamespace(
        cudagraph_capture_sizes=[5],
        max_cudagraph_capture_size=16,
    )
    manager.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    manager.decode_query_len = 4
    manager.varlen_spec_decode = True
    manager.max_num_reqs = 16
    manager.lora_capture_cases = [0]
    manager._candidates = {}
    manager._capture_descs = {}

    manager._init_candidates()

    assert any(
        desc.max_req_tokens == manager.decode_query_len
        for desc in manager._capture_descs[CUDAGraphMode.FULL]
    )


def test_varlen_cudagraph_prefers_uniform_below_capacity_knee(monkeypatch):
    monkeypatch.setenv("VLLM_DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE", "4")
    manager = object.__new__(CudaGraphManager)
    manager.vllm_config = SimpleNamespace(speculative_config=None)
    manager.compilation_config = SimpleNamespace(
        cudagraph_capture_sizes=[12, 16],
        max_cudagraph_capture_size=64,
    )
    manager.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    manager.decode_query_len = 4
    manager.varlen_spec_decode = True
    manager.max_num_reqs = 16
    manager.lora_capture_cases = [0]
    manager._lora_dispatch_map = {}
    manager._max_lora_case = 0
    manager._candidates = {}
    manager._capture_descs = {}

    manager._init_candidates()
    manager._graphs_captured = True

    low_load_desc = manager.dispatch(
        num_reqs=3,
        num_tokens=12,
        uniform_token_count=4,
        num_active_loras=0,
        max_req_tokens=4,
    )
    assert low_load_desc.uniform_token_count == 4
    assert low_load_desc.num_reqs == 3

    capacity_desc = manager.dispatch(
        num_reqs=4,
        num_tokens=13,
        uniform_token_count=None,
        num_active_loras=0,
        max_req_tokens=4,
    )
    assert capacity_desc.uniform_token_count is None
    assert capacity_desc.max_req_tokens == 4
    assert capacity_desc.num_tokens == 16


def test_varlen_cudagraph_dispatch_skips_incompatible_uniform_grid():
    manager = object.__new__(CudaGraphManager)
    manager.vllm_config = SimpleNamespace(speculative_config=None)
    manager.compilation_config = SimpleNamespace(
        cudagraph_capture_sizes=[96, 104],
        max_cudagraph_capture_size=512,
    )
    manager.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    manager.decode_query_len = 4
    manager.varlen_spec_decode = True
    manager.max_num_reqs = 128
    manager.lora_capture_cases = [0]
    manager._lora_dispatch_map = {}
    manager._max_lora_case = 0
    manager._candidates = {}
    manager._capture_descs = {}

    manager._init_candidates()
    manager._graphs_captured = True

    desc = manager.dispatch(
        num_reqs=32,
        num_tokens=97,
        uniform_token_count=None,
        num_active_loras=0,
        max_req_tokens=4,
    )

    assert desc.cg_mode == CUDAGraphMode.FULL
    assert desc.num_tokens == 104
    assert desc.max_req_tokens == 4
