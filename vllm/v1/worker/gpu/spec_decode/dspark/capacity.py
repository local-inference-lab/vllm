# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _compute_prefix_survival_probabilities_kernel(
    confidence_logits_ptr,
    survival_probs_ptr,
    inv_temperature,
    CONFIDENCE_STRIDE: tl.constexpr,
    SURVIVAL_STRIDE: tl.constexpr,
    NUM_SPECULATIVE_STEPS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    survival_prob = tl.full((), 1.0, tl.float32)
    for step in tl.static_range(0, NUM_SPECULATIVE_STEPS):
        confidence_logit = tl.load(
            confidence_logits_ptr + req_idx * CONFIDENCE_STRIDE + step
        ).to(tl.float32)
        confidence_prob = 1.0 / (1.0 + tl.exp(-confidence_logit * inv_temperature))
        survival_prob *= confidence_prob
        tl.store(
            survival_probs_ptr + req_idx * SURVIVAL_STRIDE + step,
            survival_prob,
        )


@triton.jit
def _allocate_draft_token_capacity_kernel(
    survival_probs_ptr,
    capacity_ptr,
    runtime_num_reqs_ptr,
    sps_table_ptr,
    min_survival_probability,
    REQ_BLOCK: tl.constexpr,
    NUM_SPECULATIVE_STEPS: tl.constexpr,
    MAX_ADMISSIONS: tl.constexpr,
    USE_BUDGET: tl.constexpr,
    BUDGET_FRAC: tl.constexpr,
    USE_SPS: tl.constexpr,
    SURVIVAL_STRIDE: tl.constexpr,
):
    offsets = tl.arange(0, REQ_BLOCK)
    runtime_num_reqs = tl.load(runtime_num_reqs_ptr).to(tl.int32)
    active = offsets < runtime_num_reqs

    if USE_BUDGET:
        total_admissions = runtime_num_reqs * NUM_SPECULATIVE_STEPS
        max_admissions = tl.minimum(
            tl.ceil(total_admissions.to(tl.float32) * BUDGET_FRAC).to(tl.int32),
            total_admissions,
        )
        # DSpark Algorithm 1: greedy global admission over the candidate set
        # {(r, j) | survival(r, j) > 0}, sorted by prefix-survival score. The
        # admission counts ARE the capacities, so the spent budget is exactly
        # sum(capacities). (Re-deriving capacities from the kth-score
        # threshold would blow past the budget whenever scores tie, e.g.
        # saturated sigmoids, and zero-survival tokens are never candidates.)
        # With an SPS curve, stop at the admission count k* maximizing
        # expected throughput theta = tau * SPS(B), where after k admissions
        # tau = R + sum of admitted survival scores and B = R + k
        # verification tokens (one bonus token per request).
        lengths = tl.full((REQ_BLOCK,), 0, tl.int32)
        if USE_SPS:
            tau = runtime_num_reqs * 1.0
            best_theta = tau * tl.load(sps_table_ptr + runtime_num_reqs)
            best_lengths = lengths
        for admission_idx in tl.range(0, MAX_ADMISSIONS):
            has_next = (
                active
                & (admission_idx < max_admissions)
                & (lengths < NUM_SPECULATIVE_STEPS)
            )
            next_scores = tl.load(
                survival_probs_ptr + offsets * SURVIVAL_STRIDE + lengths,
                mask=has_next,
                other=-1.0,
            )
            best_score, best_idx = tl.max(next_scores, axis=0, return_indices=True)
            admit = best_score > 0.0
            lengths += tl.where(admit & (offsets == best_idx), 1, 0)
            if USE_SPS:
                tau += tl.where(admit, best_score, 0.0)
                sps = tl.load(sps_table_ptr + runtime_num_reqs + admission_idx + 1)
                theta = tau * sps
                better = admit & (theta > best_theta)
                best_theta = tl.where(better, theta, best_theta)
                best_lengths = tl.where(better, lengths, best_lengths)
        capacities = best_lengths if USE_SPS else lengths
    else:
        capacities = tl.full((REQ_BLOCK,), 0, tl.int32)
        for step in tl.static_range(0, NUM_SPECULATIVE_STEPS):
            scores = tl.load(
                survival_probs_ptr + offsets * SURVIVAL_STRIDE + step,
                mask=active,
                other=-1.0,
            )
            capacities += tl.where(scores >= min_survival_probability, 1, 0)

    tl.store(capacity_ptr + offsets, capacities, mask=active)


def build_sps_table(
    sps_curve: list[tuple[int, float]],
    max_batch_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    """Densify (batch_num_tokens, steps_per_sec) breakpoints into a lookup
    table indexed by verification batch token count, linearly interpolated
    and clamped at the ends."""
    xs = np.array([b for b, _ in sps_curve], dtype=np.float64)
    ys = np.array([s for _, s in sps_curve], dtype=np.float64)
    table = np.interp(np.arange(max_batch_tokens + 1), xs, ys)
    return torch.tensor(table, dtype=torch.float32, device=device)


def compute_draft_token_capacity_from_confidence(
    confidence_logits: torch.Tensor,
    draft_token_capacity: torch.Tensor,
    min_survival_probability: float,
    num_reqs: int,
    num_speculative_steps: int,
    runtime_num_reqs: torch.Tensor,
    survival_probs: torch.Tensor | None = None,
    budget_frac: float = 1.0,
    sps_table: torch.Tensor | None = None,
    confidence_temperature: float = 1.0,
) -> None:
    if num_reqs == 0 or num_speculative_steps == 0:
        return
    if survival_probs is None:
        survival_probs = torch.empty_like(confidence_logits)
    _compute_prefix_survival_probabilities_kernel[(num_reqs,)](
        confidence_logits,
        survival_probs,
        1.0 / confidence_temperature,
        CONFIDENCE_STRIDE=confidence_logits.stride(0),
        SURVIVAL_STRIDE=survival_probs.stride(0),
        NUM_SPECULATIVE_STEPS=num_speculative_steps,
    )
    # Even when the budget covers every token, zero-survival tokens are
    # not admission candidates (DSpark Alg. 1), so always run the kernel.
    use_budget = min_survival_probability <= 0.0
    use_sps = use_budget and sps_table is not None
    # Pow2-padded so one compiled variant serves all runtime_num_reqs values
    # under CUDA graph capture.
    kernel_num_reqs = triton.next_power_of_2(max(num_reqs, 1))
    if use_sps:
        assert sps_table is not None
        # The theta scan reads SPS(B) for B up to kernel_num_reqs * (1 + n_spec).
        assert sps_table.shape[0] > kernel_num_reqs * (1 + num_speculative_steps), (
            f"SPS table has {sps_table.shape[0]} entries but batch token "
            f"counts can reach {kernel_num_reqs * (1 + num_speculative_steps)}"
        )
    if sps_table is None:
        sps_table = draft_token_capacity
    _allocate_draft_token_capacity_kernel[(1,)](
        survival_probs,
        draft_token_capacity,
        runtime_num_reqs,
        sps_table,
        min_survival_probability,
        REQ_BLOCK=kernel_num_reqs,
        NUM_SPECULATIVE_STEPS=num_speculative_steps,
        MAX_ADMISSIONS=kernel_num_reqs * num_speculative_steps,
        USE_BUDGET=use_budget,
        BUDGET_FRAC=budget_frac,
        USE_SPS=use_sps,
        SURVIVAL_STRIDE=survival_probs.stride(0),
    )
