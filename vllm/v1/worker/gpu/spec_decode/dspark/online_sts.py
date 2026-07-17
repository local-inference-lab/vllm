# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class DSparkOnlineSTS:
    """Online Sequential Temperature Scaling for the DSpark capacity scheduler.

    The paper (Section 3.2.1) calibrates each position's conditional survival
    probability with a per-position temperature chosen to minimize the
    Expected Calibration Error of the cumulative product on a validation set
    — an order-preserving transform. This is the serving-time analogue: per
    position it maintains binned empirical conditional acceptance
    P(accept_k | prefix accepted, position verified) from the rejection
    sampler's own outcomes (exponential decay), and each step fits the
    temperature that minimizes the trial-weighted ECE of
    sigmoid(logit / T_k) against those bins. Fitting each conditional
    directly is the chain-rule equivalent of the paper's sequential
    cumulative-product fit, with censored online observations instead of a
    held-out set.

    Observations are censored by capacity (unverified positions yield no
    trials); under light load the theta-argmax verifies every candidate,
    which is where the observation mass comes from. With few observations
    the fitted temperature is blended toward 1.0 (identity), so cold-start
    behaves like the raw confidence head.

    All buffers are persistent: ``temperatures`` is read inside the captured
    draft graph, while ``record()`` runs eagerly each step. All reductions
    are one-hot sums (no atomics) so the state stays bitwise identical
    across TP ranks.
    """

    DECAY = 0.999
    PRIOR_WEIGHT = 64.0
    NUM_BINS = 16
    LOGIT_RANGE = 8.0
    # Log-spaced temperature grid; 1.0 is on the grid so a well-calibrated
    # head fits the identity exactly.
    TEMP_GRID_MIN = 0.125
    TEMP_GRID_MAX = 8.0
    TEMP_GRID_SIZE = 49

    def __init__(self, max_num_reqs: int, num_steps: int, device: torch.device):
        self.num_steps = num_steps
        # Raw head logits of each request's latest proposal, by req-state
        # slot, so verification outcomes (one step later, possibly reordered)
        # can be joined back to the confidences that produced them.
        self.logits_by_state = torch.zeros(
            max_num_reqs, num_steps, dtype=torch.float32, device=device
        )
        self.proposal_valid_by_state = torch.zeros(
            max_num_reqs, dtype=torch.bool, device=device
        )
        # EMA counters per (position, logit bin).
        self.bin_trials = torch.zeros(
            num_steps, self.NUM_BINS, dtype=torch.float32, device=device
        )
        self.bin_hits = torch.zeros_like(self.bin_trials)
        # Per-position temperatures; persistent so captured graphs see
        # updates. Identity until observations accumulate.
        self.temperatures = torch.ones(num_steps, dtype=torch.float32, device=device)

        self._steps = torch.arange(num_steps, device=device)
        self._bins = torch.arange(self.NUM_BINS, device=device)
        bin_width = 2 * self.LOGIT_RANGE / self.NUM_BINS
        self._bin_mids = (
            -self.LOGIT_RANGE + (self._bins.to(torch.float32) + 0.5) * bin_width
        )
        self._temp_grid = torch.logspace(
            torch.log10(torch.tensor(self.TEMP_GRID_MIN)),
            torch.log10(torch.tensor(self.TEMP_GRID_MAX)),
            self.TEMP_GRID_SIZE,
            device=device,
        )
        # sigmoid(mid_b / T) for every (T, bin) pair, fixed for the run.
        self._grid_probs = torch.sigmoid(
            self._bin_mids.unsqueeze(0) / self._temp_grid.unsqueeze(1)
        )
        self._log_temp_grid = self._temp_grid.log()
        self._log_interval = int(
            os.environ.get("VLLM_DSPARK_STS_LOG_INTERVAL", "0") or "0"
        )
        self._record_count = 0

    def stage_proposal(
        self,
        req_state_indices: torch.Tensor,
        logits: torch.Tensor,
        *,
        valid: bool = True,
    ) -> None:
        """Remember the raw head logits of the current proposal."""
        if valid:
            rows = self.logits_by_state[req_state_indices]
            rows.zero_()
            rows[:, : logits.shape[1]] = logits
            self.logits_by_state[req_state_indices] = rows
        self.proposal_valid_by_state[req_state_indices] = valid

    def invalidate_all(self) -> None:
        """Discard proposals whose confidence logits were bypassed or profiled."""
        self.proposal_valid_by_state.zero_()

    def calibrate(
        self, logits: torch.Tensor, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply the per-position temperatures (order-preserving)."""
        return torch.div(logits, self.temperatures[: logits.shape[-1]], out=out)

    def record(
        self,
        req_state_indices: torch.Tensor,
        num_accepted: torch.Tensor,
        num_verified: torch.Tensor,
    ) -> None:
        """Fold one verification step's outcomes into the calibration.

        Args:
            req_state_indices: [num_reqs] req-state slot of each request.
            num_accepted: [num_reqs] accepted draft tokens this step.
            num_verified: [num_reqs] draft tokens that were verified
                (post-capacity), zero for rows without drafts.
        """
        logits = self.logits_by_state[req_state_indices]
        proposal_valid = self.proposal_valid_by_state[req_state_indices]
        bin_width = 2 * self.LOGIT_RANGE / self.NUM_BINS
        bin_idx = ((logits + self.LOGIT_RANGE) / bin_width).long()
        bin_idx.clamp_(0, self.NUM_BINS - 1)

        k = self._steps.unsqueeze(0)
        # Position k (0-based) is evaluated iff the k-token prefix before it
        # was accepted and it was inside the verified capacity.
        trial = k < torch.minimum(num_accepted + 1, num_verified).unsqueeze(1)
        trial.logical_and_(proposal_valid.unsqueeze(1))
        hit = (k < num_accepted.unsqueeze(1)) & trial

        # One-hot reduction (deterministic; index_add_ atomics are not).
        onehot = bin_idx.unsqueeze(-1) == self._bins  # [reqs, steps, bins]
        # Do not age the calibration state when every row was invalid. Keeping
        # this as a device scalar avoids a host synchronization on the hot path.
        decay = 1.0 - trial.any().to(torch.float32) * (1.0 - self.DECAY)
        self.bin_trials.mul_(decay).add_(
            (trial.unsqueeze(-1) & onehot).sum(0).to(torch.float32)
        )
        self.bin_hits.mul_(decay).add_(
            (hit.unsqueeze(-1) & onehot).sum(0).to(torch.float32)
        )
        self.proposal_valid_by_state[req_state_indices] = False

        # Per-position 1D grid search: T_k minimizing trial-weighted ECE of
        # sigmoid(mid_b / T) against the empirical bin acceptance.
        emp = self.bin_hits / self.bin_trials.clamp(min=1e-6)
        err = (self._grid_probs.unsqueeze(1) - emp).abs()  # [T, steps, bins]
        ece = (err * self.bin_trials).sum(-1)  # [T, steps]
        log_t = self._log_temp_grid[ece.argmin(0)]
        # Blend toward the identity until enough outcomes accumulate.
        total = self.bin_trials.sum(-1)
        torch.exp(log_t * (total / (total + self.PRIOR_WEIGHT)), out=self.temperatures)
        self._maybe_log_diagnostics(total)

    def _maybe_log_diagnostics(self, total_trials: torch.Tensor) -> None:
        if self._log_interval <= 0:
            return
        self._record_count += 1
        if self._record_count % self._log_interval:
            return
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return

        denom = total_trials.clamp(min=1e-6)
        empirical_mean = self.bin_hits.sum(-1) / denom
        calibrated_bin_probs = torch.sigmoid(
            self._bin_mids.unsqueeze(0) / self.temperatures.unsqueeze(1)
        )
        predicted_mean = (calibrated_bin_probs * self.bin_trials).sum(-1) / denom
        raw_logit_mean = (self._bin_mids.unsqueeze(0) * self.bin_trials).sum(-1) / denom
        logger.info(
            "DSpark online STS: trials=%s empirical_cond=%s "
            "predicted_cond=%s temperatures=%s raw_logit_mean=%s",
            [round(x, 1) for x in total_trials.tolist()],
            [round(x, 4) for x in empirical_mean.tolist()],
            [round(x, 4) for x in predicted_mean.tolist()],
            [round(x, 4) for x in self.temperatures.tolist()],
            [round(x, 4) for x in raw_logit_mean.tolist()],
        )
