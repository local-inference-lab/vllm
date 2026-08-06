# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.v1.outputs import DraftTokenIds
from vllm.v1.worker.gpu.async_utils import async_copy_to_np
from vllm.v1.worker.gpu.input_batch import InputBatch

# Salt added to the Philox offsets used for draft-token Gumbel noise.
# Positions are bounded by max_model_len, so this puts the draft stream in a
# range disjoint from the target-side offsets.
DRAFT_GUMBEL_POS_OFFSET = 1 << 30


def draft_gumbel_pos(positions: torch.Tensor) -> torch.Tensor:
    """Philox offsets for draft Gumbel noise, given draft-row positions.

    The rejection sampler keys both its acceptance uniform and recovery
    Gumbel noise for position P by offset P. Keep the draft proposal on a
    disjoint Philox range so rejection sampling remains unbiased.
    """
    return positions + (1 + DRAFT_GUMBEL_POS_OFFSET)


def limit_draft_tokens(
    draft_tokens: torch.Tensor,
    num_speculative_tokens: int,
    max_num_speculative_tokens: int,
) -> torch.Tensor:
    """Limit a speculator's output to the scheduler-selected depth."""
    if draft_tokens.ndim != 2:
        raise RuntimeError(
            "Speculator returned unsupported draft shape "
            f"{tuple(draft_tokens.shape)}; expected a 2D tensor."
        )
    if not 1 <= num_speculative_tokens <= max_num_speculative_tokens:
        raise RuntimeError(
            "Scheduler selected an invalid speculative-token count "
            f"{num_speculative_tokens}; expected a value between 1 and "
            f"{max_num_speculative_tokens}."
        )
    if draft_tokens.shape[1] == 0:
        return draft_tokens
    if draft_tokens.shape[1] < num_speculative_tokens:
        raise RuntimeError(
            "Speculator returned too few draft tokens "
            f"({draft_tokens.shape[1]}); expected at least "
            f"{num_speculative_tokens}."
        )
    return draft_tokens[:, :num_speculative_tokens]


class DraftTokensHandler:
    def __init__(
        self,
        device: torch.device | None = None,
        needs_real_draft_tokens: bool = False,
    ):
        self.device = device
        self.needs_real_draft_tokens = needs_real_draft_tokens
        self.copy_stream = torch.cuda.Stream(device)
        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
        self.copy_event = torch.cuda.Event(blocking=True)

        self.req_ids: list[str] = []
        self.draft_tokens_np: np.ndarray | None = None
        self.num_draft_tokens: int = 0

    def needs_host_copy(self, input_batch: InputBatch) -> bool:
        return self.needs_real_draft_tokens or input_batch.has_structured_output_reqs

    def set_draft_tokens(
        self, input_batch: InputBatch, draft_tokens: torch.Tensor
    ) -> None:
        self.req_ids = input_batch.req_ids
        self.num_draft_tokens = draft_tokens.shape[1]
        if not self.needs_host_copy(input_batch):
            # Fixed-width speculators use scheduler placeholders. Avoid a D2H
            # copy and per-step event synchronization on their decode path.
            self.draft_tokens_np = None
            return

        # The scheduler needs the real draft lengths. Some speculators use
        # -1 as a sentinel for fallback/no-draft slots; sending placeholder
        # -1 values back to the scheduler can cause them to be scheduled as
        # verifier inputs on the next step.
        current_stream = torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.copy_stream):
            self.draft_tokens_np = async_copy_to_np(draft_tokens)
            # draft_tokens is a temporary allocation on the main stream and read here on
            # copy_stream; without record_stream, the caching allocator may reuse its
            # memory before the async copy executes.
            draft_tokens.record_stream(self.copy_stream)
            self.copy_event.record()

    def get_draft_tokens(self) -> DraftTokenIds | None:
        if self.draft_tokens_np is not None:
            self.copy_event.synchronize()
            draft_token_ids = self.draft_tokens_np.tolist()
            for token_ids in draft_token_ids:
                for i, token_id in enumerate(token_ids):
                    if token_id < 0:
                        del token_ids[i:]
                        break
        else:
            # Fixed-width async scheduling only needs the draft count. The
            # worker retains the actual token ids for verification.
            draft_token_ids = [[-1] * self.num_draft_tokens for _ in self.req_ids]
        return DraftTokenIds(self.req_ids, draft_token_ids)


def get_parallel_drafting_token_id(hf_config) -> int:
    """Resolve the mask token id used for parallel drafting slots.

    Checks (in order): `dflash_config.mask_token_id`, top-level `mask_token_id`,
    `dspark_noise_token_id`, `pard_token`, `ptd_token_id`. Raises ValueError if
    none are present.
    """
    dflash_config = getattr(hf_config, "dflash_config", None) or {}
    if "mask_token_id" in dflash_config:
        return int(dflash_config["mask_token_id"])
    if getattr(hf_config, "mask_token_id", None) is not None:
        return int(hf_config.mask_token_id)
    if hasattr(hf_config, "dspark_noise_token_id"):
        return int(hf_config.dspark_noise_token_id)
    if hasattr(hf_config, "pard_token"):
        return int(hf_config.pard_token)
    if hasattr(hf_config, "ptd_token_id"):
        return int(hf_config.ptd_token_id)
    raise ValueError(
        "Model config must specify `dflash_config.mask_token_id`,"
        " `mask_token_id`, `dspark_noise_token_id`, `pard_token`, or"
        " `ptd_token_id` for parallel drafting."
    )
