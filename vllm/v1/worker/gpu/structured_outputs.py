# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import InputBatch


def _build_grammar_row_mapping(
    req_ids: list[str],
    grammar_req_ids: list[str],
    grammar_num_spec_tokens: list[int],
    grammar_has_bonus_token: list[bool],
    cu_num_logits_np: np.ndarray,
    num_draft_tokens_per_req: np.ndarray | None,
    num_bonus_tokens: int,
) -> tuple[list[int], list[int]]:
    """Map serialized grammar rows to the active compact logits layout."""
    assert (
        len(grammar_req_ids)
        == len(grammar_num_spec_tokens)
        == len(grammar_has_bonus_token)
    )
    assert num_bonus_tokens in (0, 1)

    req_id_to_idx = {req_id: i for i, req_id in enumerate(req_ids)}
    source_indices: list[int] = []
    logits_indices: list[int] = []
    source_offset = 0

    for grammar_req_id, num_source_drafts, has_source_bonus_token in zip(
        grammar_req_ids,
        grammar_num_spec_tokens,
        grammar_has_bonus_token,
        strict=True,
    ):
        req_idx = req_id_to_idx[grammar_req_id]
        num_active_drafts = (
            0
            if num_draft_tokens_per_req is None
            else int(num_draft_tokens_per_req[req_idx])
        )
        assert 0 <= num_active_drafts <= num_source_drafts

        logits_start = int(cu_num_logits_np[req_idx])
        num_active_logits = int(
            cu_num_logits_np[req_idx + 1] - cu_num_logits_np[req_idx]
        )
        assert num_active_logits == num_active_drafts + num_bonus_tokens

        source_indices.extend(range(source_offset, source_offset + num_active_drafts))
        logits_indices.extend(range(logits_start, logits_start + num_active_drafts))
        if num_bonus_tokens:
            assert has_source_bonus_token
            source_indices.append(source_offset + num_source_drafts)
            logits_indices.append(logits_start + num_active_drafts)

        source_offset += num_source_drafts + int(has_source_bonus_token)

    return source_indices, logits_indices


class StructuredOutputsWorker:
    def __init__(
        self,
        max_num_logits: int,
        vocab_size: int,
        device: torch.device,
        num_bonus_tokens: int,
    ):
        self.logits_indices = torch.zeros(
            max_num_logits, dtype=torch.int32, device=device
        )
        self.grammar_bitmask = torch.zeros(
            (max_num_logits, cdiv(vocab_size, 32)), dtype=torch.int32, device=device
        )
        self.device = device
        self.copy_stream = torch.cuda.Stream()
        self.num_bonus_tokens = num_bonus_tokens

    def apply_grammar_bitmask(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        grammar_req_ids: list[str],
        grammar_bitmask: np.ndarray,
        grammar_num_spec_tokens: list[int],
        grammar_has_bonus_token: list[bool],
    ) -> None:
        if not grammar_req_ids:
            return

        source_indices, mapping = _build_grammar_row_mapping(
            input_batch.req_ids,
            grammar_req_ids,
            grammar_num_spec_tokens,
            grammar_has_bonus_token,
            input_batch.cu_num_logits_np,
            input_batch.num_draft_tokens_per_req,
            self.num_bonus_tokens,
        )
        expected_source_rows = sum(
            num_drafts + int(has_bonus_token)
            for num_drafts, has_bonus_token in zip(
                grammar_num_spec_tokens, grammar_has_bonus_token, strict=True
            )
        )
        assert grammar_bitmask.shape[0] == expected_source_rows
        grammar_bitmask = grammar_bitmask[source_indices]

        # Asynchronously copy the active bitmask rows to GPU.
        with torch.cuda.stream(self.copy_stream):
            bitmask = async_copy_to_gpu(
                grammar_bitmask, out=self.grammar_bitmask[: grammar_bitmask.shape[0]]
            )

        # Asynchronously copy the mapping to GPU.
        with torch.cuda.stream(self.copy_stream):
            logits_indices = torch.tensor(
                mapping, dtype=torch.int32, device="cpu", pin_memory=True
            )
            logits_indices = self.logits_indices[: len(mapping)].copy_(
                logits_indices, non_blocking=True
            )

        # Ensure all async copies are complete before launching the kernel.
        current_stream = torch.cuda.current_stream()
        current_stream.wait_stream(self.copy_stream)

        num_masks = bitmask.shape[0]
        assert num_masks == len(mapping)
        vocab_size = logits.shape[-1]
        BLOCK_SIZE = 8192
        grid = (num_masks, triton.cdiv(vocab_size, BLOCK_SIZE))
        _apply_grammar_bitmask_kernel[grid](
            logits,
            logits.stride(0),
            logits_indices,
            bitmask,
            bitmask.stride(0),
            vocab_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # Ensure the copy stream waits for the device tensors to finish being used
        # before it re-uses or deallocates them
        self.copy_stream.wait_stream(current_stream)


# Adapted from
# https://github.com/mlc-ai/xgrammar/blob/main/python/xgrammar/kernels/apply_token_bitmask_inplace_triton.py
@triton.jit
def _apply_grammar_bitmask_kernel(
    logits_ptr,
    logits_stride,
    logits_indices_ptr,
    bitmask_ptr,
    bitmask_stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    bitmask_idx = tl.program_id(0)
    logits_idx = tl.load(logits_indices_ptr + bitmask_idx)

    # Load the bitmask.
    block_id = tl.program_id(1)
    bitmask_offset = (block_id * BLOCK_SIZE) // 32 + tl.arange(0, BLOCK_SIZE // 32)
    packed_bitmask = tl.load(
        bitmask_ptr + bitmask_idx * bitmask_stride + bitmask_offset,
        mask=bitmask_offset < bitmask_stride,
    )
    # Unpack the bitmask.
    bitmask = ((packed_bitmask[:, None] >> (tl.arange(0, 32)[None, :])) & 1) == 0
    bitmask = bitmask.reshape(BLOCK_SIZE)

    # Apply the bitmask to the logits.
    block_offset = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(
        logits_ptr + logits_idx * logits_stride + block_offset,
        -float("inf"),
        mask=bitmask & (block_offset < vocab_size),
    )
