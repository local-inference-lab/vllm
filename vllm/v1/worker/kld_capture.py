# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in raw prompt-logit capture shared by the V1 and V2 model runners."""

import os
from pathlib import Path

import torch

from vllm.distributed.parallel_state import is_global_first_rank
from vllm.logger import init_logger

logger = init_logger(__name__)


def is_kld_prompt_logit_capture_enabled() -> bool:
    return bool(os.environ.get("VLLM_KLD_CAPTURE_DIR")) and is_global_first_rank()


def maybe_capture_kld_prompt_logits(
    logits: torch.Tensor,
    *,
    req_id: str,
    start_idx: int,
    vocab_size: int,
) -> None:
    """Persist one full-vocabulary prompt-logit chunk when opted in."""
    capture_dir = os.environ.get("VLLM_KLD_CAPTURE_DIR")
    if not capture_dir or not is_global_first_rank():
        return

    safe_req_id = "".join(
        char if char.isalnum() or char in "-_." else "_" for char in req_id
    )
    request_dir = Path(capture_dir) / safe_req_id
    request_dir.mkdir(parents=True, exist_ok=True)
    end_idx = start_idx + logits.shape[0]
    output_path = request_dir / (
        f"logits.rows-{start_idx:06d}-{end_idx:06d}.safetensors"
    )
    if output_path.exists():
        raise RuntimeError(f"Refusing to overwrite KLD capture chunk: {output_path}")

    # Transfer BF16/FP16 logits first, then widen on CPU. This avoids a large
    # transient float32 allocation on a GPU that is already weight-bound.
    logits_cpu = logits[:, :vocab_size].detach().to(device="cpu").float().contiguous()
    from safetensors.torch import save_file

    save_file(
        {"logits": logits_cpu},
        str(output_path),
        metadata={
            "request_id": req_id,
            "row_start": str(start_idx),
            "row_end": str(end_idx),
            "vocab_size": str(vocab_size),
        },
    )
    logger.info(
        "Saved KLD prompt-logit rows [%d, %d) shape=%s to %s",
        start_idx,
        end_idx,
        tuple(logits_cpu.shape),
        output_path,
    )
