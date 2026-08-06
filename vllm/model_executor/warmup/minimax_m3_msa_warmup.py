# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import numpy as np
import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.models.minimax_m3.nvidia.model import MiniMaxM3SparseAttention
from vllm.platforms import current_platform
from vllm.tracing import instrument

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def _supports_minimax_m3_msa_warmup() -> bool:
    if not current_platform.is_cuda():
        return False
    if envs.VLLM_USE_B12X_MINIMAX_M3_MSA:
        return current_platform.is_device_capability_family(120)
    return current_platform.is_device_capability_family(
        100
    ) or current_platform.is_device_capability_family(120)


def _dense_attention_backend_names(worker: "Worker") -> set[str]:
    names: set[str] = set()
    for module in worker.get_model().modules():
        if not isinstance(module, Attention):
            continue
        try:
            names.add(module.get_attn_backend().get_name())
        except NotImplementedError:
            continue
    return names


def _warmup_slot_mapping(worker: "Worker", num_tokens: int) -> None:
    runner = worker.model_runner
    block_table = getattr(getattr(runner, "input_batch", None), "block_table", None)
    if block_table is None:
        return

    num_reqs = min(worker.scheduler_config.max_num_seqs, num_tokens)
    if num_reqs <= 0:
        return

    tokens_per_req = [1] * (num_reqs - 1)
    tokens_per_req.append(num_tokens - len(tokens_per_req))
    query_start_loc = np.zeros(num_reqs + 1, dtype=np.int32)
    query_start_loc[1:] = np.cumsum(tokens_per_req, dtype=np.int32)

    runner.query_start_loc.np[: num_reqs + 1] = query_start_loc
    runner.query_start_loc.copy_to_gpu(num_reqs + 1)
    runner.positions[:num_tokens].copy_(
        torch.arange(num_tokens, dtype=torch.int64, device=runner.device)
    )
    block_table.commit_block_table(num_reqs)
    block_table.compute_slot_mapping(
        num_reqs,
        runner.query_start_loc.gpu[: num_reqs + 1],
        runner.positions[:num_tokens],
    )


@instrument(span_name="MiniMax M3 MSA warmup")
def minimax_m3_msa_warmup(worker: "Worker") -> None:
    sparse_module = next(
        (
            module
            for module in worker.get_model().modules()
            if isinstance(module, MiniMaxM3SparseAttention)
        ),
        None,
    )
    if sparse_module is None:
        return
    if not _supports_minimax_m3_msa_warmup():
        return

    dense_backend_names = _dense_attention_backend_names(worker)
    dense_backend_msg = (
        f" and dense {sorted(dense_backend_names)} attention"
        if dense_backend_names
        else ""
    )
    num_tokens = (
        worker.scheduler_config.max_num_batched_tokens if dense_backend_names else 16
    )

    logger.info(
        "Warming up MiniMax M3 MSA kernels%s with %d tokens.",
        dense_backend_msg,
        num_tokens,
    )
    _warmup_slot_mapping(worker, num_tokens)

    if dense_backend_names:
        logger.info(
            "Warming up MiniMax M3 single-request prefill attention with %d tokens.",
            num_tokens,
        )
        worker.model_runner._dummy_run(
            num_tokens=num_tokens,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            include_mm_inputs=False,
            single_request_prefill=True,
        )

    # Cover sparse prefill through the normal model path. When dense attention
    # layers are present, use the scheduler max token count so their attention
    # kernels and the slot-mapping kernel are compiled before serving starts.
    worker.model_runner._dummy_run(
        num_tokens=num_tokens,
        skip_eplb=True,
        is_profile=True,
        force_attention=True,
        create_mixed_batch=True,
        include_mm_inputs=False,
    )

    if dense_backend_names:
        decode_tokens = min(worker.scheduler_config.max_num_seqs, num_tokens)
        decode_context_len = min(
            worker.scheduler_config.max_num_batched_tokens,
            worker.model_config.max_model_len,
        )
        logger.info(
            "Warming up MiniMax M3 dense decode attention with %d tokens "
            "and context length %d.",
            decode_tokens,
            decode_context_len,
        )
        worker.model_runner._dummy_run(
            num_tokens=decode_tokens,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            uniform_decode=True,
            profile_seq_lens=decode_context_len,
            include_mm_inputs=False,
        )

        logger.info(
            "Warming up MiniMax M3 single-request decode attention "
            "with context length %d.",
            decode_context_len,
        )
        worker.model_runner._dummy_run(
            num_tokens=1,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            uniform_decode=True,
            profile_seq_lens=decode_context_len,
            include_mm_inputs=False,
        )
