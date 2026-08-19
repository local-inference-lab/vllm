# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
from collections.abc import Callable
from threading import Lock

import numpy as np
import torch

import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.all2all_utils import get_ep_all2all_manager
from vllm.v1.outputs import (
    AsyncModelRunnerOutput,
    DraftTokenIds,
    LogprobsTensors,
    ModelRunnerOutput,
    PoolerOutput,
    RoutedExpertsTensors,
)
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.utils import raise_if_nan_logits


class AsyncOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampler_output: SamplerOutput,
        num_sampled_tokens: torch.Tensor,
        main_stream: torch.cuda.Stream,
        copy_stream: torch.cuda.Stream,
        check_ep_fault: bool = False,
        routed_experts: RoutedExpertsTensors | None = None,
        defer_copy_event: bool = False,
        completion_callback: Callable[[np.ndarray, np.ndarray], None] | None = None,
    ):
        # NOTE(woosuk): We must retain references to the GPU tensors,
        # as the copy operations are performed on a different CUDA stream than
        # the one where the tensors were created.
        self.model_runner_output = model_runner_output
        self.sampler_output = sampler_output
        self.num_sampled_tokens = num_sampled_tokens
        self.routed_experts = routed_experts
        self._has_fault: torch.Tensor | None = None
        self.main_stream = main_stream
        self.copy_stream = copy_stream
        self.draft_req_ids: list[str] | None = None
        self.draft_token_ids_np: np.ndarray | None = None
        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
        self.copy_event = torch.cuda.Event(blocking=True)
        self.copy_event_recorded = False

        self.completion_callback = completion_callback
        self.completion_lock = Lock() if completion_callback is not None else None
        self.num_rejected_tokens_np: np.ndarray | None = None
        with stream(copy_stream, main_stream):
            copy_stream.wait_stream(main_stream)

            self.sampled_token_ids = async_copy_to_np(sampler_output.sampled_token_ids)
            self.logprobs_tensors: LogprobsTensors | None = None
            if sampler_output.logprobs_tensors is not None:
                self.logprobs_tensors = (
                    sampler_output.logprobs_tensors.to_cpu_nonblocking()
                )
            self.num_nans: np.ndarray | None = None
            if sampler_output.num_nans is not None:
                self.num_nans = async_copy_to_np(sampler_output.num_nans)
            self.num_sampled_tokens_np = async_copy_to_np(num_sampled_tokens)
            self.routed_experts_cpu: RoutedExpertsTensors | None = None
            if routed_experts is not None:
                self.routed_experts_cpu = routed_experts.to_cpu_nonblocking()
            if completion_callback is not None:
                assert sampler_output.num_rejected is not None
                self.num_rejected_tokens_np = async_copy_to_np(
                    sampler_output.num_rejected
                )
            self.prompt_logprobs_dict = {
                k: v.to_cpu_nonblocking() if v is not None else None
                for k, v in self.model_runner_output.prompt_logprobs_dict.items()
            }
            if check_ep_fault:
                has_fault = get_ep_all2all_manager().query_fault()
                self._has_fault = has_fault.to("cpu", non_blocking=True)
            if not defer_copy_event:
                self.copy_event.record(copy_stream)
                self.copy_event_recorded = True

    def add_draft_token_ids(
        self, req_ids: list[str], draft_token_ids: torch.Tensor
    ) -> None:
        """Append draft D2H to this output's copy stream and final event."""
        assert not self.copy_event_recorded
        self.draft_req_ids = list(req_ids)

        # Draft state lives in a persistent worker buffer that the next step can
        # overwrite. Snapshot it on the main stream before handing it to the
        # asynchronous copy stream.
        draft_token_ids_snapshot = draft_token_ids.clone()
        with stream(self.copy_stream, self.main_stream):
            self.copy_stream.wait_stream(self.main_stream)
            self.draft_token_ids_np = async_copy_to_np(draft_token_ids_snapshot)
            draft_token_ids_snapshot.record_stream(self.copy_stream)
            self.copy_event.record(self.copy_stream)
            self.copy_event_recorded = True

    def _run_completion_callback(self) -> None:
        completion_lock = self.completion_lock
        if completion_lock is None:
            return
        with completion_lock:
            completion_callback = self.completion_callback
            self.completion_callback = None
            if completion_callback is None:
                return
            assert self.num_rejected_tokens_np is not None
            completion_callback(self.num_sampled_tokens_np, self.num_rejected_tokens_np)

    def resolve_completion(self) -> None:
        """Resolve accepted tokens before the next ownership update."""
        completion_lock = self.completion_lock
        if completion_lock is None:
            return
        with completion_lock:
            if self.completion_callback is None:
                return
        assert self.copy_event_recorded
        self.copy_event.synchronize()
        self._run_completion_callback()

    def get_output(self) -> ModelRunnerOutput:
        assert self.copy_event_recorded
        self.copy_event.synchronize()
        self._run_completion_callback()

        # NOTE(woosuk): The following code is to ensure compatibility with
        # the existing model runner.
        # Going forward, we should keep the data structures as NumPy arrays
        # rather than Python lists.
        sampled_token_ids: list[list[int]] = self.sampled_token_ids.tolist()
        num_sampled_tokens: list[int] = self.num_sampled_tokens_np.tolist()
        for token_ids, num_tokens in zip(sampled_token_ids, num_sampled_tokens):
            del token_ids[num_tokens:]
            for i, token_id in enumerate(token_ids):
                if token_id < 0:
                    del token_ids[i:]
                    break
        self.model_runner_output.sampled_token_ids = sampled_token_ids

        if self.draft_token_ids_np is not None:
            assert self.draft_req_ids is not None
            draft_token_ids = self.draft_token_ids_np.tolist()
            for token_ids in draft_token_ids:
                for i, token_id in enumerate(token_ids):
                    if token_id < 0:
                        del token_ids[i:]
                        break
            self.model_runner_output.draft_token_ids = DraftTokenIds(
                self.draft_req_ids, draft_token_ids
            )

        if self.num_nans is not None:
            self.model_runner_output.num_nans_in_logits = dict(
                zip(self.model_runner_output.req_ids, self.num_nans.tolist())
            )
            if envs.VLLM_RAISE_ON_LOGIT_NANS:
                raise_if_nan_logits(self.model_runner_output.num_nans_in_logits)

        if self.logprobs_tensors is not None:
            self.model_runner_output.logprobs = self.logprobs_tensors.tolists()
        self.model_runner_output.prompt_logprobs_dict = self.prompt_logprobs_dict
        if self.routed_experts_cpu is not None:
            self.model_runner_output.routed_experts = self.routed_experts_cpu.tolists()

        if self._has_fault is not None and self._has_fault.item():
            mask = get_ep_all2all_manager().query_active_mask()
            raise RuntimeError(
                "Fault detected in EP all2all communication: "
                "one or more ranks timed out during dispatch/combine. "
                f"Mask: {mask.cpu().tolist()}"
            )

        return self.model_runner_output


class AsyncPoolingOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        pooler_output: PoolerOutput,
        finished_mask: list[bool],
        main_stream: torch.cuda.Stream,
        copy_stream: torch.cuda.Stream,
    ):
        self.model_runner_output = model_runner_output
        self.pooler_output = pooler_output
        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
        self.copy_event = torch.cuda.Event(blocking=True)

        with stream(copy_stream, main_stream):
            copy_stream.wait_stream(main_stream)
            if isinstance(self.pooler_output, torch.Tensor) and all(finished_mask):
                self.pooler_output_cpu: PoolerOutput = self.pooler_output.to(
                    "cpu", non_blocking=True
                )
            else:
                outputs = (
                    self.pooler_output.unbind()
                    if isinstance(self.pooler_output, torch.Tensor)
                    else self.pooler_output
                )
                self.pooler_output_cpu = [
                    None
                    if output is None or not is_finished
                    else output.to("cpu", non_blocking=True)
                    for output, is_finished in zip(outputs, finished_mask, strict=True)
                ]
            self.copy_event.record(copy_stream)

    def get_output(self) -> ModelRunnerOutput:
        if isinstance(self.pooler_output_cpu, torch.Tensor):
            pooler_output = list(self.pooler_output_cpu.unbind(dim=0))
        else:
            pooler_output = self.pooler_output_cpu
        self.copy_event.synchronize()
        self.model_runner_output.pooler_output = pooler_output
        return self.model_runner_output


def async_copy_to_np(x: torch.Tensor) -> np.ndarray:
    return x.to("cpu", non_blocking=True).numpy()


@contextlib.contextmanager
def stream(to_stream: torch.cuda.Stream, from_stream: torch.cuda.Stream):
    """Lightweight version of torch.cuda.stream() context manager which
    avoids current_stream and device lookups.
    """
    try:
        torch.cuda.set_stream(to_stream)
        yield
    finally:
        torch.cuda.set_stream(from_stream)
