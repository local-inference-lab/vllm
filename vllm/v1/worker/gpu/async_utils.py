# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib

import numpy as np
import torch

from vllm.v1.outputs import (
    AsyncModelRunnerOutput,
    DraftTokenIds,
    LogprobsTensors,
    ModelRunnerOutput,
)
from vllm.v1.worker.gpu.sample.output import SamplerOutput


class AsyncOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampler_output: SamplerOutput,
        num_sampled_tokens: torch.Tensor,
        main_stream: torch.cuda.Stream,
        copy_stream: torch.cuda.Stream,
        defer_copy_event: bool = False,
    ):
        # NOTE(woosuk): We must retain references to the GPU tensors,
        # as the copy operations are performed on a different CUDA stream than
        # the one where the tensors were created.
        self.model_runner_output = model_runner_output
        self.sampler_output = sampler_output
        self.num_sampled_tokens = num_sampled_tokens
        self.main_stream = main_stream
        self.copy_stream = copy_stream
        self.draft_req_ids: list[str] | None = None
        self.draft_token_ids_np: np.ndarray | None = None
        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
        self.copy_event = torch.cuda.Event(blocking=True)
        self.copy_event_recorded = False

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
            self.prompt_logprobs_dict = {
                k: v.to_cpu_nonblocking() if v is not None else None
                for k, v in self.model_runner_output.prompt_logprobs_dict.items()
            }
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

    def get_output(self) -> ModelRunnerOutput:
        assert self.copy_event_recorded
        self.copy_event.synchronize()

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

        if self.logprobs_tensors is not None:
            self.model_runner_output.logprobs = self.logprobs_tensors.tolists()
        self.model_runner_output.prompt_logprobs_dict = self.prompt_logprobs_dict
        return self.model_runner_output


class AsyncPoolingOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        pooler_output: torch.Tensor,
        is_valid: torch.Tensor | None,
        main_stream: torch.cuda.Stream,
        copy_stream: torch.cuda.Stream,
    ):
        self.model_runner_output = model_runner_output
        self.pooler_output = pooler_output
        self.is_valid = is_valid
        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
        self.copy_event = torch.cuda.Event(blocking=True)

        with stream(copy_stream, main_stream):
            copy_stream.wait_stream(main_stream)
            self.pooler_output_cpu = self.pooler_output.to("cpu", non_blocking=True)
            if self.is_valid is not None:
                self.is_valid_cpu = self.is_valid.to("cpu", non_blocking=True)
            else:
                self.is_valid_cpu = None
            self.copy_event.record(copy_stream)

    def get_output(self) -> ModelRunnerOutput:
        pooler_output = list(self.pooler_output_cpu.unbind(dim=0))
        self.copy_event.synchronize()
        if self.is_valid_cpu is not None:
            is_valid_cpu = self.is_valid_cpu.tolist()
            for i, is_valid in enumerate(is_valid_cpu):
                if not is_valid:
                    pooler_output[i] = None
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
