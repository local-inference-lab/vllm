# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A queued prefill-flag upload must carry the flags of the step that queued it.

``Qwen4ExpModelState`` and ``Glm5NextModelState`` rewrite their prefill-flag
host buffer in every ``prepare_attn`` and upload it with the non-blocking
``CpuGpuBuffer.copy_to_gpu``. Under async scheduling the host can prepare step
N+1 while step N's upload is still queued behind earlier GPU work (for example
the MTP drafter), so that upload must not read the host buffer when it runs.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.glm5next.model_state import Glm5NextModelState
from vllm.models.qwen4_exp.nvidia.model_state import Qwen4ExpModelState
from vllm.platforms import current_platform
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

MAX_REQS = 8
# GPU cycles to hold the stream; far longer than the host work queued behind it.
STALL_CYCLES = 200_000_000
STEP_N = [True, False, True, False]
STEP_N1 = [False, True, False, True]


def _build_state(
    state_cls: type[MambaHybridModelState], monkeypatch: pytest.MonkeyPatch
) -> MambaHybridModelState:
    """Run the model state's own ``__init__`` over a stubbed base class."""

    def base_init(self, vllm_config, model, encoder_cache, device):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.max_num_reqs = MAX_REQS
        self.device = device

    monkeypatch.setattr(MambaHybridModelState, "__init__", base_init)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(ple_layer_ids=[])),
        kernel_config=SimpleNamespace(linear_backend=None, moe_backend=None),
    )
    model = SimpleNamespace(modules=lambda: iter(()))
    return state_cls(vllm_config, model, None, torch.device("cuda"))


def _upload_two_steps(flags: CpuGpuBuffer) -> tuple[list[bool], list[bool], bool]:
    """Queue step N's and step N+1's uploads behind a stalled stream."""
    torch.cuda._sleep(STALL_CYCLES)
    flags.np[: len(STEP_N)] = STEP_N
    # Snapshot in stream order: what step N's forward would read.
    seen_n = flags.copy_to_gpu(len(STEP_N)).clone()
    flags.np[: len(STEP_N1)] = STEP_N1
    seen_n1 = flags.copy_to_gpu(len(STEP_N1)).clone()
    still_stalled = not torch.cuda.current_stream().query()
    torch.cuda.synchronize()
    return seen_n.tolist(), seen_n1.tolist(), still_stalled


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.skipif(not PIN_MEMORY, reason="Requires pinned host memory")
@pytest.mark.parametrize(
    ("state_cls", "buffer_name"),
    [
        pytest.param(Qwen4ExpModelState, "qsa_is_prefilling", id="qwen4_exp"),
        pytest.param(Glm5NextModelState, "selector_is_prefilling", id="glm5next"),
    ],
)
def test_queued_prefill_flag_upload_ignores_next_step_rewrite(
    monkeypatch: pytest.MonkeyPatch,
    state_cls: type[MambaHybridModelState],
    buffer_name: str,
) -> None:
    flags = getattr(_build_state(state_cls, monkeypatch), buffer_name)

    # Warm-up passes leave one cached pinned staging block and one device block
    # per queued upload: a fresh allocation may wait for queued GPU work and
    # end the stall before the host rewrite.
    for _ in range(2):
        _upload_two_steps(flags)
    seen_n, seen_n1, still_stalled = _upload_two_steps(flags)

    assert still_stalled, "stream drained before the rewrite; raise STALL_CYCLES"
    assert seen_n == STEP_N, "step N's queued upload read step N+1's host flags"
    assert seen_n1 == STEP_N1
