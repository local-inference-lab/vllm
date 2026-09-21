# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A worker owns prepared resources before EngineCore initialization completes."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.v1.worker import gpu_worker


@pytest.mark.parametrize("error", [RuntimeError("warmup"), KeyboardInterrupt()])
def test_warmup_failure_releases_worker_and_preserves_error(monkeypatch, error):
    monkeypatch.setattr(gpu_worker, "set_current_vllm_config", lambda _: nullcontext())
    worker = SimpleNamespace(
        vllm_config=None,
        _compile_or_warm_up_model_after_preparation=Mock(side_effect=error),
        shutdown=Mock(),
    )
    with pytest.raises(type(error)) as caught:
        gpu_worker.Worker.compile_or_warm_up_model(worker)
    assert caught.value is error
    worker.shutdown.assert_called_once()


def test_successful_warmup_keeps_worker_alive(monkeypatch):
    monkeypatch.setattr(gpu_worker, "set_current_vllm_config", lambda _: nullcontext())
    worker = SimpleNamespace(
        vllm_config=None,
        _compile_or_warm_up_model_after_preparation=Mock(return_value="ready"),
        shutdown=Mock(),
    )
    assert gpu_worker.Worker.compile_or_warm_up_model(worker) == "ready"
    worker.shutdown.assert_not_called()
