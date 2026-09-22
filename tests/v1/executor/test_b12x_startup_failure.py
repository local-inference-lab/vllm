# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest

from vllm.v1.executor.abstract import Executor


@pytest.mark.parametrize("abort_fails", [False, True])
def test_failed_admission_closes_unowned_startup_executor(abort_fails):
    calls = []

    def rpc(method, **kwargs):
        calls.append(method)
        if method == "begin_b12x_preparation":
            raise ValueError("admission rejected")
        assert method == "abort_b12x_preparation" and kwargs["timeout"] == 10
        if abort_fails:
            raise RuntimeError("peer unavailable")
        return []

    executor = SimpleNamespace(
        collective_rpc=rpc, shutdown=lambda: calls.append("shutdown")
    )
    with pytest.raises(ValueError, match="admission rejected") as caught:
        Executor._run_b12x_preparation(executor, stage="weights")
    assert calls == ["begin_b12x_preparation", "abort_b12x_preparation", "shutdown"]
    if abort_fails:
        assert "peer unavailable" in caught.value.__notes__[0]
