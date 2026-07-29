# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.cli import main as cli_main
from vllm.utils import system_utils


def test_serve_bootstraps_forkserver_before_command_imports(monkeypatch):
    calls: list[tuple[list[str], bool]] = []
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "forkserver")
    monkeypatch.setattr("sys.argv", ["vllm", "serve", "model"])
    monkeypatch.setattr(
        system_utils,
        "ensure_cuda_clean_forkserver",
        lambda modules, *, set_start_method=False: calls.append(
            (modules, set_start_method)
        ),
    )

    cli_main._bootstrap_cuda_clean_forkserver()

    assert calls == [
        (
            [
                "vllm.v1.engine.async_llm",
                "vllm.v1.executor.multiproc_executor",
                "vllm.v1.worker.gpu_worker",
            ],
            True,
        )
    ]


def test_non_serve_command_does_not_bootstrap_forkserver(monkeypatch):
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "forkserver")
    monkeypatch.setattr("sys.argv", ["vllm", "bench"])
    monkeypatch.setattr(
        system_utils,
        "ensure_cuda_clean_forkserver",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("non-serve command must not start a forkserver")
        ),
    )

    cli_main._bootstrap_cuda_clean_forkserver()
