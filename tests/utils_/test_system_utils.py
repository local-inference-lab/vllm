# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing
import multiprocessing.forkserver as forkserver
import os
import tempfile
from pathlib import Path

import pytest

import vllm.envs as envs
from vllm.utils import system_utils
from vllm.utils.system_utils import (
    _maybe_force_spawn,
    ensure_cuda_clean_forkserver,
    unique_filepath,
)


def test_unique_filepath():
    temp_dir = tempfile.mkdtemp()
    path_fn = lambda i: Path(temp_dir) / f"file_{i}.txt"
    paths = set()
    for i in range(10):
        path = unique_filepath(path_fn)
        path.write_text("test")
        paths.add(path)
    assert len(paths) == 10
    assert len(list(Path(temp_dir).glob("*.txt"))) == 10


def test_numa_bind_forces_spawn(monkeypatch):
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    monkeypatch.setattr("sys.argv", ["vllm", "serve", "--numa-bind"])
    _maybe_force_spawn()
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"


def test_worker_multiproc_method_accepts_forkserver(monkeypatch):
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "forkserver")

    assert envs.environment_variables["VLLM_WORKER_MULTIPROC_METHOD"]() == "forkserver"


def test_ensure_cuda_clean_forkserver_starts_with_deduplicated_preload(
    monkeypatch,
):
    calls: dict[str, object] = {}
    monkeypatch.setattr(system_utils, "cuda_is_initialized", lambda: False)
    monkeypatch.setattr(system_utils, "xpu_is_initialized", lambda: False)
    monkeypatch.setattr(
        multiprocessing,
        "set_start_method",
        lambda method, force=False: calls.update(method=(method, force)),
    )
    monkeypatch.setattr(
        multiprocessing,
        "set_forkserver_preload",
        lambda modules: calls.update(preload=modules),
    )
    monkeypatch.setattr(
        forkserver,
        "ensure_running",
        lambda: calls.update(running=True),
    )

    ensure_cuda_clean_forkserver(
        ["vllm.worker", "vllm.executor", "vllm.worker"],
        set_start_method=True,
    )

    assert calls == {
        "method": ("forkserver", True),
        "preload": ["vllm.worker", "vllm.executor"],
        "running": True,
    }


@pytest.mark.parametrize("initialized", ["cuda", "xpu"])
def test_ensure_cuda_clean_forkserver_rejects_initialized_parent(
    monkeypatch, initialized
):
    monkeypatch.setattr(
        system_utils, "cuda_is_initialized", lambda: initialized == "cuda"
    )
    monkeypatch.setattr(
        system_utils, "xpu_is_initialized", lambda: initialized == "xpu"
    )

    with pytest.raises(RuntimeError, match="after CUDA/XPU initialization"):
        ensure_cuda_clean_forkserver(["vllm.worker"])
