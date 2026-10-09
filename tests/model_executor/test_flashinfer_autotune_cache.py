# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.model_executor.warmup.flashinfer_autotune_cache import (
    read_flashinfer_autotune_cache,
    resolve_flashinfer_autotune_file,
    save_flashinfer_autotune_cache,
)

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("override", [False, True])
def test_autotune_revision_isolates_existing_cache(tmp_path, monkeypatch, override):
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    if override:
        root = tmp_path / "configured"
        monkeypatch.setenv("VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR", str(root))
    else:
        monkeypatch.delenv("VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR", raising=False)
        workspace = tmp_path / "flashinfer" / "build" / "device"
        monkeypatch.setitem(
            sys.modules,
            "flashinfer.jit",
            SimpleNamespace(env=SimpleNamespace(FLASHINFER_WORKSPACE_DIR=workspace)),
        )
        root = tmp_path / "flashinfer_autotune_cache" / "build" / "device"

    runner = Mock()
    runner.vllm_config.compute_hash.return_value = "model-config"
    previous_hash = hashlib.sha256(b"model-config").hexdigest()
    previous = root / previous_hash / "autotune_configs.json"
    previous.parent.mkdir(parents=True)
    contents = b'{"configs": []}'
    previous.write_bytes(contents)

    path = resolve_flashinfer_autotune_file(runner)
    assert path.parent.parent == root
    assert read_flashinfer_autotune_cache(path) is None
    tuner = Mock()
    tuner.save_configs.side_effect = lambda output: Path(output).write_bytes(contents)
    save_flashinfer_autotune_cache(path, tuner)
    assert read_flashinfer_autotune_cache(resolve_flashinfer_autotune_file(runner)) == (
        contents
    )
    assert previous.read_bytes() == contents


@pytest.mark.parametrize("contents", [None, b"", b'{"version":', b"\xff"])
def test_interrupted_autotune_cache_is_a_miss(tmp_path, contents):
    path = tmp_path / "autotune_configs.json"
    if contents is not None:
        path.write_bytes(contents)
    assert read_flashinfer_autotune_cache(path) is None


def test_autotune_cache_preserves_serialized_bytes(tmp_path):
    path = tmp_path / "autotune_configs.json"
    contents = b'{"version": 1, "configs": []}\n'
    tuner = Mock()
    tuner.save_configs.side_effect = lambda output: Path(output).write_bytes(contents)
    save_flashinfer_autotune_cache(path, tuner)
    assert read_flashinfer_autotune_cache(path) == contents
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("serialization_failure", [False, True])
def test_failed_autotune_save_preserves_existing_cache(tmp_path, serialization_failure):
    path = tmp_path / "autotune_configs.json"
    contents = b'{"version": 1, "configs": []}\n'
    path.write_bytes(contents)

    def save(output):
        Path(output).write_bytes(b'{"version":')
        if serialization_failure:
            raise RuntimeError("Interrupted tuner serialization")

    tuner = Mock()
    tuner.save_configs.side_effect = save
    with pytest.raises(RuntimeError if serialization_failure else ValueError):
        save_flashinfer_autotune_cache(path, tuner)
    assert read_flashinfer_autotune_cache(path) == contents
    assert list(tmp_path.iterdir()) == [path]
