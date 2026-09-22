# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from unittest.mock import Mock

import pytest

from vllm.model_executor.warmup.flashinfer_autotune_cache import (
    read_flashinfer_autotune_cache,
    save_flashinfer_autotune_cache,
)

pytestmark = pytest.mark.cpu_test


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
