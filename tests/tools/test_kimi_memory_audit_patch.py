# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU나 실제 신호 없이 메모리 감사 패치의 설치 계약을 검증한다."""

import importlib.util
import pickle
import signal
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2] / "tools/profiler/kimi_memory_audit_patch.py"
)
SOURCE = """class Worker:
    def init_device(self):
        if self.device_config.device_type == "cuda":
            return "ready"
"""


@pytest.fixture
def installer(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("memory_audit_patch", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = tmp_path / "worker.py"
    target.write_text(SOURCE)
    monkeypatch.setattr(module, "PATH", target)
    return module, target


def test_install_is_idempotent_and_preserves_worker_method(installer):
    module, target = installer
    assert module.main() == 0
    patched = target.read_text()
    compile(patched, str(target), "exec")
    assert 'return "ready"' in patched
    assert patched.count("self._k3_memory_audit_install()") == 1
    assert module.main() == 0
    assert target.read_text() == patched


@pytest.mark.parametrize("source", ["class Worker: pass\n", SOURCE + SOURCE])
def test_missing_or_ambiguous_anchor_never_overwrites_source(installer, source):
    module, target = installer
    target.write_text(source)
    with pytest.raises(SystemExit):
        module.main()
    assert target.read_text() == source


def test_invalid_generated_source_is_not_written(installer, monkeypatch):
    module, target = installer
    monkeypatch.setattr(module, "NEW", "invalid syntax !!!")
    with pytest.raises(SyntaxError):
        module.main()
    assert target.read_text() == SOURCE


def test_disabled_audit_does_not_touch_cuda_or_signal(installer, monkeypatch):
    module, target = installer
    module.main()
    fake_torch = ModuleType("torch")
    fake_cuda = Mock()
    monkeypatch.setattr(fake_torch, "cuda", fake_cuda, raising=False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    install_signal = Mock()
    monkeypatch.setattr(signal, "signal", install_signal)
    monkeypatch.delenv("K3_MEMORY_AUDIT", raising=False)
    namespace = {"logger": Mock()}
    exec(compile(target.read_text(), str(target), "exec"), namespace)
    namespace["Worker"]()._k3_memory_audit_install()
    assert fake_cuda.mock_calls == []
    install_signal.assert_not_called()


def test_enabled_audit_writes_mocked_rank_snapshot(installer, monkeypatch, tmp_path):
    module, target = installer
    module.main()
    fake_torch = ModuleType("torch")
    fake_cuda = Mock()
    fake_cuda.mem_get_info.return_value = (10, 20)
    fake_cuda.memory_stats.return_value = {"allocated": 7}
    fake_cuda.memory._snapshot.return_value = {"segments": []}
    monkeypatch.setattr(fake_torch, "cuda", fake_cuda, raising=False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    install_signal = Mock()
    monkeypatch.setattr(signal, "signal", install_signal)
    monkeypatch.setenv("K3_MEMORY_AUDIT", "1")
    monkeypatch.setenv("K3_MEMORY_AUDIT_MAX_ENTRIES", "32")
    monkeypatch.setenv("K3_MEMORY_AUDIT_DIR", str(tmp_path / "snapshots"))
    namespace = {"logger": Mock()}
    exec(compile(target.read_text(), str(target), "exec"), namespace)
    worker = namespace["Worker"]()
    worker.rank = 3
    worker.device_config = SimpleNamespace(device_type="cuda")
    assert worker.init_device() == "ready"
    fake_cuda.memory._record_memory_history.assert_called_once_with(max_entries=32)
    install_signal.assert_called_once()
    signum, handler = install_signal.call_args.args
    assert signum == signal.SIGUSR1
    handler(None, None)
    files = list((tmp_path / "snapshots").glob("rank3-*.pickle"))
    assert len(files) == 1
    with files[0].open("rb") as stream:
        payload = pickle.load(stream)
    assert payload["rank"] == 3
    assert payload["mem_get_info"] == {"free": 10, "total": 20}
    assert payload["snapshot"] == {"segments": []}
