"""Foreground terminal input and restoration without a model or CUDA context."""
import os
import pty
import termios
import threading
import time
from types import SimpleNamespace

import pytest

from vllm.v1.executor import _b12x_terminal as terminal
from vllm.v1.executor.abstract import Executor


@pytest.fixture
def tty(monkeypatch):
    master, slave = pty.openpty()
    settings = termios.tcgetattr(slave)
    monkeypatch.setattr(terminal, '_open_terminal', lambda: os.dup(slave))
    try:
        yield master, slave, settings
    finally:
        os.close(master)
        os.close(slave)


def test_escape_without_enter_cancels_once_and_restores_terminal(tty):
    master, slave, settings = tty
    event, calls = threading.Event(), []
    def cancel():
        calls.append('cancel')
        event.set()
    with terminal.EscapeKey(cancel) as keyboard:
        assert keyboard.active
        live = termios.tcgetattr(slave)
        assert not live[3] & (termios.ICANON | termios.ECHO)
        assert live[3] & termios.ISIG == settings[3] & termios.ISIG
        os.write(master, b'\x1b')
        assert event.wait(2)
    assert calls == ['cancel']
    assert termios.tcgetattr(slave) == settings


def test_arrow_and_alt_sequences_do_not_cancel(tty):
    master, slave, settings = tty
    event = threading.Event()
    with terminal.EscapeKey(event.set):
        os.write(master, b'\x1b[A\x1bOB\x1bx')
        assert not event.wait(0.1)
        os.write(master, b'\x1b')
        time.sleep(0.01)
        os.write(master, b'[C')
        assert not event.wait(0.1)
    assert termios.tcgetattr(slave) == settings


def test_exception_restores_terminal(tty):
    _, slave, settings = tty
    with pytest.raises(ValueError, match='startup failed'):
        with terminal.EscapeKey(lambda: None):
            raise ValueError('startup failed')
    assert termios.tcgetattr(slave) == settings


def test_no_controlling_terminal_is_optional(monkeypatch):
    def unavailable():
        raise OSError('no controlling terminal')
    monkeypatch.setattr(terminal, '_open_terminal', unavailable)
    with terminal.EscapeKey(lambda: pytest.fail('unexpected cancellation')) as keyboard:
        assert not keyboard.active


def _executor(monkeypatch):
    import vllm.utils.b12x as b12x
    import vllm.platforms as platforms
    monkeypatch.setattr(b12x, 'has_b12x', lambda: True)
    monkeypatch.setattr(platforms, 'current_platform', SimpleNamespace(
        is_cuda=lambda: True, is_device_capability_family=lambda family: family == 120,
    ))
    cancel = threading.Event()
    return SimpleNamespace(
        vllm_config=SimpleNamespace(kernel_config=SimpleNamespace(enable_b12x_autotune=True)),
        cancel_b12x_autotuning=cancel.set, _b12x_autotuning_cancel=cancel,
    )


def test_escape_between_stages_stays_cancelled(tty, monkeypatch):
    master, slave, settings = tty
    monkeypatch.delenv('B12X_AUTOTUNE', raising=False)
    executor = _executor(monkeypatch)
    with Executor.b12x_warmup_control(executor):
        assert executor._b12x_keyboard.active
        assert not executor._b12x_autotuning_cancel.is_set()
        os.write(master, b'\x1b')
        assert executor._b12x_autotuning_cancel.wait(2)
        assert executor._b12x_autotuning_cancel.is_set()
    assert executor._b12x_keyboard is None
    assert termios.tcgetattr(slave) == settings


@pytest.mark.parametrize('disable', ['environment', 'configuration'])
def test_disabled_warmup_skips_keyboard(monkeypatch, disable):
    executor = _executor(monkeypatch)
    if disable == 'environment':
        monkeypatch.setenv('B12X_AUTOTUNE', '0')
    else:
        monkeypatch.delenv('B12X_AUTOTUNE', raising=False)
        executor.vllm_config.kernel_config.enable_b12x_autotune = False
    monkeypatch.setattr(terminal, '_open_terminal', lambda: pytest.fail('opened terminal'))
    with Executor.b12x_warmup_control(executor):
        assert executor._b12x_autotuning_cancel.is_set()
