"""Terminal output ownership and the native startup output transport."""
from __future__ import annotations

import ctypes
import io
import os
import select
import subprocess
import sys
import termios
from pathlib import Path

import pytest

from vllm.v1.executor._b12x_output import PreparationOutput


def test_terminal_capture_includes_native_child_and_partial_output():
    master, slave = os.openpty()
    lines = []
    try:
        with PreparationOutput(lines.append, descriptors=(slave,)) as output:
            assert output.stream.isatty()
            assert not os.isatty(slave)
            os.write(slave, b'first\n\n\xce')
            os.write(slave, b'\xb1\n')
            subprocess.run([
                sys.executable, '-c',
                f'import os; os.write({slave}, b"native child\\n")',
            ], pass_fds=(slave,), check=True)
            os.write(slave, b'partial')
        assert os.isatty(slave)
        assert lines == ['first', '', 'α', 'native child', 'partial']
        os.write(slave, b'normal output\n')
        assert os.read(master, 64) == b'normal output\r\n'
    finally:
        os.close(slave)
        os.close(master)


def test_capture_restores_terminal_after_failure():
    master, slave = os.openpty()
    lines = []
    try:
        with pytest.raises(ValueError, match='preparation failed'):
            with PreparationOutput(lines.append, descriptors=(slave,)):
                os.write(slave, b'diagnostic without newline')
                raise ValueError('preparation failed')
        assert os.isatty(slave)
        assert lines == ['diagnostic without newline']
    finally:
        os.close(slave)
        os.close(master)


def test_capture_leaves_pipe_destinations_alone():
    reader, writer = os.pipe()
    try:
        with PreparationOutput(lambda line: pytest.fail(line), descriptors=(writer,)) as output:
            assert output.stream is None
            os.write(writer, b'file-directed log\n')
        assert os.read(reader, 64) == b'file-directed log\n'
    finally:
        os.close(writer)
        os.close(reader)


def _startup_with_native_output():
    """Run the actual executor and worker transport with a host-only workload."""
    import logging
    import multiprocessing
    import threading
    import time
    from types import SimpleNamespace

    from b12x.preparation import PreparationProgress
    from vllm.v1.executor.abstract import Executor
    from vllm.v1.worker.worker_base import WorkerBase

    parent, child = multiprocessing.Pipe()

    def worker():
        parent.close()
        handler = logging.StreamHandler(sys.stderr)
        def advance(*, cancel_tuning=False):
            handler.emit(logging.LogRecord('worker', logging.INFO, '', 0, 'worker-bound-handler', (), None))
            os.write(2, b'\n\nworker-native-write\n')
            subprocess.run([
                sys.executable, '-c',
                'import os; os.write(1, b"compiler-child-output\\n" + b"long-log-word " * 28 + b"\\n")',
            ], check=True)
            os.write(1, b'worker-final-fragment')
            time.sleep(0.25)
            return dict(global_rank=0, done=True, error=None, progress=PreparationProgress(
                False, False, (), True, phase='ready', completed_requests=1, total_requests=1,
            ))
        try:
            result = WorkerBase.run_b12x_preparation(
                SimpleNamespace(rank=0, advance_b12x_preparation=advance), **child.recv(),
            )
            child.send(result)
        finally:
            child.close()

    process = multiprocessing.get_context('fork').Process(target=worker)
    process.start()
    child.close()
    handler = logging.StreamHandler(sys.stderr)
    def rpc(method, kwargs=None):
        if method == 'begin_b12x_preparation':
            return [dict(native=True, global_rank=0, done=False)]
        assert method == 'run_b12x_preparation'
        handler.emit(logging.LogRecord('engine', logging.INFO, '', 0, 'engine-bound-handler', (), None))
        os.write(2, b'\nengine-native-write\n')
        parent.send(kwargs)
        return [parent.recv()]
    try:
        Executor._run_b12x_preparation(SimpleNamespace(
            collective_rpc=rpc, _b12x_autotuning_cancel=threading.Event(),
        ), stage='weights')
    finally:
        parent.close()
        process.join(5)
        if process.is_alive():
            process.terminate()
            process.join()
        assert process.exitcode == 0


def _terminal_text(data, rows, columns):
    try:
        lib = ctypes.CDLL('libvterm.so.0')
    except OSError:
        pytest.skip('terminal replay requires libvterm')
    pointer = ctypes.c_void_p
    class Rect(ctypes.Structure):
        _fields_ = [(name, ctypes.c_int) for name in ('start_row', 'end_row', 'start_col', 'end_col')]
    lib.vterm_new.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.vterm_new.restype = pointer
    lib.vterm_set_utf8.argtypes = [pointer, ctypes.c_int]
    lib.vterm_obtain_screen.argtypes = [pointer]
    lib.vterm_obtain_screen.restype = pointer
    lib.vterm_screen_reset.argtypes = [pointer, ctypes.c_int]
    lib.vterm_input_write.argtypes = [pointer, ctypes.c_char_p, ctypes.c_size_t]
    lib.vterm_input_write.restype = ctypes.c_size_t
    lib.vterm_screen_get_text.argtypes = [pointer, pointer, ctypes.c_size_t, Rect]
    lib.vterm_screen_get_text.restype = ctypes.c_size_t
    lib.vterm_free.argtypes = [pointer]
    terminal = lib.vterm_new(rows, columns)
    try:
        lib.vterm_set_utf8(terminal, 1)
        screen = lib.vterm_obtain_screen(terminal)
        lib.vterm_screen_reset(screen, 1)
        lib.vterm_input_write(terminal, data, len(data))
        buffer = ctypes.create_string_buffer(rows * columns * 4)
        count = lib.vterm_screen_get_text(screen, buffer, len(buffer), Rect(0, rows, 0, columns))
        return buffer.raw[:count].decode()
    finally:
        lib.vterm_free(terminal)


@pytest.mark.parametrize('columns', [84, 133])
def test_executor_serializes_bound_handlers_native_writes_and_child_output(columns):
    master, slave = os.openpty()
    rows = 36
    termios.tcsetwinsize(slave, (rows, columns))
    os.set_blocking(master, False)
    env = dict(os.environ, TERM='xterm-256color', PYTHONDONTWRITEBYTECODE='1')
    for key in ('COLUMNS', 'LINES'):
        env.pop(key, None)
    output = bytearray()
    try:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), '--startup-replay'],
            stdin=slave, stdout=slave, stderr=slave, env=env,
        )
        while process.poll() is None:
            if select.select([master], [], [], 0.1)[0]:
                output.extend(os.read(master, 65536))
        while select.select([master], [], [], 0)[0]:
            output.extend(os.read(master, 65536))
        assert process.returncode == 0, output.decode(errors='replace')
        screen = _terminal_text(bytes(output), rows, columns)
        assert screen.count('b12x / one-time kernel autotuning') == 1, screen
        for message in (
            'engine-bound-handler', 'engine-native-write', 'worker-bound-handler',
            'worker-native-write', 'compiler-child-output', 'worker-final-fragment',
        ):
            assert screen.count(message) == 1, screen
        assert screen.count('long-log-word') == 28, screen
        top = next(index for index, line in enumerate(screen.splitlines()) if line.startswith('┌'))
        panel = screen.splitlines()[top:top + 10]
        assert len(panel) == 10 and panel[-1].startswith('└'), screen
        assert all(line.startswith('│') and line.endswith('│') for line in panel[1:-1]), screen
    finally:
        if 'process' in locals() and process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        os.close(slave)
        os.close(master)


if __name__ == '__main__' and '--startup-replay' in sys.argv:
    _startup_with_native_output()
