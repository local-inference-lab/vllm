"""Read a bare Escape key while startup owns the foreground terminal."""
from __future__ import annotations

import os
import select
import termios
import threading
import time


def _open_terminal():
    fd = os.open('/dev/tty', os.O_RDWR | os.O_NONBLOCK | os.O_NOCTTY)
    try:
        if os.tcgetpgrp(fd) != os.getpgrp():
            os.close(fd)
            return None
        return fd
    except BaseException:
        os.close(fd)
        raise


class EscapeKey:
    def __init__(self, cancel):
        self._cancel = cancel
        self._fd = None
        self._settings = None
        self._thread = None
        self._closed = threading.Event()

    @property
    def active(self):
        return self._fd is not None and not self._closed.is_set()

    def __enter__(self):
        try:
            self._fd = _open_terminal()
            if self._fd is None:
                return self
            self._settings = termios.tcgetattr(self._fd)
            settings = termios.tcgetattr(self._fd)
            settings[3] &= ~(termios.ICANON | termios.ECHO)
            settings[6][termios.VMIN] = 0
            settings[6][termios.VTIME] = 0
            termios.tcsetattr(self._fd, termios.TCSANOW, settings)
        except (OSError, termios.error):
            self._restore()
            return self
        self._thread = threading.Thread(target=self._read, name='b12x-escape', daemon=True)
        try:
            self._thread.start()
        except BaseException:
            self._restore()
            raise
        return self

    def _read(self):
        deadline = None
        try:
            while not self._closed.is_set():
                now = time.monotonic()
                if deadline is not None and now >= deadline:
                    self._cancel()
                    return
                timeout = 0.05 if deadline is None else max(0, deadline - now)
                if not select.select([self._fd], [], [], timeout)[0]:
                    continue
                data = os.read(self._fd, 64)
                if not data:
                    return
                for byte in data:
                    if byte == 27:
                        deadline = time.monotonic() + 0.05
                    elif deadline is not None:
                        # Arrow keys and Alt sequences begin with Escape.
                        deadline = None
        except (OSError, termios.error):
            pass
        finally:
            self._closed.set()
            self._restore()

    def _restore(self):
        fd, self._fd = self._fd, None
        if fd is not None:
            try:
                if self._settings is not None:
                    termios.tcsetattr(fd, termios.TCSANOW, self._settings)
            except (OSError, termios.error):
                pass
            finally:
                os.close(fd)

    def __exit__(self, *_args):
        self._closed.set()
        if self._thread is not None:
            self._thread.join()
        self._restore()
