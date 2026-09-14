"""Forward terminal output through the startup coordinator's live renderer."""
from __future__ import annotations

import codecs
import os
import select
import sys
import threading


class PreparationOutput:
    """Capture terminal descriptors, including native writes and child output.

    File and pipe destinations are left alone. ``stream`` is an independent
    terminal descriptor for the renderer, so its output cannot be recaptured.
    Compiler children must be drained before stopping capture.
    """

    def __init__(self, emit, *, enabled=True, descriptors=(1, 2)):
        self._emit = emit
        self._enabled = enabled
        self._descriptors = descriptors
        self._saved = {}
        self._reader = None
        self._thread = None
        self._stopped = threading.Event()
        self._error = None
        self.stream = None

    def _flush(self):
        if not self._saved:
            return
        for stream in (sys.stdout, sys.stderr):
            try:
                fd = stream.fileno()
            except (AttributeError, OSError, ValueError):
                continue
            if fd in self._saved:
                stream.flush()

    def start(self):
        if not self._enabled:
            return self
        try:
            for fd in self._descriptors:
                if os.isatty(fd):
                    self._saved[fd] = os.dup(fd)
            if not self._saved:
                return self
            self._flush()
            terminal = self._saved.get(2, next(reversed(self._saved.values())))
            self.stream = os.fdopen(os.dup(terminal), 'w', buffering=1, encoding='utf-8')
            self._reader, writer = os.pipe()
            try:
                for fd in self._saved:
                    os.dup2(writer, fd)
            finally:
                os.close(writer)
            self._thread = threading.Thread(target=self._read, name='b12x-output', daemon=True)
            self._thread.start()
        except BaseException:
            self.close()
            raise
        return self

    def _read(self):
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        pending = ''
        try:
            while True:
                if not select.select([self._reader], [], [], 0.05)[0]:
                    if self._stopped.is_set():
                        break
                    continue
                data = os.read(self._reader, 65536)
                if not data:
                    break
                pending += decoder.decode(data)
                while '\n' in pending:
                    line, pending = pending.split('\n', 1)
                    self._emit(line.rstrip('\r'))
            pending += decoder.decode(b'', final=True)
            if pending:
                self._emit(pending)
        except BaseException as error:
            self._error = error
        finally:
            os.close(self._reader)
            self._reader = None

    def stop(self):
        try:
            self._flush()
        finally:
            saved, self._saved = self._saved, {}
            for fd, original in saved.items():
                try:
                    os.dup2(original, fd)
                finally:
                    os.close(original)
            self._stopped.set()
            if self._thread is not None:
                if self._thread.ident is not None:
                    self._thread.join()
                self._thread = None
            if self._reader is not None:
                os.close(self._reader)
                self._reader = None
        if self._error is not None:
            error, self._error = self._error, None
            raise error

    def close(self):
        try:
            self.stop()
        finally:
            if self.stream is not None:
                self.stream.close()
                self.stream = None

    def __enter__(self):
        return self.start()

    def __exit__(self, kind, value, traceback):
        try:
            self.close()
        except BaseException as error:
            if value is None:
                raise
            value.add_note(f'startup output cleanup failed: {error!r}')
