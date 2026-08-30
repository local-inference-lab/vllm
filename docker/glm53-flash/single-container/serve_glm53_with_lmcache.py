#!/opt/venv/bin/python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""PID 1 supervisor for one-container GLM-5.3 vLLM and LMCache serving."""

from __future__ import annotations

import contextlib
import http.client
import math
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import regex as re

LMCACHE_EXECUTABLE = "/opt/venv/bin/lmcache"
VLLM_EXECUTABLE = "/opt/venv/bin/vllm"
PREFIX = "GLM53_LMCACHE_"
VLLM_PREFIX = "GLM53_VLLM_"
INTEGER = re.compile(r"[0-9]+")
IMMUTABLE_PROVENANCE_ENV = frozenset(
    {
        "GLM53_LMCACHE_ALIGNMENT_ABI_BASE_IMAGE_ID",
        "GLM53_LMCACHE_DCP_BASE_IMAGE_ID",
        "GLM53_LMCACHE_REGISTRATION_BASE_IMAGE_ID",
    }
)
IMMUTABLE_BUILD_ENV = frozenset({"GLM53_VLLM_COMMIT"})


@dataclass(frozen=True)
class Config:
    host: str
    port: int
    http_host: str
    http_port: int
    l1_size_gb: float
    l1_init_size_gb: int
    max_gpu_workers: int
    max_cpu_workers: int
    chunk_size: int
    l1_align_bytes: int
    eviction_trigger: float
    eviction_ratio: float
    eviction_policy: str
    transfer_mode: str
    startup_timeout_seconds: float
    health_poll_seconds: float
    shutdown_grace_seconds: float
    vllm_health_url: str
    vllm_health_host: str
    vllm_health_port: int
    vllm_health_path: str
    vllm_startup_timeout_seconds: float
    vllm_health_poll_seconds: float
    broker_dir: Path
    lmcache_executable: str
    vllm_executable: str


DEFAULTS = {
    "GLM53_LMCACHE_HOST": "127.0.0.1",
    "GLM53_LMCACHE_PORT": "5555",
    "GLM53_LMCACHE_HTTP_HOST": "127.0.0.1",
    "GLM53_LMCACHE_HTTP_PORT": "8080",
    "GLM53_LMCACHE_L1_SIZE_GB": "48",
    "GLM53_LMCACHE_L1_INIT_SIZE_GB": "20",
    "GLM53_LMCACHE_MAX_GPU_WORKERS": "4",
    "GLM53_LMCACHE_MAX_CPU_WORKERS": "8",
    "GLM53_LMCACHE_CHUNK_SIZE": "9216",
    "GLM53_LMCACHE_L1_ALIGN_BYTES": "16384",
    "GLM53_LMCACHE_EVICTION_TRIGGER": "0.85",
    "GLM53_LMCACHE_EVICTION_RATIO": "0.10",
    "GLM53_LMCACHE_EVICTION_POLICY": "LRU",
    "GLM53_LMCACHE_TRANSFER_MODE": "lmcache_driven",
    "GLM53_LMCACHE_STARTUP_TIMEOUT_SECONDS": "600",
    "GLM53_LMCACHE_HEALTH_POLL_SECONDS": "0.5",
    "GLM53_LMCACHE_SHUTDOWN_GRACE_SECONDS": "30",
    "GLM53_LMCACHE_CUMEM_BROKER_DIR": "/run/lmcache-cumem",
    "GLM53_LMCACHE_EXECUTABLE": LMCACHE_EXECUTABLE,
    "GLM53_VLLM_EXECUTABLE": VLLM_EXECUTABLE,
    "GLM53_VLLM_HEALTH_URL": "http://127.0.0.1:8000/health",
    "GLM53_VLLM_STARTUP_TIMEOUT_SECONDS": "3600",
    "GLM53_VLLM_HEALTH_POLL_SECONDS": "2",
}
HEALTHCHECK_ONLY_ENV = frozenset({"GLM53_VLLM_HEALTH_TIMEOUT_SECONDS"})


def _value(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name, DEFAULTS[name])
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _integer(
    environ: Mapping[str, str],
    name: str,
    *,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    raw = _value(environ, name)
    if INTEGER.fullmatch(raw) is None:
        raise ValueError(f"{name} must be a base-10 integer, got {raw!r}")
    value = int(raw)
    if value < minimum or (maximum is not None and value > maximum):
        expected = f"{minimum}..{maximum}" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name} must be {expected}, got {value}")
    return value


def _number(
    environ: Mapping[str, str],
    name: str,
    *,
    minimum_exclusive: float,
    maximum_inclusive: float | None = None,
) -> float:
    raw = _value(environ, name)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric, got {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {raw!r}")
    if value <= minimum_exclusive:
        raise ValueError(f"{name} must be > {minimum_exclusive}, got {value}")
    if maximum_inclusive is not None and value > maximum_inclusive:
        raise ValueError(f"{name} must be <= {maximum_inclusive}, got {value}")
    return value


def _host(environ: Mapping[str, str], name: str) -> str:
    value = _value(environ, name)
    if any(character.isspace() or ord(character) < 32 for character in value):
        raise ValueError(f"{name} must be a host without whitespace")
    return value


def _choice(environ: Mapping[str, str], name: str, choices: set[str]) -> str:
    value = _value(environ, name)
    if value not in choices:
        raise ValueError(f"{name} must be one of {sorted(choices)}, got {value!r}")
    return value


def _executable(environ: Mapping[str, str], name: str) -> str:
    value = _value(environ, name)
    if not os.path.isabs(value):
        raise ValueError(f"{name} must be an absolute path, got {value!r}")
    return value


def _http_url(environ: Mapping[str, str], name: str) -> tuple[str, str, int, str]:
    raw = _value(environ, name)
    if any(character.isspace() or ord(character) < 32 for character in raw):
        raise ValueError(f"{name} must not contain whitespace or control characters")
    try:
        parsed = urlsplit(raw)
        host = parsed.hostname
        parsed_port = parsed.port
        port = 80 if parsed_port is None else parsed_port
    except (UnicodeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid HTTP URL, got {raw!r}") from exc
    if (
        parsed.scheme != "http"
        or not host
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not parsed.path
        or not parsed.path.startswith("/")
        or port < 1
        or port > 65535
    ):
        raise ValueError(
            f"{name} must be an HTTP URL with a valid host, port, and path, got {raw!r}"
        )
    return raw, host, port, parsed.path


def load_config(environ: Mapping[str, str]) -> Config:
    """Load and strictly validate the complete supervisor environment."""
    unknown = sorted(
        name
        for name in environ
        if name.startswith((PREFIX, VLLM_PREFIX))
        and name not in DEFAULTS
        and name not in HEALTHCHECK_ONLY_ENV
        and name not in IMMUTABLE_BUILD_ENV
        and name not in IMMUTABLE_PROVENANCE_ENV
    )
    if unknown:
        if unknown[0].startswith(PREFIX):
            raise ValueError(
                f"unknown GLM53 LMCache environment variable: {unknown[0]}"
            )
        raise ValueError(f"unknown GLM53 runtime environment variable: {unknown[0]}")

    l1_size = _number(environ, "GLM53_LMCACHE_L1_SIZE_GB", minimum_exclusive=0)
    l1_init = _integer(environ, "GLM53_LMCACHE_L1_INIT_SIZE_GB")
    if l1_init > l1_size:
        raise ValueError(
            "GLM53_LMCACHE_L1_INIT_SIZE_GB must not exceed "
            f"GLM53_LMCACHE_L1_SIZE_GB ({l1_size:g}), got {l1_init}"
        )
    align = _integer(environ, "GLM53_LMCACHE_L1_ALIGN_BYTES")
    if align & (align - 1):
        raise ValueError(
            f"GLM53_LMCACHE_L1_ALIGN_BYTES must be a power of two, got {align}"
        )
    vllm_health_url, vllm_health_host, vllm_health_port, vllm_health_path = _http_url(
        environ, "GLM53_VLLM_HEALTH_URL"
    )

    return Config(
        host=_host(environ, "GLM53_LMCACHE_HOST"),
        port=_integer(environ, "GLM53_LMCACHE_PORT", maximum=65535),
        http_host=_host(environ, "GLM53_LMCACHE_HTTP_HOST"),
        http_port=_integer(environ, "GLM53_LMCACHE_HTTP_PORT", maximum=65535),
        l1_size_gb=l1_size,
        l1_init_size_gb=l1_init,
        max_gpu_workers=_integer(environ, "GLM53_LMCACHE_MAX_GPU_WORKERS"),
        max_cpu_workers=_integer(environ, "GLM53_LMCACHE_MAX_CPU_WORKERS"),
        chunk_size=_integer(environ, "GLM53_LMCACHE_CHUNK_SIZE"),
        l1_align_bytes=align,
        eviction_trigger=_number(
            environ,
            "GLM53_LMCACHE_EVICTION_TRIGGER",
            minimum_exclusive=0,
            maximum_inclusive=1,
        ),
        eviction_ratio=_number(
            environ,
            "GLM53_LMCACHE_EVICTION_RATIO",
            minimum_exclusive=0,
            maximum_inclusive=1,
        ),
        eviction_policy=_choice(
            environ,
            "GLM53_LMCACHE_EVICTION_POLICY",
            {"LRU", "IsolatedLRU", "noop"},
        ),
        transfer_mode=_choice(
            environ,
            "GLM53_LMCACHE_TRANSFER_MODE",
            {"lmcache_driven", "engine_driven", "auto"},
        ),
        startup_timeout_seconds=_number(
            environ,
            "GLM53_LMCACHE_STARTUP_TIMEOUT_SECONDS",
            minimum_exclusive=0,
        ),
        health_poll_seconds=_number(
            environ,
            "GLM53_LMCACHE_HEALTH_POLL_SECONDS",
            minimum_exclusive=0,
        ),
        shutdown_grace_seconds=_number(
            environ,
            "GLM53_LMCACHE_SHUTDOWN_GRACE_SECONDS",
            minimum_exclusive=0,
        ),
        vllm_health_url=vllm_health_url,
        vllm_health_host=vllm_health_host,
        vllm_health_port=vllm_health_port,
        vllm_health_path=vllm_health_path,
        vllm_startup_timeout_seconds=_number(
            environ,
            "GLM53_VLLM_STARTUP_TIMEOUT_SECONDS",
            minimum_exclusive=0,
        ),
        vllm_health_poll_seconds=_number(
            environ,
            "GLM53_VLLM_HEALTH_POLL_SECONDS",
            minimum_exclusive=0,
        ),
        broker_dir=Path(_value(environ, "GLM53_LMCACHE_CUMEM_BROKER_DIR")),
        lmcache_executable=_executable(environ, "GLM53_LMCACHE_EXECUTABLE"),
        vllm_executable=_executable(environ, "GLM53_VLLM_EXECUTABLE"),
    )


def ensure_broker_directory(path: Path) -> None:
    """Create and verify a private, writable, non-symlink broker directory."""
    if not path.is_absolute():
        raise ValueError(f"cuMem broker directory must be absolute, got {path}")
    try:
        before = path.lstat()
    except FileNotFoundError:
        path.mkdir(parents=True, mode=0o700)
        before = path.lstat()
    if stat.S_ISLNK(before.st_mode):
        raise ValueError(f"cuMem broker directory must not be a symlink: {path}")
    if not stat.S_ISDIR(before.st_mode):
        raise ValueError(f"cuMem broker path is not a directory: {path}")

    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fchmod(descriptor, 0o700)
        current = os.fstat(descriptor)
        if stat.S_IMODE(current.st_mode) != 0o700:
            raise ValueError(f"could not set cuMem broker directory mode 0700: {path}")
        with tempfile.NamedTemporaryFile(dir=path):
            pass
    except OSError as exc:
        raise ValueError(
            f"cuMem broker directory is not writable: {path}: {exc}"
        ) from exc
    finally:
        os.close(descriptor)


def _display_number(value: float) -> str:
    return f"{value:g}"


def lmcache_argv(config: Config) -> list[str]:
    return [
        config.lmcache_executable,
        "server",
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--http-host",
        config.http_host,
        "--http-port",
        str(config.http_port),
        "--l1-size-gb",
        _display_number(config.l1_size_gb),
        "--l1-init-size-gb",
        str(config.l1_init_size_gb),
        "--l1-align-bytes",
        str(config.l1_align_bytes),
        "--max-gpu-workers",
        str(config.max_gpu_workers),
        "--max-cpu-workers",
        str(config.max_cpu_workers),
        "--chunk-size",
        str(config.chunk_size),
        "--eviction-trigger-watermark",
        _display_number(config.eviction_trigger),
        "--eviction-ratio",
        _display_number(config.eviction_ratio),
        "--eviction-policy",
        config.eviction_policy,
        "--supported-transfer-mode",
        config.transfer_mode,
    ]


def child_environment(config: Config, environ: Mapping[str, str]) -> dict[str, str]:
    child_env = dict(environ)
    child_env["LMCACHE_CUMEM_BROKER_DIR"] = str(config.broker_dir)
    child_env["LMCACHE_MP_TRANSFER_MODE"] = config.transfer_mode
    return child_env


def _normalized_status(returncode: int) -> int:
    return 128 + abs(returncode) if returncode < 0 else returncode


def _group_exists(process: subprocess.Popen[object]) -> bool:
    try:
        os.killpg(process.pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _signal_group(process: subprocess.Popen[object], signum: int) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signum)


def _stop_processes(
    processes: Sequence[subprocess.Popen[object]], grace_seconds: float
) -> None:
    for process in processes:
        _signal_group(process, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        for process in processes:
            process.poll()
        if all(not _group_exists(process) for process in processes):
            break
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    for process in processes:
        if _group_exists(process):
            _signal_group(process, signal.SIGKILL)
    for process in processes:
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            print(
                f"[glm53-lifecycle] child process group {process.pid} did not reap",
                file=sys.stderr,
                flush=True,
            )


def _health_host(host: str) -> str:
    if host == "0.0.0.0":
        return "127.0.0.1"
    if host == "::":
        return "::1"
    return host


def _healthy(config: Config, timeout: float) -> bool:
    connection = http.client.HTTPConnection(
        _health_host(config.http_host), config.http_port, timeout=timeout
    )
    try:
        connection.request("GET", "/healthcheck")
        response = connection.getresponse()
        response.read()
        return response.status == 200
    except (OSError, http.client.HTTPException):
        return False
    finally:
        connection.close()


def _vllm_healthy(config: Config, timeout: float) -> bool:
    connection = http.client.HTTPConnection(
        config.vllm_health_host,
        config.vllm_health_port,
        timeout=timeout,
    )
    try:
        connection.request("GET", config.vllm_health_path)
        response = connection.getresponse()
        response.read()
        return response.status == 200
    except (OSError, http.client.HTTPException):
        return False
    finally:
        connection.close()


def _validate_executables(config: Config) -> None:
    for name, path in (
        ("LMCache", config.lmcache_executable),
        ("vLLM", config.vllm_executable),
    ):
        if not os.path.isfile(path) or not os.access(path, os.X_OK):
            raise ValueError(f"{name} executable is missing or not executable: {path}")


def supervise(vllm_args: Sequence[str], environ: Mapping[str, str]) -> int:
    config = load_config(environ)
    if not vllm_args or vllm_args[0] != "serve":
        raise ValueError(
            "vLLM arguments must use the supported form: serve MODEL [OPTIONS]"
        )
    _validate_executables(config)
    ensure_broker_directory(config.broker_dir)
    env = child_environment(config, environ)
    children: list[subprocess.Popen[object]] = []
    received_signal: int | None = None

    def forward(signum: int, _frame: object) -> None:
        nonlocal received_signal
        if received_signal is None:
            received_signal = signum
            print(
                f"[glm53-lifecycle] received signal {signum}; stopping children",
                file=sys.stderr,
                flush=True,
            )
        for child in children:
            _signal_group(child, signum)

    previous = {
        signum: signal.signal(signum, forward)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        print("[glm53-lifecycle] starting LMCache", file=sys.stderr, flush=True)
        lmcache = subprocess.Popen(
            lmcache_argv(config),
            env=env,
            start_new_session=True,
        )
        children.append(lmcache)

        deadline = time.monotonic() + config.startup_timeout_seconds
        while True:
            if received_signal is not None:
                _stop_processes(children, config.shutdown_grace_seconds)
                return 128 + received_signal
            lmcache_status = lmcache.poll()
            if lmcache_status is not None:
                print(
                    "[glm53-lifecycle] LMCache exited before healthcheck "
                    f"(status={_normalized_status(lmcache_status)})",
                    file=sys.stderr,
                    flush=True,
                )
                return _normalized_status(lmcache_status) or 1
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                print(
                    "[glm53-lifecycle] LMCache healthcheck timed out after "
                    f"{config.startup_timeout_seconds:g}s",
                    file=sys.stderr,
                    flush=True,
                )
                _stop_processes(children, config.shutdown_grace_seconds)
                return 1
            if _healthy(config, min(1.0, remaining)):
                break
            time.sleep(min(config.health_poll_seconds, max(0.0, remaining)))

        print(
            "[glm53-lifecycle] LMCache healthy; starting vLLM",
            file=sys.stderr,
            flush=True,
        )
        vllm = subprocess.Popen(
            [config.vllm_executable, *vllm_args],
            env=env,
            start_new_session=True,
        )
        children.append(vllm)

        deadline = time.monotonic() + config.vllm_startup_timeout_seconds
        while True:
            if received_signal is not None:
                _stop_processes(children, config.shutdown_grace_seconds)
                return 128 + received_signal
            vllm_status = vllm.poll()
            if vllm_status is not None:
                print(
                    "[glm53-lifecycle] vLLM exited before readiness "
                    f"(status={_normalized_status(vllm_status)}); stopping LMCache",
                    file=sys.stderr,
                    flush=True,
                )
                _stop_processes([lmcache], config.shutdown_grace_seconds)
                return _normalized_status(vllm_status)
            lmcache_status = lmcache.poll()
            if lmcache_status is not None:
                print(
                    "[glm53-lifecycle] LMCache exited during vLLM startup "
                    f"(status={_normalized_status(lmcache_status)}); stopping vLLM",
                    file=sys.stderr,
                    flush=True,
                )
                _stop_processes([vllm], config.shutdown_grace_seconds)
                return _normalized_status(lmcache_status) or 1
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                print(
                    "[glm53-lifecycle] vLLM readiness timed out after "
                    f"{config.vllm_startup_timeout_seconds:g}s",
                    file=sys.stderr,
                    flush=True,
                )
                _stop_processes(children, config.shutdown_grace_seconds)
                return 1
            if _vllm_healthy(config, min(1.0, remaining)):
                print(
                    "[glm53-lifecycle] vLLM healthy; entering steady-state supervision",
                    file=sys.stderr,
                    flush=True,
                )
                break
            time.sleep(min(config.vllm_health_poll_seconds, max(0.0, remaining)))

        while True:
            if received_signal is not None:
                _stop_processes(children, config.shutdown_grace_seconds)
                return 128 + received_signal
            vllm_status = vllm.poll()
            if vllm_status is not None:
                print(
                    f"[glm53-lifecycle] vLLM exited (status="
                    f"{_normalized_status(vllm_status)}); stopping LMCache",
                    file=sys.stderr,
                    flush=True,
                )
                _stop_processes([lmcache], config.shutdown_grace_seconds)
                return _normalized_status(vllm_status)
            lmcache_status = lmcache.poll()
            if lmcache_status is not None:
                print(
                    "[glm53-lifecycle] LMCache exited while vLLM was running "
                    f"(status={_normalized_status(lmcache_status)}); stopping vLLM",
                    file=sys.stderr,
                    flush=True,
                )
                _stop_processes([vllm], config.shutdown_grace_seconds)
                return _normalized_status(lmcache_status) or 1
            time.sleep(min(config.health_poll_seconds, 0.1))
    finally:
        _stop_processes(children, config.shutdown_grace_seconds)
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return supervise(
            list(sys.argv[1:] if argv is None else argv),
            os.environ,
        )
    except ValueError as exc:
        print(f"[glm53-lifecycle] configuration error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"[glm53-lifecycle] startup error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
