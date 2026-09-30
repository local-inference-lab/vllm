# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A server that stops because its engine died exits with a failure status.

Restart policies and supervisors (Docker ``on-failure``, systemd) only act on
a non-zero exit; a crashed engine used to end ``vllm serve`` with status 0.
"""

from types import SimpleNamespace

import pytest
import zmq

from vllm.entrypoints.launchers.api_server.entry import exit_if_engine_failed
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.core_client import BackgroundResources
from vllm.v1.engine.exceptions import EngineDeadError


def _client(resources: BackgroundResources) -> SimpleNamespace:
    return SimpleNamespace(engine_core=SimpleNamespace(resources=resources))


def test_a_reported_engine_death_is_a_failure():
    resources = BackgroundResources(ctx=zmq.Context())
    with pytest.raises(EngineDeadError):
        resources.validate_alive([zmq.Frame(EngineCoreProc.ENGINE_CORE_DEAD)])
    assert resources.engine_dead and resources.engine_failed
    with pytest.raises(SystemExit) as exited:
        exit_if_engine_failed(_client(resources))
    assert exited.value.code == 1


def test_an_orderly_shutdown_is_not_a_failure():
    resources = BackgroundResources(ctx=zmq.Context())
    resources()  # the cleanup an orderly shutdown runs
    assert resources.engine_dead and not resources.engine_failed
    exit_if_engine_failed(_client(resources))
    # Clients without these resources (for example a remote engine) never fail it.
    exit_if_engine_failed(SimpleNamespace())
