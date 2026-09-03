# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
from types import SimpleNamespace

import pytest

from vllm.entrypoints.serve.dev.fairness.api_router import (
    PrefillFairnessRequest,
    get_prefill_fairness,
    set_prefill_fairness,
)


class _FakeEngineClient:
    def __init__(self, result):
        self.result = result
        self.received = None

    async def get_prefill_fairness(self):
        return self.result

    async def set_prefill_fairness(self, config):
        self.received = config
        return self.result


def _request(client):
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(engine_client=client))
    )


def test_get_prefill_fairness_returns_current_config():
    current = {"fairness_engine": "compute_share", "prefill_compute_share": 0.5}
    response = asyncio.run(get_prefill_fairness(_request(_FakeEngineClient(current))))
    assert response.status_code == 200
    assert json.loads(response.body) == current


def test_request_accepts_auto_compute_share():
    config = PrefillFairnessRequest(
        fairness_engine="compute_share", prefill_compute_share="auto"
    )

    assert config.prefill_compute_share == "auto"


@pytest.mark.parametrize(
    ("reason", "status_code"),
    [("busy", 409), ("invalid", 422)],
)
def test_set_prefill_fairness_maps_rejection_status(reason, status_code):
    result = {
        "applied": False,
        "reason": reason,
        "message": "rejected",
        "config": {"fairness_engine": None},
    }
    client = _FakeEngineClient(result)
    config = PrefillFairnessRequest(
        fairness_engine="compute_share", prefill_compute_share=0.5
    )

    response = asyncio.run(set_prefill_fairness(_request(client), config))

    assert response.status_code == status_code
    assert client.received == config.model_dump()
