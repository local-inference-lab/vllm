# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

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
    current = {
        "prefill_compute_share": "auto",
        "prefill_compute_half_life": 0.2,
        "effective_prefill_compute_half_life_seconds": 0.2,
        "effective_prefill_compute_share": 0.55,
    }
    response = asyncio.run(get_prefill_fairness(_request(_FakeEngineClient(current))))
    assert response.status_code == 200
    assert json.loads(response.body) == current


@pytest.mark.parametrize(
    ("half_life", "expected"),
    [(None, None), ("smooth", "smooth"), ("responsive", "responsive"), (0.2, 0.2)],
)
def test_request_accepts_auto_compute_share(half_life, expected):
    config = PrefillFairnessRequest(
        prefill_compute_share="auto",
        prefill_compute_half_life=half_life,
    )

    assert config.prefill_compute_share == "auto"
    assert config.prefill_compute_half_life == expected


def test_request_rejects_removed_profile_name():
    with pytest.raises(ValidationError, match="prefill_compute_share"):
        PrefillFairnessRequest(prefill_compute_share="responsive")


def test_request_rejects_half_life_outside_auto_mode():
    with pytest.raises(ValidationError, match="requires prefill_compute_share='auto'"):
        PrefillFairnessRequest(
            prefill_compute_share=0.5,
            prefill_compute_half_life="responsive",
        )


def test_request_rejects_removed_selector():
    with pytest.raises(ValidationError, match="fairness_engine"):
        PrefillFairnessRequest(fairness_engine="micro_slicing")


def test_set_prefill_fairness_maps_invalid_rejection():
    result = {
        "applied": False,
        "reason": "invalid",
        "message": "rejected",
        "config": {"prefill_compute_share": None},
    }
    client = _FakeEngineClient(result)
    config = PrefillFairnessRequest(prefill_compute_share=0.5)

    response = asyncio.run(set_prefill_fairness(_request(client), config))

    assert response.status_code == 422
    assert client.received == config.model_dump()


@pytest.mark.parametrize("half_life", ["smooth", "responsive", 5.0, 0.2])
def test_set_prefill_fairness_applies_live_half_life_update(half_life):
    result = {
        "applied": True,
        "config": {
            "prefill_compute_share": "auto",
            "prefill_compute_half_life": half_life,
            "effective_prefill_compute_share": 0.4,
        },
    }
    client = _FakeEngineClient(result)
    config = PrefillFairnessRequest(
        prefill_compute_share="auto",
        prefill_compute_half_life=half_life,
    )

    response = asyncio.run(set_prefill_fairness(_request(client), config))

    assert response.status_code == 200
    assert client.received == config.model_dump()


def test_set_prefill_fairness_sends_complete_replacement_config():
    result = {
        "applied": True,
        "config": {
            "prefill_compute_share": "auto",
            "prefill_compute_half_life": "smooth",
        },
    }
    client = _FakeEngineClient(result)
    config = PrefillFairnessRequest(
        prefill_compute_share="auto",
        prefill_compute_half_life="smooth",
    )

    response = asyncio.run(set_prefill_fairness(_request(client), config))

    assert response.status_code == 200
    assert client.received == {
        "prefill_compute_share": "auto",
        "prefill_compute_half_life": "smooth",
    }
