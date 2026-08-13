# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.serve.dev.cartridge.api_router import attach_router


def _client(engine) -> TestClient:
    app = FastAPI()
    attach_router(app)
    app.state.engine_client = engine
    return TestClient(app)


def test_load_cartridge_returns_updated_layers():
    engine = AsyncMock()
    engine.load_exl3_cartridge.return_value = [2, 2]
    client = _client(engine)

    response = client.post(
        "/load_exl3_cartridge", json={"adapter_path": "/tmp/cart.safetensors"}
    )

    assert response.status_code == 200
    assert response.json() == {"status": "loaded", "updated_layers": [2, 2]}
    engine.load_exl3_cartridge.assert_awaited_once_with("/tmp/cart.safetensors")


def test_load_cartridge_requires_adapter_path():
    client = _client(AsyncMock())

    response = client.post("/load_exl3_cartridge", json={})

    assert response.status_code == 400


def test_load_cartridge_maps_value_error_to_bad_request():
    engine = AsyncMock()
    engine.load_exl3_cartridge.side_effect = ValueError("bad cartridge")
    client = _client(engine)

    response = client.post(
        "/load_exl3_cartridge", json={"adapter_path": "/tmp/cart.safetensors"}
    )

    assert response.status_code == 400
    assert "bad cartridge" in response.json()["error"]


def test_deactivate_cartridge_returns_updated_layers():
    engine = AsyncMock()
    engine.deactivate_exl3_cartridge.return_value = [0, 0]
    client = _client(engine)

    response = client.post("/deactivate_exl3_cartridge")

    assert response.status_code == 200
    assert response.json() == {"status": "deactivated", "updated_layers": [0, 0]}


def test_cartridge_status_reports_active_workers():
    engine = AsyncMock()
    engine.collective_rpc.return_value = [True, False]
    client = _client(engine)

    response = client.get("/exl3_cartridge_status")

    assert response.status_code == 200
    assert response.json() == {"active": [True, False]}
    engine.collective_rpc.assert_awaited_once_with("has_exl3_cartridge")


def test_endpoints_reject_engine_without_cartridge_support():
    engine = object()  # no load_exl3_cartridge attribute
    client = _client(engine)

    response = client.post(
        "/load_exl3_cartridge", json={"adapter_path": "/tmp/cart.safetensors"}
    )

    assert response.status_code == 501
