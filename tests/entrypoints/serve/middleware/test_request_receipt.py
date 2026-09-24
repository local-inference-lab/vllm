# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The receipt middleware times requests held before rendering."""

import asyncio
import time

import pytest

from vllm.entrypoints.serve.middleware.request_receipt import RequestReceiptMiddleware
from vllm.v1.engine.stall_diagnostics import REQUEST_RECEIPT, RequestTimeline

SCOPE = {"type": "http", "method": "POST", "path": "/v1/chat/completions"}


def _slow_body(delay: float):
    parts = [
        {"type": "http.request", "body": b"{", "more_body": True},
        {"type": "http.request", "body": b"}", "more_body": False},
    ]

    async def receive():
        message = parts.pop(0)
        if not message["more_body"]:
            await asyncio.sleep(delay)
        return message

    return receive


def test_timeline_includes_http_receipt_and_body_read(caplog_vllm):
    seen = {}

    async def app(scope, receive, send):
        while (await receive()).get("more_body"):
            pass
        seen["timeline"] = RequestTimeline("req-1", time.time(), time.time())
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def send(message):
        return None

    asyncio.run(RequestReceiptMiddleware(app)(SCOPE, _slow_body(0.3), send))
    timeline = seen["timeline"]
    assert timeline.received is not None and timeline.body_read is not None
    assert timeline.body_read - timeline.received >= 0.3
    timeline.close("finished", threshold=0.0, always=True)
    assert "HTTP receipt to body read 0.3" in caplog_vllm.text
    assert REQUEST_RECEIPT.get() is None


def test_request_ending_without_response_is_reported(caplog_vllm):
    async def app(scope, receive, send):
        while (await receive()).get("more_body"):
            pass
        await asyncio.sleep(0.3)
        raise RuntimeError("client disconnected")

    async def send(message):
        return None

    middleware = RequestReceiptMiddleware(app)
    middleware.threshold = 0.2
    with pytest.raises(RuntimeError, match="client disconnected"):
        asyncio.run(middleware(SCOPE, _slow_body(0.0), send))
    assert "HTTP POST /v1/chat/completions ended after 0.3" in caplog_vllm.text
    assert "without a response" in caplog_vllm.text


def test_api_server_registers_the_receipt_middleware():
    from argparse import Namespace

    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient

    from vllm.entrypoints.serve.middleware.register import (
        init_entrypoints_middleware,
    )

    app = FastAPI()

    @app.post("/probe")
    async def probe(request: Request):
        await request.body()
        receipt = REQUEST_RECEIPT.get()
        return {"received": receipt is not None and receipt.body_read is not None}

    args = Namespace(
        allowed_origins=["*"],
        allow_credentials=False,
        allowed_methods=["*"],
        allowed_headers=["*"],
        api_key=None,
        enable_request_id_headers=False,
        middleware=[],
    )
    init_entrypoints_middleware(args, app, ())
    assert TestClient(app).post("/probe", json={}).json() == {"received": True}
