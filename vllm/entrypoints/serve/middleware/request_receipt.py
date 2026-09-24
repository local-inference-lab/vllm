# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stamp when each HTTP request arrives and when its body is fully read.

The renderer stamps a request's arrival only when rendering starts, after the
body has been read and parsed. These stamps let the request timeline show a
wait before that point, such as a slowly delivered request body.
"""

import time

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from vllm import envs
from vllm.logger import init_logger
from vllm.v1.engine.stall_diagnostics import REQUEST_RECEIPT, RequestReceipt

logger = init_logger(__name__)


class RequestReceiptMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.threshold = envs.VLLM_REQUEST_STALL_WARNING_S

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        receipt = RequestReceipt(time.time())

        async def timed_receive() -> Message:
            message = await receive()
            if (
                receipt.body_read is None
                and message["type"] == "http.request"
                and not message.get("more_body", False)
            ):
                receipt.body_read = time.time()
            return message

        responded = False

        async def watched_send(message: Message) -> None:
            nonlocal responded
            if message["type"] == "http.response.start":
                responded = True
            await send(message)

        token = REQUEST_RECEIPT.set(receipt)
        try:
            await self.app(scope, timed_receive, watched_send)
        finally:
            REQUEST_RECEIPT.reset(token)
            # A request held before rendering never reaches the engine's
            # timeline; report it here if it ended without any response.
            elapsed = time.time() - receipt.received
            if not responded and 0 < self.threshold < elapsed:
                logger.warning(
                    "HTTP %s %s ended after %.2f s without a response; body read %s",
                    scope.get("method", ""),
                    scope.get("path", ""),
                    elapsed,
                    "never"
                    if receipt.body_read is None
                    else f"after {receipt.body_read - receipt.received:.2f} s",
                )
