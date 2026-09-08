#!/opt/venv/bin/python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Container healthcheck for the supervised vLLM endpoint."""

from __future__ import annotations

import http.client
import os
import sys
from urllib.parse import urlsplit

DEFAULT_URL = "http://127.0.0.1:8000/health"


def main() -> int:
    raw_url = os.environ.get("GLM53_VLLM_HEALTH_URL", DEFAULT_URL)
    parsed = urlsplit(raw_url)
    if (
        parsed.scheme != "http"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        print(f"invalid GLM53_VLLM_HEALTH_URL: {raw_url!r}", file=sys.stderr)
        return 2
    timeout_raw = os.environ.get("GLM53_VLLM_HEALTH_TIMEOUT_SECONDS", "5")
    try:
        timeout = float(timeout_raw)
        if timeout <= 0:
            raise ValueError
    except ValueError:
        print(
            "GLM53_VLLM_HEALTH_TIMEOUT_SECONDS must be positive",
            file=sys.stderr,
        )
        return 2

    connection = http.client.HTTPConnection(
        parsed.hostname, parsed.port or 80, timeout=timeout
    )
    try:
        connection.request("GET", parsed.path or "/health")
        response = connection.getresponse()
        response.read()
        if response.status != 200:
            print(f"vLLM health returned HTTP {response.status}", file=sys.stderr)
            return 1
        return 0
    except (OSError, http.client.HTTPException) as exc:
        print(f"vLLM health request failed: {exc}", file=sys.stderr)
        return 1
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
