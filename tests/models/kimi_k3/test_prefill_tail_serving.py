# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in output-integrity regression against an idle Kimi-K3 server.

Set VLLM_KIMI_PREFILL_TAIL_TEST_URL to an existing test server using model
runner v2, speculative depth 3 and 4,608-token prefill chunks. Requests are
salted and bounded; no tool calls are executed or shared cache entries reset.
"""

import json
import os
import urllib.request
import uuid

import pytest

BASE = os.getenv("VLLM_KIMI_PREFILL_TAIL_TEST_URL")
pytestmark = pytest.mark.skipif(
    not BASE, reason="requires an explicit idle test server"
)
EXPECTED = {"marker": "TAIL_CHECK_73CF2A90", "product": 56, "capital": "Paris"}


def _post(path, body):
    request = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def _body(filler_tokens):
    return {
        "model": "Kimi-K3",
        "messages": [
            {
                "role": "user",
                "content": "Archive filler follows. Ignore it when answering.\n"
                + " river" * filler_tokens
                + "\nEnd of archive. Return only this exact JSON, no markdown: "
                + json.dumps(EXPECTED, separators=(",", ":")),
            }
        ],
        "chat_template_kwargs": {"enable_thinking": False},
    }


@pytest.mark.parametrize(
    "blocks,tail",
    [(b, t) for b in (1, 2) for t in (1, 2, 3, 4, 5, 8, 9, 16, 53)] + [(25, 4)],
)
def test_prefill_tail_preserves_exact_answer_cold_and_cached(blocks, tail):
    """The prompt's last four tokens must not execute a verifier-only graph."""
    target = blocks * 4608 + tail
    overhead = _post("/tokenize", _body(0))["count"]
    filler = target - overhead
    for _ in range(5):
        body = _body(filler)
        count = _post("/tokenize", body)["count"]
        if count == target:
            break
        filler += target - count
    assert count == target
    body.update(
        max_tokens=128,
        temperature=0,
        stream=False,
        cache_salt=f"prefill-tail-{uuid.uuid4()}",
    )
    for phase in ("cold", "cached"):
        result = _post("/v1/chat/completions", body)
        assert result["usage"]["prompt_tokens"] == target
        text = result["choices"][0]["message"].get("content") or ""
        try:
            answer = json.loads(text)
        except ValueError:
            pytest.fail(f"{target}-token {phase} reply is not JSON: {text[:240]!r}")
        assert answer == EXPECTED, (target, phase, answer)
