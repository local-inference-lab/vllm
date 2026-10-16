# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Probe recurrent suffix recovery with a synthetic final-token edit.

Run against a supported request-boundary server with
--enable-prompt-tokens-details, once with the default stride and once with
--recurrent-prefill-tail-stride 256. No cache reset is needed: each run uses a
fresh cache salt. These token-ID inputs isolate checkpoint placement; they do
not reproduce the repository/tool workload or evaluate model quality.
"""

import argparse
import hashlib
import json
import os
import time
import urllib.request
import uuid


def main():
    """Print cache usage and timings for cold, repeated, and edited prompts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=64632)
    parser.add_argument("--append-tokens", type=int, default=203)
    args = parser.parse_args()
    if args.prompt_tokens < 2 or args.append_tokens < 0:
        parser.error("prompt-tokens must be >= 2 and append-tokens must be >= 0")

    # Qwen's single/double newline IDs model a thinking-prompt history edit.
    prompt = [1000] * (args.prompt_tokens - 1) + [198]
    edited = prompt[:-1] + [271] + [1001] * args.append_tokens
    salt = uuid.uuid4().hex
    headers = {"Content-Type": "application/json"}
    if key := os.environ.get("VLLM_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    for label, tokens in (
        ("cold", prompt),
        ("repeat", prompt),
        ("suffix_edit", edited),
    ):
        payload = {
            "model": args.model,
            "prompt": tokens,
            "cache_salt": salt,
            "max_tokens": 1,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        request = urllib.request.Request(
            args.base_url.rstrip("/") + "/v1/completions",
            data=json.dumps(payload).encode(),
            headers=headers,
        )
        started = time.monotonic()
        ttft = None
        usage = {}
        with urllib.request.urlopen(request, timeout=1800) as response:
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("error"):
                    raise RuntimeError(chunk["error"])
                if chunk.get("usage"):
                    usage = chunk["usage"]
                if ttft is None and any(
                    c.get("text") for c in chunk.get("choices", [])
                ):
                    ttft = time.monotonic() - started
        if not usage:
            raise RuntimeError("Stream ended without a usage chunk")
        if "prompt_tokens" not in usage:
            raise RuntimeError("Usage chunk is missing prompt_tokens")
        details = usage.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens")
        if cached is None:
            raise RuntimeError("Server must enable --enable-prompt-tokens-details")
        print(
            json.dumps(
                {
                    "case": label,
                    "input_sha256": hashlib.sha256(
                        json.dumps(tokens).encode()
                    ).hexdigest(),
                    "prompt_tokens": usage["prompt_tokens"],
                    "cached_tokens": cached,
                    "recomputed_tokens": usage["prompt_tokens"] - cached,
                    "ttft_s": ttft,
                    "elapsed_s": time.monotonic() - started,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
