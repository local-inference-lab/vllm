# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Record live GLM-5.3 completions and compare speculative token sequences."""

import argparse
import concurrent.futures
import json
import time
from pathlib import Path

import httpx
from transformers import AutoTokenizer

CASES = (
    ("arithmetic", "What is 37 * 24? Give only the integer.", 32),
    (
        "python",
        (
            "Write a Python function that merges two sorted integer lists in linear "
            "time without calling sort or sorted. Include a short example."
        ),
        256,
    ),
    (
        "systems",
        (
            "Explain the difference between an acquire load and a release store, "
            "with a short producer-consumer example."
        ),
        256,
    ),
    ("spanish", "Explica en dos frases por qué el cielo es azul.", 96),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8002")
    parser.add_argument("--model", default="zai-org/GLM-5.3-Flash")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--max-tokens", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None

    with httpx.Client(base_url=args.url, timeout=600) as client:
        client.get("/health").raise_for_status()
        before = client.get("/metrics")
        before.raise_for_status()

        def complete(case):
            name, prompt, max_tokens = case
            prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
            )
            request = {
                "model": args.model,
                "prompt": prompt_ids,
                "max_tokens": args.max_tokens or max_tokens,
                "temperature": 0,
                "seed": 23,
                "return_token_ids": True,
            }
            start = time.perf_counter()
            response = client.post("/v1/completions", json=request)
            elapsed = time.perf_counter() - start
            response.raise_for_status()
            return {
                "name": name,
                "request": request,
                "elapsed_seconds": elapsed,
                "response": response.json(),
            }

        records = [complete(case) for case in CASES]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            concurrent_records = list(pool.map(complete, CASES))
        after = client.get("/metrics")
        after.raise_for_status()

    result = {
        "url": args.url,
        "tokenizer": args.tokenizer,
        "sequential": records,
        "concurrent": concurrent_records,
        "metrics_before": before.text,
        "metrics_after": after.text,
    }
    if baseline is not None:
        comparisons = []
        for mode in ("sequential", "concurrent"):
            for actual, expected in zip(result[mode], baseline[mode], strict=True):
                assert actual["name"] == expected["name"]
                assert actual["request"] == expected["request"]
                a = actual["response"]["choices"][0]["token_ids"]
                b = expected["response"]["choices"][0]["token_ids"]
                comparisons.append(
                    {"mode": mode, "case": actual["name"], "tokens_equal": a == b}
                )
        result["baseline_comparisons"] = comparisons
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps({"output": str(args.output), "requests": len(records) * 2}))
    if baseline is not None:
        print(json.dumps(result["baseline_comparisons"]))


if __name__ == "__main__":
    main()
