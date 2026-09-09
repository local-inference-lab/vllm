# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run serial GLM prefill/cache smoke checks and fixed cold TTFT samples.

Requires /health, /metrics, /tokenize and OpenAI-compatible chat endpoints.
This client controls no model processes or hardware. Readiness and exclusive
access are operator attestations; feature activation is not verified here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

import regex as re

SIZES = (8192, 16384, 32768)
REPEATS = 3
SEMANTIC_MAX_TOKENS = 384
MEASURED_PROTOCOL_SOURCE_SHA256 = (
    "6e4bf5ff62379bb2db4dc439d27a85eda0837959e983ef5f92ab3605898e69ac"
)


def normalize_base_url(value):
    """Normalize the API prefix without accepting credentials in journaled URLs."""
    base = value.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    parsed = urllib.parse.urlsplit(base)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "Use an HTTP(S) base URL without credentials, query or fragment"
        )
    return base


def load_conditions(path):
    """Read optional operator metadata without promoting it to verification."""
    if path is None:
        return None, None
    raw = path.read_bytes()
    conditions = json.loads(raw)
    if (
        not isinstance(conditions, dict)
        or conditions.get("schema") != "glm-prefill-reproduction-conditions/v1"
        or set(conditions) - {"schema", "sources", "settings"}
        or not isinstance(conditions.get("sources", {}), dict)
        or not isinstance(conditions.get("settings", {}), dict)
    ):
        raise ValueError(
            "Conditions require schema glm-prefill-reproduction-conditions/v1 "
            "and optional sources/settings objects"
        )
    if any(
        not isinstance(value, str) or not value
        for value in conditions.get("sources", {}).values()
    ):
        raise ValueError("Condition source identities must be nonempty strings")
    return conditions, hashlib.sha256(raw).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def cached_tokens(usage):
    details = usage.get("prompt_tokens_details")
    count = details.get("cached_tokens") if isinstance(details, dict) else None
    if type(count) is not int or count < 0:
        raise ValueError("Response must provide explicit nonnegative cached_tokens")
    return count


def validate_usage(usage, expected_tokens, cache):
    if (
        type(usage.get("prompt_tokens")) is not int
        or usage["prompt_tokens"] != expected_tokens
    ):
        raise ValueError(
            f"Prompt usage differs: expected {expected_tokens}, "
            f"got {usage.get('prompt_tokens')}"
        )
    count = cached_tokens(usage)
    if cache == "cold" and count != 0:
        raise ValueError(f"Cold request reused {count} tokens")
    if cache == "reuse" and count <= 0:
        raise ValueError(
            "Expected cache reuse was absent; recomputation is not a reuse pass"
        )
    return count


def request_gauges(metrics):
    result = {}
    for metric in ("num_requests_running", "num_requests_waiting"):
        pattern = re.compile(
            r"^vllm:" + metric + r"(?:\{[^\n]*\})?\s+(\S+)(?:\s+\S+)?$"
        )
        values = [
            float(match.group(1))
            for line in metrics.splitlines()
            if (match := pattern.match(line))
        ]
        if not values or any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError(f"Missing or invalid vllm:{metric} metrics")
        result[metric] = sum(values)
    return result


def first_token_delta(chunk):
    return any(
        any(
            choice.get("delta", {}).get(key)
            for key in ("content", "reasoning", "reasoning_content")
        )
        for choice in chunk.get("choices", [])
    )


def summarize(cases):
    result = []
    for size in SIZES:
        selected = [
            row
            for row in cases
            if row["phase"] == "measured"
            and row.get("accepted")
            and row["tokens"] == size
        ]
        if not selected:
            continue
        times = [row["ttft_seconds"] for row in selected]
        if any(not math.isfinite(value) or value <= 0 for value in times):
            raise ValueError("Invalid TTFT cannot enter a summary")
        median = statistics.median(times)
        result.append(
            {
                "tokens": size,
                "samples": len(times),
                "median_ttft_seconds": median,
                "min_ttft_seconds": min(times),
                "max_ttft_seconds": max(times),
                "tokens_per_second": size / median,
            }
        )
    return result


class Journal:
    def __init__(self, path, record):
        self.path, self.record = path, record
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(record, stream, indent=2)

    def save(self):
        self.record["updated_at_utc"] = utc_now()
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=self.path.parent,
            prefix=self.path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as stream:
            json.dump(self.record, stream, indent=2)
            stream.write("\n")
            temporary = Path(stream.name)
        temporary.replace(self.path)


class PrefillChecks:
    def __init__(self, args, journal):
        self.args, self.journal = args, journal
        self.base = normalize_base_url(args.base_url)
        self.headers = {"Content-Type": "application/json"}
        if args.api_key_env:
            key = os.environ.get(args.api_key_env)
            if not key:
                raise ValueError("Selected API-key environment variable is empty")
            self.headers["Authorization"] = "Bearer " + key
        self.model = args.model

    def control(self, path):
        request = urllib.request.Request(self.base + path, headers=self.headers)
        with urllib.request.urlopen(
            request, timeout=self.args.request_timeout
        ) as response:
            return response.read().decode("utf-8")

    def idle(self, *, drain=False, stable=1):
        deadline = time.monotonic() + self.args.idle_timeout
        consecutive = 0
        observations = []
        while True:
            self.control("/health")
            gauges = request_gauges(self.control("/metrics"))
            observations.append({"at_utc": utc_now(), **gauges})
            if all(value == 0 for value in gauges.values()):
                consecutive += 1
                if consecutive >= stable:
                    break
            else:
                consecutive = 0
                if not drain:
                    raise RuntimeError("Service is busy before an exclusive check")
            if time.monotonic() >= deadline:
                raise TimeoutError("Service did not reach idle before the deadline")
            time.sleep(0.5)
        self.journal.record["idle_checks"].append(observations)
        self.journal.save()

    def post(self, path, body, *, stream=False):
        row = {
            "path": path,
            "request": body,
            "started_at_utc": utc_now(),
            "complete": False,
        }
        index = len(self.journal.record["requests"])
        self.journal.record["requests"].append(row)
        self.journal.save()
        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.base + path, data=data, headers=self.headers
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(
                request, timeout=self.args.request_timeout
            ) as response:
                row["http_status"] = response.status
                if not stream:
                    payload = json.loads(response.read())
                    row["response"] = payload
                    row["elapsed_seconds"] = time.perf_counter() - started
                else:
                    chunks, first, usage, done = [], None, None, False
                    row["sse_chunks"] = chunks
                    for line in response:
                        if not line.startswith(b"data:"):
                            continue
                        raw = line[5:].strip()
                        if raw == b"[DONE]":
                            done = True
                            break
                        chunk = json.loads(raw)
                        chunks.append(chunk)
                        if chunk.get("error"):
                            raise RuntimeError(f"Streaming API error: {chunk['error']}")
                        if first is None and first_token_delta(chunk):
                            first = time.perf_counter() - started
                        if chunk.get("usage") is not None:
                            usage = chunk["usage"]
                    row.update(
                        ttft_seconds=first,
                        usage=usage,
                        saw_done=done,
                        elapsed_seconds=time.perf_counter() - started,
                    )
                    if first is None or usage is None or not done:
                        raise RuntimeError(
                            "Stream lacks a token delta, final usage, or DONE marker"
                        )
                    payload = row
            row["complete"] = True
            return payload, index
        except urllib.error.HTTPError as exc:
            row.update(
                http_status=exc.code,
                error_response=exc.read().decode("utf-8", errors="replace"),
            )
            raise
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            row["finished_at_utc"] = utc_now()
            self.journal.save()

    def count(self, messages):
        response, index = self.post(
            "/tokenize",
            {"model": self.model, "messages": messages, "add_generation_prompt": True},
        )
        count = response.get("count")
        if type(count) is not int or count <= 0:
            raise ValueError("Tokenizer must return a positive integer count")
        return count, index

    def calibrate(self, tokens, fact):
        nonce = uuid.uuid4().hex
        prefix = f"Test {nonce}. The project code is {fact}. Remember it.\n"
        suffix = "\nWhat is the project code? Reply with just the code."
        words = tokens
        indices = []
        for _ in range(12):
            if words < 0:
                raise ValueError("Exact-token calibration exhausted prompt padding")
            text = (
                prefix
                + " ".join(
                    (["alpha", "beta", "gamma", "delta"] * ((words + 3) // 4))[:words]
                )
                + suffix
            )
            messages = [{"role": "user", "content": text}]
            count, index = self.count(messages)
            indices.append(index)
            if count == tokens:
                return messages, indices
            words += tokens - count
        raise ValueError("Exact prompt calibration did not converge in twelve attempts")

    def prefill(self, tokens, phase, sample):
        self.idle()
        messages, calibration = self.calibrate(tokens, "STONE-7482")
        self.idle()
        case = {
            "phase": phase,
            "sample": sample,
            "tokens": tokens,
            "calibration_requests": calibration,
            "accepted": False,
            "request_index": len(self.journal.record["requests"]),
        }
        self.journal.record["cases"].append(case)
        response, index = self.post(
            "/v1/chat/completions",
            {
                "model": self.model,
                "messages": messages,
                "max_tokens": 1,
                "temperature": 0,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            stream=True,
        )
        case["request_index"] = index
        case["cached_tokens"] = validate_usage(response["usage"], tokens, "cold")
        case["ttft_seconds"] = response["ttft_seconds"]
        self.idle(drain=True)
        case["accepted"] = True
        self.journal.save()
        print(
            json.dumps(
                {
                    key: case[key]
                    for key in ("phase", "sample", "tokens", "ttft_seconds", "accepted")
                }
            ),
            flush=True,
        )

    def semantic(self, tokens, *, reuse):
        self.idle()
        fact = "RIVER-" + str(1000 + int(uuid.uuid4().hex[:4], 16) % 9000)
        messages, calibration = self.calibrate(tokens, fact)
        for kind in ("cold", "repeated", "extended") if reuse else ("cold",):
            self.idle()
            query = (
                messages
                if kind != "extended"
                else [
                    {
                        "role": "user",
                        "content": messages[0]["content"]
                        + "\nFinal instruction: give exactly the project code.",
                    }
                ]
            )
            expected_tokens, count_index = self.count(query)
            if kind != "extended" and expected_tokens != tokens:
                raise ValueError(
                    "Exact semantic prompt count changed after calibration"
                )
            if kind == "extended" and expected_tokens <= tokens:
                raise ValueError("Extended prompt did not increase the token count")
            self.idle()
            case = {
                "phase": "semantic",
                "kind": kind,
                "tokens": expected_tokens,
                "base_tokens": tokens,
                "expected_answer": fact,
                "calibration_requests": calibration,
                "count_request": count_index,
                "accepted": False,
                "request_index": len(self.journal.record["requests"]),
            }
            self.journal.record["cases"].append(case)
            response, index = self.post(
                "/v1/chat/completions",
                {
                    "model": self.model,
                    "messages": query,
                    "max_tokens": self.args.semantic_max_tokens,
                    "temperature": 0,
                    "top_p": 1,
                },
            )
            case["request_index"] = index
            choice = response["choices"][0]
            answer = (choice["message"].get("content") or "").strip()
            case.update(answer=answer, finish_reason=choice.get("finish_reason"))
            case["cached_tokens"] = validate_usage(
                response["usage"],
                expected_tokens,
                "cold" if kind == "cold" else "reuse",
            )
            if answer != fact or choice.get("finish_reason") != "stop":
                raise RuntimeError(
                    "Exact-answer check failed; inspect the preserved response"
                )
            self.idle(drain=True)
            case["accepted"] = True
            self.journal.save()
            print(
                json.dumps(
                    {
                        key: case[key]
                        for key in (
                            "phase",
                            "kind",
                            "tokens",
                            "answer",
                            "cached_tokens",
                            "accepted",
                        )
                    }
                ),
                flush=True,
            )

    def run(self):
        self.journal.record["models_response"] = json.loads(self.control("/v1/models"))
        models = self.journal.record["models_response"]["data"]
        if self.model is None:
            if len(models) != 1:
                raise ValueError("Specify --model when the API serves multiple models")
            self.model = models[0]["id"]
        if self.model not in {row["id"] for row in models}:
            raise ValueError("Requested model is absent from /v1/models")
        self.journal.record["model"] = self.model
        self.idle(drain=True, stable=3)
        for size in SIZES:
            self.prefill(size, "warmup", 0)
        self.semantic(8192, reuse=True)
        self.semantic(16384, reuse=False)
        self.semantic(32768, reuse=False)
        self.journal.save()
        for sample in range(REPEATS):
            for size in SIZES:
                self.prefill(size, "measured", sample)
        self.journal.record["summary"] = summarize(self.journal.record["cases"])


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", required=True)
    p.add_argument(
        "--model", help="Served model ID; inferred only if /v1/models lists one model"
    )
    p.add_argument(
        "--conditions",
        type=Path,
        help="Optional operator-supplied sources/settings JSON",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--label",
        required=True,
        help="Operator label, not evidence that a feature executed",
    )
    p.add_argument(
        "--ready-confirmed",
        action="store_true",
        help="Attest that all-rank model warmup completed",
    )
    p.add_argument(
        "--exclusive-window",
        action="store_true",
        help="Attest that this client owns the inference window",
    )
    p.add_argument("--request-timeout", type=float, default=180)
    p.add_argument("--idle-timeout", type=float, default=60)
    p.add_argument(
        "--api-key-env",
        help="API-key environment variable; its value is never journaled",
    )
    p.set_defaults(semantic_max_tokens=SEMANTIC_MAX_TOKENS)
    return p


def main():
    args = parser().parse_args()
    if not args.ready_confirmed or not args.exclusive_window:
        raise ValueError(
            "Execution requires completed model warmup and an exclusive inference "
            "window, attested by both flags"
        )
    if any(
        not math.isfinite(value) or value <= 0
        for value in (args.request_timeout, args.idle_timeout)
    ):
        raise ValueError("Timeouts must be finite and positive")
    # Reject credentials before creating a journal that includes the command.
    args.base_url = normalize_base_url(args.base_url)
    conditions, conditions_sha = load_conditions(args.conditions)
    clock = time.get_clock_info("perf_counter")
    record = {
        "schema": "glm-prefill-checkpoints-reproduction/v1",
        "qualification": "bounded semantic/cache smoke checks and TTFT observations",
        "execution_status": "running",
        "label": args.label,
        "run_id": uuid.uuid4().hex,
        "started_at_utc": utc_now(),
        "command": sys.argv,
        "base_url": args.base_url,
        "operator_attestations": {
            "all_rank_warmup_complete": True,
            "exclusive_window": True,
        },
        "conditions": conditions,
        "conditions_sha256": conditions_sha,
        "conditions_provenance": "operator_supplied"
        if conditions is not None
        else "not_supplied",
        "feature_activation": {
            "status": "not_verified",
            "source": "separate runtime evidence required",
        },
        "model_quality_qualified": False,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "measured_protocol_source_sha256": MEASURED_PROTOCOL_SOURCE_SHA256,
        "clock": {
            name: getattr(clock, name)
            for name in ("implementation", "resolution", "monotonic", "adjustable")
        },
        "settings": {
            "sizes": list(SIZES),
            "repeats": REPEATS,
            "inference_concurrency": 1,
            "prefill_output_tokens": 1,
            "semantic_max_tokens": SEMANTIC_MAX_TOKENS,
        },
        "requests": [],
        "cases": [],
        "idle_checks": [],
        "limitations": [
            "Synthetic exact-answer/cache checks do not establish full model quality "
            "or numerical equivalence.",
            "TTFT includes API/client/network time; it is not GPU kernel time.",
            "Idle snapshots and a serial client cannot exclude outside requests; "
            "the operator controls exclusivity.",
            "Source/settings metadata and completed warmup are operator attestations, "
            "not remotely verified.",
            "Feature activation is not checked; collect request-associated per-rank "
            "dispatch logs separately before attributing timings to an optimization.",
            "The measured protocol used an external activation gate between semantic "
            "checks and timing; this standalone client omits that collection step.",
            "Positive reuse proves a cache hit, not every checkpoint destination "
            "or persistent-cache tier.",
        ],
    }
    journal = Journal(args.output, record)
    try:
        PrefillChecks(args, journal).run()
        record["execution_status"] = "passed"
    except Exception as exc:
        record["execution_status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["finished_at_utc"] = utc_now()
        journal.save()


if __name__ == "__main__":
    main()
