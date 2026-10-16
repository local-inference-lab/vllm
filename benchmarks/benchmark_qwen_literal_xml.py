# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired CPU replay of identical checkpoint-token streams; no generation variance."""

import argparse
import dataclasses
import gc
import importlib.util
import json
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path

from transformers import AutoTokenizer

from vllm.parser.engine.streaming_parser_engine import StreamingParserEngine
from vllm.parser.qwen3 import qwen3_config


def replay(engine, chunks):
    engine.reset()
    for text_chunk, token_chunk in chunks:
        engine.feed(text_chunk, token_chunk)
    engine.finish()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--baseline-engine",
        type=Path,
        required=True,
        help="Unmodified streaming_parser_engine.py from the target base commit",
    )
    cli.add_argument(
        "--tokenizer", required=True, help="Locally cached tokenizer or path"
    )
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--cpu", type=int, default=None)
    args = cli.parse_args()
    spec = importlib.util.spec_from_file_location(
        "baseline_parser_engine", args.baseline_engine
    )
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    affinity = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {args.cpu if args.cpu is not None else affinity[-1]})
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    call = (
        "<tool_call>\n<function=lookup>\n<parameter=key>\n"
        "hello\n</parameter>\n</function>\n</tool_call>"
    )
    workloads = {
        "prose": (
            (
                "The deployment keeps its existing model, context, "
                "memory, and tool settings.\n"
            )
            * 80,
            False,
        ),
        "code": (
            "```python\n"
            + ('def lookup(key):\n    return {"value": key, "ok": True}\n\n' * 40)
            + "```\nDONE",
            False,
        ),
        "real_tool": (call, False),
        "reasoning_then_tool": (
            "I will inspect the requested key and call the available lookup function. "
            * 80
            + "</think>"
            + call,
            True,
        ),
        "quoted_xml_then_tool": (
            "Here is an XML example:\n```xml\n"
            + call
            + "\n```\nNow calling the tool.\n"
            + call,
            False,
        ),
    }
    rows = []
    rng = random.Random(42)
    for name, (text, thinking) in workloads.items():
        ids = tokenizer.encode(text, add_special_tokens=False)
        for size in [1, 8, 32]:
            chunks = [
                (
                    tokenizer.decode(ids[i : i + size], skip_special_tokens=False),
                    ids[i : i + size],
                )
                for i in range(0, len(ids), size)
            ]
            configs = {
                "baseline": dataclasses.replace(
                    qwen3_config(thinking=thinking),
                    validate_tool_preamble=False,
                    guard_literal_tool_markers=False,
                ),
                "patched": qwen3_config(thinking=thinking),
            }
            engines = {
                "baseline": baseline.StreamingParserEngine,
                "patched": StreamingParserEngine,
            }
            instances = {
                kind: engines[kind](configs[kind], tokenizer) for kind in engines
            }

            for kind in engines:
                for _ in range(5):
                    replay(instances[kind], chunks)
            start = time.perf_counter()
            replay(instances["patched"], chunks)
            duration = time.perf_counter() - start
            repeats = max(5, min(200, int(0.06 / max(duration, 0.000001))))
            times = {"baseline": [], "patched": []}
            gc.collect()
            gc.disable()
            try:
                for _ in range(9):
                    order = list(engines)
                    rng.shuffle(order)
                    for kind in order:
                        start = time.process_time_ns()
                        for _ in range(repeats):
                            replay(instances[kind], chunks)
                        times[kind].append(
                            (time.process_time_ns() - start) / repeats / 1000
                        )
            finally:
                gc.enable()
            baseline_time = statistics.median(times["baseline"])
            patched_time = statistics.median(times["patched"])
            row = {
                "workload": name,
                "tokens": len(ids),
                "chunk_tokens": size,
                "repeats_per_round": repeats,
                "rounds": 9,
                "baseline_us_per_request": baseline_time,
                "patched_us_per_request": patched_time,
                "extra_us_per_token": (patched_time - baseline_time) / len(ids),
                "relative_change_pct": 100 * (patched_time / baseline_time - 1),
                "baseline_tokens_per_s": len(ids) * 1e6 / baseline_time,
                "patched_tokens_per_s": len(ids) * 1e6 / patched_time,
                "samples_us_per_request": times,
            }
            rows.append(row)
            print(
                name,
                size,
                "old_us",
                round(baseline_time, 1),
                "patched_us",
                round(patched_time, 1),
                "delta%",
                round(row["relative_change_pct"], 1),
                flush=True,
            )
    report = {
        "method": (
            "Identical predecoded checkpoint token chunks; baseline and patched "
            "engines loaded together; parser reset/feed/finish timed. Construction "
            "and tokenizer loading/encoding/decoding excluded. Five warmups, nine "
            "randomized paired rounds, median process CPU time, one pinned core. "
            "GC disabled during timing. Shared-host activity can affect results."
        ),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "rows": rows,
    }
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
