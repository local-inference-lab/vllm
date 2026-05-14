#!/usr/bin/env bash
# Warm b12x/CUTE DSL, Triton, TorchInductor and CUDA graph runtime caches for
# the GLM-5.1 NVFP4 MTP vLLM server before exposing it to real traffic.
#
# This intentionally uses a unique prompt prefix so prefix caching does not hide
# prefill work. The goal is to trigger kernel specialization/JIT, not to measure
# model quality.

set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5263}"
MODEL="${MODEL:-GLM-5}"
PROMPT_FILE="${PROMPT_FILE:-/root/testLuke5.txt}"
MAX_TOKENS_SHORT="${MAX_TOKENS_SHORT:-32}"
MAX_TOKENS_LONG="${MAX_TOKENS_LONG:-1}"
MAX_LONG_CHARS="${MAX_LONG_CHARS:-500000}"
WARMUP_MAX_PASSES="${WARMUP_MAX_PASSES:-4}"
WARMUP_DECODE_TOKENS="${WARMUP_DECODE_TOKENS:-1,8,64}"
WARMUP_PREFILL_CHARS="${WARMUP_PREFILL_CHARS:-8000,64000,256000,500000}"
TIMEOUT="${TIMEOUT:-900}"

python3 - "$HOST" "$PORT" "$MODEL" "$PROMPT_FILE" "$MAX_TOKENS_SHORT" "$MAX_TOKENS_LONG" "$MAX_LONG_CHARS" "$WARMUP_MAX_PASSES" "$WARMUP_DECODE_TOKENS" "$WARMUP_PREFILL_CHARS" "$TIMEOUT" <<'PY'
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

(
    host,
    port,
    model,
    prompt_file,
    max_tokens_short,
    max_tokens_long,
    max_long_chars,
    warmup_max_passes,
    warmup_decode_tokens,
    warmup_prefill_chars,
    timeout,
) = sys.argv[1:]
base = f"http://{host}:{port}"
max_tokens_short = int(max_tokens_short)
max_tokens_long = int(max_tokens_long)
max_long_chars = int(max_long_chars)
warmup_max_passes = int(warmup_max_passes)
decode_tokens = [int(x) for x in warmup_decode_tokens.split(",") if x]
prefill_chars = [min(int(x), max_long_chars) for x in warmup_prefill_chars.split(",") if x]
deadline = time.time() + int(timeout)
cache_roots = [
    Path("/root/.cache/cutlass_dsl"),
    Path("/root/.cache/triton"),
    Path("/root/.cache/torchinductor"),
    Path("/root/.cache/vllm"),
    Path("/cache/jit"),
]


def post_json(path, payload, timeout_s=None):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_ready():
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base + "/v1/models", timeout=5) as resp:
                if resp.status == 200:
                    return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"server did not become ready: {base}")


def chat(prompt, max_tokens, label):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    t0 = time.time()
    try:
        out = post_json("/v1/chat/completions", payload, timeout_s=None)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{label} failed: HTTP {e.code}: {body[:2000]}") from e
    dt = time.time() - t0
    usage = out.get("usage") or {}
    print(
        f"[warmup] {label}: {dt:.2f}s "
        f"prompt_tokens={usage.get('prompt_tokens')} "
        f"completion_tokens={usage.get('completion_tokens')}"
    )


def load_long_prompt():
    path = Path(prompt_file)
    if path.exists():
        text = path.read_text(encoding="utf-8").rstrip("\n")
        if len(text) > max_long_chars:
            text = text[:max_long_chars]
        return text
    # Fallback that is long enough to trigger chunked prefill specialization.
    block = (
        "This is a synthetic cache warmup document. It is intentionally long, "
        "repetitive, and semantically unimportant. "
    )
    text = block * ((max_long_chars // len(block)) + 1)
    return text[:max_long_chars]


def snapshot_cache_paths():
    paths = set()
    for root in cache_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                paths.add(str(path))
    return paths


wait_ready()
base_long_prompt = load_long_prompt()
previous_paths = snapshot_cache_paths()

for pass_idx in range(1, warmup_max_passes + 1):
    stamp = time.time_ns()
    print(f"[warmup] pass {pass_idx}/{warmup_max_passes} starting")

    # Cover decode/MTP and CUDA graph batch-size related paths.
    for tokens in decode_tokens:
        chat(
            (
                f"Runtime cache warmup decode pass={pass_idx} "
                f"tokens={tokens} unique={stamp}. Reply concisely."
            ),
            tokens,
            f"decode-{tokens}",
        )

    # Cover chunked prefill and sparse NSA extend-logits variants. Use unique
    # prefixes so prefix caching does not hide the actual prefill work.
    for chars in prefill_chars:
        body = base_long_prompt[:chars]
        long_prompt = (
            f"UNIQUE_RUNTIME_CACHE_WARMUP_{stamp}_{pass_idx}_{chars}\n"
            f"{body}\n\n"
            "Answer with the single word done."
        )
        chat(long_prompt, max_tokens_long, f"prefill-{chars}-chars")

    current_paths = snapshot_cache_paths()
    new_count = len(current_paths - previous_paths)
    print(f"[warmup] pass {pass_idx} new_cache_files={new_count}")
    if new_count == 0:
        print(f"[warmup] cache stable after pass {pass_idx}")
        break
    previous_paths = current_paths
else:
    print(f"[warmup] reached max passes ({warmup_max_passes}); cache may still grow for unseen shapes")
PY
