# Reproducing the serving measurements

Status: **research-only**. The results cover bounded correctness/cache probes
and short performance observations. Use an otherwise working four-GB10 TP4/DCP4
GLM deployment. [Source reconstruction](source-reproduction.md) identifies public
commits that reproduce the measured runtime; it does not supply the private image.

## Model and tokenizer

Weights: [local-inference-lab/GLM-5.3-Flash-NVFP4-Spark](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark/tree/df116c4fb16b1d37ae43d2cfd624de26ffbc832e),
revision `df116c4fb16b1d37ae43d2cfd624de26ffbc832e`. All 16 candidate
container specifications mounted that snapshot. No separate tokenizer override
was configured; the tokenizer is supplied by the model snapshot.

Use BF16 activations, FP8 KV cache, TP4/DCP4/PP1, native MTP with three
speculative tokens, 24 GiB KV budget per rank, 16 maximum sequences,
1,048,576 maximum model tokens and 8,192 maximum batched tokens. Enable native
prefix caching with aligned recurrent checkpoints and retention interval zero.
Keep both split-page settings at 512 on every worker:
`VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE=512` and
`VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE=512`. Keep
`VLLM_GLM53_MHC_PREFILL_DIAGNOSTICS=1` in every comparison configuration.

| Configuration | VLLM_B12X_KDA_PREFILL_COALESCING | VLLM_GLM53_MHC_PREFILL_SHARD |
| --- | ---: | ---: |
| Neither | 0 | 0 |
| Coalescing only | 1 | 0 |
| mHC only | 0 | 1 |
| Both | 1 | 1 |

The measured deployment held fused SparkRing transport constant. A switch-based
cluster can test the same feature logic with its own working TP communicator,
but changing transport does not reproduce the absolute recorded timings.
Complete automatic model warmup and reserve exclusive inference access before
running either benchmark. Serving preparation is outside these client scripts.

## Decode

The measured client was a local merge of the public benchmark. Reconstruct its
exact bytes from the public parent and the adjacent patch:

```bash
git clone https://github.com/local-inference-lab/llm-inference-bench.git
git -C llm-inference-bench checkout bd88816e9e7bcc97e1bcfd954c3053528f31af69
# Run from this evidence directory, or substitute its absolute path.
git -C llm-inference-bench apply "$PWD/decode-benchmark.patch"
sha256sum llm-inference-bench/llm_decode_bench.py
```

Expected SHA-256:
`46ace1dad13c245807bc1b4ccf4ab6b90e95d12a99dd729ee24caa51034194c6`.
The patch contains Windows portability and display/hardware-monitor integration
used by the measured client. Hardware monitoring was disabled for these tests.

```bash
uv venv .benchmark-venv
uv pip install --python .benchmark-venv/bin/python httpx==0.28.1 rich==15.0.0
.benchmark-venv/bin/python llm-inference-bench/llm_decode_bench.py \
  --host http://SERVER --port PORT --model SERVED_MODEL_ID \
  --dcp-size 4 --temperature 1.0 --token-targeting exact \
  --kv-budget 524288 --display-mode plain --no-hw-monitor --skip-prefill \
  --concurrency 1,4 --contexts 8k,32k --max-tokens 1024 --duration 20 \
  --decode-warmup-seconds 5 --cell-warmup-timeout-seconds 180 \
  --output decode-results.json
```

Replace SERVER, PORT and SERVED_MODEL_ID with the deployment endpoint. Repeat
for each flag combination. The benchmark generates run-specific padding and
uses the server tokenizer for exact context sizes. Stochastic outputs need not
match token-for-token. Report raw output tok/s, mean MTP acceptance length,
and their quotient; retain reported server step rates separately.

## Prefill and cache checks

Run the standalone client from the vLLM checkout after model warmup:

```bash
.venv/bin/python benchmarks/glm_prefill_checkpoints.py \
  --base-url http://SERVER:PORT --model SERVED_MODEL_ID \
  --label coalescing-only --output prefill-results.json \
  --ready-confirmed --exclusive-window
```

The client preserves the measured protocol's prompt generator, exact tokenizer
calibration, zero-cache checks, streaming first-token timing and phase order:
three excluded shape warmups, five exact-answer/cache checks, then three cold
samples each at 8K/16K/32K. The synthetic record queries are generated in the
script; no private prompt corpus is needed. Repeated and extended prompts test
cache reuse. Sampling and token limits are encoded in the client.

The standalone client records feature activation as **not verified**. The
recorded four-configuration experiment separately gated timing on request-bound
all-rank checkpoint/mHC diagnostics. Collect those diagnostics as described in
the feature documentation before attributing a reproduced rate to either
optimization. The client neither accesses SSH nor starts/stops the model.
