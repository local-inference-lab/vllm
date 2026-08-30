# GLM-5.3-Flash D16 single-container recipe

Build vLLM from this branch with the repository CUDA image, ensuring
FlashInfer 0.6.18 or newer is installed. Install LMCache 0.5.4, apply
`../lmcache-d16-overlay`, rebuild its CUDA extension from the pinned source
commit named there, and use that image as `BASE_IMAGE`:

```bash
docker build \
  --build-arg BASE_IMAGE=glm53-jovian-lmcache:local \
  -f docker/glm53-flash/single-container/Dockerfile \
  -t glm53-d16:local docker/glm53-flash
```

The model must be the unchanged `zai-org/GLM-5.3-Flash` FP8 checkpoint.
Its selective quantization configuration remains authoritative: MoE experts
are FP8 while attention, KDA, and other excluded modules stay BF16.

The qualified 4× RTX PRO 6000 Blackwell Max-Q launch is:

```bash
docker run --rm --restart=no --gpus all \
  --shm-size=32g --ipc=host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e CUTE_DSL_ARCH=sm_120a \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=1 \
  -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
  -e GLM53_LMCACHE_L1_SIZE_GB=48 \
  -e GLM53_LMCACHE_CHUNK_SIZE=9216 \
  glm53-d16:local \
  serve zai-org/GLM-5.3-Flash \
  --served-model-name GLM-5.3-Flash \
  --tensor-parallel-size 4 \
  --decode-context-parallel-size 4 \
  --cp-kv-cache-interleave-size 4 \
  --mamba-cache-mode align \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --enable-chunked-prefill \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --quantization modelopt_mixed \
  --attention-backend B12X \
  --linear-backend b12x \
  --moe-backend humming \
  --load-format instanttensor \
  --max-model-len 1048576 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 32768 \
  --kv-cache-memory-bytes 2952790016 \
  --max-cudagraph-capture-size 32 \
  --compilation-config '{"cudagraph_mode":"FULL"}' \
  --speculative-config \
  '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"humming","attention_backend":"B12X"}' \
  --kv-transfer-config \
  '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"127.0.0.1","lmcache.mp.port":5555}}'
```

The supervisor is PID 1. It creates a mode-0700 broker directory shared by
both children, starts LMCache first, gates vLLM on LMCache health, bounds vLLM
readiness, forwards SIGTERM/SIGINT to both child process groups, and kills and
reaps either sibling when the other exits. LMCache persistence and reload are
not part of this recipe.

LMCache is always launched with `--separate-object-groups`. Separation is
mandatory for this hybrid recipe so full attention history and the one-chunk
Mamba state use separate object semantics. The recipe depends on the
exact-boundary and sparse-transfer updates in #525 and #526.

The supervisor makes `--enable-prompt-tokens-details` mandatory so gateways
such as Bifrost can report `usage.prompt_tokens_details.cached_tokens`. This
changes cache accounting in API responses only; it does not change caching
behavior.

LMCache defaults to `WARNING` through `LMCACHE_LOG_LEVEL`. vLLM defaults to a
packaged logging config that keeps the general `vllm` namespace at `WARNING`
while restoring the periodic request metrics and speculative-decoding metrics
at `INFO`. An explicit `VLLM_LOGGING_CONFIG_PATH` or `VLLM_LOGGING_LEVEL`
preserves the operator's vLLM logging choice without injecting the packaged
config. Explicit `LMCACHE_LOG_LEVEL` values are also preserved.

Immediately after vLLM readiness, the supervisor makes one bounded
`GET /metrics` request to the readiness host and port. It reports the
`vllm:cache_config_info` KV capacity as tokens, max-context concurrency, block
size, and KV dtype. Concurrency falls back to capacity divided by an explicit
`--max-model-len` when the metric omits it. Other labels are not logged.
Unavailable, non-200, malformed, or oversized metrics produce one warning and
do not interrupt serving.

The image also installs startup-only vLLM fences around scheduler-realistic
warmup stages and before FULL graph capture. They keep TP4/DCP4/MTP3 ranks
aligned during capture without making steady-state serving synchronous.
