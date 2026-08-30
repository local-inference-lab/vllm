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
