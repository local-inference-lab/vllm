 docker run -d \
    --name abyssal-abjuration-611a842 \
    --network host \
    --ipc host \
    --gpus '"device=2,3,4,5,6,7,8,9"' \
    --security-opt label=disable \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v /root/.cache/huggingface:/root/.cache/huggingface \
    -v /mnt:/mnt \
    -v /models:/models \
    -v /cache:/cache \
    -v /root/bench-results:/root/bench-results \
    -e MODEL=/models/GLM-5.1-NVFP4-MTP-NVFP4 \
    -e PORT=8000 \
    -e VLLM_USE_V2_MODEL_RUNNER=1 \
    -e VLLM_USE_B12X_MOE=1 \
    -e VLLM_USE_B12X_FP8_GEMM=1 \
    -e VLLM_USE_B12X_SPARSE_INDEXER=1 \
    -e VLLM_USE_FLASHINFER_SAMPLER=1 \
    -e VLLM_ENABLE_PCIE_ALLREDUCE=1 \
    -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
    -e VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=64KB \
    -e VLLM_NCCL_SO_PATH=/opt/libnccl-local-inference.so.2.30.4 \
    -e LD_PRELOAD=/opt/libnccl-local-inference.so.2.30.4 \
    -e B12X_DENSE_SPLITK_TURBO=1 \
    -e B12X_W4A16_TC_DECODE=1 \
    -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
    -e NCCL_IB_DISABLE=1 \
    -e NCCL_P2P_LEVEL=SYS \
    -e NCCL_PROTO=LL,LL128,Simple \
    -e OMP_NUM_THREADS=16 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    voipmonitor/vllm:abyssal-abjuration-611a842 \
    bash -lc '
  set -euo pipefail
  unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS
  export PYTHONPATH="/opt/vllm${PYTHONPATH:+:${PYTHONPATH}}"
  cd /opt/vllm
  exec /opt/vllm/.venv/bin/python -m vllm.entrypoints.cli.main serve "${MODEL}" \
    --served-model-name GLM-5.1 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 1 \
    --decode-context-parallel-size 1 \
    --dcp-comm-backend ag_rs \
    --dcp-kv-cache-interleave-size 1 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --load-format fastsafetensors \
    --async-scheduling \
    -cc.pass_config.fuse_allreduce_rms=True \
    --gpu-memory-utilization 0.94 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 64 \
    --max-cudagraph-capture-size 64 \
    --quantization modelopt_fp4 \
   --attention-backend B12X_MLA_SPARSE \
    --moe-backend b12x \
    --kv-cache-dtype fp8 \
    --tool-call-parser glm47 \
    --enable-auto-tool-choice \
    --reasoning-parser glm45 \
    --hf-overrides "{\"index_topk_pattern\":\"FFSFSSSFSSFFFSSSFFFSFSSSSSSFFSFFSFFSSFFFFFFSFFFFFSFFSSSSSSFSFFFSFSSSFSFFSFFSSS\"}"
  '
