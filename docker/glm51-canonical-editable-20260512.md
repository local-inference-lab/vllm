# GLM/Kimi Canonical Editable Image

This records the first working canonical editable vLLM image validated on
2026-05-12.

## Validated Image

```text
voipmonitor/vllm:glm51-canonical-editable-vllm1e118a8-b12x9436cb8-20260512
```

Validated source state:

```text
vLLM branch: codex/glm51-kimi-canonical-a16-upstream-20260511
vLLM commit: 1e118a830 Fix KV cache profiling config compatibility
B12X commit: 9436cb8 runtime: add GLM B12X A16 stability fixes
Base image: voipmonitor/vllm:glm51-canonical-githead-vllmdb22839-b12x9436cb8-20260512
```

The image is intentionally editable-installed from `/opt/vllm`:

```text
PYTHONPATH=/opt/vllm
vllm.__file__=/opt/vllm/vllm/__init__.py
vllm._C=/opt/vllm/vllm/_C.abi3.so
```

Keep `PYTHONPATH=/opt/vllm`; the base image still contains a
`site-packages/vllm` tree, and without `PYTHONPATH` Python can import the
wrong copy.

## Rebuild

Run from this repository, with a clean git tree:

```bash
MAX_JOBS=128 \
CMAKE_BUILD_PARALLEL_LEVEL=128 \
TORCH_CUDA_ARCH_LIST=12.0a \
scripts/build-glm51-canonical-editable-image.sh
```

The script copies `git archive HEAD` into `/opt/vllm`, runs:

```bash
/opt/venv/bin/python -m pip install -e . --no-deps --no-build-isolation -v
```

and commits a Docker image with:

```text
ENV PYTHONPATH=/opt/vllm
ENTRYPOINT ["/usr/local/bin/run-glm51-vllm"]
```

No extra Python dependencies are installed by this script. It depends on the
base image already containing the known-good CUDA 13, FlashInfer, communicator,
launcher, and B12X runtime stack.

## GLM DCP1 MTP Smoke Run

The validated runtime was launched with the existing GLM launcher defaults plus
these host-level settings:

```bash
docker run -d \
  --name glm51-dcp1-mtp-canonical-editable-b12x-a16off \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size 16g \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e PORT=5264 \
  -e DCP_SIZE=1 \
  -e TP_SIZE=8 \
  -e GLM51_DISABLE_MTP=0 \
  -e ATTENTION_BACKEND=B12X_MLA_SPARSE \
  -e MOE_BACKEND=b12x \
  -e VLLM_B12X_FORCE_MOE_A16=0 \
  -e KV_CACHE_DTYPE=fp8 \
  -e GPU_MEMORY_UTILIZATION=0.865 \
  -e MAX_NUM_BATCHED_TOKENS=8192 \
  -e MAX_NUM_SEQS=64 \
  -e MAX_CUDAGRAPH_CAPTURE_SIZE=256 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=1 \
  -e VLLM_PCIE_ALLREDUCE_BACKEND=cpp \
  -e VLLM_CPP_AR_1STAGE_NCCL_CUTOFF=56KB \
  -e VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS=0 \
  -e NCCL_PROTO=LL,LL128,Simple \
  -e HF_OVERRIDES='{"index_topk_pattern":"FFSFSSSFSSFFFSSSFFFSFSSSSSSFFSFFSFFSSFFFFFFSFFFFFSFFSSSSSSFSFFFSFSSSFSFFSFFSSS"}' \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v /cache/jit:/cache/jit \
  voipmonitor/vllm:glm51-canonical-editable-vllm1e118a8-b12x9436cb8-20260512
```

Smoke checks:

```bash
curl -sS http://127.0.0.1:5264/v1/models
docker exec glm51-dcp1-mtp-canonical-editable-b12x-a16off \
  bash -lc 'PYTHONPATH=/opt/vllm /opt/venv/bin/python -c "import vllm, vllm._C; print(vllm.__file__); print(vllm._C.__file__)"'
```

Expected startup signs:

```text
RTX6K NCCL residual-add fusion overlay imported.
Enabled custom fusions: act_quant, allreduce_rms
Using 'B12X' NvFp4 MoE backend
Application startup complete.
```
