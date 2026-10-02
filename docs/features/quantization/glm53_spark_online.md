# GLM-5.3-Flash online quantization with Spark projection selection

Use two 96 GB RTX PRO 6000 Blackwell GPUs and a vLLM build containing online
`nvfp4_a16` and strict quantization targets. B12X must provide native FP4 scale
compression. Run the following from the vLLM repository inside that environment:

```bash
python examples/features/quantization/glm53_spark_online.py \
  --output-dir glm-spark-config

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
VLLM_USE_V2_MODEL_RUNNER=1 \
VLLM_B12X_MOE_FP4_CSF=1 \
VLLM_B12X_MOE_FP4_FORCE_A16=0 \
VLLM_ENABLE_PCIE_ALLREDUCE=1 VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE=2048 \
VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE=auto \
VLLM_GLM53_KDA_GATE_SIDE_STREAM=1 VLLM_DISABLE_SHARED_EXPERTS_STREAM=0 \
VLLM_GLM53_L2_PREFETCH=1 VLLM_LM_HEAD_A16=1 \
VLLM_GLM53_MTP_DRAFT_HEAD=bf16 \
VLLM_MTP_NVFP4_LM_HEAD=0 VLLM_MXFP8_LM_HEAD=0 \
VLLM_GLM53_VISION_MXFP8=0 VLLM_GLM53_FP8_DENSE=0 \
VLLM_GLM53_ONLINE_DENSE_MXFP8=0 \
vllm serve local-inference-lab/GLM-5.3-Flash-NVFP4 \
  --revision cfd47bd7680e68408924df09b179d5bed25b2ae9 \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 2 --pipeline-parallel-size 1 \
  --quantization modelopt_mixed --load-format safetensors --dtype bfloat16 \
  --moe-backend b12x --linear-backend b12x --attention-backend B12X \
  --additional-config '{"glm53_kda_decode_backend":"auto","kda_prefill_backend":"b12x"}' \
  --quantization-config "$(cat glm-spark-config/main.json)" \
  --speculative-config "$(cat glm-spark-config/mtp.json)" \
  --kv-cache-dtype fp8 --kv-cache-memory-bytes 3221225472 \
  --gpu-memory-utilization 0.93 \
  --block-size 256 --mamba-cache-mode align \
  --max-model-len 8192 --max-num-batched-tokens 1024 --max-num-seqs 8 \
  --max-parallel-prefills 1 --enable-chunked-prefill --enable-prefix-caching \
  --recurrent-checkpoint-policy request_boundaries \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --cudagraph-capture-sizes 1 2 4 8 16 32 64 --max-cudagraph-capture-size 64 \
  --no-enable-flashinfer-autotune --reasoning-parser glm45
```

The configuration generator creates a directory and refuses to overwrite one
that already exists. The pinned source revision belongs to the `mtp-bf16`
branch. vLLM reads the checkpoint and converts weights in memory; it does not
write a quantized checkpoint.

The adjacent projection manifest selects exactly 531 BF16 matrices from the
Spark recipe: 520 in the main model and 11 in MTP. These use MXFP8 weights and
the selected MXFP8 linear backend. MTP routed experts use NVFP4 weights with
BF16 activations. The LM head, embeddings, vision tower, routers, norms,
convolutions and the first three dense feed-forward layers retain their source
representation. Existing main-model NVFP4 experts retain their activation
recipe. MXFP8 scale compression is not enabled.

`strict_targets` defaults to `false` in vLLM; both generated target maps set it
to `true`. A missing projection, incomplete fused group, or independently
enabled online quantizer causes startup to fail. Keep the head, vision and
dense quantization flags shown above disabled when reproducing this selection.
The recipe matches the Spark **layer selection**; it does not claim byte parity
with Spark's serialized quantized weights.

`VLLM_B12X_MOE_FP4_CSF` defaults to `0`. The example sets it to `1` to compress
the main model's NVFP4 expert scales losslessly. The generated MTP configuration
inherits that setting. In the tested serialized TP2 configuration, MTP shares
the main model's reconstruction buffers, so compressing its scales also saves
memory. To compare native MTP scales,
generate a separate configuration directory:

```bash
python examples/features/quantization/glm53_spark_online.py \
  --output-dir glm-spark-native-mtp-config --mtp-scale-compression native
```

Use that directory's `main.json` and `mtp.json` in the same launch command.
`speculative_config.moe_scale_compression` defaults to `null`, which inherits
the main kernel setting and environment default. The main-model override is
`--kernel-config '{"moe_scale_compression":"native"}'` or `"csf"`.
These controls govern compression during native weight loading; they do not
reinterpret a checkpoint already stored with CSF scales.

The generator's `--mtp-scale-compression` default is `inherit`; `csf` and `native`
are explicit overrides. Compressing only the draft while the main model uses
native scales requires an additional reconstruction buffer and can increase
memory. The shared-buffer saving applies when both use CSF with matching shapes.

MTP uses three speculative tokens by default in the generator. Change this
with `--num-speculative-tokens 1`, or omit `--speculative-config` from the serve
command to disable MTP. The 3 GiB cache allocation is per GPU; it is a bounded
8K-context example, not a million-token capacity configuration.

The measured TP2 configuration trades lower weight memory and faster decode for
slower prefill. Keep the selected projections in BF16 if prefill throughput is
the priority; see the paired 4K and 32K measurements below before choosing.

Quality, memory, throughput and exact validation identities are recorded in
the [GLM online-weight validation report](glm53_spark_online_validation.md).
