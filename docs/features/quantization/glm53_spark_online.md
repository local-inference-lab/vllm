# GLM-5.3-Flash QAD with Spark projection selection

Load the QAD `mtp-bf16` checkpoint on two 96 GB RTX PRO 6000 Blackwell GPUs,
converting exactly the Spark attention/shared projection set to MXFP8 during
loading. QAD expert weights, routers and calibrated per-expert activation scales
remain from the QAD checkpoint. This reduces the loaded model's memory use while
retaining its QAD source.

Use the `karmic-kraken-beta` container release containing vLLM PR #965. Docker,
NVIDIA Container Toolkit, `curl`, and sufficient host memory to load the source
checkpoint are required. Choose two unused GPU indices in `GLM_QAD_GPUS` and run:

```bash
set -euo pipefail
GLM_QAD_IMAGE=ghcr.io/local-inference-lab/vllm:karmic-kraken-beta
GLM_QAD_GPUS=0,1
GLM_QAD_DIR="$PWD/glm-qad-spark"
mkdir -p "$GLM_QAD_DIR"

curl --fail --location --retry 3 \
  https://raw.githubusercontent.com/local-inference-lab/vllm/integration/karmic-kraken-beta/examples/features/quantization/glm53_spark_online.py \
  --output "$GLM_QAD_DIR/glm53_spark_online.py"
curl --fail --location --retry 3 \
  https://raw.githubusercontent.com/local-inference-lab/vllm/integration/karmic-kraken-beta/examples/features/quantization/glm53_spark_targets.json \
  --output "$GLM_QAD_DIR/glm53_spark_targets.json"

docker pull "$GLM_QAD_IMAGE"
docker run --rm --entrypoint /opt/venv/bin/python \
  -v "$GLM_QAD_DIR:/recipe" "$GLM_QAD_IMAGE" \
  /recipe/glm53_spark_online.py --output-dir /recipe/config

docker run -d --name glm-qad-spark-tp2 --restart no --init \
  --gpus "\"device=$GLM_QAD_GPUS\"" --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864:67108864 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$GLM_QAD_DIR/config:/recipe:ro" \
  -e OMP_NUM_THREADS=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_B12X_MOE_FP4_CSF=1 -e VLLM_B12X_MOE_FP4_FORCE_A16=0 \
  -e VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE=0 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=1 -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
  -e VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE=2048 \
  -e VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE=auto \
  -e VLLM_GLM53_KDA_GATE_SIDE_STREAM=1 -e VLLM_DISABLE_SHARED_EXPERTS_STREAM=0 \
  -e VLLM_GLM53_L2_PREFETCH=1 -e VLLM_LM_HEAD_A16=1 \
  -e VLLM_GLM53_MTP_DRAFT_HEAD=bf16 \
  -e VLLM_MTP_NVFP4_LM_HEAD=0 -e VLLM_MXFP8_LM_HEAD=0 \
  -e VLLM_GLM53_VISION_MXFP8=0 -e VLLM_GLM53_FP8_DENSE=0 \
  -e VLLM_GLM53_ONLINE_DENSE_MXFP8=0 \
  --entrypoint /bin/bash "$GLM_QAD_IMAGE" -lc '
unset NCCL_GRAPH_FILE
exec /opt/venv/bin/python -m vllm.entrypoints.cli.main \
  serve local-inference-lab/GLM-5.3-Flash-NVFP4 \
  --revision cfd47bd7680e68408924df09b179d5bed25b2ae9 \
  --served-model-name GLM-QAD-Spark \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 2 --pipeline-parallel-size 1 --decode-context-parallel-size 1 \
  --quantization modelopt_mixed --load-format safetensors --dtype bfloat16 \
  --moe-backend b12x --linear-backend b12x --attention-backend B12X \
  --additional-config "{\"glm53_kda_decode_backend\":\"auto\",\"kda_prefill_backend\":\"b12x\"}" \
  --quantization-config "$(cat /recipe/main.json)" \
  --speculative-config "$(cat /recipe/mtp.json)" \
  --kv-cache-dtype fp8 --kv-cache-memory-bytes 4294967296 \
  --gpu-memory-utilization 0.93 \
  --block-size 256 --mamba-cache-mode align \
  --max-model-len 65536 --max-num-batched-tokens 3072 --max-num-seqs 8 \
  --max-parallel-prefills 1 --enable-chunked-prefill --enable-prefix-caching \
  --recurrent-checkpoint-policy request_boundaries \
  --compilation-config "{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\"}" \
  --cudagraph-capture-sizes 1 2 4 8 16 32 64 --max-cudagraph-capture-size 64 \
  --no-enable-flashinfer-autotune --reasoning-parser glm45'

docker logs -f glm-qad-spark-tp2
```

The configuration generator refuses to overwrite an existing `config/`
directory. To resume a stopped container with the same configuration, run
`docker start glm-qad-spark-tp2`. Stop it with `docker stop glm-qad-spark-tp2`;
the example disables automatic restart. To change command-line settings, stop
and remove that named container, then repeat the serving command with the
existing configuration directory.

The pinned source revision belongs to the `mtp-bf16` branch. vLLM converts
weights in memory on each model load. It does not write a quantized checkpoint
or alter the Hugging Face source files.

## Loaded precision and memory

| Component | Loaded representation |
| --- | --- |
| QAD main routed experts | Source NVFP4, with source per-expert activation scales |
| Selected attention and shared-expert projections | Online MXFP8, using exactly the Spark selection |
| MTP routed experts from `mtp-bf16` | Online NVFP4 weights with BF16 activations |
| Routers, norms, first three dense FFNs, vision, embeddings and head | Source representation |

The adjacent projection manifest selects exactly 531 BF16 matrices: 520 in the
main model and 11 in MTP. The configuration generator creates `main.json` for
`--quantization-config` and `mtp.json` for `--speculative-config`. MXFP8 scale
compression is not enabled.

With MTP disabled, converting the selected main projections reduced loader
allocation from 92.071 to 88.548 GB per GPU in the TP2 measurement, approximately
7.047 GB saved across the pair. Total VRAM also includes KV cache, CUDA graphs
and runtime buffers. Conditions and image identities are in the
[validation report](glm53_spark_online_validation.md).

`strict_targets` defaults to `false` in vLLM; both generated target maps set it
to `true`. A missing projection, incomplete fused group, or independently
enabled online quantizer causes startup to fail. Keep the head, vision and
dense quantization flags shown above disabled when reproducing this selection.

The recipe matches the Spark **layer selection** while preserving QAD's weights
and per-expert activation calibration. It does not produce byte parity with
Spark's serialized weights. Expert-specific QAD activation scales prevent reuse
of one quantized input across experts. The optional shared-scale setting below
enables that optimization with a change to activation quantization.

## Optional shared gate/up activation scale

`VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE` defaults to `0`, which preserves the
checkpoint's expert-specific activation calibration. Keep `0` for routine use.
To share one gate/up input scale per main routed layer while retaining each
expert's down-input scale, replace its environment line in the Docker command
with:

```bash
  -e VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE=w13 \
```

This takes the layer maximum of the existing gate/up scales during loading.
The trained NVFP4 weight codes and weight scales remain unchanged; the loader
recomputes the runtime coefficients for the selected activation scale. The MTP
experts in this recipe use BF16 activations and need no shared FP4 input range.
Use `w13` to retain per-expert down scales; `1` or `all` would share them too.
This setting is separate from lossless CSF compression of **weight** scales.

On two RTX PRO 6000 Blackwell GPUs at 600 W and 16365 MHz memory clocks, the
shared setting improved fixed 32K prefill by about 1.1% after the reload control,
with no decode improvement. The output distributions changed; a small English
sample showed NLL 1.08448 to 1.08563 and did not establish equivalent quality.
The [activation-scale report](glm53_qad_activation_scales.md) records exact
image/configuration identities, repetitions and uncertainty. A maximum of
expert-specific ranges is a heuristic, not a new calibration pass.

## MTP, cache and prefill settings

MTP uses three speculative tokens by default in the generator. Change this
with `--num-speculative-tokens 1` when generating a separate configuration
directory, or omit `--speculative-config` from the serve command to disable MTP.
Without MTP, only the 520 main-model MXFP8 targets are loaded.

The 4 GiB cache allocation is per GPU; the example sets a 65536-token model
limit. It does not provide eight simultaneous 64K contexts or establish
million-token capacity. Set a lower limit with `--max-model-len 32768` if needed;
increase `--kv-cache-memory-bytes` only when the available VRAM permits it.

Set `OMP_NUM_THREADS=1` before startup, as in the example. This avoids excessive
CPU thread creation while loading two tensor-parallel workers, especially when
starting several servers on the same host. The container otherwise selected 64
threads per worker during loading before reducing that count for inference.

Set `--max-num-batched-tokens 3072` explicitly, as in the example, for prefill
throughput. With TP2/DCP1, no MTP and a 32K prompt, the same GPU pair measured
about 12.1K tok/s with BF16 selected projections and 12.7K tok/s with online
MXFP8. Those measurements use RTX PRO 6000 Blackwell GPUs at 600 W and 16365 MHz
memory clocks; exact image and configuration identities are in the report.

Reducing the budget to `--max-num-batched-tokens 1024` produces more model
executions per prompt and prevents the MXFP8 backend from entering its large
prefill regime. Both online MXFP8 and the serialized Spark checkpoint measured
about 6K tok/s at that setting, while BF16 projections measured 7.8K. Online
weight encoding runs during loading, not during prefill.

## FP4 scale compression

`VLLM_B12X_MOE_FP4_CSF` defaults to `0`. The example sets it to `1` to compress
main-model NVFP4 expert weight scales losslessly. The generated MTP configuration
inherits that setting. In the tested serialized TP2 execution, MTP shares the
main model's reconstruction buffers, so compressing its scales also saves memory.

The generator's `--mtp-scale-compression` default is `inherit`; `csf` and `native`
are explicit overrides. To compare native MTP scales, generate a separate directory:

```bash
docker run --rm --entrypoint /opt/venv/bin/python \
  -v "$GLM_QAD_DIR:/recipe" "$GLM_QAD_IMAGE" \
  /recipe/glm53_spark_online.py \
  --output-dir /recipe/config-native-mtp --mtp-scale-compression native
```

Mount `$GLM_QAD_DIR/config-native-mtp` at `/recipe` in the serving command to use
that directory's `main.json` and `mtp.json`.
`speculative_config.moe_scale_compression` defaults to `null`, inheriting the
main kernel setting and environment default. A main-model override is
`--kernel-config '{"moe_scale_compression":"native"}'` or `"csf"`. If adding that
option inside the single-quoted Docker command above, escape its JSON like the
`--compilation-config` example.

These controls govern compression during native weight loading; they do not
reinterpret a checkpoint already stored with CSF scales. Compressing only the
draft while the main model uses native scales requires an additional
reconstruction buffer and can increase memory. The shared-buffer saving applies
when both use CSF with matching shapes.
