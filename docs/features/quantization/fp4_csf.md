# MXFP4-CSF and NVFP4-CSF

These formats compress FP4 block-scale tensors losslessly. Weight nibbles,
original scale bytes, calibration values and other tensors are preserved.
The B12X backend keeps scales compressed between layers and reconstructs
the needed scales into shared GPU scratch before expert computation.

Use `--quantization mxfp4_csf --load-format mxfp4_csf` for supported Kimi-K3,
DeepSeek-V4.1-Flash, DeepSeek-V4-Flash or DeepSeek-V4-Flash-Vision-Exp MXFP4
containers. NVFP4 compressed scales use ordinary ModelOpt mixed-precision
recipes and the standard safetensors loader.

MXFP4 uses a one-bit unsigned offset from a row's base byte. NVFP4 uses a
four-bit offset. Both preserve out-of-interval scale bytes as exceptions.
Expert activations follow each model's native B12X policy. DeepSeek-V4.1-Flash
and DeepSeek-V4-Flash MXFP4-CSF quantize expert activations to MXFP8 (W4A8)
unless `VLLM_B12X_MOE_FP4_FORCE_A16=1` requests BF16 activations. Kimi-K3
MXFP4-CSF uses BF16 expert activations. NVFP4-CSF uses native calibrated FP4
expert activations.

## GLM-5.3-Flash on two 96 GiB Blackwell GPUs

The checkpoint's `config.json` identifies compressed scales in each affected
expert recipe. Other recipes, including MTP and vision, retain their own
quantization settings:

```json
{
  "quantization_config": {
    "quant_method": "modelopt",
    "quant_algo": "MIXED_PRECISION",
    "quantized_layers": {
      "model.language_model.layers.3.mlp.experts": {
        "quant_algo": "NVFP4",
        "group_size": 16,
        "weight_scale_encoding": "csf"
      }
    }
  }
}
```

Each compressed `weight_scale` is stored as `weight_scale.nvfp4_csf_fixed`
(uint8) and `weight_scale.nvfp4_csf_exceptions` (uint32) tensors in ordinary
safetensors shards. The standard shard index locates all tensors. No CSF
manifest, checkpoint-root setting, or custom load format is required.

Run in an environment with matching vLLM/B12X CSF support:

```bash
vllm serve /models/GLM-5.3-Flash-NVFP4-CSF \
  --quantization modelopt_mixed --load-format safetensors \
  --tensor-parallel-size 2 --dtype bfloat16 \
  --moe-backend b12x --linear-backend b12x --attention-backend B12X \
  --additional-config '{"glm53_kda_decode_backend":"auto","kda_prefill_backend":"b12x"}' \
  --language-model-only --kv-cache-dtype fp8 \
  --kv-cache-memory-bytes 1073741824 \
  --max-model-len 8192 --max-num-seqs 4 --max-num-batched-tokens 1024 \
  --cudagraph-capture-sizes 1 2 4 --max-cudagraph-capture-size 4 \
  --host 0.0.0.0 --port 8000
```

This example reserves 1 GiB/GPU for KV cache. Increase
`--kv-cache-memory-bytes` only when the model, execution buffers and CUDA
graphs leave enough memory. The quantization method supports tensor parallel execution
with pipeline parallel size 1, without expert/data parallelism or ubatching.
Separate concurrent execution streams need separate decoded-scale scratch.
Leave `VLLM_B12X_MOE_FP4_FORCE_A16=0` (the default) to retain the source
NVFP4 activation arithmetic.

## Qwen3.8-Flash-Next on two 96 GiB Blackwell GPUs

Use the same per-recipe scale encoding as GLM. Main routed
experts retain calibrated four-bit activations; MTP, vision, shared experts
and the PLE embedding table retain their source precision.

```bash
VLLM_PLE_TABLE_MEMORY=ram vllm serve /models/Qwen3.8-Flash-Next-NVFP4-CSF \
  --quantization modelopt_mixed --load-format safetensors \
  --tensor-parallel-size 2 --pipeline-parallel-size 1 --dtype bfloat16 \
  --moe-backend b12x --linear-backend b12x --no-enable-flashinfer-autotune \
  --mm-encoder-tp-mode data --mamba-cache-mode align \
  --no-enable-prefix-caching --enable-chunked-prefill \
  --kv-cache-dtype bfloat16 --kv-cache-memory-bytes 2147483648 --block-size 16 \
  --max-model-len 8192 --max-num-seqs 4 --max-num-batched-tokens 1024 \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4]}' \
  --reasoning-parser qwen3 --host 0.0.0.0 --port 8000
```

`VLLM_PLE_TABLE_MEMORY` is unset by default and leaves placement to the
model configuration. The explicit `ram` setting above requires about 26.8
GiB of host RAM for both TP ranks' table payloads, plus loading and process
memory. `VLLM_PLE_TABLE_MEMORY=disk` selects file-backed table access; its
performance depends on storage and is not represented by the RAM example.
The example reserves 2 GiB/GPU for KV cache. Qwen's intermediate dimension
supports TP1 and TP2 with this scale preparation; TP4 does not satisfy the 64-channel
local alignment requirement.

## DeepSeek-V4-Flash on two 96 GiB Blackwell GPUs

This also applies to DeepSeek-V4-Flash-Vision-Exp. The serving directory
contains the source tokenizer metadata and a `config.json` whose
`quantization_config` keeps the source block-FP8 fields, including
`weight_block_size` equal to `[128, 128]`, and sets `quant_method` to
`mxfp4_csf`, `format_version` to 1 and an absolute `checkpoint_root`.

The `deepseek_v4_flash` container compresses the routed-expert scales of the
43 target layers. Attention, dense and shared-expert weights retain block
FP8. MTP/DSpark draft layers (`mtp.*`), the vision tower, the aligner and the
image embeddings retain their source tensors and precision. A `config.json`
with a vision tower selects the multimodal model.

```bash
vllm serve /models/DeepSeek-V4-Flash-MXFP4-CSF/serve \
  --quantization mxfp4_csf --load-format mxfp4_csf \
  --tensor-parallel-size 2 --moe-backend b12x --attention-backend B12X \
  --block-size 256 --max-num-seqs 8 --max-num-batched-tokens 4096 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5}' \
  --tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4 \
  --tool-call-parser deepseek_v4 --host 0.0.0.0 --port 8000
```

The reader supports TP1, TP2, TP4 and TP8 with pipeline parallel size 1,
without expert/data parallelism or ubatching.

## Compatibility

Install matching vLLM and B12X CSF support. For NVFP4, the standard model
loader delivers scale components through the model's normal weight-loading
hooks. ModelOpt selects the CSF expert method from `weight_scale_encoding`;
that method validates and slices TP-local tensors. B12X receives those weights
and compressed CPU scale planes through `prepare_weights`, which uploads and
rearranges them for the selected kernels.

MXFP4's separate loader requires `lil-mxfp4-csf-checkpoint/1`, a manifest,
and `.mxfp4_csf_*` tensor components. It does not accept predecessor X4T/LSC
names, schemas, or suffixes.

Runtimes without the corresponding scale decoder require uncompressed scale
tensors. Including both representations in one download removes the storage
benefit.
