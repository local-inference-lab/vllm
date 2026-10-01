# MXFP4-CSF and NVFP4-CSF

These formats compress FP4 block-scale tensors losslessly. Weight nibbles,
original scale bytes, calibration values and other tensors are preserved.
The B12X backend keeps scales compressed between layers and reconstructs
the needed scales into shared GPU scratch before expert computation.

Use `--quantization mxfp4_csf --load-format mxfp4_csf` for supported Kimi-K3
or DeepSeek-V4.1-Flash MXFP4 containers. Use
`--quantization nvfp4_csf --load-format nvfp4_csf` for GLM-5.3-Flash or
Qwen3.8-Flash-Next NVFP4 containers. Automatic loading does not select these readers;
set the load format explicitly.

MXFP4 uses a one-bit unsigned offset from a row's base byte. NVFP4 uses a
four-bit offset. Both preserve out-of-interval scale bytes as exceptions.
They retain their source arithmetic: MXFP4-CSF uses BF16 expert activations;
NVFP4-CSF uses native calibrated FP4 expert activations.

## GLM-5.3-Flash on two 96 GiB Blackwell GPUs

Supply a local serving directory containing source tokenizer metadata and
`config.json`. Its `quantization_config` must contain `quant_method` equal
to `nvfp4_csf`, `format_version` equal to 1, an absolute `checkpoint_root`,
and the unchanged source ModelOpt configuration in
`source_quantization_config`. The root points to the compressed artifact's
manifest and tensor files; it must be visible inside the runtime container.

Run in an environment with matching vLLM/B12X CSF support:

```bash
vllm serve /models/GLM-5.3-Flash-NVFP4-CSF/serve \
  --quantization nvfp4_csf --load-format nvfp4_csf \
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
graphs leave enough memory. The loader supports tensor parallel execution
with pipeline parallel size 1, without expert/data parallelism or ubatching.
Separate concurrent execution streams need separate decoded-scale scratch.
Leave `VLLM_B12X_MOE_FP4_FORCE_A16=0` (the default) to retain the source
NVFP4 activation arithmetic.

## Qwen3.8-Flash-Next on two 96 GiB Blackwell GPUs

Use the same serving-directory metadata contract as GLM. Main routed
experts retain calibrated four-bit activations; MTP, vision, shared experts
and the PLE embedding table retain their source precision.

```bash
VLLM_PLE_TABLE_MEMORY=ram vllm serve /models/Qwen3.8-Flash-Next-NVFP4-CSF/serve \
  --quantization nvfp4_csf --load-format nvfp4_csf \
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
supports TP1 and TP2 in this reader; TP4 does not satisfy the 64-channel
local alignment requirement.

## Compatibility

Install matching vLLM and B12X CSF support. vLLM owns the checkpoint manifest,
model tensor names and file access; B12X receives tensor sources for TP slicing
and GPU preparation. This API boundary does not change checkpoint contents,
serving flags or native expert arithmetic.

Readers require `lil-mxfp4-csf-checkpoint/1` or
`lil-nvfp4-csf-checkpoint/1` and their `.mxfp4_csf_*` or `.nvfp4_csf_*`
tensor components. Predecessor X4T/LSC names, schemas and suffixes are not
accepted. Migrate the checkpoint metadata, headers and receipts, verify
the reconstructed source hashes, and update the serving configuration
before removing a predecessor. Migration preserves compressed payload
values; it does not requantize the model.

Distribute compressed weights separately from an uncompressed checkpoint.
Consumers without a CSF loader can restore the original files. Including
both original and compressed scale tensors in every download removes the
storage benefit.
