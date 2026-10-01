# MXFP4-CSF and NVFP4-CSF

These formats compress FP4 block-scale tensors losslessly. Weight nibbles,
original scale bytes, calibration values and other tensors are preserved.
The B12X backend keeps scales compressed between layers and reconstructs
the needed scales into shared GPU scratch before expert computation.

Use `--quantization mxfp4_csf --load-format mxfp4_csf` for supported Kimi-K3
or DeepSeek-V4.1-Flash MXFP4 containers. Use
`--quantization nvfp4_csf --load-format nvfp4_csf` for GLM-5.3-Flash NVFP4
containers. Automatic loading does not select these container readers;
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
NVFP4-CSF does not accept `VLLM_B12X_MOE_FP4_FORCE_A16=1` because that changes
the source activation arithmetic.

## Compatibility

`kimi_x4t` and `exact_mxfp4` quantization names remain supported. The
`exact_mxfp4` load format remains an alias for the MXFP4 container reader.
The `nvfp4_lsc` quantization and load names remain supported for existing
serving configurations. Original schema identifiers, tensor suffixes and
checkpoint files remain unchanged; renaming does not require re-encoding.

Distribute compressed weights separately from an uncompressed checkpoint.
Consumers without a CSF loader can restore the original files. Including
both original and compressed scale tensors in every download removes the
storage benefit.
