# GLM-5.2 NVFP4 MLA KV outer scales

Per-layer calibration for the `nvfp4_ds_mla` KV cache writer
(`VLLM_NVFP4_MLA_SCALES_FILE`, format `nvfp4_ds_mla_outer_scale_v1`).

GLM-5.2's post-RMSNorm 512-dim `kv_c` latent spans a ~240x amplitude range
across layers. With the default outer scale of 1.0, shallow layers quantize
with E4M3 block scales at or below the subnormal floor, and shallow-layer KV
error is strongly amplified downstream. `s_l = max_abs(kv_c_normed) / (6*448)`
re-centers every layer so its largest block scale lands at the top of the
E4M3 range.

## Files

- `glm52-nvfp4-nf3-hybrid_mla_outer_scales_v1.json` — calibrated on
  `madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid` (K64R16), Salesforce/wikitext
  (wikitext-2-raw-v1, test), 2048-token context, TP4; per-layer envelope with
  an independent community capture of the same base model for shallow-layer
  headroom. `max_abs` per layer is included for auditability.

## Usage

```bash
VLLM_NVFP4_MLA_SCALES_FILE=/path/to/glm52-nvfp4-nf3-hybrid_mla_outer_scales_v1.json \
  ./serve-glm52.sh   # nvfp4_ds_mla + B12X_MLA_SPARSE only; inert otherwise
```

## Results (teacher-forced prefill KLD vs BF16 reference, 5 fresh boots each)

| KV config | mean +/- sd | max ctx (4x96GB) |
|---|---|---|
| fp8_ds_mla | 0.1263 +/- 0.0030 | 373k |
| nvfp4_ds_mla + scales, bf16 rope | 0.1345 +/- 0.0035 | 550k |
| nvfp4_ds_mla + scales, fp8 rope (`KV_FP8_ROPE=1`) | 0.1356 +/- 0.0054 | 600k+ |
| nvfp4_ds_mla, no scales, bf16 rope | 0.158 | 550k |
| nvfp4_ds_mla, no scales, fp8 rope | 0.168 | 600k+ |

Protocol: local-inference-lab/rtx6kpro `benchmarks/glm52-kld-evaluation.md`
(festr2 2026-07-08 reference logits, one fixed 2048-token window, 2047
positions, full 154,880 vocab, `KL(ref || candidate)`).

## Cache invalidation (required)

CuTeDSL folds the outer-scale multiply out of the kernel when `latent_scale`
traces at exactly 1.0. b12x builds without the identity/dynamic compile-spec
fact (see lukealonso/b12x PR "mla: split latent_scale identity/dynamic
compile-cache entries") replay stale identity cubins from persistent compile
caches, silently dropping the restore. Clear mounted b12x compile caches once
when enabling scales on such builds.
