# GLM Spark MTP precision contract

The serialized Spark checkpoint contains exactly eleven MXFP8 MTP projection
matrices and NVFP4 routed-expert weights with BF16 activations. The online
QAD recipe implements that selection and those precision contracts. It does
not establish identical quantized bytes or identical model outputs.

## Conditions and evidence

**Qualified:** a direct audit of the safetensors headers, quantization config
and checked-in projection manifest for the following immutable checkpoints:

- Spark: `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark`,
  revision `a608241037e4c2565356bff7ca293f2133888f88`.
- QAD source: `local-inference-lab/GLM-5.3-Flash-NVFP4`, branch `mtp-bf16`,
  revision `cfd47bd7680e68408924df09b179d5bed25b2ae9`.

The MTP prefix is `model.language_model.layers.45`. Tensor dtypes, shapes,
config groups and manifest hash are retained in the
[audit receipt](glm53_spark_mtp_precision_results.json).
This audit reads headers and configuration; serving qualification is recorded
separately in the [online-loading report](glm53_spark_online_validation.md).

## Projection selection

Names below are relative to the MTP prefix. Eight attention/indexer projections
and three shared-expert projections have E4M3 weight tensors and `[1, 32]`
MXFP8 scale blocks in Spark. Its config declares dynamic group-32 FP8 input
activations. All eleven corresponding QAD source matrices are BF16.

| Projection | Spark weights | QAD source weights |
| --- | --- | --- |
| `mlp.shared_experts.down_proj` | MXFP8 | BF16 |
| `mlp.shared_experts.gate_proj` | MXFP8 | BF16 |
| `mlp.shared_experts.up_proj` | MXFP8 | BF16 |
| `self_attn.indexer.weights_proj` | MXFP8 | BF16 |
| `self_attn.indexer.wk` | MXFP8 | BF16 |
| `self_attn.indexer.wq_b` | MXFP8 | BF16 |
| `self_attn.kv_a_proj_with_mqa` | MXFP8 | BF16 |
| `self_attn.kv_b_proj` | MXFP8 | BF16 |
| `self_attn.o_proj` | MXFP8 | BF16 |
| `self_attn.q_a_proj` | MXFP8 | BF16 |
| `self_attn.q_b_proj` | MXFP8 | BF16 |

## Routed experts and unchanged tensors

Spark's `group_w4a16_nvfp4_mtp_routed_experts` config selects the entire MTP
`mlp.experts` module. Its 288 experts each have gate, up and down matrices:
864 packed U8 weight tensors, E4M3 group-16 block scales and FP32 global scales.
The config has no quantized-input declaration, and the expert tensors have no
`input_scale` entries. The corresponding serving contract is NVFP4 W4A16.
The QAD source contains the 864 expert matrices in BF16; the generated
`nvfp4_a16` target converts them during loading without quantizing activations.

The MTP `eh_proj`, router, norms and head remain outside this conversion.
The complete manifest also selects 520 main-model matrices. Strict matching
rejects a missing projection or an incomplete fused projection group.

**Implemented:** the generated configuration preserves this exact target set,
loads the QAD source and performs conversion in memory. It writes no checkpoint.
**Unsupported by this audit:** byte parity with Spark, matching Spark's
calibration data or broad model-quality equivalence. The optional main-model
`w13` activation-scale policy is a separate choice: it shares gate/up ranges
and retains down ranges per expert; it is unnecessary for BF16 MTP activations.

Use the [complete launch guide](glm53_spark_online.md) to reproduce the recipe.
