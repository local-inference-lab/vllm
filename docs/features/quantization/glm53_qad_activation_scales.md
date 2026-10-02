# GLM QAD gate/up activation-scale sharing: validation

For the tested GLM-5.3-Flash QAD source, sharing the maximum gate/up input scale
across a routed-expert layer is a usable loading option. It preserves the trained
NVFP4 weight representation and each expert's down-projection input scale.
It provides a small prefill improvement; the measurements do not show a decode
improvement or establish equivalent model quality. Keep the checkpoint's
calibration as the default. Use the shared-scale option explicitly when testing
that tradeoff.

The [TP2 usage guide](glm53_spark_online.md) provides the complete online MXFP8
and NVFP4-MTP launch. This report evaluates the additional environment setting
`VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE=w13`; its default is `0`.

## Behavior, status and compatibility

- **Implemented:** `w13` replaces the loaded gate/up input activation scale with
  its layer maximum. B12X can quantize a token's common gate/up input once and
  reuse it across routed experts. The `w2` activation scale stays per expert.
- **Qualified:** both TP2 policies complete startup, C1/C8 decode and fixed
  prefill through 32K on the published image below. Loaded-scale invariants pass
  for all 42 routed layers on both ranks. Repeated full-logit captures are
  identical for both NVFP4 policies in the deterministic evaluation mode.
- **Research-only:** the short throughput comparison and 16-document English
  quality sample. They support a bounded comparison, not broad quality parity.
- **Unsupported by this evidence:** deterministic BF16 expert-activation
  reference scoring, exact parity with the serialized Spark model,
  million-token quality or capacity, and untested devices or parallelism.

This option changes activation quantization during inference. It does not
retrain QAD, re-encode the routed weights, export a checkpoint, or modify source
files. A layer maximum derived from existing expert-specific calibration is a
heuristic shared range, not a recalibration on a layer-wide activation corpus.
The [source model card](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4/blob/cfd47bd7680e68408924df09b179d5bed25b2ae9/README.md)
describes BF16-activation distillation followed by expert-specific inference
calibration. This supports treating per-expert ranges as an intentional source
configuration, not evidence of damaged QAD weights. The calibration algorithm
itself was not independently reproduced.

For each rank and routed layer, the audit verifies:

1. The original 288 reciprocal gate/up input scales become one scalar, exactly
   the minimum of those reciprocals: the reciprocal of the layer maximum.
2. Down-input scales, down runtime coefficients and both raw weight-global
   scale tensors remain bitwise identical.
3. Gate/up runtime coefficients are recomputed consistently. The kernel uses
   `weight_global_scale / reciprocal_input_scale`; those folded coefficients
   must change when the activation scale changes.
4. The online projection selection remains 520 source matrices fused into
   331 main-model linear modules per rank.

Do not copy the original folded gate/up coefficients over the shared-scale
preparation. That would change the represented multiplication incorrectly.

## Artifacts and conditions

| Artifact or condition | Identity or value |
| --- | --- |
| QAD source | `local-inference-lab/GLM-5.3-Flash-NVFP4@cfd47bd7680e68408924df09b179d5bed25b2ae9`, `mtp-bf16` branch |
| Performance image | `ghcr.io/local-inference-lab/vllm@sha256:ef547bdcf146e2c9b183e2fa0b82a3a01ada680765720d01a25c258b45a801a7` |
| Installed performance code | vLLM `58d05bc7626dd87ee401155e9b52e6c6fd92bd5d`, B12X `a77b3f85e5e2a81315ff4a90088912f187708005` |
| Hardware | Frank1, RTX PRO 6000 Blackwell Workstation 96 GB; performance on GPUs 12/13 |
| Clocks | 600 W limits, 16365 MHz loaded memory clocks, dynamic graphics clocks |
| Serving | TP2/DCP1, three MTP speculative tokens, online Spark-selected MXFP8 projections, main/draft NVFP4-CSF |
| Capacity | 4 GiB FP8 KV per rank, 3072-token prefill budget, eight sequences, 65536-token model limit |
| Host/runtime controls | `OMP_NUM_THREADS=1`, `FULL_AND_PIECEWISE` CUDA graphs, capture sizes 1/2/4/8/16/32/64 |

Performance uses the installed packages without Python source overlays.
Normalized launch commands differ only in container identity and the activation
scale policy. The same persistent B12X preparation cache retains all 115
configuration records across the accepted loads. Separate services on GPUs 0–9
remain active; these are not isolated-host measurements. No second task-owned
serving benchmark runs concurrently with a performance measurement.

## Throughput

Decode is the median of three complete 30-second sustained runs per policy.
C8 is aggregate output throughput across eight concurrent requests. Each run
uses approximately 78 prompt tokens and up to 2048 output tokens per request.

| Policy | C1 output tok/s | C8 output tok/s |
| --- | ---: | ---: |
| Source expert-specific calibration | 221.477 | 768.102 |
| Layer-maximum gate/up input scale | 219.161 | 768.673 |
| Shared/source change | −1.05% | +0.07% |

These short runs do not resolve a decode improvement. The complete accepted
three-repetition batches pass the error, looping, underfill and capacity checks.

Prefill uses identical prompt hashes and server prefill duration, with zero
cached tokens. The source policy is measured before and after the shared policy;
the second source load is a dedicated prefill control. Each paired comparison
normalizes the shared result by the geometric mean of its two source controls.

| Prompt | Source median tok/s | Shared median tok/s | Paired change | Prompt-bootstrap 95% interval |
| --- | ---: | ---: | ---: | ---: |
| 2K, four prompts | 11066.708 | 11212.735 | +0.812% | +0.575% to +1.038% |
| 8K, four prompts | 11916.979 | 12067.970 | +0.834% | +0.785% to +0.896% |
| 32K, ten prompts | 12114.033 | 12308.330 | +1.119% | +1.066% to +1.170% |

The displayed source median is from the first load; paired changes include the
second source load. At 32K, source-control throughput drifted upward by 0.907%.
The confidence intervals resample prompts within these particular loads, not
independent server restarts or clock states. They must not be interpreted as
subpercent reproducibility across systems. Quantization also changes routing,
so this is a serving-policy comparison rather than an isolated packing-kernel
speed measurement.

Three other decode loads are excluded in full after sampled queues triggered
`capacity_limited`: two shared-policy loads with 3 GiB KV and a source-policy
control with 4 GiB. Their unflagged repetitions are also excluded. The data do
not establish that extra KV eliminates the queue condition or that scale
sharing causes it. Raw samples and reasons remain in the results JSON.

## Quality

Both accepted policies retain the same QAD routed weights and online MXFP8
projections. Evaluation disables MTP and CUDA graphs, uses a 1024-token
scheduler budget and 2 GiB FP8 KV per rank, and enables deterministic dynamic
MoE reduction. Prefix caching remains enabled for aligned recurrent pages;
distinct request salts prevent prefix reuse. The per-expert policy uses GPUs
12/13 and the shared policy uses GPUs 14/15 on the same host.

The corpus contains 16 distinct 1024-token WikiText test prefixes selected with
seed 53145 from `Salesforce/wikitext`, `wikitext-2-raw-v1`, revision
`b08601e04326c79dfdd32d625aee71d232d685c3`. The input-suite SHA256 is
`dece045c0b507d3288d667e281a76b12cbe75c2e4afc16e91f26657e234e821a`.
Full FP32 logits cover all 154880 vocabulary entries; comparison normalizes in
FP64. KLD uses 16384 positions and NLL uses 16368 known successor tokens.

| Measurement | Result |
| --- | ---: |
| KLD: source activation policy to shared policy | 0.100458 nats/token |
| Document-bootstrap KLD 95% interval | 0.086818 to 0.114027 |
| Top-1 agreement | 90.570% |
| Source-policy next-token NLL | 1.084481 |
| Shared-policy next-token NLL | 1.085628 |
| Paired NLL increase | 0.001147 nats/token |
| Document-bootstrap NLL-difference 95% interval | −0.004441 to +0.005916 |
| Perplexity change | +0.115% |
| Repeated-forward KLD, two documents per accepted policy | Exactly 0 for both |

The output distributions change measurably. The mean held-out NLL change is
small and its interval includes zero; that is not proof of equivalent quality.
KLD measures a distribution change, not a percentage of wrong answers. NLL uses
the corpus's actual next tokens and does not depend on a model-generated
reference distribution.

BF16 expert-activation reference probes are excluded because they fail the
repeated-forward control: KLD 0.026257 with B12X and 0.021490 with independently
selected Marlin. The underlying cause is not isolated here. No KLD against
either reference is used to judge the two activation policies. The accepted
comparison instead uses the repeatable source NVFP4 policy and independent
ground-truth NLL. This does not measure error against the original BF16 teacher.

## Deterministic preparation correction

The quality runtime copies the installed vLLM package and adds two observational
call sites: loaded-weight auditing and complete prompt-logit capture. It also
binds the preparation correction in
[B12X PR #464](https://github.com/local-inference-lab/b12x/pull/464).

The correction captures `B12X_DYNAMIC_DETERMINISTIC_OUTPUT` in the immutable
preparation query before candidate selection. Otherwise an unspecified
`RoutingSpec` can admit an atomic NVFP4 split kernel while kernel compilation
reads the environment and requires deterministic reduction. The kernel's
unsupported combination remains rejected; preparation selects an eligible
implementation. Explicit routing overrides retain precedence. No kernel
arithmetic or source weight representation changes.

The minimal reproducer fails both implicit-default cases before correction;
all four explicit-override cases pass. After correction, 128 focused tests pass,
four CUDA tests skip in the CPU-only test container, and one GPU lifetime test
is deselected. Full-model NVFP4 captures then complete and repeat exactly.
The correction is merged into B12X `integration/karmic-kraken-beta` at
`66dba3aca0ab32e197bacb2eea120a067de767cc`. The bound preparation file SHA256 is
`f4a8c6a895eda7a9f876a215d55c056ffa0d35395a3c8d7262f6f2ded2f8349f`.
Performance numbers above use the unmodified published image, not this quality
overlay.

## Evidence and recommendation

The adjacent [results JSON](glm53_qad_activation_scales_results.json) contains
raw accepted timings, paired intervals, scale-contract checks, exclusions and
artifact hashes. Raw local evidence is retained on Frank1 at
`/data/trellis-quant/glm53-online-mxfp8-csf-20261002/`:

- `qad-layer-input-scale/results.json` is the combined result;
  `quality-qualified.json` identifies the accepted repeat controls.
- `serving/beta-qad-scale-{expert,w13}-kv4-tp2/` contains accepted launches,
  decode repetitions, telemetry and fixed prefill measurements.
- `serving/beta-qad-scale-expert-prefill-control-kv4-tp2/` is the prefill drift
  control; it has no accepted decode measurement.
- `serving/beta-qad-input-prepared-{expert,w13}/` contains quality launches,
  loaded-target/scale audits, full-logit chunk manifests and repeat captures.
- `runtime/beta-58d05-scale-audit-2/source-manifest.json` and
  `runtime/beta-58d05-deterministic-preparation/manifest.json` identify the
  observational runtime and preparation correction.

The reproducible controllers are under
`/root/vllm/kimi/glm53-online-mxfp8-csf-20261002/`: `verify_beta_qad_recipe.py`,
`run_qad_scale_quality.py`, `capture_glm_logits.py`, `compare_glm_logits.py` and
`summarize_qad_scale_comparison.py`. They preserve source checkpoint files and
stop their test containers after use.

Keep source calibration as the routine setting because the measured benefit is
small and the source ranges have a substantially broader calibration basis.
The optional shared maximum is suitable for targeted evaluation with unchanged
trained weights. A permanent shared-scale recipe should calibrate one gate/up
range per layer on representative data and retain per-expert down ranges, then
validate on independent data. This report does not claim that recalibration or
broader evaluation has been performed.
