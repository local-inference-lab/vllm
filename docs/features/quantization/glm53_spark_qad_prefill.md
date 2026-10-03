# GLM-5.3-Flash: serialized Spark versus QAD with online MXFP8

The layer-maximum w1/w3 input-scale option closes the measured long-prefill
throughput gap between Spark and online QAD in this TP2 configuration. Against
the mean latency of two Spark controls, QAD with shared w1/w3 differs by
+0.175% on the generated benchmark prompts and
-0.054% on WikiText prompts. These are bounded measurements
with dynamic GPU clocks, not a claim of an exact hardware-independent tie.

Sharing w1/w3 improves QAD by 4.122% on the generated prompts
and 1.601% on WikiText in the same no-MTP run configuration.
The previously reported 1.119% gain was measured on WikiText with MTP enabled.
It was not a measurement of the generated-prompt benchmark's approximately
4.2% Spark gap. Comparing those percentages directly was unjustified.

## Behavior and qualification

- Implemented: the existing loader setting
  VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE=w13 replaces gate/up input scales
  with their layer maximum and retains per-expert down input scales. No
  serving implementation changed during this investigation.
- Qualified: all five serving loads, identical prompt-token sequences,
  uncached completions, retained kernel-selection choices and completed
  profiler captures on the pinned image and GPU pair below.
- Research-only: the throughput comparison, individual profiler captures,
  and the diagnostic all-scales policy. No model-quality evaluation was
  repeated for this comparison.
- Unsupported by these measurements: one universal percentage for other
  text distributions, prompt lengths, concurrency, chunks, MTP settings
  or hardware, and quality equivalence after changing activation ranges.

## Artifacts and conditions

Image:
  ghcr.io/local-inference-lab/vllm@sha256:ed1fe72b2c733492ebeda264f9c49906c677b901ff135997c1abfb042425e287
Installed vLLM:
  58d05bc7626dd87ee401155e9b52e6c6fd92bd5d
Installed B12X:
  66dba3aca0ab32e197bacb2eea120a067de767cc
QAD source:
  local-inference-lab/GLM-5.3-Flash-NVFP4, branch mtp-bf16,
  cfd47bd7680e68408924df09b179d5bed25b2ae9
Spark source:
  local-inference-lab/GLM-5.3-Flash-NVFP4-Spark,
  a608241037e4c2565356bff7ca293f2133888f88

Frank1 GPUs 12/13 are RTX PRO 6000 Blackwell Workstation 96 GB cards.
Power limits are 600 W and loaded memory clocks are 16365 MHz. Graphics
clocks remain dynamic; per-load clock/power/temperature ranges are in
comparison.json and the raw telemetry.csv files. Existing services on
GPUs 0–9 remain active. Task measurements run serially on the same pair.

Every load uses TP2/DCP1, no MTP, FP8 KV with 3 GiB/rank, a 3072-token
prefill budget, a 65536-token model limit, eight maximum sequences,
one parallel prefill, OMP_NUM_THREADS=1, B12X MoE/linear/attention backends,
main NVFP4-CSF, and FULL_AND_PIECEWISE CUDA graphs with sizes
1/2/4/8/16/32/64. Source model directories are read-only mounts at the same
container path. Installed serving packages have no Python source overlays.
Only QAD loads add the strict online MXFP8 projection map; no MTP quantization
is invoked. Complete commands, environment values and revisions are in each
load's launch.json.

## Method

The generated prompts use the architecture-text generator in version 0.7.5
of /opt/lil/bench/llm_decode_bench.py from the pinned image. The calibration
ratio and prefix from the Spark input-scale control are frozen, then tokenized
once. The long generated prompts contain exactly 32322 tokens. WikiText
prompts contain exactly 32768 tokens, formed from the deterministic shuffled
document-token sequences used in the activation-scale evaluation.

Ten long prompts and four short prompts from each family are replayed with
identical token hashes across every model load. Four long warmups precede
measurement. The two text families alternate to reduce correlation with
temperature drift. Every completion has a unique cache salt, temperature 0,
one output token and matching usage/server counters. Prefix caching is enabled
for recurrent-page alignment, but measured cached tokens are zero throughout.
Timing uses server request-prefill latency, excluding tokenization and HTTP
serialization. Client wall time is retained independently.

Spark is measured before and after the three QAD policies. For each prompt,
the reference is the arithmetic mean of its two Spark latencies. Reported
paired changes are medians of the per-prompt throughput ratios. The two Spark
controls drift by -0.141% on generated prompts and
-0.135% on WikiText. Multiple prompts within one load do
not establish subpercent reproducibility across independent restarts.

The persistent B12X preparation cache is shared across loads. Existing
configuration entries must remain unchanged, while different legal input-scale
contracts may add their own queries. Before/after snapshots are retained,
and serving does not mutate the selected configurations. Nonuniform input
scales are never falsely declared shareable to force a faster kernel.

## Long-prefill measurements

All values are prompt tokens per second; medians over ten prompts per cell.

| Model and activation-scale policy | Generated, 32322 tokens | WikiText, 32768 tokens |
| --- | ---: | ---: |
| Spark, first load | 13,233.4 | 12,797.9 |
| QAD online MXFP8, source input scales | 12,723.8 | 12,583.3 |
| QAD online MXFP8, layer-maximum w1/w3 | 13,244.8 | 12,779.9 |
| QAD online MXFP8, layer-maximum w1/w3 and w2 | 13,247.7 | 12,779.6 |
| Spark, reload control | 13,209.7 | 12,777.8 |

Paired change against the bracketing Spark controls:

| Model and activation-scale policy | Generated | WikiText |
| --- | ---: | ---: |
| QAD online MXFP8, source input scales | -3.724% | -1.651% |
| QAD online MXFP8, layer-maximum w1/w3 | +0.175% | -0.054% |
| QAD online MXFP8, layer-maximum w1/w3 and w2 | +0.226% | -0.053% |

The source-policy gap remains approximately four percent on generated text
and approximately one to two percent on WikiText. The shared-w1/w3 policy
reaches Spark throughput in both workloads within the observed variation.
The all-scales diagnostic provides no material additional gain. It is not
necessary to replace the per-expert w2 activation ranges to close this gap.

## Short-prefill measurements

The generated short prompts contain 2161 tokens; WikiText contains 2048.
These results must not be described as the same input length or extrapolated
from the long-prefill percentage. Source calibration can be faster at this
shorter length. All four samples per cell are retained.

| Model and activation-scale policy | Generated, 2161 tokens | WikiText, 2048 tokens |
| --- | ---: | ---: |
| Spark, first load | 12,062.7 | 11,885.5 |
| QAD online MXFP8, source input scales | 12,324.5 | 12,112.7 |
| QAD online MXFP8, layer-maximum w1/w3 | 12,073.9 | 11,915.7 |
| QAD online MXFP8, layer-maximum w1/w3 and w2 | 12,100.7 | 11,926.5 |
| Spark, reload control | 12,072.2 | 11,899.8 |

## GPU attribution

B12X has a prefill implementation that separates the NVFP4 gate/up and down
matrix work from routing and input quantization. It requires uniform gate/up
input scales. In B12X 66dba3aca, _nvfp4_materialization_eligible in
b12x/moe/fused_moe/_tuning.py checks shared_input_scales and nvfp4_share_input,
along with the numeric, tile and shape constraints. The QAD source ranges
are expert-specific and cannot satisfy this contract. Spark's ranges already
do; the QAD w13 option makes its gate/up range eligible too.

Separate profiling requests confirm the actual dispatch:

- QAD source: one MoEDynamicKernelSilu compute kernel per routed layer.
- Spark, QAD w13 and QAD all: a routing/input front end followed by
  Nvfp4MaterializedPhase1Kernel for gate/up and
  Nvfp4MaterializedPhase2Kernel for down.

There are 462 routed-layer invocations in each long capture: 42 main MoE
layers across 11 prefill chunks. The faster path therefore changes the
execution of substantial expert matrix work; the gain is not merely the
cost of reading a smaller activation-scale tensor.

GPU 12 / TP rank 0 summed kernel times, milliseconds, for one generated prompt:

| Policy | MoE total | Dense GEMMs | MLA attention | KDA attention | CSF reconstruction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spark, first load | 691.43 | 501.83 | 340.89 | 109.55 | 89.47 |
| QAD online MXFP8, source input scales | 780.35 | 504.35 | 341.18 | 109.13 | 89.66 |
| QAD online MXFP8, layer-maximum w1/w3 | 682.96 | 502.73 | 340.85 | 110.05 | 89.65 |
| QAD online MXFP8, layer-maximum w1/w3 and w2 | 682.68 | 503.25 | 341.11 | 110.02 | 89.67 |
| Spark, reload control | 689.54 | 502.16 | 340.73 | 109.66 | 89.48 |

Both ranks' complete summaries and raw traces are retained. Kernel durations
can overlap, and profiler overhead changes host scheduling and communication
waits. These sums locate the work; they are not substitutes for unprofiled
throughput or an exact additive decomposition of wall time. In the first
Spark/source-QAD pair on rank 0, the MoE sum rises by about 89 ms while the
instrumented GPU span rises by about 96 ms. Dense MXFP8 GEMMs, attention and
CSF reconstruction remain similar. QAD w13 restores the same prefill path
and approximately the same expert-compute duration as Spark.

The speedup is workload-dependent. The materialized gate/up and down kernels
process expert-major tiles. Their work depends on routed-token distribution
and partially occupied tiles; prompt families and lengths are different here.
No claim is made that a specific change in router probabilities was isolated.
The direct same-token comparison is sufficient to establish that shared w13
closes the Spark/QAD gap for both measured workloads.

Online MXFP8 weight encoding runs during model loading. It is not called
for each prefill request. The missing long-prefill performance in this
comparison is attributable primarily to the NVFP4 expert execution contract,
not recurring online weight encoding or slower compressed-scale decoding.

## Practical conclusion

For the measured long-prefill configuration, the setting is:

  VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE=w13

Keep w2 per expert. The default value 0 retains source calibration.
The w13 option preserves trained packed NVFP4 weights but changes activation
quantization; throughput parity does not imply identical model outputs.
The separate [activation-scale quality evaluation](glm53_qad_activation_scales.md) remains
the evidence for that tradeoff. This investigation does not recommend
silently changing the default or broadening the online projection selection.

## Evidence and reproduction

Work directory:
  /root/vllm/kimi/glm53-prefill-attribution-20261003
Data directory:
  /data/trellis-quant/glm53-prefill-attribution-20261003

- comparison.json: validated summary, raw timing rows, profiler categories,
  telemetry ranges and evidence hashes.
- prompts.json: complete frozen token sequences and hashes.
- <case>/launch.json, results.json, telemetry.csv, server.log,
  selection-before.json and selection-after.json: serving evidence.
- <case>/profile-<family>.json and profile/: separate request receipts,
  raw per-rank traces and machine-readable kernel summaries.
- compare_prefill.py: serving/measurement controller. It requires free
  GPUs 12/13 and preserves existing checkpoint files.
- summarize_traces.py and summarize_comparison.py: raw-data analysis.
- write_report.py: renders this report from comparison.json.

All serving containers are configured with restart=no and stopped after
measurement. The comparison controller skips completed case directories
and refuses to overwrite incomplete directories; reproductions must use
a separate data directory. No checkpoint or serving source was modified.

The [machine-readable comparison](glm53_spark_qad_prefill_results.json) retains
the paired samples, telemetry summaries, profiler categories and source hashes.
The model-tested container release also distributes the reproduction scripts
and compact timing evidence as `glm53-spark-qad-prefill-evidence.tar.gz`:
[release assets](https://github.com/local-inference-lab/blackwell-llm-docker/releases/tag/karmic-kraken-beta-a2d687e9120bba38fbf3c6b43ab928df486fbcc70baa3628ec2293f889e29691).
Its SHA256 is
`fed855430536b3ba511b8d04d32b4339ceb5269064c8695dfd9393a61501d343`.
