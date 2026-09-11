# TrellisMX on Jovian Judgement — implementation draft

## September9 selected reference

The selected image and matched cache KLD are documented in [REFERENCE-20260909.md](REFERENCE-20260909.md). `compose.r27.yaml` pins that tested image with24slots, MTP3, TP4/DCP4, NCCL8 and selected collective cutoffs. `serve-r27.sh` supplies the same settings for source builds. This changes the serving recipe and companion B12X pin; it does not replace unrelated vLLM source.

This branch adds executable support code for the coupled
**GLM-5.3-Flash-TrellisMX-MXFP8, 17-K5 / 25-K4** routed overlay. It is a
review draft, **not a serving-qualified Jovian release**. No production
services, clocks or GPU jobs are changed by preparing this PR.

## Review map

- `vllm/utils/trellismx.py`: complete TP4 inventory, file sizes, design hashes,
  rate/header validation and rank-local payload hashing.
- `vllm/model_executor/layers/quantization/trellismx.py`: actual ModelOpt
  carrier loader ABI, native P8 dispatch, routed-storage release and Jovian
  warmup provider. There is no process-wide class/factory monkey patch.
- `modelopt.py`: explicit opt-in selection plus the text-only GLM MTP name
  alias. Dense, attention, shared experts, router and MTP stay with ModelOpt.
- Companion B12X fork: native P8 kernels, coupled transforms and internal
  runtime. JJ imports `b12x.moe._shared.trellismx`; no kernel dependency tree
  is vendored here. No encoder is needed or included.
- `tests/quantization/test_trellismx_*.py`: CPU manifest, loader and dispatch
  tests. Native runtime launches are not emulated as evidence of GPU closure.
- This directory: source-build Docker recipe, Compose and `serve.sh`.

The public opt-in is `VLLM_TRELLISMX_CHECKPOINT=/checkpoint`, while the
carrier retains `--quantization modelopt`. This is an explicit overlay
adapter, not a new standalone Transformers quantization checkpoint loader.
An absent or corrupt requested sidecar fails startup; it cannot fall back to
stock weights for that layer. Hot weight updates are unsupported.

## Download and build

The overlay must finish uploading before it is usable. Read its
`release-status.json`; the HF repo is private during upload verification.

```bash
hf download brandonmusic/GLM-5.3-Flash-TrellisMX-MXFP8 \
  --local-dir /your/models/trellismx
hf download local-inference-lab/GLM-5.3-Flash-NVFP4 \
  --revision 520de24eabf507659eaef7c70f14fd584527facc \
  --local-dir /your/models/glm53-flash-carrier

# From this PR checkout; builds current Jovian Python AND extension binaries.
bash examples/trellismx/build.sh
export MODEL_ROOT=/your/models/glm53-flash-carrier
export P8_CHECKPOINT_ROOT=/your/models/trellismx
docker compose -f examples/trellismx/compose.yaml up
```

The build is expensive and has **not been executed end-to-end for this PR**.
It uses the repository's normal Docker build rather than overlaying latest
Python on RC5's old native extensions. The addon pins the companion B12X fork
at `ba682a8308b3352c45a3dba457aadc2ea56d6d48`, based on current B12X
`6483963275dcf32eb2eec6d100e644d1ea647ed6`. P8, attention and PCIe now use
one B12X installation. Their combined serving compatibility needs device
validation; RC5 measurements do not qualify this source reconciliation.

The default recipe requests TP4/DCP1, MTP3, BF16 external activations with
native E4M3 P8 internal activations, B12X_MLA_SPARSE, NVFP4 MLA KV, graphs,
prefix caching, 8192 scheduler tokens and **1,000,000 maximum context**.
One-million-context startup, capacity and retrieval are **not tested** by
this PR. Port 8033 avoids production port 8000. No power or clock changes.
TP2, EP, LoRA, other architectures and other coupling laws are not supported
by this first adapter. It is not general ModelOpt replacement software.

The sidecars replace routed carrier tensors; they do not add another expert
ensemble. Both downloads are still required in this transitional layout.
The runtime frees only replaced routed parameters after successful native
loading. No claim is made that total download size equals logical model size.

## Results and boundaries

The table below is **historical campaign/RC5 evidence, not new-Jovian results**.
The coupled allocation averages **4.6587417643 routed stored bpw**, including
metadata. E4M3/UE8M0-32 describes native computation, not 8-bit stored weights.
P8 uses `mxf8f6f4`: twice NVFP4's MMA issue count for equal dimensions.

| Measurement | Historical result | Conditions |
| --- | --- | --- |
| Mean true-decode KLD | **0.0341811459**, BCa95 [0.0291483518, 0.0409784257] | 32 opened conditional-fit windows; historical B12X attention/NVFP4 MLA KV/native P8 E4M3, not fresh RC5/Jovian KLD |
| Uniform coupled K4 | 0.0369674524 | K4/K5 is 7.54% lower at a larger bit budget; not matched-size or matched-stock evidence |
| Estonia | 30/30 PASS | RC5 TP4/DCP1, MTP3, B12X attention/PCIe, NVFP4 MLA KV, native P8 E4M3, 4.65874 bpw; C10, reasoning max |
| LAVD | 22 EXACT + 6 NEAR + 2 token-cap truncations | Same RC5 regime; official 28/30; the two unfinished outputs are not proven correct |
| Hotel | 28 EXACT + 2 FAIL | Same RC5 regime; neither failure hit the token limit |
| Decode C1 / C2 / C4 | 178.336464 / 235.605573 / 300.874345 aggregate tok/s | Same RC5 model/backend regime, graphs, separate cache-disabled decode launch; one configuration run |
| Prefill 8k / 32k | 8063 / 7998 tok/s | Same checkpoint/compute family; prefix caching enabled |

Reasoning tests: 30 requests/profile, 100k output cap, no request errors.
Speed is from client `llm_decode_bench` JSON/logs, not rolling vLLM throughput;
four RTX PRO 6000 96GB, PCIe/no NVLink, 300W and +6000 memory offset.
Top-1 agreement and KLD p99 are not available in these receipts; do not invent
them or substitute p99 output length. The 32 CF windows are development data,
not an untouched final holdout. Protected confirmation logits stay unopened.

[Raw results and measurement receipts](https://github.com/brandonmmusic-max/glm53-hadamard-shapleymcg-kld/blob/a0c3407e79228ac06a1abf6a79484f38f38bd90a/results/RC5_RELEASE_RESULTS_20260907.md)
include the historical KLD protocol: teacher-to-student KL, CPU FP64,
2047 true-decode rows/window, excluding the prefill row.

## Validation required before promotion

1. Build this exact branch's native vLLM image and import the companion B12X
   P8 implementation; verify the image's commit and dependency identities.
2. Device closure at both K4 and K5, small-M/MTP and grouped prefill, five-run
   determinism, and native decoder/reference agreement. Keep documented GEMM
   rounding tolerances; do not weaken them to get a pass.
3. Clean HF-overlay serving, all 42 routed layer receipts, coherent target-only
   and MTP3 output, native dispatch and activation-path checks. Verify KV/tail
   semantics on this current B12X path instead of assuming RC5 equivalence.
4. Matched full-model KLD on authorized opened CF32; never label historical
   KLD as this port's measurement.
5. Client benchmark JSON/logs, prefill with prefix caching, decode C1/C2/C4
   without prefix-cache inflation. Document graph and MTP acceptance settings.

## Attribution and overlap

The adapter/glue uses the vLLM Apache-2.0 convention. The companion P8 kernel
code retains SHAPLEYMCG and third-party licenses in the B12X fork;
it is **not implicitly relicensed Apache-2.0**. Upstream inclusion requires a
maintainer decision about that boundary. Attribution: Brandon M. Music,
Z.ai, vLLM, Local Inference Lab/B12X, ExLlamaV3, KQuant, QSRT and
`w4a8_trellis`; see that fork's `licenses/trellismx/` and
`docs/trellismx-review.md`. The full historical source bundle remains in
commit `8433d53c19bce3462a2a68f95a315e1a3e3e55bb`, not in this PR's net diff.

EXL3 [PR 562](https://github.com/local-inference-lab/vllm/pull/562) is related
Jovian work, not this native P8 path. Coupled QSRT
[PR 566](https://github.com/local-inference-lab/vllm/pull/566) and wide W4A16
[PR 563](https://github.com/local-inference-lab/vllm/pull/563) target Infernal.
Their work is acknowledged, not represented as merged or duplicated here.
This owner-fork draft requests human review; no review, sign-off, successful
serving port or upstream approval is asserted.

## September 8 runtime evidence and overlay review

See the [four-row KLD matrix and overlay reconciliation](evidence-r27-20260908/README.md).
The historical FP8/DCP1 score is0.0318077613; historical NVFP4/DCP1 is0.0341811459.
Current r27 DCP4 scores are0.0350078183 (NVFP4) and0.0310574767 (FP8).
These are external measured-image references, not GPU qualification of this PR head.

Focused CPU checks after this review:31 loader/method tests passed;38 DCP tests
passed with21 GPU tests skipped;14 B12X contract tests passed.

## Explicit r27 source-build launcher

`Dockerfile` keeps the default DCP1 source recipe above. Its named `r27` target
instead installs `serve-r27.sh` as `/opt/trellismx/serve.sh`, enforcing TP4/DCP4
before delegating backend/cache/scheduler selection to
`/usr/local/bin/serve-glm53-flash.sh`. The supplied core image must contain that
r27 backend launcher and matching Python/native extensions. This target is not
an upgrade of an older core image.

```bash
docker build -f examples/trellismx/Dockerfile --target r27 \
  --build-arg JOVIAN_IMAGE=your-matching-r27-core-image \
  -t local/trellismx-r27:review .
printf 'services:\n  trellismx:\n    image: local/trellismx-r27:review\n' > /tmp/trellismx-r27-source.yaml
# MODEL_ROOT and TRELLISMX_ROOT must identify the complete local artifacts.
docker compose -f examples/trellismx/compose.r27.yaml \
  -f /tmp/trellismx-r27-source.yaml config
# After qualification, use the same two files with `up` to select this build.
```

`compose.r27.yaml` alone continues to identify the immutable published image;
it does not build this Dockerfile. Source-target launcher admission/delegation
can be tested separately from model loading. Such checks do not qualify the
full source-built image, GPU kernels, KLD, or performance.
