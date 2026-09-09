# TP4 GLM prefill at DCP1 and DCP2

Status: **qualified** for the bounded combined-runtime checks described below; **research-only** for broader model accuracy and DCP2 cache-invariant probabilities.

## Recorded serving composition

These measurements enable or disable **both continuation coalescing and token-sharded mHC together**. They establish no isolated speedup for either PR. The standalone coalescing implementation at vLLM `a6c8407645cf5e751711883f24e8822e73dba9a0` and mHC implementation at vLLM `138ffcdbc97141ef4600d33ae12077a7b5bf926a` have CPU coverage; neither complete standalone revision has GPU or full-model validation.

- Model: `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark`, revision `df116c4fb16b1d37ae43d2cfd624de26ffbc832e`.
- Image: `sha256:e585e8b3ebeeea2852011d319f4a4d781bf2ae33388210b7cd16ff16d3765d9b`.
- vLLM: `8f8ea47be212bbdd91b2172d5958ea2aae2b0e50`; B12X: `0b6d61c37c87ae49d2f9d20d38b9da023146e243`.
- Four GB10 Sparks, TP4, MTP3, BF16 activations, FP8 KV, B12X attention/KDA/MoE/linear, 8192-token scheduler budget, explicit 512-token target and recurrent cache grids, aligned checkpoint policy, prefix retention interval 0, 24 GiB KV per rank, max 16 sequences.
- `FULL_AND_PIECEWISE` graphs with capture sizes 4,8,...,64; async scheduling; prefill schedule interval 2. DCP1 uses cache interleave 1; DCP2 uses 4. These choices are fixed within each on/off pair.
- TP4 mesh and dual-domain NCCL are retained. NCCL library SHA-256: `768a450b5eb84bf3d1191795350e43c96de75aeba4783ec314d47672fe6e1fc6`. CKV gathering uses the DCP2 group and bypasses at DCP1. SparkCache, compact index cache and DCP4-only top-k owner exchange/fused endpoints are disabled.
- Both flags are 0 in controls and 1 in enabled arms: `VLLM_B12X_KDA_PREFILL_COALESCING`, `VLLM_GLM53_MHC_PREFILL_SHARD`. Both native geometry overrides are 512: `VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE`, `VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE`.

## Prefill

Median of three cold exact-token requests per size after excluded warmups and semantic checks; each allows one generated token. Tokens/s is prompt tokens divided by TTFT. Arms ran sequentially, not interleaved.

| DCP | Coalescing + mHC | 8K tok/s | 16K tok/s | 32K tok/s |
| --- | --- | ---: | ---: | ---: |
| 1 | Off | 2,263.0 | 2,615.7 | 2,828.4 |
| 1 | On | 3,616.3 | 3,615.1 | 3,586.1 |
| 2 | Off | 1,938.7 | 2,345.5 | 2,619.0 |
| 2 | On | 3,493.9 | 3,461.8 | 3,445.9 |

## MTP-normalized decode

Aggregate emitted tokens/s divided by observed speculative acceptance length. Each cell is one approximately 10-second observation at temperature 1 with a 2048-token output cap. These short observations do not demonstrate repeatable decode gains. At C4, the value aggregates request decode steps rather than counting GPU launches.

| DCP | Coalescing + mHC | Context | C1 steps/s | C4 aggregate steps/s |
| --- | --- | ---: | ---: | ---: |
| 1 | Off | 8,192 | 21.17 | 47.93 |
| 1 | Off | 32,768 | 21.23 | 46.21 |
| 1 | On | 8,192 | 21.40 | 48.00 |
| 1 | On | 32,768 | 21.17 | 49.93 |
| 2 | Off | 8,192 | 19.73 | 45.41 |
| 2 | Off | 32,768 | 19.86 | 45.01 |
| 2 | On | 8,192 | 20.05 | 46.62 |
| 2 | On | 32,768 | 19.80 | 44.47 |

## Correctness and numerical limits

The measured composition passed 18 GB10 recurrent-export, continuation/reload and convolution-history component tests, with zero skips. Each serving arm has six exact-answer cases, nine cold prefill timings and four decode cells; four-rank logs prove enabled paths executed and disabled paths did not. Those component tests do not validate distributed mHC intermediate tensors or broad model accuracy.

Fixed-input comparisons require identical calibrated prompt token IDs, generation options and output token sequences. The preselected generated-token logprob tolerance is `abs(on-off) <= 0.1 + 0.02*abs(off)`. All six matched on/off comparisons pass. DCP1 also passes cached-versus-cold comparisons.

**DCP2 cached-versus-cold behavior remains research-only.** With both patches enabled, the maximum generated-token logprob difference is 0.127015 and exceeds that comparison's bound; the disabled comparison is 0.018938. DCP2's enabled/disabled extended-cold maximum difference is 0.100147, narrowly within its approximately 0.100867 bound. Exact 13-token answers match, but this does not prove the numerical difference harmless for other prompts. Separate on/off passes do not establish cache invariance, and the enabled cached-versus-cold fixture has status `needs-review`.

The DCP2 enabled 8K/C1 decode cell records 575 server and 571 client output tokens, within the benchmark's accepted boundary tolerance. Small normalized differences should not be described as speedups. Mixed-request scheduling, preemption, every checkpoint boundary, long-context capacity, four-rank cache restoration and broad accuracy are outside this performance matrix. The DCP2 disabled numerical fixture uses a separate correctness-only deployment requiring a node reboot. The prefill/decode tables use the four performance deployments identified in the measurement artifact.

AI assistance was used for implementation, testing and evidence preparation.

## Checkpoint and collective contracts

With TP4 and 512-token physical cache blocks, the scheduler grids are 512,
1024 and 2048 tokens for DCP1, DCP2 and DCP4 respectively. Retention requires
two, three and four active checkpoint destinations respectively. B12X keeps
planned capacity four; three active destinations do not require capacity three.
The checkpoint kernel receives local head geometry, packed ranges and destinations;
the vLLM adapter determines distributed cache boundaries.

Token-sharded mHC always uses the four-rank TP communicator. Each rank owns
2048 rows of an eligible 8192-token prefill, independently of DCP group size.
The CKV attention gather uses the two-rank DCP subgroup at DCP2 and bypasses
at DCP1. These group identities do not change mHC token ownership.

## Standalone CPU validation

Coalescing revision `a6c8407645cf5e751711883f24e8822e73dba9a0` passes 62
checkpoint/allocator and worker-binding cases. Its scheduler configuration
interface matches the PR base `2a979314dc97b03173a0a76fc15664ec924db32b`.
It requires the four-checkpoint B12X API and contains no mHC implementation.

mHC revision `138ffcdbc97141ef4600d33ae12077a7b5bf926a` passes 62 cases,
including all four TP ranks with DCP1/2/4. Tests use B12X
`06b4de7c723e6f166d65abf5909c5b7d0f8acc68`, whose capacity type lacks
`max_checkpoints`, demonstrating independence from the checkpoint extension.

Both suites run on Windows with Python 3.12 and Torch 2.13.0+cpu. The launcher
maps `uvloop` to `winloop`; GPU operations in the suites use CPU doubles.
Commands executed from the corresponding standalone checkout:

```powershell
# Continuation coalescing
.venv/Scripts/python.exe -c "import sys,winloop; sys.modules['uvloop']=winloop; from b12x.sequence.kda_prefill._impl import Caps; assert 'max_checkpoints' in Caps.__dataclass_fields__; import pytest; raise SystemExit(pytest.main(['tests/v1/core/test_recurrent_prefill_checkpoint.py','tests/v1/worker/test_kda_prefill_checkpoint_binding.py','--confcutdir=tests/v1','-q','--tb=short']))"

# Token-sharded mHC
.venv/Scripts/python.exe -c "import sys,winloop; sys.modules['uvloop']=winloop; from b12x.sequence.kda_prefill._impl import Caps; assert 'max_checkpoints' not in Caps.__dataclass_fields__; import pytest; raise SystemExit(pytest.main(['tests/models/test_glm5next_mhc_prefill.py','--confcutdir=tests/models','-q','--tb=short']))"
```

## Measurement artifact

[Serving evidence](evidence.json), schema `dcp-prefill-review-evidence/v1`,
contains numeric samples, semantic serving arguments, runtime identities and
source receipt hashes. Configuration labels `dcp1-off`, `dcp1-on`, `dcp2-off`
and `dcp2-on` identify the DCP size and whether both prefill features are disabled
or enabled. Hashes identify source receipts; the artifact does not include the
complete private deployment receipts or request-associated worker logs.
This artifact supports inspection of the reported measurements; it is not a
complete image-build or cluster-deployment recipe.
