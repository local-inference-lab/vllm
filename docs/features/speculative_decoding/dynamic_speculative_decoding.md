# Dynamic Speculative Decoding

## Why is Dynamic SD needed?

SD methods need to verify K tokens for each sequence during decoding. As BS increases, the effective BS becomes BS\*K which increases the compute requirement during verification. When this BS\*K goes beyond a critical BS then SD negatively impacts the decode speed (TPOT). DSD helps by tuning the K to an optimal value such that we continue to reap the benefits from SD.

## Use cases

* Variable concurrency workload using same deployment. K would decrease as concurrency increases.
* During RL rollout where we start off with high BS but then end up with small BS due to very few long tail request which end up generating a lot of tokens stalling the progress of the current rollout. Here K would go up during the end of rollout.

## `--speculative-config` schema

Dynamic speculative decoding supports two controls:

* `num_speculative_tokens_per_batch_size` caps K based on the current batch
  size.
* `adaptive_speculative_tokens_window` adjusts K from recent accepted draft
  lengths.

The controls can be enabled separately or together. When both are configured,
each contiguous batch-size cap band has an independent acceptance controller.

### Batch-size schedule

`num_speculative_tokens_per_batch_size` is a list of
`[start_bs, end_bs, optimal_K]` entries. Each entry selects K for the inclusive
batch-size range `[start_bs, end_bs]`. For example:

```bash
--speculative-config '{
    "method": "eagle",
    "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
    "num_speculative_tokens": 3,
    "num_speculative_tokens_per_batch_size": [
      [1, 64, 3],
      [65, 128, 1],
      [129, 512, 0]
    ]
  }'
```

implies that:

* K=3 will be used when the concurrency is in range [1, 64]
* K=1 will be used when the concurrency is in range [65, 128]
* K=0 will be used when the concurrency is in range [129, 512], i.e., no draft tokens will be produced.

### Acceptance-length adaptation

Set `adaptive_speculative_tokens_window` to the number of non-empty
verification steps to observe between K updates.
`adaptive_speculative_tokens_initial` sets the starting K and defaults to
`num_speculative_tokens`; `num_speculative_tokens` remains the upper bound.
The controller never reduces K below 1 unless a batch-size cap selects K=0.

For each window, the controller computes the request-weighted mean accepted
draft length and targets:

```text
target K = round_half_up(mean accepted draft tokens + 1)
```

K drops directly to a lower target, but rises by at most one token per window.
This reacts quickly to wasted verification work without immediately returning
to an expensive depth after a transient high-acceptance window.

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --speculative-config '{
    "method": "eagle3",
    "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
    "num_speculative_tokens": 8,
    "adaptive_speculative_tokens_initial": 3,
    "adaptive_speculative_tokens_window": 32
  }'
```

Every selectable depth is captured separately when full CUDA graphs are in
use. If acceptance adaptation is combined with batch-size caps, capture covers
all intermediate depths up to the largest cap and K=0 when a cap disables
drafting.

## Online Examples

### Dynamic SD Eagle Drafter

```bash
VLLM_USE_V2_MODEL_RUNNER=0 vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --speculative-config '{
    "method": "eagle",
    "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
    "num_speculative_tokens": 3,
    "num_speculative_tokens_per_batch_size": [
      [1, 64, 3],
      [65, 128, 1],
      [129, 512, 0]
    ]
  }'
```

### Dynamic SD Eagle3 Drafter

```bash
VLLM_USE_V2_MODEL_RUNNER=0 vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --speculative-config '{
    "method": "eagle3",
    "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
    "num_speculative_tokens": 3,
    "num_speculative_tokens_per_batch_size": [
      [1, 16, 5],
      [17, 32, 4],
      [33, 64, 3],
      [65, 128, 1],
      [129, 512, 0]
    ]
  }'

```

## Limitations

* Tested with Eagle, Eagle-3, and DFlash. Other SD methods may or may not work out of the box
* Full Cudagraph only works with Model Runner V2. MRv1 only supports piece-wise cuda graph with this feature
* Not compatible with data parallelism (`--data-parallel-size > 1`). Each DP rank schedules independently, so ranks can pick different K values, causing DP collective divergence and deadlocks. When DP is enabled, vLLM disables both dynamic controls and falls back to the static `num_speculative_tokens` value.
