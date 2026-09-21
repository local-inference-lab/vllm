# MiMo-V2.6-Flash-RL on Karmic Kraken

This recipe targets `XiaomiMiMo/MiMo-V2.6-Flash-RL` at revision
`3b38d063180c3e4aed9691fdc735f3d10b266ee4`. On `dev/karmic-kraken`, include
[the QKV/MTP loader backport](https://github.com/local-inference-lab/vllm/pull/828)
and [the router/DFlash backport](https://github.com/local-inference-lab/vllm/pull/829)
for this checkpoint. Hardware qualification of this recipe is pending.

The model combines 9 global and 39 sliding-window decoder layers. Global layers
have 64 query heads and 4 KV heads; sliding layers have 64 query heads and 8 KV
heads with a 128-token window and attention sinks. Q/K head size is 192 and V
head size is 128. The DiffKV implementation stores packed K/V elements; it is
not an MLA latent cache. At TP4, each GPU owns one global KV head or two sliding
KV heads per layer.

The weights mix MXFP4 experts, FP8 projections, and BF16 projections/encoders.
Weight precision does not select KV precision. BF16 KV stores two bytes per
packed element. E4M3 KV stores one byte per element with separate scalar K/V
scales and BF16 queries. The reader dequantizes K/V before its dot products.
E5M2, INT8, and per-token-head KV modes are not implemented by this backend.

## Prepare the checkpoint

The pinned `dflash/config.json` has one trailing comma. Create a metadata overlay
that shares all weights and verifies the exact correction. Do not apply this
repair to an unverified later revision.

```python
import hashlib
from pathlib import Path
from huggingface_hub import snapshot_download

source = Path(snapshot_download(
    "XiaomiMiMo/MiMo-V2.6-Flash-RL",
    revision="3b38d063180c3e4aed9691fdc735f3d10b266ee4",
)).resolve()
overlay = Path("models/mimo-v26-flash-rl").resolve()
overlay.mkdir(parents=True, exist_ok=False)
for original in source.rglob("*"):
    if not original.is_file():
        continue
    relative = original.relative_to(source)
    target = overlay / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if relative.as_posix() == "dflash/config.json":
        raw = original.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == (
            "29f18def0d74535771b2364b28107f4914ebb88872abb93189012f7573e10e4b"
        )
        fixed = raw.replace(b'"use_cache": true,\n', b'"use_cache": true\n')
        assert hashlib.sha256(fixed).hexdigest() == (
            "3c4152b4ef45c9b7dd22e4c5c54b18d4cd1a12fc72a31c6cddc5b4089acaa925"
        )
        target.write_bytes(fixed)
    else:
        target.symlink_to(original.resolve())
```

When using a container, mount both the overlay and the original weight directory
read-only, preserving the absolute symlink destinations. The complete snapshot,
including encoders and the DFlash decoder, is about 178 GB.

## Launch

Build/install the combined source with CUDA architecture 12.0 for SM120. Use the
fork's Dockerfile and retain the source revision, dependency versions, build
arguments, and immutable image identity.

The current MiMo processor reads its limits from `processor_config`. Preserve
that configuration while overriding the media budget:

```sh
MIMO_PROCESSOR_OVERRIDE=$(python3 - <<'PY'
import json
from pathlib import Path
config = json.loads(Path("models/mimo-v26-flash-rl/config.json").read_text())
processor = config["processor_config"]
processor.update(image_max_pixels=1048576, video_max_pixels=262144, max_frames=8)
print(json.dumps({"processor_config": processor}))
PY
)

VLLM_USE_V2_MODEL_RUNNER=1 vllm serve models/mimo-v26-flash-rl \
  --served-model-name MiMo-V2.6-Flash-RL \
  --trust-remote-code --tensor-parallel-size 4 --dtype bfloat16 \
  --kv-cache-dtype bfloat16 --attention-backend TRITON_ATTN \
  --moe-backend marlin --gpu-memory-utilization 0.90 \
  --max-model-len 1048576 --max-num-seqs 16 --max-num-batched-tokens 8192 \
  --enable-chunked-prefill --enable-prefix-caching \
  --reasoning-parser mimo --tool-call-parser mimo --enable-auto-tool-choice \
  --generation-config vllm \
  --enable-prompt-tokens-details \
  --limit-mm-per-prompt '{"image":1,"audio":1,"video":1}' \
  --hf-overrides "$MIMO_PROCESSOR_OVERRIDE"
```

For E4M3 KV, replace only `--kv-cache-dtype bfloat16` with
`--kv-cache-dtype fp8_e4m3`. Record actual buffers and scale values; an accepted
CLI option alone is not proof of one-byte storage or acceptable model quality.
The standard scale loader uses checkpoint K/V scales when supplied and fixed
1.0 defaults otherwise. This recipe does not recalibrate scales per request.

For DFlash K7, add:

```sh
--speculative-config '{"method":"dflash","model":"models/mimo-v26-flash-rl/dflash","num_speculative_tokens":7}'
```

The separate draft has five layers and block size eight. Record draft cache
precision independently of the target. The three embedded MTP layers are a
different artifact; the fork's supported native MTP mode uses one layer and
`--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`.

The publisher recommends temperature 1.0 and top-p 0.95. Its template uses
`chat_template_kwargs: {"enable_thinking": true}`. Deterministic qualification
uses temperature 0 and frozen inputs. A configured 1M maximum is a capacity
request; only successful long-context inference establishes a tested limit.

## Focused tests

```sh
python3 -m pytest tests/models/quantization/test_mimo_v2_qkv_shard.py \
  tests/models/quantization/test_mimo_v2_router_dflash.py \
  tests/v1/attention/test_triton_diffkv_config.py -q
python3 -m pytest tests/kernels/attention/test_triton_diffkv_fp8.py -q
python3 -m pytest tests/models/multimodal/test_mimo_v2_omni.py -q
```

The independent DiffKV tests require CUDA and must execute on SM120 without
skips. They compare the packed writer, backend, 2D reader, split-KV reader, and
graph replay against independently quantized K/V and FP32 reference attention
at `atol=rtol=0.02`. Model-quality comparisons against BF16 remain a separate
requirement; numerical agreement with quantized inputs does not bound the
quality loss from quantization.
