# Quantization

Quantization trades off model precision for smaller memory footprint, allowing large models to be run on a wider range of devices.

!!! tip
    To get started with quantization, see [LLM Compressor](llm_compressor/README.md), a library for optimizing models for deployment with vLLM that supports FP8, INT8, INT4, and other quantization formats.

The following are the supported quantization formats for vLLM:

- [AutoAWQ](auto_awq.md)
- [BitsAndBytes](bnb.md)
- [GPTQModel](gptqmodel.md)
- [Intel Neural Compressor](inc.md)
- [LLM Compressor](llm_compressor/README.md)
    - [FP8 W8A8](llm_compressor/fp8.md)
    - [INT4 W4A16](llm_compressor/int4.md)
    - [INT8 W4A8](llm_compressor/int8_w4a8.md)
    - [INT8 W8A8](llm_compressor/int8_w8a8.md)
- [NVIDIA Model Optimizer](modelopt.md)
- [Online Quantization](online.md)
- [AMD Quark](quark.md)
- [Quantized KV Cache](quantized_kvcache.md)
- [TorchAO](torchao.md)
- [FP8 ViT Encoder Attention](fp8_vit_attn.md)

## Supported Hardware

The table below shows the compatibility of various quantization implementations with different hardware platforms in vLLM:

<style>
td:not(:first-child) {
  text-align: center !important;
}
td {
  padding: 0.5rem !important;
  white-space: nowrap;
}

th {
  padding: 0.5rem !important;
  min-width: 0 !important;
}

th:not(:first-child) {
  writing-mode: vertical-lr;
  transform: rotate(180deg)
}
</style>

| Implementation            | Volta | Turing | Ampere | Ada | Hopper | AMD GPU | Intel GPU | x86 CPU | Arm CPU |
| ------------------------- | ----- | ------ | ------ | --- | ------ | ------- | --------- | ------- | ------- |
| AWQ                       | ❌    | ✅︎     | ✅︎     | ✅︎  | ✅︎     | ❌      | ✅︎        | ✅︎      | ❌      |
| GPTQ                      | ✅︎    | ✅︎     | ✅︎     | ✅︎  | ✅︎     | ❌      | ✅︎        | ✅︎      | ❌      |
| Marlin (GPTQ/AWQ/FP8/FP4) | ❌    | ✅︎*    | ✅︎     | ✅︎  | ✅︎     | ❌      | ❌        | ❌      | ❌      |
| llm-compressor INT8 (W8A8)| ❌    | ✅︎     | ✅︎     | ✅︎  | ✅︎     | ❌      | ❌        | ✅︎      | ✅︎      |
| llm-compressor INT8 (W4A8)| ❌    | ❌     | ❌     | ❌  | ❌     | ❌      | ❌        | ❌      | ✅︎      |
| llm-compressor FP8 (W8A8) | ❌    | ❌     | ❌     | ✅︎  | ✅︎     | ✅︎      | ❌        | ❌      | ❌      |
| bitsandbytes              | ✅︎    | ✅︎     | ✅︎     | ✅︎  | ✅︎     | ❌      | ❌        | ❌      | ❌      |
| DeepSpeedFP               | ✅︎    | ✅︎     | ✅︎     | ✅︎  | ✅︎     | ❌      | ❌        | ❌      | ❌      |
| GGUF                      | ✅︎    | ✅︎     | ✅︎     | ✅︎  | ✅︎     | ✅︎      | ❌        | ❌      | ❌      |

- Volta refers to SM 7.0, Turing to SM 7.5, Ampere to SM 8.0/8.6, Ada to SM 8.9, and Hopper to SM 9.0.
- ✅︎ indicates that the quantization method is supported on the specified hardware.
- ❌ indicates that the quantization method is not supported on the specified hardware.
- All Intel Gaudi quantization support has been migrated to [vLLM-Gaudi](https://github.com/vllm-project/vllm-gaudi).
- *Turing does not support Marlin MXFP4.

!!! note
    For information on quantization support on Google TPU, please refer to the [TPU-Inference Recommended Models and Features](https://docs.vllm.ai/projects/tpu/en/latest/recommended_models_features/) documentation.

!!! note
    This compatibility chart is subject to change as vLLM continues to evolve and expand its support for different hardware platforms and quantization methods.

    For the most up-to-date information on hardware support and quantization methods, please refer to [vllm/model_executor/layers/quantization](../../../vllm/model_executor/layers/quantization) or consult with the vLLM development team.

## EXL3 MSRT cartridges

EXL3 MSRT cartridges hot-swap additive, trellis-quantized routed-expert
residuals without materializing dense expert weights. They are not LoRA
adapters and cannot be loaded through the LoRA manager.

The base checkpoint must advertise the closed `exl3-msrt-base/1` rank-sliced
EXL3 profile. Runtime loading remains an operator policy: set
`VLLM_ENABLE_EXL3_CARTRIDGE=1` independently of checkpoint metadata. The
adapter path must name a
directory containing `adapter_config.json` conforming to
[`fq-cartridge-adapter/3`](https://github.com/malaiwah/progressive-tensors/blob/f8b5c491574bf2a7efa13d588a009d5ea8843e69/schemas/fq-cartridge-adapter-3.schema.json)
and every digest-pinned safetensors shard listed by that manifest. Loading
fails before the serving pause when the manifest is missing, its recomputed
per-layer compatibility fingerprint does not match the loaded base tensors, a
shard digest or declared byte size changed, the MCG layout differs, or the
declared TP topology,
stage order, bitrate, expert count, projection set, or tensor geometry is
incompatible.

The runtime supports one model-wide active cartridge. It supports a manifest-
declared full artifact sliced locally by TP workers, or a rank-sharded artifact
whose explicit ranks and world size exactly match the runtime (including a
rank-sharded world size of one). Pipeline, data, and expert parallelism, the V2
model runner, and simultaneous vLLM LoRA adapters are not supported. The
ExLlamaV3 extension must export `exl3_moe_additive_fused` with
`EXL3_MOE_ADDITIVE_ABI_VERSION = 1`.

`fq-cartridge-adapter/3` is a closed schema: unknown fields are rejected and
any schema change requires a new version. `producer_verified_signer` is
unauthenticated producer provenance; vLLM does not use it as a trust decision.
Dynamic cartridge loading is a trusted-admin operation and should use local,
trusted storage. Verification timeouts are cooperative between regular-file
reads; remote or FUSE filesystems can still block inside the operating system.

Async engine users call:

```python
await async_llm.load_exl3_cartridge("/srv/cartridges/k4like")
await async_llm.deactivate_exl3_cartridge()
```

The synchronous `LLMEngine` API is unsupported because it cannot serialize
request admission with this model-wide graph transition. Each model worker
copies and digest-verifies its shards into private staging before generation is
paused; the drained transaction consumes only those verified bytes.

The server endpoints are development APIs and are registered only with
`VLLM_SERVER_DEV_MODE=1`:

```bash
curl -X POST http://localhost:8000/load_exl3_cartridge \
  -H 'content-type: application/json' \
  -d '{"adapter_path":"/srv/cartridges/k4like"}'
curl http://localhost:8000/exl3_cartridge_status
curl -X POST http://localhost:8000/deactivate_exl3_cartridge
```

Load and deactivation pause admission, drain in-flight requests, release the
old CUDA graphs, switch packed tensors, recapture, and then resume. A failed
load restores the compressed base path. The adapter path must resolve to the
same files on every worker. These development endpoints are unauthenticated,
can read worker-local paths, mutate model output, and consume GPU memory; do
not expose them to untrusted clients.

## Out-of-Tree Quantization Plugins

vLLM supports registering custom, out-of-tree quantization methods using the `@register_quantization_config` decorator. This allows you to implement and use your own quantization schemes without modifying the vLLM codebase.

### Registering a Custom Quantization Method

To register a custom quantization method, create a class that inherits from `QuantizationConfig` and decorate it with `@register_quantization_config`. The `get_quant_method` dispatches to the appropriate quantize method based on the layer type:

```python
import torch
from vllm.model_executor.layers.quantization import (
    register_quantization_config,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.fused_moe import FusedMoE

@register_quantization_config("my_quant")
class MyQuantConfig(QuantizationConfig):
    """Custom quantization config."""

    def get_name(self) -> str:
        return "my_quant"

    def get_supported_act_dtypes(self) -> list:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # Minimum GPU compute capability, -1 for no restriction
        return -1

    @staticmethod
    def get_config_filenames() -> list[str]:
        # Config files to search for in model directory
        return []

    @classmethod
    def from_config(cls, config: dict) -> "MyQuantConfig":
        # Create config from model's quantization config
        return cls()

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        # Dispatch based on layer type
        # NOTE: you only need to implement methods you care about
        if isinstance(layer, LinearBase):
            return MyQuantLinearMethod()
        elif isinstance(layer, FusedMoE):
            return MyQuantMoEMethod(layer.moe_config)
        return None
```

### Required QuantizationConfig Methods

Your custom `QuantizationConfig` subclass must implement these abstract methods:

| Method | Description |
| ------ | ----------- |
| `get_name()` | Returns the name of the quantization method |
| `get_supported_act_dtypes()` | Returns list of supported activation dtypes (e.g., `torch.float16`) |
| `get_min_capability()` | Returns minimum GPU compute capability (e.g., 80 for Ampere, -1 for no restriction) |
| `get_config_filenames()` | Returns list of config filenames to search for in model directory |
| `from_config(config)` | Class method to create config from model's quantization config dict |
| `get_quant_method(layer, prefix)` | Returns the quantization method for a given layer, or `None` to skip |

### Implementing a Quantized Linear Method

For linear layers, return a `QuantizeMethodBase` subclass from `get_quant_method`. You can extend `UnquantizedLinearMethod` as a starting point:

```python
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

class MyQuantLinearMethod(UnquantizedLinearMethod):
    """Custom quantization method for linear layers."""

    def create_weights(
        self, layer: torch.nn.Module, *weight_args, **extra_weight_attrs
    ):
        # Create quantized weights for the layer
        ...

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Apply custom quantization logic here
        ...
```

### Implementing a Quantized MoE Method

For Mixture of Experts (MoE) models, return a `FusedMoEMethodBase` subclass from `get_quant_method`. You can use `UnquantizedFusedMoEMethod` to skip MoE quantization:

```python
from vllm.model_executor.layers.fused_moe.layer import UnquantizedFusedMoEMethod
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig

class MyQuantMoEMethod(FusedMoEMethodBase):
    """Custom quantization method for MoE layers."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Create quantized weights for the MoE layer
        ...

    def apply(
        self,
        layer: torch.nn.Module,
        router: "FusedMoERouter",
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        # Apply MoE computation with quantized weights
        ...

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        # Return the MoE quantization configuration
        ...
```

See existing implementations like `Fp8MoEMethod` in `vllm/model_executor/layers/quantization/fp8.py` for reference.

### Using the Plugin

Once registered, you can use your custom quantization method with vLLM:

```python
# Register your quantization method (import the module containing your config)
import my_quant_plugin

from vllm import LLM

# Use the custom quantization method
llm = LLM(model="your-model", quantization="my_quant")
```

For more information on the plugin system, see the [Plugin System documentation](../../design/plugin_system.md).
