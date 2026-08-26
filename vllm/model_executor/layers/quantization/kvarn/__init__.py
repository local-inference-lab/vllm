# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVarN variance-normalized KV-cache quantization.

Hadamard rotation and iterative log-domain variance normalization precede
dense little-endian K5/V5 packing. Keys are quantized per channel and values
per token. Completed history is packed while sink and recent tiles remain in
an FP8 precision-tail pool by default.
"""

from vllm.model_executor.layers.quantization.kvarn.config import KVarNConfig

__all__ = ["KVarNConfig"]
