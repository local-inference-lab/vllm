# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 V4.1 attention entry point; no foreign-provider dispatch."""

from vllm.models.deepseek_v4_1.attention import DeepseekV4Attention
from vllm.models.deepseek_v4_1.sparse_mla import DeepseekV41B12xBackend


class DeepseekV41B12xAttention(DeepseekV4Attention):
    """Native full-head b12x MLA with Full/Reindex/Reuse ownership."""


__all__ = ["DeepseekV41B12xAttention", "DeepseekV41B12xBackend"]
