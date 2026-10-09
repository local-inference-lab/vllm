# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 execution components for the DeepSeek V4.1 architecture."""

from vllm.platforms import current_platform


def is_enabled() -> bool:
    return current_platform.is_cuda() and current_platform.is_device_capability_family(
        120
    )
