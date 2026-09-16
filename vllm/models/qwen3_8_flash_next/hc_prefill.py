# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility path for the Qwen4Exp HC prefill helpers."""

from vllm.models.qwen4_exp.common.hc_prefill import *  # noqa: F401,F403
from vllm.models.qwen4_exp.common.hc_prefill import (  # noqa: F401
    configure,
    eligible,
    report,
    RowOwnership,
)
