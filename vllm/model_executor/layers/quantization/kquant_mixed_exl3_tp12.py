# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compatibility import for pre-QSRT Kimi-K3 checkpoints.

New serving code must import :mod:`kquant_kimi_k3_qsrt_tp12` directly. This
module remains only so the validated interim ``mixed_exl3_tp12`` checkpoint
can be loaded while it is still used as a comparison teacher.
"""

from vllm.model_executor.layers.quantization.kquant_kimi_k3_qsrt_tp12 import *  # noqa: F403
