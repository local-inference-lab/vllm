# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.layers.attention.attention import (
    _largest_kernel_block_within,
)
from vllm.v1.attention.backend import MultipleOf


class _FixedPageBackend:
    @staticmethod
    def get_supported_kernel_block_sizes():
        return [16, 32, 64]


class _MultipleOfPageBackend:
    @staticmethod
    def get_supported_kernel_block_sizes():
        return [MultipleOf(16)]


def test_largest_fixed_block_within_shared_page():
    assert _largest_kernel_block_within(_FixedPageBackend, 1024, 65536, 256) == 64


def test_largest_multiple_within_shared_page():
    assert (
        _largest_kernel_block_within(
            _MultipleOfPageBackend,
            per_token_bytes=1024,
            page_budget=1_511_424,
            fallback=256,
        )
        == 1472
    )


def test_smallest_block_without_shared_page():
    assert _largest_kernel_block_within(_MultipleOfPageBackend, 1024, None, 256) == 16
