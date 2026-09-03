# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for image-layout detection and placeholder handling."""

import numpy as np
import pytest

from vllm.multimodal.parse import (
    ImageProcessorItems,
    ImageSize,
    looks_like_chw,
)

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        # RGB CHW with widths 1, 3, and 4 must not be misread as HWC.
        ((3, 8, 1), True),
        ((3, 8, 3), True),
        ((3, 6, 4), True),
        # Conventional CHW shapes.
        ((3, 8, 5), True),
        ((3, 224, 224), True),
        # HWC shapes.
        ((4, 6, 3), False),
        ((224, 224, 3), False),
        # Gray/RGBA-leading shapes follow the unambiguous last-axis rule.
        ((1, 8, 4), False),  # c=1 and w=4: not RGB, so HWC
        ((4, 8, 4), False),  # c=4 and w=4: not RGB, so HWC
        # Not 3D.
        ((3, 8), False),
        ((1, 3, 8, 4), False),
    ],
)
def test_looks_like_chw(shape, expected):
    assert looks_like_chw(shape) == expected


@pytest.mark.parametrize(
    ("array", "expected_size"),
    [
        # CHW, widths 1/3/4.
        (np.zeros((3, 8, 1), dtype=np.uint8), ImageSize(1, 8)),
        (np.zeros((3, 8, 3), dtype=np.uint8), ImageSize(3, 8)),
        (np.zeros((3, 6, 4), dtype=np.uint8), ImageSize(4, 6)),
        # Conventional CHW.
        (np.zeros((3, 8, 5), dtype=np.uint8), ImageSize(5, 8)),
        # HWC.
        (np.zeros((4, 6, 3), dtype=np.uint8), ImageSize(6, 4)),
        # Gray (C=1) and width 4 is HWC under the unambiguous rule:
        # it is interpreted as (H=1, W=8, C=4).
        (np.zeros((1, 8, 4), dtype=np.uint8), ImageSize(8, 1)),
    ],
)
def test_image_get_image_size_chw_hwc(array, expected_size):
    items = ImageProcessorItems(data=[array])
    assert items.get_image_size(0) == expected_size


def test_plain_text_image_token_mention_stays_text():
    """A textual placeholder mention must not be treated as image input."""
    from vllm.tokenizers.deepseek_v4_encoding import (
        IMAGE_PLACEHOLDER,
        flatten_content_blocks,
    )

    # Image blocks become placeholders; text stays literal.
    content = [
        {"type": "text", "text": f"text containing {IMAGE_PLACEHOLDER}"},
    ]
    out = flatten_content_blocks(content)
    assert IMAGE_PLACEHOLDER in out  # Literal text, not an injected image block.

    # An image block does insert a placeholder.
    content2 = [
        {"type": "text", "text": "before:"},
        {"type": "image", "source": {"data": "AAAA"}},
    ]
    out2 = flatten_content_blocks(content2)
    assert out2 == f"before:{IMAGE_PLACEHOLDER}"
