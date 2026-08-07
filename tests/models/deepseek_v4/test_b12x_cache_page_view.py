# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.deepseek_v4.nvidia.b12x import _b12x_cache_page_view


_PAGE_SIZE = 64
_PAYLOAD_BYTES = _PAGE_SIZE * 584
_PADDED_PAGE_BYTES = 37_440


def test_b12x_cache_page_view_accepts_contiguous_payload_pages() -> None:
    cache = torch.empty((3, _PAGE_SIZE, 584), dtype=torch.uint8)

    view = _b12x_cache_page_view(cache, _PAGE_SIZE, "cache")

    assert view.shape == (3, _PAYLOAD_BYTES)
    assert view.stride() == (_PAYLOAD_BYTES, 1)


def test_b12x_cache_page_view_preserves_padded_physical_stride() -> None:
    storage = torch.empty(3 * _PADDED_PAGE_BYTES, dtype=torch.uint8)
    cache = torch.as_strided(
        storage,
        size=(3, _PAGE_SIZE, 584),
        stride=(_PADDED_PAGE_BYTES, 584, 1),
    )

    view = _b12x_cache_page_view(cache, _PAGE_SIZE, "cache")

    assert view.shape == (3, _PAYLOAD_BYTES)
    assert view.stride() == (_PADDED_PAGE_BYTES, 1)


def test_b12x_cache_page_view_rejects_short_physical_stride() -> None:
    storage = torch.empty(2 * _PAYLOAD_BYTES, dtype=torch.uint8)
    cache = torch.as_strided(
        storage,
        size=(2, _PAGE_SIZE, 584),
        stride=(_PAYLOAD_BYTES - 1, 584, 1),
    )

    with pytest.raises(RuntimeError, match=r"page stride .* is smaller"):
        _b12x_cache_page_view(cache, _PAGE_SIZE, "cache")


def test_b12x_cache_page_view_rejects_overlapping_2d_pages() -> None:
    storage = torch.empty(2 * _PAYLOAD_BYTES, dtype=torch.uint8)
    cache = torch.as_strided(
        storage,
        size=(2, _PAYLOAD_BYTES),
        stride=(_PAYLOAD_BYTES - 1, 1),
    )

    with pytest.raises(RuntimeError, match=r"page stride .* is smaller"):
        _b12x_cache_page_view(cache, _PAGE_SIZE, "cache")


def test_b12x_cache_page_view_rejects_gapped_page_payload() -> None:
    storage = torch.empty(2 * _PAYLOAD_BYTES + _PAGE_SIZE, dtype=torch.uint8)
    cache = torch.as_strided(
        storage,
        size=(2, _PAGE_SIZE, 584),
        stride=(_PAYLOAD_BYTES, 585, 1),
    )

    with pytest.raises(RuntimeError, match=r"page payload must be contiguous"):
        _b12x_cache_page_view(cache, _PAGE_SIZE, "cache")
