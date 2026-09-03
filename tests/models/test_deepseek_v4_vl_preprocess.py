# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for DeepSeek-V4 VL multimodal preprocessing."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from vllm.models.deepseek_v4.common import mm_preprocess as ours
from vllm.models.deepseek_v4.common.mm_preprocess import (
    COMPRESS_PAD_TO,
    IMAGE,
    IMAGE_END,
    IMAGE_PAD,
    IMAGE_START,
    DeepseekV4VLMultiModalProcessor,
    DeepseekV4VLProcessor,
)
from vllm.multimodal.parse import MultiModalDataParser
from vllm.multimodal.processing import PromptReplacement, PromptUpdateDetails
from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config

PATCH_SIZE = 14
DOWNSAMPLE_RATIO = 3
MAX_N_TOKEN = 384

OURS_KWARGS = dict(
    patch_size=PATCH_SIZE,
    downsample_ratio=DOWNSAMPLE_RATIO,
    max_n_token=MAX_N_TOKEN,
    min_pixels=147456,
    max_wh_ratio=8,
)


@pytest.mark.parametrize(
    ("height", "width", "expected"),
    [
        (42, 84, (1, 2, 10)),
        (84, 84, (2, 2, 10)),
        (126, 126, (3, 3, 18)),
    ],
)
def test_grid_tokens(height: int, width: int, expected: tuple[int, int, int]):
    assert ours.grid_tokens(height, width, PATCH_SIZE, DOWNSAMPLE_RATIO) == expected


@pytest.mark.parametrize("max_n_token", [32, 128, MAX_N_TOKEN, 1024])
@pytest.mark.parametrize(
    "height,width", [(50, 3000), (3000, 50), (137, 400), (756, 756), (100, 100)]
)
def test_solve_resize_ratio_returns_consistent_grid(
    height: int, width: int, max_n_token: int
):
    n_llm_h, n_llm_w, best_height, best_width, num_tokens = ours.solve_resize_ratio(
        height,
        width,
        PATCH_SIZE,
        DOWNSAMPLE_RATIO,
        max_n_token,
    )
    assert best_height % PATCH_SIZE == 0
    assert best_width % PATCH_SIZE == 0
    assert (n_llm_h, n_llm_w, num_tokens) == ours.grid_tokens(
        best_height, best_width, PATCH_SIZE, DOWNSAMPLE_RATIO
    )


@pytest.mark.parametrize("max_n_token", [32, 128, MAX_N_TOKEN, 1024])
@pytest.mark.parametrize(
    "height,width", [(50, 3000), (3000, 50), (137, 400), (756, 756)]
)
def test_safe_resize_reserves_alignment_padding(
    height: int, width: int, max_n_token: int
):
    best_h = -(-height // PATCH_SIZE) * PATCH_SIZE
    best_w = -(-width // PATCH_SIZE) * PATCH_SIZE
    n_llm_h, n_llm_w, resized_h, resized_w = ours.safe_resize(
        height, width, best_h, best_w, PATCH_SIZE, DOWNSAMPLE_RATIO, max_n_token
    )
    assert (n_llm_h, n_llm_w) == ours.grid_tokens(
        resized_h, resized_w, PATCH_SIZE, DOWNSAMPLE_RATIO
    )[:2]
    assert ours.grid_tokens(resized_h, resized_w, PATCH_SIZE, DOWNSAMPLE_RATIO)[
        2
    ] <= max_n_token - (COMPRESS_PAD_TO - 1)


@pytest.mark.parametrize("start_pos", range(9))
@pytest.mark.parametrize(
    "n_llm_h,n_llm_w", [(h, w) for h in range(1, 9) for w in range(1, 9)]
)
def test_build_image_block_returns_a_permutation(
    n_llm_h: int, n_llm_w: int, start_pos: int
):
    types, perm = ours.build_image_block(n_llm_h, n_llm_w, start_pos)
    compress_pad = COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO
    assert types[0] == (IMAGE_PAD if compress_pad else IMAGE_START)
    assert types[-1] == IMAGE_END
    assert torch.equal(perm.sort().values, torch.arange(n_llm_h * n_llm_w))


@pytest.mark.parametrize(
    "width,height",
    [(800, 600), (100, 2000), (2000, 100), (50, 50), (13, 7), (384, 384)],
)
def test_load_image_returns_normalized_patches(width: int, height: int):
    rng = np.random.default_rng(width * 10000 + height)
    array = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    image = Image.fromarray(array)

    our_out = ours.load_image(image, **OURS_KWARGS)
    patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = our_out
    assert patches.shape == (n_vit_h * n_vit_w, 3, PATCH_SIZE, PATCH_SIZE)
    assert patches.dtype == torch.bfloat16
    assert (n_llm_h, n_llm_w) == ours.grid_tokens(
        n_vit_h * PATCH_SIZE,
        n_vit_w * PATCH_SIZE,
        PATCH_SIZE,
        DOWNSAMPLE_RATIO,
    )[:2]
    assert patches.min() >= -1
    assert patches.max() <= 1


@pytest.mark.parametrize("start_pos", range(8))
@pytest.mark.parametrize("n_llm_h,n_llm_w", [(1, 1), (2, 3), (3, 2), (5, 4)])
def test_block_semantics(n_llm_h: int, n_llm_w: int, start_pos: int):
    types, perm = ours.build_image_block(n_llm_h, n_llm_w, start_pos)

    # The grid occupies the pad-free part of the block.
    _, _, num_tokens = ours.grid_tokens(
        n_llm_h * PATCH_SIZE * DOWNSAMPLE_RATIO,
        n_llm_w * PATCH_SIZE * DOWNSAMPLE_RATIO,
        PATCH_SIZE,
        DOWNSAMPLE_RATIO,
    )
    compress_pad = COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO
    assert len(types) == num_tokens + compress_pad
    assert (types == IMAGE_PAD).sum() >= compress_pad
    assert (types[:compress_pad] == IMAGE_PAD).all()
    assert types[compress_pad] == IMAGE_START
    assert types[-1] == IMAGE_END
    assert (types == IMAGE).sum() == n_llm_h * n_llm_w
    assert len(perm) == n_llm_h * n_llm_w
    # ``perm`` selects every IMAGE slot exactly once.
    assert torch.equal(perm.sort().values, torch.arange(n_llm_h * n_llm_w))

    # The is_embed mask must mark exactly the IMAGE positions.
    is_embed = types == IMAGE
    assert is_embed.sum() == n_llm_h * n_llm_w
    assert not is_embed[:compress_pad].any()


@pytest.mark.parametrize("n_llm_h,n_llm_w", [(1, 1), (3, 2), (4, 5)])
def test_pad_free_block(n_llm_h: int, n_llm_w: int):
    types, perm = ours.build_image_block_pad_free(n_llm_h, n_llm_w)
    ref_types, ref_perm = ours.build_image_block(n_llm_h, n_llm_w, start_pos=0)
    compress_pad = COMPRESS_PAD_TO - 1
    assert torch.equal(types, ref_types[compress_pad:])
    assert torch.equal(perm, ref_perm)
    assert types[0] == IMAGE_START and types[-1] == IMAGE_END


def test_processor_output():
    config = DeepseekV4Config(vocab_size=129280)
    processor = DeepseekV4VLProcessor(config)
    images = [
        Image.new("RGB", (800, 600), color=(10, 20, 30)),
        Image.new("RGB", (100, 2000), color=(200, 100, 50)),
    ]
    out = processor(images=images)

    vit_grid = out["vit_grid"]
    llm_grid = out["llm_grid"]
    assert vit_grid.shape == llm_grid.shape == (2, 2)
    assert out["patches"].shape[0] == vit_grid.prod(-1).sum()
    assert out["patches"].dtype == torch.bfloat16
    assert out["patches"].shape[1:] == (3, PATCH_SIZE, PATCH_SIZE)
    assert out["perm"].shape[0] == llm_grid.prod(-1).sum()

    # Per-image pad-free types: [IMAGE_START, ..., IMAGE_END], with
    # n_llm_h * n_llm_w IMAGE entries.
    offset = 0
    for i in range(2):
        n_llm_h, n_llm_w = llm_grid[i].tolist()
        _, _, num_tokens = ours.grid_tokens(
            n_llm_h * PATCH_SIZE * DOWNSAMPLE_RATIO,
            n_llm_w * PATCH_SIZE * DOWNSAMPLE_RATIO,
            PATCH_SIZE,
            DOWNSAMPLE_RATIO,
        )
        types = out["types"][offset : offset + num_tokens]
        offset += num_tokens
        assert types[0] == IMAGE_START and types[-1] == IMAGE_END
        assert (types == IMAGE).sum() == n_llm_h * n_llm_w
    assert offset == out["types"].shape[0]

    assert processor(images=[]) == {}


class _StubInfo:
    """Minimum ``ProcessingInfo`` surface for the placeholder-splicing test."""

    def get_hf_config(self):
        return SimpleNamespace()

    def get_tokenizer(self):
        # Token-id replacements do not require a tokenizer because the text
        # fallback is not exercised in this test.
        return None

    def get_data_parser(self):
        return MultiModalDataParser()


def test_apply_token_matches_adds_compress_pad():
    base = ours.IMAGE_SENTINEL_BASE_ID
    image_token_id = 7
    n_llm_h, n_llm_w = 3, 2

    processor = DeepseekV4VLMultiModalProcessor(_StubInfo(), None)

    types, _ = ours.build_image_block_pad_free(n_llm_h, n_llm_w)
    full = (base + types).tolist()
    update = PromptReplacement(
        modality="image",
        target=[image_token_id],
        replacement=PromptUpdateDetails.select_token_id(full, base + IMAGE),
    )
    prompt = [11, 12, image_token_id, 13, 14, 15, image_token_id, 16]
    mm_prompt_updates = {"image": [[update.resolve(0)], [update.resolve(1)]]}

    new_token_ids, match_result, placeholders = (
        processor._apply_token_matches_with_placeholders(prompt, mm_prompt_updates)
    )
    assert match_result == {"image": [0, 0]}

    # Build the expected blocks at their final prompt positions.
    ref_ids = [11, 12]
    ref_types, _ = ours.build_image_block(n_llm_h, n_llm_w, len(ref_ids))
    ref_ids += (base + ref_types).tolist()
    ref_ids += [13, 14, 15]
    ref_types, _ = ours.build_image_block(n_llm_h, n_llm_w, len(ref_ids))
    ref_ids += (base + ref_types).tolist()
    ref_ids += [16]
    assert new_token_ids == ref_ids

    image_placeholders = placeholders["image"]
    assert len(image_placeholders) == 2
    for placeholder in image_placeholders:
        compress_pad = COMPRESS_PAD_TO - 1 - placeholder.start_idx % COMPRESS_PAD_TO
        assert len(placeholder.tokens) == len(full) + compress_pad
        assert placeholder.tokens[:compress_pad] == [base + IMAGE_PAD] * compress_pad
        assert placeholder.tokens[compress_pad:] == full
        assert (
            new_token_ids[
                placeholder.start_idx : placeholder.start_idx + len(placeholder.tokens)
            ]
            == placeholder.tokens
        )
        is_embed = placeholder.is_embed
        assert is_embed.sum() == n_llm_h * n_llm_w
        assert not is_embed[:compress_pad].any()
        assert is_embed.tolist() == [
            token == base + IMAGE for token in placeholder.tokens
        ]
