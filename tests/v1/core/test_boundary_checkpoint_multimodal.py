# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Boundary checkpoints for requests with image or video placeholders."""

from types import SimpleNamespace

import pytest

from tests.v1.core.test_boundary_admission import drain
from tests.v1.core.test_boundary_admission import (
    initialize_hash as initialize_hash,  # noqa: F401
)
from tests.v1.core.test_boundary_admission import manager as make_manager
from tests.v1.core.test_prefix_caching import make_request
from vllm.multimodal.inputs import PlaceholderRange
from vllm.utils.hashing import sha256
from vllm.v1.core.boundary_checkpoint import (
    MIN_CHECKPOINT_ITEM_TOKENS,
    BoundaryCheckpointCache,
    checkpoint_end_allowed,
    content_token_ids,
)
from vllm.v1.request import RequestStatus


def image_request(name, image, offset, length=140, span=40):
    result = make_request(
        name,
        list(range(length)),
        16,
        sha256,
        mm_positions=[PlaceholderRange(offset=offset, length=span)],
        mm_hashes=[image],
    )
    result.max_tokens = result.sampling_params.max_tokens = 256
    return result


def publish(cache, producer, *ends):
    """Compute the producer's prompt and publish checkpoints at ``ends``.

    All but the last end are instruction boundaries; the last is the prompt.
    """
    if len(ends) > 1:
        producer.recurrent_instruction_boundary = ends[0]
    cache.get_computed_blocks(producer)
    computed = 0
    for index, end in enumerate(ends):
        assert cache.allocate_slots(producer, end - computed, num_lookahead_tokens=3)
        producer.num_computed_tokens = computed = end
        drain(cache)
        kind = "prompt" if index == len(ends) - 1 else "instruction"
        assert cache.publish_boundary_checkpoint(producer, end, kind=kind)
    producer.status = RequestStatus.FINISHED_STOPPED
    cache.free(producer)


def hit(cache, request):
    return cache.get_computed_blocks(request)[1]


def test_content_tokens_name_placeholder_spans_by_content():
    def fake(identifier):
        return SimpleNamespace(
            all_token_ids=list(range(20)),
            mm_features=[
                SimpleNamespace(
                    identifier=identifier,
                    mm_position=PlaceholderRange(offset=5, length=10),
                )
            ],
        )

    text = SimpleNamespace(all_token_ids=list(range(20)), mm_features=None)
    assert content_token_ids(text, 3, 12) == tuple(range(3, 12))
    a, b = fake("image-a"), fake("image-b")
    assert content_token_ids(a, 0, 5) == tuple(range(5))
    assert content_token_ids(a, 15, 20) == tuple(range(15, 20))
    assert content_token_ids(a, 0, 20) != content_token_ids(b, 0, 20)
    assert content_token_ids(a, 0, 20) == content_token_ids(fake("image-a"), 0, 20)
    # A window inside the span matches the same positions of the full view.
    assert content_token_ids(a, 7, 9) == content_token_ids(a, 0, 20)[7:9]
    # Salt (for example encoder precision) separates otherwise equal content.
    assert content_token_ids(a, 0, 20, b"x") != content_token_ids(a, 0, 20)
    assert all(1 << 30 <= value < 1 << 31 for value in content_token_ids(a, 5, 15))


@pytest.mark.parametrize(
    "modality,supported", [("image", True), ("video", True), ("audio", False)]
)
def test_supported_modalities(modality, supported):
    request = image_request("r", "image-a", 20)
    request.mm_features[0].modality = modality
    assert BoundaryCheckpointCache.supports_request(request) is supported


def test_image_checkpoint_restores_only_for_the_same_image():
    manager = make_manager()
    publish(manager, image_request("producer", "image-a", 20), 10, 140)
    # Same text and image: the whole prompt restores.
    assert hit(manager, image_request("same", "image-a", 20)) == 140
    # Different image: only the checkpoint that ends before the image applies.
    assert hit(manager, image_request("other", "image-b", 20)) == 10


def test_image_inside_the_final_partial_block_is_part_of_the_identity():
    manager = make_manager()
    # The image lies wholly after the last full hash block (128), so the block
    # hashes of both requests agree and only the tail comparison differs.
    publish(manager, image_request("producer", "image-a", 130, span=8), 140)
    assert hit(manager, image_request("same", "image-a", 130, span=8)) == 140
    assert hit(manager, image_request("other", "image-b", 130, span=8)) == 0


def fake_image(identifier, offset=5, length=40, total=60):
    return SimpleNamespace(
        all_token_ids=list(range(total)),
        mm_features=[
            SimpleNamespace(
                identifier=identifier,
                mm_position=PlaceholderRange(offset=offset, length=length),
            )
        ],
    )


def test_content_tokens_carry_independent_bits_per_position():
    # Two real PNGs whose processor hashes collided under the v1 scheme, where
    # a whole span was a function of one 30-bit seed (Discord report, #904).
    red = "2373f7d58a45946947a5179eff2f1a6fc5c6ccac30b9ef4e6999b17e54901057"
    blue = "cd07abfe9657e3a97e5a4c82b6baf82ab190e9a150aba160577c7e92b7b82219"
    salt = b'{"VLLM_GLM53_VISION_MXFP8": "1"}'
    a = content_token_ids(fake_image(red), 5, 45, salt)
    b = content_token_ids(fake_image(blue), 5, 45, salt)
    assert all(x != y for x, y in zip(a, b))
    # Positions are not an arithmetic progression of one seed.
    steps = {(y - x) % (1 << 30) for x, y in zip(a, a[1:])}
    assert len(steps) > 1
    # Windows crossing counter blocks match the full view.
    full = content_token_ids(fake_image(red), 0, 60)
    for start, end in ((3, 22), (20, 21), (21, 38), (44, 60)):
        assert content_token_ids(fake_image(red), start, end) == full[start:end]


def test_checkpoints_cannot_end_at_the_start_of_an_image():
    request = fake_image("image-a", offset=5, length=40)
    allowed = [checkpoint_end_allowed(request, end) for end in range(1, 60)]
    shallow = range(6, 5 + MIN_CHECKPOINT_ITEM_TOKENS)
    assert [end for end, ok in zip(range(1, 60), allowed) if not ok] == list(shallow)
    # A tiny item may end a checkpoint once it is complete.
    assert checkpoint_end_allowed(fake_image("icon", offset=5, length=2), 7)
    assert not checkpoint_end_allowed(fake_image("icon", offset=5, length=2), 6)


def test_shallow_image_endpoint_is_not_published():
    manager = make_manager()
    producer = image_request("producer", "image-a", 20)
    producer.recurrent_instruction_boundary = 22
    manager.get_computed_blocks(producer)
    assert manager.allocate_slots(producer, 22, num_lookahead_tokens=3)
    producer.num_computed_tokens = 22
    drain(manager)
    assert manager.publish_boundary_checkpoint(producer, 22, kind="instruction") is None
