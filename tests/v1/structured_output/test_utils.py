# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.core.sched.output import GrammarOutput
from vllm.v1.structured_output import utils
from vllm.v1.structured_output.backend_xgrammar import (
    has_xgrammar_unsupported_json_features,
)

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def unsupported_string_schemas():
    return [
        {"type": "string", "format": "non_existing_format"},
    ]


@pytest.fixture
def unsupported_integer_schemas():
    return [
        {"type": "integer", "multipleOf": 120},
    ]


@pytest.fixture
def unsupported_number_schemas():
    return [
        {"type": "number", "multipleOf": 120},
    ]


@pytest.fixture
def unsupported_array_schemas():
    return [
        {"type": "array", "uniqueItems": True},
        {"type": "array", "contains": {"type": "string"}},
        {"type": "array", "minContains": 1},
        {"type": "array", "maxContains": 5},
    ]


@pytest.fixture
def unsupported_object_schemas():
    return [
        {"type": "object", "propertyNames": {"pattern": "^[a-z]+$"}},
        {"type": "object", "patternProperties": {"^S": {"type": "string"}}},
    ]


@pytest.fixture
def supported_schema():
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "email": {"type": "string", "format": "email"},
            "status": {"type": "string"},
            "scores": {"type": "array", "items": {"type": "number"}},
            "car_type": {"type": "string", "enum": ["sedan", "suv", "truck"]},
            "car_brand": {"type": "string", "pattern": "^[a-zA-Z]+$"},
            "short_description": {"type": "string", "maxLength": 50},
            "mileage": {"type": "number", "minimum": 0, "maximum": 1000000},
            "model_year": {
                "type": "integer",
                "exclusiveMinimum": 1900,
                "exclusiveMaximum": 2100,
            },
            "long_description": {"type": "string", "minLength": 50, "maxLength": 2000},
            "address": {
                "type": "object",
                "properties": {
                    "street": {"type": "string"},
                    "city": {"type": "string"},
                },
            },
        },
        "minProperties": 1,
        "maxProperties": 100,
    }


@pytest.mark.parametrize(
    "schema_type",
    [
        "unsupported_string_schemas",
        "unsupported_integer_schemas",
        "unsupported_number_schemas",
        "unsupported_array_schemas",
        "unsupported_object_schemas",
    ],
)
def test_unsupported_json_features_by_type(schema_type, request):
    schemas = request.getfixturevalue(schema_type)
    for schema in schemas:
        assert has_xgrammar_unsupported_json_features(schema), (
            f"Schema should be unsupported: {schema}"
        )


def test_supported_json_features(supported_schema):
    assert not has_xgrammar_unsupported_json_features(supported_schema), (
        "Schema should be supported"
    )


def test_apply_grammar_bitmask_preserves_source_offsets_after_draft_trimming(
    monkeypatch,
):
    """A trimmed request must not shift another request's grammar rows.

    The scheduler serializes masks at its scheduled speculative width. A worker
    may trim grammar-invalid drafts before applying those masks, so source and
    destination offsets must be advanced with their respective widths.
    """
    scheduler_output = SimpleNamespace(
        scheduled_spec_decode_tokens={
            "trimmed-request": [1],
            "full-request": [2, 3, 4],
        }
    )
    grammar_output = GrammarOutput(
        structured_output_request_ids=["trimmed-request", "full-request"],
        grammar_bitmask=np.array(
            [[10], [11], [12], [13], [20], [21], [22], [23]],
            dtype=np.int32,
        ),
        num_spec_tokens=[3, 3],
        has_bonus_token=[True, True],
        num_invalid_spec_tokens=[0, 0],
    )
    input_batch = SimpleNamespace(req_ids=["trimmed-request", "full-request"])
    logits = torch.zeros((6, 32))
    applied_bitmask = None

    def capture_bitmask(logits, bitmask, indices):
        nonlocal applied_bitmask
        applied_bitmask = bitmask.clone()
        assert indices is None

    monkeypatch.setattr(
        utils,
        "xgr",
        SimpleNamespace(apply_token_bitmask_inplace=capture_bitmask),
    )
    monkeypatch.setattr(utils, "PIN_MEMORY", False)

    utils.apply_grammar_bitmask(
        scheduler_output,
        grammar_output,
        input_batch,
        logits,
    )

    assert applied_bitmask is not None
    assert applied_bitmask[:, 0].tolist() == [10, 13, 20, 21, 22, 23]


def test_apply_grammar_bitmask_skips_omitted_diffusion_bonus_rows(monkeypatch):
    """An omitted source bonus row must not consume the next request's mask."""
    scheduler_output = SimpleNamespace(
        scheduled_spec_decode_tokens={
            "diffusion-request": [1],
            "later-request": [2, 3, 4],
        }
    )
    grammar_output = GrammarOutput(
        structured_output_request_ids=["diffusion-request", "later-request"],
        grammar_bitmask=np.array([[10], [20], [21], [22], [23]], dtype=np.int32),
        num_spec_tokens=[1, 3],
        has_bonus_token=[False, True],
        num_invalid_spec_tokens=[0, 0],
    )
    input_batch = SimpleNamespace(req_ids=["diffusion-request", "later-request"])
    logits = torch.zeros((6, 32))
    applied_bitmask = None
    applied_indices = None

    def capture_bitmask(logits, bitmask, indices):
        nonlocal applied_bitmask, applied_indices
        applied_bitmask = bitmask.clone()
        applied_indices = indices

    monkeypatch.setattr(
        utils,
        "xgr",
        SimpleNamespace(apply_token_bitmask_inplace=capture_bitmask),
    )
    monkeypatch.setattr(utils, "PIN_MEMORY", False)

    utils.apply_grammar_bitmask(
        scheduler_output,
        grammar_output,
        input_batch,
        logits,
    )

    assert applied_bitmask is not None
    assert applied_bitmask[:, 0].tolist() == [10, -1, 20, 21, 22, 23]
    assert applied_indices == [0, 2, 3, 4, 5]
