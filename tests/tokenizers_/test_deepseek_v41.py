# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy

import pytest

from tests.tokenizers_.test_deepseek_v4 import FakeHfTokenizer
from vllm.tokenizers.deepseek_v41 import get_deepseek_v41_tokenizer


@pytest.mark.parametrize(
    ("role", "responses_type"),
    [("user", "input_text"), ("assistant", "output_text")],
)
def test_responses_text_parts_match_chat_text_parts(role, responses_type):
    tokenizer = get_deepseek_v41_tokenizer(FakeHfTokenizer())
    responses_message = {
        "role": role,
        "content": [{"type": responses_type, "text": "hello"}],
    }
    chat_message = {
        "role": role,
        "content": [{"type": "text", "text": "hello"}],
    }
    original = copy.deepcopy(responses_message)
    assert tokenizer.apply_chat_template(
        [responses_message], tokenize=False, thinking=False
    ) == tokenizer.apply_chat_template([chat_message], tokenize=False, thinking=False)
    assert responses_message == original
