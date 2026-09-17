# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 prompts preserve reference-format conversation semantics."""

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from tests.tokenizers_.test_deepseek_v4 import FakeHfTokenizer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tokenizers.deepseek_v41 import get_deepseek_v41_tokenizer
from vllm.tokenizers.deepseek_v41_encoding import (
    encode_messages,
    render_reasoning_effort,
)


@pytest.mark.parametrize(
    ("effort", "budget"),
    [("low", 50), ("high", 75), ("max", 100), ("xhigh", 75), (None, 75)],
)
def test_named_and_default_effort_render_official_numeric_budget(effort, budget):
    assert render_reasoning_effort(0, "thinking", effort).startswith(
        f"Reasoning Effort: {budget} "
    )


@pytest.mark.parametrize("budget", [1, 25, 50, 75, 100])
def test_numeric_effort_keeps_explicit_budget(budget):
    assert render_reasoning_effort(0, "thinking", budget).startswith(
        f"Reasoning Effort: {budget} "
    )


@pytest.mark.parametrize(("index", "mode"), [(0, "chat"), (1, "thinking")])
def test_reasoning_prefix_is_only_written_for_first_thinking_message(index, mode):
    assert render_reasoning_effort(index, mode, "high") == ""


def _tokenizer():
    return get_deepseek_v41_tokenizer(FakeHfTokenizer())


_FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures/deepseek_v41/encoding.json").read_text()
)


@pytest.mark.parametrize("case", _FIXTURES["cases"], ids=lambda case: case["name"])
def test_prompt_matches_reference_fixture(case):
    original = copy.deepcopy(case["messages"])
    request = ChatCompletionRequest(
        messages=case["messages"],
        tools=case.get("tools"),
    )
    prompt = _tokenizer().apply_chat_template(
        request.messages,
        tools=[tool.model_dump(exclude_none=True) for tool in request.tools]
        if request.tools
        else None,
        tokenize=False,
        **case["kwargs"],
    )
    assert prompt == case["expected"]
    assert case["messages"] == original


@pytest.mark.parametrize("location", ["tool", "function"])
@pytest.mark.parametrize("namespace", ["inventory", {"name": "inventory"}])
def test_direct_encoder_and_api_preserve_the_same_tool_identity(location, namespace):
    tool: dict[str, Any] = {"type": "function", "function": {"name": "lookup"}}
    call: dict[str, Any] = {
        "id": "call_a",
        "type": "function",
        "function": {"name": "lookup", "arguments": '{"sku":"A1"}'},
    }
    for item in (tool, call):
        (item if location == "tool" else item["function"])["namespace"] = namespace
    messages = [
        {"role": "system", "content": "Use tools.", "tools": [tool]},
        {"role": "user", "content": "Find A1."},
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "call_a", "content": "In stock."},
    ]
    original = copy.deepcopy(messages)
    direct = encode_messages(messages, thinking_mode="chat")
    request = ChatCompletionRequest(messages=messages)
    rendered = _tokenizer().apply_chat_template(
        request.messages,
        tokenize=False,
        thinking=False,
    )
    assert rendered == direct
    assert '"name": "inventory::lookup"' in rendered
    assert 'name="inventory::lookup"' in rendered
    assert "<tool_result>In stock.</tool_result>" in rendered
    assert messages == original


def test_reminder_does_not_enable_arbitrary_message_roles():
    with pytest.raises(ValueError, match="Invalid role: SYSTEM"):
        _tokenizer().apply_chat_template(
            [{"role": "SYSTEM", "content": "Hello"}],
            tokenize=False,
        )


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
