# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""An empty tools array must validate like an omitted field (OpenAI parity);
a tool_choice that forces a call with zero tools must still be rejected."""

import pydantic
import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

pytestmark = pytest.mark.cpu_test

MESSAGES = [{"role": "user", "content": "hello"}]

TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {}},
    },
}

NAMED_CHOICE = {"type": "function", "function": {"name": "get_weather"}}


@pytest.mark.parametrize(
    "tool_choice",
    [
        pytest.param("absent", id="tool_choice-absent"),
        pytest.param("auto", id="tool_choice-auto"),
        pytest.param("none", id="tool_choice-none"),
        pytest.param(None, id="tool_choice-null"),
    ],
)
def test_empty_tools_accepted_like_omitted(tool_choice):
    """tools=[] with a non-forcing tool_choice validates like a request
    that sent neither field."""
    data = {"messages": MESSAGES, "model": "x", "tools": []}
    if tool_choice != "absent":
        data["tool_choice"] = tool_choice

    request = ChatCompletionRequest.model_validate(data)

    assert request.tools is None
    assert request.tool_choice == "none"


@pytest.mark.parametrize(
    "tool_choice",
    [
        pytest.param(NAMED_CHOICE, id="tool_choice-named"),
        pytest.param("required", id="tool_choice-required"),
    ],
)
def test_empty_tools_with_forcing_tool_choice_rejected(tool_choice):
    """A forced tool call with zero tools is a genuinely invalid request."""
    data = {
        "messages": MESSAGES,
        "model": "x",
        "tools": [],
        "tool_choice": tool_choice,
    }

    with pytest.raises(pydantic.ValidationError, match="`tools` must be set"):
        ChatCompletionRequest.model_validate(data)


def test_null_tools_accepted():
    request = ChatCompletionRequest.model_validate(
        {"messages": MESSAGES, "model": "x", "tools": None}
    )

    assert request.tools is None
    assert request.tool_choice == "none"


def test_tools_default_tool_choice_auto():
    """Existing behavior: providing tools defaults tool_choice to auto."""
    request = ChatCompletionRequest.model_validate(
        {"messages": MESSAGES, "model": "x", "tools": [TOOL]}
    )

    assert request.tools is not None
    assert request.tool_choice == "auto"


def test_tool_choice_without_tools_rejected():
    """Existing behavior: tool_choice=auto without any tools field."""
    with pytest.raises(pydantic.ValidationError, match="`tools` must be set"):
        ChatCompletionRequest.model_validate(
            {"messages": MESSAGES, "model": "x", "tool_choice": "auto"}
        )
