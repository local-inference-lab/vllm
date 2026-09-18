# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for tool_calls Iterable → list materialisation.

Regression tests for https://github.com/vllm-project/vllm/issues/34792.

Setting VLLM_LOGGING_LEVEL=debug caused tool calling to break for Mistral
models because:
  1. The OpenAI Python SDK types tool_calls as Iterable[...] in
     ChatCompletionAssistantMessageParam.
  2. Pydantic v2, when validating from Python objects (not from raw JSON),
     wraps Iterable fields in a one-shot lazy iterator.
  3. Debug logging called model_dump_json() which consumed that iterator.
  4. The Mistral tokenizer then saw empty tool_calls and raised
     "ValueError: Unexpected tool call id ...".
"""

import copy

import pytest
from pydantic import ValidationError

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.exceptions import VLLMValidationError


def _make_tool_call(tc_id: str, name: str, args: str) -> dict:
    return {
        "id": tc_id,
        "type": "function",
        "function": {"name": name, "arguments": args},
    }


def _make_request(messages: list) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test-model",
        messages=messages,
    )


def test_tool_calls_list_preserved_after_model_dump():
    """tool_calls in assistant messages must be readable after model_dump_json.

    When the request is built from Python dicts (as in the Anthropic → OpenAI
    conversion path), Pydantic v2 previously wrapped the Iterable tool_calls
    in a one-shot iterator.  model_dump_json() consumed it, leaving subsequent
    readers (e.g. the Mistral tokenizer) with an empty sequence.
    """
    tool_call = _make_tool_call("call_abc123", "get_weather", '{"city": "Paris"}')
    messages = [
        {"role": "user", "content": "What is the weather in Paris?"},
        {"role": "assistant", "content": None, "tool_calls": [tool_call]},
        {
            "role": "tool",
            "tool_call_id": "call_abc123",
            "content": '{"temperature": 20}',
        },
    ]

    req = _make_request(messages)

    # Simulate debug logging: serialize the model (this was the trigger)
    _ = req.model_dump_json()

    # The assistant message must still have accessible tool_calls afterwards
    assistant_msg = req.messages[1]
    assert isinstance(assistant_msg, dict)
    tool_calls = assistant_msg.get("tool_calls")
    assert tool_calls is not None, "tool_calls must not be None after model_dump_json"
    assert isinstance(tool_calls, list), "tool_calls must be a list"
    assert len(tool_calls) > 0, "tool_calls must not be empty after model_dump_json"


def test_tool_calls_from_generator_are_materialised():
    """tool_calls passed as a generator must be converted to list on validation."""
    tool_call = _make_tool_call("call_gen1", "search", '{"query": "vllm"}')

    def tool_calls_gen():
        yield tool_call

    messages = [
        {"role": "user", "content": "Search for vllm"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls_gen(),  # one-shot generator
        },
    ]

    req = _make_request(messages)
    assistant_msg = req.messages[1]
    assert isinstance(assistant_msg, dict)

    # Iterate twice — must not raise or return empty on second pass
    tool_calls_first = list(assistant_msg.get("tool_calls", []))
    tool_calls_second = list(assistant_msg.get("tool_calls", []))

    assert len(tool_calls_first) == 1, "First read must return the tool call"
    assert len(tool_calls_second) == 1, "Second read must also return the tool call"


def test_tool_calls_list_passthrough():
    """tool_calls already provided as a list must remain a list."""
    tool_call = _make_tool_call("call_list1", "calculate", '{"expr": "2+2"}')
    messages = [
        {"role": "user", "content": "Calculate 2+2"},
        {"role": "assistant", "content": None, "tool_calls": [tool_call]},
    ]

    req = _make_request(messages)
    assistant_msg = req.messages[1]
    assert isinstance(assistant_msg, dict)
    assert isinstance(assistant_msg.get("tool_calls"), list)


def test_messages_without_tool_calls_unaffected():
    """Messages without tool_calls must be handled correctly."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    req = _make_request(messages)
    # None of the messages should have tool_calls injected
    for msg in req.messages:
        assert isinstance(msg, dict)
        assert msg.get("tool_calls") is None or msg.get("tool_calls") == []


@pytest.mark.parametrize("num_tool_calls", [1, 3])
def test_multiple_tool_calls_materialised(num_tool_calls: int):
    """Multiple tool calls in a single message are all preserved."""
    tool_calls = [
        _make_tool_call(f"call_{i}", f"func_{i}", f'{{"arg": {i}}}')
        for i in range(num_tool_calls)
    ]
    messages = [
        {"role": "user", "content": "Do things"},
        {"role": "assistant", "content": None, "tool_calls": iter(tool_calls)},
    ]

    req = _make_request(messages)
    assistant_msg = req.messages[1]
    assert isinstance(assistant_msg, dict)

    result_tool_calls = assistant_msg.get("tool_calls")
    assert isinstance(result_tool_calls, list)
    assert len(result_tool_calls) == num_tool_calls

    # Verify after model_dump_json too
    _ = req.model_dump_json()
    assert len(assistant_msg.get("tool_calls", [])) == num_tool_calls


@pytest.mark.parametrize("location", ["tool", "function"])
@pytest.mark.parametrize("namespace", ["inventory", {"name": "inventory"}])
@pytest.mark.parametrize("iterator", [False, True])
@pytest.mark.parametrize("tools_collection", [list, tuple, iter])
def test_namespaces_survive_definitions_choices_and_history(
    location, namespace, iterator, tools_collection
):
    def qualify(item, value):
        target = item if location == "tool" else item["function"]
        target["namespace"] = value
        return item

    tools = [
        qualify({"type": "function", "function": {"name": "lookup"}}, namespace),
        qualify({"type": "function", "function": {"name": "lookup"}}, "billing"),
    ]
    calls = [qualify(_make_tool_call("call_1", "lookup", "{}"), namespace)]
    payload = {
        "model": "DeepSeek-V4.1-Flash",
        "tools": tools,
        "tool_choice": qualify(
            {"type": "function", "function": {"name": "lookup"}}, namespace
        ),
        "messages": [
            {"role": "user", "content": "Look up stock."},
            {"role": "assistant", "content": None, "tool_calls": calls},
        ],
    }
    original = copy.deepcopy(payload)
    payload["tools"] = tools_collection(tools)
    if iterator:
        payload["messages"][1]["tool_calls"] = iter(calls)
    request = ChatCompletionRequest.model_validate(payload)
    assert [tool.function.name for tool in request.tools] == [
        "inventory::lookup",
        "billing::lookup",
    ]
    assert request.tool_choice.function.name == "inventory::lookup"
    assert request.messages[1]["tool_calls"][0]["function"]["name"] == (
        "inventory::lookup"
    )
    serialized = request.model_dump_json()
    assert request.model_dump_json() == serialized
    restored = ChatCompletionRequest.model_validate_json(serialized)
    assert restored.model_dump_json() == serialized
    if not iterator and tools_collection is list:
        assert payload == original
    assert tools == original["tools"]
    assert calls == original["messages"][1]["tool_calls"]


def test_namespace_description_is_preserved_exactly_once():
    tool = {
        "type": "function",
        "namespace": {"name": "inventory", "description": "Stock operations."},
        "function": {
            "name": "inventory::lookup",
            "description": "Find a SKU.",
            "strict": True,
        },
    }
    original = copy.deepcopy(tool)
    parsed = ChatCompletionToolsParam.model_validate(tool)
    assert parsed.function.name == "inventory::lookup"
    assert parsed.function.description == "Stock operations.\nFind a SKU."
    assert parsed.function.strict is True
    assert tool == original
    assert ChatCompletionToolsParam.model_validate(
        parsed.model_dump()
    ).model_dump() == (parsed.model_dump())


@pytest.mark.parametrize("collection", [list, tuple])
def test_iterable_history_preserves_namespaced_tool_identity(collection):
    call = _make_tool_call("call_history", "lookup", "{}")
    call["namespace"] = "inventory"
    messages = [{"role": "assistant", "content": None, "tool_calls": [call]}]
    original = copy.deepcopy(messages)
    request = ChatCompletionRequest.model_validate(
        {"model": "test-model", "messages": collection(messages)}
    )
    assert request.messages[0]["tool_calls"][0]["function"]["name"] == (
        "inventory::lookup"
    )
    serialized = request.model_dump_json()
    restored = ChatCompletionRequest.model_validate_json(serialized)
    assert restored.messages[0]["tool_calls"][0]["function"]["name"] == (
        "inventory::lookup"
    )
    assert request.model_dump_json() == serialized
    assert messages == original


@pytest.mark.parametrize("messages", ["text", b"text", bytearray(b"text"), {}])
def test_invalid_history_collection_remains_a_client_error(messages):
    with pytest.raises(ValidationError) as error:
        ChatCompletionRequest.model_validate(
            {"model": "test-model", "messages": messages}
        )
    assert any(item["loc"] == ("messages",) for item in error.value.errors())


@pytest.mark.parametrize("tool_calls", [1, False, 1.5])
def test_invalid_tool_calls_shape_reports_the_field(tool_calls):
    with pytest.raises(ValidationError) as error:
        _make_request(
            [{"role": "assistant", "content": None, "tool_calls": tool_calls}]
        )
    assert any("tool_calls" in item["loc"] for item in error.value.errors())


@pytest.mark.parametrize(
    "outer, inner, expected",
    [
        (None, "Stock operations.", "Stock operations.\nFind a SKU."),
        ("Outer description.", "Inner description.", "Outer description.\nFind a SKU."),
        ("", "Inner description.", "Find a SKU."),
    ],
)
def test_consistent_namespaces_preserve_first_non_null_description(
    outer, inner, expected
):
    tool = {
        "type": "function",
        "namespace": {"name": "inventory", "description": outer},
        "function": {
            "name": "lookup",
            "namespace": {"name": "inventory", "description": inner},
            "description": "Find a SKU.",
        },
    }
    original = copy.deepcopy(tool)
    parsed = ChatCompletionToolsParam.model_validate(tool)
    assert parsed.function.name == "inventory::lookup"
    assert parsed.function.description == expected
    assert tool == original
    assert ChatCompletionToolsParam.model_validate(parsed.model_dump()) == parsed


@pytest.mark.parametrize("tool_choice", ["none", "auto", "required"])
@pytest.mark.parametrize(
    "namespace, name",
    [
        ("inventory", "billing::lookup"),
        ("a::b", "lookup"),
        ("", "lookup"),
        ({}, "lookup"),
        ({"name": 7}, "lookup"),
        (False, "lookup"),
        ("inventory", "inventory::x::lookup"),
        ("inventory", "inventory::"),
    ],
)
def test_invalid_namespaces_are_client_errors(namespace, name, tool_choice):
    with pytest.raises(ValidationError, match="[Nn]amespace"):
        ChatCompletionRequest.model_validate(
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [
                    {
                        "type": "function",
                        "namespace": namespace,
                        "function": {"name": name},
                    }
                ],
                "tool_choice": tool_choice,
            }
        )


def test_conflicting_namespace_locations_are_rejected():
    with pytest.raises(ValidationError, match="Conflicting tool namespaces"):
        ChatCompletionToolsParam.model_validate(
            {
                "type": "function",
                "namespace": "inventory",
                "function": {"name": "lookup", "namespace": "billing"},
            }
        )


def test_consistent_namespace_locations_do_not_duplicate_the_prefix():
    parsed = ChatCompletionToolsParam.model_validate(
        {
            "type": "function",
            "namespace": {"name": "inventory"},
            "function": {"name": "inventory::lookup", "namespace": "inventory"},
        }
    )
    assert parsed.function.name == "inventory::lookup"


def test_conflicting_history_namespace_is_not_silently_dropped():
    call = _make_tool_call("call_a", "inventory::lookup", "{}")
    call["namespace"] = "billing"
    with pytest.raises(ValidationError, match="Conflicting tool namespaces"):
        _make_request([{"role": "assistant", "content": None, "tool_calls": [call]}])


def test_bare_choice_cannot_select_a_namespaced_tool():
    with pytest.raises(VLLMValidationError, match="does not match"):
        ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "lookup"}],
                "tools": [
                    {
                        "type": "function",
                        "namespace": "inventory",
                        "function": {"name": "lookup"},
                    }
                ],
                "tool_choice": {"type": "function", "function": {"name": "lookup"}},
            }
        )


def test_plain_and_already_qualified_names_keep_their_spelling():
    for name in ("lookup", "inventory::lookup", "opaque::legacy::name"):
        tool = {"type": "function", "function": {"name": name}}
        assert ChatCompletionToolsParam.model_validate(tool).function.name == name
