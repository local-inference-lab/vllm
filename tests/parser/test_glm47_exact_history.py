# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM parser output must re-render to exactly the generated text.

GLM-4.7/5.x chat templates emit an assistant turn as reasoning between
``<think>`` and ``</think>``, then content, then tool calls, adding no
whitespace of their own. When the parser returns every generated character,
a template that renders content verbatim reproduces the model's tokens, so
the next request can reuse the cached state of the whole response instead of
prefilling it again.
"""

import json
import random

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.parser_manager import ParserManager

GLM_MODEL = "zai-org/GLM-4.7"
QWEN_MODEL = "Qwen/Qwen3-32B"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string"},
                    "limit": {"type": "integer"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
            },
        },
    },
]

READ = (
    "<tool_call>read<arg_key>filePath</arg_key><arg_value>/repo/a.py</arg_value>"
    "<arg_key>limit</arg_key><arg_value>200</arg_value></tool_call>"
)
BASH = (
    "<tool_call>bash<arg_key>command</arg_key>"
    "<arg_value>ls -la\n</arg_value></tool_call>"
)

# Raw generations that follow the prompt's trailing <think>.
OUTPUTS = {
    "answer_trailing_newline": "Plan the fix.</think>Use move_to_end.\n",
    "answer_code_block": "Plan.</think>Fixed:\n\n```python\nx = 1\n```\n",
    "answer_leading_newlines": "Plan.</think>\n\nThe answer.",
    "reasoning_trailing_newlines": "Plan the fix.\n\n</think>Done.",
    "empty_reasoning": "</think>Done.\n",
    "content_before_tool": "Read it.</think>Let me read the file.\n" + READ,
    "whitespace_only_before_tool": "Read it.</think>\n" + READ,
    "two_tools": "Check both.</think>" + READ + BASH,
}


@pytest.fixture(scope="module")
def glm_tokenizer():
    from vllm.tokenizers import get_tokenizer

    return get_tokenizer(GLM_MODEL)


@pytest.fixture(scope="module")
def glm_parser_cls():
    return ParserManager.get_parser(
        tool_parser_name="glm47",
        reasoning_parser_name="glm45",
        enable_auto_tools=True,
    )


def _request() -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(
        {
            "model": "glm",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": TOOLS,
            "tool_choice": "auto",
        }
    )


def _generated_ids(tokenizer, raw: str) -> list[int]:
    stop = "<|observation|>" if "<tool_call>" in raw else "<|user|>"
    ids = tokenizer.encode(raw, add_special_tokens=False)
    return ids + [tokenizer.convert_tokens_to_ids(stop)]


def _render_turn(reasoning, content, calls) -> str:
    """The assistant turn after the prompt's <think>, as a GLM template
    that renders content verbatim emits it."""
    text = (reasoning or "") + "</think>" + (content or "")
    for name, arguments in calls:
        text += "<tool_call>" + name
        for key, value in json.loads(arguments).items():
            if not isinstance(value, str):
                value = json.dumps(value, ensure_ascii=False)
            text += f"<arg_key>{key}</arg_key><arg_value>{value}</arg_value>"
        text += "</tool_call>"
    return text


def _parse(parser_cls, tokenizer, raw):
    request = _request()
    ids = _generated_ids(tokenizer, raw)
    parser = parser_cls(tokenizer, request.tools)
    reasoning, content, calls = parser.parse(
        tokenizer.decode(ids, skip_special_tokens=True),
        request,
        enable_auto_tools=True,
        model_output_token_ids=ids,
    )
    return reasoning, content, [(c.name, c.arguments) for c in calls or []]


def _stream(parser_cls, tokenizer, raw, tokens_per_delta):
    request = _request()
    ids = _generated_ids(tokenizer, raw)
    parser = parser_cls(tokenizer, request.tools)
    reasoning, content = "", ""
    calls: dict[int, list[str]] = {}
    previous = ""
    for start in range(0, len(ids), tokens_per_delta):
        end = min(start + tokens_per_delta, len(ids))
        text = tokenizer.decode(ids[:end], skip_special_tokens=True)
        delta = parser.parse_delta(
            delta_text=text[len(previous) :],
            delta_token_ids=ids[start:end],
            request=request,
            prompt_token_ids=[tokenizer.convert_tokens_to_ids("<think>")],
            finished=end == len(ids),
        )
        previous = text
        if delta is None:
            continue
        reasoning += delta.reasoning or ""
        content += delta.content or ""
        for call in delta.tool_calls or []:
            slot = calls.setdefault(call.index, ["", ""])
            if call.function is not None:
                slot[0] += call.function.name or ""
                slot[1] += call.function.arguments or ""
    return (
        reasoning or None,
        content or None,
        [tuple(calls[index]) for index in sorted(calls)],
    )


@pytest.mark.parametrize("name", OUTPUTS)
def test_parse_output_rerenders_generated_text(glm_parser_cls, glm_tokenizer, name):
    raw = OUTPUTS[name]
    assert _render_turn(*_parse(glm_parser_cls, glm_tokenizer, raw)) == raw


@pytest.mark.parametrize("tokens_per_delta", [1, 2, 3, 4])
@pytest.mark.parametrize("name", OUTPUTS)
def test_streamed_output_matches_parse(
    glm_parser_cls, glm_tokenizer, name, tokens_per_delta
):
    raw = OUTPUTS[name]
    reasoning, content, calls = _stream(
        glm_parser_cls, glm_tokenizer, raw, tokens_per_delta
    )
    parsed_reasoning, parsed_content, parsed_calls = _parse(
        glm_parser_cls, glm_tokenizer, raw
    )
    assert (reasoning, content) == (parsed_reasoning, parsed_content)
    assert [name for name, _ in calls] == [name for name, _ in parsed_calls]
    assert [json.loads(args) for _, args in calls] == [
        json.loads(args) for _, args in parsed_calls
    ]
    assert _render_turn(reasoning, content, calls) == raw


@pytest.mark.parametrize("tokens_per_delta", [None, 1, 3])
def test_whitespace_after_tool_calls_is_not_content(
    glm_parser_cls, glm_tokenizer, tokens_per_delta
):
    # Content precedes all calls in a message, so text between or after
    # calls cannot be placed faithfully; it must not leak into content.
    raw = "Check.</think>" + READ + "\n" + BASH + "\n"
    if tokens_per_delta is None:
        _, content, calls = _parse(glm_parser_cls, glm_tokenizer, raw)
    else:
        _, content, calls = _stream(
            glm_parser_cls, glm_tokenizer, raw, tokens_per_delta
        )
    assert content is None
    assert [name for name, _ in calls] == ["read", "bash"]


def _random_turn(rng: random.Random) -> str:
    """A GLM turn whose every character the message format can represent."""
    pieces = ["Plan", " it", ".", "\n", "\n\n", " ", "```python\nx = 1\n```", "`y`"]

    def text() -> str:
        return "".join(rng.choice(pieces) for _ in range(rng.randint(0, 6)))

    turn = (text() + "</think>" if rng.random() < 0.9 else "</think>") + text()
    return turn + "".join(rng.choice([READ, BASH]) for _ in range(rng.randint(0, 2)))


@pytest.mark.parametrize("seed", range(40))
def test_random_turns_round_trip(glm_parser_cls, glm_tokenizer, seed):
    rng = random.Random(seed)
    raw = _random_turn(rng)
    parsed = _parse(glm_parser_cls, glm_tokenizer, raw)
    assert _render_turn(*parsed) == raw
    streamed = _stream(glm_parser_cls, glm_tokenizer, raw, rng.randint(1, 5))
    assert _render_turn(*streamed) == raw


def test_other_parsers_still_drop_whitespace_only_content():
    from vllm.tokenizers import get_tokenizer

    tokenizer = get_tokenizer(QWEN_MODEL)
    parser_cls = ParserManager.get_parser(
        tool_parser_name="qwen3_xml",
        reasoning_parser_name="qwen3",
        enable_auto_tools=True,
    )
    request = _request()
    raw = (
        "Plan.\n</think>\n\n<tool_call>\n<function=read>\n"
        "<parameter=filePath>\n/repo/a.py\n</parameter>\n</function>\n</tool_call>"
    )
    parser = parser_cls(tokenizer, request.tools)
    _, content, calls = parser.parse(raw, request, enable_auto_tools=True)
    assert content is None
    assert [call.name for call in calls or []] == ["read"]
