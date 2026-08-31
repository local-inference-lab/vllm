# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-4.7 parser for reasoning and tool calls.

GLM-4.7 uses XML-like tool calls::

    <tool_call>func_name<arg_key>key</arg_key><arg_value>value</arg_value></tool_call>

The function name can be followed directly by the first ``<arg_key>`` tag,
and tool calls may have no arguments.
"""

from __future__ import annotations

import functools
import json
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

THINK_START = "<think>"
THINK_END = "</think>"
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
ARG_KEY_START = "<arg_key>"
ARG_KEY_END = "</arg_key>"
ARG_VALUE_START = "<arg_value>"
ARG_VALUE_END = "</arg_value>"


def _skip_whitespace(text: str, offset: int) -> int:
    while offset < len(text) and text[offset].isspace():
        offset += 1
    return offset


def _glm47_arg_converter(raw_args: str, partial: bool) -> str:
    params: dict[str, object] = {}
    cursor = 0

    while True:
        key_start = raw_args.find(ARG_KEY_START, cursor)
        if key_start < 0:
            break
        key_value_start = key_start + len(ARG_KEY_START)

        key_end_search = key_value_start
        key_end = -1
        value_start = -1
        while True:
            candidate = raw_args.find(ARG_KEY_END, key_end_search)
            if candidate < 0:
                break
            after_key = _skip_whitespace(raw_args, candidate + len(ARG_KEY_END))
            if raw_args.startswith(ARG_VALUE_START, after_key):
                key_end = candidate
                value_start = after_key + len(ARG_VALUE_START)
                break
            key_end_search = candidate + 1

        if key_end < 0:
            break

        key = raw_args[key_value_start:key_end].strip()
        value_end_search = value_start
        value_end = -1
        next_cursor = -1
        pending_value_end = -1
        while True:
            candidate = raw_args.find(ARG_VALUE_END, value_end_search)
            if candidate < 0:
                break
            after_value = _skip_whitespace(
                raw_args,
                candidate + len(ARG_VALUE_END),
            )
            if raw_args.startswith(ARG_KEY_START, after_value):
                value_end = candidate
                next_cursor = after_value
                break
            if after_value == len(raw_args):
                if partial:
                    pending_value_end = candidate
                else:
                    value_end = candidate
                    next_cursor = after_value
                break
            value_end_search = candidate + 1

        if value_end >= 0:
            params[key] = raw_args[value_start:value_end]
            cursor = next_cursor
            if cursor >= len(raw_args):
                break
            continue

        if partial and key:
            partial_end = pending_value_end if pending_value_end >= 0 else len(raw_args)
            params[key] = raw_args[value_start:partial_end]
        break

    return json.dumps(params, ensure_ascii=False)


@functools.cache
def glm47_moe_config(thinking: bool = True) -> ParserEngineConfig:
    arg_tag_transitions = {
        (ParserState.TOOL_NAME, "ARG_KEY_START"): Transition(
            ParserState.TOOL_BETWEEN,
            (EventType.ARG_VALUE_CHUNK,),
        ),
        (ParserState.TOOL_BETWEEN, "ARG_KEY_START"): Transition(
            ParserState.TOOL_BETWEEN,
            (EventType.ARG_VALUE_CHUNK,),
        ),
        (ParserState.TOOL_BETWEEN, "ARG_KEY_END"): Transition(
            ParserState.TOOL_BETWEEN,
            (EventType.ARG_VALUE_CHUNK,),
        ),
        (ParserState.TOOL_BETWEEN, "ARG_VALUE_START"): Transition(
            ParserState.TOOL_ARGS,
            (EventType.ARG_VALUE_CHUNK,),
        ),
        (ParserState.TOOL_ARGS, "ARG_VALUE_END"): Transition(
            ParserState.TOOL_ARG_END_PENDING,
            (EventType.ARG_VALUE_CHUNK,),
        ),
        (ParserState.TOOL_ARG_END_PENDING, "ARG_VALUE_END"): Transition(
            ParserState.TOOL_ARG_END_PENDING,
            (EventType.ARG_VALUE_CHUNK,),
        ),
        (ParserState.TOOL_ARG_END_PENDING, "ARG_KEY_START"): Transition(
            ParserState.TOOL_BETWEEN,
            (EventType.ARG_VALUE_CHUNK,),
        ),
    }

    reasoning_terminals = (
        {
            "THINK_START": THINK_START,
            "THINK_END": THINK_END,
        }
        if thinking
        else {}
    )
    reasoning_token_id_terminals = (
        {
            "THINK_START": THINK_START,
            "THINK_END": THINK_END,
        }
        if thinking
        else {}
    )
    reasoning_transitions = (
        {
            (ParserState.CONTENT, "THINK_START"): Transition(
                ParserState.REASONING,
                (EventType.REASONING_START,),
            ),
            (ParserState.REASONING, "THINK_END"): Transition(
                ParserState.CONTENT,
                (EventType.REASONING_END,),
            ),
            (ParserState.CONTENT, "THINK_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
        }
        if thinking
        else {}
    )

    return ParserEngineConfig(
        name="glm47_moe",
        initial_state=ParserState.REASONING if thinking else ParserState.CONTENT,
        terminals={
            **reasoning_terminals,
            "TOOL_START": TOOL_CALL_START,
            "TOOL_END": TOOL_CALL_END,
            "ARG_KEY_START": ARG_KEY_START,
            "ARG_KEY_END": ARG_KEY_END,
            "ARG_VALUE_START": ARG_VALUE_START,
            "ARG_VALUE_END": ARG_VALUE_END,
        },
        token_id_terminals={
            **reasoning_token_id_terminals,
            "TOOL_START": TOOL_CALL_START,
            "TOOL_END": TOOL_CALL_END,
        },
        transitions={
            **reasoning_transitions,
            (ParserState.REASONING, "THINK_START"): Transition(
                ParserState.REASONING,
                (),
            ),
            (ParserState.REASONING, "TOOL_START"): Transition(
                ParserState.TOOL_NAME,
                (EventType.REASONING_END, EventType.TOOL_CALL_START),
            ),
            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_NAME, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
            (ParserState.TOOL_BETWEEN, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
            (ParserState.TOOL_ARG_END_PENDING, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
            **arg_tag_transitions,
        },
        non_whitespace_transitions={
            ParserState.TOOL_ARG_END_PENDING: Transition(
                ParserState.TOOL_ARGS,
                (EventType.ARG_VALUE_CHUNK,),
            )
        },
        content_events={
            ParserState.CONTENT: EventType.TEXT_CHUNK,
            ParserState.REASONING: EventType.REASONING_CHUNK,
            ParserState.TOOL_NAME: EventType.TOOL_NAME,
            ParserState.TOOL_ARGS: EventType.ARG_VALUE_CHUNK,
            ParserState.TOOL_BETWEEN: EventType.ARG_VALUE_CHUNK,
            ParserState.TOOL_ARG_END_PENDING: EventType.ARG_VALUE_CHUNK,
        },
        arg_converter=_glm47_arg_converter,
        stream_arg_deltas=True,
        tool_args_json=False,
        validate_tool_names=True,
    )


class Glm47MoeParser(ParserEngine):
    """GLM-4.7 parser backed by the declarative parser engine."""

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        thinking = chat_kwargs.get("thinking", None)
        enable_thinking = chat_kwargs.get("enable_thinking", None)
        self.thinking_enabled = (
            True
            if thinking is None and enable_thinking is None
            else bool(thinking) or bool(enable_thinking)
        )
        kwargs.setdefault(
            "parser_engine_config",
            glm47_moe_config(thinking=self.thinking_enabled),
        )
        super().__init__(tokenizer, tools, **kwargs)

    def _emit_name_delta(self, idx: int, deltas, name: str | None) -> None:
        if name is not None:
            name = name.strip()
        super()._emit_name_delta(idx, deltas, name)

    def _handle_tool_end(self, event, deltas) -> None:
        idx = event.tool_index
        if 0 <= idx < len(self._tool_slots):
            self._tool_slots[idx].name = self._tool_slots[idx].name.strip()
        super()._handle_tool_end(event, deltas)

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        if not self.thinking_enabled:
            return True
        return super().is_reasoning_end(input_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if not self.thinking_enabled:
            return input_ids
        return super().extract_content_ids(input_ids)

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        if not self.thinking_enabled:
            return None, model_output
        return super().extract_reasoning(model_output, request)
