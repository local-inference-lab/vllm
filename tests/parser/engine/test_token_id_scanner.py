# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for TokenIDScanner."""

from unittest.mock import MagicMock

import pytest

from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)
from vllm.parser.engine.streaming_parser_engine import StreamingParserEngine
from vllm.parser.engine.token_id_scanner import (
    DROP_TERMINAL,
    PreLexedTerminal,
    TextChunk,
    TokenIDScanner,
)
from vllm.parser.gemma4 import gemma4_config

CHANNEL_START = "<|channel>"
CHANNEL_END = "<channel|>"
CHANNEL_START_ID = 100
CHANNEL_END_ID = 101
REGULAR_TOKEN_ID = 200
TOOL_START = "<tool_call>"
TOOL_END = "</tool_call>"
TOOL_START_ID = 110
TOOL_END_ID = 111


@pytest.fixture
def tokenizer():
    tok = MagicMock()
    tok.get_vocab.return_value = {
        CHANNEL_START: CHANNEL_START_ID,
        CHANNEL_END: CHANNEL_END_ID,
    }
    tok.decode.side_effect = lambda ids: {
        CHANNEL_START_ID: CHANNEL_START,
        CHANNEL_END_ID: CHANNEL_END,
        REGULAR_TOKEN_ID: "regular",
    }.get(ids[0], f"<unk:{ids[0]}>")
    return tok


@pytest.fixture
def scanner(tokenizer):
    return TokenIDScanner(
        token_id_to_terminal={
            CHANNEL_START_ID: "THINK_START",
            CHANNEL_END_ID: "THINK_END",
        },
        tokenizer=tokenizer,
    )


class TestJoinDecodedTextReturnsStr:
    """_join_decoded_text always returns str."""

    @pytest.fixture
    def bare_scanner(self):
        return TokenIDScanner({}, tokenizer=None)

    def test_mixed_items(self, bare_scanner):
        items = [
            TextChunk("hello "),
            PreLexedTerminal("TOOL_START", 42, "<tool_call>"),
            TextChunk(" world"),
        ]
        result = bare_scanner._join_decoded_text(items)
        assert isinstance(result, str)
        assert result == "hello <tool_call> world"

    def test_empty_list(self, bare_scanner):
        result = bare_scanner._join_decoded_text([])
        assert isinstance(result, str)
        assert result == ""

    def test_only_text_chunks(self, bare_scanner):
        result = bare_scanner._join_decoded_text([TextChunk("abc"), TextChunk("def")])
        assert result == "abcdef"


class TestHoldbackTextRecovery:
    def test_holdback_text_with_special_token_text_absent(self, scanner):
        """Terminal deferred when its text is absent from delta_text."""
        result = scanner.scan(
            delta_text="processed is appropriate.",
            delta_token_ids=[CHANNEL_END_ID],
        )

        assert len(result) == 0

        result2 = scanner.scan(
            delta_text="<channel|>Understood.",
            delta_token_ids=[20, 21],
        )
        pre_lexed = [r for r in result2 if isinstance(r, PreLexedTerminal)]
        assert len(pre_lexed) == 1
        assert pre_lexed[0].terminal == "THINK_END"
        texts = [r.text for r in result2 if isinstance(r, TextChunk)]
        combined = "".join(texts)
        assert "processed is appropriate." in combined
        assert "Understood." in combined

    def test_holdback_text_with_special_token_text_present(self, scanner):
        """Hold-back text + special token text both in delta_text."""
        result = scanner.scan(
            delta_text="holdback text<channel|>",
            delta_token_ids=[CHANNEL_END_ID],
        )

        assert len(result) == 2
        assert isinstance(result[0], TextChunk)
        assert result[0].text == "holdback text"
        assert isinstance(result[1], PreLexedTerminal)
        assert result[1].terminal == "THINK_END"

    def test_no_holdback_text(self, scanner):
        """delta_text is exactly the special token text."""
        result = scanner.scan(
            delta_text="<channel|>",
            delta_token_ids=[CHANNEL_END_ID],
        )

        assert len(result) == 1
        assert isinstance(result[0], PreLexedTerminal)
        assert result[0].terminal == "THINK_END"

    def test_empty_delta_text(self, scanner):
        """Empty delta_text defers the terminal until text arrives."""
        result = scanner.scan(
            delta_text="",
            delta_token_ids=[CHANNEL_END_ID],
        )

        assert len(result) == 0

        flushed = scanner.flush_pending()
        assert len(flushed) == 1
        assert isinstance(flushed[0], PreLexedTerminal)
        assert flushed[0].terminal == "THINK_END"

    def test_empty_delta_text_drops_individual_decode_text(self, tokenizer):
        """Empty delta_text with multiple tokens: all results deferred."""
        tool_start_id = 400
        tok_a = 201
        tok_b = 202
        tokenizer.decode.side_effect = lambda ids: {
            tool_start_id: "<|tool_call>",
            tok_a: "call:",
            tok_b: "get_weather",
        }.get(ids[0], "?")

        scanner = TokenIDScanner(
            token_id_to_terminal={tool_start_id: "TOOL_START"},
            tokenizer=tokenizer,
        )

        result = scanner.scan(
            delta_text="",
            delta_token_ids=[tool_start_id, tok_a, tok_b],
        )

        assert len(result) == 0

        flushed = scanner.flush_pending()
        assert len(flushed) == 2
        assert isinstance(flushed[0], PreLexedTerminal)
        assert flushed[0].terminal == "TOOL_START"
        assert isinstance(flushed[1], TextChunk)
        assert flushed[1].text == ""
        assert flushed[1].token_count == 2

    def test_deferred_terminal_preserves_trailing_token_count(self, tokenizer):
        tokenizer.decode.side_effect = lambda ids: {
            CHANNEL_START_ID: CHANNEL_START,
            201: "alpha",
            202: "beta",
        }[ids[0]]
        scanner = TokenIDScanner({CHANNEL_START_ID: "THINK_START"}, tokenizer)

        assert scanner.scan("", [CHANNEL_START_ID, 201, 202]) == []

        result = scanner.scan(f"{CHANNEL_START}alphabeta", [])
        assert isinstance(result[0], PreLexedTerminal)
        assert isinstance(result[1], TextChunk)
        assert result[1].text == "alphabeta"
        assert result[1].token_count == 2

    def test_holdback_before_start_tag(self, scanner):
        result = scanner.scan(
            delta_text="prefix text<|channel>",
            delta_token_ids=[CHANNEL_START_ID],
        )

        assert len(result) == 2
        assert isinstance(result[0], TextChunk)
        assert result[0].text == "prefix text"
        assert isinstance(result[1], PreLexedTerminal)
        assert result[1].terminal == "THINK_START"

    def test_multi_token_batch_special_in_middle(self, scanner, tokenizer):
        """Multi-token batch with special token in the middle."""
        tok_a = 201
        tok_b = 202
        tokenizer.decode.side_effect = lambda ids: {
            tok_a: "wordA",
            tok_b: "wordB",
            CHANNEL_END_ID: CHANNEL_END,
        }.get(ids[0], "?")

        scanner_multi = TokenIDScanner(
            token_id_to_terminal={CHANNEL_END_ID: "THINK_END"},
            tokenizer=tokenizer,
        )

        result = scanner_multi.scan(
            delta_text="holdback wordA<channel|> wordB",
            delta_token_ids=[tok_a, CHANNEL_END_ID, tok_b],
        )

        texts = [r.text for r in result if isinstance(r, TextChunk)]
        terminals = [r.terminal for r in result if isinstance(r, PreLexedTerminal)]
        assert "THINK_END" in terminals
        assert "holdback wordA" in "".join(texts)

    def test_multi_token_batch_special_token_text_absent(self, scanner, tokenizer):
        """Multi-token batch where special token text is absent."""
        tok_a = 201
        tok_b = 202
        tokenizer.decode.side_effect = lambda ids: {
            tok_a: "alpha",
            tok_b: "beta",
            CHANNEL_END_ID: CHANNEL_END,
        }.get(ids[0], "?")

        scanner_multi = TokenIDScanner(
            token_id_to_terminal={CHANNEL_END_ID: "THINK_END"},
            tokenizer=tokenizer,
        )

        result = scanner_multi.scan(
            delta_text="holdback alpha",
            delta_token_ids=[tok_a, CHANNEL_END_ID, tok_b],
        )

        assert len(result) == 0

        result2 = scanner_multi.scan(
            delta_text="<channel|> more text",
            delta_token_ids=[300],
        )
        pre_lexed = [r for r in result2 if isinstance(r, PreLexedTerminal)]
        assert len(pre_lexed) == 1
        assert pre_lexed[0].terminal == "THINK_END"
        text_chunks = [r for r in result2 if isinstance(r, TextChunk)]
        combined = "".join(t.text for t in text_chunks)
        assert "holdback alpha" in combined
        assert "more text" in combined

    def test_holdback_with_content_after_special_token(self, tokenizer):
        """Hold-back + special token + content after in one delta."""
        tok_content = 210
        tokenizer.decode.side_effect = lambda ids: {
            CHANNEL_END_ID: CHANNEL_END,
            tok_content: "content start",
        }.get(ids[0], "?")

        scanner = TokenIDScanner(
            token_id_to_terminal={CHANNEL_END_ID: "THINK_END"},
            tokenizer=tokenizer,
        )

        result = scanner.scan(
            delta_text="reasoning end.<channel|>content start",
            delta_token_ids=[CHANNEL_END_ID, tok_content],
        )

        pre_lexed = [r for r in result if isinstance(r, PreLexedTerminal)]
        assert len(pre_lexed) == 1
        assert pre_lexed[0].terminal == "THINK_END"

        text_chunks = [r for r in result if isinstance(r, TextChunk)]
        combined = "".join(t.text for t in text_chunks)
        assert "reasoning end." in combined


class TestEndToEndReasoningHoldback:
    """End-to-end engine tests with detokenizer hold-back."""

    def test_reasoning_content_not_truncated(self):
        config = gemma4_config()
        tok = MagicMock()
        vocab = {
            CHANNEL_START: CHANNEL_START_ID,
            CHANNEL_END: CHANNEL_END_ID,
        }
        tok.get_vocab.return_value = vocab
        tok.decode.side_effect = lambda ids: {
            CHANNEL_START_ID: CHANNEL_START,
            CHANNEL_END_ID: CHANNEL_END,
        }.get(ids[0], f"tok{ids[0]}")

        engine = StreamingParserEngine(config, tok)
        all_events = []

        all_events.extend(engine.feed(CHANNEL_START, [CHANNEL_START_ID]))
        all_events.extend(
            engine.feed(
                "thought\nThe request was received and ",
                [10, 11, 12, 13, 14],
            )
        )
        # CHANNEL_END token arrives but its text is held back.
        all_events.extend(
            engine.feed(
                "processed is appropriate.",
                [CHANNEL_END_ID],
            )
        )
        # Detokenizer flushes the held-back text.
        all_events.extend(
            engine.feed(
                "<channel|>Understood.",
                [20, 21],
            )
        )

        all_events.extend(engine.finish())

        reasoning_text = "".join(
            e.value for e in all_events if e.type == EventType.REASONING_CHUNK
        )
        content_text = "".join(
            e.value for e in all_events if e.type == EventType.TEXT_CHUNK
        )

        assert "processed is appropriate." in reasoning_text
        assert "Understood." in content_text

    def test_backtick_content_not_truncated(self):
        config = gemma4_config()
        tok = MagicMock()
        vocab = {
            CHANNEL_START: CHANNEL_START_ID,
            CHANNEL_END: CHANNEL_END_ID,
        }
        tok.get_vocab.return_value = vocab
        tok.decode.side_effect = lambda ids: {
            CHANNEL_START_ID: CHANNEL_START,
            CHANNEL_END_ID: CHANNEL_END,
        }.get(ids[0], f"tok{ids[0]}")

        engine = StreamingParserEngine(config, tok)
        all_events = []

        all_events.extend(engine.feed(CHANNEL_START, [CHANNEL_START_ID]))
        all_events.extend(
            engine.feed(
                "thought\n1/10 completed. Next: ",
                [10, 11, 12, 13],
            )
        )
        all_events.extend(
            engine.feed(
                "`hostname`.\n",
                [CHANNEL_END_ID],
            )
        )
        all_events.extend(
            engine.feed(
                "<channel|>tool output",
                [20, 21],
            )
        )

        all_events.extend(engine.finish())

        reasoning_text = "".join(
            e.value for e in all_events if e.type == EventType.REASONING_CHUNK
        )

        assert "`hostname`." in reasoning_text


_CHANNEL_START_TAG = "<|channel>"
_CHANNEL_END_TAG = "<channel|>"
_TOOL_START_TAG = "<|tool_call>"
_TOOL_END_TAG = "<tool_call|>"
_QUOTE_TAG = '<|"|>'

_CHANNEL_START_TID = 100
_CHANNEL_END_TID = 101
_TOOL_START_TID = 102
_TOOL_END_TID = 103
_QUOTE_TID = 104
_TOK = list(range(200, 215))


def _gemma4_vocab() -> dict[str, int]:
    return {
        _CHANNEL_START_TAG: _CHANNEL_START_TID,
        _CHANNEL_END_TAG: _CHANNEL_END_TID,
        _TOOL_START_TAG: _TOOL_START_TID,
        _TOOL_END_TAG: _TOOL_END_TID,
        _QUOTE_TAG: _QUOTE_TID,
    }


def _make_gemma4_tokenizer(
    extra_decode: dict[int, str] | None = None,
) -> MagicMock:
    special = {
        _CHANNEL_START_TID: _CHANNEL_START_TAG,
        _CHANNEL_END_TID: _CHANNEL_END_TAG,
        _TOOL_START_TID: _TOOL_START_TAG,
        _TOOL_END_TID: _TOOL_END_TAG,
        _QUOTE_TID: _QUOTE_TAG,
    }
    decode_map = {**special, **(extra_decode or {})}

    tok = MagicMock()
    tok.get_vocab.return_value = _gemma4_vocab()
    tok.decode.side_effect = lambda ids: decode_map.get(ids[0], f"tok{ids[0]}")
    return tok


def _collect_events(engine, deltas):
    from vllm.parser.engine.events import SemanticEvent

    all_events: list[SemanticEvent] = []
    for delta_text, delta_token_ids in deltas:
        all_events.extend(engine.feed(delta_text, delta_token_ids))
    all_events.extend(engine.finish())
    return all_events


def _reasoning_text(events) -> str:
    return "".join(e.value for e in events if e.type == EventType.REASONING_CHUNK)


def _content_text(events) -> str:
    return "".join(e.value for e in events if e.type == EventType.TEXT_CHUNK)


def _arg_text(events) -> str:
    return "".join(e.value for e in events if e.type == EventType.ARG_VALUE_CHUNK)


def _has_event(events, event_type) -> bool:
    return any(e.type == event_type for e in events)


class TestMultiTokenBoundaryPreservation:
    """No text lost at state boundaries with multi-token deltas."""

    def test_empty_delta_text_at_channel_end_unified(self):
        """Empty delta_text when CHANNEL_END arrives; text comes later."""
        tok = _make_gemma4_tokenizer()
        engine = StreamingParserEngine(gemma4_config(), tok)

        events = _collect_events(
            engine,
            [
                ("", [_CHANNEL_START_TID]),
                ("<|channel>thought\nSome reasoning.", [_TOK[0], _TOK[1]]),
                ("", [_CHANNEL_END_TID]),
                ("<channel|>Final answer.", [_TOK[2], _TOK[3]]),
            ],
        )

        reasoning = _reasoning_text(events)
        content = _content_text(events)
        assert "Some reasoning." in reasoning
        assert "Final answer." in content
        assert _has_event(events, EventType.REASONING_START)
        assert _has_event(events, EventType.REASONING_END)

    def test_deferred_channel_end_flushed_at_finish_unified(self):
        """Deferred CHANNEL_END flushed at end-of-stream."""
        tok = _make_gemma4_tokenizer()
        engine = StreamingParserEngine(gemma4_config(), tok)

        events = _collect_events(
            engine,
            [
                (_CHANNEL_START_TAG, [_CHANNEL_START_TID]),
                ("thought\nReasoning text.", [_TOK[0]]),
                (" Final thought.", [_CHANNEL_END_TID]),
            ],
        )

        reasoning = _reasoning_text(events)
        assert "Reasoning text. Final thought." in reasoning
        assert _has_event(events, EventType.REASONING_END)

    def test_reasoning_to_tool_call_handoff_unified(self):
        """Full reasoning -> content -> tool call flow."""
        tok = _make_gemma4_tokenizer()
        engine = StreamingParserEngine(gemma4_config(), tok)

        events = _collect_events(
            engine,
            [
                (_CHANNEL_START_TAG, [_CHANNEL_START_TID]),
                ("thought\nI need to check the weather.", [_TOK[0], _TOK[1], _TOK[2]]),
                (_CHANNEL_END_TAG, [_CHANNEL_END_TID]),
                ("Let me call a tool.", [_TOK[3], _TOK[4]]),
                (_TOOL_START_TAG, [_TOOL_START_TID]),
                ("call:get_weather{city:", [_TOK[5], _TOK[6]]),
                ('<|"|>SF<|"|>}', [_QUOTE_TID, _TOK[7], _QUOTE_TID, _TOK[8]]),
                (_TOOL_END_TAG, [_TOOL_END_TID]),
            ],
        )

        reasoning = _reasoning_text(events)
        content = _content_text(events)

        assert "I need to check the weather." in reasoning
        assert "Let me call a tool." in content
        assert _has_event(events, EventType.REASONING_START)
        assert _has_event(events, EventType.REASONING_END)
        assert _has_event(events, EventType.TOOL_CALL_START)
        assert _has_event(events, EventType.TOOL_CALL_END)
        assert "SF" in _arg_text(events)

    def test_multiple_tool_calls_rapid_transitions_unified(self):
        """Two back-to-back tool calls with correct tool_index tracking."""
        tok = _make_gemma4_tokenizer()
        engine = StreamingParserEngine(gemma4_config(), tok)

        events = _collect_events(
            engine,
            [
                (_TOOL_START_TAG, [_TOOL_START_TID]),
                ("call:get_weather{city:", [_TOK[0], _TOK[1]]),
                ('<|"|>NYC<|"|>}', [_QUOTE_TID, _TOK[2], _QUOTE_TID, _TOK[3]]),
                (_TOOL_END_TAG, [_TOOL_END_TID]),
                (_TOOL_START_TAG, [_TOOL_START_TID]),
                ("call:get_time{tz:", [_TOK[4], _TOK[5]]),
                ('<|"|>EST<|"|>}', [_QUOTE_TID, _TOK[6], _QUOTE_TID, _TOK[7]]),
                (_TOOL_END_TAG, [_TOOL_END_TID]),
            ],
        )

        starts = [e for e in events if e.type == EventType.TOOL_CALL_START]
        ends = [e for e in events if e.type == EventType.TOOL_CALL_END]
        assert len(starts) == 2
        assert len(ends) == 2
        assert starts[0].tool_index == 0
        assert starts[1].tool_index == 1

        names = "".join(e.value for e in events if e.type == EventType.TOOL_NAME)
        assert "get_weather" in names
        assert "get_time" in names

    def test_deferred_channel_end_before_tool_call_unified(self):
        """Deferred CHANNEL_END followed by a tool call."""
        tok = _make_gemma4_tokenizer()
        engine = StreamingParserEngine(gemma4_config(), tok)

        events = _collect_events(
            engine,
            [
                (_CHANNEL_START_TAG, [_CHANNEL_START_TID]),
                ("thought\nNeed to call a tool.", [_TOK[0], _TOK[1]]),
                (" Let me proceed.", [_CHANNEL_END_TID]),
                (_CHANNEL_END_TAG, [_TOK[2]]),
                (_TOOL_START_TAG, [_TOOL_START_TID]),
                ("call:get_weather{city:", [_TOK[3], _TOK[4]]),
                ('<|"|>Tokyo<|"|>}', [_QUOTE_TID, _TOK[5], _QUOTE_TID, _TOK[6]]),
                (_TOOL_END_TAG, [_TOOL_END_TID]),
            ],
        )

        reasoning = _reasoning_text(events)
        assert "Need to call a tool. Let me proceed." in reasoning
        assert _has_event(events, EventType.REASONING_END)
        assert _has_event(events, EventType.TOOL_CALL_START)
        assert _has_event(events, EventType.TOOL_CALL_END)
        assert "Tokyo" in _arg_text(events)


class TestStreamInterval10:
    """Tests with stream_interval=10 (large multi-token batches)."""

    def test_channel_end_mid_batch_text_present(self):
        """<channel|> mid-batch with its text present in delta_text."""
        tok = _make_gemma4_tokenizer({_TOK[i]: f"word{i} " for i in range(15)})
        engine = StreamingParserEngine(gemma4_config(), tok)

        events: list = []
        events.extend(
            engine.feed(
                "<|channel>thought\nword0 word1 word2 word3 word4 "
                "word5 word6 word7 word8 ",
                [
                    _CHANNEL_START_TID,
                    _TOK[0],
                    _TOK[1],
                    _TOK[2],
                    _TOK[3],
                    _TOK[4],
                    _TOK[5],
                    _TOK[6],
                    _TOK[7],
                    _TOK[8],
                ],
            )
        )

        events.extend(
            engine.feed(
                "word9 word10 word11 <channel|>word12 word13 word14 word0 word1 word2 ",
                [
                    _TOK[9],
                    _TOK[10],
                    _TOK[11],
                    _CHANNEL_END_TID,
                    _TOK[12],
                    _TOK[13],
                    _TOK[14],
                    _TOK[0],
                    _TOK[1],
                    _TOK[2],
                ],
            )
        )

        events.extend(engine.finish())

        reasoning = _reasoning_text(events)
        content = _content_text(events)

        for w in ("word9", "word10", "word11"):
            assert w in reasoning, f"{w!r} missing from reasoning"

        for w in ("word12", "word13", "word14"):
            assert w in content, f"{w!r} missing from content"

        assert _has_event(events, EventType.REASONING_END)

    def test_channel_end_and_tool_start_same_batch_unified(self):
        """Both <channel|> and <|tool_call> in a single batch."""
        tok = _make_gemma4_tokenizer({_TOK[i]: f"w{i} " for i in range(15)})
        engine = StreamingParserEngine(gemma4_config(), tok)

        events: list = []

        events.extend(
            engine.feed(
                "<|channel>thought\nw0 w1 w2 w3 w4 w5 w6 w7 w8 ",
                [
                    _CHANNEL_START_TID,
                    _TOK[0],
                    _TOK[1],
                    _TOK[2],
                    _TOK[3],
                    _TOK[4],
                    _TOK[5],
                    _TOK[6],
                    _TOK[7],
                    _TOK[8],
                ],
            )
        )

        events.extend(
            engine.feed(
                "w9 w10 <channel|>w11 <|tool_call>",
                [
                    _TOK[9],
                    _TOK[10],
                    _CHANNEL_END_TID,
                    _TOK[11],
                    _TOOL_START_TID,
                    _TOK[12],
                    _TOK[13],
                    _TOK[14],
                    _TOK[0],
                    _TOK[1],
                ],
            )
        )
        events.extend(engine.finish())

        reasoning = _reasoning_text(events)

        assert "w9" in reasoning
        assert "w10" in reasoning
        assert _has_event(events, EventType.REASONING_END)
        assert _has_event(events, EventType.TOOL_CALL_START)

    def test_channel_end_mid_batch_text_absent(self):
        """<channel|> mid-batch with its text absent from delta_text."""
        tok = _make_gemma4_tokenizer({_TOK[i]: f"word{i} " for i in range(15)})
        engine = StreamingParserEngine(gemma4_config(), tok)

        events: list = []
        events.extend(
            engine.feed(
                "<|channel>thought\nword0 word1 word2 word3 word4 "
                "word5 word6 word7 word8 ",
                [
                    _CHANNEL_START_TID,
                    _TOK[0],
                    _TOK[1],
                    _TOK[2],
                    _TOK[3],
                    _TOK[4],
                    _TOK[5],
                    _TOK[6],
                    _TOK[7],
                    _TOK[8],
                ],
            )
        )

        events.extend(
            engine.feed(
                "word9 word10 word11 ",
                [
                    _TOK[9],
                    _TOK[10],
                    _TOK[11],
                    _CHANNEL_END_TID,
                    _TOK[12],
                    _TOK[13],
                    _TOK[14],
                    _TOK[0],
                    _TOK[1],
                    _TOK[2],
                ],
            )
        )

        events.extend(
            engine.feed(
                "<channel|>word12 word13 word14 word0 word1 word2 ",
                [_TOK[3], _TOK[4], _TOK[5]],
            )
        )

        events.extend(engine.finish())

        reasoning = _reasoning_text(events)
        content = _content_text(events)

        for w in ("word9", "word10", "word11"):
            assert w in reasoning, f"{w!r} missing from reasoning"

        for w in ("word12", "word13", "word14"):
            assert w in content, f"{w!r} missing from content"

        assert _has_event(events, EventType.REASONING_END)

    def test_tool_end_mid_batch_text_absent_unified(self):
        """<tool_call|> mid-batch with text absent."""
        tok = _make_gemma4_tokenizer({_TOK[i]: f"w{i}" for i in range(15)})
        engine = StreamingParserEngine(gemma4_config(), tok)

        events: list = []
        events.extend(
            engine.feed(
                _CHANNEL_START_TAG,
                [_CHANNEL_START_TID],
            )
        )
        events.extend(
            engine.feed(
                "thought\nNeed a tool.",
                [_TOK[0], _TOK[1]],
            )
        )
        events.extend(
            engine.feed(
                _TOOL_START_TAG,
                [_TOOL_START_TID],
            )
        )
        events.extend(
            engine.feed(
                "call:get_weather{city:",
                [_TOK[2], _TOK[3], _TOK[4]],
            )
        )

        events.extend(
            engine.feed(
                '<|"|>San Francisco<|"|>}',
                [
                    _QUOTE_TID,
                    _TOK[5],
                    _TOK[6],
                    _QUOTE_TID,
                    _TOK[7],
                    _TOOL_END_TID,
                    _TOK[8],
                    _TOK[9],
                    _TOK[10],
                    _TOK[11],
                ],
            )
        )

        events.extend(
            engine.feed(
                "<tool_call|>w8w9w10w11w12",
                [_TOK[12], _TOK[13]],
            )
        )

        events.extend(engine.finish())

        assert _has_event(events, EventType.TOOL_CALL_END)
        assert "San Francisco" in _arg_text(events)

    def test_large_batch_holdback_spans_two_batches(self):
        """Holdback text spanning two batches with <channel|> in the second."""
        tok = _make_gemma4_tokenizer({_TOK[i]: f"w{i} " for i in range(15)})
        engine = StreamingParserEngine(gemma4_config(), tok)

        events: list = []

        events.extend(
            engine.feed(
                "<|channel>thought\nThe user asked about machine learning "
                "and I need to think about the best approach to",
                [
                    _CHANNEL_START_TID,
                    _TOK[0],
                    _TOK[1],
                    _TOK[2],
                    _TOK[3],
                    _TOK[4],
                    _TOK[5],
                    _TOK[6],
                    _TOK[7],
                    _TOK[8],
                ],
            )
        )

        events.extend(
            engine.feed(
                " explain this complex topic. Let me organize my thoughts.",
                [
                    _TOK[9],
                    _TOK[10],
                    _TOK[11],
                    _TOK[12],
                    _TOK[13],
                    _TOK[14],
                    _CHANNEL_END_TID,
                    _TOK[0],
                    _TOK[1],
                    _TOK[2],
                ],
            )
        )

        events.extend(
            engine.feed(
                "<channel|>w0 w1 w2 Here is what I recommend: start with "
                "the fundamentals and build up from there.",
                [
                    _TOK[3],
                    _TOK[4],
                    _TOK[5],
                    _TOK[6],
                    _TOK[7],
                    _TOK[8],
                    _TOK[9],
                    _TOK[10],
                    _TOK[11],
                    _TOK[12],
                ],
            )
        )

        events.extend(engine.finish())

        reasoning = _reasoning_text(events)
        content = _content_text(events)

        assert "organize my thoughts." in reasoning
        assert "explain" in reasoning
        assert "recommend" in content

        assert _has_event(events, EventType.REASONING_START)
        assert _has_event(events, EventType.REASONING_END)


class TestRebuildFromAnchorsLiteralLookalike:
    """Literal token text in prose must not be consumed as an anchor."""

    @pytest.fixture
    def tool_scanner(self):
        tok = MagicMock()
        tok.get_vocab.return_value = {
            TOOL_START: TOOL_START_ID,
            TOOL_END: TOOL_END_ID,
        }
        tok.decode.side_effect = lambda ids: {
            TOOL_START_ID: TOOL_START,
            TOOL_END_ID: TOOL_END,
        }.get(ids[0], f"t{ids[0]}")
        return TokenIDScanner(
            {TOOL_START_ID: "TOOL_START", TOOL_END_ID: "TOOL_END"},
            tok,
        )

    def test_literal_before_real_anchor(self, tool_scanner):
        delta_text = 'Use <tool_call> like this: <tool_call>{"name":"f"}</tool_call>'
        delta_token_ids = [1, 2, 3, 4, 5, TOOL_START_ID, 6, 7, TOOL_END_ID]
        items = tool_scanner.scan(delta_text, delta_token_ids)

        text_parts = [it.text for it in items if isinstance(it, TextChunk)]
        terminals = [it for it in items if isinstance(it, PreLexedTerminal)]

        assert len(terminals) == 2
        assert terminals[0].terminal == "TOOL_START"
        assert terminals[1].terminal == "TOOL_END"

        joined_text = "".join(text_parts)
        assert "<tool_call>" in joined_text
        assert '{"name":"f"}' in joined_text

    def test_multiple_tool_calls_with_literal_between(self, tool_scanner):
        delta_text = (
            '<tool_call>{"name":"a"}</tool_call>'
            " see <tool_call> syntax "
            '<tool_call>{"name":"b"}</tool_call>'
        )
        delta_token_ids = [
            TOOL_START_ID,
            1,
            TOOL_END_ID,
            2,
            3,
            4,
            TOOL_START_ID,
            5,
            TOOL_END_ID,
        ]
        items = tool_scanner.scan(delta_text, delta_token_ids)

        terminals = [it for it in items if isinstance(it, PreLexedTerminal)]
        assert len(terminals) == 4

        text_parts = [it.text for it in items if isinstance(it, TextChunk)]
        joined_text = "".join(text_parts)
        assert "<tool_call> syntax" in joined_text


class TestRebuildFromAnchorsCascadingDeferral:
    """Missing middle anchor defers only itself, not subsequent ones."""

    @pytest.fixture
    def bare_scanner(self):
        tok = MagicMock()
        tok.decode.side_effect = lambda ids: f"t{ids[0]}"
        return TokenIDScanner({}, tok)

    def test_middle_anchor_missing_does_not_cascade(self, bare_scanner):
        a = PreLexedTerminal("TOOL_START", TOOL_START_ID, TOOL_START)
        b = PreLexedTerminal("THINK_END", CHANNEL_END_ID, CHANNEL_END)
        c = PreLexedTerminal("TOOL_END", TOOL_END_ID, TOOL_END)
        delta_text = f"prefix{TOOL_START}middle{TOOL_END}suffix"
        results = [a, b, c]

        rebuilt = bare_scanner._rebuild_from_anchors(delta_text, results)

        terminals = [r for r in rebuilt if isinstance(r, PreLexedTerminal)]
        texts = [r for r in rebuilt if isinstance(r, TextChunk)]
        joined = "".join(t.text for t in texts)

        assert len(terminals) == 2
        assert terminals[0].terminal == "TOOL_START"
        assert terminals[1].terminal == "TOOL_END"
        assert "prefix" in joined
        assert "middle" in joined
        assert "suffix" in joined
        assert len(bare_scanner._deferred_terminals) == 1
        assert bare_scanner._deferred_terminals[0].terminal == "THINK_END"
        assert bare_scanner._deferred_post_text == ""

    def test_first_anchor_missing_rest_still_emitted(self, bare_scanner):
        a = PreLexedTerminal("THINK_END", CHANNEL_END_ID, CHANNEL_END)
        b = PreLexedTerminal("TOOL_START", TOOL_START_ID, TOOL_START)
        delta_text = f"text{TOOL_START}more"
        results = [a, b]

        rebuilt = bare_scanner._rebuild_from_anchors(delta_text, results)

        terminals = [r for r in rebuilt if isinstance(r, PreLexedTerminal)]
        assert len(terminals) == 1
        assert terminals[0].terminal == "TOOL_START"
        assert len(bare_scanner._deferred_terminals) == 1
        assert bare_scanner._deferred_terminals[0].terminal == "THINK_END"

    def test_last_anchor_missing_preceding_still_emitted(self, bare_scanner):
        a = PreLexedTerminal("TOOL_START", TOOL_START_ID, TOOL_START)
        b = PreLexedTerminal("THINK_END", CHANNEL_END_ID, CHANNEL_END)
        delta_text = f"text{TOOL_START}more"
        results = [a, b]

        rebuilt = bare_scanner._rebuild_from_anchors(delta_text, results)

        terminals = [r for r in rebuilt if isinstance(r, PreLexedTerminal)]
        assert len(terminals) == 1
        assert terminals[0].terminal == "TOOL_START"
        texts = [r for r in rebuilt if isinstance(r, TextChunk)]
        joined = "".join(t.text for t in texts)
        assert "text" in joined
        assert bare_scanner._deferred_post_text == "more"
        assert len(bare_scanner._deferred_terminals) == 1
        assert bare_scanner._deferred_terminals[0].terminal == "THINK_END"


class TestTrailingDropAnchorRecovery:
    DROP_TEXT = "<|observation|>"
    DROP_TEXT_2 = "<|user|>"
    DROP_ID = 112
    DROP_ID_2 = 113
    TEXT_ID = 201
    BODY_ID = 202
    AFTER_ID = 203
    BODY = (
        "record_value<arg_key>value</arg_key><arg_value>"
        "</tool_call> and <tool_call></arg_value>"
    )

    def _scanner(self) -> TokenIDScanner:
        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids: {
            self.TEXT_ID: "prefix",
            self.BODY_ID: self.BODY,
            self.AFTER_ID: "after",
            TOOL_START_ID: TOOL_START,
            TOOL_END_ID: TOOL_END,
            self.DROP_ID: self.DROP_TEXT,
            self.DROP_ID_2: self.DROP_TEXT_2,
        }[ids[0]]
        return TokenIDScanner(
            {
                TOOL_START_ID: "TOOL_START",
                TOOL_END_ID: "TOOL_END",
                self.DROP_ID: DROP_TERMINAL,
                self.DROP_ID_2: DROP_TERMINAL,
            },
            tokenizer,
        )

    @staticmethod
    def _signature(items, *, without_drop: bool = False):
        result: list[tuple[object, ...]] = []
        for item in items:
            if isinstance(item, TextChunk):
                result.append(("text", item.text, item.token_count))
            elif not without_drop or item.terminal != DROP_TERMINAL:
                result.append(("terminal", item.terminal))
        return result

    def _texts_and_ids(self, *drop_ids: int):
        ids = [
            self.TEXT_ID,
            TOOL_START_ID,
            self.BODY_ID,
            TOOL_END_ID,
            *drop_ids,
        ]
        matching = f"prefix{TOOL_START}{self.BODY}{TOOL_END}"
        for drop_id in drop_ids:
            matching += {
                self.DROP_ID: self.DROP_TEXT,
                self.DROP_ID_2: self.DROP_TEXT_2,
            }[drop_id]
        stripped = f"prefix{TOOL_START}{self.BODY}{TOOL_END}"
        return matching, stripped, ids

    def test_stop_stripped_trailing_drop_preserves_structural_anchors(self):
        matching, stripped, ids = self._texts_and_ids(self.DROP_ID)
        matching_scanner = self._scanner()
        stripped_scanner = self._scanner()

        matching_items = matching_scanner.scan(matching, ids)
        stripped_items = stripped_scanner.scan(stripped, ids)

        expected = [
            ("text", "prefix", 1),
            ("terminal", "TOOL_START"),
            ("text", self.BODY, 1),
            ("terminal", "TOOL_END"),
        ]
        assert self._signature(matching_items, without_drop=True) == expected
        assert self._signature(stripped_items) == expected
        assert [item.terminal for item in stripped_scanner._deferred_terminals] == [
            DROP_TERMINAL
        ]
        assert stripped_scanner._deferred_prefix_token_counts == [0]
        assert stripped_scanner._deferred_post_text == ""

    def test_stop_stripped_drop_preserves_text_token_count(self):
        matching_scanner = self._scanner()
        stripped_scanner = self._scanner()

        matching = matching_scanner.scan(
            "prefix" + self.DROP_TEXT,
            [self.TEXT_ID, self.DROP_ID],
        )
        stripped = stripped_scanner.scan(
            "prefix",
            [self.TEXT_ID, self.DROP_ID],
        )

        matching_count = sum(
            item.token_count for item in matching if isinstance(item, TextChunk)
        )
        stripped_count = sum(
            item.token_count for item in stripped if isinstance(item, TextChunk)
        )
        assert matching_count == stripped_count == 1

    def test_deferred_trailing_drop_resolves_on_next_delta(self):
        _, stripped, ids = self._texts_and_ids(self.DROP_ID)
        scanner = self._scanner()
        scanner.scan(stripped, ids)

        items = scanner.scan(self.DROP_TEXT + "after", [self.AFTER_ID])

        assert self._signature(items) == [
            ("terminal", DROP_TERMINAL),
            ("text", "after", 1),
        ]
        assert scanner._deferred_terminals == []

    def test_absent_trailing_drop_is_flushed_in_order(self):
        _, stripped, ids = self._texts_and_ids(self.DROP_ID, self.DROP_ID_2)
        scanner = self._scanner()

        stripped_items = scanner.scan(stripped, ids)
        pending = scanner.flush_pending()

        assert self._signature(stripped_items) == [
            ("text", "prefix", 1),
            ("terminal", "TOOL_START"),
            ("text", self.BODY, 1),
            ("terminal", "TOOL_END"),
        ]
        assert [item.token_id for item in pending] == [self.DROP_ID, self.DROP_ID_2]
        assert all(item.terminal == DROP_TERMINAL for item in pending)


def _cleanup_tokenizer(decoded: dict[int, str]) -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.decode.side_effect = lambda ids: decoded[ids[0]]
    return tokenizer


def test_reconstructed_current_text_binds_to_suffix():
    body_id = 10
    end_id = 11
    scanner = TokenIDScanner(
        {end_id: "END"},
        _cleanup_tokenizer({body_id: "tail", end_id: "<end>"}),
    )

    items = scanner.scan("tail<end> held tail<end>", [body_id, end_id])

    assert items == [
        TextChunk("tail<end> held "),
        TextChunk("tail", ("tail",), 1),
        PreLexedTerminal("END", end_id, "<end>"),
    ]


def test_stripped_trailing_drop_preserves_holdback_and_token_count():
    body_id = 10
    drop_id = 11
    scanner = TokenIDScanner(
        {drop_id: DROP_TERMINAL},
        _cleanup_tokenizer({body_id: "prefix", drop_id: "<drop>"}),
    )

    items = scanner.scan("holdback prefix", [body_id, drop_id])

    assert items == [
        TextChunk("holdback "),
        TextChunk("prefix", ("prefix",), 1),
    ]
    assert scanner.flush_pending() == [
        PreLexedTerminal(DROP_TERMINAL, drop_id, "<drop>")
    ]


def test_resolved_terminal_preserves_deferred_trailing_count():
    body_id = 10
    end_id = 11
    scanner = TokenIDScanner(
        {end_id: "END"},
        _cleanup_tokenizer({body_id: "body", end_id: "<end>"}),
    )

    assert scanner.scan("", [end_id, body_id]) == []
    assert scanner.scan("<end>", []) == [
        PreLexedTerminal("END", end_id, "<end>"),
        TextChunk("", token_count=1),
    ]


@pytest.mark.parametrize("finish", [False, True], ids=["resolve", "flush"])
def test_inter_terminal_count_carrier_stays_ordered(finish: bool):
    start_id = 10
    body_id = 11
    end_id = 12
    scanner = TokenIDScanner(
        {start_id: "START", end_id: "END"},
        _cleanup_tokenizer(
            {
                start_id: "<start>",
                body_id: "body",
                end_id: "<end>",
            }
        ),
    )

    assert scanner.scan("", [start_id, body_id, end_id]) == []
    items = scanner.flush_pending() if finish else scanner.scan("<start><end>", [])

    assert items == [
        PreLexedTerminal("START", start_id, "<start>"),
        TextChunk("", token_count=1),
        PreLexedTerminal("END", end_id, "<end>"),
    ]


@pytest.mark.parametrize(
    "protection",
    ["terminals", "token_id_terminals", "preserve_tokens"],
)
def test_auto_drop_alias_does_not_override_protected_id(protection: str):
    protected = "<protected>"
    alias = "<alias>"
    dropped = "<dropped>"
    protected_id = 10
    dropped_id = 11
    tokenizer = _cleanup_tokenizer({protected_id: protected, dropped_id: dropped})
    tokenizer.get_vocab.return_value = {
        protected: protected_id,
        alias: protected_id,
        dropped: dropped_id,
    }
    tokenizer.all_special_tokens = [protected, alias, dropped]
    tokenizer.all_special_ids = [protected_id, protected_id, dropped_id]

    terminals = {"PROTECTED": protected} if protection == "terminals" else {}
    token_id_terminals = (
        {"PROTECTED": protected} if protection == "token_id_terminals" else {}
    )
    preserve_tokens = (
        frozenset({protected}) if protection == "preserve_tokens" else frozenset()
    )
    transitions = (
        {
            (ParserState.CONTENT, "PROTECTED"): Transition(
                ParserState.CONTENT,
                (EventType.TEXT_CHUNK,),
            )
        }
        if protection != "preserve_tokens"
        else {}
    )
    config = ParserEngineConfig(
        name=f"protected_{protection}",
        terminals=terminals,
        token_id_terminals=token_id_terminals,
        transitions=transitions,
        content_events={ParserState.CONTENT: EventType.TEXT_CHUNK},
        initial_state=ParserState.CONTENT,
        preserve_tokens=preserve_tokens,
    )
    engine = StreamingParserEngine(config, tokenizer)

    events = engine.feed(protected, [protected_id])
    events.extend(engine.finish())

    assert (
        "".join(event.value for event in events if event.type == EventType.TEXT_CHUNK)
        == protected
    )

    alias_events = StreamingParserEngine(config, tokenizer).parse_complete(alias)
    assert (
        "".join(
            event.value for event in alias_events if event.type == EventType.TEXT_CHUNK
        )
        == ""
    )


@pytest.mark.parametrize(
    "protection",
    ["terminals", "token_id_terminals", "preserve_tokens"],
)
def test_alias_spelling_survives_when_its_only_id_is_protected(
    protection: str,
):
    protected = "<protected>"
    alias = "<alias>"
    protected_id = 10
    tokenizer = _cleanup_tokenizer({protected_id: alias})
    tokenizer.get_vocab.return_value = {
        protected: protected_id,
        alias: protected_id,
    }
    tokenizer.all_special_tokens = [protected, alias]
    tokenizer.all_special_ids = [protected_id, protected_id]

    terminals = {"PROTECTED": protected} if protection == "terminals" else {}
    token_id_terminals = (
        {"PROTECTED": protected} if protection == "token_id_terminals" else {}
    )
    preserve_tokens = (
        frozenset({protected}) if protection == "preserve_tokens" else frozenset()
    )
    transitions = (
        {
            (ParserState.CONTENT, "PROTECTED"): Transition(
                ParserState.CONTENT,
                (EventType.TEXT_CHUNK,),
            )
        }
        if protection != "preserve_tokens"
        else {}
    )
    config = ParserEngineConfig(
        name=f"alias_only_{protection}",
        terminals=terminals,
        token_id_terminals=token_id_terminals,
        transitions=transitions,
        content_events={ParserState.CONTENT: EventType.TEXT_CHUNK},
        initial_state=ParserState.CONTENT,
        preserve_tokens=preserve_tokens,
    )

    events = StreamingParserEngine(config, tokenizer).feed(alias, [protected_id])
    assert (
        "".join(event.value for event in events if event.type == EventType.TEXT_CHUNK)
        == alias
    )

    text_only = StreamingParserEngine(config, tokenizer).parse_complete(alias)
    assert not any(
        event.value == alias
        for event in text_only
        if event.type == EventType.TEXT_CHUNK
    )


def test_deferred_terminal_binds_after_literal_lookalike():
    body_id = 10
    end_id = 11
    scanner = TokenIDScanner(
        {end_id: "END"},
        _cleanup_tokenizer({body_id: "after", end_id: "<end>"}),
    )

    assert scanner.scan("", [end_id]) == []
    items = scanner.scan("literal <end> real <end>after", [body_id])

    assert items == [
        TextChunk("literal <end> real "),
        PreLexedTerminal("END", end_id, "<end>"),
        TextChunk("after", ("after",), 1),
    ]


def test_deferred_terminal_sequence_uses_current_suffix_boundary():
    start_id = 10
    body_id = 11
    end_id = 12
    after_id = 13
    scanner = TokenIDScanner(
        {start_id: "START", end_id: "END"},
        _cleanup_tokenizer(
            {
                start_id: "<start>",
                body_id: "between",
                end_id: "<end>",
                after_id: "after",
            }
        ),
    )

    assert scanner.scan("", [start_id, body_id, end_id]) == []
    items = scanner.scan(
        "literal <start> real <start><end>after",
        [after_id],
    )

    assert items == [
        TextChunk("literal <start> real "),
        PreLexedTerminal("START", start_id, "<start>"),
        TextChunk("", token_count=1),
        PreLexedTerminal("END", end_id, "<end>"),
        TextChunk("after", ("after",), 1),
    ]
