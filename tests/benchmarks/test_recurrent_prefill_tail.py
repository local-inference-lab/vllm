# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check usage validation without a model server or GPU dependencies."""

import contextlib
import io
import json
import unittest
from unittest.mock import patch

from benchmarks import benchmark_recurrent_prefill_tail as benchmark


class TestRecurrentPrefillTail(unittest.TestCase):
    def run_probe(self, usage):
        """Run the CLI against an in-memory completion stream.

        Args:
            usage: The usage object reported by the mocked completion chunk.

        Returns:
            The JSON rows printed by the benchmark, as parsed dicts.
        """
        chunks = [{"choices": [{"text": "ok"}], "usage": usage}]
        stream = (
            b"".join(
                b"data: " + json.dumps(chunk).encode() + b"\n\n" for chunk in chunks
            )
            + b"data: [DONE]\n\n"
        )
        output = io.StringIO()
        with (
            patch("sys.argv", ["probe", "--model", "test", "--prompt-tokens", "2"]),
            patch.object(
                benchmark.urllib.request,
                "urlopen",
                side_effect=lambda *a, **kw: io.BytesIO(stream),
            ),
            contextlib.redirect_stdout(output),
        ):
            benchmark.main()
        return [json.loads(line) for line in output.getvalue().splitlines()]

    def test_missing_usage_has_distinct_error(self):
        """Absent usage must not be blamed on prompt-token details."""
        missing_usage: tuple[None, dict[str, int]] = (None, {})
        for usage in missing_usage:
            with (
                self.subTest(usage=usage),
                self.assertRaisesRegex(
                    RuntimeError, "Stream ended without a usage chunk"
                ),
            ):
                self.run_probe(usage)

    def test_missing_prompt_tokens_has_clear_error(self):
        """Incomplete usage must not escape as a bare KeyError."""
        with self.assertRaisesRegex(
            RuntimeError, "Usage chunk is missing prompt_tokens"
        ):
            self.run_probe({"prompt_tokens_details": {"cached_tokens": 0}})

    def test_missing_cache_details_keeps_flag_hint(self):
        """Usage without cache accounting still needs the server flag."""
        with self.assertRaisesRegex(RuntimeError, "--enable-prompt-tokens-details"):
            self.run_probe({"prompt_tokens": 2})

    def test_zero_cached_tokens_is_valid(self):
        """A cold cache is a valid measurement, not missing usage."""
        rows = self.run_probe(
            {"prompt_tokens": 2, "prompt_tokens_details": {"cached_tokens": 0}}
        )
        self.assertEqual(
            [row["case"] for row in rows], ["cold", "repeat", "suffix_edit"]
        )
        for row in rows:
            self.assertEqual(row["cached_tokens"], 0)
            self.assertEqual(row["recomputed_tokens"], 2)
            self.assertIsNotNone(row["ttft_s"])
