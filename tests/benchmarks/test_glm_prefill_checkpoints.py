# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline response/gate tests; never contact an inference endpoint."""

import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

PATH = Path(__file__).resolve().parents[2] / "benchmarks/glm_prefill_checkpoints.py"
SPEC = importlib.util.spec_from_file_location("glm_prefill_checks", PATH)
assert SPEC is not None and SPEC.loader is not None
CHECKS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKS)


@pytest.mark.parametrize(
    "details",
    [None, {}, {"cached_tokens": False}, {"cached_tokens": -1}, {"cached_tokens": "0"}],
)
def test_unknown_or_malformed_cache_usage_is_rejected(details):
    with pytest.raises(ValueError, match="cached_tokens"):
        CHECKS.cached_tokens({"prompt_tokens_details": details})


def test_exact_usage_and_expected_cache_hit_are_both_required():
    details = {"cached_tokens": 0}
    usage = {"prompt_tokens": 8192, "prompt_tokens_details": details}
    assert CHECKS.validate_usage(usage, 8192, "cold") == 0
    with pytest.raises(ValueError, match="Prompt usage"):
        CHECKS.validate_usage(usage, 16384, "cold")
    with pytest.raises(ValueError, match="reuse was absent"):
        CHECKS.validate_usage(usage, 8192, "reuse")
    details["cached_tokens"] = 4096
    assert CHECKS.validate_usage(usage, 8192, "reuse") == 4096
    with pytest.raises(ValueError, match="Cold request"):
        CHECKS.validate_usage(usage, 8192, "cold")


def test_metrics_require_present_finite_gauges():
    assert CHECKS.request_gauges(
        'vllm:num_requests_running{model="m"} 0\nvllm:num_requests_waiting 0\n'
    ) == {"num_requests_running": 0, "num_requests_waiting": 0}
    for text in (
        "",
        "vllm:num_requests_running 0\n",
        "vllm:num_requests_running NaN\nvllm:num_requests_waiting 0",
    ):
        with pytest.raises(ValueError):
            CHECKS.request_gauges(text)


def test_summary_excludes_shape_warmup_and_failed_samples():
    cases = [
        {"phase": phase, "accepted": accepted, "tokens": 8192, "ttft_seconds": elapsed}
        for phase, accepted, elapsed in [
            ("warmup", True, 100),
            ("measured", True, 2),
            ("measured", True, 4),
            ("measured", False, 1),
        ]
    ]
    summary = CHECKS.summarize(cases)
    assert summary[0]["samples"] == 2
    assert summary[0]["median_ttft_seconds"] == 3
    assert summary[0]["tokens_per_second"] == 8192 / 3


@pytest.mark.parametrize("done", [True, False])
def test_stream_records_full_chunks_and_requires_final_completion(monkeypatch, done):
    journal = NS(record={"requests": []}, save=lambda: None)
    args = NS(
        base_url="http://unit.invalid/v1",
        api_key_env=None,
        model="m",
        request_timeout=1,
    )
    client = CHECKS.PrefillChecks(args, journal)
    chunks = [
        {"choices": [{"delta": {"role": "assistant", "content": ""}}]},
        {"choices": [{"delta": {"reasoning": "x"}}]},
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 8192,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        },
    ]
    raw = b"".join(b"data: " + json.dumps(chunk).encode() + b"\n" for chunk in chunks)
    if done:
        raw += b"data: [DONE]\n"

    class Response(io.BytesIO):
        status = 200

    monkeypatch.setattr(
        CHECKS.urllib.request, "urlopen", lambda *a, **kw: Response(raw)
    )
    values = iter([100.0, 100.25, 100.5])
    monkeypatch.setattr(CHECKS.time, "perf_counter", lambda: next(values))
    if done:
        response, index = client.post(
            "/v1/chat/completions", {"model": "m"}, stream=True
        )
        assert index == 0 and response["ttft_seconds"] == 0.25
        assert response["complete"] and response["sse_chunks"] == chunks
    else:
        with pytest.raises(RuntimeError, match="DONE"):
            client.post("/v1/chat/completions", {"model": "m"}, stream=True)
        assert journal.record["requests"][0]["sse_chunks"] == chunks
        assert not journal.record["requests"][0]["complete"]


def test_journal_refuses_overwrite_and_persists_updates_atomically(tmp_path):
    path = tmp_path / "result.json"
    journal = CHECKS.Journal(path, {"execution_status": "running"})
    with pytest.raises(FileExistsError):
        CHECKS.Journal(path, {})
    journal.record["execution_status"] = "failed"
    journal.save()
    assert json.loads(path.read_text())["execution_status"] == "failed"
    assert not list(tmp_path.glob("*.tmp"))


def test_fixed_protocol_orders_excluded_warmups_cache_checks_then_timing(monkeypatch):
    args = NS(
        base_url="http://unit.invalid",
        api_key_env=None,
        model="m",
        request_timeout=1,
        idle_timeout=1,
        semantic_max_tokens=384,
    )
    journal = NS(record={"requests": [], "cases": []}, save=lambda: None)
    client = CHECKS.PrefillChecks(args, journal)
    phases = []
    monkeypatch.setattr(
        client, "control", lambda path: json.dumps({"data": [{"id": "m"}]})
    )
    monkeypatch.setattr(client, "idle", lambda **kw: None)
    monkeypatch.setattr(
        client,
        "prefill",
        lambda size, phase, sample: phases.append((phase, size, sample)),
    )
    monkeypatch.setattr(
        client, "semantic", lambda size, reuse: phases.append(("semantic", size, reuse))
    )
    client.run()
    assert phases == (
        [("warmup", size, 0) for size in CHECKS.SIZES]
        + [
            ("semantic", 8192, True),
            ("semantic", 16384, False),
            ("semantic", 32768, False),
        ]
        + [("measured", size, sample) for sample in range(3) for size in CHECKS.SIZES]
    )


def test_prompt_padding_calibrates_exact_token_count(monkeypatch):
    journal = NS(record={"requests": []}, save=lambda: None)
    client = CHECKS.PrefillChecks(
        NS(base_url="http://unit.invalid", api_key_env=None, model="m"), journal
    )
    monkeypatch.setattr(CHECKS.uuid, "uuid4", lambda: NS(hex="1" * 32))
    counts = []

    def count(messages):
        words = messages[0]["content"].split()
        observed = (
            sum(word in ("alpha", "beta", "gamma", "delta") for word in words) + 37
        )
        counts.append(observed)
        return observed, len(counts) - 1

    monkeypatch.setattr(client, "count", count)
    messages, requests = client.calibrate(8192, "STONE-7482")
    assert counts == [8229, 8192]
    assert requests == [0, 1]
    assert messages[0]["content"].startswith("Test " + "1" * 32)
    assert "The project code is STONE-7482. Remember it." in messages[0]["content"]


def test_generic_conditions_cannot_claim_activation(tmp_path):
    path = tmp_path / "conditions.json"
    path.write_text(
        json.dumps(
            {
                "schema": "glm-prefill-reproduction-conditions/v1",
                "sources": {"vllm": "revision"},
                "settings": {"tp": 4},
            }
        ),
        encoding="utf-8",
    )
    conditions, sha = CHECKS.load_conditions(path)
    assert conditions["sources"] == {"vllm": "revision"} and len(sha) == 64
    conditions["activation_verified"] = True
    path.write_text(json.dumps(conditions), encoding="utf-8")
    with pytest.raises(ValueError, match="Conditions require"):
        CHECKS.load_conditions(path)
    assert CHECKS.load_conditions(None) == (None, None)


def test_successful_run_does_not_assert_feature_activation(tmp_path, monkeypatch):
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        CHECKS.sys,
        "argv",
        [
            str(PATH),
            "--base-url",
            "http://unit.invalid",
            "--output",
            str(output),
            "--label",
            "both",
            "--ready-confirmed",
            "--exclusive-window",
        ],
    )
    monkeypatch.setattr(CHECKS.PrefillChecks, "run", lambda self: None)
    CHECKS.main()
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["execution_status"] == "passed"
    assert result["feature_activation"]["status"] == "not_verified"
    assert result["conditions_provenance"] == "not_supplied"
    assert result["settings"]["repeats"] == 3
    assert result["model_quality_qualified"] is False


@pytest.mark.parametrize(
    "url", ["http://user:secret@unit.invalid", "http://:secret@unit.invalid"]
)
def test_url_credentials_are_rejected_before_journal_creation(
    tmp_path, monkeypatch, url
):
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        CHECKS.sys,
        "argv",
        [
            str(PATH),
            "--base-url",
            url,
            "--output",
            str(output),
            "--label",
            "case",
            "--ready-confirmed",
            "--exclusive-window",
        ],
    )
    with pytest.raises(ValueError, match="without credentials"):
        CHECKS.main()
    assert not output.exists()


def test_api_key_is_sent_but_never_journaled(monkeypatch):
    monkeypatch.setenv("PREFILL_UNIT_KEY", "synthetic-secret")
    journal = NS(record={"requests": []}, save=lambda: None)
    args = NS(
        base_url="http://unit.invalid",
        api_key_env="PREFILL_UNIT_KEY",
        model="m",
        request_timeout=1,
    )
    client = CHECKS.PrefillChecks(args, journal)

    class Response(io.BytesIO):
        status = 200

    def urlopen(request, **kwargs):
        assert request.get_header("Authorization") == "Bearer synthetic-secret"
        return Response(b'{"count": 8192}')

    monkeypatch.setattr(CHECKS.urllib.request, "urlopen", urlopen)
    assert client.count([{"role": "user", "content": "synthetic input"}])[0] == 8192
    assert "synthetic-secret" not in json.dumps(journal.record)
