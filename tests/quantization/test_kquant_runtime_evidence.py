# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import pytest

from vllm.compilation import kquant_runtime_evidence as evidence


def test_runtime_evidence_aggregates_all_layers_capture_and_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_path = tmp_path / "qsrt-runtime-paths.json"
    writer = evidence.RuntimeEvidence(evidence_path)
    monkeypatch.setattr(evidence, "_runtime_evidence", writer)
    monkeypatch.setattr(evidence, "runtime_evidence_enabled", True)
    monkeypatch.setattr(evidence, "_capture_active", False)
    monkeypatch.setattr(evidence.torch.cuda, "synchronize", lambda: None)

    assert evidence.begin_graph_capture()
    for layer in range(3, 13):
        evidence.record_layer_execution(layer, "w4a16", False, 2)
        evidence.record_layer_execution(layer, "w4a8", True, 2)
    evidence.record_layer_execution(13, "w4a8", True, 2)
    evidence.record_layer_execution(13, "w4a16", False, 1)
    assert not evidence_path.exists()

    observations = evidence.finish_graph_capture(True, True)
    assert observations is not None
    evidence.record_graph_replay(observations)

    raw = evidence_path.read_text(encoding="utf-8")
    runtime_paths = json.loads(raw)
    assert (
        raw
        == json.dumps(
            runtime_paths,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )
    assert set(runtime_paths) == {
        "schema",
        "version",
        "layers",
        "cudagraph",
        "speculative",
    }
    assert runtime_paths["schema"] == "qsrt_fruit_runtime_paths_v2"
    assert runtime_paths["version"] == 2
    assert set(runtime_paths["layers"]) == {str(layer) for layer in range(3, 14)}
    for layer in range(3, 13):
        layer_evidence = runtime_paths["layers"][str(layer)]
        assert layer_evidence["prefill"] == {"mode": "w4a16", "calls": 1}
        assert layer_evidence["decode"] == {
            "mode": "w4a8",
            "calls": 1,
            "part_count": 2,
            "capture_calls": 1,
            "replay_calls": 1,
        }
    assert runtime_paths["layers"]["13"] == {
        "mtp_prefill": {
            "mode": "w4a16",
            "calls": 1,
            "capture_calls": 1,
            "replay_calls": 1,
        },
        "mtp_decode": {
            "mode": "w4a8",
            "calls": 1,
            "part_count": 2,
            "capture_calls": 1,
            "replay_calls": 1,
        },
    }
    assert runtime_paths["cudagraph"] == {
        "mode": "FULL_AND_PIECEWISE",
        "capture_count": 1,
        "replay_count": 1,
    }
    assert runtime_paths["speculative"] == {
        "method": "mtp",
        "num_speculative_tokens": 1,
        "draft_tokens": 1,
    }


def test_replay_accounting_is_limited_to_captured_layers(tmp_path: Path) -> None:
    evidence_path = tmp_path / "qsrt-runtime-paths.json"
    writer = evidence.RuntimeEvidence(evidence_path)
    observations = bytearray(22)
    observations[10] = 2  # Layer 3 W4A8 decode.

    writer.observe_capture(bytes(observations))
    writer.observe_replay(bytes(observations))

    runtime_paths = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert runtime_paths["layers"]["3"]["decode"]["capture_calls"] == 1
    assert runtime_paths["layers"]["3"]["decode"]["replay_calls"] == 1
    assert runtime_paths["layers"]["4"]["decode"]["capture_calls"] == 0
    assert runtime_paths["layers"]["4"]["decode"]["replay_calls"] == 0


def test_graph_replay_stops_synchronizing_after_evidence_saturates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_path = tmp_path / "qsrt-runtime-paths.json"
    writer = evidence.RuntimeEvidence(evidence_path)
    monkeypatch.setattr(evidence, "_runtime_evidence", writer)
    synchronizations: list[None] = []
    monkeypatch.setattr(
        evidence.torch.cuda,
        "synchronize",
        lambda: synchronizations.append(None),
    )
    layer_three = bytes([0] * 10 + [2] + [0] * 11)

    for _ in range(100):
        evidence.record_graph_replay(layer_three)

    assert len(synchronizations) == 1
    assert writer.replay_saturated(layer_three)

    layer_four = bytes([0] * 11 + [2] + [0] * 10)
    writer.observe_capture(layer_four)
    evidence.record_graph_replay(layer_four)

    assert len(synchronizations) == 2
    assert writer.replay_saturated(layer_four)
    runtime_paths = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert runtime_paths["layers"]["4"]["decode"]["replay_calls"] == 1


def test_disjoint_captures_before_replay_each_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_path = tmp_path / "qsrt-runtime-paths.json"
    writer = evidence.RuntimeEvidence(evidence_path)
    monkeypatch.setattr(evidence, "_runtime_evidence", writer)
    synchronizations: list[None] = []
    monkeypatch.setattr(
        evidence.torch.cuda,
        "synchronize",
        lambda: synchronizations.append(None),
    )
    prefill = bytes([2] + [0] * 21)
    decode = bytes([0] * 10 + [2] + [0] * 11)
    mtp_decode = bytes([0] * 20 + [2, 0])
    mtp_prefill = bytes([0] * 21 + [2])

    for observations in (prefill, decode, mtp_decode, mtp_prefill):
        writer.observe_capture(observations)
    for observations in (prefill, decode, mtp_decode, mtp_prefill):
        evidence.record_graph_replay(observations)
        evidence.record_graph_replay(observations)

    assert len(synchronizations) == 4
    runtime_paths = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert runtime_paths["layers"]["3"]["decode"]["replay_calls"] == 1
    assert runtime_paths["layers"]["13"]["mtp_decode"]["replay_calls"] == 1
    assert runtime_paths["layers"]["13"]["mtp_prefill"]["replay_calls"] == 1
    assert runtime_paths["speculative"]["draft_tokens"] == 1


def test_runtime_evidence_counters_are_bounded(tmp_path: Path) -> None:
    evidence_path = tmp_path / "qsrt-runtime-paths.json"
    writer = evidence.RuntimeEvidence(evidence_path)

    for _ in range(100):
        writer.observe_layer(3, "w4a8", True, 2)
        writer.observe_layer(13, "w4a16", False, 1)
        writer.observe_capture(bytes([0] * 10 + [2] + [0] * 11))
        writer.observe_replay(bytes([0] * 10 + [2] + [0] * 11))

    runtime_paths = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert runtime_paths["layers"]["3"]["decode"] == {
        "mode": "w4a8",
        "calls": 1,
        "part_count": 2,
        "capture_calls": 1,
        "replay_calls": 1,
    }
    assert runtime_paths["layers"]["13"]["mtp_prefill"]["calls"] == 1
    assert runtime_paths["cudagraph"]["capture_count"] == 1
    assert runtime_paths["cudagraph"]["replay_count"] == 1
