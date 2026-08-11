# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests: decision records explain every swap and every non-swap."""
import json
import logging

import numpy as np
import pytest

try:
    from vllm.model_executor.layers.quantization.exl3_fungible import (
        decision_log as DL,
        policy as P,
    )
except ImportError:  # standalone: load by path with a stub package
    import importlib.util
    import sys
    import types
    from pathlib import Path as _P

    _dir = (_P(__file__).resolve().parents[2] / "vllm" / "model_executor"
            / "layers" / "quantization" / "exl3_fungible")

    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    P = _load("vllm.model_executor.layers.quantization.exl3_fungible.policy",
              _dir / "policy.py")
    pkg = types.ModuleType("vllm.model_executor.layers.quantization.exl3_fungible")
    pkg.policy = P
    sys.modules["vllm.model_executor.layers.quantization.exl3_fungible"] = pkg
    DL = _load("fq_decision_log_standalone", _dir / "decision_log.py")


def scenario():
    """2 layers x 8 experts; expert 4 is hot+sensitive (wants K4)."""
    L, E = 2, 8
    tier = np.full((L, E), 3)
    tier[:, :3] = 4                       # budget 3 K4 per layer
    count = np.ones((L, E)); count[:, 4] = 500.0
    mass = np.ones((L, E)); mass[:, 4] = 50.0
    eps = {3: np.full((L, E), 0.02), 4: np.full((L, E), 0.005)}
    stats = {"count": count, "mass": mass}
    cfg = {"n_k4": 3, "dwell_steps": 0}
    return stats, eps, tier, cfg


def test_every_swap_has_rationale():
    stats, eps, tier, cfg = scenario()
    swaps = P.decide(stats, eps, tier, cfg=cfg)
    assert swaps, "scenario must produce swaps"
    rec = DL.explain(stats, eps, tier, swaps, cfg=cfg, step=123,
                     policy_sha_before="a" * 64, policy_sha_after="b" * 64)
    assert len(rec["swaps"]) == len(swaps)
    for sw in rec["swaps"]:
        assert sw["expert_in"] == 4
        assert sw["score_in"] > sw["score_out"]
        assert sw["hysteresis_ratio"] > 1.25
        assert sw["mass_in"] == 50.0
    assert rec["totals"]["executed"] == len(swaps)
    json.loads(DL.to_json(rec))  # serializable


def test_blocked_tallies_dwell():
    stats, eps, tier, cfg = scenario()
    cfg = dict(cfg, dwell_steps=1000)
    dwell = np.zeros((2, 8))              # nobody has dwelled long enough
    swaps = P.decide(stats, eps, tier, dwell=dwell, cfg=cfg)
    assert swaps == []
    rec = DL.explain(stats, eps, tier, swaps, dwell=dwell, cfg=cfg)
    assert rec["blocked"]["dwell"] >= 1
    assert rec["totals"]["executed"] == 0


def test_log_lines_emitted(caplog, monkeypatch):
    stats, eps, tier, cfg = scenario()
    swaps = P.decide(stats, eps, tier, cfg=cfg)
    rec = DL.explain(stats, eps, tier, swaps, cfg=cfg, step=7,
                     policy_sha_before="c" * 64, policy_sha_after="d" * 64)
    # When DL is the real package module its logger sits under the "vllm"
    # hierarchy, whose root vllm configures with propagate=False — attach
    # caplog's handler directly so it sees the records either way.
    DL.logger.addHandler(caplog.handler)
    monkeypatch.setattr(DL.logger, "level", logging.INFO)
    try:
        with caplog.at_level(logging.INFO):
            DL.log_decision(rec)
    finally:
        DL.logger.removeHandler(caplog.handler)
    text = caplog.text
    assert "FQ interval step=7" in text
    assert "FQ swap L" in text and "e4" in text
    assert "cccccccc -> dddddddd" in text
