# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for PolicyStore: atomic commit, rotation, rehydration, refusal."""
import json

import pytest

try:
    from vllm.model_executor.layers.quantization.exl3_fungible import store as st
except ImportError:  # standalone run against an env without built vllm._C
    import importlib.util
    from pathlib import Path as _P

    _p = (_P(__file__).resolve().parents[2] / "vllm" / "model_executor"
          / "layers" / "quantization" / "exl3_fungible" / "store.py")
    _s = importlib.util.spec_from_file_location("fq_store_standalone", _p)
    st = importlib.util.module_from_spec(_s)
    _s.loader.exec_module(st)


def make_doc(manifest="m0", n_k4=2, e=4, layer="3"):
    bits = [4] * n_k4 + [3] * (e - n_k4)
    return {
        "schema": "fq-policy/2",
        "manifest": manifest,
        "budget": {"mode": "fixed_cardinality", "n_k4_per_layer": {layer: n_k4}},
        "bits_per_expert": {layer: bits},
        "pinned": {},
        "provenance": {"parent": "generic-v1"},
    }


def test_commit_and_rehydrate(tmp_path):
    s = st.PolicyStore(tmp_path, "m0")
    h = s.commit(make_doc(), num_experts=4)
    assert len(h) == 64
    doc = s.load_current(num_experts=4)
    assert doc["bits_per_expert"]["3"] == [4, 4, 3, 3]


def test_history_rotation_bounded(tmp_path):
    s = st.PolicyStore(tmp_path, "m0")
    for i in range(st.HISTORY_KEEP + 4):
        d = make_doc()
        d["provenance"]["gen"] = i
        s.commit(d, num_experts=4)
    assert len(s.history()) == st.HISTORY_KEEP


def test_manifest_mismatch_refused(tmp_path):
    s = st.PolicyStore(tmp_path, "m0")
    with pytest.raises(ValueError, match="manifest"):
        s.commit(make_doc(manifest="OTHER"), num_experts=4)
    s2 = st.PolicyStore(tmp_path, "m0")
    s2.commit(make_doc(), num_experts=4)
    cur = s2.root / "current.json"
    doc = json.loads(cur.read_text())
    doc["manifest"] = "TAMPERED"
    cur.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="mismatch"):
        s2.load_current()


def test_validation_rejects_bad_docs(tmp_path):
    s = st.PolicyStore(tmp_path, "m0")
    bad = make_doc()
    bad["bits_per_expert"]["3"][0] = 5
    with pytest.raises(ValueError, match=r"non-\{3,4\}"):
        s.commit(bad, num_experts=4)
    topo = make_doc()
    topo["rank"] = 0
    with pytest.raises(ValueError, match="topology-neutral"):
        s.commit(topo, num_experts=4)
    off_budget = make_doc()
    off_budget["bits_per_expert"]["3"] = [4, 4, 4, 3]  # 3 K4 vs declared 2
    with pytest.raises(ValueError, match="occupancy"):
        s.commit(off_budget, num_experts=4)


def test_no_torn_current_on_crash_sim(tmp_path):
    s = st.PolicyStore(tmp_path, "m0")
    s.commit(make_doc(), num_experts=4)
    # simulate a crashed writer leaving a .tmp behind
    (s.root / "current.tmp").write_text("{ definitely not json")
    doc = s.load_current(num_experts=4)
    assert doc is not None, "stray .tmp must never shadow current.json"
