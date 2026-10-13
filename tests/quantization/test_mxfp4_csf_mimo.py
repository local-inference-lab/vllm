# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiMo-family loader contract tests (Codex §12.2 / §9.3 requests).

The strict geometry + source-index identity pin applies ONLY to the
``mimo_v26_pro_mxfp4`` family. Upstream families (deepseek_v41, kimi_k3,
deepseek_v4_flash) keep stock pass-through behavior for geometry-mismatched
configs and are covered by their own upstream fixtures.
"""

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.mxfp4_csf_loader import (
    CODEC,
    SCHEMA,
    Mxfp4CsfModelLoader,
)


def _mimo_container(tmp_path, *, with_identity=True, source_index=None):
    """A minimal valid MiMo CSF container + matching model root."""
    tensor_dir = tmp_path / "container"
    tensor_dir.mkdir()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    name = "model-00001.safetensors"
    tensors = {
        "model.layers.5.mlp.experts.0.w1.weight": torch.ones(
            8, dtype=torch.uint8
        ),
        "model.layers.5.mlp.experts.0.w1.scale.mxfp4_csf_fixed": torch.ones(
            8, dtype=torch.uint8
        ),
        "norm.weight": torch.ones(8, dtype=torch.bfloat16),
    }
    save_file(tensors, tensor_dir / name)
    common = {
        "schema": SCHEMA,
        "codec": CODEC,
        "family": "mimo_v26_pro_mxfp4",
    }
    contract = {
        **common,
        "shards": [{"file": name}],
        "source_names": {k: name for k in tensors},
    }
    if with_identity:
        contract["source_index_sha256"] = (
            source_index
            if source_index is not None
            else hashlib.sha256(b"model.safetensors.index.json").hexdigest()
        )
    (tensor_dir / "build-contract.json").write_text(json.dumps(contract))
    (tensor_dir / "manifest.json").write_text(
        json.dumps({**common, "shards": [{"file": name}]})
    )
    # model root with the index the identity pin checks
    index_bytes = b"model.safetensors.index.json"
    (model_dir / "model.safetensors.index.json").write_bytes(index_bytes)
    default_sha = hashlib.sha256(index_bytes).hexdigest()
    return tensor_dir, model_dir, default_sha


def _mimo_model_config(container, model_dir, *, geometry=(384, 6144, 2048)):
    e, h, n = geometry
    return SimpleNamespace(
        hf_config=SimpleNamespace(
            quantization_config={
                "quant_method": "mxfp4_csf",
                "checkpoint_root": str(container),
            },
            num_hidden_layers=70,
        ),
        hf_text_config=SimpleNamespace(
            model_type="mimo_v2",
            n_routed_experts=e,
            hidden_size=h,
            moe_intermediate_size=n,
        ),
        model=str(model_dir),
    )


def test_mimo_identity_pin_accepts_matching_source_index(tmp_path):
    """Positive MiMo case: container + matching source index resolves."""
    container, model_dir, sha = _mimo_container(tmp_path)
    contract = json.loads(
        (container / "build-contract.json").read_text()
    )
    contract["source_index_sha256"] = sha
    (container / "build-contract.json").write_text(json.dumps(contract))

    loader = Mxfp4CsfModelLoader(LoadConfig(load_format="mxfp4_csf"))
    resolved = loader._resolve_csf_root(
        contract_like := {
            "quant_method": "mxfp4_csf",
            "checkpoint_root": str(container),
        },
        _mimo_model_config(container, model_dir).hf_text_config,
        model_root=str(model_dir),
    )
    assert resolved is not None
    assert resolved[1]["family"] == "mimo_v26_pro_mxfp4"


def test_mimo_identity_pin_rejects_foreign_source_index(tmp_path):
    """Foreign source checkpoint (index sha mismatch) must refuse."""
    import pytest

    container, model_dir, _ = _mimo_container(tmp_path)
    contract = json.loads(
        (container / "build-contract.json").read_text()
    )
    contract["source_index_sha256"] = (
        hashlib.sha256(b"some-other-checkpoints-index").hexdigest()
    )
    (container / "build-contract.json").write_text(json.dumps(contract))

    loader = Mxfp4CsfModelLoader(LoadConfig(load_format="mxfp4_csf"))
    with pytest.raises(ValueError, match="built from a different source"):
        loader._resolve_csf_root(
            {
                "quant_method": "mxfp4_csf",
                "checkpoint_root": str(container),
            },
            _mimo_model_config(container, model_dir).hf_text_config,
            model_root=str(model_dir),
        )


def test_mimo_identity_pin_requires_recorded_sha(tmp_path):
    """MiMo container without source_index_sha256 must refuse (no bypass)."""
    import pytest

    container, model_dir, _ = _mimo_container(tmp_path, with_identity=False)
    loader = Mxfp4CsfModelLoader(LoadConfig(load_format="mxfp4_csf"))
    with pytest.raises(ValueError, match="lacks source_index_sha256"):
        loader._resolve_csf_root(
            {
                "quant_method": "mxfp4_csf",
                "checkpoint_root": str(container),
            },
            _mimo_model_config(container, model_dir).hf_text_config,
            model_root=str(model_dir),
        )


def test_mimo_geometry_mismatch_refuses_csf_load(tmp_path):
    """Explicit CSF intent on a geometry-mismatched MiMo config refuses."""
    import pytest

    container, model_dir, _ = _mimo_container(tmp_path)
    loader = Mxfp4CsfModelLoader(LoadConfig(load_format="mxfp4_csf"))
    with pytest.raises(ValueError, match="geometry does not match"):
        loader._resolve_csf_root(
            {
                "quant_method": "mxfp4_csf",
                "checkpoint_root": str(container),
            },
            _mimo_model_config(
                container, model_dir, geometry=(256, 6144, 2048)
            ).hf_text_config,
            model_root=str(model_dir),
        )


def test_mimo_plain_source_checkpoint_passes_through(tmp_path):
    """Hybrid MiMo source WITHOUT CSF intent: plain FP8 launch, no raise.

    The hybrid intent gate (fp8 + store_dtype mxfp4 + model_type mimo_v2*)
    requires MIMO_CSF_ROOT or checkpoint_root; absent that, this is the plain
    source checkpoint served as FP8 — must return None (probe contract).
    """
    container, model_dir, _ = _mimo_container(tmp_path)
    loader = Mxfp4CsfModelLoader(LoadConfig(load_format="mxfp4_csf"))
    text_config = _mimo_model_config(container, model_dir).hf_text_config
    resolved = loader._resolve_csf_root(
        {"quant_method": "fp8"}, text_config, model_root=str(model_dir)
    )
    assert resolved is None


def test_mimo_dflash_draft_passes_through(tmp_path):
    """DFlash draft model (no quantization_config at all) passes through."""
    container, model_dir, _ = _mimo_container(tmp_path)
    loader = Mxfp4CsfModelLoader(LoadConfig(load_format="mxfp4_csf"))
    text_config = _mimo_model_config(container, model_dir).hf_text_config
    resolved = loader._resolve_csf_root(
        None, text_config, model_root=str(model_dir)
    )
    assert resolved is None


def test_non_mimo_family_keeps_stock_geometry_pass_through(tmp_path):
    """Codex §12.2 regression: upstream family + geometry-mismatched config
    must return None (pass-through), NOT raise — stock upstream behavior."""
    import pytest

    container, model_dir, _ = _mimo_container(tmp_path)
    # rewrite container as deepseek_v41 family
    contract = json.loads(
        (container / "build-contract.json").read_text()
    )
    contract["family"] = "deepseek_v41"
    (container / "build-contract.json").write_text(json.dumps(contract))
    manifest = json.loads((container / "manifest.json").read_text())
    manifest["family"] = "deepseek_v41"
    (container / "manifest.json").write_text(json.dumps(manifest))

    loader = Mxfp4CsfModelLoader(LoadConfig(load_format="mxfp4_csf"))
    # MiMo geometry config against a deepseek_v41 container with explicit
    # mxfp4_csf intent: stock resolves any valid container (no geometry gate
    # for non-MiMo families); the MiMo-scoped gate must not raise here.
    resolved = loader._resolve_csf_root(
        {
            "quant_method": "mxfp4_csf",
            "checkpoint_root": str(container),
        },
        _mimo_model_config(container, model_dir).hf_text_config,
        model_root=str(model_dir),
    )
    assert resolved is not None
    assert resolved[1]["family"] == "deepseek_v41"


def test_mimo_hybrid_intent_with_csf_root_resolves(tmp_path):
    """Hybrid intent (fp8 + store_dtype mxfp4 + mimo_v2*) + MIMO_CSF_ROOT
    resolves the CSF container (this is the production fast profile)."""
    import os

    container, model_dir, sha = _mimo_container(tmp_path)
    contract = json.loads(
        (container / "build-contract.json").read_text()
    )
    contract["source_index_sha256"] = sha
    (container / "build-contract.json").write_text(json.dumps(contract))

    loader = Mxfp4CsfModelLoader(LoadConfig(load_format="mxfp4_csf"))
    text_config = _mimo_model_config(container, model_dir).hf_text_config
    resolved = loader._resolve_csf_root(
        {
            "quant_method": "fp8",
            "store_dtype": "mxfp4",
            "checkpoint_root": str(container),
        },
        text_config,
        model_root=str(model_dir),
    )
    assert resolved is not None
    assert resolved[1]["family"] == "mimo_v26_pro_mxfp4"


# ---------------------------------------------------------------------------
# MXFP4/CSF scale prefetch (Codex §9.2/§9.3/§9.5): MiMo-owned orchestration.
# ---------------------------------------------------------------------------


def _stub_method(owner, index, *, backend=None, prepared=None):
    """A Mxfp4CsfMoEMethod with the prefetch fields but no weights.

    ``owner`` is the prefetch STATE (MimoMxfp4CsfScalePrefetch); the owner
    namespace mirrors MimoMxfp4CsfConfig's capability contract. ``prepared``
    defaults to a fake CSF payload: a namespace whose ``_impl`` retains the
    native CSF decoder, which is what the (fixed) eligibility derivation
    and the load-time support check consult.
    """
    from vllm.models.deepseek_v4_1.mxfp4_csf import Mxfp4CsfMoEMethod

    owner_ns = SimpleNamespace(
        x4t_scale_prefetch=True, scale_prefetch_state=owner
    )
    if owner.scale_stream is None:
        # register() is never called on stubs; give the state a fake side stream.
        owner.scale_stream = _FakeStream()
    method = object.__new__(Mxfp4CsfMoEMethod)
    method.owner = owner_ns
    method.layer_index = index
    method.scale_prefetch_owner = owner
    method.backend = backend if backend is not None else SimpleNamespace(
        scales_expanded=False
    )
    method.moe_done = SimpleNamespace(record=lambda stream: None)
    method.scales_ready = SimpleNamespace(record=lambda stream: None)
    method.prepared = (
        prepared
        if prepared is not None
        else SimpleNamespace(_impl=SimpleNamespace(nvfp4_csf=object()))
    )
    method.moe_kernel = SimpleNamespace()
    return method


class _FakeEvent:
    def __init__(self):
        self.recorded_on = None

    def record(self, stream):
        self.recorded_on = stream


class _FakeStream:
    """Minimal torch.cuda.Stream stand-in (records waits, no CUDA)."""

    def __init__(self, device=None):
        self.waits = []
        self.entered = 0
        # torch.cuda.StreamContext.__enter__ reads `.device` on CUDA-visible
        # hosts (to decide whether a device switch is needed before
        # set_stream); on CPU-only hosts it early-outs (idx == -1) and never
        # reads it.  Mirror the current device so the stub works in both
        # environments; None stays inert where it is never read.
        self.device = device if device is not None else (
            torch.cuda.current_device() if torch.cuda.is_available() else None
        )

    def wait_event(self, event):
        self.waits.append(event)

    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, *exc):
        return False


def _arm_stub_cuda(monkeypatch):
    """Replace the CUDA stream/event surface with CPU stubs."""
    main = _FakeStream()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a, **k: main)
    monkeypatch.setattr(torch.cuda, "Stream", lambda *a, **k: _FakeStream(*a))
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    # torch.cuda.StreamContext.__enter__ installs the selected stream via
    # set_stream when a GPU is visible; record the enter on the fake and keep
    # torch's bookkeeping out of the way (we never launch real work).  On
    # CPU-only hosts StreamContext early-outs (idx == -1) and this is never
    # called — kept harmless there.
    monkeypatch.setattr(
        torch.cuda, "set_stream",
        lambda s: setattr(s, "entered", getattr(s, "entered", 0) + 1),
    )
    # torch.cuda.is_current_stream_capturing raises on CPU-only hosts; the
    # production helper's contract is "False when not capturing".
    monkeypatch.setattr(
        "vllm.models.deepseek_v4_1.mxfp4_csf._is_current_stream_capturing",
        lambda: False,
    )
    return main


def _step(monkeypatch, *, starts, spec_tokens=3, gdn_prefills=None):
    """One forward-context step with a decodes-first batch (Codex 9.5 shape).

    Real-shaped host metadata: the paged metadata carries query_start_loc /
    num_actual_tokens / max_query_len. ``gdn_prefills`` adds a second
    (GDN/KDA-style) metadata entry whose ``num_spec_decode_tokens`` follows
    the REAL gdn_attn.py:555 builder semantics: the AGGREGATE speculative
    token count for the whole batch (sum of spec-decode query rows), NOT a
    per-request width. Codex §10.2.2: ``_batch_has_prefill_rows`` must
    classify with the speculative-config width only and IGNORE that
    aggregate — an aggregate-as-width refinement lets many verification
    requests suppress recognition of a shorter real prefill. (Production
    MiMo's B12xPagedMetadata has no num_spec_decode_tokens field at all, so
    MiMo behavior is unchanged; this fixture pins the corrected semantics
    for backends that do report it.)
    """
    import vllm.config as vllm_config
    import vllm.forward_context as forward_context

    # Production width source: the speculative config only (B12xPagedMetadata
    # does not carry num_spec_decode_tokens; GDN's aggregate is not a width).
    monkeypatch.setattr(
        vllm_config,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(
            speculative_config=SimpleNamespace(num_speculative_tokens=spec_tokens)
        ),
    )
    max_query = max(b - a for a, b in zip(starts, starts[1:]))
    metadata = {
        "attn": SimpleNamespace(
            query_start_loc=torch.tensor(starts, dtype=torch.int32),
            num_actual_tokens=starts[-1],
            max_query_len=max_query,
        )
    }
    if gdn_prefills is not None:
        # gdn_attn.py:555: num_spec_decode_tokens = total spec-decode rows.
        # For a uniform-8 verification batch of n requests this is 8*n (the
        # AGGREGATE), which is NOT any request's query width (that is 8).
        spec_rows = starts[-1] - gdn_prefills * max_query if gdn_prefills else 0
        metadata["kda"] = SimpleNamespace(
            num_prefills=gdn_prefills,
            num_spec_decode_tokens=spec_rows,
            max_query_len=max_query,
        )
    context = SimpleNamespace(attn_metadata=metadata)
    monkeypatch.setattr(
        forward_context, "is_forward_context_available", lambda: True
    )
    monkeypatch.setattr(forward_context, "get_forward_context", lambda: context)
    return context


def test_mimo_owner_carries_prefetch_capability():
    """MiMo owner is the ONLY owner that arms (Codex 9.1-3)."""
    from vllm.model_executor.layers.quantization.mxfp4_csf import (
        MimoMxfp4CsfConfig,
    )
    from vllm.models.deepseek_v4_1.mxfp4_csf import DeepseekV41Mxfp4CsfConfig

    assert getattr(MimoMxfp4CsfConfig, "x4t_scale_prefetch", False) is True
    # Non-MiMo owners never set the capability attribute: unarmed even with
    # the env on (blast radius).
    assert not getattr(DeepseekV41Mxfp4CsfConfig, "x4t_scale_prefetch", False)


def test_mxfp4_scale_prefetch_env_defaults_off():
    """VLLM_B12X_MXFP4_CSF_SCALE_PREFETCH defaults OFF (unlike the NVFP4 twin)."""
    import vllm.envs as envs

    assert envs.VLLM_B12X_MXFP4_CSF_SCALE_PREFETCH is False
    # The NVFP4 twin keeps its own default; this diff must not touch it.
    assert envs.VLLM_B12X_CSF_SCALE_PREFETCH is True


def test_row_type_eligibility_rejects_dflash_verification_and_decode(monkeypatch):
    """Codex 9.5: arm only for actual eligible prefill rows.

    DFlash verification batches (uniform 1+spec-token queries) and pure decode
    batches must NOT arm, whatever their row count — a total-row threshold
    would arm for exactly these cases.
    """
    from vllm.models.deepseek_v4_1.mxfp4_csf import _batch_has_prefill_rows

    # Pure decode: 4 requests, uniform_query_len=4 rows each (MTP=3 ->
    # 1 + 3 spec tokens). No prefill rows -> must NOT arm.
    _step(monkeypatch, starts=[0, 4, 8, 12, 16], spec_tokens=3)
    assert _batch_has_prefill_rows() is False

    # DFlash verification: uniform 1+spec-token queries, 8 rows each
    # (DFlash=7 -> uniform_query_len=8). Many rows, but ALL decode-class.
    _step(monkeypatch, starts=[0, 8, 16, 24, 32], spec_tokens=7, gdn_prefills=0)
    assert _batch_has_prefill_rows() is False

    # A real prefill request (query 512) among decodes: arm.
    _step(monkeypatch, starts=[0, 8, 520])
    assert _batch_has_prefill_rows() is True

    # Missing metadata: never arm (fail closed).
    import vllm.forward_context as forward_context

    monkeypatch.setattr(
        forward_context, "is_forward_context_available", lambda: True
    )
    monkeypatch.setattr(
        forward_context,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=None),
    )
    assert _batch_has_prefill_rows() is False


def test_aggregate_spec_tokens_never_inflate_the_classification_width(monkeypatch):
    """Codex §10.2.2: metadata ``num_spec_decode_tokens`` is the batch-wide
    AGGREGATE speculative token count (gdn_attn.py:555), never a per-request
    width. With MANY verification requests it can far exceed a shorter REAL
    prefill query; treating it as a width would suppress arming for a batch
    that does contain prefill rows. Classification must use the
    speculative-config width (``_spec_decode_query_width``) only."""

    from vllm.models.deepseek_v4_1.mxfp4_csf import _batch_has_prefill_rows

    # DFlash7 verification requests (width 8 each) plus one 16-token prefill.
    # The GDN aggregate (8 * 63 = 504 spec rows) dwarfs the 16-token prefill;
    # the aggregate-as-width refinement would misread every query as
    # decode-class and refuse to arm. The corrected classifier arms.
    _step(
        monkeypatch,
        starts=[0] + [8 * (i + 1) for i in range(63)] + [8 * 63 + 16],
        spec_tokens=7,
        gdn_prefills=1,
    )
    assert _batch_has_prefill_rows() is True, (
        "a batch with a real 16-token prefill must arm even when the "
        "GDN aggregate spec-token count (504) exceeds that query length"
    )

    # Control: the same verification-heavy batch with NO prefill rows must
    # still refuse to arm (the aggregate is ignored, not the prefill rows).
    _step(
        monkeypatch,
        starts=[0] + [8 * (i + 1) for i in range(63)],
        spec_tokens=7,
        gdn_prefills=0,
    )
    assert _batch_has_prefill_rows() is False, (
        "a pure verification batch must never arm, whatever its row count"
    )


def test_prefetch_off_is_unarmed(monkeypatch):
    """With the env OFF the method runs today's path exactly: no skip flag,
    no side stream, no expansion."""
    from vllm.models.deepseek_v4_1.mxfp4_csf import MimoMxfp4CsfScalePrefetch

    _arm_stub_cuda(monkeypatch)
    monkeypatch.setattr("vllm.envs.VLLM_B12X_MXFP4_CSF_SCALE_PREFETCH", False)
    state = MimoMxfp4CsfScalePrefetch()
    calls = []
    backend = SimpleNamespace(scales_expanded=False)
    method = _stub_method(state, 3, backend=backend)
    # env OFF -> the method never registers (scale_prefetch_owner is None).
    method.scale_prefetch_owner = None
    method.moe_kernel = SimpleNamespace(
        apply=lambda **_: calls.append(backend.scales_expanded) or "out"
    )
    layer = SimpleNamespace(
        w13_weight=None,
        w2_weight=None,
        activation=None,
        global_num_experts=4,
        expert_map=None,
        apply_router_weight_on_input=False,
    )
    result = method.apply(layer, torch.zeros(8, 4), None, None, None, None)
    assert result == "out"
    assert calls == [False], "prefetch OFF must never set the skip flag"
    assert state.scale_prefetch is None


def test_stale_generation_is_awaited_and_never_reused(monkeypatch):
    """Codex §9.3 point 3/4: a stale/mismatched generation still needs a wait
    before an inline expansion overwrites the shared scratch, and its scales
    are NEVER consumed."""
    from vllm.models.deepseek_v4_1.mxfp4_csf import (
        MimoMxfp4CsfScalePrefetch,
        _forward_token,
    )

    main = _arm_stub_cuda(monkeypatch)
    monkeypatch.setattr("vllm.envs.VLLM_B12X_MXFP4_CSF_SCALE_PREFETCH", True)
    state = MimoMxfp4CsfScalePrefetch()
    backend = SimpleNamespace(scales_expanded=False)
    method = _stub_method(state, 3, backend=backend)
    seen = []
    method.moe_kernel = SimpleNamespace(
        apply=lambda **_: seen.append(backend.scales_expanded)
    )
    layer = SimpleNamespace(
        w13_weight=None,
        w2_weight=None,
        activation=None,
        global_num_experts=4,
        expert_map=None,
        apply_router_weight_on_input=False,
    )
    x = torch.zeros(512, 4)

    # A pending prefetch for a DIFFERENT generation (wrong token, wrong layer).
    stale_event = _FakeEvent()
    state.scale_prefetch = (3, 999999, stale_event)
    _step(monkeypatch, starts=[0, 8, 520])
    method.apply(layer, x, None, None, None, None)

    # The stale event was awaited (scratch race protection) ...
    assert stale_event in main.waits, (
        "a stale prefetch must still be awaited before the inline expansion"
    )
    # ... but NOT consumed as this call's scales.
    assert seen == [False], "stale generation must never be reused"
    assert state.scale_prefetch is None
    # The skip flag is cleared after the call (no leakage into the next one).
    assert backend.scales_expanded is False

    # A matching generation IS consumed. The token must be created for the
    # SAME forward context the apply runs in (each step gets its own token).
    fresh = _FakeEvent()
    _step(monkeypatch, starts=[0, 8, 520])
    token = _forward_token()
    state.scale_prefetch = (3, token, fresh)
    method.apply(layer, x, None, None, None, None)
    assert seen[-1] is True, "matching (layer, token) must consume the prefetch"
    assert fresh in main.waits
    assert backend.scales_expanded is False, "flag must be cleared in finally"


def test_cancellation_clears_skip_flag_and_capture_clears_pending(monkeypatch):
    """A failed forward clears the per-call skip flag; capture entry CLEARS
    (not drains) a stale pending triple (Codex §10.1)."""
    from vllm.models.deepseek_v4_1.mxfp4_csf import MimoMxfp4CsfScalePrefetch

    main = _arm_stub_cuda(monkeypatch)
    monkeypatch.setattr("vllm.envs.VLLM_B12X_MXFP4_CSF_SCALE_PREFETCH", True)
    state = MimoMxfp4CsfScalePrefetch()
    backend = SimpleNamespace(scales_expanded=False)
    method = _stub_method(state, 3, backend=backend)

    def boom(**_):
        raise RuntimeError("forward failed")

    method.moe_kernel = SimpleNamespace(apply=boom)
    layer = SimpleNamespace(
        w13_weight=None,
        w2_weight=None,
        activation=None,
        global_num_experts=4,
        expert_map=None,
        apply_router_weight_on_input=False,
    )
    x = torch.zeros(512, 4)
    _step(monkeypatch, starts=[0, 8, 520])
    with pytest.raises(RuntimeError, match="forward failed"):
        method.apply(layer, x, None, None, None, None)
    assert backend.scales_expanded is False, (
        "the skip flag must be cleared in finally even on a failed forward"
    )

    # Graph transition (Codex §10.1): capture entry CLEARS a pending triple
    # WITHOUT waiting — an event wait inside capture raises
    # cudaErrorStreamCaptureIsolation (invalidating the capture) even for a
    # completed event, and the serving runner device-synchronizes BEFORE
    # capture so the side-stream expansion is already complete. Draining
    # (waiting) here is not merely unnecessary, it is illegal.
    inflight = _FakeEvent()
    state.scale_prefetch = (3, 1, inflight)
    monkeypatch.setattr(
        "vllm.models.deepseek_v4_1.mxfp4_csf._is_current_stream_capturing",
        lambda: True,
    )
    method.moe_kernel = SimpleNamespace(apply=lambda **_: "captured")
    assert method.apply(layer, x, None, None, None, None) == "captured"
    assert state.scale_prefetch is None, (
        "capture entry must CLEAR the pending triple (the captured graph "
        "runs inline and can never consume it)"
    )
    assert inflight not in main.waits, (
        "capture entry must NOT wait the pending event: an event wait inside "
        "capture raises cudaErrorStreamCaptureIsolation (§10.1)"
    )


def test_mixed_decode_prefill_arms_for_prefill_rows_only(monkeypatch):
    """Mixed decode/prefill splits: the producer arms only for a call with
    actual prefill rows (row-type eligibility), never by total row count.

    Codex review fix: eligibility is derived from the prepared X4T consumer
    with the REAL environment readers -- the production MiMo environment
    (B12X_W4A16_MXFP4_PREFILL_MIN_TOKENS=512, NVFP4 knob UNSET) arms. The
    reviewed commit consulted the NVFP4 knob and never armed; the old test
    monkeypatched _w4a16_a4_prefill_enabled=True and masked that. No
    eligibility helper is patched here.
    """
    from vllm.models.deepseek_v4_1.mxfp4_csf import MimoMxfp4CsfScalePrefetch

    monkeypatch.setenv("B12X_W4A16_MXFP4_PREFILL_MIN_TOKENS", "512")
    monkeypatch.delenv("B12X_W4A16_A4_PREFILL_MIN_TOKENS", raising=False)
    _arm_stub_cuda(monkeypatch)
    monkeypatch.setattr("vllm.envs.VLLM_B12X_MXFP4_CSF_SCALE_PREFETCH", True)
    expanded = []

    def fake_expand(prepared):
        expanded.append(prepared)
        return True

    import b12x.moe.fused_moe as fused_moe

    monkeypatch.setattr(fused_moe, "expand_scales", fake_expand)

    state = MimoMxfp4CsfScalePrefetch()
    layers = {i: _stub_method(state, i) for i in (3, 4)}
    for method in layers.values():
        state.scale_layers[method.layer_index] = method
        method.moe_kernel = SimpleNamespace(apply=lambda **_: None)
    layer = SimpleNamespace(
        w13_weight=None,
        w2_weight=None,
        activation=None,
        global_num_experts=4,
        expert_map=None,
        apply_router_weight_on_input=False,
    )

    # Mixed batch: 8 decode-class rows (DFlash7 verification width) + 512
    # prefill rows. The call is prefill-eligible -> arm for layer 4.
    _step(monkeypatch, starts=[0, 8, 520], spec_tokens=7, gdn_prefills=1)
    x = torch.zeros(520, 4)
    for method in layers.values():
        method.apply(layer, x, None, None, None, None)
    assert expanded == [layers[4].prepared], (
        "mixed decode/prefill must arm for the prefill rows' next layer"
    )

    # Pure-decode batch (DFlash verification shape): 32 rows, 0 prefill rows.
    # Even with many rows, no arming (Codex 9.5).
    expanded.clear()
    _step(monkeypatch, starts=[0, 8, 16, 24, 32], spec_tokens=7, gdn_prefills=0)
    x = torch.zeros(32, 4)
    for method in layers.values():
        method.apply(layer, x, None, None, None, None)
    assert expanded == [], (
        "DFlash verification / pure decode must never arm the prefetch"
    )


def test_first_and_last_layer_handled(monkeypatch):
    """First layer (nothing pending) and last layer (nothing to arm) are safe."""
    from vllm.models.deepseek_v4_1.mxfp4_csf import MimoMxfp4CsfScalePrefetch

    _arm_stub_cuda(monkeypatch)
    monkeypatch.setattr("vllm.envs.VLLM_B12X_MXFP4_CSF_SCALE_PREFETCH", True)
    state = MimoMxfp4CsfScalePrefetch()
    calls = []
    last = _stub_method(state, 3)
    last.moe_kernel = SimpleNamespace(
        apply=lambda **_: calls.append(last.backend.scales_expanded)
    )
    state.scale_layers[3] = last
    layer = SimpleNamespace(
        w13_weight=None,
        w2_weight=None,
        activation=None,
        global_num_experts=4,
        expert_map=None,
        apply_router_weight_on_input=False,
    )
    _step(monkeypatch, starts=[0, 8, 520])
    # Layer 3 is the last registered layer: no layer 4 to arm, and no pending
    # prefetch (first call of the forward).
    last.apply(layer, torch.zeros(512, 4), None, None, None, None)
    assert calls == [False], "first/last layer must run the inline path"
    assert state.scale_prefetch is None


# ---------------------------------------------------------------------------
# Codex §8 review fix tests (2026-10-07): the reviewed pair's P1 regressions.
# ---------------------------------------------------------------------------


def test_mimo_config_factory_constructs_prefetch_state_gate_off_and_on(monkeypatch, tmp_path):
    """Codex §8.1 P1 regression: the REAL config factory must construct the
    owner state with the prefetch gate OFF and ON.

    The reviewed commit called MimoMxfp4CsfScalePrefetch(layer_index=-1) --
    a constructor that takes no arguments -- so every
    MimoMxfp4CsfConfig.from_config() raised TypeError and MiMo could not load
    at all. Separate config instances must carry SEPARATE owner state (one
    per model load); the loader-root resolution here needs no real
    checkpoint because from_config never touches tensors.
    """
    from vllm.model_executor.layers.quantization.mxfp4_csf import (
        MimoMxfp4CsfConfig,
    )
    from vllm.models.deepseek_v4_1.mxfp4_csf import MimoMxfp4CsfScalePrefetch

    root = str(tmp_path / "unused-cpu-config-probe")

    # Gate OFF (the default): construction must succeed -- this is the path
    # every MiMo load takes even with the prefetch disabled.
    monkeypatch.setattr("vllm.envs.VLLM_B12X_MXFP4_CSF_SCALE_PREFETCH", False)
    cfg_off = MimoMxfp4CsfConfig.from_config({"checkpoint_root": root})
    assert isinstance(cfg_off.scale_prefetch_state, MimoMxfp4CsfScalePrefetch)
    assert cfg_off.scale_prefetch_state.scale_layers == {}
    assert cfg_off.scale_prefetch_state.scale_prefetch is None

    # Gate ON: same construction path, same owner contract.
    monkeypatch.setattr("vllm.envs.VLLM_B12X_MXFP4_CSF_SCALE_PREFETCH", True)
    cfg_on = MimoMxfp4CsfConfig.from_config({"checkpoint_root": root})
    assert isinstance(cfg_on.scale_prefetch_state, MimoMxfp4CsfScalePrefetch)

    # Separate config instances carry SEPARATE state: registering a layer on
    # one owner never leaks into another (two model loads, two owners).
    assert cfg_off.scale_prefetch_state is not cfg_on.scale_prefetch_state
    method = _stub_method(cfg_on.scale_prefetch_state, 0)
    cfg_on.scale_prefetch_state.scale_layers[0] = method
    assert 0 not in cfg_off.scale_prefetch_state.scale_layers
    assert cfg_off.scale_prefetch_state.scale_prefetch is None


def test_expand_returns_false_never_publishes_a_pending_prefetch(monkeypatch):
    """Codex §8.4: the (layer, token, event) triple is published ONLY after
    expand_scales() returns True; False fails closed (no event recorded, no
    pending triple) and an exception after enqueued side-stream work
    publishes an await-only dependency (token None) that can never be
    consumed but is always waited."""
    from vllm.models.deepseek_v4_1.mxfp4_csf import MimoMxfp4CsfScalePrefetch

    main = _arm_stub_cuda(monkeypatch)
    monkeypatch.setattr("vllm.envs.VLLM_B12X_MXFP4_CSF_SCALE_PREFETCH", True)
    monkeypatch.setenv("B12X_W4A16_MXFP4_PREFILL_MIN_TOKENS", "512")
    monkeypatch.delenv("B12X_W4A16_A4_PREFILL_MIN_TOKENS", raising=False)
    import b12x.moe.fused_moe as fused_moe

    state = MimoMxfp4CsfScalePrefetch()
    layers = {i: _stub_method(state, i) for i in (3, 4)}
    for method in layers.values():
        state.scale_layers[method.layer_index] = method
        method.moe_kernel = SimpleNamespace(apply=lambda **_: None)
    layer = SimpleNamespace(
        w13_weight=None,
        w2_weight=None,
        activation=None,
        global_num_experts=4,
        expert_map=None,
        apply_router_weight_on_input=False,
    )
    _step(monkeypatch, starts=[0, 8, 520], spec_tokens=7, gdn_prefills=1)
    x = torch.zeros(520, 4)

    # expand_scales -> False: nothing published, no readiness event recorded.
    monkeypatch.setattr(fused_moe, "expand_scales", lambda prepared: False)
    layers[4].scales_ready = _FakeEvent()
    layers[3].apply(layer, x, None, None, None, None)
    assert state.scale_prefetch is None, (
        "expand_scales returning False must fail closed: no pending triple"
    )
    assert layers[4].scales_ready.recorded_on is None, (
        "no readiness event may be recorded on False"
    )

    # expand_scales raises AFTER side-stream work may be enqueued: an
    # await-only dependency (token None) is preserved for later scratch
    # reuse -- waited, never consumed.
    def boom(prepared):
        raise RuntimeError("expansion failed mid-stream")

    monkeypatch.setattr(fused_moe, "expand_scales", boom)
    with pytest.raises(RuntimeError, match="expansion failed mid-stream"):
        layers[3].apply(layer, x, None, None, None, None)
    pending = state.scale_prefetch
    assert pending is not None and pending[1] is None, (
        "an exception after enqueued side-stream work must preserve an "
        "await-only dependency"
    )
    # The next call waits it (scratch race protection) but never consumes it.
    seen = []
    layers[4].backend = SimpleNamespace(scales_expanded=False)
    layers[4].moe_kernel = SimpleNamespace(
        apply=lambda **_: seen.append(layers[4].backend.scales_expanded)
    )
    _step(monkeypatch, starts=[0, 8, 520], spec_tokens=7, gdn_prefills=1)
    layers[4].apply(layer, x, None, None, None, None)
    assert pending[2] in main.waits, "the await-only dependency must be waited"
    assert seen == [False], "an await-only triple must never be consumed"
    assert state.scale_prefetch is None


def test_production_mimo_environment_arms_and_consumes_one_prefetch(monkeypatch):
    """Codex §8.2 P1 regression: with the REAL environment readers -- the
    production MiMo launcher environment (B12X_W4A16_MXFP4_PREFILL_MIN_TOKENS
    set, the NVFP4 knob B12X_W4A16_A4_PREFILL_MIN_TOKENS UNSET) -- a
    DFlash7-configured forward with real-shaped host metadata actually
    enqueues ONE prefetch and the next layer consumes it.

    The reviewed _expands() consulted the NVFP4 knob and never armed in this
    environment. Nothing eligibility-related is patched here: the real
    _expands, the real _batch_has_prefill_rows (reading real-shaped
    query_start_loc/num_actual_tokens/max_query_len metadata and the
    DFlash7 speculative config), the real _b12x_csf_prefetch_supported, and
    the real env readers.
    """
    from vllm.models.deepseek_v4_1.mxfp4_csf import (
        MimoMxfp4CsfScalePrefetch,
        _b12x_csf_prefetch_supported,
        _forward_token,
    )

    # Production MiMo launcher environment: MXFP4 knob alone.
    monkeypatch.setenv("B12X_W4A16_MXFP4_PREFILL_MIN_TOKENS", "512")
    monkeypatch.delenv("B12X_W4A16_A4_PREFILL_MIN_TOKENS", raising=False)
    main = _arm_stub_cuda(monkeypatch)
    monkeypatch.setattr("vllm.envs.VLLM_B12X_MXFP4_CSF_SCALE_PREFETCH", True)

    # The NVFP4 reader (the WRONG knob the reviewed code consulted) stays
    # False in this environment -- prove it with the real reader, unpatched.
    from vllm.model_executor.layers.fused_moe.b12x import (
        _w4a16_a4_prefill_enabled,
    )

    assert _w4a16_a4_prefill_enabled() is False, (
        "the NVFP4 A4 knob must be unset in the production MiMo environment"
    )

    state = MimoMxfp4CsfScalePrefetch()
    prepared = SimpleNamespace(_impl=SimpleNamespace(nvfp4_csf=object()))
    # The real load-time support check accepts this prepared CSF payload.
    assert _b12x_csf_prefetch_supported(prepared) is True
    layers = {i: _stub_method(state, i, prepared=prepared) for i in (3, 4)}
    for method in layers.values():
        state.scale_layers[method.layer_index] = method
        method.moe_kernel = SimpleNamespace(apply=lambda **_: None)
    # Track the real event-record ordering through the fake CUDA surface.
    layers[3].moe_done = _FakeEvent()
    layers[4].scales_ready = _FakeEvent()
    layer = SimpleNamespace(
        w13_weight=None,
        w2_weight=None,
        activation=None,
        global_num_experts=4,
        expert_map=None,
        apply_router_weight_on_input=False,
    )

    expanded = []
    ready_events = []

    def fake_expand(payload):
        expanded.append(payload)
        return True

    import b12x.moe.fused_moe as fused_moe

    monkeypatch.setattr(fused_moe, "expand_scales", fake_expand)

    # DFlash7 config (num_speculative_tokens=7 -> verification width 8) and
    # real-shaped host metadata: a decodes-first mixed batch whose last
    # request is a 512-token prefill.
    _step(monkeypatch, starts=[0, 8, 520], spec_tokens=7, gdn_prefills=1)
    token = _forward_token()
    x = torch.zeros(520, 4)

    # Layer 3 runs: it must enqueue exactly ONE prefetch for layer 4.
    layers[3].apply(layer, x, None, None, None, None)
    assert expanded == [prepared], (
        "the production MiMo environment (MXFP4 knob alone, NVFP4 unset) "
        "must arm exactly one prefetch for the following layer"
    )
    assert state.scale_prefetch == (4, token, layers[4].scales_ready), (
        "the pending triple must be (following layer, this forward's token, "
        "the following layer's readiness event)"
    )
    # The side stream waited this layer's completion, and the readiness
    # event was recorded on the side stream (real ordering, fake CUDA).
    assert layers[3].moe_done.recorded_on is main
    assert layers[4].scales_ready.recorded_on is state.scale_stream

    # Layer 4 runs in the SAME forward: it waits the pending event and
    # CONSUMES the prefetch (per-call skip flag set, cleared afterwards).
    seen = []
    layers[4].backend = SimpleNamespace(scales_expanded=False)
    layers[4].moe_kernel = SimpleNamespace(
        apply=lambda **_: seen.append(layers[4].backend.scales_expanded)
    )
    layers[4].apply(layer, x, None, None, None, None)
    assert layers[4].scales_ready in main.waits, (
        "the consumer must wait the pending readiness event"
    )
    assert seen == [True], (
        "the matching (layer, token) generation must be consumed"
    )
    assert layers[4].backend.scales_expanded is False, (
        "the skip flag must be cleared in finally"
    )
    assert state.scale_prefetch is None
