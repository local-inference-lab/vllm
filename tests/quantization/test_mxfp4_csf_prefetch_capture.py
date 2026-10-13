# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CSF scale-prefetch capture-transition integration (Codex §10.1).

GPU integration of the REAL orchestration: the actual
``MimoMxfp4CsfScalePrefetch`` owner state, real CUDA streams/events, a real
pending prefetch armed by a real ``Mxfp4CsfMoEMethod.apply()`` call, and the
REAL serving capture wrapper
``vllm.distributed.parallel_state.graph_capture`` (single-process
``init_distributed_environment`` + ``initialize_model_parallel``, world size
1 — the same TP/PP/DP group contexts the runner enters), with a real
``torch.cuda.CUDAGraph`` captured on the wrapper's dedicated stream while
``apply()`` itself runs inside the capture.

Empirical contract (pinned on this build, CUDA 12.x / torch 2.14):

* ``cudaStreamWaitEvent`` on an EXTERNALLY-RECORDED event inside a capture
  raises ``cudaErrorStreamCaptureIsolation`` (surfaced by torch as
  ``torch.AcceleratorError: CUDA error: dependency created on uncaptured
  work in another stream``) and the subsequent ``capture_end`` fails with
  ``cudaErrorStreamCaptureInvalidated`` — EVEN WHEN THE EVENT HAS ALREADY
  COMPLETED. A captured graph can never adopt an external event dependency.
* The serving capture path therefore performs a device-wide
  ``torch.accelerator.synchronize()`` BEFORE entering capture
  (``gpu_model_runner.capture_model`` / ``_warmup_and_capture``;
  ``torch.cuda.graph``'s own entry also synchronizes), so any side-stream
  expansion armed by an earlier forward is already complete when capture
  begins — there is nothing left to wait.
* The capture-entry policy is CLEAR-ONLY: ``apply()``'s capturing branch
  drops the pending (layer, token, event) triple WITHOUT waiting, and the
  captured graph runs flag-OFF/inline (each replay re-expands inline, so a
  cleared triple is never missed).

Red control: this test FAILS at the pre-§10.1 parent ``d7aacf3e5``, whose
capturing branch called ``owner.drain()`` (a ``wait_event`` inside capture)
whenever a pending triple existed at capture entry — the capture is
invalidated with ``cudaErrorStreamCaptureInvalidated``. It passes at the
§10.1 fix commit.

No model boot, small geometry; the MoE math runs through the real b12x
public path (plan/prepare/PreparationSession/bind/run) and the production
``Mxfp4CsfMoEMethod.apply()`` with only the checkpoint loader replaced by
in-memory tensors.
"""

from __future__ import annotations

import itertools
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

# Small admissible MXFP4-CSF geometry (Kimi-list): hidden % 256 == 0 and
# intermediate % 128 == 0, usable by the source-native (w31) A16 native-scale
# path under real CUDA-graph capture.
H, N, E, TOPK = 3584, 256, 4, 2
W13_ROWS, W13_COLS = 2 * N, H // 32
W2_ROWS, W2_COLS = H, N // 32

_FORWARD_TOKENS = itertools.count()


def _planes(rows, columns, *, seed):
    """Bounded e8m0 CSF slab planes (finite, nontrivial, sparse exceptions).

    Builds the checkpoint's legacy ``"csf"`` encoding — per-16-row-slab base
    bytes plus packed one-bit selector offsets, u24-position exception words
    — batched over experts exactly the way ``_load_mxfp4_csf_weights`` does,
    so ``prepare_weights`` routes through the native CSF-block preparation
    (FlashInfer 97fde4bd5/7e5a142e9) and the payload carries a real
    ``nvfp4_csf`` decoder for ``expand_scales`` to drive.
    """
    from b12x.loader._csf_source import CsfSource

    rng = np.random.default_rng(seed)
    fixed, exceptions = [], []
    for _ in range(E):
        bases = rng.integers(118, 127, rows, dtype=np.uint8)
        bits = rng.integers(0, 2, (rows, columns), dtype=np.uint8)
        logical = bases[:, None] + bits
        positions = np.unique(
            np.array(
                [
                    0,
                    63 * columns,
                    rows // 2 * columns,
                    (rows // 2 + 1) * columns,
                    rows * columns - 1,
                ],
                dtype=np.uint32,
            )
        )
        values = rng.integers(128, 136, len(positions), dtype=np.uint32)
        logical.ravel()[positions] = values
        selectors = np.packbits(bits, axis=1, bitorder="little")
        fixed.append(
            torch.from_numpy(
                np.concatenate(
                    (bases.reshape(-1, 16), selectors.reshape(rows // 16, -1)), 1
                )
            )
        )
        exceptions.append(torch.from_numpy(positions | (values << 24)))
    sizes = [int(t.numel()) for t in exceptions]
    starts = torch.zeros(len(sizes) + 1, dtype=torch.int64)
    torch.cumsum(torch.tensor(sizes, dtype=torch.int64), 0, out=starts[1:])
    return CsfSource(
        (torch.stack(fixed), torch.cat(exceptions)),
        starts,
        rows,
        columns,
        "mxfp4",
        len(sizes),
        encoding="csf",
    )


def _weights(seed=7):
    rng = np.random.default_rng(seed)
    w13 = torch.from_numpy(
        rng.integers(1, 256, (E, W13_ROWS, H // 2), dtype=np.uint8)
    ).to("cuda")
    w2 = torch.from_numpy(
        rng.integers(1, 256, (E, W2_ROWS, N // 2), dtype=np.uint8)
    ).to("cuda")
    return w13, w2


def _activations(tokens, *, seed=5):
    """Deterministic activations/routes for a token count. CUDA generators
    cannot be constructed under capture, so results are CACHED per (tokens,
    seed) and reused — exactly like the serving graph, which binds the same
    input buffers on every replay."""
    key = (tokens, seed)
    cached = _ACTIVATION_CACHE.get(key)
    if cached is not None:
        return cached
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = (torch.randn(tokens, H, device="cuda", generator=g) * 0.1).to(
        torch.bfloat16
    )
    shares = torch.tensor([1.0, 2.0, 3.0, 8.0], device="cuda")
    ids = torch.multinomial(shares.expand(tokens, -1), TOPK, generator=g)
    ids = ids.to(torch.int32).contiguous()
    weights = torch.softmax(
        torch.randn(tokens, TOPK, device="cuda", generator=g), -1
    ).contiguous()
    _ACTIVATION_CACHE[key] = (a, ids, weights)
    return a, ids, weights


_ACTIVATION_CACHE: dict = {}


class _Harness:
    """Two real ``Mxfp4CsfMoEMethod`` methods (one real owner, real streams)
    over a real prepared b12x MXFP4-CSF payload, with the production
    ``apply()`` driven end-to-end.

    Only the checkpoint loader is replaced (in-memory tensors); the method
    objects get exactly the attributes ``process_weights_after_loading``
    sets, and ``apply()`` itself — arming, consume-or-drop, skip-flag
    handling and the capture branch — is the production code under test.
    ``run()`` inside apply binds the shared prepared plan through the public
    b12x path.
    """

    def __init__(self):
        from b12x.moe import fused_moe as b12x_fused_moe
        from b12x.preparation import PreparationSession
        from b12x.preparation.types import require_prepared
        from vllm.models.deepseek_v4_1.mxfp4_csf import (
            MimoMxfp4CsfScalePrefetch,
            Mxfp4CsfMoEMethod,
        )

        self.owner = MimoMxfp4CsfScalePrefetch()
        self.scratch = (
            torch.empty((E, W13_COLS, W13_ROWS), dtype=torch.uint8, device="cuda"),
            torch.empty((E, W2_COLS, W2_ROWS), dtype=torch.uint8, device="cuda"),
        )
        fc1 = _planes(W13_ROWS, W13_COLS, seed=1000)
        fc2 = _planes(W2_ROWS, W2_COLS, seed=1500)
        w13, w2 = _weights()
        plan_w = b12x_fused_moe.plan_weights(
            source=b12x_fused_moe.PackedSource(
                format="fp4_e8m0_k32", w13_layout="w31"
            ),
            activation=b12x_fused_moe.ActivationSpec(
                mode="a16", nonlinearity="silu", io_dtype=torch.bfloat16
            ),
            geometry=b12x_fused_moe.MoEGeometry(
                num_experts=E, hidden_size=H, intermediate_size=N
            ),
            constraints=b12x_fused_moe.WeightPlanConstraints(
                required_packing=b12x_fused_moe.WeightPacking.SOURCE_NATIVE
            ),
        )
        experts = b12x_fused_moe.prepare_weights(
            plan=plan_w,
            weights=b12x_fused_moe.Mxfp4CsfWeights(
                w13=w13,
                w2=w2,
                w13_scales=fc1,
                w2_scales=fc2,
                w13_scale_scratch=self.scratch[0],
                w2_scale_scratch=self.scratch[1],
            ),
        )
        self.experts = experts
        exec_plan = b12x_fused_moe.plan_execution(
            experts=experts,
            capacity=b12x_fused_moe.ExecutionCapacity(
                max_tokens=64, top_k=TOPK
            ),
            routing=b12x_fused_moe.RoutingSpec(),
        )
        self.exec_plan = exec_plan
        self.session = PreparationSession(
            device=torch.device("cuda"), autotune=False, compile_workers=0
        )
        # allocated before session.prepare: the primer binds through it
        self.state = None
        self.bind_scratch = None
        self.session.prepare(
            (exec_plan.request(name="main", prepare_call=self._primer),)
        )
        self.state = require_prepared(exec_plan, "moe.decode")
        self.bind_scratch = tuple(
            torch.empty(s.shape, dtype=s.dtype, device=s.device)
            for s in self.state.scratch.scratch_specs()
        )

        self.methods: dict[int, Mxfp4CsfMoEMethod] = {}
        for index in (3, 4):
            method = object.__new__(Mxfp4CsfMoEMethod)
            method.owner = SimpleNamespace(
                x4t_scale_prefetch=True, scale_prefetch_state=self.owner
            )
            method.layer_index = index
            method.scale_prefetch_owner = self.owner
            method.backend = SimpleNamespace(scales_expanded=False)
            method.moe_done = torch.cuda.Event()
            method.scales_ready = torch.cuda.Event()
            method.prepared = experts
            method.moe_kernel = SimpleNamespace(apply=self._moe_apply)
            self.owner.register(method, torch.device("cuda"))
            self.methods[index] = method
        self.b12x_fused_moe = b12x_fused_moe

    # ---- production-plumbed pieces -------------------------------
    def _moe_apply(self, *, hidden_states, **_):
        """The ``moe_kernel.apply`` the production ``run()`` closure calls:
        the real public bind/run over the shared prepared plan, consulting
        the per-call consumer skip flag exactly as B12xExperts does (the
        harness wires this stub to the CALLING method's backend flag)."""
        tokens = hidden_states.shape[0]
        _, ids, weights = _activations(tokens)
        out = torch.empty(tokens, H, dtype=torch.bfloat16, device="cuda")
        method = self._consumer_method
        consumer = bool(
            method is not None and method.backend.scales_expanded
        )
        binding = self.b12x_fused_moe.bind(
            self.exec_plan,
            scratch=self.bind_scratch,
            a=hidden_states,
            topk_weights=weights,
            topk_ids=ids,
            output=out,
            input_scales_static=True,
            **({"scales_expanded": True} if consumer else {}),
        )
        self.b12x_fused_moe.run(binding=binding)
        return out

    def _primer(self, state):
        # primer-local scratch (like the b12x suite's _primer): the serving
        # bind_scratch is allocated after the session prepares.
        primer_scratch = tuple(
            torch.empty(s.shape, dtype=s.dtype, device=s.device)
            for s in state.scratch.scratch_specs()
        )
        a, ids, weights = _activations(8)
        output = torch.empty(8, H, dtype=torch.bfloat16, device="cuda")
        binding = state.bind(
            scratch=primer_scratch,
            a=a,
            topk_weights=weights,
            topk_ids=ids,
            output=output,
            input_scales_static=True,
        )
        from b12x.preparation import PreparedCall

        return PreparedCall(
            run=lambda: state.run(binding), owners=primer_scratch
        )

    # ---- driving -------------------------------------------------
    def apply_layer(self, index, x):
        """One real ``apply()`` call on the given layer's method. The stub
        moe_kernel reads the calling method's per-call skip flag (production
        B12xExperts consults the same backend attribute apply() sets)."""
        method = self.methods[index]
        self._consumer_method = method
        try:
            return method.apply(
                SimpleNamespace(
                    w13_weight=None,
                    w2_weight=None,
                    activation=None,
                    global_num_experts=E,
                    expert_map=None,
                    apply_router_weight_on_input=False,
                ),
                x,
                None,
                None,
                None,
                None,
            )
        finally:
            self._consumer_method = None

    def forward(self, x, *, prefill_rows=True):
        """A two-layer forward through real apply() calls: layer 3 runs (and,
        for a prefill-eligible batch, arms layer 4's prefetch), layer 4 runs
        (and consumes/drops the pending triple)."""
        with _forward_context(prefill_rows=prefill_rows):
            out3 = self.apply_layer(3, x)
            out4 = self.apply_layer(4, x)
        return out3, out4

    def close(self):
        self.session.__exit__(None, None, None)


class _forward_context:
    """Minimal real forward context: a per-forward token id (the production
    ``_forward_token`` contract) plus prefill-eligibility metadata shaped
    like the real host metadata (max_query_len)."""

    def __init__(self, *, prefill_rows):
        self.prefill_rows = prefill_rows

    def __enter__(self):
        from vllm import forward_context as fc

        token = next(_FORWARD_TOKENS)
        self.ctx = SimpleNamespace(
            _b12x_csf_token=token,
            attn_metadata={
                "attn": SimpleNamespace(
                    max_query_len=512 if self.prefill_rows else 1
                )
            },
        )
        self._prev_get = fc.get_forward_context
        self._prev_avail = fc.is_forward_context_available
        fc.get_forward_context = lambda: self.ctx
        fc.is_forward_context_available = lambda: True
        return self

    def __exit__(self, *exc):
        from vllm import forward_context as fc

        fc.get_forward_context = self._prev_get
        fc.is_forward_context_available = self._prev_avail
        return False


@pytest.fixture(scope="module")
def distributed():
    """Single-process distributed init + model-parallel groups so the REAL
    ``vllm.distributed.parallel_state.graph_capture`` wrapper can run (gloo,
    world size 1 — the standalone-initializable form of the serving path)."""
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29577")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    import torch.distributed as dist

    if not dist.is_initialized():
        from vllm.config import (
            DeviceConfig,
            ParallelConfig,
            VllmConfig,
            set_current_vllm_config,
        )

        cfg = VllmConfig(
            parallel_config=ParallelConfig(
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                data_parallel_size=1,
            ),
            device_config=DeviceConfig(device=torch.device("cuda:0")),
        )
        with set_current_vllm_config(cfg):
            from vllm.distributed import (
                init_distributed_environment,
                initialize_model_parallel,
            )

            init_distributed_environment(
                world_size=1, rank=0, local_rank=0, backend="gloo"
            )
            initialize_model_parallel(tensor_model_parallel_size=1)
    yield


def test_capture_entry_clears_pending_prefetch_and_captures_cleanly(
    distributed,
):
    """Codex §10.1: the real pending-prefetch transition into CUDA-graph
    capture through the REAL serving capture wrapper.

    * a real forward through ``apply()`` arms a real pending prefetch
      (side-stream expansion in flight, readiness event recorded, triple
      published);
    * the serving capture sequence runs — device synchronize, the real
      ``graph_capture`` wrapper, a real ``torch.cuda.CUDAGraph`` captured
      around a real ``apply()`` call — and must SUCCEED (the pre-§10.1
      code called ``owner.drain()`` inside the capture branch and died with
      ``cudaErrorStreamCaptureInvalidated``);
    * the pending triple is CLEARED at capture entry (clear-not-wait);
    * the captured graph replays BIT-EXACT twice over poisoned scratch;
    * capture entry after an INTERRUPTED forward that left a STALE pending
      triple also captures cleanly and clears it;
    * the owner re-arms normally after the capture cycle.
    """
    harness = _Harness()
    try:
        from vllm.distributed import graph_capture
        from vllm.model_executor.layers.fused_moe.b12x import (
            _is_current_stream_capturing,
        )

        x, _, _ = _activations(32)
        # eager reference outputs (inline path), for replay equality. A FULL
        # two-layer forward arms layer 4's prefetch in layer 3 and CONSUMES
        # it in layer 4 — the normal production cycle, bit-exact.
        ref3, ref4 = harness.forward(x)
        torch.cuda.synchronize()
        ref3, ref4 = ref3.clone(), ref4.clone()

        # ---- (1) a PARTIALLY-EXECUTED forward leaves a real pending
        # prefetch: layer 3 ran (and armed layer 4's side-stream expansion),
        # layer 4 never ran — e.g. the scheduler dropped the batch after the
        # producer. The triple is pending at capture entry.
        harness.owner.scale_prefetch = None
        with _forward_context(prefill_rows=True):
            out3 = harness.apply_layer(3, x)
        torch.testing.assert_close(out3, ref3, rtol=0, atol=0)
        pending = harness.owner.scale_prefetch
        assert pending is not None, (
            "a partially-executed forward must leave a pending prefetch "
            "armed by the real apply() for layer 4"
        )
        assert pending[0] == 4, "the armed triple targets the following layer"

        # ---- (2) the serving capture sequence: device-wide synchronize
        # BEFORE capture (gpu_model_runner.capture_model does exactly this),
        # then the REAL graph_capture wrapper + torch.cuda.graph around a
        # real apply() call of the layer the pending triple targeted.
        torch.cuda.synchronize()
        device = torch.device("cuda:0")
        graph = torch.cuda.CUDAGraph()
        with graph_capture(device=device):
            assert torch.cuda.current_stream() != torch.cuda.default_stream(), (
                "the wrapper must capture on a dedicated stream"
            )
            with torch.cuda.graph(graph):
                assert _is_current_stream_capturing(), (
                    "apply()'s capture branch must engage inside the graph"
                )
                captured = harness.apply_layer(4, x)

        # ---- (3) the transition contract
        assert harness.owner.scale_prefetch is None, (
            "capture entry must CLEAR the pending triple (clear-not-wait, "
            "Codex §10.1) — a captured graph can never consume it and must "
            "not adopt its external event"
        )
        # two bit-exact replays over poisoned scratch
        for buf in harness.scratch:
            buf.fill_(0xD6)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, ref4, rtol=0, atol=0)
        for buf in harness.scratch:
            buf.fill_(0xD6)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, ref4, rtol=0, atol=0)
        del graph

        # ---- (4) the Codex-required INTERRUPTED-forward variant: the
        # forward dies AFTER arming (stale generation — its forward token
        # can never match a live forward again), leaving the triple stale.
        # Capture entry must clear it and capture cleanly.
        with _forward_context(prefill_rows=True):
            harness.apply_layer(3, x)  # arms for layer 4 ...
        # ... and the forward is interrupted here: layer 4 never runs, the
        # armed token is already dead (the next forward gets a new token).
        assert harness.owner.scale_prefetch is not None
        torch.cuda.synchronize()  # the runner's pre-capture device sync
        graph2 = torch.cuda.CUDAGraph()
        with graph_capture(device=device):
            with torch.cuda.graph(graph2):
                captured2 = harness.apply_layer(4, x)
        assert harness.owner.scale_prefetch is None, (
            "capture entry after an interrupted forward must CLEAR the stale "
            "pending triple"
        )
        graph2.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured2, ref4, rtol=0, atol=0)
        del graph2

        # ---- (5) the owner re-arms and consumes normally after the capture
        # cycle: a full forward arms in layer 3 and consumes in layer 4,
        # bit-exact with the reference.
        out3, out4 = harness.forward(x)
        torch.testing.assert_close(out3, ref3, rtol=0, atol=0)
        torch.testing.assert_close(out4, ref4, rtol=0, atol=0)
        assert harness.owner.scale_prefetch is None
    finally:
        harness.close()
