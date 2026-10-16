# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Losslessly compressed DS4.1 routed experts with native dense tensors."""

import itertools

import regex as re
import torch

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.b12x import (
    B12xExperts,
    _is_current_stream_capturing,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    FusedMoEQuantDesc,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.utils.torch_utils import set_default_torch_num_threads

from .quant_config import DeepseekV41FP8Config

logger = init_logger(__name__)

# Per-forward token counter (shared with nvfp4_csf.py via the same forward
# context attribute; see _forward_token).
_FORWARD_TOKENS = itertools.count()


def _spec_decode_query_width() -> int:
    """Rows per request in a uniform decode/verification batch.

    ``1 + num_speculative_tokens`` (the runner's ``uniform_decode_query_len``):
    MTP=3 gives 4, DFlash=7 gives 8. Queries at or below this width are
    decode or speculative-verification rows, not prefill rows.
    """
    from vllm.config import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    spec = getattr(config, "speculative_config", None) if config is not None else None
    return 1 + int(getattr(spec, "num_speculative_tokens", 0) or 0)


def _batch_has_prefill_rows() -> bool:
    """Whether the current batch contains actual prefill rows (Codex 9.5).

    Row-type eligibility, never a total-row threshold: a batch qualifies only
    when some request's query exceeds the decode/verification width
    (``1 + num_speculative_tokens``, taken from the speculative config).
    DFlash verification batches (uniform 1+spec-token queries) and pure decode
    batches never qualify, whatever their row count. Host-side metadata only
    (``max_query_len``): no device read, no synchronization. Missing metadata
    means no arming.

    Codex §10.2.2 correction: the classification width is derived from the
    SPECULATIVE CONFIG ONLY (``_spec_decode_query_width``). Metadata
    ``num_spec_decode_tokens`` is deliberately NOT consulted: in the real
    builder (``gdn_attn.py:555``) it is the AGGREGATE speculative token
    count for the whole batch (``query_lens.sum() - prefill_tokens -
    decode_tokens``), not a per-request width. Refining the width with that
    aggregate lets many verification requests inflate it past a shorter REAL
    prefill query and suppress arming for a batch that does contain prefill
    rows. Production MiMo's ``B12xPagedMetadata`` has no
    ``num_spec_decode_tokens`` field at all, so this correction cannot change
    MiMo behavior -- it fixes the semantics for backends (GDN/KDA) that do
    report it.
    """
    from vllm.forward_context import (
        get_forward_context,
        is_forward_context_available,
    )

    if not is_forward_context_available():
        return False
    metadata = getattr(get_forward_context(), "attn_metadata", None)
    if metadata is None:
        return False
    values: list = []
    if isinstance(metadata, dict):
        values = list(metadata.values())
    elif isinstance(metadata, (list, tuple)):
        for entry in metadata:
            if isinstance(entry, dict):
                values.extend(entry.values())
    width = _spec_decode_query_width()
    return any(
        isinstance(getattr(value, "max_query_len", None), int)
        and int(value.max_query_len) > width
        for value in values
    )


def _b12x_csf_prefetch_supported(prepared) -> bool:
    """Whether b12x exposes prepared CSF scale expansion + consumer skip.

    Requires both halves of the coordinated contract: ``expand_scales`` must
    handle this prepared payload (the ``nvfp4_csf`` decoder branch installed
    by the FlashInfer CSF-block preparation — 97fde4bd5/7e5a142e9 made it
    carry BOTH the NVFP4 and the MXFP4 compact-scale payloads) and ``bind``
    must accept ``scales_expanded`` (the per-call consumer skip). Older b12x
    builds have neither; arming there would duplicate work (a side-stream
    expansion with the per-call decode still running), so prefetch stays off.
    Fails closed on any missing piece.
    """
    if getattr(getattr(prepared, "_impl", None), "nvfp4_csf", None) is None:
        return False
    try:
        import inspect

        from b12x.moe import fused_moe
        from b12x.moe.fused_moe._impl import TPMoEScratchPlan

        if not hasattr(fused_moe, "expand_scales"):
            return False
        return "scales_expanded" in inspect.signature(
            TPMoEScratchPlan.bind
        ).parameters
    except Exception:
        return False


class MimoMxfp4CsfScalePrefetch:
    """Owner-carried MXFP4/CSF scale-prefetch state for one MiMo expert owner.

    Mirrors ``Nvfp4CsfConfig``'s scale_layers/scale_stream/scale_prefetch +
    moe_done/scales_ready pattern (nvfp4_csf.py), with Codex §9.3 corrections
    for the shared per-rank scale scratch:

    * The overlap window is layer L+1's ATTENTION: after L's MoE completes
      (``moe_done``), the side stream expands L+1's scales into the shared
      scratch while L+1's attention runs; L+1's MoE consumes them.
    * A pending prefetch is always awaited before the consumer runs OR before
      any inline expansion overwrites the same scratch — a stale/mismatched
      generation still needs that wait (§9.3 point 3/4).
    * The per-call consumer skip flag is set only for the matching
      (layer, forward-token) and cleared in ``finally``.
    * CUDA-graph capture: new prefetch is disabled and any pending triple is
      CLEARED at capture entry without waiting (Codex §10.1) — the serving
      runner device-synchronizes before capture, and adopting an external
      event dependency inside a capture raises
      cudaErrorStreamCaptureIsolation even for a completed event; the
      captured graph runs inline (flag never declared under capture).
    * Eligibility is per-row-type (``_batch_has_prefill_rows``): DFlash
      verification and pure-decode batches never arm (Codex 9.5).

    2026-10-12 port (csfdelta): the mechanism now targets the NATIVE CSF
    scale machinery the FlashInfer 97fde4bd5/7e5a142e9 bases installed in
    place of the old X4T paired-program surface: an MXFP4-CSF prepared
    payload carries an ``nvfp4_csf`` decoder whose ``decode_all`` is what
    ``b12x_fused_moe.expand_scales`` drives on the side stream, and whose
    per-call selective decode is what ``bind(..., scales_expanded=True)``
    (the stock ``B12xExperts.scales_expanded`` flag) skips. The class and
    env names keep their X4T-era spelling for launcher compatibility; the
    owner capability attribute ``x4t_scale_prefetch`` is unchanged.

    PP1, no ubatching, no concurrent forwards sharing the owner (enforced by
    the shared method's PP1/no-ubatching requirement). IDs/events/streams are
    allocated at preparation; nothing is allocated per request.
    """

    def __init__(self):
        self.scale_layers: dict[int, "Mxfp4CsfMoEMethod"] = {}
        self.scale_stream: torch.cuda.Stream | None = None
        # (destination layer index, forward token, ready event)
        self.scale_prefetch: tuple[int, int, torch.cuda.Event] | None = None

    def register(self, method: "Mxfp4CsfMoEMethod", device: torch.device) -> None:
        self.scale_layers[method.layer_index] = method
        if self.scale_stream is None:
            self.scale_stream = torch.cuda.Stream(device)

    def drain(self) -> None:
        """Wait out any in-flight side-stream expansion OUTSIDE capture only.

        Codex §10.1: this is the EAGER-transition helper (a caller that is NOT
        capturing and wants a device-ordered join). The capture transition no
        longer calls it: the serving runner synchronizes the device before
        capture begins, and waiting an externally-recorded event INSIDE a
        capture raises cudaErrorStreamCaptureIsolation (invalidating the
        capture) even for a completed event — ``apply()``'s capture branch
        therefore clears the pending triple WITHOUT waiting.
        """
        pending = self.scale_prefetch
        self.scale_prefetch = None
        if pending is not None:
            torch.cuda.current_stream().wait_event(pending[2])


def _forward_token() -> int | None:
    """An id of the current forward pass, or None outside of one.

    Duplicated from nvfp4_csf.py's tiny helper with the same context attribute
    (``_b12x_csf_token``), so the MXFP4 and NVFP4 prefetch mechanisms share one
    per-forward identity. The import is avoided deliberately: nvfp4_csf imports
    the modelopt graph, and this module is imported by the shared CSF config.
    """
    from vllm.forward_context import (
        get_forward_context,
        is_forward_context_available,
    )

    if not is_forward_context_available():
        return None
    context = get_forward_context()
    token = getattr(context, "_b12x_csf_token", None)
    if token is None:
        token = next(_FORWARD_TOKENS)
        context._b12x_csf_token = token
    return token


class DeepseekV41Mxfp4CsfConfig(DeepseekV41FP8Config):
    """MXFP4-CSF main experts; unchanged native dense, shared and draft weights."""

    checkpoint_root: str
    scale_scratch: tuple[torch.Tensor, ...] | None

    @classmethod
    def get_name(cls):
        return "mxfp4_csf"

    @classmethod
    def get_min_capability(cls):
        return 120

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        if (
            user_quant in (None, cls.get_name())
            and hf_quant_cfg is not None
            and hf_quant_cfg.get("quant_method") == cls.get_name()
            and getattr(hf_config, "model_type", None)
            in ("deepseek_v41", "deepseek_v41_text")
        ):
            return cls.get_name()
        return None

    @classmethod
    def from_config(cls, config):
        if config.get("format_version") != 1 or not config.get("checkpoint_root"):
            raise ValueError("MXFP4-CSF requires format_version=1 and checkpoint_root")
        result = cls(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[32, 32],
        )
        result.checkpoint_root = config["checkpoint_root"]
        result.scale_scratch = None
        return result

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, RoutedExperts):
            match = re.search(r"(?:^|\.)layers\.(\d+)\.", prefix)
            if match is None:
                raise ValueError("DS4.1 routed experts require a numbered layer")
            config = get_current_vllm_config()
            if int(match.group(1)) < config.model_config.hf_config.num_hidden_layers:
                if (
                    config.parallel_config.pipeline_parallel_size != 1
                    or config.parallel_config.use_ubatching
                ):
                    raise NotImplementedError(
                        "MXFP4-CSF shared scale scratch requires PP1 without ubatching"
                    )
                if config.load_config.load_format not in ("mxfp4_csf",):
                    raise ValueError("MXFP4-CSF requires --load-format mxfp4_csf ")
                return Mxfp4CsfMoEMethod(
                    layer.moe_config,
                    self,
                    activation_mode="a16" if envs.VLLM_B12X_MOE_FP4_FORCE_A16 else "a8",
                )
        return super().get_quant_method(layer, prefix)


class Mxfp4CsfMoEMethod(FusedMoEMethodBase):
    """Expand scales for routed experts into serialized, model-owned scratch."""

    def __init__(self, moe, owner, *, activation_mode="a16"):
        super().__init__(moe)
        self.owner = owner
        if activation_mode not in ("a16", "a8"):
            raise ValueError("MXFP4-CSF requires A16 or MXFP8 activations")
        self.activation_mode = activation_mode
        # Set at weight-load time when the owner arms the CSF scale prefetch.
        self.scale_prefetch_owner = None
        parallel = moe.moe_parallel_config
        if (
            parallel.use_ep
            or parallel.ep_size != 1
            or parallel.dp_size != 1
            or parallel.use_all2all_kernels
            or parallel.enable_eplb
        ):
            raise NotImplementedError("MXFP4-CSF experts support TP without EP/DP")
        if (
            moe.activation not in (MoEActivation.SILU, MoEActivation.SITU)
            or moe.in_dtype != torch.bfloat16
            or moe.has_bias
            or (
                moe.activation == MoEActivation.SITU
                and (
                    moe.activation_situ_beta != 4.0
                    or moe.activation_situ_linear_beta != 25.0
                )
            )
        ):
            raise ValueError(
                "MXFP4-CSF requires bias-free BF16 SwiGLU or SiTU(4,25) experts"
            )

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        match = re.search(r"(?:^|\.)layers\.(\d+)\.", layer.layer_name)
        if match is None or params_dtype != torch.bfloat16:
            raise ValueError("MXFP4-CSF requires a numbered BF16 expert layer")
        self.layer_index = int(match.group(1))
        self.num_experts, self.hidden_size = num_experts, hidden_size
        self.local_intermediate = intermediate_size_per_partition
        for name in ("w13_weight", "w2_weight"):
            layer.register_buffer(
                name, torch.empty(0, dtype=torch.uint8), persistent=False
            )

    def get_fused_moe_quant_config(self, layer):
        activation = "mxfp8" if self.activation_mode == "a8" else None
        return FusedMoEQuantConfig(
            _a1=FusedMoEQuantDesc(dtype=activation),
            _a2=FusedMoEQuantDesc(dtype=activation),
            _w1=FusedMoEQuantDesc(dtype="mxfp4"),
            _w2=FusedMoEQuantDesc(dtype="mxfp4"),
        )

    def process_weights_after_loading(self, layer):
        from b12x.moe import fused_moe

        from vllm.model_executor.model_loader.mxfp4_csf_loader import (
            read_mxfp4_csf_layer,
        )

        tp, rank = (
            get_tensor_model_parallel_world_size(),
            get_tensor_model_parallel_rank(),
        )
        device = layer.w13_weight.device
        e, h, n = self.num_experts, self.hidden_size, self.local_intermediate
        shapes = ((e, h // 32, 2 * n), (e, n // 32, h))
        if self.owner.scale_scratch is None:
            # MXFP4-CSF disallows ubatching: every layer consumes these scale grids
            # before its successor may overwrite them on the same stream.
            self.owner.scale_scratch = tuple(
                torch.empty(s, dtype=torch.uint8, device=device) for s in shapes
            )
        scratch = self.owner.scale_scratch
        if any(
            t.shape != shape or t.device != device for t, shape in zip(scratch, shapes)
        ):
            raise ValueError("MXFP4-CSF shared scale scratch geometry/device mismatch")
        # Per-expert CPU slices are too small to amortize intra-op barriers.
        # Restore the serving thread policy before kernel preparation.
        with set_default_torch_num_threads(1):
            weights = read_mxfp4_csf_layer(
                self.owner.checkpoint_root,
                self.layer_index,
                num_experts=e,
                hidden_size=h,
                intermediate_size=n * tp,
                tp_rank=rank,
                tp_size=tp,
                device=device,
                w13_scale_scratch=scratch[0],
                w2_scale_scratch=scratch[1],
            )
        plan = fused_moe.plan_weights(
            source=fused_moe.PackedSource(format="fp4_e8m0_k32", w13_layout="w31"),
            activation=fused_moe.ActivationSpec(
                mode=self.activation_mode,
                nonlinearity="situ"
                if self.moe.activation == MoEActivation.SITU
                else "silu",
                io_dtype=torch.bfloat16,
                swiglu_limit=self.moe.swiglu_limit,
            ),
            geometry=fused_moe.MoEGeometry(
                num_experts=e, hidden_size=h, intermediate_size=n
            ),
        )
        prepared = fused_moe.prepare_weights(plan=plan, weights=weights)
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        backend = B12xExperts(self.moe, self.moe_quant_config)
        backend.install_prepared_experts(layer, prepared)
        self.moe_kernel = mk.FusedMoEKernel(
            MoEPrepareAndFinalizeNoDPEPModular(), backend
        )
        self.backend, self.prepared = backend, prepared
        # MiMo-owned CSF scale prefetch (Codex §9.2/§9.3): default OFF, owner
        # capability attribute only (never an isinstance check against the
        # MiMo config, which would be a circular import for non-MiMo owners).
        # Codex review fix: the two CUDA events exist only for an ARMED layer
        # -- with the gate OFF nothing is allocated (gate-OFF cost is zero,
        # not "small").
        self.scale_prefetch_owner = None
        if (
            getattr(self.owner, "x4t_scale_prefetch", False)
            and self.activation_mode == "a16"
            and envs.VLLM_B12X_MXFP4_CSF_SCALE_PREFETCH
            and _b12x_csf_prefetch_supported(prepared)
        ):
            state = getattr(self.owner, "scale_prefetch_state", None)
            if state is not None:
                self.moe_done = torch.cuda.Event()
                self.scales_ready = torch.cuda.Event()
                self.scale_prefetch_owner = state
                state.register(self, device)
        logger.info(
            "MXFP4-CSF lossless MXFP4 layer %d rank %d/%d: %s activations, "
            "compressed scales, shared scratch %d bytes, csf-scale-prefetch=%s",
            self.layer_index,
            rank,
            tp,
            "MXFP8" if self.activation_mode == "a8" else "BF16",
            sum(t.numel() for t in scratch),
            self.scale_prefetch_owner is not None,
        )

    def apply(
        self,
        layer,
        x,
        topk_weights,
        topk_ids,
        shared_experts,
        shared_experts_input,
        workspace=None,
    ):
        assert self.moe_kernel is not None

        def run():
            return self.moe_kernel.apply(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=layer.activation,
                global_num_experts=layer.global_num_experts,
                expert_map=layer.expert_map,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                shared_experts=shared_experts,
                shared_experts_input=shared_experts_input,
                workspace=workspace,
            )

        owner = self.scale_prefetch_owner
        if owner is None:
            return run()
        stream = torch.cuda.current_stream()
        if _is_current_stream_capturing():
            # Codex §10.1: capture-entry CLEAR-ONLY, never a wait. The real
            # serving capture path (gpu_model_runner.capture_model /
            # _warmup_and_capture) performs a device-wide
            # torch.accelerator.synchronize() BEFORE entering capture (and
            # torch.cuda.graph's own entry does torch.cuda.synchronize()), so
            # any side-stream expansion armed by an earlier forward is already
            # complete when capture begins: there is nothing left to wait.
            # Waiting inside capture is not merely unnecessary, it is ILLEGAL:
            # cudaStreamWaitEvent on an externally-recorded event inside a
            # capture raises cudaErrorStreamCaptureIsolation and invalidates
            # the capture -- even when the event has already completed (pinned
            # empirically on this build; the captured graph must never adopt
            # an external event dependency). So the transition drops the
            # pending triple without touching the event. The captured graph
            # runs flag-OFF/inline anyway (new arming is disabled under
            # capture below and the flag is never declared inside capture),
            # so the cleared triple cannot be missed: each replay re-expands
            # inline exactly as if no prefetch had ever been armed.
            owner.scale_prefetch = None
            return run()
        token = _forward_token()
        # Codex §9.3 point 3: consume-or-drop the pending prefetch, but always
        # WAIT it before this call: whether or not the generation matches, the
        # side-stream expansion may be in flight over the same shared scratch
        # this call's inline expansion would overwrite.
        pending, owner.scale_prefetch = owner.scale_prefetch, None
        if pending is not None:
            stream.wait_event(pending[2])
        self.backend.scales_expanded = (
            token is not None
            and pending is not None
            and (pending[:2] == (self.layer_index, token))
        )
        try:
            result = run()
        finally:
            self.backend.scales_expanded = False
        following = owner.scale_layers.get(self.layer_index + 1)
        if (
            token is not None
            and following is not None
            and self._expands(x)
            and _batch_has_prefill_rows()
        ):
            # The next layer's MoE would expand its scales into the same
            # scratch: do it now on a side stream, overlapping that layer's
            # attention (NOT this layer's: it is already done).
            from b12x.moe import fused_moe as b12x_fused_moe

            self.moe_done.record(stream)
            side = owner.scale_stream
            side.wait_event(self.moe_done)
            with torch.cuda.stream(side):
                # Codex review fix: publish the (layer, token, event) triple
                # ONLY after expand_scales() returns True. Its Boolean is the
                # readiness contract -- fail closed on False (nothing was
                # enqueued; no event, no pending triple) and, if it RAISES
                # after side-stream work may already be enqueued over the
                # shared scratch, preserve a dependency that later scratch
                # reuse must wait: an await-only triple whose token is None
                # can never match a live forward (consumption requires a
                # real forward token), so it is drained/waited and dropped,
                # never consumed.
                try:
                    expanded = bool(
                        b12x_fused_moe.expand_scales(following.prepared)
                    )
                except Exception:
                    following.scales_ready.record(side)
                    owner.scale_prefetch = (
                        following.layer_index,
                        None,
                        following.scales_ready,
                    )
                    raise
                if expanded:
                    following.scales_ready.record(side)
                    owner.scale_prefetch = (
                        following.layer_index,
                        token,
                        following.scales_ready,
                    )
        return result

    def _expands(self, x: torch.Tensor) -> bool:
        """Whether this call's consumers decode CSF scales into the scratch.

        Codex review fix (P1) carried over from the X4T design: eligibility
        is derived from the PREPARED CSF PAYLOAD, never from the NVFP4 knob.
        History: the reviewed commit consulted
        ``_w4a16_a4_prefill_enabled()`` (``B12X_W4A16_A4_PREFILL_MIN_TOKENS``,
        an NVFP4 switch) while the MiMo launcher sets
        ``B12X_W4A16_MXFP4_PREFILL_MIN_TOKENS`` and leaves the NVFP4 knob
        unset, so production MiMo never armed. On the 97fde4bd5-era bases
        every consumer of a CSF payload consumes the shared native scale
        grids: the per-call selective decode at launch fills them unless
        ``bind(..., scales_expanded=True)`` skips it, and ALL routed
        consumers -- the A16 runners (small-M direct arm, packed/direct
        route) and the MXFP4 A4 prefill runner -- read the grids, so a
        prepared payload retaining the ``nvfp4_csf`` decoder IS the
        eligibility, and the knobs only pick WHICH consumer runs.

        Codex §9.5 still holds: this says the consumer decodes into the
        shared scratch, not that this call should arm -- the caller
        separately requires actual prefill rows (``_batch_has_prefill_rows``)
        and CUDA-graph capture excludes arming in ``apply``. Row
        classification uses the host metadata's verification width
        (``_spec_decode_query_width``), never
        ``_num_leading_decode_tokens`` (that helper assumes query width <= 4
        and reads ``query_start_loc.tolist()``; it is not a DFlash7
        classifier and is not used here).
        """
        prepared = getattr(self, "prepared", None)
        decoder = getattr(getattr(prepared, "_impl", None), "nvfp4_csf", None)
        return decoder is not None
