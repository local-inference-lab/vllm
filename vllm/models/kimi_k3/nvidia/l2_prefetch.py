# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cache immutable dense weights during Kimi's tensor-parallel reductions.

The CuTeDSL cache-hint primitive is shared with GLM. Kimi supplies separate
weight ranges and budgets; no activation, weight, or reduction is modified.
Plans are rebuilt after weight loading and remain fixed through graph replay.
"""

from vllm.logger import init_logger
from vllm.models.glm5next.nvidia import l2_prefetch as cache_hints

logger = init_logger(__name__)


class KimiDecodePrefetch:
    """Bounded cache hints before attention-output and MoE all-reduces."""

    def __init__(self, layers, device):
        self.prefetcher = cache_hints.L2Prefetcher(device, persisting_l2_request="0")
        self.plans = []
        self.hooks = []
        # CUDA graphs contain raw addresses. Retain the corresponding tensors
        # until this owner and its captured graphs have been released.
        self.weights = tuple(p for layer in layers for p in layer.parameters())
        for index, layer in enumerate(layers):
            moe = layer.mlp
            segments = []
            paired = getattr(moe, "_paired_decode_weight", None)
            if paired is not None:
                segments.append(
                    cache_hints.tensor_segment("paired_down_router", paired)
                )
                self.weights += (paired,)
            else:
                for name in ("gate", "routed_expert_down_proj"):
                    projection = getattr(moe, name, None)
                    if projection is not None:
                        segments.extend(cache_hints.segments_of(projection, name + "."))
            shared = getattr(moe, "shared_experts", None)
            if shared is not None:
                segments.extend(
                    cache_hints.segments_of(shared, "shared.", include_attrs=False)
                )
            self._attach(
                getattr(layer.self_attn, "o_proj", None), segments, 24 * 1024**2, device
            )
            if index + 1 == len(layers):
                continue
            attention = layers[index + 1].self_attn
            segments = []
            for name in (
                "in_proj_qkvgfab",
                "in_proj_qkv",
                "in_proj_gfab",
                "fused_qkv_a_g_proj",
                "fused_qkv_a_proj",
                "q_a_proj",
                "q_proj",
                "kv_a_proj_with_mqa",
            ):
                projection = getattr(attention, name, None)
                if projection is not None:
                    segments.extend(
                        cache_hints.segments_of(
                            projection, name + ".", include_attrs=False
                        )
                    )
            experts = getattr(moe, "experts", None)
            target = getattr(experts, "runner", experts)
            if target is None:
                target = getattr(moe, "down_proj", None)
            self._attach(target, segments, 48 * 1024**2, device)

    def _attach(self, owner, segments, budget, device):
        if owner is None:
            return
        plan, _ = cache_hints.make_plan(
            [s for s in segments if s is not None], budget, device
        )
        if plan is None:
            return
        if getattr(owner, "_l2_prefetch_pre_reduce_hook", None) is not None:
            raise ValueError("Kimi cache hints require an unclaimed pre-reduction hook")
        self.plans.append(plan)

        def issue(rows):
            if rows <= 8:
                self.prefetcher.issue(plan, rows)

        owner._l2_prefetch_pre_reduce_hook = issue
        self.hooks.append((owner, issue))

    def join(self):
        self.prefetcher.join()

    def close(self):
        self.join()
        for owner, callback in self.hooks:
            if getattr(owner, "_l2_prefetch_pre_reduce_hook", None) is callback:
                del owner._l2_prefetch_pre_reduce_hook
        self.hooks.clear()


def prepare_decode_prefetch(model):
    """Create ranges only after all linear weights have their serving layout."""
    from vllm import envs
    from vllm.platforms import current_platform

    previous = getattr(model, "_decode_prefetch", None)
    if previous is not None:
        previous.close()
    model._decode_prefetch = None
    if not envs.VLLM_KIMI_L2_PREFETCH:
        return
    if not current_platform.is_device_capability(120) or model.use_sequence_parallel:
        raise ValueError("Kimi L2 prefetch requires SM120 without sequence parallelism")
    if cache_hints._get_launcher() is None:
        raise RuntimeError("Kimi L2 prefetch could not prepare its CuTeDSL kernel")
    layers = list(model.layers[model.start_layer : model.end_layer])
    device = next(model.parameters()).device
    model._decode_prefetch = KimiDecodePrefetch(layers, device)
    logger.info_once(
        "Kimi dense-weight L2 prefetch: %d bounded plans, at most 8 rows",
        len(model._decode_prefetch.plans),
    )
