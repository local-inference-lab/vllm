# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""b12x expert residency for routed MXFP4 experts on SM103.

Expert residency ("residency" is a working name) keeps each layer's frequently
routed experts in HBM and serves the rest from coherent Grace memory, which
the GPU reads directly. One prepared b12x operator serves both tiers, so a
checkpoint whose experts exceed HBM can run on one GPU.

Loading stages the routed expert tensors in host memory. Before the weights
preparation stage, :func:`declare_expert_residency` places every layer of the
target and draft models within one model-wide HBM/Grace budget and gives each
layer its b12x plan. Draft layers stay entirely in HBM. Placement is static:
a validated b12x profile, or a balanced placement derived from the budget.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.model import ModelConfig
    from vllm.utils.b12x import B12xWorkload

logger = init_logger(__name__)


def expert_residency_enabled() -> bool:
    """Whether the engine being constructed serves experts through residency."""
    from vllm.config import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    return config is not None and config.expert_residency_config is not None


@dataclass
class ResidencyDeclaration:
    """One layer's source-native weight contract, staged in host memory."""

    layer_name: str
    weight_plan: Any
    w13: torch.Tensor
    w2: torch.Tensor
    w13_block_scales: torch.Tensor
    w2_block_scales: torch.Tensor
    top_k: int
    plan: Any = None
    max_tokens: int = 0

    def packed_weights(self, fused_moe: Any, fingerprint: str) -> Any:
        experts = int(self.w13.shape[0])
        unit = torch.ones(experts, dtype=torch.float32)
        return fused_moe.PackedWeights(
            w13=self.w13,
            w2=self.w2,
            w13_block_scales=self.w13_block_scales,
            w2_block_scales=self.w2_block_scales,
            w13_global_scales=unit,
            w2_global_scales=unit,
            immutable_input_scales=True,
            checkpoint_fingerprint=fingerprint,
            layer_name=self.layer_name,
        )


def checkpoint_fingerprint(model_config: ModelConfig) -> str:
    """Identify the local checkpoint by file names, sizes and safetensors headers.

    Headers fix every tensor's name, dtype, shape and byte range; placement
    profiles are rejected when any of them changes. Tensor bytes are not
    hashed.
    """
    path = Path(model_config.model)
    files = sorted(path.glob("*.safetensors")) if path.is_dir() else []
    if not files:
        raise ValueError(
            "expert residency requires a local safetensors checkpoint directory; "
            f"{model_config.model!r} has none"
        )
    digest = hashlib.sha256()
    digest.update(str(model_config.revision or "").encode())
    for file in files:
        with file.open("rb") as handle:
            size = int.from_bytes(handle.read(8), "little")
            header = handle.read(size)
        digest.update(file.name.encode())
        digest.update(file.stat().st_size.to_bytes(8, "little"))
        digest.update(header)
    return "sha256:" + digest.hexdigest()


def _residency_providers(model: torch.nn.Module | None) -> list[tuple[Any, Any]]:
    """Every b12x MoE provider in ``model`` that declared host-staged experts."""
    if model is None:
        return []
    found = []
    for module in model.modules():
        provider = getattr(module, "b12x_preparation_provider", None)
        declaration = getattr(provider, "residency_declaration", None)
        if declaration is not None:
            found.append((provider, declaration))
    return found


def _capacity(workload: B12xWorkload) -> int:
    return max(workload.max_tokens, *workload.fixed_token_counts)


def declare_expert_residency(
    vllm_config: VllmConfig,
    *,
    model: torch.nn.Module,
    draft: torch.nn.Module | None,
    workload: B12xWorkload,
    draft_workload: B12xWorkload | None,
    device: torch.device,
) -> None:
    """Place every declared layer and install its b12x residency plan.

    The model-wide budget reserves the configured KV cache and HBM headroom
    once. Draft layers are placed entirely in HBM; target layers share the
    remaining HBM and the Grace envelope. Runs outside CUDA graph capture,
    before the weights preparation stage.
    """
    from b12x.moe import fused_moe

    config = vllm_config.expert_residency_config
    assert config is not None
    target = _residency_providers(model)
    drafts = _residency_providers(draft)
    if not target:
        raise ValueError(
            "expert residency is enabled, but the model declared no routed "
            "MXFP4 experts for it"
        )
    if all(declaration.plan is not None for _, declaration in (*target, *drafts)):
        return

    fingerprint = checkpoint_fingerprint(vllm_config.model_config)
    lanes: list[tuple[Any, Any, int, dict]] = [
        (provider, declaration, _capacity(workload), {})
        for provider, declaration in target
    ]
    if drafts:
        assert draft_workload is not None
        lanes.extend(
            (
                provider,
                declaration,
                _capacity(draft_workload),
                {
                    "minimum_hot": int(declaration.w13.shape[0]),
                    "maximum_hot": int(declaration.w13.shape[0]),
                },
            )
            for provider, declaration in drafts
        )
    bounds = {
        name: value
        for name, value in (
            ("minimum_hot", config.min_hot_experts),
            ("maximum_hot", config.max_hot_experts),
        )
        if value is not None
    }
    specs = []
    for _, declaration, capacity, fixed in lanes:
        experts = int(declaration.w13.shape[0])
        layer_bounds = fixed or {
            name: min(value, experts) for name, value in bounds.items()
        }
        declaration.max_tokens = capacity
        specs.append(
            fused_moe.ResidencyLayerSpec.from_weight_plan(
                layer=declaration.layer_name,
                weight_plan=declaration.weight_plan,
                capacity=fused_moe.ExecutionCapacity(
                    max_tokens=capacity, top_k=declaration.top_k
                ),
                **layer_bounds,
            )
        )
    model_config = vllm_config.model_config
    model_spec = fused_moe.ResidencyModelSpec(
        checkpoint_fingerprint=fingerprint,
        model_revision=str(model_config.revision or "unspecified"),
        tokenizer_revision=str(model_config.tokenizer_revision or "unspecified"),
        layers=tuple(specs),
    )
    cache_config = vllm_config.cache_config
    budget = fused_moe.ModelExpertMemoryBudget.from_available(
        device=device,
        grace_bytes=config.grace_bytes,
        kv_reserved_bytes=int(cache_config.kv_cache_memory_bytes or 0),
        hbm_safety_bytes=config.hbm_reserve_bytes,
        grace_safety_bytes=config.grace_reserve_bytes,
    )
    controller = fused_moe.ResidencyController(
        config=fused_moe.AutomaticResidencyConfig(
            mode="static",
            workload=config.workload,
            provenance="vLLM static expert residency",
            profile_path=config.profile_path,
        ),
        model=model_spec,
        budget=budget,
        hardware=fused_moe.ResidencyHardware.detect(device),
        store=fused_moe.ResidencyProfileStore(config.profile_directory),
    )
    progress = controller.startup()
    for _, declaration, _, _ in lanes:
        declaration.plan = controller.plan_execution(
            layer=declaration.layer_name,
            weight_plan=declaration.weight_plan,
            weights=declaration.packed_weights(fused_moe, fingerprint),
        )
    _log_placement(controller.placements(), progress, budget)


def _log_placement(placements: Iterable[Any], progress: Any, budget: Any) -> None:
    placements = tuple(placements)
    hot = sum(len(p.hbm_expert_ids) for p in placements)
    total = sum(p.total_experts for p in placements)
    logger.info(
        "Expert residency: %d of %d experts in HBM across %d layers (%s); "
        "KV reserve %.2f GiB, HBM headroom %.2f GiB",
        hot,
        total,
        len(placements),
        progress.reason,
        budget.kv_reserved_bytes / (1 << 30),
        budget.hbm_safety_bytes / (1 << 30),
    )


def residency_prepare_call_factory(
    *, tokens: int, topk: int, hidden_size: int, num_experts: int, device: torch.device
):
    """Prime the prepared residency operator with seeded routes at capacity."""

    def factory(state: Any):
        from b12x.moe.fused_moe.workloads import make_tuning_routes
        from b12x.preparation import PreparedCall

        generator = torch.Generator(device=device).manual_seed(42)
        hidden = torch.empty((tokens, hidden_size), dtype=torch.bfloat16, device=device)
        hidden.normal_(mean=0.0, std=0.125, generator=generator)
        ids = make_tuning_routes(tokens, topk, num_experts, device=device)[0]
        ids = ids.to(torch.int32).contiguous()
        logits = torch.arange(topk, dtype=torch.float32, device=device) * 0.125
        weights = torch.softmax(logits, dim=-1).expand(tokens, topk).contiguous()
        output = torch.empty_like(hidden)
        binding = state.bind(
            a=hidden, topk_ids=ids, topk_weights=weights, output=output
        )
        return PreparedCall(
            run=binding.run,
            output=output,
            owners=(state, binding, hidden, ids, weights, output),
        )

    return factory
