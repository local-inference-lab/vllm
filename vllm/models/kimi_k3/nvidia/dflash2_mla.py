# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash2 draft model for Kimi-K3 on the colocated MLA draft backbone.

`lightseekorg/kimi-k3-dflash2` (`DFlash2DraftModel`) is a block-diffusion
draft: one parallel forward over an anchor token plus mask tokens proposes a
whole block. Its backbone is the same as the Kimi-K3 DSpark draft's
(`K3DSparkModel`: five MLA decoder layers over a context projection of the
target's auxiliary hidden states, TP-sharded, fused latent context KV), with
three additions from the DFlash2 checkpoint:

* grouped dynamic convolutions around every attention and MLP
  (`attention_conv`, `mlp_conv`: kernel taps ``conv_kernel_size``, groups of
  ``conv_group_size`` channels, per-token coefficients from a
  ``kernel_projection`` of the normalized layer input, mixing each token with
  its predecessors inside the drafted block);
* a pairwise candidate selector (`candidate_selector`: predecessor and
  successor codebooks of rank ``selector_rank`` plus a hidden projection,
  scoring transitions between the top-``selector_top_k`` unary candidates of
  neighbouring block positions);
* no Markov head and no confidence head.

The checkpoint declares ``model_type: qwen3`` with ``attention_mode: mla``
and the DFlash nested ``dflash_config`` (block size, taps, selector, mask
token, target taps). Under speculative method ``dflash`` the draft config is
wrapped in ``EAGLEConfig`` (model type ``eagle``, architectures unchanged,
every checkpoint field copied); ``normalize_dflash2_config`` lifts the fields
the MLA backbone reads (``target_layer_ids``, ``num_target_layers``,
``target_hidden_size``, ``draft_vocab_size``) to the top level, exactly as the
serving runtime of the checkpoint's authors does, and pins the DFlash query
layout (anchor + mask tokens, no sampling from the anchor). Target taps are
0-based completed-layer outputs; the backbone and the target's capture both
shift them by one to vLLM's auxiliary-state ids (embedding output is id 0).
Every layer attends non-causally inside the block, as in the reference; the
per-layer sliding windows of the checkpoint (4096 on four layers, unbounded
on one) are served with one uniform bounded draft KV window
(``VLLM_DSPARK_DRAFT_KV_WINDOW``), so the full-attention layer sees at most
that window of context.

Serving contract: the embedding and LM head are the target's (the checkpoint
ships copies of the target embedding only). Proposals are sampled per block
position from the LM-head distribution of the draft hidden states (the
speculator's probabilistic path, draft probabilities handed to the rejection
sampler). With the selector enabled (``VLLM_DFLASH2_SELECTOR``, the default)
the positions are sampled in order and each position's logits carry the
selector's transition scores from the token sampled before it to the
position's unary top-k candidates (``DFlash2CandidateSelector.condition``),
so the proposal distribution is the reference lattice's chain and the
rejection sampler sees exactly that distribution. With the selector disabled
the block is sampled in one parallel pass.

Weight names of the checkpoint and their module paths here:
``fc.weight`` -> ``model.context_proj.weight``; ``hidden_norm.weight`` ->
``model.context_norm.weight``; ``norm.weight`` -> ``model.final_norm.weight``;
``layers.N.*`` unchanged under ``model.`` (MLA projections fuse as in the
DSpark loader); ``candidate_selector.*`` under ``model.``;
``embed_tokens.weight`` skipped.
"""

from __future__ import annotations

from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.models.utils import maybe_prefix
from vllm.models.common.ops.fused_allreduce_rms_norm import (
    fused_allreduce_rms_norm,
)
from vllm.models.kimi_k3.nvidia.dspark_mla import (
    K3DSparkDecoderLayer,
    K3DSparkForCausalLM,
    K3DSparkModel,
)

logger = init_logger(__name__)

DFLASH2_ARCHITECTURE = "DFlash2DraftModel"

_CHECKPOINT_RENAMES = {
    "fc.weight": "context_proj.weight",
    "hidden_norm.weight": "context_norm.weight",
    "norm.weight": "final_norm.weight",
}


def is_dflash2_draft(hf_config: Any) -> bool:
    """Whether a draft config is the DFlash2 MLA checkpoint."""
    architectures = getattr(hf_config, "architectures", None) or ()
    if DFLASH2_ARCHITECTURE not in architectures:
        return False
    nested = getattr(hf_config, "dflash_config", None) or {}
    return str(nested.get("attention_mode", "gqa")).lower() == "mla"


def normalize_dflash2_config(hf_config: Any) -> Any:
    """Lift the nested DFlash2 fields the MLA backbone reads to the top level.

    Idempotent. ``num_target_layers`` of the checkpoint counts the target's
    layers (93); the backbone reads it as the number of tapped layers, so it
    becomes the length of ``target_layer_ids``.
    """
    nested = dict(getattr(hf_config, "dflash_config", None) or {})
    target_layer_ids = nested.get("target_layer_ids")
    if target_layer_ids is None:
        raise ValueError("DFlash2 config lacks dflash_config.target_layer_ids")
    target_layer_ids = [int(layer_id) for layer_id in target_layer_ids]
    hf_config.target_layer_ids = target_layer_ids
    hf_config.num_target_layers = len(target_layer_ids)
    hf_config.target_hidden_size = int(
        getattr(hf_config, "target_hidden_size", None) or hf_config.hidden_size
    )
    hf_config.draft_vocab_size = int(
        getattr(hf_config, "draft_vocab_size", None) or hf_config.vocab_size
    )
    # The DFlash block: the anchor (bonus) token plus mask tokens; the anchor
    # position is not a prediction.
    hf_config.sample_from_anchor = False
    if (
        "mask_token_id" not in nested
        and getattr(hf_config, "mask_token_id", None) is None
    ):
        raise ValueError("DFlash2 config lacks dflash_config.mask_token_id")
    if float(nested.get("input_embedding_scale", 1.0)) != 1.0:
        # The reference scales only the mask-token embedding; the shared
        # target embedding table is not rescaled here.
        raise ValueError("DFlash2 input_embedding_scale != 1.0 is unsupported")
    hf_config.dflash_config = nested
    return hf_config


def rename_dflash2_checkpoint_name(name: str) -> str | None:
    """Map one checkpoint tensor name to the backbone's parameter name.

    Returns ``None`` for tensors the serving model does not own (the target's
    embedding is shared, rotary tables are rebuilt).
    """
    name = name.removeprefix("model.")
    if name == "embed_tokens.weight" or "rotary_emb.inv_freq" in name:
        return None
    return _CHECKPOINT_RENAMES.get(name, name)


def _grouped_conv(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    num_groups: int,
    group_size: int,
    taps: int,
) -> torch.Tensor:
    """DFlash2's grouped dynamic convolution over flat draft blocks.

    ``hidden_states`` is ``[tokens, hidden]`` laid out as consecutive blocks
    of ``block_size`` rows (one per request); ``delta`` is ``[tokens, taps,
    groups]`` and ``base`` ``[taps, hidden]``. Row ``t`` of a block mixes rows
    ``t - tap`` for ``tap < taps`` with coefficient ``base + delta``; taps that
    reach before the block start are dropped.
    """
    blocks = hidden_states.unflatten(-1, (num_groups, group_size))
    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    output = coefficients[:, 0] * blocks
    position = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    if block_size & (block_size - 1) == 0:
        position = position & (block_size - 1)
    else:
        position = position % block_size
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        output = output + coefficients[:, tap] * shifted * (position >= tap).view(
            -1, 1, 1
        )
    return output.flatten(-2)


class DFlash2GroupedConv(nn.Module):
    """Grouped convolution pair of one DFlash2 sublayer (prepare / finish)."""

    def __init__(
        self,
        hidden_size: int,
        taps: int,
        group_size: int,
        block_size: int,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        if hidden_size % group_size:
            raise ValueError(
                f"conv_group_size={group_size} must divide hidden_size={hidden_size}."
            )
        if not 1 <= taps <= block_size:
            raise ValueError(
                f"conv_kernel_size={taps} must be in [1, block_size={block_size}]."
            )
        self.block_size = int(block_size)
        self.taps = int(taps)
        self.group_size = int(group_size)
        self.num_groups = hidden_size // self.group_size
        self.base_kernel = nn.Parameter(
            torch.empty(2, self.taps, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        self.kernel_projection = ReplicatedLinear(
            hidden_size,
            2 * self.taps * self.num_groups,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "kernel_projection"),
            return_bias=False,
        )

    def _convolve(
        self, hidden_states: torch.Tensor, delta: torch.Tensor, side: int
    ) -> torch.Tensor:
        return _grouped_conv(
            hidden_states,
            delta,
            self.base_kernel[side],
            self.block_size,
            self.num_groups,
            self.group_size,
            self.taps,
        )

    def prepare(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convolve the sublayer input; return it with the output coefficients."""
        coefficients = self.kernel_projection(hidden_states).reshape(
            hidden_states.shape[0], 2, self.taps, self.num_groups
        )
        return self._convolve(hidden_states, coefficients[:, 0], 0), coefficients[:, 1]

    def finish(
        self, hidden_states: torch.Tensor, coefficients: torch.Tensor
    ) -> torch.Tensor:
        """Convolve the sublayer output with the coefficients of ``prepare``.

        The convolution is linear in ``hidden_states`` for fixed coefficients,
        and the coefficients are computed from the replicated normalized
        input, so applying it to a TP-partial output before the fused
        all-reduce of the next norm sums to the convolution of the reduced
        output.
        """
        return self._convolve(hidden_states, coefficients, 1)


class DFlash2DecoderLayer(K3DSparkDecoderLayer):
    """MLA decoder layer with DFlash2's convolutions around attention and MLP."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config,
        layer_idx: int,
        start_layer_id: int,
        prefix: str,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            config=config,
            layer_idx=layer_idx,
            start_layer_id=start_layer_id,
            prefix=prefix,
        )
        nested = config.dflash_config
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        layer_prefix = f"layers.{start_layer_id + layer_idx}"

        def grouped_conv(name: str) -> DFlash2GroupedConv:
            return DFlash2GroupedConv(
                hidden_size=int(config.hidden_size),
                taps=int(nested["conv_kernel_size"]),
                group_size=int(nested["conv_group_size"]),
                # The drafted block: the anchor plus the mask tokens of one
                # request, the row layout of the speculator's query batch.
                block_size=1 + int(speculative_config.num_speculative_tokens),
                params_dtype=vllm_config.model_config.dtype,
                prefix=maybe_prefix(prefix, f"{layer_prefix}.{name}"),
            )

        self.attention_conv = grouped_conv("attention_conv")
        self.mlp_conv = grouped_conv("mlp_conv")

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        rope_cos_sin_cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = fused_allreduce_rms_norm(
                hidden_states, residual, self.input_layernorm
            )
        hidden_states, coefficients = self.attention_conv.prepare(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            rope_cos_sin_cache=rope_cos_sin_cache,
        )
        hidden_states = self.attention_conv.finish(hidden_states, coefficients)
        hidden_states, residual = fused_allreduce_rms_norm(
            hidden_states, residual, self.post_attention_layernorm
        )
        hidden_states, coefficients = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_conv.finish(hidden_states, coefficients)
        return hidden_states, residual


def score_selector_edges(
    predecessor_table: torch.Tensor,
    successor_table: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    hidden: torch.Tensor,
    anchor_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Score every (predecessor candidate -> successor candidate) edge.

    ``candidate_ids`` and ``unary_logits`` are ``[batch, steps, top_k]``,
    ``hidden`` is the projected draft hidden state ``[batch, steps, rank]``.
    Step 0's predecessors are the anchor tokens; step ``s > 0``'s predecessors
    are step ``s - 1``'s candidates. Returns ``[batch, steps, top_k
    (predecessor), top_k (successor)]``: unary score of the successor plus the
    bilinear transition ``<predecessor * hidden, successor>``.
    """
    top_k = candidate_ids.shape[-1]
    successors = successor_table[candidate_ids]
    predecessor_ids = torch.cat(
        (
            anchor_token_ids[:, None, None].expand(-1, 1, top_k),
            candidate_ids[:, :-1],
        ),
        dim=1,
    )
    predecessors = predecessor_table[predecessor_ids]
    return unary_logits[:, :, None] + torch.einsum(
        "blpr,blcr->blpc", predecessors * hidden[:, :, None], successors
    )


def add_selector_transitions(
    logits: torch.Tensor,
    candidate_ids: torch.Tensor,
    successor_rows: torch.Tensor,
    predecessor_rows: torch.Tensor,
    hidden: torch.Tensor,
) -> torch.Tensor:
    """Add the transition score of each candidate to its logit, in place.

    ``logits`` is ``[rows, vocab]``, ``candidate_ids`` ``[rows, k]`` (the
    unary top-k of each row), ``successor_rows`` ``[rows, k, rank]`` (their
    successor codebook rows), ``predecessor_rows`` ``[rows, rank]`` (the
    predecessor codebook row of the token preceding each row) and ``hidden``
    ``[rows, rank]`` (the projected draft hidden state of the row). Candidates
    outside the top-k keep their unary logit, as in the reference lattice.
    """
    transition = torch.einsum("br,bcr->bc", predecessor_rows * hidden, successor_rows)
    return logits.scatter_add_(1, candidate_ids, transition.to(logits.dtype))


class DFlash2CandidateSelector(nn.Module):
    """DFlash2's low-rank transition model over top-k candidate lattices."""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        rank: int,
        top_k: int,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError(f"selector_rank must be positive, got {rank}.")
        if not 2 <= top_k <= vocab_size:
            raise ValueError(
                f"selector_top_k must be in [2, {vocab_size}], got {top_k}."
            )
        self.top_k = int(top_k)
        self.predecessor_codebook = nn.Parameter(
            torch.empty(vocab_size, rank, dtype=params_dtype), requires_grad=False
        )
        self.successor_codebook = nn.Parameter(
            torch.empty(vocab_size, rank, dtype=params_dtype), requires_grad=False
        )
        self.hidden_projection = ReplicatedLinear(
            hidden_size,
            rank,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "hidden_projection"),
            return_bias=False,
        )

    def forward(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.hidden_projection(hidden_states)
        return score_selector_edges(
            self.predecessor_codebook,
            self.successor_codebook,
            candidate_ids,
            unary_logits,
            hidden,
            anchor_token_ids,
        )

    def prepare_rows(
        self, hidden_states: torch.Tensor, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-row inputs of the sequential conditioning.

        Returns the projected hidden states ``[rows, rank]``, the unary top-k
        candidate ids ``[rows, k]`` and their successor rows ``[rows, k, rank]``.
        """
        hidden = self.hidden_projection(hidden_states)
        candidate_ids = logits.topk(self.top_k, dim=-1).indices
        return hidden, candidate_ids, self.successor_codebook[candidate_ids]

    def condition(
        self,
        logits: torch.Tensor,
        previous_token_ids: torch.Tensor,
        hidden: torch.Tensor,
        candidate_ids: torch.Tensor,
        successor_rows: torch.Tensor,
    ) -> torch.Tensor:
        """Condition one block position's logits on the tokens before it."""
        return add_selector_transitions(
            logits,
            candidate_ids,
            successor_rows,
            self.predecessor_codebook[previous_token_ids],
            hidden,
        )


class DFlash2Model(K3DSparkModel):
    decoder_layer_cls = DFlash2DecoderLayer
    uses_markov_head = False

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            start_layer_id=start_layer_id,
            prefix=prefix,
        )
        nested = self.config.dflash_config
        self.candidate_selector = DFlash2CandidateSelector(
            hidden_size=int(self.config.hidden_size),
            vocab_size=int(self.config.vocab_size),
            rank=int(nested["selector_rank"]),
            top_k=int(nested["selector_top_k"]),
            params_dtype=vllm_config.model_config.dtype,
            prefix=maybe_prefix(prefix, "candidate_selector"),
        )


class DFlash2ForCausalLM(K3DSparkForCausalLM):
    """`DFlash2DraftModel` on the colocated Kimi-K3 MLA draft backbone."""

    model_cls = DFlash2Model
    checkpoint_skip_substrs = ("embed_tokens", "lm_head")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        assert vllm_config.speculative_config is not None
        normalize_dflash2_config(
            vllm_config.speculative_config.draft_model_config.hf_config
        )
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    @property
    def candidate_selector(self) -> DFlash2CandidateSelector:
        return self.model.candidate_selector

    def supports_local_draft_argmax(self) -> bool:
        return False

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def renamed() -> Iterable[tuple[str, torch.Tensor]]:
            for name, weight in weights:
                target = rename_dflash2_checkpoint_name(name)
                if target is not None:
                    yield target, weight

        return super().load_weights(renamed())


def describe_dflash2_draft(
    hf_config: Any, num_speculative_steps: int, sampled: bool
) -> str:
    """One-line description of the served DFlash2 geometry for the boot log."""
    geometry = dflash2_config_summary(hf_config)
    layer_types = "/".join(t.replace("_attention", "") for t in geometry.layer_types)
    taps = ",".join(str(layer_id) for layer_id in geometry.target_layer_ids)
    return (
        f"DFlash2 draft: {geometry.layers} MLA layers ({layer_types}, window "
        f"{geometry.sliding_window}), block {geometry.block_size}, conv taps "
        f"{geometry.taps} x groups of {geometry.group_size}, selector rank "
        f"{geometry.selector_rank} top-{geometry.selector_top_k}, target taps "
        f"[{taps}], {num_speculative_steps} speculative tokens per step, "
        f"{'sampled' if sampled else 'greedy'} proposals."
    )


def dflash2_config_summary(hf_config: Any) -> SimpleNamespace:
    """Small read-only view of the DFlash2 geometry for logs and tests."""
    nested = getattr(hf_config, "dflash_config", None) or {}
    return SimpleNamespace(
        layers=int(hf_config.num_hidden_layers),
        layer_types=list(getattr(hf_config, "layer_types", ()) or ()),
        sliding_window=getattr(hf_config, "sliding_window", None),
        block_size=int(nested.get("block_size", 0)),
        taps=int(nested.get("conv_kernel_size", 0)),
        group_size=int(nested.get("conv_group_size", 0)),
        selector_rank=int(nested.get("selector_rank", 0)),
        selector_top_k=int(nested.get("selector_top_k", 0)),
        target_layer_ids=list(nested.get("target_layer_ids", ())),
        mask_token_id=nested.get("mask_token_id"),
    )
