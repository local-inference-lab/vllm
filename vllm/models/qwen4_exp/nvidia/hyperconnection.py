# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HyperConnection (Gated Residual) utilities — NVIDIA model variant.

Implements the HyperConnection residual scheme proposed in
"HyperConnections" (https://arxiv.org/abs/2409.19606). This NVIDIA variant
delays each HC combine to the following HC mix boundary. HC glue kernels,
including fused combine+RMSNorm, live in ``ops/hc.py``; projections remain
standard vLLM Linear modules.

Hidden states between layers have shape ``[..., HC*HS]`` with HS inner
(HC outer, HS inner — checkpoint-native layout).

Typical usage inside a transformer decoder layer::

    self.attn_hc = GatedResidual(hc_config)

    hidden_states, block_input, injection = self.attn_hc.mix(hidden_states)
    attention_output = attention(block_input)
    hidden_states, block_input, injection = self.mlp_hc.combine_and_mix(
        hidden_states, attention_output, injection
    )
"""

from collections.abc import Iterable, Sequence
from typing import Any

import torch
from torch import nn

import vllm.envs as envs
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.weight_transfer import copy_weight
from vllm.platforms import current_platform
from vllm.utils.b12x import (
    B12xPreparationUnit,
    B12xWorkload,
    PreparationResourceUnavailableError,
    get_b12x_hyperconnection,
    set_b12x_preparation_provider,
)

from ..common.hyperconnection import (
    GroupedGemmaRMSNorm,
    HyperConnectionConfig,
)
from .ops.hc import (
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_combine_norm,
    hc_gate_mix,
    hc_silu,
)

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Gated-residual variant
# ---------------------------------------------------------------------------
def _hyperconnection_api() -> Any:
    api = get_b12x_hyperconnection()
    if api is None:
        raise ImportError(
            "Qwen4Exp requires b12x.norm.hyperconnection; "
            "install the b12x serving extra"
        )
    return api


class _ShardedDownProjection(ReplicatedLinear):
    """Local bottleneck rows with replicated injection rows in one GEMM."""

    def __init__(
        self,
        config: HyperConnectionConfig,
        tp_rank: int,
        tp_size: int,
        use_combine: bool,
        prefix: str,
    ) -> None:
        self.hc_tp_rank = tp_rank
        self.full_lowrank = config.hc_lowrank
        self.local_lowrank = config.hc_lowrank // tp_size
        self.injection_size = config.hc_count if use_combine else 0
        rows = self.local_lowrank + self.injection_size
        self.padding = -rows % 16
        super().__init__(
            config.hc_count * config.hidden_size,
            rows + self.padding,
            bias=False,
            params_dtype=config.params_dtype,
            return_bias=False,
            disable_tp=True,
            prefix=prefix,
        )

    def weight_loader(
        self,
        param: torch.Tensor,
        loaded_weight: torch.Tensor,
        shard_id: int | None = None,
    ) -> None:
        if shard_id is None and self.injection_size:
            self.weight_loader(param, loaded_weight[: self.full_lowrank], 0)
            self.weight_loader(
                param,
                loaded_weight[
                    self.full_lowrank : self.full_lowrank + self.injection_size
                ],
                1,
            )
            return
        if shard_id in (None, 0):
            assert loaded_weight.shape == (self.full_lowrank, self.input_size)
            source = loaded_weight.narrow(
                0, self.hc_tp_rank * self.local_lowrank, self.local_lowrank
            )
            destination = param.data[: self.local_lowrank]
        elif shard_id == 1 and self.injection_size:
            assert loaded_weight.shape == (self.injection_size, self.input_size)
            source = loaded_weight
            destination = param.data[
                self.local_lowrank : self.local_lowrank + self.injection_size
            ]
        else:
            raise ValueError(f"Invalid HC projection shard: {shard_id}")
        copy_weight(destination, source)
        if self.padding and shard_id != 1:
            copy_weight(
                param.data[-self.padding :],
                torch.zeros(
                    self.padding, self.input_size, dtype=param.dtype, device="cpu"
                ),
            )

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[str]:
        for name, weight in weights:
            if name != "weight":
                raise ValueError(f"Unexpected HC projection parameter: {name}")
            self.weight_loader(self.weight, weight, getattr(weight, "shard_id", None))
            yield name


class _ShardedUpProjection(ReplicatedLinear):
    """Shard features within every residual stream, preserving stream order."""

    def __init__(
        self, config: HyperConnectionConfig, tp_rank: int, tp_size: int, prefix: str
    ) -> None:
        self.hc_tp_rank = tp_rank
        self.streams = config.hc_count
        self.full_hidden = config.hidden_size
        self.local_hidden = config.hidden_size // tp_size
        super().__init__(
            config.hc_lowrank,
            self.streams * self.local_hidden,
            bias=False,
            params_dtype=config.params_dtype,
            return_bias=False,
            disable_tp=True,
            prefix=prefix,
        )

    def weight_loader(self, param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        assert loaded_weight.shape == (
            self.streams * self.full_hidden,
            self.input_size,
        )
        # Separate views retain b12x's file-backed source ranges during loading.
        for stream in range(self.streams):
            source = loaded_weight.narrow(
                0,
                stream * self.full_hidden + self.hc_tp_rank * self.local_hidden,
                self.local_hidden,
            )
            destination = param.data.narrow(
                0, stream * self.local_hidden, self.local_hidden
            )
            copy_weight(destination, source)


class HyperConnectionWorkspace(nn.Module):
    """Fixed-capacity storage shared by all HC modules in one model."""

    def __init__(self, config: HyperConnectionConfig, max_tokens: int) -> None:
        super().__init__()
        if not config.hc_per_branch_norm:
            raise NotImplementedError(
                "Qwen4Exp requires one RMSNorm group per HC stream"
            )
        self.config = config
        self.max_tokens = int(max_tokens)
        self.device = torch.device(current_platform.current_device())
        tp_size = get_tensor_model_parallel_world_size()
        self.tp_size = (
            tp_size
            if envs.VLLM_QWEN3_8_FLASH_NEXT_HC_TP
            and current_platform.is_cuda()
            and config.params_dtype == torch.bfloat16
            and config.hc_lowrank % tp_size == 0
            and config.hidden_size % tp_size == 0
            else 1
        )
        self.tp_rank = get_tensor_model_parallel_rank() if self.tp_size > 1 else 0
        width = config.hc_count * config.hidden_size
        factory = dict(device=self.device, dtype=config.params_dtype)
        self.register_buffer(
            "normalized", torch.empty(max_tokens, width, **factory), persistent=False
        )
        self.register_buffer(
            "bottleneck",
            torch.empty(max_tokens, config.hc_lowrank, **factory),
            persistent=False,
        )
        self.register_buffer(
            "block_input",
            torch.empty(max_tokens, config.hidden_size, **factory),
            persistent=False,
        )
        if self.tp_size > 1:
            local_hidden = config.hidden_size // self.tp_size
            local_lowrank = config.hc_lowrank // self.tp_size
            logger.info_once(
                "Sharding HyperConnection projections across %d ranks: "
                "local bottleneck=%d, features/stream=%d; FP32 down outputs.",
                self.tp_size,
                local_lowrank,
                local_hidden,
            )
            down_width = local_lowrank + config.hc_count
            down_width += -down_width % 16
            for name, columns in (
                ("local_normalized", config.hc_count * local_hidden),
                ("local_bottleneck", local_lowrank),
                ("local_block_input", local_hidden),
                ("local_gates", config.hc_count * local_hidden),
                ("local_down", down_width),
            ):
                self.register_buffer(
                    name, torch.empty(max_tokens, columns, **factory), persistent=False
                )
            self.register_buffer(
                "local_down_fp32",
                torch.empty(
                    max_tokens, down_width, device=self.device, dtype=torch.float32
                ),
                persistent=False,
            )

    def caps(self, max_tokens: int, *, local: bool = False):
        api = _hyperconnection_api()
        partitions = self.tp_size if local else 1
        return api.Caps(
            device=self.device,
            max_tokens=max_tokens,
            hidden_size=self.config.hidden_size // partitions,
            streams=self.config.hc_count,
            lowrank=self.config.hc_lowrank // partitions,
            dtype=self.config.params_dtype,
        )

    def bind(self, plan, tokens: int, *, local: bool = False):
        return _hyperconnection_api().bind(
            plan,
            normalized=self.local_normalized if local else self.normalized,
            bottleneck=self.local_bottleneck if local else self.bottleneck,
            block_input=self.local_block_input if local else self.block_input,
            tokens=tokens,
        )


class GatedResidual(nn.Module):
    """Gated HyperConnection with learnable low-rank mixing and injection.

    ``combine_and_mix()`` runs the pre pipeline (grouped GemmaRMSNorm -> merged
    low-rank down+inject GEMM -> silu -> up GEMM -> sigmoid -> gated mean
    over the HC streams). When passed a pending block output, it fuses its
    residual combine with the RMSNorm. A missing injection selects unit-weight
    combine. Final mixers use ``use_combine=False`` and do not produce a new
    injection.

    TP shards bottleneck rows and output features within each stream, with a
    gather at each boundary. The down GEMM returns FP32 before its BF16 cast.
    Norm and residual state remain replicated; TP1 uses standard Linear dispatch.
    """

    def __init__(
        self,
        config: HyperConnectionConfig,
        use_combine: bool = True,
        prefix: str = "",
        *,
        workspace: HyperConnectionWorkspace | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.tp_size = workspace.tp_size if workspace is not None else 1
        self.tp_rank = workspace.tp_rank if workspace is not None else 0
        self.lora_rank = config.hc_lowrank // self.tp_size
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.use_combine = use_combine

        norm_size = (
            self.hyper_hidden_size if config.hc_per_branch_norm else config.hidden_size
        )
        group_size = config.hidden_size if config.hc_per_branch_norm else None
        # Normalize each H-sized HC stream independently while retaining a
        # separate affine weight for every element of the HC*H layout.
        self.hc_norm = GroupedGemmaRMSNorm(
            norm_size,
            eps=config.rms_norm_eps,
            group_size=group_size,
            dtype=config.params_dtype,
        )

        # -- vLLM Linear weights --------------------------------------------
        # The merged skinny-GEMM shape is physically padded to 16 rows to ensure
        # good alignment and performant implementation chosen by CuBLAS heuristics.
        self.pad_size = (-(self.lora_rank + self.hc_count)) % 16 if use_combine else 0
        if self.tp_size > 1:
            name = (
                "input_mix_weight_down_block_inject"
                if use_combine
                else "input_mix_weight_down"
            )
            setattr(
                self,
                name,
                _ShardedDownProjection(
                    config,
                    self.tp_rank,
                    self.tp_size,
                    use_combine,
                    maybe_prefix(prefix, name),
                ),
            )
        elif use_combine:
            self.input_mix_weight_down_block_inject = MergedColumnParallelLinear(
                self.hyper_hidden_size,
                [self.lora_rank, self.hc_count]
                + ([self.pad_size] if self.pad_size else []),
                bias=False,
                params_dtype=config.params_dtype,
                quant_config=None,
                prefix=maybe_prefix(prefix, "input_mix_weight_down_block_inject"),
                return_bias=False,
                disable_tp=True,
            )
        else:
            self.input_mix_weight_down = ReplicatedLinear(
                self.hyper_hidden_size,
                self.lora_rank,
                bias=False,
                params_dtype=config.params_dtype,
                quant_config=None,
                prefix=maybe_prefix(prefix, "input_mix_weight_down"),
                return_bias=False,
            )
        self.input_mix_weight_up = (
            _ShardedUpProjection(
                config,
                self.tp_rank,
                self.tp_size,
                maybe_prefix(prefix, "input_mix_weight_up"),
            )
            if self.tp_size > 1
            else ReplicatedLinear(
                self.lora_rank,
                self.hyper_hidden_size,
                bias=False,
                params_dtype=config.params_dtype,
                quant_config=None,
                prefix=maybe_prefix(prefix, "input_mix_weight_up"),
                return_bias=False,
            )
        )
        object.__setattr__(self, "_workspace", workspace)
        self._preparation_prefix = prefix or "qwen4_exp.hyperconnection"
        self._plans: dict[str, object] = {}
        if workspace is not None and not getattr(
            self, "b12x_preparation_suppressed", False
        ):
            set_b12x_preparation_provider(self, self)

    def mix(self, hidden_states: torch.Tensor):
        if self.workspace is not None:
            normalized = _hyperconnection_api().run_grouped_rmsnorm(
                hidden_states,
                self.hc_norm.weight,
                eps=self.config.rms_norm_eps,
                binding=self._binding(hidden_states, "grouped_rmsnorm"),
            )
        else:
            normalized = grouped_gemma_rmsnorm(
                hidden_states,
                self.hc_norm.weight,
                self.config.rms_norm_eps,
                self.hc_count,
            )
        block_input, injection = self._mix_normalized(normalized)
        return hidden_states, block_input, injection

    def combine_and_mix(self, hidden_states, prev_block_output, prev_injection):
        if prev_block_output is None:
            return self.mix(hidden_states)
        if self.workspace is not None and prev_injection is not None:
            combined, normalized = _hyperconnection_api().run_combine_norm(
                hidden_states,
                prev_block_output,
                prev_injection,
                self.hc_norm.weight,
                eps=self.config.rms_norm_eps,
                plan=self._plan_for("combine_norm"),
            )
        else:
            combined, normalized = hc_combine_norm(
                hidden_states,
                prev_block_output,
                prev_injection,
                self.hc_norm.weight,
                self.config.rms_norm_eps,
                self.hc_count,
            )
        block_input, injection = self._mix_normalized(normalized)
        return combined, block_input, injection

    def combine(self, hidden_states, block_output, injection):
        if self.workspace is not None and injection is not None:
            return _hyperconnection_api().run_combine(
                hidden_states,
                block_output,
                injection,
                plan=self._plan_for("combine"),
            )
        return hc_combine(hidden_states, block_output, injection, self.hc_count)

    @property
    def hyper_hidden_size(self) -> int:
        return self.hc_count * self.hidden_size

    def _request_name(self, operation: str, tokens: int) -> str:
        return f"{self._preparation_prefix}.hc.{operation}.m{tokens}"

    def get_b12x_preparation_units(
        self, layer: torch.nn.Module, workload: B12xWorkload
    ) -> Sequence[B12xPreparationUnit]:
        if layer is not self:
            raise ValueError("HC preparation owner mismatch")
        if self.workspace is None or workload.stage != "weights":
            return ()
        if self.hc_norm.weight.is_meta:
            return ()
        if workload.max_tokens > self.workspace.max_tokens:
            raise PreparationResourceUnavailableError(
                f"{self._preparation_prefix} HC workspace capacity "
                f"{self.workspace.max_tokens} cannot serve {workload.max_tokens}"
            )
        api = _hyperconnection_api()
        operations = (
            "grouped_rmsnorm",
            "scaled_silu",
            "gate_mean",
            "combine",
            "combine_norm",
        )
        requests = []
        tokens = workload.max_tokens
        plans = {}
        for operation in operations:
            local = self.tp_size > 1 and operation in ("scaled_silu", "gate_mean")
            plan = api.plan(
                self.workspace.caps(tokens, local=True)
                if local
                else self.workspace.caps(tokens),
                invocation={"operation": operation, "eps": self.config.rms_norm_eps},
            )
            plans[operation] = plan
            requests.append(
                plan.request(
                    name=self._request_name(operation, tokens),
                    prepare_call=self._prepare_call(operation, tokens),
                    benchmark_call=self._benchmark_call(operation, tokens),
                )
            )
        self._plans = plans
        return (
            B12xPreparationUnit(
                name="HYPERCONNECTION",
                key=(self._preparation_prefix, tokens),
                requests=tuple(requests),
                stage="weights",
            ),
        )

    def _prepare_call(self, operation: str, tokens: int):
        """Prime the installed operation against its durable workspace."""
        return self._call_factory(operation, tokens, benchmark=False)

    def _benchmark_call(self, operation: str, tokens: int):
        """Measure an isolated binding; it is never retained for serving."""
        return self._call_factory(operation, tokens, benchmark=True)

    def _call_factory(self, operation: str, tokens: int, *, benchmark: bool):
        def prepare(state):
            from b12x.norm.hyperconnection import _impl
            from b12x.preparation import PreparedCall

            factory = dict(device=self.workspace.device, dtype=self.config.params_dtype)
            local = self.tp_size > 1 and operation in ("scaled_silu", "gate_mean")
            hidden_size = (
                self.hidden_size // self.tp_size if local else self.hidden_size
            )
            width = self.hc_count * hidden_size
            activation_inputs = []
            activation_owners: list[torch.Tensor] = []
            owners: tuple[torch.Tensor, ...]

            def activation(shape):
                value = torch.empty(shape, **factory)
                template = torch.arange(value.numel(), **factory).reshape(shape)
                template.div_(max(value.numel(), 1))
                activation_inputs.append((value, template))
                activation_owners.extend((value, template))
                return value

            def produce():
                for value, template in activation_inputs:
                    value.copy_(template)

            # Serving priming writes its owned workspace. Trials instead own
            # independent output storage so no measured binding can escape.
            def output(shape, serving):
                return torch.empty(shape, **factory) if benchmark else serving

            if operation == "grouped_rmsnorm":
                source = activation((tokens, width))
                out = output((tokens, width), self.workspace.normalized)
                run = lambda: _impl.run_grouped_rmsnorm_impl(
                    source,
                    self.hc_norm.weight,
                    eps=self.config.rms_norm_eps,
                    plan=state,
                    out=out,
                )
                owners = (out,)
            elif operation == "scaled_silu":
                source = activation((tokens, self.lora_rank))
                out = output(
                    (tokens, self.lora_rank),
                    self.workspace.local_bottleneck
                    if local
                    else self.workspace.bottleneck,
                )
                run = lambda: _impl.run_scaled_silu_impl(source, plan=state, out=out)
                owners = (out,)
            elif operation == "gate_mean":
                source = activation((tokens, width))
                gates = activation((tokens, width))
                out = output(
                    (tokens, hidden_size),
                    self.workspace.local_block_input
                    if local
                    else self.workspace.block_input,
                )
                run = lambda: _impl.run_gate_mean_impl(
                    source, gates, plan=state, out=out
                )
                owners = (out,)
            else:
                hidden = activation((tokens, width))
                block = activation((tokens, self.hidden_size))
                injection = activation((tokens, self.hc_count))
                if operation == "combine":
                    run = lambda: _impl.run_combine_impl(
                        hidden,
                        block,
                        injection,
                        plan=state,
                    )
                else:
                    run = lambda: _impl.run_combine_norm_impl(
                        hidden,
                        block,
                        injection,
                        self.hc_norm.weight,
                        eps=self.config.rms_norm_eps,
                        plan=state,
                    )
                owners = ()
            return PreparedCall(
                run=run,
                produce=produce,
                owners=(*activation_owners, *owners),
            )

        return prepare

    def _plan_for(self, operation: str):
        try:
            return self._plans[operation]
        except KeyError:
            raise PreparationResourceUnavailableError(
                f"{self._preparation_prefix} lacks a declared {operation} plan"
            ) from None

    @property
    def workspace(self) -> HyperConnectionWorkspace:
        return self._workspace

    def _binding(self, hidden_states: torch.Tensor, operation: str):
        if self.tp_size > 1 and operation in ("scaled_silu", "gate_mean"):
            return self.workspace.bind(
                self._plan_for(operation), hidden_states.shape[0], local=True
            )
        return self.workspace.bind(self._plan_for(operation), hidden_states.shape[0])

    def _mix_normalized(self, normalized: torch.Tensor):
        if self.tp_size > 1:
            return self._mix_sharded(normalized)
        api = _hyperconnection_api() if self.workspace is not None else None
        if self.use_combine:
            down_and_injection = self.input_mix_weight_down_block_inject(normalized)
            projected_down = down_and_injection[:, : self.lora_rank]
            injection_start = self.lora_rank
            # The projection owner stays live through the downstream residual
            # combine; readers consume row-strided slices without staging.
            injection = down_and_injection[
                :, injection_start : injection_start + self.hc_count
            ]
        else:
            projected_down = self.input_mix_weight_down(normalized)
            injection = None

        bottleneck = (
            api.run_scaled_silu(
                projected_down, binding=self._binding(normalized, "scaled_silu")
            )
            if api is not None
            else hc_silu(projected_down, self.hc_count)
        )
        gate_logits = self.input_mix_weight_up(bottleneck)
        block_input = (
            api.run_gate_mean(
                normalized, gate_logits, binding=self._binding(normalized, "gate_mean")
            )
            if api is not None
            else hc_gate_mix(normalized, gate_logits, self.hc_count)
        )
        return block_input, injection

    def _mix_sharded(self, normalized: torch.Tensor):
        api = _hyperconnection_api()
        workspace = self.workspace
        tokens = normalized.shape[0]
        down = (
            self.input_mix_weight_down_block_inject
            if self.use_combine
            else self.input_mix_weight_down
        )
        width = down.weight.shape[0]
        projected_fp32 = workspace.local_down_fp32.view(-1)[: tokens * width].view(
            tokens, width
        )
        torch.mm(normalized, down.weight.T, out=projected_fp32, out_dtype=torch.float32)
        projected = workspace.local_down.view(-1)[: tokens * width].view(tokens, width)
        projected.copy_(projected_fp32)
        injection = (
            projected[:, self.lora_rank : self.lora_rank + self.hc_count]
            if self.use_combine
            else None
        )
        bottleneck = api.run_scaled_silu(
            projected[:, : self.lora_rank],
            binding=self._binding(normalized, "scaled_silu"),
        )
        bottleneck = tensor_model_parallel_all_gather(bottleneck, dim=-1)
        logits = workspace.local_gates[:tokens]
        torch.mm(bottleneck, self.input_mix_weight_up.weight.T, out=logits)
        local_hidden = self.hidden_size // self.tp_size
        local_normalized = workspace.local_normalized[:tokens]
        local_normalized.view(tokens, self.hc_count, local_hidden).copy_(
            normalized.view(tokens, self.hc_count, self.hidden_size)[
                :, :, self.tp_rank * local_hidden : (self.tp_rank + 1) * local_hidden
            ]
        )
        block_input = api.run_gate_mean(
            local_normalized, logits, binding=self._binding(normalized, "gate_mean")
        )
        return tensor_model_parallel_all_gather(block_input, dim=-1), injection


__all__ = [
    "GatedResidual",
    "GroupedGemmaRMSNorm",
    "HyperConnectionConfig",
]
