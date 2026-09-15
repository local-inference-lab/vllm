# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer-held block-scaled linear state shared by the b12x linear kernels.

A holder owns one packed weight and the exact-M ``Plan`` declared for the
serving shapes. The kernel's ``apply_weights`` reaches ``run`` through the
``vllm::b12x_blockscaled_linear`` custom op, so compiled graphs carry only
tensors, the output width, and the layer name. ``run`` resolves the prepared
regime for the live row count and draws its workspace from the worker's
workspace manager; nothing here allocates during graph capture once the
workspace is locked.
"""

from __future__ import annotations

import weakref

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.b12x import (
    B12xPreparationUnit,
    B12xWorkload,
    PreparationResourceUnavailableError,
    get_b12x_blockscaled,
)

COMPONENT = "gemm.blockscaled_precision"
logger = init_logger(__name__)


def _operands(packed, recipe: str):
    if recipe == "nvfp4":
        return packed.values, packed.scale_mma, packed.global_scale, packed.global_scale_kind
    weight = packed.weight
    return weight.values, weight.scale_mma, None, "none"


class B12xBlockscaledLinear:
    def __init__(
        self,
        packed,
        *,
        recipe: str,
        activation_mode: str,
        layer_name: str,
        activation_scale: torch.Tensor | None = None,
    ) -> None:
        if recipe not in ("nvfp4", "mxfp8"):
            raise ValueError("block-scaled linear recipe must be nvfp4 or mxfp8")
        self.packed = packed
        self.recipe = recipe
        self.activation_mode = activation_mode
        self.layer_name = layer_name
        self.activation_scale = activation_scale
        self.plan = None
        self._plan_key = None

    @property
    def out_features(self) -> int:
        return int(self.packed.out_features)

    @property
    def in_features(self) -> int:
        return int(self.packed.in_features)

    @property
    def device(self) -> torch.device:
        return self._operands()[0].device

    def _operands(self):
        return _operands(self.packed, self.recipe)

    def holds(self, packed) -> bool:
        """True when ``packed`` shares the storage this holder's plan was declared on."""
        mine = _operands(self.packed, self.recipe)[:2]
        theirs = _operands(packed, self.recipe)[:2]
        return all(
            a.data_ptr() == b.data_ptr() and a.shape == b.shape and a.dtype == b.dtype
            for a, b in zip(mine, theirs)
        )

    def signature(self, workload: B12xWorkload):
        return (
            self.recipe, self.activation_mode, self.in_features,
            int(self.packed.padded_in_features), self.out_features,
            self.activation_scale is not None, workload.max_tokens,
            workload.fixed_token_counts, workload.output_dtype,
        )

    def ensure_plan(self, workload: B12xWorkload):
        """Declare the exact-M regimes for the first workload; reuse afterward.

        The plan is never replaced once declared, so a prepared plan stays
        installed. A later workload that asks for more exact-M regimes is
        served by the capacity regime for those counts.
        """
        key = (workload.max_tokens, workload.fixed_token_counts)
        if self.plan is not None:
            if self._plan_key != key:
                if workload.max_tokens != self._plan_key[0]:
                    raise ValueError(
                        f"{self.layer_name}: block-scaled linear capacity changed "
                        f"from {self._plan_key[0]} to {workload.max_tokens}"
                    )
                missing = sorted(set(workload.fixed_token_counts) - set(self._plan_key[1]))
                if missing:
                    logger.warning_once(
                        "%s: exact-M regimes for %s were not declared in the weights "
                        "stage; the capacity regime serves those counts.",
                        self.layer_name, tuple(missing),
                    )
            return self.plan
        api = get_b12x_blockscaled()
        assert api is not None
        _, _, _, global_scale_kind = self._operands()
        query = api.BlockscaledQuery(
            recipe=self.recipe,
            num_tokens=workload.max_tokens,
            in_features=self.in_features,
            padded_in_features=int(self.packed.padded_in_features),
            out_features=self.out_features,
            activation_mode=self.activation_mode,
            activation_scale_available=self.activation_scale is not None,
            global_scale_kind=global_scale_kind,
            source_contiguous=True,
            source_aligned=True,
            workspace_form="provided",
            workspace_nbytes=envs.VLLM_B12X_BLOCKSCALED_WORKSPACE_MAX_BYTES,
            expected_m=None,
        )
        self.plan = api.plan_regimes(query, exact_m=workload.fixed_token_counts)
        self._plan_key = key
        return self.plan

    def _call_factory(self, rows: int):
        values, scales, global_scale, _ = self._operands()
        activation_scale = self.activation_scale
        shared: tuple[weakref.ref, ...] | None = None
        holder = self

        def prepare(state):
            from b12x.preparation import PreparedCall

            nonlocal shared
            tensors = None if shared is None else tuple(ref() for ref in shared)
            if tensors is None or any(tensor is None for tensor in tensors):
                source = torch.empty(
                    (rows, holder.in_features), dtype=torch.bfloat16, device=values.device,
                )
                shared = (weakref.ref(source),)
            else:
                (source,) = tensors
            workspace = (
                torch.empty(state.required_workspace, dtype=torch.uint8, device=values.device)
                if state.required_workspace else None
            )

            def produce() -> None:
                source.fill_(0.125)

            def run() -> None:
                state.run(
                    source, values, scales, global_scale,
                    activation_scale=activation_scale, workspace=workspace,
                )

            return PreparedCall(
                run=run, produce=produce, owners=(values, scales), capture_safe=False,
            )

        return prepare

    def unit(self, workload: B12xWorkload, *, name: str) -> B12xPreparationUnit:
        plan = self.ensure_plan(workload)
        calls = {rows: self._call_factory(rows) for rows in plan.token_counts}
        request = plan.request(name=name, prepare_calls=calls, benchmark_calls=calls)
        return B12xPreparationUnit(
            name=self.recipe.upper(), key=self.signature(workload), requests=(request,),
            stage="weights", autotune=not workload.eager_only,
        )

    def get_workspace_size(self, rows: int) -> int:
        """Scratch bytes a call needs, so a caller can reserve them ahead of time.

        The shared-experts runner reserves this beside the routed experts'
        buffers before either branch runs, because the shared experts execute
        on a side stream and must not draw an overlapping view from the
        workspace manager.
        """
        del rows
        plan = self.plan
        if plan is None:
            raise PreparationResourceUnavailableError(
                f"{self.layer_name}: block-scaled linear has no declared plan"
            )
        if plan.prepared is None:
            return sum(spec.nbytes for spec in plan.scratch_specs())
        from b12x.preparation import require_prepared

        return int(require_prepared(plan, COMPONENT).required_workspace)

    def run(self, source: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        """Execute the prepared regime for ``source``; op-body only."""
        plan = self.plan
        if plan is None:
            raise PreparationResourceUnavailableError(
                f"{self.layer_name}: block-scaled linear has no declared plan"
            )
        from b12x.preparation import require_prepared

        state = require_prepared(plan, COMPONENT, source.device)
        workspace = None
        if state.required_workspace:
            from vllm.v1.worker.workspace import (
                current_preallocated_workspace,
                current_workspace_manager,
            )

            reserved = current_preallocated_workspace()
            if reserved is not None:
                reserved = reserved.view(torch.uint8)
                if reserved.numel() < state.required_workspace:
                    raise ValueError(
                        f"{self.layer_name}: reserved scratch holds {reserved.numel()} "
                        f"bytes, the prepared regime needs {state.required_workspace}"
                    )
                workspace = reserved[: state.required_workspace]
            else:
                (workspace,) = current_workspace_manager().get_simultaneous(
                    ((state.required_workspace,), torch.uint8)
                )
        api = get_b12x_blockscaled()
        assert api is not None
        return api.mm(
            source, self.packed, plan=plan, bias=bias, workspace=workspace,
            activation_global_scale=self.activation_scale,
        )
