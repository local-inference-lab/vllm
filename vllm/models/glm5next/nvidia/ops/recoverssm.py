# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12X recurrent-state recovery with vLLM's cache-boundary metadata."""

from typing import Any, cast

import torch

from vllm.models.kimi_k3.nvidia.ops.recoverssm import KDARecoverSSMCommitContext


class B12XKDARecoverSSMCommitContext(KDARecoverSSMCommitContext):
    _b12x_layer: Any
    _b12x_scratch: torch.Tensor

    @classmethod
    def create(cls, layers, *, spec_query_len, max_num_reqs):
        context = cast(
            "B12XKDARecoverSSMCommitContext",
            super().create(
                layers, spec_query_len=spec_query_len, max_num_reqs=max_num_reqs
            ),
        )
        layer = layers[0]
        if layer._b12x_kda_plan is None:
            raise RuntimeError("B12X KDA recovery requires a prepared decode plan")
        context._b12x_layer = layer
        (spec,) = layer._b12x_kda_plan.scratch_specs()
        context._b12x_scratch = torch.empty(
            spec.shape, dtype=spec.dtype, device=context.checkpoints[0].device
        )
        return context

    def _commit_recurrent_state(self, state_indices, batch, align_mode):
        layer = self._b12x_layer
        api = layer._b12x_kda_api
        binding = api.bind_kda_commit(
            layer._b12x_kda_plan,
            scratch=self._b12x_scratch,
            state_base_addrs=self.state_base_addrs,
            state_block_strides=self.state_block_strides,
            correction_cache_base_addrs=self.correction_cache_base_addrs,
            correction_cache_block_strides=self.correction_cache_block_strides,
            kg_cache_base_addrs=self.kg_cache_base_addrs,
            kg_cache_block_strides=self.kg_cache_block_strides,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            state_indices=state_indices,
            commit_lens=self.commit_lens[:batch],
            final_state_indices=self.final_state_indices[:batch],
            boundary_state_indices=self.boundary_state_indices[:batch],
            boundary_recovery_lens=self.boundary_recovery_lens[:batch],
        )
        api.run_kda_commit(binding, lower_bound=self.lower_bound)
