# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepared BF16 linear projections with b12x-owned backend selection."""

from __future__ import annotations

import weakref

import torch

from vllm.utils.b12x import B12xPreparationUnit

_OWNERS: weakref.WeakValueDictionary[str, B12xUnquantizedLinear] = (
    weakref.WeakValueDictionary()
)


class B12xUnquantizedLinear:
    def __init__(self, layer):
        from b12x.gemm import bf16_gemv

        self.projection = bf16_gemv
        self.weight = layer.weight
        self.bias = None if layer.skip_bias_add else layer.bias
        self.name = layer.prefix
        self.plans = {}
        _OWNERS[self.name] = self

    def get_b12x_preparation_units(self, layer, workload):
        from b12x.preparation import PreparedCall

        if workload.stage != "weights":
            return ()
        requests = []
        for rows in sorted(set(workload.token_counts)):
            if rows not in self.plans:
                self.plans[rows] = self.projection.plan(
                    self.projection.GemvQuery(
                        source_dtype="bfloat16",
                        weight_dtype="bfloat16",
                        max_rows=rows,
                        in_features=self.weight.shape[1],
                        out_features=self.weight.shape[0],
                        source_contiguous=True,
                        source_aligned=True,
                        weight_contiguous=True,
                        weight_aligned=True,
                        bias_dtype=None if self.bias is None else "bfloat16",
                    )
                )

            def prepare(state, count=rows):
                source = torch.empty(
                    (count, self.weight.shape[1]),
                    device=self.weight.device,
                    dtype=torch.bfloat16,
                )
                return PreparedCall(
                    run=lambda: state.run(source, self.weight, bias=self.bias),
                    produce=lambda: source.fill_(0.125),
                    owners=(self.weight, self.bias),
                )

            requests.append(
                self.plans[rows].request(
                    name=f"{self.name}.bf16.m{rows}.lane{workload.lane}",
                    prepare_call=prepare,
                    benchmark_call=prepare,
                )
            )
        return (
            B12xPreparationUnit(
                name="BF16_LINEAR",
                key=(self.name, tuple(sorted(self.plans))),
                requests=tuple(requests),
                stage="weights",
                autotune=not workload.eager_only,
            ),
        )

    def run(self, source, weight, bias):
        rows = source.numel() // weight.shape[1]
        capacity = min((n for n in self.plans if n >= rows), default=None)
        if capacity is None:
            raise RuntimeError(
                f"BF16 projection {self.name} has no prepared capacity for {rows} rows"
            )
        return self.projection.mm(
            source.reshape(rows, weight.shape[1]),
            weight,
            bias=bias,
            plan=self.plans[capacity],
        ).reshape(*source.shape[:-1], weight.shape[0])

    def apply(self, source, weight, bias):
        return torch.ops.vllm.b12x_unquantized_linear(source, weight, bias, self.name)


@torch.library.custom_op("vllm::b12x_unquantized_linear", mutates_args=())
def b12x_unquantized_linear(
    source: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, owner: str
) -> torch.Tensor:
    return _OWNERS[owner].run(source, weight, bias)


@b12x_unquantized_linear.register_fake
def _fake(source, weight, bias, owner):
    return source.new_empty((*source.shape[:-1], weight.shape[0]))
