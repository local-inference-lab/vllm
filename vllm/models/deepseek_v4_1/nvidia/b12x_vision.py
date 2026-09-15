# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replicated V4.1 ViT and aligner using prepared native b12x compute.

One image is evaluated per call.  The preparation registry installs immutable
rotary, merge, and GELU executions together with fixed owner workspaces before
capture. Attention binds full planned-capacity tensors and a device
[0, live_patches] prefix, so padded/stale rows cannot participate.
"""

from __future__ import annotations

import itertools

import torch
from b12x.attention import varlen
from b12x.gemm import bf16_gemv
from b12x.norm import hyperconnection
from b12x.norm import vision as vision_api
from b12x.norm.vision import VisionQuery, run_gelu, run_rope_qkv, run_spatial_merge
from b12x.preparation import PreparedCall
from torch import nn

from vllm.model_executor.weight_transfer import allocate_weights
from vllm.utils.b12x import (
    set_b12x_preparation_provider,
    B12xPreparationUnit,
    B12xWorkload,
    PreparationResourceUnavailableError,
)


def _capacity(config):
    capacity = getattr(config, "vision_max_tokens", None)
    if capacity is None:
        capacity = config.vision_max_n_token * config.vision_downsample_ratio**2
    if capacity <= 0:
        raise ValueError("vision token capacity must be positive")
    return int(capacity)


class _Linear(nn.Module):
    """Replicated checkpoint layout; bias is accumulated before BF16 rounding."""

    def __init__(self, in_features, out_features, bias=True, *, max_rows=None):
        super().__init__()
        self.weight = nn.Parameter(
            allocate_weights(
                torch.empty, out_features, in_features, dtype=torch.bfloat16
            ),
            requires_grad=False,
        )
        self.bias = (
            nn.Parameter(
                allocate_weights(torch.empty, out_features, dtype=torch.bfloat16),
                requires_grad=False,
            )
            if bias
            else None
        )

        self._plans = {}
        self._max_rows = max_rows
        self._capacity = 0
        set_b12x_preparation_provider(self, self)

    def forward(self, x, *, out):
        plan = self._plans.get(x.shape[0], self._plans[self._capacity])
        return bf16_gemv.mm(
            x, self.weight, plan=plan, bias=self.bias, out=out,
            output_dtype=torch.bfloat16,
        )

    def get_b12x_preparation_units(
        self, layer: nn.Module, workload: B12xWorkload
    ) -> tuple[B12xPreparationUnit, ...]:
        if workload.stage != "weights" or self.weight.is_meta:
            return ()
        self._capacity = workload.max_tokens if self._max_rows is None else self._max_rows
        capacities = tuple(sorted({
            self._capacity, *(rows for rows in workload.token_counts if rows <= self._capacity),
        }))

        def make_call(state):
            offset = int(not state.query.source_aligned)
            source = torch.empty(
                (state.query.max_rows + offset, self.weight.shape[1]),
                dtype=torch.bfloat16, device=self.weight.device,
            )[offset:]
            out = torch.empty(
                (state.query.max_rows, self.weight.shape[0]),
                dtype=torch.bfloat16, device=self.weight.device,
            )
            return PreparedCall(
                run=lambda: state.run(source, self.weight, out=out, bias=self.bias),
                produce=lambda: source.normal_(std=0.25),
                owners=(self.weight, self.bias),
            )

        requests = []
        for capacity in capacities:
            if capacity not in self._plans:
                self._plans[capacity] = bf16_gemv.plan(bf16_gemv.GemvQuery(
                    source_dtype="bfloat16", weight_dtype="bfloat16",
                    max_rows=capacity, in_features=self.weight.shape[1],
                    out_features=self.weight.shape[0], source_contiguous=True,
                    source_aligned=self.weight.shape[1] % 8 == 0,
                    weight_contiguous=self.weight.is_contiguous(),
                    weight_aligned=self.weight.data_ptr() % 16 == 0,
                    bias_dtype=None if self.bias is None else "bfloat16",
                ))
            requests.append(self._plans[capacity].request(
                name=f"deepseek_v41.vision.linear.{id(self):x}.m{capacity}",
                prepare_call=make_call, benchmark_call=make_call,
            ))
        return (B12xPreparationUnit(
            name="V41VisionLinear", key=(id(self), capacities),
            requests=tuple(requests), stage="weights", autotune=not workload.eager_only,
        ),)


class _Norm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(
            allocate_weights(torch.ones, dim, dtype=torch.bfloat16), requires_grad=False
        )

    def forward(self, x, *, out, plan):
        binding = hyperconnection.bind(
            plan,
            normalized=out,
            bottleneck=x.view(-1)[: x.shape[0]].view(-1, 1),
            block_input=x,
            tokens=x.shape[0],
        )
        return hyperconnection.run_grouped_rmsnorm(
            x, self.weight, eps=1e-6, zero_centered=False, binding=binding
        )


class _PatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = _Linear(
            3 * config.vision_patch_size**2, config.vision_dim, max_rows=_capacity(config),
        )

    def forward(self, patches, *, out):
        return self.proj(patches.flatten(1), out=out)


class _Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wqkv = _Linear(
            config.vision_dim, 3 * config.vision_dim, max_rows=_capacity(config),
        )
        self.wo = _Linear(
            config.vision_dim, config.vision_dim, max_rows=_capacity(config),
        )

    def forward(self, x, height, width, workspace, *, out):
        rows = x.shape[0]
        qkv = self.wqkv(x, out=workspace.qkv[:rows])
        run_rope_qkv(
            qkv,
            height,
            width,
            workspace.inv_freq,
            q=workspace.q,
            k=workspace.k,
            v=workspace.v,
            cu_seqlens=workspace.cu,
            plan=workspace.rope_plan,
        )
        attended, _ = varlen.run(binding=workspace.attention_binding)
        return self.wo(attended[:rows].reshape(rows, -1), out=out)


class _MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = _Linear(
            config.vision_dim, 2 * config.vision_inter_dim, False, max_rows=_capacity(config),
        )
        self.w2 = _Linear(
            config.vision_inter_dim, config.vision_dim, False, max_rows=_capacity(config),
        )

    def forward(self, x, workspace, *, out):
        rows = x.shape[0]
        gate_up = self.w1(x, out=workspace.gate_up[:rows])
        activated = hyperconnection.run_swiglu(
            gate_up, limit=float("inf"), round_silu=True, out=workspace.activated[:rows],
            plan=workspace.swiglu_plan,
        )
        return self.w2(activated, out=out)


class _Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1, self.norm2 = _Norm(config.vision_dim), _Norm(config.vision_dim)
        self.attn, self.mlp = _Attention(config), _MLP(config)

    def forward(self, x, height, width, workspace):
        rows = x.shape[0]
        norm, branch = workspace.normalized[:rows], workspace.branch[:rows]
        self.norm1(x, out=norm, plan=workspace.norm_plan)
        self.attn(norm, height, width, workspace, out=branch)
        residual = hyperconnection.run_add(
            x, branch, out=workspace.residual[:rows], plan=workspace.add_plan
        )
        self.norm2(residual, out=norm, plan=workspace.norm_plan)
        self.mlp(norm, workspace, out=branch)
        return hyperconnection.run_add(residual, branch, out=x, plan=workspace.add_plan)


class _VisionWorkspace:
    def __init__(self, config, capacity, device):
        hidden, inter = config.vision_dim, config.vision_inter_dim
        heads = config.vision_n_heads
        dim = hidden // heads

        def empty(*shape):
            return torch.empty(shape, dtype=torch.bfloat16, device=device)

        self.state, self.normalized, self.branch = (
            empty(capacity, hidden) for _ in range(3)
        )
        self.residual = empty(capacity, hidden)
        self.qkv = empty(capacity, 3 * hidden)
        self.q, self.k, self.v = (empty(capacity, heads, dim) for _ in range(3))
        self.gate_up, self.activated = (
            empty(capacity, 2 * inter),
            empty(capacity, inter),
        )
        # This is static head geometry, not a table keyed by live image grids.
        inv = 1.0 / (
            config.vision_rope_theta
            ** (
                torch.arange(0, dim // 2, 2, dtype=torch.float32, device="cpu")
                / (dim // 2)
            )
        )
        self.inv_freq = inv.to(device=device)
        self.cu = torch.zeros(2, dtype=torch.int32, device=device)
        self.scratch = None
        self.attention_binding = None


class DeepseekV4ViT(nn.Module):
    """Reference 2D split-half RoPE and full bidirectional attention per image."""

    def __init__(self, config):
        super().__init__()
        if (
            config.vision_dim % config.vision_n_heads
            or (config.vision_dim // config.vision_n_heads) % 4
        ):
            raise ValueError("vision head dimension must be divisible by four")
        self.config, self.capacity = config, _capacity(config)
        self.rope_dim = config.vision_dim // config.vision_n_heads // 2
        self.rope_theta = config.vision_rope_theta
        self.patch_embed = _PatchEmbed(config)
        self.blocks = nn.ModuleList(
            [_Block(config) for _ in range(config.vision_n_layers)]
        )
        self.norm = _Norm(config.vision_dim)
        self._workspace = None
        self._rope_plan = None
        self._attention_plan = None
        self._norm_plan = None
        self._add_plan = None
        self._swiglu_plan = None
        set_b12x_preparation_provider(self, self)
    def _provision_workspace(self):
        """Create fixed serving buffers only while the session is priming us."""
        device = self.patch_embed.proj.weight.device
        if device.type != "cuda":
            raise ValueError("native vision requires CUDA weights")
        if self._workspace is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("vision preparation cannot allocate during CUDA graph capture")
            self._workspace = _VisionWorkspace(self.config, self.capacity, device)
            self._workspace.rope_plan = self._rope_plan
            self._workspace.norm_plan = self._norm_plan
            self._workspace.add_plan = self._add_plan
            self._workspace.swiglu_plan = self._swiglu_plan
        elif self._workspace.state.device != device:
            raise ValueError("vision moved devices after preparation")
        return self._workspace

    def get_b12x_preparation_units(
        self, layer: nn.Module, workload: B12xWorkload
    ) -> tuple[B12xPreparationUnit, ...]:
        if workload.stage != "weights" or any(parameter.is_meta for parameter in self.parameters()):
            return ()
        if self._rope_plan is None:
            rope = VisionQuery(
                operation="rope",
                channels=self.config.vision_dim,
                heads=self.config.vision_n_heads,
                ratio=1,
            )
            self._rope_plan = vision_api.plan(
                rope, device=self.patch_embed.proj.weight.device
            )

        device = self.patch_embed.proj.weight.device
        if self._attention_plan is None:
            from torch._subclasses.fake_tensor import FakeTensorMode

            heads = self.config.vision_n_heads
            with FakeTensorMode():
                q, k, v = (
                    torch.empty(
                        (self.capacity, heads, self.config.vision_dim // heads),
                        dtype=torch.bfloat16, device=device,
                    ) for _ in range(3)
                )
                cu = torch.empty(2, dtype=torch.int32, device=device)
            self._attention_plan = varlen.plan(
                q, k, v, cu, max_seqlen_q=self.capacity,
                max_seqlen_k=self.capacity, causal=False,
            )
        for attribute, width, invocation in (
            ("_norm_plan", self.config.vision_dim, {
                "operation": "grouped_rmsnorm", "zero_centered": False,
                "weight_dtype": "bfloat16", "eps": 1e-6,
            }),
            ("_add_plan", self.config.vision_dim, {"operation": "add"}),
            ("_swiglu_plan", self.config.vision_inter_dim, {
                "operation": "swiglu", "round_silu": True,
            }),
        ):
            if getattr(self, attribute) is None:
                setattr(self, attribute, hyperconnection.plan(
                    hyperconnection.Caps(
                        device=device, max_tokens=self.capacity, hidden_size=width,
                        streams=1, lowrank=1,
                    ),
                    invocation=invocation,
                ))

        def attention_call(state, *, serving=False):
            if serving:
                workspace = self._provision_workspace()
                q, k, v, cu = workspace.q, workspace.k, workspace.v, workspace.cu
            else:
                q, k, v = (
                    torch.empty(state.plan.q_shape, dtype=torch.bfloat16, device=device)
                    for _ in range(3)
                )
                cu = torch.empty(2, dtype=torch.int32, device=device)
            (spec,) = self._attention_plan.scratch_specs()
            scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
            binding = state.bind(
                plan=self._attention_plan, scratch=scratch, q=q, k=k, v=v,
                cu_seqlens_q=cu, max_seqlen_q=self.capacity,
                max_seqlen_k=self.capacity, causal=False,
            )
            if serving:
                workspace.scratch, workspace.attention_binding = scratch, binding

            def produce():
                for tensor in (q, k, v):
                    tensor.normal_(std=0.25)
                cu[0].zero_()
                cu[1].fill_(self.capacity)

            return PreparedCall(
                run=lambda: state.run(binding), produce=produce,
                owners=(q, k, v, cu, scratch),
            )

        def norm_call(state):
            from b12x.norm.hyperconnection._impl import run_grouped_rmsnorm_impl

            source = torch.empty(
                (self.capacity, self.config.vision_dim), dtype=torch.bfloat16, device=device,
            )
            out = torch.empty_like(source)
            return PreparedCall(
                run=lambda: run_grouped_rmsnorm_impl(
                    source, self.norm.weight, eps=1e-6, zero_centered=False,
                    plan=state, out=out,
                ),
                produce=lambda: source.normal_(std=0.25), owners=(self.norm.weight,),
            )

        def pointwise_call(state):
            from b12x.norm.hyperconnection._impl import run_add_impl, run_swiglu_impl

            width = state.query.hidden_size
            swiglu = state.query.operation == "swiglu"
            left = torch.empty(
                (self.capacity, width * (2 if swiglu else 1)),
                dtype=torch.bfloat16, device=device,
            )
            right, out = (
                torch.empty((self.capacity, width), dtype=torch.bfloat16, device=device)
                for _ in range(2)
            )

            def produce():
                left.normal_(std=0.25)
                right.normal_(std=0.25)

            return PreparedCall(
                run=(lambda: run_swiglu_impl(
                    left, limit=float("inf"), round_silu=True, plan=state, out=out,
                )) if swiglu else (lambda: run_add_impl(left, right, plan=state, out=out)),
                produce=produce,
            )

        def call(state):
            workspace = self._provision_workspace()
            rows = min(self.capacity, 1)
            return PreparedCall(
                run=lambda: vision_api._run_state(
                    state, workspace.qkv[:rows], workspace.q, rows, 1, rows,
                    k=workspace.k, v=workspace.v, inv=workspace.inv_freq, cu=workspace.cu,
                ),
                reset=lambda: workspace.qkv[:rows].zero_(),
                owners=(workspace.qkv, workspace.q, workspace.k, workspace.v, workspace.inv_freq, workspace.cu),
            )

        base = f"deepseek_v41.vision.{id(self):x}"
        requests = (
            self._rope_plan.request(
                name=f"{base}.rope", prepare_call=call, benchmark_call=call,
            ),
            self._attention_plan.request(
                name=f"{base}.attention",
                prepare_call=lambda state: attention_call(state, serving=True),
                benchmark_call=attention_call,
            ),
            self._norm_plan.request(
                name=f"{base}.norm", prepare_call=norm_call, benchmark_call=norm_call,
            ),
            self._add_plan.request(
                name=f"{base}.add", prepare_call=pointwise_call, benchmark_call=pointwise_call,
            ),
            self._swiglu_plan.request(
                name=f"{base}.swiglu", prepare_call=pointwise_call, benchmark_call=pointwise_call,
            ),
        )
        return (B12xPreparationUnit(
            name="DeepseekV4ViT",
            key=(id(self), self.config.vision_dim, self.config.vision_n_heads),
            requests=requests, stage="weights", autotune=not workload.eager_only,
        ),)

    def forward(self, patches, n_vit_h: int, n_vit_w: int, *, out=None):
        rows = n_vit_h * n_vit_w
        if (
            n_vit_h <= 0
            or n_vit_w <= 0
            or rows != patches.shape[0]
            or rows > self.capacity
        ):
            raise ValueError(
                "vision image grid exceeds capacity or does not match patches"
            )
        if patches.dtype != torch.bfloat16 or not patches.is_contiguous():
            raise ValueError("vision patches must be contiguous BF16")
        if (
            self._workspace is None
            or self._rope_plan is None
            or any(plan is None or plan.prepared is None for plan in (
                self._rope_plan, self._attention_plan, self._norm_plan,
                self._add_plan, self._swiglu_plan,
            ))
        ):
            raise RuntimeError("vision plan was not prepared")
        ws = self._workspace
        if out is None:
            out = torch.empty(
                (rows, self.config.vision_dim),
                dtype=torch.bfloat16,
                device=patches.device,
            )
        x = self.patch_embed(patches, out=ws.state[:rows])
        for block in self.blocks:
            x = block(x, n_vit_h, n_vit_w, ws)
        return self.norm(x, out=out, plan=ws.norm_plan)


class DeepseekV4Aligner(nn.Module):
    """Channel-major padded spatial merge, biased projections, exact GELU."""

    def __init__(self, config):
        super().__init__()
        self.downsample_ratio = config.vision_downsample_ratio
        if self.downsample_ratio <= 0:
            raise ValueError("vision downsample ratio must be positive")
        self.out_dim = config.hidden_size
        self.hidden_size, self.capacity = config.vision_dim, _capacity(config)
        capacity = (self.capacity + self.downsample_ratio - 1) // self.downsample_ratio
        self.w1 = _Linear(
            config.vision_dim * self.downsample_ratio**2, self.out_dim, max_rows=capacity,
        )
        self.w2 = _Linear(self.out_dim, self.out_dim, max_rows=capacity)
        self._workspace = None
        self._merge_plan = None
        self._gelu_plan = None
        set_b12x_preparation_provider(self, self)
    def _provision_workspace(self):
        device = self.w1.weight.device
        if device.type != "cuda":
            raise ValueError("native aligner requires CUDA weights")
        if self._workspace is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("aligner preparation cannot allocate during CUDA graph capture")
            r = self.downsample_ratio
            capacity = (self.capacity + r - 1) // r
            self._workspace = (
                torch.empty((self.capacity, self.hidden_size), device=device, dtype=torch.bfloat16),
                *(torch.empty((capacity, width), device=device, dtype=torch.bfloat16)
                  for width in (self.hidden_size * r * r, self.out_dim, self.out_dim)),
            )
        elif self._workspace[0].device != device:
            raise ValueError("aligner moved devices after preparation")
        return self._workspace

    def get_b12x_preparation_units(
        self, layer: nn.Module, workload: B12xWorkload
    ) -> tuple[B12xPreparationUnit, ...]:
        if any(parameter.is_meta for parameter in self.parameters()):
            return ()
        device = self.w1.weight.device
        if self._merge_plan is None:
            merge = VisionQuery(
                operation="merge", channels=self.hidden_size, heads=1,
                ratio=self.downsample_ratio,
            )
            self._merge_plan = vision_api.plan(merge, device=device)
        if self._gelu_plan is None:
            gelu = VisionQuery(operation="gelu", channels=self.out_dim, heads=1, ratio=1)
            self._gelu_plan = vision_api.plan(gelu, device=device)

        def merge_call(state):
            source, merged, _, _ = self._provision_workspace()
            return PreparedCall(
                run=lambda: vision_api._run_state(state, source[:1], merged[:1], 1, 1, 1),
                reset=lambda: source[:1].zero_(),
                owners=(source, merged),
            )

        def gelu_call(state):
            _, _, hidden, activated = self._provision_workspace()
            return PreparedCall(
                run=lambda: vision_api._run_state(state, hidden[:1], activated[:1], 1),
                reset=lambda: hidden[:1].zero_(),
                owners=(hidden, activated),
            )

        base = f"deepseek_v41.aligner.{id(self):x}"
        requests = (
            self._merge_plan.request(
                name=f"{base}.merge", prepare_call=merge_call, benchmark_call=merge_call,
            ),
            self._gelu_plan.request(
                name=f"{base}.gelu", prepare_call=gelu_call, benchmark_call=gelu_call,
            ),
        )
        return (B12xPreparationUnit(
            name="DeepseekV4Aligner",
            key=(id(self), self.hidden_size, self.out_dim, self.downsample_ratio),
            requests=requests, stage="weights", autotune=not workload.eager_only,
        ),)

    def forward(self, x, n_vit_h: int, n_vit_w: int, *, out=None):
        if (
            n_vit_h <= 0
            or n_vit_w <= 0
            or x.shape != (n_vit_h * n_vit_w, self.hidden_size)
            or x.shape[0] > self.capacity
        ):
            raise ValueError(
                "aligner image grid exceeds capacity or does not match input"
            )
        if (
            self._workspace is None
            or self._merge_plan is None
            or self._gelu_plan is None
            or self._merge_plan.prepared is None
            or self._gelu_plan.prepared is None
        ):
            raise RuntimeError("aligner plan was not prepared")
        r = self.downsample_ratio
        rows = ((n_vit_h + r - 1) // r) * ((n_vit_w + r - 1) // r)
        _, merged_buffer, hidden_buffer, activated_buffer = self._workspace
        merged, hidden, activated = (
            merged_buffer[:rows], hidden_buffer[:rows], activated_buffer[:rows]
        )
        run_spatial_merge(
            x, n_vit_h, n_vit_w, ratio=r, out=merged,
            plan=self._merge_plan,
        )
        self.w1(merged, out=hidden)
        run_gelu(hidden, out=activated, plan=self._gelu_plan)
        if out is None:
            out = torch.empty(
                (rows, self.out_dim), dtype=torch.bfloat16, device=x.device
            )
        return self.w2(activated, out=out)



def run_dp_sharded_vision_tower(vision_model, aligner, patches, vit_grid):
    """Shard images across replicated TP ranks and perform a real all-gather."""
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
        tensor_model_parallel_all_gather,
    )
    from vllm.model_executor.models.vision import get_load_balance_assignment

    if not vit_grid:
        if patches.shape[0] != 0:
            raise ValueError("nonempty patches require image grids")
        return []
    sizes = [h * w for h, w in vit_grid]
    if any(h <= 0 or w <= 0 for h, w in vit_grid) or sum(sizes) != patches.shape[0]:
        raise ValueError("image grids do not partition patches")
    world, rank = (
        get_tensor_model_parallel_world_size(),
        get_tensor_model_parallel_rank(),
    )
    offsets = [0, *itertools.accumulate(sizes)]
    assignments, counts, _ = get_load_balance_assignment(sizes, world)
    rank_offsets = [0, *itertools.accumulate(counts)]
    r = aligner.downsample_ratio
    output_rows = [((h + r - 1) // r) * ((w + r - 1) // r) for h, w in vit_grid]

    def images(g):
        return assignments[rank_offsets[g] : rank_offsets[g + 1]]

    max_rows = max(sum(output_rows[i] for i in images(g)) for g in range(world))
    local = torch.zeros(
        (max_rows, aligner.out_dim), dtype=patches.dtype, device=patches.device
    )
    offset = 0
    for i in images(rank):
        h, w = vit_grid[i]
        encoded = vision_model(patches[offsets[i] : offsets[i + 1]], h, w)
        aligner(encoded, h, w, out=local[offset : offset + output_rows[i]])
        offset += output_rows[i]
    gathered = tensor_model_parallel_all_gather(local, dim=0) if world > 1 else local
    result = [None] * len(vit_grid)
    for g in range(world):
        offset = g * max_rows
        for i in images(g):
            result[i] = gathered[offset : offset + output_rows[i]]
            offset += output_rows[i]
    return result
