# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3 KDA modeling adapter.

Decode-latency variant of the shared KDA forward: the two low-rank output-gate
projections (``g_a_proj`` then ``g_b_proj``) depend only on the layer input, so
they are issued on a side CUDA stream and overlap the large fused
``in_proj_qkvgfab`` GEMM instead of trailing it on the main stream. Kernels,
shapes and reduction orders are unchanged, so results are bitwise identical to
the sequential path; only the stream fork/join edges are new. The fork/join
uses ``wait_stream``, which CUDA graph capture records as dependency edges.
``VLLM_GLM53_KDA_GATE_SIDE_STREAM=0`` restores the sequential forward.
"""

import os
from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from einops import rearrange

from vllm.config import VllmConfig
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    KimiGatedDeltaNetAttention,
)
from vllm.models.kimi_k3.nvidia.kda_metadata import (
    KDARecoverSSMAlignMetadata,
    KDARecoverSSMCommitMetadata,
    KimiK3KDAMetadata,
)
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import AttentionBackend, CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    use_preallocated_workspace,
)

if TYPE_CHECKING:
    from vllm.models.kimi_k3.nvidia.ops.recoverssm import (
        KDARecoverSSMCommitContext,
    )


class Glm5NextKDAMetadataBuilder(GDNAttentionMetadataBuilder):
    supports_varlen_decode_cudagraph = True

    def __init__(
        self,
        kv_cache_spec: MambaSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # Adaptive verification supplies packed boundaries on device, while the
        # CPU lengths describe an even distribution of the same token budget.
        self._reuse_spec_decode_inputs = False


class Glm5NextKDAAttentionBackend(GDNAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "GLM5NEXT_KDA"

    @staticmethod
    def get_builder_cls() -> type[Glm5NextKDAMetadataBuilder]:
        return Glm5NextKDAMetadataBuilder

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        return True


@dataclass
class Glm5NextRecoverKDAMetadata(KimiK3KDAMetadata):
    """GLM prefill metadata with KDA speculative-state recovery."""


class Glm5NextRecoverKDAMetadataBuilder(Glm5NextKDAMetadataBuilder):
    supports_update_block_table = False

    def __init__(
        self,
        kv_cache_spec: MambaSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._neutral_accepted = torch.ones(
            vllm_config.scheduler_config.max_num_seqs,
            dtype=torch.int32,
            device=device,
        )
        self._recoverssm_context: KDARecoverSSMCommitContext | None = None

    def _get_recoverssm_context(self) -> "KDARecoverSSMCommitContext":
        if self._recoverssm_context is None:
            from vllm.models.glm5next.nvidia.ops.recoverssm import (
                B12XKDARecoverSSMCommitContext,
            )

            layers = self.vllm_config.compilation_config.static_forward_context
            self._recoverssm_context = B12XKDARecoverSSMCommitContext.create(
                [layers[name] for name in self.layer_names],
                spec_query_len=self.num_spec + 1,
                max_num_reqs=self.vllm_config.scheduler_config.max_num_seqs,
            )
        return self._recoverssm_context

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> Glm5NextRecoverKDAMetadata:
        common = common_attn_metadata
        if num_decode_draft_tokens_cpu is not None:
            assert common.is_prefilling is not None
            active_decodes = (~common.is_prefilling) & (
                common.query_start_loc_cpu.diff() > 0
            )
            missing = active_decodes & (num_decode_draft_tokens_cpu < 0)
            if bool(torch.any(missing)):
                num_decode_draft_tokens_cpu = num_decode_draft_tokens_cpu.clone()
                num_decode_draft_tokens_cpu[missing] = 0
        metadata = super().build(
            common_prefix_len,
            common,
            num_accepted_tokens=self._neutral_accepted[: common.num_reqs],
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
            fast_build=fast_build,
        )
        commit = None
        if metadata.num_spec_decodes > 0:
            assert metadata.spec_sequence_masks_cpu is not None
            assert metadata.spec_state_indices_tensor is not None
            assert metadata.spec_query_start_loc is not None
            request_indices = async_tensor_h2d(
                metadata.spec_sequence_masks_cpu.nonzero(as_tuple=True)[0],
                dtype=torch.int32,
                device=common.query_start_loc.device,
            )
            align = None
            if self.kv_cache_spec.mamba_cache_mode == "align":
                align = KDARecoverSSMAlignMetadata(
                    block_table=common.block_table_tensor,
                    num_computed_tokens=common.compute_num_computed_tokens(),
                    block_size=self.kv_cache_spec.block_size,
                )
            commit = KDARecoverSSMCommitMetadata(
                state_indices=metadata.spec_state_indices_tensor,
                query_start_loc=metadata.spec_query_start_loc,
                request_indices=request_indices,
                align=align,
            )
        return Glm5NextRecoverKDAMetadata(
            **vars(metadata),
            recoverssm_commit=commit,
            recoverssm_context=(
                self._get_recoverssm_context() if commit is not None else None
            ),
        )


class Glm5NextRecoverKDAAttentionBackend(Glm5NextKDAAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "GLM5NEXT_RECOVER_KDA"

    @staticmethod
    def get_builder_cls() -> type[Glm5NextRecoverKDAMetadataBuilder]:
        return Glm5NextRecoverKDAMetadataBuilder


_GATE_SIDE_STREAM = os.getenv("VLLM_GLM53_KDA_GATE_SIDE_STREAM", "1") != "0"
_side_streams: dict[int, torch.cuda.Stream] = {}


def _gate_overlap_allowed() -> bool:
    """Use the gate side stream only in runtime or regular FULL capture.

    Breakable capture alternates captured segments with eager operations.  A
    side-stream fork in either portion can outlive a segment boundary even when
    the main stream has enqueued a later event wait.  Graph preparation also
    runs uncaptured warmup forwards; overlapping multiple auxiliary streams in
    those forwards can race allocator reuse before the following capture.
    Regular FULL capture and serving execution retain the overlapped path.
    """
    try:
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
        from vllm.compilation.monitor import is_cudagraph_capturing_enabled

        if BreakableCUDAGraphCapture.current() is not None:
            return False
        return (
            not is_cudagraph_capturing_enabled()
            or torch.cuda.is_current_stream_capturing()
        )
    except Exception:  # noqa: BLE001
        return not torch.cuda.is_current_stream_capturing()


def _gate_stream(device: torch.device) -> torch.cuda.Stream:
    index = (
        device.index
        if device.index is not None
        else torch.accelerator.current_device_index()
    )
    stream = _side_streams.get(index)
    if stream is None:
        stream = torch.cuda.Stream(device=torch.device("cuda", index))
        _side_streams[index] = stream
    return stream


class Glm5NextLinearAttention(KimiGatedDeltaNetAttention):
    """Adapt the shared out-buffer KDA layer to GLM's tensor-returning block."""

    enable_b12x_kda_decode = True
    b12x_kda_null_state_index = 0

    def __init__(self, config, vllm_config: VllmConfig, prefix: str = "") -> None:
        quant_config = vllm_config.quant_config
        if (
            quant_config is not None
            and quant_config.get_name() == "fp8"
            and getattr(quant_config, "is_checkpoint_fp8_serialized", False)
        ):
            # Native GLM FP8 exports keep KDA weights in BF16. Their unfused
            # exclusion names do not always match this fused adapter. ModelOpt
            # exports describe each projection independently and must retain
            # that quantization contract. Do not mutate the model-wide config.
            vllm_config = copy(vllm_config)
            vllm_config.quant_config = None
        super().__init__(config, vllm_config, prefix)

    def get_attn_backend(self) -> type[AttentionBackend]:
        if self.cache_config.use_kda_recoverssm:
            return Glm5NextRecoverKDAAttentionBackend
        if (
            self.speculative_config is not None
            and self.speculative_config.enable_adaptive_verification
            and self.cache_config.mamba_cache_mode == "align"
        ):
            return Glm5NextKDAAttentionBackend
        return super().get_attn_backend()

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.empty_like(hidden_states)
        if _GATE_SIDE_STREAM and not self.use_full_rank_gate:
            # Reserve both branches even in serialized graph warmup, before
            # workspace growth is forbidden by capture.
            gate_workspace, main_workspace = self._projection_workspaces(
                hidden_states.size(0)
            )
            if _gate_overlap_allowed():
                self._forward_gate_overlap(
                    hidden_states, output, gate_workspace, main_workspace
                )
                return output
        super().forward(hidden_states, positions, output)
        return output

    def _projection_workspaces(
        self, num_tokens: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        def size(projection) -> int:
            report = getattr(
                getattr(projection, "quant_method", None), "get_workspace_size", None
            )
            return 0 if report is None else report(projection, num_tokens)

        gate_bytes = max(size(self.g_a_proj), size(self.g_b_proj))
        main_bytes = max(size(self.in_proj_qkvgfab), size(self.f_b_proj))
        if not (gate_bytes or main_bytes):
            return None, None
        gate_workspace, main_workspace = current_workspace_manager().get_simultaneous(
            ((gate_bytes,), torch.uint8), ((main_bytes,), torch.uint8)
        )
        return gate_workspace, main_workspace

    def _forward_gate_overlap(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        gate_workspace: torch.Tensor | None,
        main_workspace: torch.Tensor | None,
    ) -> None:
        num_tokens = hidden_states.size(0)
        main = torch.cuda.current_stream(hidden_states.device)
        side = _gate_stream(hidden_states.device)

        # Fork: the gate projections read only hidden_states.
        side.wait_stream(main)
        # Keep hidden_states alive for the side stream (allocator safety in
        # eager/warmup mode; a no-op inside graph capture).
        hidden_states.record_stream(side)
        with torch.cuda.stream(side), use_preallocated_workspace(gate_workspace):
            g_a_states = self.g_a_proj(hidden_states)[0]
            # Some linear backends allocate their caller-owned output before
            # entering the active CUDA stream. Record the asynchronous consumer
            # explicitly so the caching allocator cannot recycle this
            # intermediate while g_b_proj is still reading it.
            g_a_states.record_stream(side)
            g_proj_states = self.g_b_proj(g_a_states)[0]

        with use_preallocated_workspace(main_workspace):
            projected_qkvgfab = self.in_proj_qkvgfab(hidden_states)[0]
            # Same optional callback the shared forward offers after its first
            # projection (e.g. the L2 weight prefetch of o_proj).
            hook = getattr(self, "_l2_prefetch_hook", None)
            if hook is not None:
                hook(num_tokens)
            mixed_qkv, beta, f_a = projected_qkvgfab.split(
                [
                    3 * self.local_projection_size,
                    self.local_num_heads,
                    self.head_dim,
                ],
                dim=-1,
            )
            g1 = self.f_b_proj(f_a)[0]
        beta = beta.unsqueeze(0)
        g1 = rearrange(g1, "n (h d) -> 1 n h d", d=self.head_dim)

        # Join before anything reads the gate states.
        main.wait_stream(side)
        g_proj_states.record_stream(main)
        g2 = rearrange(g_proj_states, "... (h d) -> ... h d", d=self.head_dim)

        core_attn_out = torch.empty(
            (1, num_tokens, self.local_num_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        self._forward(
            mixed_qkv=mixed_qkv,
            g1=g1,
            g2=g2,
            beta=beta,
            core_attn_out=core_attn_out,
        )
        core_attn_out = rearrange(core_attn_out, "1 n h d -> n (h d)")
        output[:] = self.o_proj(core_attn_out)[0]


__all__ = ["Glm5NextLinearAttention"]
