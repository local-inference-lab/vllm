# SPDX-License-Identifier: Apache-2.0
"""Load a Kimi-K3 draft model on a dedicated single GPU.

The process loads either DSpark or DFlash plus the target embedding and LM
head, without loading the Kimi-K3 target transformer. Draft weights, KV cache,
attention workspace, and proposal compute stay on this process's GPU.
"""

from __future__ import annotations

import argparse
import json
import signal
import threading
import time
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors import safe_open

from vllm.config.vllm import set_current_vllm_config
from vllm.engine.arg_utils import EngineArgs
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.worker.workspace import init_workspace_manager

logger = init_logger(__name__)

EMBED_TENSOR = "language_model.model.embed_tokens.weight"
LM_HEAD_TENSOR = "language_model.lm_head.weight"
SHARED_TENSORS = (EMBED_TENSOR, LM_HEAD_TENSOR)


@dataclass
class RuntimeStatus:
    phase: str = "starting"
    ready: bool = False
    proposal_transport_ready: bool = False
    device: str = ""
    compute_capability: str = ""
    torch_version: str = torch.__version__
    torch_arches: tuple[str, ...] = ()
    draft_model: str = ""
    method: str = ""
    target_weights: str = ""
    num_speculative_tokens: int = 0
    max_model_len: int = 0
    allocated_gib: float = 0.0
    reserved_gib: float = 0.0
    draft_kv_cache_gib: float = 0.0
    draft_kv_cache_blocks: int = 0
    draft_kv_cache_token_capacity: int = 0
    draft_kv_cache_smoke: bool = False
    draft_kv_window: int = 0
    proposal_address: str | None = None
    proposal_count: int = 0
    proposal_active_requests: int = 0
    proposal_last_latency_ms: float = 0.0
    smoke_token: int | None = None
    load_seconds: float = 0.0
    error: str | None = None


class _TargetLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            vocab_size,
            hidden_size,
            params_dtype=torch.bfloat16,
            prefix="model.embed_tokens",
            disable_tp=True,
        )


class StandaloneTargetFacade(nn.Module):
    """Only the two frozen target modules that DSpark shares."""

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.model = _TargetLanguageModel(vocab_size, hidden_size)
        self.lm_head = ParallelLMHead(
            vocab_size,
            hidden_size,
            params_dtype=torch.bfloat16,
            prefix="lm_head",
            disable_tp=True,
        )

    def get_language_model(self) -> StandaloneTargetFacade:
        return self


@dataclass
class StandaloneRuntime:
    vllm_config: Any
    draft_vllm_config: Any
    target_facade: StandaloneTargetFacade
    model: nn.Module
    kv_caches: dict[str, torch.Tensor]
    kv_cache_block_size: int
    method: str
    attn_metadata_builder: Any | None = None


def resolve_shared_weight_files(target_weights: Path) -> dict[str, Path]:
    """Resolve the two shared tensors through a safetensors index."""
    root = target_weights.resolve()
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Target weight index is missing: {index_path}")

    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"Invalid safetensors weight_map in {index_path}")

    resolved: dict[str, Path] = {}
    for tensor_name in SHARED_TENSORS:
        relative = weight_map.get(tensor_name)
        if not isinstance(relative, str):
            raise KeyError(f"Target tensor is absent from the index: {tensor_name}")
        tensor_path = (root / relative).resolve()
        if not tensor_path.is_relative_to(root):
            raise ValueError(
                f"Target tensor path escapes the checkpoint root: {tensor_path}"
            )
        if not tensor_path.is_file():
            raise FileNotFoundError(f"Target tensor file is missing: {tensor_path}")
        resolved[tensor_name] = tensor_path
    return resolved


def _load_module_weight(
    module: VocabParallelEmbedding,
    tensor_name: str,
    tensor_path: Path,
    device: torch.device,
) -> None:
    logger.info("Loading shared target tensor %s from %s", tensor_name, tensor_path)
    with safe_open(str(tensor_path), framework="pt", device=str(device)) as handle:
        if tensor_name not in set(handle.keys()):
            raise KeyError(f"{tensor_name} is absent from {tensor_path}")
        loaded_weight = handle.get_tensor(tensor_name)
        if loaded_weight.dtype != torch.bfloat16:
            raise TypeError(
                f"{tensor_name} must be bfloat16, got {loaded_weight.dtype}"
            )
        module.weight_loader(module.weight, loaded_weight)
        del loaded_weight
    torch.accelerator.empty_cache()


def load_shared_target_weights(
    target: StandaloneTargetFacade,
    target_weights: Path,
    device: torch.device,
) -> None:
    files = resolve_shared_weight_files(target_weights)
    _load_module_weight(
        target.model.embed_tokens,
        EMBED_TENSOR,
        files[EMBED_TENSOR],
        device,
    )
    _load_module_weight(
        target.lm_head,
        LM_HEAD_TENSOR,
        files[LM_HEAD_TENSOR],
        device,
    )


def _validate_cuda_runtime(device: torch.device) -> tuple[str, tuple[str, ...]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the DSpark container")
    major, minor = torch.cuda.get_device_capability(device)
    expected_arch = f"sm_{major}{minor}"
    arches = tuple(torch.cuda.get_arch_list())
    if expected_arch not in arches:
        raise RuntimeError(
            f"PyTorch does not contain {expected_arch}; compiled arches are {arches}"
        )
    if major < 8:
        raise RuntimeError(
            f"Kimi-K3 DSpark BF16 requires compute capability >= 8.0, got "
            f"{major}.{minor}"
        )
    return f"{major}.{minor}", arches


def _init_single_gpu_distributed() -> None:
    from vllm.distributed.parallel_state import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
        model_parallel_is_initialized,
    )
    from vllm.utils.network_utils import get_open_port

    if model_parallel_is_initialized():
        return
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
        backend="gloo",
    )
    ensure_model_parallel_initialized(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )


def _build_vllm_config(args: argparse.Namespace):
    attention_backend = "TRITON_ATTN" if args.method == "dflash" else "TRITON_MLA"
    speculative_config = {
        "model": str(args.draft_model),
        "method": args.method,
        "num_speculative_tokens": args.num_speculative_tokens,
        "attention_backend": attention_backend,
        "kv_cache_dtype": "bfloat16",
        "draft_sample_method": "greedy",
        "rejection_sample_method": "block",
        "max_model_len": args.max_model_len,
    }
    engine_args = EngineArgs(
        model=str(args.target_config),
        tokenizer_mode="skip",
        skip_tokenizer_init=True,
        dtype="bfloat16",
        kv_cache_dtype="bfloat16",
        max_model_len=args.max_model_len,
        tensor_parallel_size=1,
        decode_context_parallel_size=1,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        block_size=16,
        enable_prefix_caching=False,
        enforce_eager=True,
        compilation_config={"custom_ops": ["none"]},
        kernel_config={
            "ir_op_priority": {
                "rms_norm": ["native"],
                "fused_add_rms_norm": ["native"],
            },
            "linear_backend": "torch",
        },
        load_format="safetensors",
        use_tqdm_on_load=False,
        speculative_config=speculative_config,
    )
    return engine_args.create_engine_config(headless=True)


def _allocate_draft_kv_cache(
    model: nn.Module,
    *,
    method: str,
    block_size: int,
    cache_gib: float,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], int]:
    """Allocate and bind the draft model's private BF16 KV cache."""
    if cache_gib <= 0:
        raise ValueError(f"--draft-kv-cache-gib must be positive, got {cache_gib}")

    if method == "dflash":
        attentions = [layer.self_attn.attn for layer in model.model.layers]
    else:
        attentions = [layer.self_attn for layer in model.model.layers]
    if not attentions:
        raise RuntimeError("K3 draft model does not expose any attention layers")

    if method == "dflash":
        cache_shapes = {
            (
                int(attn.impl.num_kv_heads),
                int(attn.impl.head_size),
            )
            for attn in attentions
        }
        if len(cache_shapes) != 1:
            raise ValueError(f"K3 DFlash layers have mixed KV shapes: {cache_shapes}")
        num_kv_heads, head_size = cache_shapes.pop()
        bytes_per_block_all_layers = (
            len(attentions)
            * block_size
            * num_kv_heads
            * 2
            * head_size
            * torch.tensor([], dtype=torch.bfloat16).element_size()
        )
    else:
        latent_widths = {
            int(attn.kv_lora_rank + attn.qk_rope_head_dim) for attn in attentions
        }
        if len(latent_widths) != 1:
            raise ValueError(
                f"K3 DSpark draft layers have mixed MLA widths: {latent_widths}"
            )
        latent_width = latent_widths.pop()
        bytes_per_block_all_layers = (
            len(attentions)
            * block_size
            * latent_width
            * torch.tensor([], dtype=torch.bfloat16).element_size()
        )
    num_blocks = int(cache_gib * 1024**3) // bytes_per_block_all_layers
    # Block zero is deliberately never assigned to a live request.
    if num_blocks < 2:
        minimum_mib = 2 * bytes_per_block_all_layers / 1024**2
        raise ValueError(
            "The draft KV cache must fit a null block plus one data block; "
            f"minimum={minimum_mib:.2f} MiB"
        )

    caches: dict[str, torch.Tensor] = {}
    for attn in attentions:
        if method == "dflash":
            # TritonAttention exposes logical B,H,N,2D while its preferred
            # physical layout is B,N,H,2D on CUDA.
            physical = torch.zeros(
                (num_blocks, block_size, num_kv_heads, 2 * head_size),
                dtype=torch.bfloat16,
                device=device,
            )
            cache = physical.permute(0, 2, 1, 3)
        else:
            cache = torch.zeros(
                (num_blocks, block_size, latent_width),
                dtype=torch.bfloat16,
                device=device,
            )
        attn.bind_kv_cache(cache)
        caches[attn.layer_name] = cache
    return caches, num_blocks


def _build_dflash_metadata_builder(
    model: nn.Module,
    draft_vllm_config: Any,
    device: torch.device,
) -> Any:
    from vllm.v1.worker.utils import AttentionGroup

    attentions = [layer.self_attn.attn for layer in model.model.layers]
    layer_names = [attn.layer_name for attn in attentions]
    first = attentions[0]
    backend = first.get_attn_backend()
    if backend.get_name() != "TRITON_ATTN":
        raise ValueError(
            f"Standalone K3 DFlash requires TRITON_ATTN, got {backend.get_name()}"
        )
    kv_spec = first.get_kv_cache_spec(draft_vllm_config)
    if kv_spec is None:
        raise RuntimeError("K3 DFlash attention did not return a KV cache spec")
    group = AttentionGroup(backend, layer_names, kv_spec, 0)
    group.create_metadata_builders(
        draft_vllm_config,
        device,
        kernel_block_size=int(draft_vllm_config.cache_config.block_size),
    )
    return group.get_metadata_builder()


def _load_runtime(args: argparse.Namespace, status: RuntimeStatus) -> StandaloneRuntime:
    start = time.perf_counter()
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    status.device = torch.cuda.get_device_name(device)
    status.compute_capability, status.torch_arches = _validate_cuda_runtime(device)
    status.phase = "building_config"

    vllm_config = _build_vllm_config(args)
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_config = speculative_config.draft_model_config.hf_config

    status.phase = "loading_shared_target_weights"
    with set_current_vllm_config(vllm_config):
        _init_single_gpu_distributed()
        init_workspace_manager(device, num_lanes=2)
        with torch.device(device), set_default_torch_dtype(torch.bfloat16):
            target = StandaloneTargetFacade(
                vocab_size=draft_config.vocab_size,
                hidden_size=draft_config.hidden_size,
            )
        load_shared_target_weights(target, args.target_weights, device)

        status.phase = f"loading_{args.method}"
        from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
            _create_draft_vllm_config,
        )

        if args.method == "dflash":
            from vllm.v1.worker.gpu.spec_decode.dflash.utils import (
                load_dflash_model,
                maybe_load_mask_embedding,
            )

            model = load_dflash_model(target, vllm_config)
            maybe_load_mask_embedding(
                model,
                str(args.draft_model),
                int(draft_config.dflash_config["mask_token_id"]),
            )
        else:
            from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model

            model = load_dspark_model(target, vllm_config)
        model.eval()
        draft_vllm_config = _create_draft_vllm_config(vllm_config)

        status.phase = "allocating_draft_kv_cache"
        block_size = int(vllm_config.cache_config.block_size)
        kv_caches, num_blocks = _allocate_draft_kv_cache(
            model,
            method=args.method,
            block_size=block_size,
            cache_gib=args.draft_kv_cache_gib,
            device=device,
        )
        status.draft_kv_cache_blocks = num_blocks
        status.draft_kv_cache_token_capacity = (num_blocks - 1) * block_size
        status.draft_kv_cache_gib = (
            sum(cache.numel() * cache.element_size() for cache in kv_caches.values())
            / 1024**3
        )

    torch.accelerator.synchronize()
    status.load_seconds = time.perf_counter() - start
    status.allocated_gib = torch.cuda.memory_allocated(device) / 1024**3
    status.reserved_gib = torch.cuda.memory_reserved(device) / 1024**3
    metadata_builder = (
        _build_dflash_metadata_builder(model, draft_vllm_config, device)
        if args.method == "dflash"
        else None
    )
    return StandaloneRuntime(
        vllm_config,
        draft_vllm_config,
        target,
        model,
        kv_caches,
        block_size,
        args.method,
        metadata_builder,
    )


@torch.inference_mode()
def _run_eager_smoke(runtime: StandaloneRuntime, device: torch.device) -> int:
    if runtime.method == "dflash":
        return _run_dflash_eager_smoke(runtime, device)

    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonDecodeMetadata,
        MLACommonMetadata,
    )
    from vllm.v1.worker.gpu.spec_decode.utils import (
        get_parallel_drafting_token_id,
    )

    model = runtime.model
    draft_config = runtime.vllm_config.speculative_config.draft_model_config.hf_config
    num_aux_layers = int(draft_config.num_target_layers)
    hidden_size = int(draft_config.hidden_size)

    aux = torch.zeros(
        (1, num_aux_layers * hidden_size),
        dtype=torch.bfloat16,
        device=device,
    )
    context_len = 1
    positions = torch.zeros(context_len, dtype=torch.int64, device=device)
    context = model.combine_hidden_states(aux)
    data_block = 1
    context_slots = torch.tensor(
        [data_block * runtime.kv_cache_block_size],
        dtype=torch.int64,
        device=device,
    )
    model.precompute_and_store_context_kv(context, positions, context_slots)

    sample_from_anchor = bool(getattr(draft_config, "sample_from_anchor", True))
    query_len = runtime.vllm_config.speculative_config.num_speculative_tokens
    if not sample_from_anchor:
        query_len += 1
    mask_token_id = get_parallel_drafting_token_id(draft_config)
    input_ids = torch.tensor(
        [draft_config.bos_token_id] + [mask_token_id] * (query_len - 1),
        dtype=torch.int64,
        device=device,
    )
    query_positions = torch.arange(
        context_len,
        context_len + query_len,
        dtype=torch.int64,
        device=device,
    )
    query_slots = torch.arange(
        data_block * runtime.kv_cache_block_size + context_len,
        data_block * runtime.kv_cache_block_size + context_len + query_len,
        dtype=torch.int64,
        device=device,
    )
    query_start_loc = torch.tensor([0, query_len], dtype=torch.int32, device=device)
    block_table = torch.tensor([[data_block]], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([context_len + query_len], dtype=torch.int32, device=device)
    metadata = MLACommonMetadata(
        num_reqs=1,
        max_query_len=query_len,
        max_seq_len=context_len + query_len,
        num_actual_tokens=query_len,
        query_start_loc=query_start_loc,
        slot_mapping=query_slots,
        num_decodes=1,
        num_decode_tokens=query_len,
        num_prefills=0,
        causal=False,
        # MLA metadata validates the cached latent width, not the full Q/K
        # projection width.
        head_dim=int(next(iter(runtime.kv_caches.values())).shape[-1]),
        prefill=None,
        decode=MLACommonDecodeMetadata(
            block_table=block_table,
            seq_lens=seq_lens,
            dcp_tot_seq_lens=None,
        ),
    )
    attn_metadata = {layer_name: metadata for layer_name in runtime.kv_caches}
    slot_mapping = {layer_name: query_slots for layer_name in runtime.kv_caches}
    with (
        set_current_vllm_config(runtime.draft_vllm_config),
        set_forward_context(
            attn_metadata,
            runtime.draft_vllm_config,
            num_tokens=query_len,
            skip_compiled=True,
            slot_mapping=slot_mapping,
        ),
    ):
        hidden = model(input_ids=input_ids, positions=query_positions)
    base_logits = model.compute_draft_logits(hidden[-1:])
    markov = model.markov_bias(model.markov_embed(input_ids[-1:]))
    token = int((base_logits + markov).argmax(dim=-1).item())
    torch.accelerator.synchronize()
    for cache in runtime.kv_caches.values():
        cache[data_block].zero_()
    return token


@torch.inference_mode()
def _run_dflash_eager_smoke(
    runtime: StandaloneRuntime,
    device: torch.device,
) -> int:
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
        get_eagle3_aux_layers_from_config,
    )
    from vllm.v1.worker.gpu.spec_decode.utils import (
        get_parallel_drafting_token_id,
    )

    speculative_config = runtime.vllm_config.speculative_config
    assert speculative_config is not None
    draft_config = speculative_config.draft_model_config.hf_config
    aux_layers = get_eagle3_aux_layers_from_config(speculative_config)
    if not aux_layers:
        raise ValueError("K3 DFlash config does not declare target auxiliary layers")

    raw_context = torch.zeros(
        (1, len(aux_layers) * int(draft_config.hidden_size)),
        dtype=torch.bfloat16,
        device=device,
    )
    context = runtime.model.combine_hidden_states(raw_context)
    block_size = runtime.kv_cache_block_size
    context_position = torch.zeros(1, dtype=torch.int64, device=device)
    context_slot = torch.tensor([block_size], dtype=torch.int64, device=device)
    runtime.model.precompute_and_store_context_kv(
        context,
        context_position,
        context_slot,
    )

    query_len = int(speculative_config.num_speculative_tokens) + 1
    sequence_end = 1 + query_len
    block_ids = list(range(1, 1 + (sequence_end + block_size - 1) // block_size))
    input_ids = torch.tensor(
        [draft_config.bos_token_id]
        + [get_parallel_drafting_token_id(draft_config)] * (query_len - 1),
        dtype=torch.int64,
        device=device,
    )
    positions = torch.arange(1, sequence_end, dtype=torch.int64, device=device)
    slots = torch.tensor(
        [
            block_ids[position // block_size] * block_size + position % block_size
            for position in range(1, sequence_end)
        ],
        dtype=torch.int64,
        device=device,
    )
    query_start_cpu = torch.tensor([0, query_len], dtype=torch.int32)
    query_start_gpu = query_start_cpu.to(device)
    seq_lens = torch.tensor([sequence_end], dtype=torch.int32, device=device)
    block_table = torch.tensor([block_ids], dtype=torch.int32, device=device)
    common = CommonAttentionMetadata(
        query_start_loc=query_start_gpu,
        query_start_loc_cpu=query_start_cpu,
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=torch.tensor([sequence_end], dtype=torch.int32),
        max_seq_len=sequence_end,
        num_reqs=1,
        num_actual_tokens=query_len,
        max_query_len=query_len,
        block_table_tensor=block_table,
        slot_mapping=slots,
        causal=True,
    )
    assert runtime.attn_metadata_builder is not None
    metadata = runtime.attn_metadata_builder.build(0, common)
    attn_metadata = {layer_name: metadata for layer_name in runtime.kv_caches}
    slot_mapping = {layer_name: slots for layer_name in runtime.kv_caches}
    with (
        set_current_vllm_config(runtime.draft_vllm_config),
        set_forward_context(
            attn_metadata,
            runtime.draft_vllm_config,
            num_tokens=query_len,
            skip_compiled=True,
            slot_mapping=slot_mapping,
        ),
    ):
        hidden = runtime.model(input_ids=input_ids, positions=positions)
    token = int(runtime.model.compute_logits(hidden[1:2]).argmax(dim=-1).item())
    torch.accelerator.synchronize()
    for cache in runtime.kv_caches.values():
        cache[1 : 1 + len(block_ids)].zero_()
    return token


def _make_handler(status: RuntimeStatus, proposal_engine: Any | None = None):
    class StatusHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path not in ("/", "/healthz", "/readyz", "/v1/status"):
                self.send_error(404)
                return
            payload = asdict(status)
            if proposal_engine is not None:
                payload["proposal_count"] = proposal_engine.proposal_count
                payload["proposal_active_requests"] = (
                    proposal_engine.allocator.active_requests
                )
                payload["proposal_max_requests"] = (
                    proposal_engine.allocator.max_requests
                )
                payload["proposal_last_latency_ms"] = proposal_engine.last_latency_ms
                payload["proposal_last_timing_ms"] = proposal_engine.last_timing_ms
                payload["proposal_mean_timing_ms"] = proposal_engine.mean_timing_ms
                payload["proposal_cold_bootstrap_count"] = (
                    proposal_engine.cold_bootstrap_count
                )
                payload["proposal_reconnect_count"] = proposal_engine.reconnect_count
                payload["proposal_last_reconnect_latency_ms"] = (
                    proposal_engine.last_reconnect_latency_ms
                )
                payload["proposal_prefix_cache_tokens"] = (
                    proposal_engine.prefix_cache_tokens
                )
                payload["proposal_prefix_cache_host_gib"] = (
                    proposal_engine.prefix_cache_host_bytes / 1024**3
                )
                payload["proposal_prefix_cache_gpu_gib"] = (
                    proposal_engine.prefix_cache_device_bytes / 1024**3
                )
                payload["proposal_cuda_graph_enabled"] = (
                    proposal_engine.cuda_graph_enabled
                )
                payload["proposal_cuda_graph_shapes"] = (
                    proposal_engine.cuda_graph_shapes
                )
                payload["proposal_cuda_graph_capture_seconds"] = (
                    proposal_engine.cuda_graph_capture_seconds
                )
                payload["proposal_cuda_graph_memory_gib"] = (
                    proposal_engine.cuda_graph_memory_gib
                )
                payload["proposal_cuda_graph_replay_count"] = (
                    proposal_engine.cuda_graph_replay_count
                )
                payload["proposal_cuda_graph_eager_fallback_count"] = (
                    proposal_engine.cuda_graph_eager_fallback_count
                )
            body = json.dumps(payload, sort_keys=True).encode()
            code = 200 if status.ready else 503
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            logger.info("DSpark status HTTP: %s", format % args)

    return StatusHandler


def _serve_status(
    args: argparse.Namespace,
    status: RuntimeStatus,
    stop: threading.Event,
    proposal_engine: Any | None = None,
) -> None:
    server = ThreadingHTTPServer(
        (args.host, args.port), _make_handler(status, proposal_engine)
    )
    server.timeout = 0.5

    def request_stop(signum: int, _frame: object) -> None:
        logger.info("Received signal %d; stopping DSpark status server", signum)
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    logger.info(
        "DSpark status endpoint listening on http://%s:%d", args.host, args.port
    )
    while not stop.is_set():
        server.handle_request()
    server.server_close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("dspark", "dflash"), default="dspark")
    parser.add_argument("--draft-model", type=Path, required=True)
    parser.add_argument("--target-weights", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--num-speculative-tokens", type=int, default=3)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-batched-tokens", type=int, default=64)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--max-retained-requests", type=int)
    parser.add_argument("--draft-kv-cache-gib", type=float, default=1.0)
    parser.add_argument("--draft-kv-window", type=int, default=32768)
    parser.add_argument("--proposal-address", default="tcp://127.0.0.1:8092")
    parser.add_argument("--disable-proposal-transport", action="store_true")
    parser.add_argument("--enable-cuda-graph", action="store_true")
    parser.add_argument("--cuda-graph-warmups", type=int, default=2)
    parser.add_argument("--skip-smoke-test", action="store_true")
    parser.add_argument("--exit-after-load", action="store_true")
    args = parser.parse_args()
    if (
        args.max_retained_requests is not None
        and args.max_retained_requests < args.max_num_seqs
    ):
        parser.error("--max-retained-requests must be >= --max-num-seqs")
    return args


def main() -> None:
    args = _parse_args()
    status = RuntimeStatus(
        draft_model=str(args.draft_model),
        method=args.method,
        target_weights=str(args.target_weights),
        num_speculative_tokens=args.num_speculative_tokens,
        max_model_len=args.max_model_len,
        draft_kv_window=args.draft_kv_window,
    )
    try:
        runtime = _load_runtime(args, status)
        if not args.skip_smoke_test:
            status.phase = "eager_smoke_test"
            status.smoke_token = _run_eager_smoke(
                runtime, torch.device("cuda", args.device)
            )
            status.draft_kv_cache_smoke = True
        stop = threading.Event()
        proposal_engine = None
        proposal_server = None
        if not args.disable_proposal_transport:
            status.phase = "starting_proposal_transport"
            from vllm.entrypoints.k3_dspark_rpc import (
                K3DSparkDraftEngine,
                K3DSparkZMQServer,
            )

            proposal_engine = K3DSparkDraftEngine(
                runtime,
                max_requests=(
                    args.max_retained_requests
                    if args.max_retained_requests is not None
                    else args.max_num_seqs
                ),
                window_size=args.draft_kv_window,
                device=torch.device("cuda", args.device),
            )
            if args.enable_cuda_graph:
                status.phase = f"capturing_{args.method}_cuda_graphs"
                proposal_engine.capture_cuda_graphs(warmups=args.cuda_graph_warmups)
            proposal_server = K3DSparkZMQServer(
                proposal_engine,
                address=args.proposal_address,
                stop=stop,
            )
            proposal_server.start()
            status.proposal_transport_ready = True
            status.proposal_address = args.proposal_address
            status.phase = "ready"
        else:
            status.phase = "ready_without_transport"
        status.ready = True
        status.allocated_gib = torch.cuda.memory_allocated(args.device) / 1024**3
        status.reserved_gib = torch.cuda.memory_reserved(args.device) / 1024**3
        logger.info("Standalone K3 draft is loaded: %s", json.dumps(asdict(status)))
        if args.exit_after_load:
            print(json.dumps(asdict(status), sort_keys=True), flush=True)
            return
        _serve_status(args, status, stop, proposal_engine)
        if proposal_server is not None:
            proposal_server.join()
    except Exception as exc:
        status.phase = "failed"
        status.error = f"{type(exc).__name__}: {exc}"
        logger.exception("Standalone K3 draft startup failed")
        raise


if __name__ == "__main__":
    main()
