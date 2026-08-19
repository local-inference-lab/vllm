# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVarN configuration."""

import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# Named KVarN presets: each maps to a frozen set of config parameters.
# The trailing g<N> encodes the variance-normalization tile size, which must
# equal the vLLM block size. g128 is the current design point; g64 trades a
# little compression (more per-tile scale overhead per token) for finer
# quantization granularity (each tile's scales adapt to fewer tokens).
#
# Bit-width is fully parameterized in the quantizer and kernels (key_bits /
# value_bits), and the tile size flows through cfg.group everywhere (storage
# layout, Triton GROUP constexpr, flush / slot math), so additional presets are
# a one-line addition here. Keys carry more quantization sensitivity than values
# (key error propagates through the softmax exponentials, value error is averaged
# out by the softmax weights), so the shipped preset spends more bits on keys.
KVARN_PRESETS: dict[str, dict[str, int]] = {
    "kvarn_k4v2_g128": {"key_bits": 4, "value_bits": 2, "group": 128},
    "kvarn_k4v4_g128": {"key_bits": 4, "value_bits": 4, "group": 128},
    "kvarn_k4v2_g64": {"key_bits": 4, "value_bits": 2, "group": 64},
    "kvarn_k4v4_g64": {"key_bits": 4, "value_bits": 4, "group": 64},
    "kvarn_k5v5_g64": {"key_bits": 5, "value_bits": 5, "group": 64},
}

DEFAULT_KVARN_TAIL_TOKENS = 1024
DEFAULT_KVARN_K5V5_TAIL_TOKENS = 0
DEFAULT_KVARN_TAIL_DTYPE = "float16"

logger = init_logger(__name__)

_MLA_SPEC_DECODE_MAX_Q_ENV = "VLLM_B12X_MLA_SPEC_DECODE_MAX_Q"
_MLA_SPEC_EXTEND_MODE_ENV = "VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE"
_BF16_BYTES = 2
_INT32_BYTES = 4


def _cdiv(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


@dataclass(frozen=True)
class KVarNMLAWorkspaceEnvelope:
    dense_rows: int
    remap_elements: int
    rotation_rows: int
    physical_slot_rows: int
    dense_bytes: int
    total_bytes: int


def kvarn_mla_workspace_envelope(
    *,
    num_kv_pages: int,
    group_size: int,
    latent_dim: int,
    rope_dim: int,
    max_batched_tokens: int,
    max_active_rows: int,
    topk_tokens: int,
    boundary_blocks: int,
    rollback_blocks: int,
) -> KVarNMLAWorkspaceEnvelope:
    """Return the shared dense-workspace envelope for one local worker."""
    positive = {
        "num_kv_pages": num_kv_pages,
        "group_size": group_size,
        "latent_dim": latent_dim,
        "rope_dim": rope_dim,
        "max_batched_tokens": max_batched_tokens,
        "max_active_rows": max_active_rows,
        "topk_tokens": topk_tokens,
    }
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if boundary_blocks < 0 or rollback_blocks < 0:
        invalid.update(
            boundary_blocks=boundary_blocks,
            rollback_blocks=rollback_blocks,
        )
    if invalid:
        raise ValueError(f"Invalid KVarN MLA workspace dimensions: {invalid}")

    page_rows = num_kv_pages * group_size
    selected_rows = max_active_rows * topk_tokens
    transient_rows = (
        _cdiv(max_batched_tokens, group_size) + boundary_blocks + rollback_blocks
    ) * group_size
    dense_rows = (
        _cdiv(max(page_rows, selected_rows, transient_rows), group_size) * group_size
    )
    remap_elements = max_batched_tokens * topk_tokens
    dense_bytes = dense_rows * (latent_dim + rope_dim) * _BF16_BYTES
    total_bytes = (
        dense_bytes
        + remap_elements * _INT32_BYTES
        + max_batched_tokens * latent_dim * _BF16_BYTES
        + page_rows * _INT32_BYTES
    )
    return KVarNMLAWorkspaceEnvelope(
        dense_rows=dense_rows,
        remap_elements=remap_elements,
        rotation_rows=max_batched_tokens,
        physical_slot_rows=page_rows,
        dense_bytes=dense_bytes,
        total_bytes=total_bytes,
    )


def _positive_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %d", name, value, default)
        return default
    if parsed <= 0:
        logger.warning("Ignoring non-positive %s=%r; using %d", name, value, default)
        return default
    return parsed


def _kvarn_mla_decode_settings() -> tuple[int, bool, bool]:
    spec_decode_max_q = _positive_env_int(_MLA_SPEC_DECODE_MAX_Q_ENV, 8)
    mode = os.getenv(_MLA_SPEC_EXTEND_MODE_ENV, "auto").strip().lower()
    disabled_modes = {"0", "false", "off", "no"}
    forced_modes = {"1", "true", "on", "yes"}
    if mode not in {"auto", *disabled_modes, *forced_modes}:
        raise ValueError(
            f"{_MLA_SPEC_EXTEND_MODE_ENV} must be auto, 0, or 1 (got {mode!r})"
        )
    return spec_decode_max_q, mode not in disabled_modes, mode in forced_modes


@dataclass
class KVarNConfig:
    """Configuration for KVarN KV-cache quantization.

    Pipeline per (block, head):
      1. Hadamard rotation along head_dim (orthonormal, applied via external GEMM).
      2. Iterative log-domain variance-normalization (Sinkhorn-like) over the
         [D, group] tile for K (per-channel orientation) and [group, D] tile for
         V (per-token orientation).
      3. Asymmetric per-row RTN at `key_bits` / `value_bits`.
      4. Absorb the per-row RTN scale and zero-point into the matching
         sinkhorn scale axis (K: into per-channel; V: into per-token-in-tile).
         Reconstruction: ``x = (q * absorbed_scale + absorbed_zp) * other_scale``.

    Cache layout (per (block, head)) is a single packed record — see the
    backend's `get_kv_cache_shape` override. There is no per-token slot
    because the scales are tile-shared; the block boundary IS the tile.

    Args:
        head_dim: Attention head dimension (power of 2; tested at 128).
        key_bits: Bits per key element (default 4).
        value_bits: Bits per value element (default 4).
        group: KVarN tile size in tokens. Must equal vLLM block_size so that
            one vLLM block = one KVarN tile per head.
        sinkhorn_iters: Iterations of the alternating column/row std-norm in
            the variance-normalization loop (default 8; lossless vs 16).
        boundary_skip_layers: Number of leading / trailing transformer layers
            to keep in fp16 (KVarN's sink/residual analogue). Default 2 mirrors
            TurboQuant's default.
    """

    head_dim: int = 128
    key_bits: int = 4
    value_bits: int = 4
    group: int = 128
    # converges by ~4 iters; 8 lossless vs 16 (validated Qwen3-4B + Qwen3.6-27B AIME)
    sinkhorn_iters: int = 8
    sink_tokens: int = 128  # leading tokens retained in the precision tail
    precision_tail_tokens: int = DEFAULT_KVARN_TAIL_TOKENS
    """Most-recent tokens retained in the higher-precision side pool."""
    tail_dtype: str = DEFAULT_KVARN_TAIL_DTYPE
    boundary_skip_layers: int = (
        0  # layer-level skipping off by default; sink_tokens replaces it
    )

    # ── derived: storage layout ──────────────────────────────────────────────
    @property
    def k_packed_bytes(self) -> int:
        """Packed bytes for one K tile per head: D * group * key_bits / 8."""
        return math.ceil(self.head_dim * self.group * self.key_bits / 8)

    @property
    def v_packed_bytes(self) -> int:
        """Packed bytes for one V tile per head: group * D * value_bits / 8."""
        return math.ceil(self.group * self.head_dim * self.value_bits / 8)

    @property
    def k_scale_bytes(self) -> int:
        """fp16 bytes for K scales: s_col_K' [D] + zp_K' [D] + s_row_K [group].

        s_col_K' = rtn_scale ⊙ s_chan_sinkhorn  (per-channel absorbed scale)
        zp_K'    = rtn_zp    ⊙ s_chan_sinkhorn  (per-channel absorbed zero)
        s_row_K  = s_tok_sinkhorn               (per-token-in-tile)
        """
        return (2 * self.head_dim + self.group) * 2

    @property
    def v_scale_bytes(self) -> int:
        """fp16 bytes for V scales: s_col_V [D] + s_row_V' [group] + zp_V' [group].

        s_col_V  = s_chan_sinkhorn              (per-channel, untouched)
        s_row_V' = rtn_scale ⊙ s_tok_sinkhorn   (per-token-in-tile absorbed scale)
        zp_V'    = rtn_zp    ⊙ s_tok_sinkhorn   (per-token-in-tile absorbed zero)
        """
        return (self.head_dim + 2 * self.group) * 2

    @property
    def tile_bytes(self) -> int:
        """Total packed bytes per (block, head): K + V combined."""
        return (
            self.k_packed_bytes
            + self.k_scale_bytes
            + self.v_packed_bytes
            + self.v_scale_bytes
        )

    @property
    def tile_bytes_aligned(self) -> int:
        """tile_bytes rounded up for nicer Triton loads.

        For head_dim >= 256 we round the PER-TOKEN slot (tile_bytes / group) up to
        a power of 2. This is required for models with heterogeneous head_dim
        (e.g. Gemma-4: 256 sliding-window layers + 512 global layers): the raw
        slot has a fixed per-token-group scale term that doesn't scale with D, so
        slot(512)/slot(256) is not an integer and vLLM's KV-cache page-size
        unification (which scales block_size by that ratio) fails. Power-of-2 slots
        make the ratio an exact power of 2. head_dim<=128 keeps the tight 8-byte
        alignment (the common case; no padding). Trailing pad only — offsets are
        unchanged, so the layout/kernels are byte-compatible."""
        if self.head_dim >= 256:
            slot = math.ceil(self.tile_bytes / self.group)
            slot_pow2 = 1 << (slot - 1).bit_length()
            return slot_pow2 * self.group
        return ((self.tile_bytes + 7) // 8) * 8

    # ── slot byte offsets within one tile (used by the kernels) ──────────────
    @property
    def k_packed_offset(self) -> int:
        return 0

    @property
    def k_s_col_offset(self) -> int:
        return self.k_packed_offset + self.k_packed_bytes

    @property
    def k_zp_offset(self) -> int:
        return self.k_s_col_offset + self.head_dim * 2

    @property
    def k_s_row_offset(self) -> int:
        return self.k_zp_offset + self.head_dim * 2

    @property
    def v_packed_offset(self) -> int:
        return self.k_s_row_offset + self.group * 2

    @property
    def v_s_col_offset(self) -> int:
        return self.v_packed_offset + self.v_packed_bytes

    @property
    def v_s_row_offset(self) -> int:
        return self.v_s_col_offset + self.head_dim * 2

    @property
    def v_zp_offset(self) -> int:
        return self.v_s_row_offset + self.group * 2

    # ── precision-tail pool sizing ───────────────────────────────────────────
    # A tile cannot be packed until all `group` tokens exist. The fixed side
    # pool retains sink, recent, and in-progress tiles in the configured tail
    # dtype (FP8 by default) and must be allocated before CUDA graph capture.
    # Its size bounds scheduler concurrency; `max_supported_seqs` caps the
    # configured request count to the chosen memory budget.
    # The pool and the paged KV cache draw from the SAME pot: the memory left
    # after model weights (i.e. `gpu_memory_utilization · total − weights`).
    # Sizing the pool as a fixed fraction of *total* GPU memory was the bug
    # behind the concurrency cap: on a 4B/24GB card the pool got 0.08·24≈1.9 GB and
    # concurrency capped to ~30 while the KV cache sat at ~3% utilization —
    # ~10 GB of usable memory wasted. We instead give the pool a share of the
    # post-weight usable envelope (POOL_USABLE_SHARE), which auto-scales: a small
    # model on a big card gets a large pool (high concurrency), a model that
    # nearly fills the card gets a small one (degrades to cap≈1, never OOMs).
    # The legacy fraction-of-total path is kept as a fallback for when the weight
    # size can't be read. Both are tunable via KVARN_POOL_MEM_FRAC (interpreted
    # as share-of-usable when weights are known, else fraction-of-total).
    POOL_MEM_FRAC_DEFAULT = 0.08  # legacy: fraction of TOTAL (fallback)
    POOL_USABLE_SHARE_DEFAULT = 0.5  # share of (util·total − weights)

    def _slot_bytes_per_layer(self, num_kv_heads: int) -> int:
        """Bytes for one K/V precision-tail pool slot in one layer."""
        element_size = 1 if self.tail_dtype == "fp8" else 2
        data_bytes = self.group * num_kv_heads * self.head_dim * 2 * element_size
        scale_bytes = (
            self.group * num_kv_heads * 2 * 2 if self.tail_dtype == "fp8" else 0
        )
        return data_bytes + scale_bytes

    @property
    def resident_blocks_per_seq(self) -> int:
        """Maximum exact blocks intersecting the sink or precision tail."""
        tail_blocks = 0
        if self.precision_tail_tokens > 0:
            tail_blocks = math.ceil(
                (self.precision_tail_tokens + self.group - 1) / self.group
            )
        sink_blocks = math.ceil(self.sink_tokens / self.group)
        return tail_blocks + sink_blocks

    def pool_slots(self, max_num_seqs: int, max_num_batched_tokens: int) -> int:
        """Structural peak of exact pool slots needed in one scheduler step."""
        prefill_blocks = math.ceil(max_num_batched_tokens / self.group)
        resident = self.resident_blocks_per_seq * max_num_seqs
        return max(resident + prefill_blocks + 8, 8)

    def pool_budget_bytes(
        self,
        total_gpu_bytes: int,
        gpu_memory_utilization: float | None = None,
        weight_bytes: int | None = None,
    ) -> int:
        """GPU bytes the precision-tail pool is allowed to occupy.

        Preferred (weight-aware): a share of the post-weight usable envelope,
        ``share · (gpu_memory_utilization · total − weight_bytes)``. This is the
        memory the pool and the paged KV cache actually compete for, so the
        budget tracks real headroom instead of an arbitrary slice of the whole
        card. ``share`` comes from KVARN_POOL_MEM_FRAC or
        POOL_USABLE_SHARE_DEFAULT.

        Fallback (weights unknown): the legacy ``frac · total`` with
        POOL_MEM_FRAC_DEFAULT, so behaviour is unchanged when we cannot read the
        weight size."""
        env = os.environ.get("KVARN_POOL_MEM_FRAC")
        if weight_bytes is not None and gpu_memory_utilization is not None:
            share = float(env) if env is not None else self.POOL_USABLE_SHARE_DEFAULT
            usable = gpu_memory_utilization * total_gpu_bytes - weight_bytes
            return max(0, int(share * usable))
        frac = float(env) if env is not None else self.POOL_MEM_FRAC_DEFAULT
        return int(total_gpu_bytes * frac)

    def max_supported_seqs(
        self,
        total_gpu_bytes: int,
        num_kv_heads: int,
        num_layers: int,
        max_num_batched_tokens: int,
        frac: float | None = None,
        gpu_memory_utilization: float | None = None,
        weight_bytes: int | None = None,
    ) -> int:
        """Largest max_num_seqs whose exact pool fits the configured budget.

        The bound includes every block that can intersect the precision tail,
        the optional sink block, blocks touched by one chunked-prefill step,
        and fixed allocator headroom.
        """
        if frac is not None:
            budget = int(total_gpu_bytes * frac)
        else:
            budget = self.pool_budget_bytes(
                total_gpu_bytes, gpu_memory_utilization, weight_bytes
            )
        slot_bytes = self._slot_bytes_per_layer(num_kv_heads) * max(num_layers, 1)
        max_slots = int(budget / slot_bytes)
        prefill_blocks = math.ceil(max_num_batched_tokens / self.group)
        per_seq = max(self.resident_blocks_per_seq, 1)
        return max(1, (max_slots - prefill_blocks - 8) // per_seq)

    def pool_bytes(
        self,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        num_kv_heads: int,
        num_layers: int,
    ) -> int:
        """Total GPU bytes occupied by the precision-tail pool on this rank.

        Pool slots are summed over every KVarN layer and reserved before the
        lazy layer-level allocation can compete with the paged cache.
        """
        slots = self.pool_slots(max_num_seqs, max_num_batched_tokens)
        return slots * self._slot_bytes_per_layer(num_kv_heads) * max(num_layers, 1)

    @staticmethod
    def num_kvarn_layers(model_config, parallel_config) -> int:
        """Number of attention layers spanned by the precision-tail pool.

        Hybrid models allocate the pool only for attention layers; dense
        transformers use their total layer count.
        """
        try:
            n = model_config.get_num_layers_by_block_type(parallel_config, "attention")
            if n and n > 0:
                return n
        except Exception:
            pass
        return model_config.get_num_layers(parallel_config)

    @staticmethod
    def estimate_weight_bytes(model: str, tensor_parallel_size: int = 1) -> int | None:
        """Best-effort per-rank model weight size in bytes, read from the
        checkpoint files on disk (exact, and cheap, with no CUDA context, which
        the early `check_and_update_config` hook must avoid). Returns None if the
        files can't be located, so the caller falls back to the legacy budget.

        Resolves a local directory directly, or the local HF cache snapshot for
        a repo id (never downloads). Prefers the shards named in a
        `*.safetensors.index.json` (or `*.bin.index.json`) manifest, which is
        exactly the set the loader reads. This avoids double-counting a repo that
        ships both a single consolidated checkpoint and the sharded HF set (e.g.
        Mistral-7B-Instruct-v0.3 carries `consolidated.safetensors` alongside
        `model-0000n-of-0000m.safetensors`, which a plain glob sums to ~2x the
        real weight size). Divides by the tensor-parallel degree (weights shard
        ~evenly across ranks)."""
        import glob as _glob
        import json as _json

        try:
            d = model
            if not os.path.isdir(d):
                # Repo id: resolve the already-cached snapshot, if any.
                try:
                    from vllm.transformers_utils.repo_utils import hf_api

                    d = hf_api().snapshot_download(model, local_files_only=True)
                except Exception:
                    return None

            # 1) Prefer the loader's own manifest: sum only the shards it lists,
            #    so a stray consolidated/single-file copy is not double-counted.
            for ext in ("safetensors", "bin"):
                indexes = _glob.glob(
                    os.path.join(d, "**", f"*.{ext}.index.json"), recursive=True
                )
                if not indexes:
                    continue
                try:
                    with open(indexes[0]) as fh:
                        weight_map = _json.load(fh).get("weight_map", {})
                    base = os.path.dirname(indexes[0])
                    names = sorted(set(weight_map.values()))
                    shards = [os.path.join(base, s) for s in names]
                    # Trust the manifest only when every listed shard is on
                    # disk: a partial set would under-estimate the weights and
                    # over-grow the pool budget, so fall through to the
                    # conservative glob instead.
                    if names and all(os.path.exists(p) for p in shards):
                        total = sum(os.path.getsize(p) for p in shards)
                        if total > 0:
                            return total // max(tensor_parallel_size, 1)
                except Exception:
                    pass  # fall through to the single-file / glob paths

            # 2) No usable manifest: prefer a canonical single-file checkpoint.
            for single in ("model.safetensors", "consolidated.safetensors"):
                p = os.path.join(d, single)
                if os.path.exists(p):
                    total = os.path.getsize(p)
                    if total > 0:
                        return total // max(tensor_parallel_size, 1)

            # 3) Fallback: sum whatever weight shards are present.
            files = _glob.glob(os.path.join(d, "**", "*.safetensors"), recursive=True)
            if not files:
                files = _glob.glob(os.path.join(d, "**", "*.bin"), recursive=True)
            if not files:
                return None
            total = sum(os.path.getsize(f) for f in files)
            if total <= 0:
                return None
            return total // max(tensor_parallel_size, 1)
        except Exception:
            return None

    @staticmethod
    def get_boundary_skip_layers(num_layers: int, n: int = 2) -> list[str]:
        """First-N + last-N transformer layer indices as strings, suitable
        for vLLM's ``kv_cache_dtype_skip_layers``. Mirrors TurboQuant
        (`TurboQuantConfig.get_boundary_skip_layers`)."""
        if n <= 0 or num_layers <= 0:
            return []
        n = min(n, num_layers // 2)
        first = list(range(n))
        last = list(range(num_layers - n, num_layers))
        return [str(i) for i in sorted(set(first + last))]

    @staticmethod
    def from_cache_dtype(cache_dtype: str, head_dim: int) -> "KVarNConfig":
        """Create a config from a preset string like ``"kvarn_k4v4"``."""
        if cache_dtype not in KVARN_PRESETS:
            valid = ", ".join(KVARN_PRESETS.keys())
            raise ValueError(
                f"Unknown KVarN cache dtype: {cache_dtype!r}. Valid: {valid}"
            )
        preset = KVARN_PRESETS[cache_dtype]
        # Optional env override for Sinkhorn iteration count (KVARN_SINKHORN_ITERS).
        # Default 16 mirrors the paper; useful for testing convergence at large
        # model scale (e.g. 48-layer 30B-A3B-Thinking-2507 may benefit from more).
        iters = int(os.environ.get("KVARN_SINKHORN_ITERS", "8"))
        sink_tokens = int(os.environ.get("KVARN_SINK_TOKENS", "128"))
        default_tail_tokens = (
            DEFAULT_KVARN_K5V5_TAIL_TOKENS
            if cache_dtype == "kvarn_k5v5_g64"
            else DEFAULT_KVARN_TAIL_TOKENS
        )
        precision_tail_tokens = int(
            os.environ.get("KVARN_TAIL_TOKENS", str(default_tail_tokens))
        )
        tail_dtype = os.environ.get(
            "KVARN_TAIL_DTYPE", DEFAULT_KVARN_TAIL_DTYPE
        ).lower()
        if tail_dtype not in ("float16", "bfloat16", "fp8"):
            raise ValueError(
                "KVARN_TAIL_DTYPE must be 'float16', 'bfloat16', or 'fp8', "
                f"got {tail_dtype!r}"
            )
        return KVarNConfig(
            head_dim=head_dim,
            key_bits=preset["key_bits"],
            value_bits=preset["value_bits"],
            group=preset["group"],
            sinkhorn_iters=iters,
            sink_tokens=sink_tokens,
            precision_tail_tokens=precision_tail_tokens,
            tail_dtype=tail_dtype,
        )


@dataclass(frozen=True)
class KVarNMLAConfig:
    """Packed KVarN latent plus exact BF16 RoPE layout for MLA caches.

    Live boundary, sink, and current latent blocks always use FP8 E4M3 side
    storage. Their RoPE values always use BF16 side storage.
    """

    latent_dim: int = 512
    rope_dim: int = 64
    bits: int = 5
    group: int = 64
    sinkhorn_iters: int = 8
    boundary_tokens: ClassVar[int] = 128

    @property
    def latent_packed_bytes(self) -> int:
        return math.ceil(self.latent_dim * self.group * self.bits / 8)

    @property
    def latent_scale_bytes(self) -> int:
        return (2 * self.latent_dim + self.group) * 2

    @property
    def rope_bytes(self) -> int:
        return self.group * self.rope_dim * 2

    @property
    def latent_s_col_offset(self) -> int:
        return self.latent_packed_bytes

    @property
    def latent_zp_offset(self) -> int:
        return self.latent_s_col_offset + self.latent_dim * 2

    @property
    def latent_s_row_offset(self) -> int:
        return self.latent_zp_offset + self.latent_dim * 2

    @property
    def rope_offset(self) -> int:
        return self.latent_s_row_offset + self.group * 2

    @property
    def tile_bytes(self) -> int:
        return self.rope_offset + self.rope_bytes

    @property
    def bytes_per_token(self) -> int:
        assert self.tile_bytes % self.group == 0
        return self.tile_bytes // self.group

    @property
    def resident_blocks_per_seq(self) -> int:
        return math.ceil(self.boundary_tokens / self.group) + 1

    def pool_slots(
        self,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        max_rollback_tokens: int = 0,
    ) -> int:
        prefill_blocks = math.ceil(max_num_batched_tokens / self.group)
        resident = self.resident_blocks_per_seq * max_num_seqs
        boundary_overlap = max_num_seqs
        rollback_blocks = (
            math.ceil(max_num_seqs * max_rollback_tokens / self.group) + max_num_seqs
            if max_rollback_tokens
            else 0
        )
        return max(
            resident + prefill_blocks + boundary_overlap + rollback_blocks + 8,
            8,
        )

    @property
    def pool_slot_bytes(self) -> int:
        return self.group * (self.latent_dim + self.rope_dim * 2)

    def max_active_rows(self, vllm_config: "VllmConfig") -> int:
        """Maximum decode/verify rows sharing the dense physical arena."""
        scheduler_config = vllm_config.scheduler_config
        max_batched = int(scheduler_config.max_num_batched_tokens)
        max_num_seqs = int(scheduler_config.max_num_seqs)
        spec_decode_max_q, spec_extend_enabled, spec_extend_forced = (
            _kvarn_mla_decode_settings()
        )
        q_per_req = 1
        spec_config = getattr(vllm_config, "speculative_config", None)
        if (
            spec_extend_enabled
            and spec_config is not None
            and getattr(spec_config, "num_speculative_tokens", None)
        ):
            q_per_req = 1 + int(spec_config.num_speculative_tokens)
        if spec_extend_forced:
            q_per_req = max(q_per_req, spec_decode_max_q)
        max_active_rows = min(max_num_seqs * q_per_req, max_batched)
        return max(max_active_rows, max_num_seqs)

    def workspace_envelope(
        self, vllm_config: "VllmConfig", num_kv_pages: int
    ) -> KVarNMLAWorkspaceEnvelope:
        """Size shared physical staging allocated once by a local worker."""
        scheduler_config = vllm_config.scheduler_config
        max_num_seqs = int(scheduler_config.max_num_seqs)
        max_rollback_tokens = 0
        spec_config = getattr(vllm_config, "speculative_config", None)
        if (
            scheduler_config.async_scheduling
            and spec_config is not None
            and getattr(spec_config, "num_speculative_tokens", None)
        ):
            max_rollback_tokens = int(spec_config.num_speculative_tokens)
        rollback_blocks = (
            _cdiv(max_num_seqs * max_rollback_tokens, self.group) + max_num_seqs
            if max_rollback_tokens
            else 0
        )
        return kvarn_mla_workspace_envelope(
            num_kv_pages=num_kv_pages,
            group_size=self.group,
            latent_dim=self.latent_dim,
            rope_dim=self.rope_dim,
            max_batched_tokens=int(scheduler_config.max_num_batched_tokens),
            max_active_rows=self.max_active_rows(vllm_config),
            topk_tokens=int(vllm_config.model_config.hf_config.index_topk),
            boundary_blocks=max_num_seqs,
            rollback_blocks=rollback_blocks,
        )

    @classmethod
    def from_cache_dtype(cls, cache_dtype: str) -> "KVarNMLAConfig":
        if cache_dtype != "kvarn_mla_k5_g64":
            raise ValueError(
                f"MLA KVarN requires 'kvarn_mla_k5_g64', got {cache_dtype!r}"
            )
        return cls(
            sinkhorn_iters=int(os.environ.get("KVARN_SINKHORN_ITERS", "8")),
        )
