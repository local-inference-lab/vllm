# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import field
from typing import TYPE_CHECKING, Literal, Self

from pydantic import model_validator

from vllm.config.utils import config, get_hash_factors, hash_factors
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config.model import ModelConfig
    from vllm.config.parallel import ParallelConfig

logger = init_logger(__name__)

# Architecture -> the hf_text_config field naming its n-gram layers. A model is
# only configurable here if it actually has such layers to store.
_NGRAM_LAYER_FIELDS = {
    "DeepseekV41ForCausalLM": "engram_layer_ids",
    "Qwen4ExpForCausalLM": "ple_layer_ids",
    "Qwen4ExpForConditionalGeneration": "ple_layer_ids",
}


def model_has_engram_layers(model_config: "ModelConfig | None") -> bool:
    """Whether the model carries n-gram embedding layers."""
    if model_config is None:
        return False
    field = _NGRAM_LAYER_FIELDS.get(model_config.architecture)
    if field is None:
        return False
    return bool(getattr(model_config.hf_text_config, field, None))


_RAM_RESERVE_GIB = 4
_RAM_RESERVE_BYTES = _RAM_RESERVE_GIB << 30


def _ram_table_nbytes(hf_config, tp_size: int = 1) -> int:
    """Full-model packed E4M3/E8M0 storage, including ceil-row TP padding."""
    # Native V4.1 rows contain 256 E4M3 bytes and eight E8M0 scale bytes.
    return sum(
        ((rows + tp_size - 1) // tp_size) * tp_size * (256 + 8)
        for rows in hf_config.engram_num_embeddings
    )


@config
class EngramConfig:
    """Configuration for Engram embedding storage and sharding."""

    cpu_offload: bool = True
    """Store embedding weights in pinned CPU memory for UVA lookup."""

    embedding_across_dp: bool = False
    """Shard embeddings across TP and all DP ranks when enabled.
    Otherwise, each DP rank has a separate TP-sharded embedding replica."""

    dp_shared_memory: bool | None = None
    """Share CPU-offloaded embedding weights between co-located
    DP replicas. Each node stores one copy of every TP shard, reducing host
    memory without per-step Engram DP collectives. Requires sufficient
    /dev/shm capacity and a shared IPC namespace. Defaults to enabled whenever
    the other settings allow it, falling back to per-replica tables when DP
    replicas are not co-located on one node or /dev/shm cannot hold them."""

    use_thp: bool = False
    """Back private CPU-offloaded tables with transparent huge pages (best
    effort, falls back to ordinary pinned pages). Prefaulting the tables at
    startup takes longer. Requires cpu_offload without dp_shared_memory."""

    table_memory: Literal["device", "ram", "disk"] = "device"
    """Native V4.1 table storage: resident CUDA rows, mapped pinned host RAM,
    or SSD io_uring staging. Independent of the legacy cpu_offload option."""

    disk_resident_scales: bool = False
    """Keep original E8M0 scale bytes in mapped host RAM for disk tables.
    Weights still use io_uring. Requires about 5.72 GiB total host RAM on V4.1."""

    projection_tp: bool = False
    """Shard Engram WKV output columns over TP and gather the BF16 result.
    Opt-in pending matched whole-serving performance and precision checks."""

    _ram_budget_checked_nbytes: int | None = field(default=None, init=False, repr=False)
    """Preflight state carried to workers; never re-charge allocated TP peers."""

    @model_validator(mode="after")
    def _validate_shared_memory(self) -> Self:
        if self.dp_shared_memory and not self.cpu_offload:
            raise ValueError("dp_shared_memory requires cpu_offload=True")
        if self.use_thp and (not self.cpu_offload or self.dp_shared_memory):
            raise ValueError(
                "use_thp requires cpu_offload=True and dp_shared_memory=False"
            )
        return self

    def verify_model_config(
        self, model_config: "ModelConfig | None", *, tp_size: int = 1
    ) -> None:
        """Reject Engram configuration for models without n-gram embeddings."""
        field = (
            _NGRAM_LAYER_FIELDS.get(model_config.architecture)
            if model_config is not None
            else None
        )
        if (
            model_config is None
            or field is None
            or not getattr(model_config.hf_text_config, field, None)
        ):
            raise ValueError(
                "EngramConfig requires a model with supported Engram "
                "embeddings and non-empty n-gram layer ids."
            )

        if self.table_memory != "disk" and self.disk_resident_scales:
            raise ValueError("disk Engram controls require table_memory='disk'")
        if self.table_memory == "ram":
            self._verify_ram_budget(model_config.hf_text_config, tp_size)
        elif self.disk_resident_scales:
            self._verify_ram_budget(
                model_config.hf_text_config, tp_size, scales_only=True
            )

    def resolve_dp_shared_memory(self, parallel_config: "ParallelConfig") -> None:
        """Share host tables by default wherever the configuration permits."""
        if self.dp_shared_memory is None:
            self.dp_shared_memory = (
                self.cpu_offload
                and not self.use_thp
                and parallel_config.data_parallel_size > 1
                and not parallel_config.enable_elastic_ep
            )

    def _verify_ram_budget(
        self, hf_config, tp_size: int, *, scales_only: bool = False
    ) -> None:
        from vllm.model_executor.model_loader.weight_utils import (
            _get_available_ram_bytes,
        )

        planned = _ram_table_nbytes(hf_config, tp_size)
        if scales_only:
            planned = planned // 264 * 8
        if self._ram_budget_checked_nbytes == planned:
            return
        reserve = _RAM_RESERVE_BYTES
        try:
            available = _get_available_ram_bytes()
        except (OSError, ValueError) as exc:
            logger.warning("Unable to check Engram mapped-host RAM budget: %s", exc)
        else:
            logger.info(
                "Engram mapped-host RAM: %.2f GiB packed tables across TP%d, "
                "%.2f GiB available, %d GiB reserve",
                planned / (1 << 30),
                tp_size,
                available / (1 << 30),
                _RAM_RESERVE_GIB,
            )
            if planned + reserve > available:
                raise ValueError(
                    "Insufficient RAM for Engram mapped-host tables: "
                    f"{planned / (1 << 30):.2f} GiB packed tables + "
                    f"{_RAM_RESERVE_GIB} GiB "
                    f"reserve required, {available / (1 << 30):.2f} GiB available. "
                    "RAM mode never falls back to disk or device storage."
                )
        # Validation runs before model construction. Preserve this decision
        # through worker serialization and draft-config revalidation instead of
        # comparing the full model against memory left after other ranks load.
        self._ram_budget_checked_nbytes = planned

    def verify_parallel_config(self, parallel_config: "ParallelConfig") -> None:
        """Reject unsupported embedding parallel topologies."""
        if parallel_config.data_parallel_mode == "independent" and (
            self.embedding_across_dp or self.dp_shared_memory
        ):
            raise ValueError(
                "Independent data parallelism does not support Engram "
                "embedding_across_dp or dp_shared_memory."
            )
        if self.dp_shared_memory:
            if parallel_config.data_parallel_size <= 1:
                raise ValueError("dp_shared_memory requires data_parallel_size > 1.")
            if parallel_config.enable_elastic_ep:
                raise ValueError("dp_shared_memory is not supported with elastic EP.")
        if (
            self.embedding_across_dp
            and parallel_config.data_parallel_size > 1
            and parallel_config.enable_elastic_ep
        ):
            raise ValueError(
                "Engram embedding_across_dp is not supported with elastic EP yet."
            )

    def get_parallel_size(self, parallel_config: "ParallelConfig") -> int:
        """Derive the embedding group size from the parallel configuration."""
        size = parallel_config.tensor_parallel_size
        if self.embedding_across_dp and parallel_config.data_parallel_size > 1:
            size *= parallel_config.data_parallel_size
        return size

    def compute_hash(self) -> str:
        """Hash settings that affect embedding execution and graph structure."""
        return hash_factors(get_hash_factors(self, {"_ram_budget_checked_nbytes"}))
