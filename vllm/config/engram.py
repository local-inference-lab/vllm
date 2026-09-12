# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import field
from typing import TYPE_CHECKING, Literal

from vllm.config.utils import config, get_hash_factors, hash_factors
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config.model import ModelConfig

logger = init_logger(__name__)

# Architecture -> the hf_text_config field naming its n-gram layers. A model is
# only configurable here if it actually has such layers to store.
_NGRAM_LAYER_FIELDS = {
    "DeepseekV41ForCausalLM": "engram_layer_ids",
}

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
    """Legacy compatibility setting. Native V4.1 table placement is selected
    exclusively by table_memory; this flag does not move its tables."""

    table_memory: Literal["device", "ram", "disk"] = "device"
    """Native V4.1 table storage: resident CUDA rows, mapped pinned host RAM,
    or SSD io_uring staging. Independent of the legacy cpu_offload option."""

    _ram_budget_checked_nbytes: int | None = field(default=None, init=False, repr=False)
    """Preflight state carried to workers; never re-charge allocated TP peers."""

    def verify_model_config(
        self, model_config: "ModelConfig | None", *, tp_size: int = 1
    ) -> None:
        """Reject Engram configuration for models without n-gram embeddings."""
        from vllm.platforms import current_platform

        field = (
            _NGRAM_LAYER_FIELDS.get(model_config.architecture)
            if model_config is not None
            else None
        )
        if (
            model_config is None
            or field is None
            or not current_platform.is_cuda()
            or not getattr(model_config.hf_text_config, field, None)
        ):
            raise ValueError(
                "EngramConfig requires a model with supported Engram "
                "embeddings. Currently only the CUDA DeepSeek V4.1 "
                f"implementation with non-empty {field or 'engram_layer_ids'} "
                "is supported."
            )
        if self.table_memory == "ram":
            self._verify_ram_budget(model_config.hf_text_config, tp_size)

    def _verify_ram_budget(self, hf_config, tp_size: int) -> None:
        from vllm.model_executor.model_loader.weight_utils import (
            _get_available_ram_bytes,
        )

        planned = _ram_table_nbytes(hf_config, tp_size)
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

    def compute_hash(self) -> str:
        """Hash settings that affect embedding execution and graph structure."""
        return hash_factors(get_hash_factors(self, {"_ram_budget_checked_nbytes"}))
