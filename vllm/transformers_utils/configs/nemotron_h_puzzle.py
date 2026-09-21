# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Heterogeneous Nemotron-H layer geometry, including checkpoints without MTP."""

from copy import deepcopy

from vllm.transformers_utils.configs.nemotron_h import NemotronHConfig


class NemotronHPuzzleConfig(NemotronHConfig):
    model_type = "nemotron_h_puzzle"

    def __init__(self, block_configs=None, mtp_block_configs=None, **kwargs):
        self.block_configs = deepcopy(block_configs or [])
        self.mtp_block_configs = deepcopy(mtp_block_configs or [])
        patterns = {"mamba": "M", "attention": "*", "moe": "E", "mlp": "-"}
        if self.block_configs:
            kwargs["hybrid_override_pattern"] = "".join(
                patterns[block["block_type"]] for block in self.block_configs
            )
            kwargs["num_hidden_layers"] = len(self.block_configs)
        if self.mtp_block_configs:
            pattern = "".join(
                patterns[block["block_type"]] for block in self.mtp_block_configs
            )
            if pattern != "*E" or self.mtp_block_configs[0].get("sliding_window"):
                raise ValueError("Puzzle MTP requires one full-attention/MoE pair")
            kwargs["mtp_hybrid_override_pattern"] = pattern
            kwargs["num_nextn_predict_layers"] = 1
        else:
            if kwargs.get("num_nextn_predict_layers", 0):
                raise ValueError("Puzzle MTP layers require mtp_block_configs")
            kwargs["num_nextn_predict_layers"] = 0
        kwargs.pop("blockwise_members", None)
        super().__init__(**kwargs)
        self.blockwise_members = sorted(
            {
                name
                for block in self.block_configs + self.mtp_block_configs
                for name, value in block.items()
                if value is not None
            }
        )
        for name in self.blockwise_members:
            if hasattr(self, name):
                delattr(self, name)

    def get_nemotron_h_config_for_layer(self, layer_idx):
        blocks = self.block_configs + self.mtp_block_configs
        if not 0 <= layer_idx < len(blocks):
            raise IndexError(
                f"Puzzle layer {layer_idx} is outside {len(blocks)} blocks"
            )
        values = self.to_dict()
        values.update({k: v for k, v in blocks[layer_idx].items() if v is not None})
        for name in ("block_configs", "mtp_block_configs", "blockwise_members"):
            values.pop(name, None)
        config = NemotronHConfig.from_dict(values)
        config._attn_implementation = self._attn_implementation
        return config

    @property
    def mtp_n_routed_experts(self):
        return self.get_nemotron_h_config_for_layer(
            self.num_hidden_layers + 1
        ).n_routed_experts
