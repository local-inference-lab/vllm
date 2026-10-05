# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Draft-step metadata policy of the B12X sparse indexer, checked without CUDA.

The B12X decode metadata carries its scan width in a device buffer and no
DeepGEMM schedule table, so the generic in-place draft refresh cannot maintain
it. The builder therefore reports no in-place update support, which keeps the
speculator on the per-step metadata rebuild path.
"""

import torch

from vllm.v1.attention.backends.mla import b12x_indexer as indexer


def test_b12x_indexer_builder_rebuilds_draft_decode_metadata(monkeypatch):
    def base_init(self, *args, block_table_width, **kwargs):
        # The generic builder advertises in-place refresh on CUDA with DeepGEMM.
        self.supports_draft_decode_metadata_update = True
        self.use_flattening = True
        self.supports_varlen = True
        self.dcp_world_size = 1
        self.device = torch.device("cpu")

    monkeypatch.setattr(
        indexer.DeepseekV32IndexerMetadataBuilder, "__init__", base_init
    )

    builder = indexer.B12xIndexerMetadataBuilder(block_table_width=1)

    assert builder.supports_draft_decode_metadata_update is False
    assert builder.use_flattening is False
    assert builder.supports_varlen is False
