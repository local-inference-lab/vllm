# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.models.glm5next.nvidia.b12x import (
    Glm5NextB12xMLASparseBackend,
    Glm5NextB12xMLASparseImpl,
)
from vllm.platforms.cuda import CudaPlatform
from vllm.v1.attention.backends.mla.b12x_mla_sparse import B12xMLASparseImpl


def test_glm5next_b12x_backend_preserves_logical_head_size() -> None:
    assert Glm5NextB12xMLASparseBackend.get_name() == "B12X"
    assert Glm5NextB12xMLASparseBackend.get_supported_head_sizes() == [512]


def test_glm5next_sm120_indexer_uses_64_state_pages(monkeypatch) -> None:
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(index_kpool=4))
    )
    monkeypatch.setattr(
        CudaPlatform,
        "is_device_capability_family",
        classmethod(lambda cls, family: family == 120),
    )

    assert CudaPlatform._get_indexer_block_alignment(config) == 256


def test_glm5next_b12x_query_padding_is_exact() -> None:
    impl = object.__new__(Glm5NextB12xMLASparseImpl)
    q_nope = torch.arange(2 * 3 * 512, dtype=torch.bfloat16).view(2, 3, 512)
    q_pe = torch.empty((2, 3, 0), dtype=torch.bfloat16)
    workspace = torch.full((4, 3, 576), 7, dtype=torch.bfloat16)

    num_tokens, packed = impl._copy_query_to_buffer((q_nope, q_pe), workspace)

    assert num_tokens == 2
    torch.testing.assert_close(packed[..., :512], q_nope)
    assert torch.count_nonzero(packed[..., 512:]) == 0
    assert torch.count_nonzero(workspace[2:]) > 0


def test_glm5next_b12x_key_padding_is_exact(monkeypatch) -> None:
    impl = object.__new__(Glm5NextB12xMLASparseImpl)
    captured: dict[str, torch.Tensor] = {}

    def capture_cache_update(
        self,
        kv_c_normed,
        k_pe,
        kv_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
    ) -> None:
        captured["k_pe"] = k_pe

    monkeypatch.setattr(B12xMLASparseImpl, "do_kv_cache_update", capture_cache_update)
    impl.do_kv_cache_update(
        torch.ones((2, 512), dtype=torch.bfloat16),
        torch.empty((2, 1, 0), dtype=torch.bfloat16),
        torch.empty((1, 1, 64, 656), dtype=torch.uint8),
        torch.arange(2, dtype=torch.int64),
        "fp8_ds_mla",
        torch.ones((), dtype=torch.float32),
    )

    assert captured["k_pe"].shape == (2, 1, 64)
    assert torch.count_nonzero(captured["k_pe"]) == 0
