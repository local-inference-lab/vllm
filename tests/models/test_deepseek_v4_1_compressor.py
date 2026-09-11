# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Accepted-prefix ownership through the real native compressor and metadata."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.cpu_test
def test_compressor_cache_rejects_duplicate_prefix_without_replacing_owner():
    from vllm.models.deepseek_v4_1.compressor import CompressorStateCache

    config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={}),
        speculative_config=None,
    )
    owner = CompressorStateCache(config, "compressor.state_cache")
    with pytest.raises(ValueError):
        CompressorStateCache(config, "compressor.state_cache")
    assert config.compilation_config.static_forward_context[owner.prefix] is owner
    other = CompressorStateCache(config, "other.state_cache")
    assert config.compilation_config.static_forward_context[other.prefix] is other


@pytest.fixture
def native_compressor_runners(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("native b12x compression requires SM12x")

    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear
    from vllm.models.deepseek_v4_1 import b12x_layers
    from vllm.models.deepseek_v4_1.compressor import DeepseekCompressor
    from vllm.models.deepseek_v4_1.sparse_mla import (
        DeepseekV41B12xMetadataBuilder,
    )
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)

    device = torch.device("cuda", torch.cuda.current_device())
    capacity, requests, width = 96, 3, 16
    monkeypatch.setattr(b12x_layers, "_capacity", lambda: capacity)
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=capacity, max_num_seqs=requests
        ),
        compilation_config=SimpleNamespace(static_forward_context={}),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(rms_norm_eps=1e-20)),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        speculative_config=None,
    )
    torch.manual_seed(411)
    original = torch.randn(3, 64, 128, device=device, dtype=torch.bfloat16)
    replacement = torch.randn_like(original)

    class Runner:
        def __init__(self, name):
            self.module = DeepseekCompressor(config, 2, 128, 512, prefix=name).to(
                device
            )
            self.spec = self.module.state_cache.get_kv_cache_spec(config)
            # Match allocator block-outer views, including a non-dense pool stride.
            self.state_width = self.spec.block_size * 1024
            self.backing = torch.full((64, self.state_width + 16), 123.0, device=device)
            self.pool = self.backing[:, : self.state_width].view(
                64, self.spec.block_size, 1024
            )
            self.module.state_cache.bind_kv_cache(self.pool)
            self.hidden = torch.empty(
                capacity, 128, device=device, dtype=torch.bfloat16
            )
            self.starts = torch.zeros(requests + 1, device=device, dtype=torch.int32)
            self.seq = torch.zeros(requests, device=device, dtype=torch.int32)
            self.input_slots = torch.full(
                (capacity,), -1, device=device, dtype=torch.int64
            )
            self.table = torch.zeros(requests, width, device=device, dtype=torch.int32)
            # Each identity owns a private state ring in its first physical block.
            self.pages = torch.arange(1, 49, device=device, dtype=torch.int32).view(
                3, width
            )
            main_spec = MLAAttentionSpec(
                block_size=128,
                num_kv_heads=1,
                head_size=288,
                dtype=torch.uint8,
                tokens_per_state=2,
            )
            self.builders = [
                DeepseekV41B12xMetadataBuilder(spec, [name], config, device)
                for spec in (main_spec, self.spec)
            ]
            self.graph = None

        def stage(self, identities, positions, lengths, source, pad=False, null=None):
            starts = [0]
            for length in lengths:
                starts.append(starts[-1] + length)
            n = starts[-1]
            self.starts.copy_(
                torch.tensor(
                    starts + [n] * (requests + 1 - len(starts)),
                    device=device,
                    dtype=torch.int32,
                )
            )
            self.seq.zero_()
            self.table.zero_()
            self.hidden.fill_(float("nan"))
            self.input_slots.fill_(-1)
            for row, (identity, pos, length) in enumerate(
                zip(identities, positions, lengths)
            ):
                self.seq[row] = pos + length
                self.table[row].copy_(self.pages[identity])
                self.hidden[starts[row] : starts[row + 1]].copy_(
                    source[identity, pos : pos + length]
                )
                self.input_slots[starts[row] : starts[row + 1]].fill_(1)
            if pad:
                self.input_slots[starts[-2] : n].fill_(-1)
            if null is not None:
                self.table[:, null] = 0
            cm = SimpleNamespace(
                query_start_loc=self.starts,
                seq_lens=self.seq,
                block_table_tensor=self.table,
                slot_mapping=self.input_slots,
                num_reqs=len(identities),
                num_actual_tokens=n,
                max_query_len=max(lengths, default=0),
            )
            self.main = self.builders[0].build(0, cm)
            state_cm = SimpleNamespace(**vars(cm))
            state_cm.block_table_tensor = self.table[:, :1]
            self.state = self.builders[1].build(0, state_cm)
            return n

        def run(self):
            if self.graph is None:
                self.module(self.hidden, self.main, self.state)
            else:
                self.graph.replay()
            n = self.main.num_actual_tokens
            return self.module._latent[:n].clone(), self.module._slots[:n].clone()

    with torch.no_grad():
        candidate, reference = Runner("candidate"), Runner("reference")
        candidate.module.fused_wkv_wgate.weight.normal_(0, 0.1)
        candidate.module.norm.weight.uniform_(0.5, 1.5)
        reference.module.load_state_dict(candidate.module.state_dict())
        for runner in (candidate, reference):
            runner.stage([0, 1], [0, 0], [10, 10], original)
            runner.run()
        yield candidate, reference, original, replacement


def test_native_compressor_rejection_reorder_padding_and_page_boundaries(
    native_compressor_runners,
):
    from vllm.models.deepseek_v4_1.attention import _pages
    from vllm.models.deepseek_v4_1.sparse_mla import _chunk

    candidate, reference, original, replacement = native_compressor_runners
    device = original.device
    width = candidate.table.shape[1]
    with torch.no_grad():
        # Capture fixed-capacity forward once; counts, tables and ordering change.
        candidate.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(candidate.graph):
            candidate.module(candidate.hidden, candidate.main, candidate.state)

        for boundary in (10, 11, 31, 32):
            for accepted in range(6):
                candidate.backing.fill_(123)
                reference.backing.fill_(123)
                candidate.stage([0, 1], [0, 0], [boundary, boundary], original)
                candidate.run()
                candidate.stage([0, 1], [boundary, boundary], [6, 6], original)
                candidate.run()
                next_pos = boundary + accepted + 1
                # The verifier has touched the whole suffix, but only its accepted
                # prefix belongs to history. The oracle never executes that suffix.
                reference.stage([0, 1], [0, 0], [next_pos, next_pos], original)
                reference.run()
                for runner in (candidate, reference):
                    runner.stage([1, 0], [next_pos, next_pos], [2, 2], replacement)
                actual, slots = candidate.run()
                expected, expected_slots = reference.run()
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                torch.testing.assert_close(slots, expected_slots, atol=0, rtol=0)
                assert (slots >= 0).sum().item() == 2
                assert (candidate.backing[0] == 123).all()
                assert (candidate.backing[:, candidate.state_width :] == 123).all()

        # A rejected/PAD request must not be resurrected by a real block table.
        before = candidate.pool[33:49].clone()
        candidate.stage([0, 1, 2], [20, 20, 20], [2, 2, 2], replacement, pad=True)
        _, slots = candidate.run()
        assert slots[-2:].tolist() == [-1, -1]
        torch.testing.assert_close(candidate.pool[33:49], before, atol=0, rtol=0)
        assert candidate.state.slot_mapping[6:].eq(-1).all()
        assert candidate.state.req_id_per_token[-1].item() == -1

        # An absent request-state block must neither supply carry nor receive
        # partials, even if incoming token slots claim it is writable.
        candidate.stage([0], [5], [1], replacement, null=0)
        _, slots = candidate.run()
        assert slots.tolist() == [-1]
        assert candidate.state.slot_mapping[0].item() == -1
        assert (candidate.backing[0] == 123).all()

        # Sparse reads must also preserve holes rather than attending null page0.
        candidate.stage([0], [8], [1], replacement, null=1)
        sparse = torch.empty(1, 16, device=device, dtype=torch.int64)
        lengths = torch.empty(1, device=device, dtype=torch.int32)
        top_lengths = torch.empty_like(lengths)
        for draft in (False, True):
            _chunk[(1,)](
                candidate.state.positions,
                candidate.state.req_id_per_token,
                candidate.table,
                sparse,
                lengths,
                top_lengths,
                candidate.state.cache_lengths,
                candidate.state.query_start_loc,
                candidate.state.request_positions,
                0,
                candidate.table.stride(0),
                4,
                8,
                16,
                draft,
                16,
            )
            logical = list(range(0 if draft else 1, 9))
            expected = [
                -1 if pos // 4 == 1 else (1 + pos // 4) * 4 + pos % 4 for pos in logical
            ]
            expected += [-1] * (16 - len(expected))
            assert sparse[0].tolist() == expected

        # Compressed/index page tables use the same null-block convention.
        pages = torch.empty((1, width), device=device, dtype=torch.int32)
        _pages[(1, 1)](
            candidate.state.req_id_per_token,
            candidate.table,
            pages,
            0,
            candidate.table.stride(0),
            width,
            width,
            16,
        )
        expected_pages = list(range(1, width + 1))
        expected_pages[1] = -1
        assert pages[0].tolist() == expected_pages

        # Empty graph replay must not change any previously accepted partial row.
        before = candidate.backing.clone()
        candidate.stage([], [], [], original)
        candidate.run()
        torch.testing.assert_close(candidate.backing, before, atol=0, rtol=0)


def test_native_compressor_high_page_offsets(native_compressor_runners):
    candidate, reference, original, replacement = native_compressor_runners
    device = original.device
    with torch.no_grad():
        # Recycled state pages can exceed Int32 element offsets even though
        # their page IDs still fit Int32. Warm/capture with that actual pool.
        stride = candidate.state_width + 16
        high_page = 2**31 // stride + 2
        required = (high_page + 1) * stride * 4
        if torch.cuda.mem_get_info()[0] < required + 512 * 1024**2:
            pytest.skip("requires just over 8 GiB for the partial-state offset check")
        high_backing = torch.empty((high_page + 1, stride), device=device)
        high_backing[high_page].fill_(123)
        candidate.graph = None
        candidate.module.state_cache.bind_kv_cache(
            high_backing[:, : candidate.state_width].view(
                high_page + 1, candidate.spec.block_size, 1024
            )
        )
        candidate.pages[0, 0] = high_page
        for runner in (candidate, reference):
            runner.stage([0], [0], [1], original)
            runner.run()
        candidate.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(candidate.graph):
            candidate.module(candidate.hidden, candidate.main, candidate.state)
        for runner in (candidate, reference):
            runner.stage([0], [1], [1], replacement)
        actual, _ = candidate.run()
        expected, _ = reference.run()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert (high_backing[high_page, candidate.state_width :] == 123).all()
