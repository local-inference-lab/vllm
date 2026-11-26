# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The uniform speculative-decode metadata fast path equals the generic path.

Runs on CPU (Python fill) and, when CUDA is available, on the GPU (Triton
kernel). Buffers are poisoned before every build, so a row the fast path
leaves unwritten shows up as a difference.
"""

import dataclasses
import sys
import types
import unittest

import torch

import vllm.v1.attention.backends.gdn_attn as gdn
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

NUM_SPEC = 3
WINDOW = NUM_SPEC + 1
MAX_SEQS = 64
MAX_BS = MAX_SEQS * WINDOW
POISON = 12345


def make_builder(device, seed):
    b = object.__new__(gdn.GDNAttentionMetadataBuilder)
    b.vllm_config = types.SimpleNamespace(
        cache_config=types.SimpleNamespace(mamba_cache_mode="align")
    )
    b.kv_cache_spec = types.SimpleNamespace(num_prefill_checkpoint_blocks=0)
    b.gdn_prefill_backend = "b12x"
    b._b12x_prefill_live_counts = None
    b._b12x_mixed = None
    b.num_spec = NUM_SPEC
    b.use_spec_decode = True
    b.state_index_columns = WINDOW
    b.use_full_cuda_graph = True
    b.decode_cudagraph_max_bs = MAX_BS
    g = torch.Generator().manual_seed(seed)
    b.mamba_aligned_state_indices = torch.randint(
        1, 100000, (MAX_SEQS, WINDOW), dtype=torch.int32, generator=g
    ).to(device)
    i32 = dict(dtype=torch.int32, device=device)
    b.spec_state_indices_tensor = torch.empty((MAX_BS, WINDOW), **i32)
    b.non_spec_state_indices_tensor = torch.empty((MAX_BS,), **i32)
    b.spec_sequence_masks = torch.empty((MAX_BS,), dtype=torch.bool, device=device)
    b.spec_token_indx = torch.empty((MAX_BS * WINDOW,), **i32)
    b.non_spec_token_indx = torch.empty((MAX_BS * WINDOW,), **i32)
    b.spec_query_start_loc = torch.empty((MAX_BS + 1,), **i32)
    b.non_spec_query_start_loc = torch.empty((MAX_BS + 1,), **i32)
    b.num_accepted_tokens = torch.empty((MAX_BS,), **i32)
    b._decode_state_indices_source = None
    b._decode_state_indices_view = None
    b._reuse_spec_decode_inputs = True
    b._uniform_spec_masks_cpu = torch.ones(MAX_BS, dtype=torch.bool)
    b._uniform_spec_rows_cpu = torch.arange(MAX_BS)
    b._uniform_spec_tokens = torch.arange(MAX_BS, **i32)
    b._uniform_spec_query_start = torch.arange(MAX_BS + 1, **i32) * WINDOW
    return b


def poison(b):
    b.spec_state_indices_tensor.fill_(POISON)
    b.non_spec_state_indices_tensor.fill_(POISON)
    b.spec_sequence_masks.fill_(True)
    b.spec_token_indx.fill_(POISON)
    b.non_spec_token_indx.fill_(POISON)
    b.spec_query_start_loc.fill_(POISON)
    b.non_spec_query_start_loc.fill_(POISON)
    b.num_accepted_tokens.fill_(POISON)


def batch(device, lens, drafts, padded_tokens, uniform_graph, seed):
    num_reqs = len(lens)
    qsl = torch.zeros(num_reqs + 1, dtype=torch.int32)
    qsl[1:] = torch.tensor(lens, dtype=torch.int32).cumsum(0)
    g = torch.Generator().manual_seed(seed)
    m = CommonAttentionMetadata(
        query_start_loc=qsl.to(device),
        query_start_loc_cpu=qsl,
        seq_lens=torch.randint(
            10, 5000, (num_reqs,), dtype=torch.int32, generator=g
        ).to(device),
        num_reqs=num_reqs,
        num_actual_tokens=padded_tokens,
        max_query_len=WINDOW,
        max_seq_len=5000,
        block_table_tensor=torch.zeros((num_reqs, 8), dtype=torch.int32, device=device),
        slot_mapping=torch.zeros(padded_tokens, dtype=torch.int64, device=device),
        is_prefilling=torch.zeros(num_reqs, dtype=torch.bool),
        uniform_decode_graph=uniform_graph,
    )
    accepted = torch.randint(
        1, WINDOW + 1, (num_reqs,), dtype=torch.int32, generator=g
    ).to(device)
    return m, accepted, torch.tensor(drafts, dtype=torch.int32)


def _h2d_without_cuda(data, dtype=None, device=None, **_):
    """Stand-in for async_tensor_h2d, which pins memory and so needs CUDA."""
    return torch.as_tensor(data, dtype=dtype).to(device)


if not torch.cuda.is_available():
    gdn.async_tensor_h2d = _h2d_without_cuda


def snapshot(metadata):
    out = {}
    for f in dataclasses.fields(metadata):
        v = getattr(metadata, f.name)
        out[f.name] = v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
    return out


def differences(a, b):
    diff = []
    for name in a:
        if name == "is_uniform_spec_decode":
            continue
        x, y = a[name], b[name]
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            if x.shape != y.shape or x.dtype != y.dtype or not torch.equal(x, y):
                diff.append(name)
        elif isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor) or x != y:
            diff.append(name)
    return diff


def run(device, lens, drafts, padded_tokens, uniform_graph, fast, seed=0):
    """Build on one builder, then refresh a second one, as two cache groups."""
    old = (
        gdn._PADDED_SPEC_DECODE_FASTPATH,
        gdn.envs.VLLM_GDN_SPEC_DECODE_METADATA_FASTPATH,
    )
    first, second = make_builder(device, seed), make_builder(device, seed + 1)
    first._reuse_spec_decode_inputs = second._reuse_spec_decode_inputs = fast
    gdn._PADDED_SPEC_DECODE_FASTPATH = fast
    try:
        m, accepted, drafts_cpu = batch(
            device, lens, drafts, padded_tokens, uniform_graph, seed
        )
        poison(first)
        poison(second)
        built = first.build(
            common_prefix_len=0,
            common_attn_metadata=m,
            num_accepted_tokens=accepted,
            num_decode_draft_tokens_cpu=drafts_cpu,
        )
        built_snapshot = snapshot(built)
        refreshed = second.update_block_table(
            built, m.block_table_tensor, m.slot_mapping
        )
        if device == "cuda":
            torch.cuda.synchronize()
        return built_snapshot, snapshot(refreshed), built.is_uniform_spec_decode
    finally:
        gdn._PADDED_SPEC_DECODE_FASTPATH = old[0]


def padded(live, reqs):
    return [WINDOW] * live + [0] * (reqs - live), [NUM_SPEC] * live + [-1] * (
        reqs - live
    )


class PaddedFastPath(unittest.TestCase):
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    def test_fast_path_equals_generic_path(self):
        shapes = [
            (1, 1),
            (1, 2),
            (2, 2),
            (3, 4),
            (13, 14),
            (17, 18),
            (31, 32),
            (32, 32),
            (33, 48),
            (47, 48),
            (49, 64),
            (63, 64),
            (64, 64),
        ]
        for device in self.devices:
            for live, reqs in shapes:
                for uniform_graph in (True, False):
                    for token_padding in (reqs * WINDOW, live * WINDOW):
                        lens, drafts = padded(live, reqs)
                        args = (device, lens, drafts, token_padding, uniform_graph)
                        fast = run(*args, fast=True, seed=live * 100 + reqs)
                        slow = run(*args, fast=False, seed=live * 100 + reqs)
                        label = (
                            f"{device} live={live} reqs={reqs} "
                            f"graph={uniform_graph} tokens={token_padding}"
                        )
                        self.assertFalse(slow[2], label)
                        if live == reqs and token_padding != reqs * WINDOW:
                            continue
                        self.assertTrue(fast[2], label)
                        self.assertEqual(
                            differences(fast[0], slow[0]), [], "build " + label
                        )
                        # The refreshed fast path never had this field (the
                        # unpadded one neither); layers read it only for
                        # batches with non-speculative rows.
                        self.assertIsNone(fast[1]["non_spec_state_indices_tensor"])
                        self.assertEqual(
                            [
                                d
                                for d in differences(fast[1], slow[1])
                                if d != "non_spec_state_indices_tensor"
                            ],
                            [],
                            "update " + label,
                        )
                        self.assertEqual(fast[0]["num_spec_decodes"], live)
                        state = fast[1]["spec_state_indices_tensor"]
                        self.assertTrue(
                            bool((state[live:] == NULL_BLOCK_ID).all()), label
                        )
                        self.assertFalse(bool((state == POISON).any()), label)

    def test_other_batches_keep_the_generic_path(self):
        cases = {
            "short draft": (
                [WINDOW, WINDOW - 1, WINDOW],
                [NUM_SPEC, NUM_SPEC - 1, NUM_SPEC],
            ),
            "padding first": ([0, WINDOW], [-1, NUM_SPEC]),
            "padding inside": ([WINDOW, 0, WINDOW], [NUM_SPEC, -1, NUM_SPEC]),
            "padding with tokens": ([WINDOW, 1], [NUM_SPEC, -1]),
            "no drafts": ([1, 1], [-1, -1]),
        }
        for device in self.devices:
            for name, (lens, drafts) in cases.items():
                b = make_builder(device, 1)
                m, accepted, drafts_cpu = batch(
                    device, lens, drafts, sum(lens), True, 1
                )
                self.assertEqual(
                    b._uniform_spec_live_reqs(m, accepted, drafts_cpu), 0, name
                )

    def test_switch_off_keeps_padded_batches_on_the_generic_path(self):
        lens, drafts = padded(3, 4)
        b = make_builder("cpu", 1)
        m, accepted, drafts_cpu = batch("cpu", lens, drafts, 16, True, 1)
        self.assertEqual(b._uniform_spec_live_reqs(m, accepted, drafts_cpu), 3)
        old = gdn._PADDED_SPEC_DECODE_FASTPATH
        gdn._PADDED_SPEC_DECODE_FASTPATH = False
        try:
            self.assertEqual(b._uniform_spec_live_reqs(m, accepted, drafts_cpu), 0)
        finally:
            gdn._PADDED_SPEC_DECODE_FASTPATH = old


if __name__ == "__main__":
    print("devices:", PaddedFastPath.devices, flush=True)
    unittest.main(argv=[sys.argv[0], "-v"])
