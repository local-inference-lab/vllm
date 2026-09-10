# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stable worklists for mixed GDN prefill and speculative verification."""

from types import SimpleNamespace

import torch

from vllm.utils.torch_utils import PIN_MEMORY


class B12xGdnMixedMetadata:
    """Builder-owned buffers; live partitions change without changing pointers."""

    def __init__(
        self,
        *,
        max_tokens: int,
        max_seqs: int,
        state_columns: int,
        device: torch.device,
    ) -> None:
        self.max_tokens = max_tokens
        self.max_seqs = max_seqs
        self.state_columns = state_columns

        def zeros(*shape, dtype=torch.int32):
            return torch.zeros(shape, dtype=dtype, device=device)

        self.query_start_loc = zeros(max_seqs + 1)
        self.state_indices = zeros(max_seqs)
        self.has_initial_state = zeros(max_seqs, dtype=torch.bool)
        self.live_counts = zeros(2)
        self.token_indices = zeros(max_tokens, dtype=torch.int64)
        self.request_rows = zeros(max_seqs, dtype=torch.int64)
        self.spec_query_start_loc = zeros(max_seqs + 1)
        self.spec_state_indices = zeros(max_seqs, state_columns)
        self.spec_accepted = torch.ones(max_seqs, dtype=torch.int32, device=device)
        self.spec_counts = zeros(2)
        self.spec_token_indices = zeros(max_seqs * state_columns, dtype=torch.int64)
        self.spec_request_rows = zeros(max_seqs, dtype=torch.int64)
        self.checkpoint = SimpleNamespace(
            state_indices=zeros(max_seqs),
            checkpoint_offsets=zeros(max_seqs),
        )
        self.checkpoint_columns = zeros(max_seqs, dtype=torch.int64)
        self.batch_ptr = torch.full(
            ((max_tokens + 7) // 8 + max_seqs,), -1, dtype=torch.int32, device=device
        )
        self.token_chunk_offset_ptr = torch.zeros_like(self.batch_ptr)
        worklists = (
            self.query_start_loc,
            self.has_initial_state,
            self.live_counts,
            self.token_indices,
            self.request_rows,
            self.spec_query_start_loc,
            self.spec_accepted,
            self.spec_counts,
            self.spec_token_indices,
            self.spec_request_rows,
            self.checkpoint_columns,
            self.batch_ptr,
            self.token_chunk_offset_ptr,
            self.checkpoint.checkpoint_offsets,
        )
        self._worklists_by_dtype = tuple(
            tuple(value for value in worklists if value.dtype == dtype)
            for dtype in (torch.int32, torch.int64, torch.bool)
        )
        self._num_non_spec = 0
        self._num_spec = 0

    @staticmethod
    def _copy_list(target: torch.Tensor, values: list[int], *, padding: int = 0):
        target.fill_(padding)
        if values:
            source = torch.tensor(
                values,
                dtype=target.dtype,
                device="cpu",
                pin_memory=PIN_MEMORY and target.is_cuda,
            )
            target[: len(values)].copy_(source, non_blocking=True)

    def stage(
        self,
        common,
        state_indices: torch.Tensor,
        accepted: torch.Tensor | None,
        draft_tokens_cpu: torch.Tensor | None,
        *,
        checkpoint_block_size: int | None,
    ) -> None:
        starts = common.query_start_loc_cpu[: common.num_reqs + 1].tolist()
        if len(starts) - 1 > self.max_seqs or starts[-1] > self.max_tokens:
            raise ValueError("GDN batch exceeds planned mixed-batch capacity")
        drafts = None if draft_tokens_cpu is None else draft_tokens_cpu.tolist()
        spec_rows, non_spec_rows = [], []
        for row, (start, end) in enumerate(zip(starts, starts[1:])):
            if end == start:
                continue
            if drafts is not None and drafts[row] >= 0 and end - start > 1:
                if end - start > self.state_columns:
                    raise ValueError("GDN verification exceeds planned state columns")
                spec_rows.append(row)
            else:
                non_spec_rows.append(row)
        self._num_non_spec, self._num_spec = len(non_spec_rows), len(spec_rows)
        for rows, cu, indices, row_indices, counts in (
            (
                non_spec_rows,
                self.query_start_loc,
                self.token_indices,
                self.request_rows,
                self.live_counts,
            ),
            (
                spec_rows,
                self.spec_query_start_loc,
                self.spec_token_indices,
                self.spec_request_rows,
                self.spec_counts,
            ),
        ):
            offsets = [0]
            tokens: list[int] = []
            for row in rows:
                tokens.extend(range(starts[row], starts[row + 1]))
                offsets.append(len(tokens))
            self._copy_list(cu, offsets, padding=len(tokens))
            self._copy_list(indices, tokens)
            self._copy_list(row_indices, rows)
            self._copy_list(counts, [len(rows), len(tokens)])

        self.has_initial_state.zero_()
        if non_spec_rows:
            rows = self.request_rows[: len(non_spec_rows)]
            computed = common.seq_lens - common.query_start_loc.diff()
            self.has_initial_state[: len(non_spec_rows)].copy_(computed[rows] > 0)
        self.spec_accepted.fill_(1)
        if spec_rows:
            if accepted is None:
                raise ValueError("GDN verification requires accepted-token counts")
            self.spec_accepted[: len(spec_rows)].copy_(
                accepted[self.spec_request_rows[: len(spec_rows)]]
            )

        conv_rows: list[int] = []
        conv_offsets: list[int] = []
        for packed_row, row in enumerate(non_spec_rows):
            chunks = (starts[row + 1] - starts[row] + 7) // 8
            conv_rows.extend([packed_row] * chunks)
            conv_offsets.extend(range(chunks))
        self._copy_list(self.batch_ptr, conv_rows, padding=-1)
        self._copy_list(self.token_chunk_offset_ptr, conv_offsets)

        checkpoint_offsets, checkpoint_columns = [], []
        if checkpoint_block_size is not None:
            upper = common.seq_lens_cpu_upper_bound
            if upper is None:
                raise ValueError("GDN checkpoints require CPU sequence-length bounds")
            lengths = upper.tolist()
            for row in non_spec_rows:
                query_len = starts[row + 1] - starts[row]
                length = lengths[row]
                offset = length // checkpoint_block_size * checkpoint_block_size - (
                    length - query_len
                )
                valid = (
                    length % checkpoint_block_size
                    and 0 < offset < query_len
                    and offset % 16 == 0
                )
                checkpoint_offsets.append(offset if valid else 0)
                checkpoint_columns.append(
                    length // checkpoint_block_size - 1 if valid else 0
                )
        self._copy_list(self.checkpoint.checkpoint_offsets, checkpoint_offsets)
        self._copy_list(self.checkpoint_columns, checkpoint_columns)
        self.refresh_state_indices(state_indices, common.block_table_tensor)

    def copy_worklists_from(self, source: "B12xGdnMixedMetadata") -> None:
        if source is self:
            return
        if (self.max_tokens, self.max_seqs, self.state_columns) != (
            source.max_tokens,
            source.max_seqs,
            source.state_columns,
        ):
            raise ValueError("GDN metadata reuse requires matching planned capacities")
        self._num_non_spec = source._num_non_spec
        self._num_spec = source._num_spec
        for destination, inputs in zip(
            self._worklists_by_dtype, source._worklists_by_dtype
        ):
            torch._foreach_copy_(destination, inputs, non_blocking=True)

    def refresh_state_indices(self, state_indices, block_table) -> None:
        self.state_indices.zero_()
        self.spec_state_indices.zero_()
        self.checkpoint.state_indices.zero_()
        if self._num_non_spec:
            rows = self.request_rows[: self._num_non_spec]
            self.state_indices[: self._num_non_spec].copy_(state_indices[rows, 0])
            columns = self.checkpoint_columns[: self._num_non_spec]
            self.checkpoint.state_indices[: self._num_non_spec].copy_(
                torch.where(
                    self.checkpoint.checkpoint_offsets[: self._num_non_spec] > 0,
                    block_table[rows, columns],
                    0,
                )
            )
        if self._num_spec:
            rows = self.spec_request_rows[: self._num_spec]
            self.spec_state_indices[: self._num_spec].copy_(
                state_indices[rows, : self.state_columns]
            )

    def convolution_metadata(self, token_capacity: int):
        programs = (token_capacity + 7) // 8 + self.max_seqs
        return SimpleNamespace(
            batch_ptr=self.batch_ptr,
            token_chunk_offset_ptr=self.token_chunk_offset_ptr,
            nums_dict={
                8: dict(
                    tot=programs,
                    mlist=None,
                    mlist_len=0,
                    offsetlist=None,
                    batch_ptr=self.batch_ptr,
                    token_chunk_offset_ptr=self.token_chunk_offset_ptr,
                )
            },
        )
