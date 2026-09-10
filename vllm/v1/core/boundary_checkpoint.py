# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exact request-boundary lookup and ownership for recurrent prefix caches."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from vllm.v1.core.kv_cache_utils import BlockHash, generate_block_hash_extra_keys
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool

MAX_BOUNDARY_STOP_TOKENS = 128

# Prompt and response retain their original slots so cached request state and
# worker metadata remain compatible when an instruction checkpoint is absent.
PROMPT_CHECKPOINT_SLOT = 0
RESPONSE_CHECKPOINT_SLOT = 1
INSTRUCTION_CHECKPOINT_SLOT = 2
NUM_BOUNDARY_CHECKPOINT_SLOTS = 3

BoundaryCheckpointKind = Literal["instruction", "prompt", "response"]


@dataclass(frozen=True)
class BoundaryCheckpoint:
    checkpoint_id: int
    num_tokens: int
    block_ids: tuple[tuple[int, ...], ...]
    auxiliary_block_ids: tuple[int, ...] = ()
    draft_prefix_len: int = 0
    kind: BoundaryCheckpointKind = "prompt"

    @property
    def dependencies(self) -> frozenset[int]:
        return frozenset(
            block_id
            for group in (*self.block_ids, self.auxiliary_block_ids)
            for block_id in group
            if block_id != 0
        )


@dataclass
class _RadixNode:
    tokens: tuple[int, ...] = ()
    checkpoint_id: int | None = None
    children: dict[int, "_RadixNode"] = field(default_factory=dict)

    def insert(self, tokens: tuple[int, ...], checkpoint_id: int) -> None:
        node = self
        while tokens:
            child = node.children.get(tokens[0])
            if child is None:
                node.children[tokens[0]] = _RadixNode(tokens, checkpoint_id)
                return
            common = 0
            for a, b in zip(tokens, child.tokens):
                if a != b:
                    break
                common += 1
            if common < len(child.tokens):
                split = _RadixNode(child.tokens[:common])
                node.children[tokens[0]] = split
                child.tokens = child.tokens[common:]
                split.children[child.tokens[0]] = child
                child = split
            node = child
            tokens = tokens[common:]
        node.checkpoint_id = checkpoint_id

    def find(self, tokens: tuple[int, ...]) -> int | None:
        node = self
        found = node.checkpoint_id
        offset = 0
        while offset < len(tokens):
            child = node.children.get(tokens[offset])
            if child is None or tokens[offset : offset + len(child.tokens)] != (
                child.tokens
            ):
                break
            offset += len(child.tokens)
            node = child
            if node.checkpoint_id is not None:
                found = node.checkpoint_id
        return found

    def remove(self, tokens: tuple[int, ...]) -> None:
        node = self
        path: list[tuple[_RadixNode, int]] = []
        offset = 0
        while offset < len(tokens):
            key = tokens[offset]
            path.append((node, key))
            node = node.children[key]
            offset += len(node.tokens)
        node.checkpoint_id = None
        for parent, key in reversed(path):
            child = parent.children[key]
            if child.checkpoint_id is not None:
                break
            if not child.children:
                del parent.children[key]
            elif len(child.children) == 1:
                grandchild = next(iter(child.children.values()))
                grandchild.tokens = child.tokens + grandchild.tokens
                parent.children[key] = grandchild


# A full hash authenticates preceding blocks; the first block uses the same
# discriminators as the normal prefix cache, including cache salt and LoRA.
_RootKey = tuple[int, BlockHash | None, tuple[Any, ...] | None]


@dataclass
class _PendingCheckpoint:
    checkpoint: BoundaryCheckpoint
    root_key: _RootKey
    tokens: tuple[int, ...]
    num_ranks: int
    completed_ranks: set[int] = field(default_factory=set)
    invalidated: bool = False


class BoundaryCheckpointCache:
    """Cache immutable bundles in the ordinary, evictable KV block pool.

    Staged bundles pin every dependency until all workers acknowledge their
    copies. Published bundles release those pins and participate in the pool's
    LRU. Evicting any dependency invalidates the entire bundle. Readers acquire
    their own pins and must copy mutable tails before running a continuation.
    """

    def __init__(self, block_pool: "BlockPool") -> None:
        if block_pool.boundary_checkpoints is not None:
            raise ValueError("Block pool already has a boundary checkpoint cache")
        block_pool.boundary_checkpoints = self
        self.block_pool = block_pool
        self.hash_block_size = block_pool.hash_block_size
        self._roots: dict[_RootKey, _RadixNode] = {}
        self._entries: dict[int, _PendingCheckpoint] = {}
        self._pending: dict[int, _PendingCheckpoint] = {}
        self._by_block: dict[int, set[int]] = {}
        self._pending_by_block: dict[int, set[int]] = {}
        self._next_id = 1

    def __len__(self) -> int:
        return len(self._entries)

    def next_id(self) -> int:
        checkpoint_id = self._next_id
        self._next_id += 1
        return checkpoint_id

    def _root_key(self, request: Request, start: int) -> _RootKey:
        if start:
            return start, request.block_hashes[start // self.hash_block_size - 1], None
        extra_keys, _ = generate_block_hash_extra_keys(
            request, 0, self.hash_block_size, 0
        )
        return 0, None, extra_keys

    @staticmethod
    def supports_request(request: Request) -> bool:
        return (
            request.prompt_token_ids is not None
            and request.prompt_embeds is None
            and not request.mm_features
            and not request.resumable
            and request.sampling_params is not None
            and len(request.sampling_params.stop_token_ids or ())
            + int(request.sampling_params.eos_token_id is not None)
            <= MAX_BOUNDARY_STOP_TOKENS
        )

    def stage(
        self,
        request: Request,
        checkpoint: BoundaryCheckpoint,
        num_ranks: int,
    ) -> None:
        """Pin a complete bundle while its asynchronous copies finish."""
        if not self.supports_request(request):
            raise ValueError("Boundary checkpoints require a text generation request")
        if not 0 < checkpoint.num_tokens <= request.num_tokens:
            raise ValueError("Checkpoint must cover an existing, nonempty prefix")
        if not 0 <= checkpoint.draft_prefix_len <= checkpoint.num_tokens:
            raise ValueError("Draft prefix cannot extend beyond the target prefix")
        if num_ranks < 1:
            raise ValueError("Checkpoint must be acknowledged by at least one rank")
        cid = checkpoint.checkpoint_id
        if cid in self._entries or cid in self._pending:
            raise ValueError("Checkpoint ID is already in use")
        blocks = [self.block_pool.blocks[i] for i in checkpoint.dependencies]
        if not blocks or any(block.ref_cnt == 0 for block in blocks):
            raise ValueError("Checkpoint dependencies must still be owned by producer")
        start = ((checkpoint.num_tokens - 1) // self.hash_block_size) * (
            self.hash_block_size
        )
        pending = _PendingCheckpoint(
            checkpoint,
            self._root_key(request, start),
            tuple(request.all_token_ids[start : checkpoint.num_tokens]),
            num_ranks,
        )
        self.block_pool.touch(blocks)
        self._pending[cid] = pending
        for block_id in checkpoint.dependencies:
            self._pending_by_block.setdefault(block_id, set()).add(cid)

    def acknowledge(self, checkpoint_id: int, rank: int) -> bool:
        """Publish only after every rank's copy completion has been observed."""
        pending = self._pending.get(checkpoint_id)
        if pending is None:
            return False
        if not 0 <= rank < pending.num_ranks:
            raise ValueError("Invalid checkpoint rank")
        pending.completed_ranks.add(rank)
        if len(pending.completed_ranks) != pending.num_ranks:
            return False
        if pending.invalidated:
            self.discard(checkpoint_id)
            return False
        del self._pending[checkpoint_id]
        self._forget_pending(pending.checkpoint)
        root = self._roots.setdefault(pending.root_key, _RadixNode())
        previous_id = root.find(pending.tokens)
        if previous_id is not None:
            previous = self._entries[previous_id]
            if previous.tokens == pending.tokens:
                # Deduplicate exact endpoints without disturbing active readers.
                self.invalidate(previous_id)
                root = self._roots.setdefault(pending.root_key, _RadixNode())
        root.insert(pending.tokens, checkpoint_id)
        self._entries[checkpoint_id] = pending
        for block_id in pending.checkpoint.dependencies:
            self._by_block.setdefault(block_id, set()).add(checkpoint_id)
        self._release(pending.checkpoint)
        return True

    def discard(self, checkpoint_id: int) -> None:
        """Release an unpublished bundle after its GPU writers have completed."""
        pending = self._pending.pop(checkpoint_id, None)
        if pending is not None:
            self._forget_pending(pending.checkpoint)
            self._release(pending.checkpoint)

    def _forget_pending(self, checkpoint: BoundaryCheckpoint) -> None:
        for block_id in checkpoint.dependencies:
            ids = self._pending_by_block[block_id]
            ids.remove(checkpoint.checkpoint_id)
            if not ids:
                del self._pending_by_block[block_id]

    def find(self, request: Request, max_length: int) -> BoundaryCheckpoint | None:
        if not self.supports_request(request) or request.skip_reading_prefix_cache:
            return None
        max_length = min(max_length, request.num_tokens)
        if max_length <= 0:
            return None
        start = ((max_length - 1) // self.hash_block_size) * self.hash_block_size
        for offset in range(start, -1, -self.hash_block_size):
            root = self._roots.get(self._root_key(request, offset))
            if root is None:
                continue
            tokens = tuple(
                request.all_token_ids[
                    offset : min(offset + self.hash_block_size, max_length)
                ]
            )
            checkpoint_id = root.find(tokens)
            if checkpoint_id is not None:
                return self._entries[checkpoint_id].checkpoint
        return None

    def acquire(self, checkpoint_id: int) -> BoundaryCheckpoint | None:
        entry = self._entries.get(checkpoint_id)
        if entry is None:
            return None
        checkpoint = entry.checkpoint
        self.block_pool.touch(
            [self.block_pool.blocks[i] for i in checkpoint.dependencies]
        )
        return checkpoint

    def release(self, checkpoint: BoundaryCheckpoint) -> None:
        self._release(checkpoint)

    def _release(self, checkpoint: BoundaryCheckpoint) -> None:
        self.block_pool.free_blocks(
            self.block_pool.blocks[i] for i in checkpoint.dependencies
        )

    def contains_block(self, block_id: int) -> bool:
        return block_id in self._by_block

    def invalidate_block(self, block_id: int) -> None:
        for checkpoint_id in self._pending_by_block.get(block_id, ()):
            self._pending[checkpoint_id].invalidated = True
        for checkpoint_id in tuple(self._by_block.get(block_id, ())):
            self.invalidate(checkpoint_id)

    def invalidate(self, checkpoint_id: int) -> None:
        entry = self._entries.pop(checkpoint_id, None)
        if entry is None:
            return
        root = self._roots[entry.root_key]
        root.remove(entry.tokens)
        if not root.children:
            del self._roots[entry.root_key]
        for block_id in entry.checkpoint.dependencies:
            ids = self._by_block[block_id]
            ids.remove(checkpoint_id)
            if not ids:
                del self._by_block[block_id]

    def clear(self) -> None:
        if self._pending:
            raise RuntimeError("Cannot clear checkpoints with pending GPU writers")
        self._entries.clear()
        self._roots.clear()
        self._by_block.clear()
