# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for tensor-loader-variant compatibility.

Backs the claims in ``runs/m5-serve/loader-compatibility.md``.  vLLM/GG
ships a dozen ``--load-format`` variants and they do NOT all hand out
tensors with the same lifetime.  Two families exist:

* **Owning** — the yielded tensor keeps its storage alive by refcount
  (default lazy safetensors mmap, ``eager``, ``runai_streamer`` which
  clones, and our ``progressive`` stream).  A consumer may retain the
  tensor and copy it later.
* **Borrowed** — the tensor is a view into a buffer the loader recycles
  on the next yield and frees on context exit (``instanttensor`` with
  ``INSTANTTENSOR_COPY=0`` per GG PR #281, ``fastsafetensors`` when the
  process group has size > 1).  A consumer MUST copy before it returns.

The EXL3 quant methods retain (``Exl3Parameter.load_exl3_weight`` stores
``loaded_weight.contiguous()``, a no-op on an already-contiguous tensor)
and only copy in ``process_weights_after_loading``, so they are an
owning-only consumer.  These tests pin the two halves of the story that
belong to fungible quant:

1. our loader-side code is an *owning producer* — every tensor it yields
   stays valid after the generator, the resolver and the spec are gone;
2. our runtime-side code (``swap.py``, the only FQ code that consumes
   somebody else's tensors) is a *copying consumer* — it survives a
   payload buffer that gets recycled the moment the call returns.

Plus a control test that demonstrates the hazard itself, so the matrix's
"incompatible" verdicts are not an assertion of faith.
"""
import ast
import gc
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_fragments_cpu as tf  # noqa: E402
import test_progressive_cpu as tp  # noqa: E402
import toy_segments as toy  # noqa: E402

fq_swap = toy.load_tree_module("swap")
pg = tp.pg
fr = tf.fr

REPO_ROOT = Path(__file__).resolve().parents[2]
LOADER_INIT = REPO_ROOT / "vllm" / "model_executor" / "model_loader" / "__init__.py"


# ------------------------------------------------------- registry wiring
#
# `import vllm` needs the compiled extension, so the registry is checked
# statically.  That is the point: this guards the one line of ours that an
# upstream rebase of model_loader/__init__.py can silently drop.


def _loader_init_ast() -> ast.Module:
    return ast.parse(LOADER_INIT.read_text())


def _literal_formats(tree: ast.Module) -> set[str]:
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "LoadFormats"
                for t in node.targets
            )
            and isinstance(node.value, ast.Subscript)
        ):
            return {ast.literal_eval(e) for e in node.value.slice.elts}
    raise AssertionError("LoadFormats Literal not found")


def _registry(tree: ast.Module) -> dict[str, str]:
    for node in ast.walk(tree):
        target = node.targets[0] if isinstance(node, ast.Assign) else getattr(
            node, "target", None)
        if (
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and isinstance(target, ast.Name)
            and target.id == "_LOAD_FORMAT_TO_MODEL_LOADER"
        ):
            return {
                ast.literal_eval(k): ast.unparse(v)
                for k, v in zip(node.value.keys, node.value.values, strict=True)
            }
    raise AssertionError("_LOAD_FORMAT_TO_MODEL_LOADER not found")


def test_progressive_is_registered_and_declared():
    tree = _loader_init_ast()
    formats, registry = _literal_formats(tree), _registry(tree)
    assert "progressive" in formats, "progressive dropped from LoadFormats"
    assert registry["progressive"] == "_progressive_loader_cls"
    # Every declared format must be resolvable, and nothing may be
    # resolvable that is not declared -- get_model_loader() raises on the
    # first mismatch and --load-format rejects on the second.
    assert formats == set(registry)


def test_progressive_loader_is_resolved_lazily():
    """model_loader/__init__.py must not import exl3_fungible at module
    scope: progressive_loader.py imports base_loader, so an eager import
    here makes that module unimportable on its own (circular init)."""
    tree = _loader_init_ast()
    for node in tree.body:  # module scope only
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            assert "exl3_fungible" not in ast.unparse(node)
    fn = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_progressive_loader_cls"
    )
    assert "exl3_fungible.progressive_loader" in ast.unparse(fn)


# ---------------------------------------------- producer: owning tensors


def _all_k3_spec(tmp_path):
    return tp.make_spec(
        tmp_path, tp.make_policy({"3": [3] * tp.NUM_EXPERTS}))


def _expected_expert_bytes(layer=3, k=3):
    return {
        name: data
        for expert in range(tp.NUM_EXPERTS)
        for name, _dtype, _shape, data in tf._expert_tensors(
            layer, expert, k, seed=0)
    }


def _raw(t: torch.Tensor) -> bytes:
    # reshape(-1) first: 0-dim tensors (the mcg scalars) cannot be viewed
    # as a wider/narrower dtype.
    return t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def test_progressive_stream_tensors_outlive_the_generator(tmp_path):
    """The progressive stream is an OWNING producer.

    EXL3 keeps every loaded tensor until process_weights_after_loading,
    which runs long after the weights iterator is exhausted and dropped.
    The mmap behind each yielded view must be kept alive by the view
    itself, not by a local in the generator frame.
    """
    spec = _all_k3_spec(tmp_path)
    resolver = spec.make_resolver(cache_dir=tmp_path / "cache", environ={})
    it = pg.progressive_weights_iterator(spec, resolver)
    retained = dict(it)

    # Drop everything the loader owned, exactly as vLLM does once
    # model.load_weights() returns, and force finalizers to run.
    del it, resolver, spec
    gc.collect()

    expected = _expected_expert_bytes()
    assert expected  # the fixture actually produced expert tensors
    for name, data in expected.items():
        assert _raw(retained[name]) == data, name


def _fd_count(path: str) -> int:
    n = 0
    for fd in os.listdir("/proc/self/fd"):
        try:
            n += os.readlink(f"/proc/self/fd/{fd}") == path
        except OSError:
            continue
    return n


@pytest.mark.skipif(
    not os.path.isdir("/proc/self/fd"), reason="needs /proc/self/fd")
def test_progressive_stream_keeps_one_fd_per_shard(tmp_path):
    """The mapping outlives the stream; the *file object* must not.

    CPython's mmap dup()s the descriptor, so one fd per mapped shard is
    unavoidable and is what the retained views cost.  Keeping the original
    file object open on top of it doubles the descriptor cost of the load
    window and makes the close depend on refcounting rather than on scope.
    Measured mid-stream, where the difference is observable.
    """
    spec = _all_k3_spec(tmp_path)
    first_shard = str(spec.shard_files()[0].resolve())
    resolver = spec.make_resolver(cache_dir=tmp_path / "cache", environ={})

    it = pg.progressive_weights_iterator(spec, resolver)
    name, tensor = next(it)  # first shard is now open and mapped
    assert _fd_count(first_shard) == 1, (
        "the shard file object outlives its mmap during the load window")
    assert tensor.numel() > 0 and name

    retained = dict(it)
    retained[name] = tensor
    del it
    gc.collect()

    # Every mapping is still readable with no file object holding it open.
    for expected_name, data in _expected_expert_bytes().items():
        assert _raw(retained[expected_name]) == data, expected_name


def test_fragment_views_outlive_the_resolver(tmp_path):
    """fragments.materialize() views must survive the resolver going away
    -- swap.py stages from them and boot retains them until
    process_weights_after_loading."""
    seg_dir = tf.make_manifest_dir(tmp_path, layers=(3,), ks=(3, 4))
    resolver = tf.resolver(seg_dir, tmp_path / "c")
    pairs = dict(resolver.expert_tensors(3, 0, 3))
    assert pairs

    del resolver
    gc.collect()

    expected = {
        name: data for name, _d, _s, data in tf._expert_tensors(3, 0, 3, seed=0)
    }
    for name, tensor in pairs.items():
        assert _raw(tensor) == expected[name], name


# ------------------------------------------ consumer: borrowed payloads


class _BorrowedPayloadResolver:
    """A resolver that serves tensors the InstantTensor ``copy=False`` way.

    Every fragment is laid out into ONE scratch buffer that is recycled
    (filled with a poison byte) when the next fragment is materialized and
    on :meth:`close`.  Tensors carry the ``_vllm_instanttensor_borrowed``
    marker GG PR #281 stamps.  A consumer that copies out before returning
    is unaffected; a consumer that retains sees poison.
    """

    POISON = 0xA5

    def __init__(self, inner):
        self.inner = inner
        self._buf: torch.Tensor | None = None

    def resolve(self, layer, expert, k):
        return self.inner.resolve(layer, expert, k)

    def _recycle(self, nbytes: int) -> torch.Tensor:
        if self._buf is not None:
            self._buf.fill_(self.POISON)
        if self._buf is None or self._buf.numel() < nbytes:
            self._buf = torch.empty(nbytes, dtype=torch.uint8)
        return self._buf

    def materialize(self, fragment, *, name_filter=None):
        pairs = self.inner.materialize(fragment, name_filter=name_filter)
        sizes = [t.numel() * t.element_size() for _, t in pairs]
        buf = self._recycle(sum(sizes))
        out, off = [], 0
        for (name, src), nbytes in zip(pairs, sizes, strict=True):
            flat = buf[off:off + nbytes]
            flat.copy_(src.contiguous().reshape(-1).view(torch.uint8))
            view = flat.view(src.dtype).reshape(src.shape)
            view._vllm_instanttensor_borrowed = True
            out.append((name, view))
            off += nbytes
        return out

    def close(self):
        if self._buf is not None:
            self._buf.fill_(self.POISON)

    @property
    def stats(self):
        return self.inner.stats


class _RetainingConsumer:
    """The EXL3 parameter pattern, reduced: keep ``.contiguous()`` at load
    time, copy only in a later ``process()``."""

    def __init__(self):
        self.tensors: dict[str, torch.Tensor] = {}

    def load(self, name, loaded_weight):
        self.tensors[name] = loaded_weight.contiguous()

    def process(self):
        for name, t in list(self.tensors.items()):
            self.tensors[name] = t.to(device=t.device, non_blocking=True).contiguous()


def test_contiguous_and_same_device_to_are_not_copies():
    """The mechanism behind every 'incompatible' cell in the matrix.

    ``.contiguous()`` on a contiguous tensor and ``.to(<same device>)``
    both return *self*, so the retain-now/copy-later pattern never
    materializes owned storage.  If torch ever changes this, the matrix
    needs revisiting -- hence the assert.
    """
    t = torch.arange(8, dtype=torch.uint8)
    assert t.contiguous() is t
    assert t.to(device=t.device, non_blocking=True) is t
    assert t.to(device=t.device, non_blocking=True).contiguous() is t


def test_retaining_consumer_is_corrupted_by_borrowed_payloads(tmp_path):
    """Control: the hazard is real, not hypothetical."""
    seg_dir = tf.make_manifest_dir(tmp_path, layers=(3,), ks=(3, 4))
    inner = tf.resolver(seg_dir, tmp_path / "cache")
    borrowed = _BorrowedPayloadResolver(inner)

    consumer = _RetainingConsumer()
    for name, tensor in borrowed.materialize(borrowed.resolve(3, 0, 3)):
        assert getattr(tensor, "_vllm_instanttensor_borrowed", False)
        consumer.load(name, tensor)
    borrowed.close()          # loader context exits
    consumer.process()        # process_weights_after_loading

    poisoned = [
        name for name, t in consumer.tensors.items()
        if set(_raw(t)) == {_BorrowedPayloadResolver.POISON}
    ]
    assert len(poisoned) == len(consumer.tensors), (
        "the fake borrowed loader failed to model buffer recycling")


def test_fragment_reader_copies_out_of_borrowed_payloads(tmp_path):
    """``swap.ResolverFragmentSource`` is a COPYING consumer.

    It is the only FQ code that consumes tensors it did not produce, so it
    is the only place a borrowed buffer could reach us.  After the payload
    is recycled the stage must still hold the right bytes, and no stage
    buffer may alias the payload.
    """
    import test_swap_resolver_cpu as tsr

    root = tmp_path / "segments"
    tsr.make_segments(root)

    truth = fq_swap.ResolverFragmentSource(tf.resolver(root, tmp_path / "c2"))
    expected = fq_swap.ExpertStage(
        3, toy.HIDDEN, toy.INTERMEDIATE, pin_memory=False)
    truth.read_expert(layer=toy.LAYER_ID, k=3, expert=0, rank=0, dest=expected)

    borrowed = _BorrowedPayloadResolver(tf.resolver(root, tmp_path / "cache"))
    stage = fq_swap.ExpertStage(
        3, toy.HIDDEN, toy.INTERMEDIATE, pin_memory=False)
    fq_swap.ResolverFragmentSource(borrowed).read_expert(
        layer=toy.LAYER_ID, k=3, expert=0, rank=0, dest=stage)

    payload_ptr = borrowed._buf.data_ptr()
    payload_end = payload_ptr + borrowed._buf.numel()
    for proj in ("gate", "up", "down"):
        for comp in ("trellis", "suh", "svh"):
            t = stage.dest_tensor(proj, comp)
            assert not (payload_ptr <= t.data_ptr() < payload_end), (
                f"{proj}.{comp} aliases the borrowed payload")

    borrowed.close()  # recycle/free the payload

    for proj in ("gate", "up", "down"):
        for comp in ("trellis", "suh", "svh"):
            assert torch.equal(
                stage.dest_tensor(proj, comp), expected.dest_tensor(proj, comp)
            ), f"{proj}.{comp} corrupted by payload recycling"
    assert stage.mcg == expected.mcg


def test_swap_through_borrowed_payloads_is_byte_identical(tmp_path):
    """End to end: a whole swap staged through recycled payloads must land
    the same layer state as one staged from local segments."""
    import test_swap_resolver_cpu as tsr

    root = tmp_path / "segments"
    ckpt = tsr.make_segments(root)

    local_state = toy.cpu_layer_state(
        fq_swap, ckpt, tsr.T0_GLOBALS, tsr.T1_GLOBALS)
    tsr.make_engine(fq_swap.LocalSegmentSource(root), local_state).apply(
        tsr.PLAN, quiesce=nullcontext())

    borrowed = _BorrowedPayloadResolver(tf.resolver(root, tmp_path / "cache"))
    state = toy.cpu_layer_state(
        fq_swap, ckpt, tsr.T0_GLOBALS, tsr.T1_GLOBALS)
    report = tsr.make_engine(
        fq_swap.ResolverFragmentSource(borrowed), state).apply(
            tsr.PLAN, quiesce=nullcontext())
    borrowed.close()

    assert report.pairs == 1 and report.dropped == ()
    toy.assert_states_equal(state, local_state)


# --------------------------------------------------- progressive loader
#
# Gaps the matrix records as "works, with caveats" -- pinned here so they
# stay honest documentation rather than folklore.


def test_progressive_extra_config_rejects_default_loader_keys(tmp_path):
    """``--model-loader-extra-config`` is loader-specific: the keys the
    default loader accepts are NOT accepted here, and vice versa."""
    src = (REPO_ROOT / "vllm" / "model_executor" / "layers" / "quantization"
           / "exl3_fungible" / "progressive_loader.py").read_text()
    tree = ast.parse(src)
    keys = next(
        ast.literal_eval(n.value)
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "_EXTRA_CONFIG_KEYS"
    )
    assert keys == {"manifest_dir", "policy", "dense_source"}
    assert "enable_multithread_load" not in keys
    del tmp_path


def test_progressive_ignores_checkpoint_weight_name_prefixes():
    """Documented gap: DefaultModelLoader honours a model's
    ``checkpoint_weight_name_prefixes`` (GLM/DeepSeek MTP heads set it to
    read only their own shard tensors); the progressive loader has no such
    filter, so an MTP draft booted progressive streams the whole dense
    source.  Perf-only today -- assert it so a future fix updates the doc.
    """
    src = (REPO_ROOT / "vllm" / "model_executor" / "layers" / "quantization"
           / "exl3_fungible" / "progressive_loader.py").read_text()
    assert "checkpoint_weight_name_prefixes" not in src
    assert "secondary_weights" not in src


def test_progressive_spec_rejects_a_non_directory_dense_source(tmp_path):
    """Loader variants that take an object-storage URI (runai_streamer,
    modelexpress) cannot feed progressive: it needs a local directory and
    says so instead of failing deep inside the stream."""
    seg_dir = tf.make_manifest_dir(tmp_path, layers=(3,), ks=(3, 4))
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(
        tp.make_policy({"3": [3] * tp.NUM_EXPERTS})))
    with pytest.raises(ValueError, match="not a directory"):
        pg.ProgressiveSpec.from_env(
            None,
            environ={},
            overrides={"manifest_dir": str(seg_dir),
                       "policy": str(policy_path),
                       "dense_source": "s3://bucket/model"},
        )
