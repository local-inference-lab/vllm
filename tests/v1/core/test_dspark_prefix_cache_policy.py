from types import SimpleNamespace

from vllm.v1.core.sched import scheduler as scheduler_module


def _spec(method: str, use_eagle: bool = True):
    return SimpleNamespace(method=method, use_eagle=lambda: use_eagle)


def _groups(*flags: bool):
    return [SimpleNamespace(is_eagle_group=flag) for flag in flags]


def test_dspark_without_target_eagle_group_does_not_drop_target_cache_tail():
    selector = getattr(scheduler_module, "use_eagle_for_target_cache", None)
    assert selector is not None, "target-cache EAGLE policy selector is missing"
    assert selector(_spec("dspark"), _groups(False, False)) is False


def test_dflash_without_target_eagle_group_does_not_drop_target_cache_tail():
    selector = getattr(scheduler_module, "use_eagle_for_target_cache", None)
    assert selector is not None, "target-cache EAGLE policy selector is missing"
    assert selector(_spec("dflash"), _groups(False, False)) is False


def test_explicit_target_eagle_group_keeps_cache_drop():
    selector = getattr(scheduler_module, "use_eagle_for_target_cache", None)
    assert selector is not None, "target-cache EAGLE policy selector is missing"
    assert selector(_spec("dspark"), _groups(False, True)) is True


def test_classic_eagle_keeps_legacy_unannotated_fallback():
    selector = getattr(scheduler_module, "use_eagle_for_target_cache", None)
    assert selector is not None, "target-cache EAGLE policy selector is missing"
    assert selector(_spec("eagle3"), _groups(False, False)) is True


def test_non_eagle_speculation_never_drops_target_cache_tail():
    selector = getattr(scheduler_module, "use_eagle_for_target_cache", None)
    assert selector is not None, "target-cache EAGLE policy selector is missing"
    assert selector(_spec("ngram", use_eagle=False), _groups(True)) is False
