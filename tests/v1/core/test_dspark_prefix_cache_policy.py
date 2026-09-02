from types import SimpleNamespace

from vllm.v1.core.sched.scheduler import use_eagle_for_target_cache


def _spec(method: str, use_eagle: bool = True) -> SimpleNamespace:
    """Build a speculative-config stand-in.

    Args:
        method: Speculative method name the selector inspects.
        use_eagle: Value returned by the config's ``use_eagle()`` query.

    Returns:
        An object exposing ``method`` and ``use_eagle()``.
    """
    return SimpleNamespace(method=method, use_eagle=lambda: use_eagle)


def _groups(*flags: bool) -> list[SimpleNamespace]:
    """Build target KV-cache group stand-ins.

    Args:
        *flags: One ``is_eagle_group`` value per group.

    Returns:
        Objects exposing ``is_eagle_group`` in the given order.
    """
    return [SimpleNamespace(is_eagle_group=flag) for flag in flags]


def test_dspark_without_target_eagle_group_does_not_drop_target_cache_tail():
    assert use_eagle_for_target_cache(_spec("dspark"), _groups(False, False)) is False


def test_dflash_without_target_eagle_group_does_not_drop_target_cache_tail():
    assert use_eagle_for_target_cache(_spec("dflash"), _groups(False, False)) is False


def test_explicit_target_eagle_group_keeps_cache_drop():
    assert use_eagle_for_target_cache(_spec("dspark"), _groups(False, True)) is True


def test_classic_eagle_keeps_legacy_unannotated_fallback():
    assert use_eagle_for_target_cache(_spec("eagle3"), _groups(False, False)) is True


def test_non_eagle_speculation_never_drops_target_cache_tail():
    assert (
        use_eagle_for_target_cache(_spec("ngram", use_eagle=False), _groups(True))
        is False
    )
