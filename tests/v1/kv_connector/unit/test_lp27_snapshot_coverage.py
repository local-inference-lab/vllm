import pytest
from unittest.mock import MagicMock

from tests.v1.core.utils import create_requests, create_scheduler, mock_kv
from tests.v1.core.test_micro_slicing import _update
from tests.v1.kv_connector.unit.test_mooncake_store_scheduler import (
    _make_bare_scheduler,
)


@pytest.mark.parametrize(
    "batch_tokens,save_decode_cache,steps", [(8, False, 3), (8192, True, 18)]
)
def test_store_has_current_snapshot_without_new_allocation(
    tmp_path, batch_tokens, save_decode_cache, steps
):
    (tmp_path / "config.json").write_text(
        '{"architectures":["OPTForCausalLM"],"model_type":"opt","max_position_embeddings":32768}'
    )
    core = create_scheduler(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        device="cpu",
        enable_prefix_caching=True,
        block_size=16,
        max_num_seqs=2,
        max_model_len=32768,
        max_num_batched_tokens=batch_tokens,
        use_kv_connector=mock_kv(0, False),
        kv_role="kv_both",
    )
    mooncake = _make_bare_scheduler(
        kv_role="kv_both", save_decode_cache=save_decode_cache
    )
    mooncake._gpu_block_pool = core.kv_cache_manager.block_pool
    mooncake._boundary_state_group_ids = frozenset()
    mooncake.client = MagicMock()
    core.connector.update_state_after_alloc = mooncake.update_state_after_alloc
    captured = []

    def build(output):
        captured.append(
            (
                dict(output.num_scheduled_tokens),
                dict(output.kv_connector_block_state.block_ids),
                list(output.scheduled_cached_reqs.new_block_ids),
            )
        )
        print("snapshot", captured[-1])
        return mooncake.build_connector_meta(output)

    core.connector.build_connector_meta = build
    (request,) = create_requests(
        1, num_tokens=32, block_size=16, max_tokens=32, req_ids=["req-0"]
    )
    core.add_request(request)
    for step in range(steps):
        output = core.schedule()
        _update(core, output)
    assert captured
