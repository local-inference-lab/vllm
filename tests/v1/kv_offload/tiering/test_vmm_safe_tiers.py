# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TieringOffloadingSpec is VMM-safe when every secondary tier works on the
host primary region only."""

import pytest

from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)


def _config(spec_name: str, tiers: list[dict] | None = None) -> KVTransferConfig:
    extra: dict = {"spec_name": spec_name, "cpu_bytes_to_use": 1 << 30}
    if tiers is not None:
        extra["secondary_tiers"] = tiers
    return KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config=extra,
    )


@pytest.mark.parametrize(
    ("spec_name", "tiers", "safe"),
    [
        ("CPUOffloadingSpec", None, True),
        ("TieringOffloadingSpec", [], True),
        ("TieringOffloadingSpec", [{"type": "fs", "root_dir": "/x"}], True),
        # tiers that do not declare host-only I/O stay excluded
        ("TieringOffloadingSpec", [{"type": "example"}], False),
        (
            "TieringOffloadingSpec",
            [{"type": "fs", "root_dir": "/x"}, {"type": "p2p"}],
            False,
        ),
        ("SomeOtherSpec", None, False),
    ],
)
def test_vmm_safety_follows_the_secondary_tiers(spec_name, tiers, safe):
    config = _config(spec_name, tiers)
    assert OffloadingConnector.supports_vmm_safe_transfer_config(config) is safe
