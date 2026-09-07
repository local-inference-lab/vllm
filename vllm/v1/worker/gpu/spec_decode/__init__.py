# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import torch

from vllm.config import VllmConfig


def init_speculator(vllm_config: VllmConfig, device: torch.device):
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    if speculative_config.method == "dflash":
        remote_address = os.environ.get("VLLM_K3_DRAFT_REMOTE_ADDRESS")
        if remote_address:
            from vllm.v1.worker.gpu.spec_decode.dspark.remote_speculator import (
                RemoteK3DSparkSpeculator,
            )

            return RemoteK3DSparkSpeculator(
                vllm_config,
                device,
                address=remote_address,
            )
        from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
            DFlashSpeculator,
        )

        return DFlashSpeculator(vllm_config, device)
    elif speculative_config.method == "dspark":
        remote_address = os.environ.get(
            "VLLM_K3_DRAFT_REMOTE_ADDRESS"
        ) or os.environ.get("VLLM_K3_DSPARK_REMOTE_ADDRESS")
        if remote_address:
            from vllm.v1.worker.gpu.spec_decode.dspark.remote_speculator import (
                RemoteK3DSparkSpeculator,
            )

            return RemoteK3DSparkSpeculator(
                vllm_config,
                device,
                address=remote_address,
            )
        from vllm.v1.worker.gpu.spec_decode.dspark.speculator import (
            DSparkSpeculator,
        )

        return DSparkSpeculator(vllm_config, device)
    elif speculative_config.use_gemma4_mtp():
        from vllm.v1.worker.gpu.spec_decode.gemma4.speculator import (
            Gemma4Speculator,
        )

        return Gemma4Speculator(vllm_config, device)
    elif speculative_config.method == "mtp":
        from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

        return MTPSpeculator(vllm_config, device)
    elif speculative_config.use_eagle():
        from vllm.v1.worker.gpu.spec_decode.eagle.speculator import (
            EagleSpeculator,
        )

        return EagleSpeculator(vllm_config, device)
    else:
        raise NotImplementedError(f"{speculative_config.method} is not supported yet.")
