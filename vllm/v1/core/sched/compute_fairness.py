# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Literal

ComputeServiceClass = Literal["decode", "prefill"]


@dataclass
class PrefillComputeShareController:
    """Allocate contended execution time with weighted virtual runtime."""

    prefill_compute_share: float
    decode_virtual_runtime: float = 0.0
    prefill_virtual_runtime: float = 0.0
    contention_active: bool = False

    def select(
        self, *, decode_runnable: bool, prefill_runnable: bool
    ) -> ComputeServiceClass | None:
        """Select the next runnable service class."""
        if not decode_runnable or not prefill_runnable:
            self.reset()
            if decode_runnable:
                return "decode"
            if prefill_runnable:
                return "prefill"
            return None

        if not self.contention_active:
            self.decode_virtual_runtime = 0.0
            self.prefill_virtual_runtime = 0.0
            self.contention_active = True

        if self.prefill_virtual_runtime < self.decode_virtual_runtime:
            return "prefill"
        return "decode"

    def record(
        self,
        service_class: ComputeServiceClass,
        elapsed_seconds: float,
        *,
        contended: bool,
    ) -> None:
        """Charge a completed execution quantum to its actual service class."""
        if not contended or elapsed_seconds <= 0.0:
            return

        if service_class == "prefill":
            charge = elapsed_seconds / self.prefill_compute_share
            self.prefill_virtual_runtime += charge
            if self.prefill_virtual_runtime - self.decode_virtual_runtime > charge:
                self.decode_virtual_runtime = self.prefill_virtual_runtime - charge
        else:
            charge = elapsed_seconds / (1.0 - self.prefill_compute_share)
            self.decode_virtual_runtime += charge
            if self.decode_virtual_runtime - self.prefill_virtual_runtime > charge:
                self.prefill_virtual_runtime = self.decode_virtual_runtime - charge

    def reset(self) -> None:
        """Discard credit when both classes are no longer runnable."""
        self.decode_virtual_runtime = 0.0
        self.prefill_virtual_runtime = 0.0
        self.contention_active = False
