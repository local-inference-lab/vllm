# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check the companion B12X source is used without loading a CUDA device."""

from pathlib import Path

import b12x
from b12x.moe._shared.trellismx.p8_native_kernel import P8NativeTPMoE

assert Path(b12x.__file__).resolve().is_relative_to(Path("/review"))
assert P8NativeTPMoE.__module__.startswith("b12x.")
print("Native P8 imports from the pinned companion B12X source")
