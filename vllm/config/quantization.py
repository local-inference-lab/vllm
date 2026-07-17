# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Annotated, Any

from pydantic import Field, GetPydanticSchema, ValidationInfo, field_validator
from pydantic_core import core_schema

from vllm.config.utils import config
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
    kInt8StaticChannelSym,
    kMxfp4Dynamic,
    kMxfp8Dynamic,
    kNvfp4Static,
)

# User-facing names addressable from quantization_config.
QUANT_KEY_NAMES: dict[str, QuantKey] = {
    "fp8_per_tensor_static": kFp8StaticTensorSym,
    "fp8_per_tensor_dynamic": kFp8DynamicTensorSym,
    "fp8_per_token": kFp8DynamicTokenSym,
    "fp8_per_channel_static": kFp8StaticChannelSym,
    "fp8_per_block_static": kFp8Static128BlockSym,
    "fp8_per_block_dynamic": kFp8Dynamic128Sym,
    "mxfp8": kMxfp8Dynamic,
    "mxfp4": kMxfp4Dynamic,
    "int8_per_channel_static": kInt8StaticChannelSym,
}


def _coerce_quant_key(v: Any) -> QuantKey | None:
    if v is None or isinstance(v, QuantKey):
        return v
    if not isinstance(v, str):
        raise TypeError(f"expected str or QuantKey, got {type(v).__name__}")
    try:
        return QUANT_KEY_NAMES[v]
    except KeyError:
        raise ValueError(
            f"unknown quantization name {v!r}; "
            f"expected one of {sorted(QUANT_KEY_NAMES)}"
        ) from None


# Stop pydantic from introspecting QuantKey: it transitively contains a
# NamedTuple with `ClassVar[GroupShape]` declarations that pydantic refuses.
QuantKeyField = Annotated[
    QuantKey | None,
    GetPydanticSchema(
        lambda _src, _handler: core_schema.no_info_plain_validator_function(
            _coerce_quant_key
        )
    ),
]


@config
class QuantSpec:
    """Quantization spec for one layer kind (linear or MoE).

    `None` on either side means the method class falls back to its own default
    (typically inherited from the checkpoint, or unquantized for online).
    """

    weight: QuantKeyField = None
    """Weight quantization key, or a name from QUANT_KEY_NAMES."""

    activation: QuantKeyField = None
    """Activation quantization key, or a name from QUANT_KEY_NAMES."""


@config
class QuantizationConfigArgs:
    """User-facing quantization configuration.

    See `docs/features/quantization/online.md` for the schema and shorthand
    string forms accepted on `linear`, `moe`, and `shared_experts`.
    """

    linear: QuantSpec | None = None
    """Spec applied to ``LinearBase`` layers."""

    moe: QuantSpec | None = None
    """Spec applied to ``FusedMoE`` layers."""

    shared_experts: QuantSpec | None = None
    """Spec applied only to shared-expert gate/up/down projections."""

    ignore: list[str] = Field(default_factory=list)
    """Layers to skip quantization for."""

    @field_validator("linear", "moe", "shared_experts", mode="before")
    @classmethod
    def _coerce_spec(cls, v: Any, info: ValidationInfo) -> Any:
        if not isinstance(v, str):
            return v
        field_name = info.field_name
        assert field_name is not None
        if v in _ONLINE_SHORTHANDS and field_name != "shared_experts":
            spec = getattr(_ONLINE_SHORTHANDS[v], field_name)
            if spec is None:
                raise ValueError(
                    f"online shorthand {v!r} does not define a {field_name} spec"
                )
            return spec
        return QuantSpec(weight=_coerce_quant_key(v))


# CLI shorthands accepted by `--quantization`. Each desugars to a full
# QuantizationConfigArgs; activation overrides go through quantization_config.
_ONLINE_SHORTHANDS: dict[str, QuantizationConfigArgs] = {
    "fp8_per_tensor": QuantizationConfigArgs(
        linear=QuantSpec(weight=kFp8StaticTensorSym),
        moe=QuantSpec(weight=kFp8StaticTensorSym),
    ),
    "fp8_per_block": QuantizationConfigArgs(
        linear=QuantSpec(weight=kFp8Static128BlockSym),
        moe=QuantSpec(weight=kFp8Static128BlockSym),
    ),
    # Per-output-channel weight scale + dynamic per-token activation.
    # Same shape as llmcompressor's FP8_DYNAMIC recipe.
    "fp8_per_channel": QuantizationConfigArgs(
        linear=QuantSpec(weight=kFp8StaticChannelSym),
        moe=QuantSpec(weight=kFp8StaticChannelSym),
    ),
    "mxfp8": QuantizationConfigArgs(
        linear=QuantSpec(weight=kMxfp8Dynamic),
        moe=QuantSpec(weight=kMxfp8Dynamic),
    ),
    # INT8 weight-only on MoE; linear stays unquantized (no `linear` field).
    "int8_per_channel_weight_only": QuantizationConfigArgs(
        moe=QuantSpec(weight=kInt8StaticChannelSym),
    ),
    # Online NVFP4 on MoE with per-token dynamic activation scales (Blackwell +
    # FlashInfer TRTLLM only); linear stays unquantized (no `linear` field).
    "nvfp4_per_token": QuantizationConfigArgs(
        moe=QuantSpec(weight=kNvfp4Static),
    ),
}


# Names accepted by `--quantization`; "online" means "use quantization_config".
ONLINE_QUANT_SHORTHAND_NAMES: tuple[str, ...] = (
    *_ONLINE_SHORTHANDS.keys(),
    "online",
)


# Checkpoint formats that support overlaying online quantization on BF16 projections
# which the checkpoint explicitly leaves unquantized (shared experts via
# `shared_experts`, other dense linears via `linear`).
_MODELOPT_ONLINE_OVERLAY_NAMES = frozenset(
    {
        "modelopt",
        "modelopt_fp4",
        "modelopt_mxfp8",
        "modelopt_mixed",
        "mxfp4",
        "nvfp4_nf3_hybrid",
    }
)


_MODELOPT_ONLINE_OVERLAY_WEIGHTS = frozenset({kMxfp8Dynamic, kFp8Static128BlockSym})


def _is_modelopt_online_overlay(
    quantization: str | None, args: QuantizationConfigArgs
) -> bool:
    if quantization not in _MODELOPT_ONLINE_OVERLAY_NAMES or args.moe is not None:
        return False
    specs = [s for s in (args.linear, args.shared_experts) if s is not None]
    if not specs:
        return False
    if args.ignore and args.linear is None:
        # `ignore` only filters the dense-linear overlay.
        return False
    return all(
        spec.weight in _MODELOPT_ONLINE_OVERLAY_WEIGHTS and spec.activation is None
        for spec in specs
    )


def resolve_quantization_config(
    quantization: str | None,
    quantization_config: dict[str, Any] | QuantizationConfigArgs | None,
) -> QuantizationConfigArgs | None:
    """Resolve `--quantization` shorthand and `--quantization-config` into a
    QuantizationConfigArgs.

    `quantization` may be a CLI shorthand that desugars into a base config via
    `_ONLINE_SHORTHANDS`. `quantization_config` is a dict or pre-built args
    object. When both are online settings, fields explicitly set in
    `quantization_config` take precedence over the shorthand. ModelOpt accepts
    online overlays for BF16 dense/shared-expert projections.
    """
    if isinstance(quantization_config, dict):
        quantization_config = QuantizationConfigArgs(**quantization_config)

    if quantization is not None and quantization not in ONLINE_QUANT_SHORTHAND_NAMES:
        if quantization_config is not None and not _is_modelopt_online_overlay(
            quantization, quantization_config
        ):
            raise ValueError(
                f"quantization_config is only supported when quantization is "
                f"one of {sorted(ONLINE_QUANT_SHORTHAND_NAMES)}, or when "
                f"using the ModelOpt online overlay (weight='mxfp8' or "
                f"weight='fp8_per_block_static' on 'shared_experts' and/or "
                f"'linear', optional 'ignore'), "
                f"got quantization={quantization!r}"
            )
        return quantization_config

    base = _ONLINE_SHORTHANDS.get(quantization) if quantization else None

    if quantization_config is None:
        return base

    if base is None:
        return quantization_config

    return QuantizationConfigArgs(
        linear=quantization_config.linear or base.linear,
        moe=quantization_config.moe or base.moe,
        shared_experts=quantization_config.shared_experts,
        ignore=quantization_config.ignore or base.ignore,
    )
