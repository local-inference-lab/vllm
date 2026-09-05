# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Annotated, Literal

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from vllm.engine.protocol import EngineClient

router = APIRouter()


class PrefillFairnessRequest(BaseModel):
    """Replacement configuration for prefill scheduling fairness."""

    model_config = ConfigDict(extra="forbid")

    prefill_compute_share: (
        Annotated[float, Field(gt=0.0, lt=1.0)] | Literal["auto"] | None
    ) = None
    prefill_compute_half_life: (
        Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
        | Literal["smooth", "responsive"]
        | None
    ) = None

    @model_validator(mode="after")
    def validate_half_life(self):
        if (
            self.prefill_compute_half_life is not None
            and self.prefill_compute_share != "auto"
        ):
            raise ValueError(
                "prefill_compute_half_life requires prefill_compute_share='auto'"
            )
        return self


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.get("/prefill_fairness")
async def get_prefill_fairness(raw_request: Request):
    """Return the active prefill fairness configuration."""
    config = await engine_client(raw_request).get_prefill_fairness()
    return JSONResponse(content=config)


@router.post("/prefill_fairness")
async def set_prefill_fairness(raw_request: Request, config: PrefillFairnessRequest):
    """Switch policy live without reloading weights or clearing caches."""
    result = await engine_client(raw_request).set_prefill_fairness(config.model_dump())
    status_code = 200 if result["applied"] else 422
    return JSONResponse(content=result, status_code=status_code)


def attach_router(app: FastAPI):
    app.include_router(router)
